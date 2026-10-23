package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * Top-k graph pooling, PyTorch Geometric's {@code TopKPooling} (its {@code SelectTopK} selection, multiplier 1): a
 * learned projection vector {@code p} scores every node by {@code x[n] . p} and a subset of the nodes is kept, each
 * row multiplied by a gate derived from its score. Two selection rules:
 * <ul>
 *   <li><b>Ratio rule</b> ({@code minScore} is {@link Float#NEGATIVE_INFINITY}, the default): the score is
 *       {@code act(x[n] . p / ||p||)} with {@code act} the module's {@link Nonlinearity} ({@code tanh} by default, as
 *       PyG); the highest-scoring {@code ceil(ratio * n_g)} nodes of each graph {@code g} of the batch
 *       ({@link GraphData#graphNodes()}, {@code n_g} its node count) are kept, each gated by its score.</li>
 *   <li><b>Min-score rule</b> (any other {@code minScore}; {@code ratio} and the nonlinearity are then ignored): the
 *       raw scores {@code x[n] . p} are normalized by a softmax over the nodes of each graph of the batch, and a node
 *       is kept when its normalized score exceeds {@code min(minScore, max - 1e-7)}, {@code max} being the largest
 *       normalized score of its graph, so every graph keeps at least its best node. The kept rows are gated by their
 *       normalized score.</li>
 * </ul>
 * <p>Under either rule, nodes that belong to no graph (a negative batch index) are never kept, and an unbatched graph
 * is one graph of every node. The projection {@code p} is the module's only parameter, {@code weight}, initialized as
 * PyG does from {@code U(-1/sqrt(inFeatures), 1/sqrt(inFeatures))}.</p>
 *
 * <p><b>Autograd.</b> The choice of nodes is a host decision on the scores read back and has no
 * gradient. The projection, the norm of {@code p}, the nonlinearity or the per-graph softmax, the row selections and
 * the gated product are recorded, so {@code weight} receives its gradient through the gate of the selected nodes, and
 * the node features (when they require grad) through both the selected rows and the scores.</p>
 */
public final class TopKPool extends Module {
    /** Subtracted from each graph's best normalized score in the min-score threshold (PyG {@code topk}'s tol). */
    private static final float MIN_SCORE_TOLERANCE = 1e-7f;

    /** The function of the ratio rule that turns {@code x[n] . p / ||p||} into the node's score and gate. */
    public enum Nonlinearity {
        /** {@code tanh(s)}, PyG's default. */
        TANH,
        /** {@code 1 / (1 + exp(-s))}. */
        SIGMOID
    }

    private final int inFeatures;
    private final float ratio;
    private final float minScore;
    private final Nonlinearity nonlinearity;

    private Parameter weight;

    /** The ratio rule with the {@link Nonlinearity#TANH} gate. */
    public TopKPool(int inFeatures, float ratio) {
        this(inFeatures, ratio, Float.NEGATIVE_INFINITY, Nonlinearity.TANH);
    }

    /** The ratio rule with the given gate. */
    public TopKPool(int inFeatures, float ratio, Nonlinearity nonlinearity) {
        this(inFeatures, ratio, Float.NEGATIVE_INFINITY, nonlinearity);
    }

    /** {@link #TopKPool(int, float, float, Nonlinearity)} with the {@link Nonlinearity#TANH} gate. */
    public TopKPool(int inFeatures, float ratio, float minScore) {
        this(inFeatures, ratio, minScore, Nonlinearity.TANH);
    }

    /**
     * @param ratio        the kept fraction of the ratio rule, in {@code (0, 1]}; validated even when the min-score
     *                     rule ignores it
     * @param minScore     {@link Float#NEGATIVE_INFINITY} for the ratio rule, any other non-NaN value for the
     *                     min-score rule
     * @param nonlinearity the gate of the ratio rule; kept but unused under the min-score rule, as in PyG
     * @throws IllegalArgumentException when {@code inFeatures} is not positive, {@code ratio} is outside
     *                                  {@code (0, 1]} or {@code minScore} is NaN
     * @throws NullPointerException     when {@code nonlinearity} is null
     */
    public TopKPool(int inFeatures, float ratio, float minScore, Nonlinearity nonlinearity) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (ratio <= 0 || ratio > 1) {
            throw new IllegalArgumentException("ratio must be in (0, 1], got: " + ratio);
        }
        if (Float.isNaN(minScore)) {
            throw new IllegalArgumentException(
                "minScore must not be NaN; Float.NEGATIVE_INFINITY selects the ratio rule");
        }

        this.inFeatures = inFeatures;
        this.ratio = ratio;
        this.minScore = minScore;
        this.nonlinearity = Objects.requireNonNull(nonlinearity, "nonlinearity cannot be null");
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Registers {@code weight}, {@code [inFeatures]} values drawn from {@code U(-b, b)} with
     * {@code b = 1 / sqrt(inFeatures)} (PyG's {@code uniform(in_channels, weight)}): the host stream is seeded by one
     * {@code nextLong()} of {@code random}. Records nothing.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        float bound = (float) (1.0 / Math.sqrt(inFeatures));
        Random stream = new Random(random.nextLong());
        float[] values = new float[inFeatures];
        for (int i = 0; i < values.length; i++) {
            values[i] = (2.0f * stream.nextFloat() - 1.0f) * bound;
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            this.weight = registerParameter("weight",
                MessagePassing.ownedParameter(Tensor.fromArray(backend, values), "weight"));
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException(
            "TopKPool requires GraphData input. Use forward(graph, x) instead.");
    }

    public TopKPoolResult forward(GraphData graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        return forward(graph, graph.nodeFeatures());
    }

    /**
     * Pools the graph's nodes by the ratio rule or the min-score rule (see the class description). The projections
     * {@code x[n] . p} are computed on the node features' backend; the values the selection compares are read back
     * from detached handles, and the scoring, row selection and gating stay on that backend and
     * are recorded.
     *
     * <ul>
     *   <li>Ratio rule: the scores {@code act(x[n] . p / sqrt(p . p))}; per graph {@code g} of {@code n_g} nodes,
     *       the {@code k_g = ceil(n_g * ratio)} (float product; at least 1 and at most {@code n_g}) nodes with the
     *       highest score; the kept nodes come graph by graph in ascending graph order, each graph's in descending
     *       score order (ties keep the lower node id; NaN ranks first), as PyTorch Geometric orders them, and each
     *       row is scaled by its score.</li>
     *   <li>Min-score rule: the kept nodes in ascending node id, each row scaled by its normalized score. The
     *       normalization subtracts each graph's largest projection (read back, a constant that the softmax does not
     *       depend on) before {@code exp}, sums the exponentials per graph in ascending node order and divides.</li>
     * </ul>
     *
     * @throws IllegalArgumentException when the graph has no node, or {@code x} is not
     *                                  {@code [numNodes, inFeatures]} on the graph's backend
     * @throws IllegalStateException    when no node belongs to a graph of the batch, or the min-score rule keeps no
     *                                  node because the normalized scores are NaN
     */
    public TopKPoolResult forward(GraphData graph, Tensor x) {
        ensureInitialized();
        Objects.requireNonNull(graph, "graph cannot be null");
        Objects.requireNonNull(x, "x cannot be null");
        graph.requireNodeFeatures(x, "TopKPool");

        if (x.shape().dim(1) != inFeatures) {
            throw new IllegalArgumentException(
                "Input feature dimension must be " + inFeatures + ", got " + x.shape().dim(1));
        }

        int numNodes = (int) graph.numNodes();
        if (numNodes == 0) {
            throw new IllegalArgumentException("TopKPool needs a graph with at least one node");
        }

        try (Tensor column = weight.tensor().reshape(inFeatures, 1);
             Tensor projected = TensorOps.matmul(x, column)) {
            return hasMinScore() ? selectByMinScore(graph, x, projected)
                : selectByRatio(graph, x, projected, column);
        }
    }

    /**
     * The ratio rule over the {@code [numNodes, 1]} projections; {@code column} is {@code p} as {@code [inFeatures,
     * 1]}. The result's scores are the gated scores {@code act(x[n] . p / ||p||)} of every node.
     */
    private TopKPoolResult selectByRatio(GraphData graph, Tensor x, Tensor projected, Tensor column) {
        GraphMembership membership = GraphMembership.of(graph);
        if (membership.memberCount == 0) {
            throw new IllegalStateException(
                "TopKPool's ratio rule kept no node: no node belongs to a graph of the batch");
        }
        try (Tensor row = weight.tensor().reshape(1, inFeatures);
             Tensor squaredNorm = TensorOps.matmul(row, column);
             Tensor norm = TensorOps.sqrt(squaredNorm);
             Tensor normalized = TensorOps.div(projected, norm);
             Tensor activated = activate(normalized)) {
            float[] scores = hostScores(activated);
            int[] selectedIndices = topKPerGraph(scores, membership);
            try (Tensor selected = Tensor.fromArray(x.backend(), selectedIndices);
                 Tensor rows = TensorOps.indexSelectRows(x, selected, false);
                 Tensor gate = TensorOps.indexSelectRows(activated, selected, false)) {
                return new TopKPoolResult(TensorOps.mul(rows, gate), selectedIndices, scores);
            }
        }
    }

    /** The module's nonlinearity applied to {@code scores}, as a new tensor. */
    private Tensor activate(Tensor scores) {
        switch (nonlinearity) {
            case TANH:
                return TensorOps.tanh(scores);
            case SIGMOID:
                return TensorOps.sigmoid(scores);
            default:
                throw new IllegalStateException("Unhandled TopKPool nonlinearity " + nonlinearity);
        }
    }

    /**
     * The min-score rule over the {@code [numNodes, 1]} raw projections; the result's scores are the normalized
     * ones, NaN for nodes outside every graph.
     */
    private TopKPoolResult selectByMinScore(GraphData graph, Tensor x, Tensor projected) {
        GraphMembership membership = GraphMembership.of(graph);
        try (Tensor normalized = graphSoftmax(projected, graph, membership)) {
            float[] scores = hostScores(normalized);
            int[] selectedIndices = aboveThreshold(scores, membership);
            if (selectedIndices.length == 0) {
                throw new IllegalStateException("TopKPool's min-score rule kept no node: "
                    + (membership.memberCount == 0 ? "no node belongs to a graph of the batch"
                        : "the normalized scores are NaN"));
            }
            for (int n = 0; n < scores.length; n++) {
                if (membership.graphOf[n] < 0) {
                    scores[n] = Float.NaN;
                }
            }
            try (Tensor selected = Tensor.fromArray(x.backend(), selectedIndices);
                 Tensor rows = TensorOps.indexSelectRows(x, selected, false);
                 Tensor gate = TensorOps.indexSelectRows(normalized, selected, false)) {
                return new TopKPoolResult(TensorOps.mul(rows, gate), selectedIndices, scores);
            }
        }
    }

    /**
     * The per-graph softmax of the {@code [numNodes, 1]} scores, as a new {@code [numNodes, 1]} tensor:
     * {@code exp(s - m_g) / sum_{n' in g} exp(s_n' - m_g)} with {@code m_g} the largest score of graph {@code g}
     * (a host constant read through a detached handle), PyG's {@code softmax(score, batch)}. Scores of nodes outside
     * every graph are replaced by 0 through {@code where} first, so their entries are finite and pass no gradient;
     * their values are never selected. The divisor is the gathered graph sum where it is positive and 1 elsewhere,
     * so no entry divides by 0.
     */
    private static Tensor graphSoftmax(Tensor scoreColumn, GraphData graph, GraphMembership membership) {
        ComputeBackend backend = scoreColumn.backend();
        Shape shape = scoreColumn.shape();
        Tensor memberScores = membership.memberCount == membership.graphOf.length ? scoreColumn
            : maskedToMembers(scoreColumn, membership);
        try (Tensor shift = column(backend, graphMaxima(hostScores(memberScores), membership));
             Tensor shifted = TensorOps.sub(memberScores, shift);
             Tensor exps = TensorOps.exp(shifted);
             Tensor sums = TensorOps.segmentReduce(exps, graph.graphOffsets(), graph.graphNodes(),
                 membership.numGraphs, SegmentOp.SUM, 0.0f);
             Tensor graphIndex = Tensor.fromArray(backend, membership.clampedGraphOf());
             Tensor nodeSums = TensorOps.indexSelectRows(sums, graphIndex, false);
             Tensor zeros = Tensor.zeros(backend, shape);
             Tensor ones = Tensor.ones(backend, shape);
             Tensor positive = TensorOps.gt(nodeSums, zeros);
             Tensor divisor = TensorOps.where(positive, nodeSums, ones)) {
            return TensorOps.div(exps, divisor);
        } finally {
            if (memberScores != scoreColumn) {
                memberScores.close();
            }
        }
    }

    /** {@code scores} with the entries of nodes outside every graph replaced by 0, as a new tensor. */
    private static Tensor maskedToMembers(Tensor scores, GraphMembership membership) {
        float[] member = new float[membership.graphOf.length];
        for (int n = 0; n < member.length; n++) {
            member[n] = membership.graphOf[n] >= 0 ? 1.0f : 0.0f;
        }
        try (Tensor mask = column(scores.backend(), member);
             Tensor zeros = Tensor.zeros(scores.backend(), scores.shape())) {
            return TensorOps.where(mask, scores, zeros);
        }
    }

    /** Per node, the largest of its graph's scores ({@link Math#max} semantics, so NaN wins); 0 outside every graph. */
    private static float[] graphMaxima(float[] scores, GraphMembership membership) {
        float[] max = new float[membership.numGraphs];
        Arrays.fill(max, Float.NEGATIVE_INFINITY);
        for (int n = 0; n < scores.length; n++) {
            int g = membership.graphOf[n];
            if (g >= 0) {
                max[g] = Math.max(max[g], scores[n]);
            }
        }
        float[] shift = new float[scores.length];
        for (int n = 0; n < scores.length; n++) {
            int g = membership.graphOf[n];
            shift[n] = g >= 0 ? max[g] : 0.0f;
        }
        return shift;
    }

    /**
     * The nodes, in ascending id, whose normalized score exceeds {@code min(minScore, max_g - 1e-7f)} in float
     * arithmetic, {@code max_g} being the largest normalized score of the node's graph.
     */
    private int[] aboveThreshold(float[] normalized, GraphMembership membership) {
        float[] max = new float[membership.numGraphs];
        Arrays.fill(max, Float.NEGATIVE_INFINITY);
        for (int n = 0; n < normalized.length; n++) {
            int g = membership.graphOf[n];
            if (g >= 0) {
                max[g] = Math.max(max[g], normalized[n]);
            }
        }
        int[] kept = new int[normalized.length];
        int count = 0;
        for (int n = 0; n < normalized.length; n++) {
            int g = membership.graphOf[n];
            if (g >= 0 && normalized[n] > Math.min(minScore, max[g] - MIN_SCORE_TOLERANCE)) {
                kept[count++] = n;
            }
        }
        return Arrays.copyOf(kept, count);
    }

    /** A new FLOAT32 {@code [values.length, 1]} tensor of {@code values} on {@code backend}. */
    private static Tensor column(ComputeBackend backend, float[] values) {
        try (Tensor flat = Tensor.fromArray(backend, values)) {
            return flat.reshape(values.length, 1);
        }
    }

    /** The scores on the host, read through a detached handle: a selection input, never part of the graph. */
    private static float[] hostScores(Tensor scoreColumn) {
        try (Tensor constant = scoreColumn.detach()) {
            return constant.toFloatArray();
        }
    }

    /**
     * The kept node ids of the ratio rule: graph by graph in ascending order, each graph's first
     * {@code ceil(n_g * ratio)} members by descending score. The sort is stable over ascending ids, so equal scores
     * keep ascending ids.
     */
    private int[] topKPerGraph(float[] scores, GraphMembership membership) {
        Integer[] members = new Integer[membership.memberCount];
        int[] graphSize = new int[membership.numGraphs];
        int count = 0;
        for (int n = 0; n < scores.length; n++) {
            int g = membership.graphOf[n];
            if (g >= 0) {
                members[count++] = n;
                graphSize[g]++;
            }
        }
        Arrays.sort(members, (a, b) -> {
            int byGraph = Integer.compare(membership.graphOf[a], membership.graphOf[b]);
            return byGraph != 0 ? byGraph : Float.compare(scores[b], scores[a]);
        });
        int[] kept = new int[members.length];
        int keptCount = 0;
        int start = 0;
        for (int g = 0; g < membership.numGraphs; g++) {
            int k = Math.min(graphSize[g], (int) Math.ceil(graphSize[g] * ratio));
            for (int i = 0; i < k; i++) {
                kept[keptCount++] = members[start + i];
            }
            start += graphSize[g];
        }
        return Arrays.copyOf(kept, keptCount);
    }

    /**
     * The graph of the batch each node belongs to, read back from the graph's cached structure
     * ({@link GraphData#graphOffsets()}, {@link GraphData#graphNodes()}).
     */
    private static final class GraphMembership {
        final int numGraphs;
        /** Per node, its graph, or -1 when its batch index names no graph. */
        final int[] graphOf;
        final int memberCount;

        private GraphMembership(int numGraphs, int[] graphOf, int memberCount) {
            this.numGraphs = numGraphs;
            this.graphOf = graphOf;
            this.memberCount = memberCount;
        }

        static GraphMembership of(GraphData graph) {
            int[] offsets = graph.graphOffsets().toIntArray();
            int[] nodes = graph.graphNodes().toIntArray();
            int numGraphs = offsets.length - 1;
            int[] graphOf = new int[(int) graph.numNodes()];
            Arrays.fill(graphOf, -1);
            for (int g = 0; g < numGraphs; g++) {
                for (int slot = offsets[g]; slot < offsets[g + 1]; slot++) {
                    graphOf[nodes[slot]] = g;
                }
            }
            return new GraphMembership(numGraphs, graphOf, nodes.length);
        }

        /** {@link #graphOf} with -1 replaced by graph 0: a valid gather index for every node. */
        int[] clampedGraphOf() {
            int[] clamped = new int[graphOf.length];
            for (int n = 0; n < clamped.length; n++) {
                clamped[n] = Math.max(graphOf[n], 0);
            }
            return clamped;
        }
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("TopKPool not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() { return inFeatures; }
    public float ratio() { return ratio; }
    public float minScore() { return minScore; }
    /** The gate of the ratio rule. */
    public Nonlinearity nonlinearity() { return nonlinearity; }
    /** Whether the min-score rule selects the nodes ({@code minScore} is not {@link Float#NEGATIVE_INFINITY}). */
    public boolean hasMinScore() { return minScore != Float.NEGATIVE_INFINITY; }

    /** The projection vector {@code p}, {@code [inFeatures]}. */
    public Parameter weight() { ensureInitialized(); return weight; }

    @Override
    public String toString() {
        return "TopKPool{" +
            "inFeatures=" + inFeatures +
            ", ratio=" + ratio +
            (hasMinScore() ? ", minScore=" + minScore : ", nonlinearity=" + nonlinearity) +
            ", initialized=" + (weight != null) +
            '}';
    }

    /**
     * The pooled features {@code [k, inFeatures]} with the selected node ids and every node's score (host copies):
     * the gated score {@code act(x[n] . p / ||p||)} under the ratio rule, the per-graph normalized score under the
     * min-score rule (NaN for a node outside every graph). The caller owns the pooled features; when the forward
     * recorded them they are valid until the loss computed from them is back-propagated, which releases them. The
     * host arrays stay valid.
     */
    public static final class TopKPoolResult {
        private final Tensor pooledFeatures;
        private final int[] selectedIndices;
        private final float[] scores;

        public TopKPoolResult(Tensor pooledFeatures, int[] selectedIndices, float[] scores) {
            this.pooledFeatures = pooledFeatures;
            this.selectedIndices = selectedIndices;
            this.scores = scores;
        }

        public Tensor pooledFeatures() { return pooledFeatures; }
        public int[] selectedIndices() { return selectedIndices; }
        public float[] scores() { return scores; }
        public int numSelected() { return selectedIndices.length; }
    }
}
