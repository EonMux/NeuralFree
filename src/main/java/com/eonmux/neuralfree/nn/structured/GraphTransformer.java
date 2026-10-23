package com.eonmux.neuralfree.nn.structured;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.graph.GraphData;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.nn.norm.LayerNorm;
import com.eonmux.neuralfree.nn.activations.ReLU;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Graph transformer: per layer, edge-wise multi-head attention aggregated per target node, an output projection with
 * a residual connection, and a ReLU feed-forward block with a residual connection; a final linear projection maps
 * {@code hiddenDim} to {@code outFeatures} when they differ. Every step is a {@code TensorOps} composition, so the
 * model runs on the device when the train kernels are available on the features' backend; intermediates are
 * released before returning.
 *
 * <p>The first layer normalizes and projects the {@code inFeatures}-wide input; when that width differs from
 * {@code hiddenDim} its attention output has no residual connection. With {@code dropoutRate > 0}, in training mode
 * the feed-forward block applies the registered {@code dropout} child after the ReLU and after its second projection;
 * its keep decisions are host draws through {@code CheckpointFunction.replayable}, seeded by {@link #initialize} after
 * every weight.</p>
 *
 * <p><b>Edge features.</b> With {@code useEdgeFeatures}, layer {@code i} owns {@code edge_i =
 * Linear(edgeFeatureDim, numHeads)}, which maps each edge's features to one additive bias per head on that edge's
 * attention logit, as Graphormer's edge encoding and the Graph Transformer's edge bias: for edge {@code e = (s -> t)}
 * the logit of head {@code h} is {@code (q_t . k_s) * scale + edge_i(f_e)[h]}. The graph must then carry FLOAT32
 * edge features {@code [numEdges, edgeFeatureDim]} on the node features' backend. Without {@code useEdgeFeatures} no
 * edge layer is created and the graph's edge features are not read.</p>
 *
 * <p><b>Gradients.</b> Every operation records, the row gathers ({@code indexSelectRows}), the per-target sums
 * ({@code scatterAdd}) and the {@code where} selections included, so a loss on the output reaches every parameter the
 * forward reads; the edge indices, the {@code sum > 0} masks and the per-target score maxima are constants.</p>
 */
public final class GraphTransformer extends Module {
    private final int inFeatures;
    private final int hiddenDim;
    private final int outFeatures;
    private final int numHeads;
    private final int numLayers;
    private final float dropoutRate;
    private final boolean useEdgeFeatures;
    private final int edgeFeatureDim;

    private final List<Linear> queryLayers;
    private final List<Linear> keyLayers;
    private final List<Linear> valueLayers;
    private final List<Linear> edgeLayers;
    private final List<Linear> outLayers;
    private final List<Linear> ffLayers1;
    private final List<Linear> ffLayers2;
    private final List<LayerNorm> normLayers1;
    private final List<LayerNorm> normLayers2;
    private Linear outputProjection;
    private Dropout dropout;
    private ReLU relu;

    private boolean initialized = false;

    public GraphTransformer(int inFeatures, int hiddenDim, int outFeatures, int numHeads, int numLayers) {
        this(inFeatures, hiddenDim, outFeatures, numHeads, numLayers, 0.1f, false);
    }

    /** A transformer whose edge layers, with {@code useEdgeFeatures}, read one edge feature per edge. */
    public GraphTransformer(int inFeatures, int hiddenDim, int outFeatures, int numHeads,
                            int numLayers, float dropoutRate, boolean useEdgeFeatures) {
        this(inFeatures, hiddenDim, outFeatures, numHeads, numLayers, dropoutRate, useEdgeFeatures, 1);
    }

    /**
     * @param edgeFeatureDim the width of the graph's edge features that the {@code edge_i} layers read when
     *                       {@code useEdgeFeatures}; must be positive
     */
    public GraphTransformer(int inFeatures, int hiddenDim, int outFeatures, int numHeads,
                            int numLayers, float dropoutRate, boolean useEdgeFeatures, int edgeFeatureDim) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (hiddenDim <= 0) {
            throw new IllegalArgumentException("hiddenDim must be positive: " + hiddenDim);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        if (numHeads <= 0) {
            throw new IllegalArgumentException("numHeads must be positive: " + numHeads);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (hiddenDim % numHeads != 0) {
            throw new IllegalArgumentException("hiddenDim (" + hiddenDim + ") must be divisible by numHeads (" + numHeads + ")");
        }
        if (!(dropoutRate >= 0.0f && dropoutRate < 1.0f)) {
            throw new IllegalArgumentException("dropoutRate must be in [0.0, 1.0): " + dropoutRate);
        }
        if (edgeFeatureDim <= 0) {
            throw new IllegalArgumentException("edgeFeatureDim must be positive: " + edgeFeatureDim);
        }

        this.inFeatures = inFeatures;
        this.hiddenDim = hiddenDim;
        this.outFeatures = outFeatures;
        this.numHeads = numHeads;
        this.numLayers = numLayers;
        this.dropoutRate = dropoutRate;
        this.useEdgeFeatures = useEdgeFeatures;
        this.edgeFeatureDim = edgeFeatureDim;

        this.queryLayers = new ArrayList<>();
        this.keyLayers = new ArrayList<>();
        this.valueLayers = new ArrayList<>();
        this.edgeLayers = new ArrayList<>();
        this.outLayers = new ArrayList<>();
        this.ffLayers1 = new ArrayList<>();
        this.ffLayers2 = new ArrayList<>();
        this.normLayers1 = new ArrayList<>();
        this.normLayers2 = new ArrayList<>();
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Creates every layer on {@code backend}, drawing the weights from {@code random}, then, when
     * {@code dropoutRate > 0}, the seed of the feed-forward dropout masks, so equal seeds give equal masks on every
     * backend.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        for (int i = 0; i < numLayers; i++) {
            int inFeat = (i == 0) ? inFeatures : hiddenDim;

            Linear qLayer = new Linear(inFeat, hiddenDim);
            qLayer.initializeXavier(backend, random);
            queryLayers.add(qLayer);
            registerModule("query_" + i, qLayer);

            Linear kLayer = new Linear(inFeat, hiddenDim);
            kLayer.initializeXavier(backend, random);
            keyLayers.add(kLayer);
            registerModule("key_" + i, kLayer);

            Linear vLayer = new Linear(inFeat, hiddenDim);
            vLayer.initializeXavier(backend, random);
            valueLayers.add(vLayer);
            registerModule("value_" + i, vLayer);

            if (useEdgeFeatures) {
                Linear eLayer = new Linear(edgeFeatureDim, numHeads);
                eLayer.initializeXavier(backend, random);
                edgeLayers.add(eLayer);
                registerModule("edge_" + i, eLayer);
            }

            Linear outLayer = new Linear(hiddenDim, hiddenDim);
            outLayer.initializeXavier(backend, random);
            outLayers.add(outLayer);
            registerModule("out_" + i, outLayer);

            Linear ff1 = new Linear(hiddenDim, hiddenDim * 4);
            ff1.initializeXavier(backend, random);
            ffLayers1.add(ff1);
            registerModule("ff1_" + i, ff1);

            Linear ff2 = new Linear(hiddenDim * 4, hiddenDim);
            ff2.initializeXavier(backend, random);
            ffLayers2.add(ff2);
            registerModule("ff2_" + i, ff2);

            LayerNorm norm1 = new LayerNorm(inFeat);
            norm1.initialize(backend);
            normLayers1.add(norm1);
            registerModule("norm1_" + i, norm1);

            LayerNorm norm2 = new LayerNorm(hiddenDim);
            norm2.initialize(backend);
            normLayers2.add(norm2);
            registerModule("norm2_" + i, norm2);
        }

        if (outFeatures != hiddenDim) {
            outputProjection = new Linear(hiddenDim, outFeatures);
            outputProjection.initializeXavier(backend, random);
            registerModule("output_proj", outputProjection);
        }

        if (dropoutRate > 0.0f) {
            dropout = new Dropout(dropoutRate);
            dropout.setSeed(random.nextLong());
            registerModule("dropout", dropout);
        }

        relu = new ReLU();
        registerModule("relu", relu);

        initialized = true;
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException("GraphTransformer requires GraphData input. Use forward(graph) instead.");
    }

    public Tensor forward(GraphData graph) {
        Objects.requireNonNull(graph, "Graph cannot be null");
        return forward(graph, graph.nodeFeatures());
    }

    /**
     * Runs every layer over node features {@code x} ({@code [numNodes, inFeatures]}) and the graph's edges, and
     * returns new {@code [numNodes, outFeatures]} features. The edge index ({@code [2, numEdges]} source and target
     * rows; FLOAT32 values converted by Java {@code (int)}, INT64 narrowed, or INT32) is converted on its backend to
     * INT32 index vectors; the per-target score maxima use the graph's cached CSR structure, which {@link GraphData}
     * builds on the host once. With {@code useEdgeFeatures} the graph's edge features add the
     * per-head edge bias to the attention logits (see the class description).
     *
     * @throws IllegalArgumentException  when {@code x} does not have one row per node or is not on the graph's
     *                                   backend, or, with {@code useEdgeFeatures}, the graph has no FLOAT32
     *                                   {@code [numEdges, edgeFeatureDim]} edge features on that backend
     * @throws IndexOutOfBoundsException when an edge names a node outside {@code [0, numNodes)} (on the device path,
     *                                   at the backend's next host synchronization)
     */
    public Tensor forward(GraphData graph, Tensor x) {
        ensureInitialized();
        Objects.requireNonNull(graph, "Graph cannot be null");
        Objects.requireNonNull(x, "Input tensor cannot be null");

        int numEdges = Math.toIntExact(graph.numEdges());
        int numNodes = Math.toIntExact(graph.numNodes());
        if (x.shape().rank() != 2 || x.shape().dim(0) != numNodes) {
            throw new IllegalArgumentException("node features must be [" + numNodes + ", features], got " + x.shape());
        }
        if (x.backend() != graph.backend()) {
            throw new IllegalArgumentException("node features must be on the graph's backend ("
                + graph.backend().backendType() + "), got " + x.backend().backendType());
        }
        Tensor edgeFeatures = useEdgeFeatures ? requireEdgeFeatures(graph) : null;

        Tensor sources = null;
        Tensor targets = null;
        Tensor h = x;
        try {
            if (numEdges > 0) {
                sources = edgeRow(graph.edgeIndex(), 0, numEdges);
                targets = edgeRow(graph.edgeIndex(), 1, numEdges);
            }
            for (int layer = 0; layer < numLayers; layer++) {
                Tensor attnOut;
                try (Tensor hNorm = normLayers1.get(layer).forward(h);
                     Tensor aggregated = graphAttention(hNorm, graph, sources, targets, edgeFeatures, numNodes,
                         layer)) {
                    attnOut = outLayers.get(layer).forward(aggregated);
                }
                Tensor mixed;
                if (h.shape().dim(1) == hiddenDim) {
                    try (Tensor a = attnOut) {
                        mixed = TensorOps.add(h, a);
                    }
                } else {
                    mixed = attnOut;
                }
                h = replace(h, mixed, x);

                try (Tensor ff = feedForward(h, layer)) {
                    h = replace(h, TensorOps.add(h, ff), x);
                }
            }

            if (outputProjection != null) {
                h = replace(h, outputProjection.forward(h), x);
            }
            return h;
        } catch (RuntimeException e) {
            if (h != x) {
                h.close();
            }
            throw e;
        } finally {
            if (sources != null) {
                sources.close();
            }
            if (targets != null) {
                targets.close();
            }
        }
    }

    /** The graph's edge features, checked to be what the {@code edge_i} layers read. */
    private Tensor requireEdgeFeatures(GraphData graph) {
        if (!graph.hasEdgeFeatures()) {
            throw new IllegalArgumentException("GraphTransformer with useEdgeFeatures needs a graph with edge "
                + "features [" + graph.numEdges() + ", " + edgeFeatureDim + "]");
        }
        Tensor edgeFeatures = graph.edgeFeatures();
        if (edgeFeatures.shape().dim(1) != edgeFeatureDim) {
            throw new IllegalArgumentException("edge features must be [numEdges, " + edgeFeatureDim + "], got "
                + edgeFeatures.shape());
        }
        if (edgeFeatures.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("edge features must be FLOAT32, got " + edgeFeatures.dtype());
        }
        if (edgeFeatures.backend() != graph.backend()) {
            throw new IllegalArgumentException("edge features must be on the graph's backend ("
                + graph.backend().backendType() + "), got " + edgeFeatures.backend().backendType());
        }
        return edgeFeatures;
    }

    /** {@code next}, closing {@code previous} unless it is the caller's {@code input}. */
    private static Tensor replace(Tensor previous, Tensor next, Tensor input) {
        if (previous != input && previous != next) {
            previous.close();
        }
        return next;
    }

    private Tensor feedForward(Tensor h, int layer) {
        Tensor activated;
        try (Tensor normed = normLayers2.get(layer).forward(h);
             Tensor expanded = ffLayers1.get(layer).forward(normed)) {
            activated = applyDropout(relu.forward(expanded));
        }
        Tensor projected;
        try (Tensor a = activated) {
            projected = ffLayers2.get(layer).forward(a);
        }
        return applyDropout(projected);
    }

    private Tensor applyDropout(Tensor t) {
        if (dropout == null) {
            return t;
        }
        Tensor dropped;
        try {
            dropped = dropout.forward(t);
        } catch (RuntimeException e) {
            t.close();
            throw e;
        }
        if (dropped != t) {
            t.close();
        }
        return dropped;
    }

    /**
     * Edge-wise multi-head attention aggregated per target node, as tensor ops on the device when the train kernels
     * are available. For edge {@code e = (s -> t)} and head {@code h}: {@code score = (q_t . k_s) * scale}
     * (plus the edge bias {@code edge_i(f_e)[h]} when {@code edgeFeatures} is not null) with {@code q = Wq x + bq},
     * {@code k = Wk x + bk}, {@code v = Wv x + bv} (the layer's linear maps),
     * {@code w_e = exp(score - m_t)} with {@code m_t} the largest score into {@code t} (per head, taken from detached
     * scores, so it is a constant that the normalized weights do not depend on and that keeps {@code exp} from
     * overflowing), {@code alpha_e = w_e / sum_{e' -> t} w_e'} when that sum is positive and 0 otherwise, and
     * {@code out_t = sum_{e -> t} alpha_e * v_s}. Rows are gathered with {@code indexSelectRows} and both sums run
     * with {@code scatterAdd} in ascending edge order. Nodes without incoming edges get 0. The dot products are in the
     * reduction class, {@code exp} in the transcendental class and the edge sums in the single-rounding class of
     * the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.
     */
    private Tensor graphAttention(Tensor x, GraphData graph, Tensor sources, Tensor targets, Tensor edgeFeatures,
                                  int numNodes, int layerIdx) {
        ComputeBackend backend = x.backend();
        if (sources == null) {
            return Tensor.zeros(backend, Shape.matrix(numNodes, hiddenDim));
        }
        int numEdges = Math.toIntExact(sources.numElements());
        int headDim = hiddenDim / numHeads;
        float scale = 1.0f / (float) Math.sqrt(headDim);

        Tensor scores;
        try (Tensor q = queryLayers.get(layerIdx).forward(x);
             Tensor k = keyLayers.get(layerIdx).forward(x);
             Tensor qEdges = TensorOps.indexSelectRows(q, targets, false);
             Tensor kEdges = TensorOps.indexSelectRows(k, sources, false);
             Tensor products = TensorOps.mul(qEdges, kEdges);
             Tensor perHead = products.reshape(Shape.of(numEdges, numHeads, headDim));
             Tensor dots = TensorOps.sum(perHead, 2)) {
            scores = TensorOps.mul(dots, scale);
        }
        if (edgeFeatures != null) {
            try (Tensor attentionScores = scores;
                 Tensor edgeBias = edgeLayers.get(layerIdx).forward(edgeFeatures)) {
                scores = TensorOps.add(attentionScores, edgeBias);
            }
        }

        Tensor weights;
        try (Tensor s = scores;
             Tensor shifted = subtractTargetMax(s, graph, targets, numNodes)) {
            weights = TensorOps.exp(shifted);
        }

        try (Tensor w = weights;
             Tensor v = valueLayers.get(layerIdx).forward(x);
             Tensor vEdges = TensorOps.indexSelectRows(v, sources, false);
             Tensor alpha = normalizePerTarget(w, targets, numNodes, numEdges);
             Tensor alpha3 = alpha.reshape(Shape.of(numEdges, numHeads, 1));
             Tensor vHeads = vEdges.reshape(Shape.of(numEdges, numHeads, headDim));
             Tensor messages = TensorOps.mul(alpha3, vHeads);
             Tensor targetIndex = expandIndex(targets, Shape.of(numEdges, numHeads, headDim));
             Tensor zeros = Tensor.zeros(backend, Shape.of(numNodes, numHeads, headDim));
             Tensor aggregated = TensorOps.scatterAdd(zeros, 0, targetIndex, messages)) {
            return aggregated.reshape(Shape.matrix(numNodes, hiddenDim));
        }
    }

    /**
     * {@code scores [numEdges, heads]} minus, per head, the largest score among the edges into the same target node.
     * The maxima are a {@code MAX} segment reduction over the graph's CSR of a detached handle of {@code scores}, so
     * the shift records nothing and receives no gradient; the gradient reaches {@code scores} through the subtraction.
     */
    private static Tensor subtractTargetMax(Tensor scores, GraphData graph, Tensor targets, int numNodes) {
        try (Tensor constantScores = scores.detach();
             Tensor max = TensorOps.segmentReduce(constantScores, graph.csrOffsets(), graph.csrEdges(), numNodes,
                     SegmentOp.MAX, 0.0f);
             Tensor edgeMax = TensorOps.indexSelectRows(max, targets, false)) {
            return TensorOps.sub(scores, edgeMax);
        }
    }

    /**
     * {@code w_e / sum_{e' -> t(e)} w_e'} per head, or 0 where that sum is not positive. The divisor is 1 where the
     * sum is not positive, so the unselected quotient there is finite and its zero gradient stays zero.
     */
    private Tensor normalizePerTarget(Tensor weights, Tensor targets, int numNodes, int numEdges) {
        ComputeBackend backend = weights.backend();
        Shape edgeHeads = Shape.matrix(numEdges, numHeads);
        try (Tensor targetIndex = expandIndex(targets, edgeHeads);
             Tensor zeros = Tensor.zeros(backend, Shape.matrix(numNodes, numHeads));
             Tensor sums = TensorOps.scatterAdd(zeros, 0, targetIndex, weights);
             Tensor edgeSums = TensorOps.indexSelectRows(sums, targets, false);
             Tensor edgeZeros = Tensor.zeros(backend, edgeHeads);
             Tensor edgeOnes = Tensor.ones(backend, edgeHeads);
             Tensor positive = TensorOps.gt(edgeSums, edgeZeros);
             Tensor divisor = TensorOps.where(positive, edgeSums, edgeOnes);
             Tensor ratio = TensorOps.div(weights, divisor)) {
            return TensorOps.where(positive, ratio, edgeZeros);
        }
    }

    /** The {@code [numEdges]} index broadcast along new trailing dims to {@code shape}. */
    private static Tensor expandIndex(Tensor index, Shape shape) {
        int[] dims = new int[shape.rank()];
        Arrays.fill(dims, 1);
        dims[0] = shape.dim(0);
        try (Tensor column = index.reshape(Shape.of(dims))) {
            return column.expand(shape);
        }
    }

    /** Row {@code row} of the {@code [2, numEdges]} edge index as an owned INT32 vector on its backend. */
    private static Tensor edgeRow(Tensor edgeIndex, int row, int numEdges) {
        try (Tensor slice = edgeIndex.slice(0, row, row + 1);
             Tensor flat = slice.reshape(Shape.vector(numEdges))) {
            Tensor ints = flat.to(DType.INT32);
            return ints != flat ? ints : flat.reshape(flat.shape());
        }
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("GraphTransformer not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() { return inFeatures; }
    public int hiddenDim() { return hiddenDim; }
    public int outFeatures() { return outFeatures; }
    public int numHeads() { return numHeads; }
    public int numLayers() { return numLayers; }
    public float dropoutRate() { return dropoutRate; }
    public boolean useEdgeFeatures() { return useEdgeFeatures; }
    public int edgeFeatureDim() { return edgeFeatureDim; }

    @Override
    public String toString() {
        return "GraphTransformer{" +
            "inFeatures=" + inFeatures +
            ", hiddenDim=" + hiddenDim +
            ", outFeatures=" + outFeatures +
            ", numHeads=" + numHeads +
            ", numLayers=" + numLayers +
            ", useEdgeFeatures=" + useEdgeFeatures +
            ", initialized=" + initialized +
            '}';
    }
}
