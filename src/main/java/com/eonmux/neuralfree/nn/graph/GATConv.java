package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.tensor.GpuReduceOps.SegmentOp;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * Graph attention convolution: node features are projected per head ({@code x @ weight}), each edge gets a per-head
 * attention weight from a LeakyReLU logit normalized by a softmax over the edges into its target, the target sums its
 * weighted source projections, and the heads are concatenated or averaged before the bias. Every step is a
 * {@link TensorOps} composition on the node features' backend; the softmax runs per target through the graph's CSR.
 *
 * <p><b>Attention dropout</b> (PyTorch Geometric's {@code GATConv}): in training mode the normalized attention
 * coefficients {@code alpha [numEdges, heads]} pass through the registered {@link Dropout} child
 * {@value #ATTENTION_DROPOUT} with rate {@code dropout}, which zeroes each coefficient with that probability and scales
 * the kept ones by {@code 1 / (1 - dropout)} before they weight the messages. Its keep decisions are host draws, one
 * per coefficient in row-major order, through {@code CheckpointFunction.replayable}, so a checkpointed recompute
 * reuses the forward's mask; {@link #initialize(ComputeBackend, Random)} seeds them from its {@code Random} after the
 * parameters when {@code dropout > 0}, so equal seeds give equal masks on every backend. In evaluation mode, or with
 * {@code dropout == 0}, the coefficients pass unchanged and the forward is identical to that of a layer without
 * dropout.</p>
 *
 * <p><b>Autograd.</b> The forward records every step, so {@code weight}, {@code attSrc}, {@code attDst},
 * {@code bias} and the node features (when they require grad) receive gradients; see {@link MessagePassing}. The
 * per-target maximum subtracted before {@code exp} is computed from detached scores: the softmax is invariant to that
 * shift, so treating it as a constant leaves the gradient that of the unshifted softmax. A training-mode dropout
 * records {@code MaskScaleGrad}, so a dropped coefficient passes no gradient.</p>
 */
public final class GATConv extends MessagePassing {
    /** Registered name of the attention-dropout child module. */
    public static final String ATTENTION_DROPOUT = "attention_dropout";

    private final int inFeatures;
    private final int outFeatures;
    private final int heads;
    private final boolean concat;
    private final boolean useBias;
    private final float negativeSlope;
    private final float dropout;
    private final Dropout attentionDropout;
    
    private Parameter weight;
    private Parameter attSrc;
    private Parameter attDst;
    private Parameter bias;

    public GATConv(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, 1, true);
    }

    public GATConv(int inFeatures, int outFeatures, int heads, boolean concat) {
        this(inFeatures, outFeatures, heads, concat, true, 0.2f, 0.0f);
    }

    public GATConv(int inFeatures, int outFeatures, int heads, boolean concat, 
                   boolean useBias, float negativeSlope, float dropout) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        if (heads <= 0) {
            throw new IllegalArgumentException("heads must be positive: " + heads);
        }
        if (!(dropout >= 0.0f && dropout < 1.0f)) {
            throw new IllegalArgumentException("dropout must be in [0.0, 1.0): " + dropout);
        }
        
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.heads = heads;
        this.concat = concat;
        this.useBias = useBias;
        this.negativeSlope = negativeSlope;
        this.dropout = dropout;
        this.attentionDropout = registerModule(ATTENTION_DROPOUT, new Dropout(dropout));
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    /**
     * Creates the parameters on {@code backend} from {@code random}: one seed each for {@code weight},
     * {@code att_src} and {@code att_dst}, then, when {@code dropout > 0}, one seed for the attention-dropout masks.
     */
    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        float std = (float) Math.sqrt(2.0 / (inFeatures + outFeatures * heads));
        
        this.weight = registerParameter("weight", normalParameter(backend,
            Shape.matrix(inFeatures, outFeatures * heads), std, random.nextLong(), "weight"));
        this.attSrc = registerParameter("att_src", normalParameter(backend,
            Shape.matrix(1, heads * outFeatures), std, random.nextLong(), "att_src"));
        this.attDst = registerParameter("att_dst", normalParameter(backend,
            Shape.matrix(1, heads * outFeatures), std, random.nextLong(), "att_dst"));
        
        if (useBias) {
            int biasSize = concat ? outFeatures * heads : outFeatures;
            this.bias = registerParameter("bias", zeroParameter(backend, Shape.vector(biasSize), "bias"));
        }
        if (dropout > 0.0f) {
            attentionDropout.setSeed(random.nextLong());
        }
    }

    /** {@code x @ weight}: the per-head features {@code [numNodes, heads * outFeatures]}, head-major. */
    @Override
    protected Tensor transform(Tensor x) {
        ensureInitialized();
        return TensorOps.matmul(x, weight.tensor());
    }

    @Override
    protected boolean needsTargetFeatures() {
        return true;
    }

    /**
     * Every edge target must name a node of the graph.
     *
     * @throws IndexOutOfBoundsException when an edge's target lies outside the graph
     */
    @Override
    protected void requireSupportedEdges(GraphData graph) {
        if (!graph.targetsInRange()) {
            throw new IndexOutOfBoundsException("GATConv needs every edge target in [0, " + graph.numNodes()
                + "), the edge index of this graph has one outside");
        }
    }

    /**
     * Each edge's source features scaled per head by its attention weight {@code alpha}: the logit
     * {@code sum_f(src[h, f] * attSrc[h, f] + dst[h, f] * attDst[h, f])} goes through LeakyReLU, a softmax over the
     * edges into the same target node (see {@link #edgeSoftmax}) and, in training mode, the attention dropout.
     * {@link #requireSupportedEdges} has checked that every target lies in the graph.
     */
    @Override
    protected Tensor message(Tensor srcFeatures, Tensor dstFeatures, GraphData graph) {
        int numEdges = srcFeatures.shape().dim(0);
        Tensor alpha;
        try (Tensor srcTerms = TensorOps.mul(srcFeatures, attSrc.tensor());
             Tensor dstTerms = TensorOps.mul(dstFeatures, attDst.tensor());
             Tensor terms = TensorOps.add(srcTerms, dstTerms);
             Tensor perHead = terms.reshape(numEdges, heads, outFeatures);
             Tensor logits = TensorOps.sum(perHead, 2);
             Tensor scores = TensorOps.leakyRelu(logits, negativeSlope)) {
            alpha = attentionWeights(scores, graph);
        }
        try (Tensor attention = alpha;
             Tensor alphaColumn = attention.reshape(numEdges, heads, 1);
             Tensor src = srcFeatures.reshape(numEdges, heads, outFeatures);
             Tensor weighted = TensorOps.mul(src, alphaColumn)) {
            return weighted.reshape(numEdges, heads * outFeatures);
        }
    }

    /**
     * The edge softmax of {@code scores} after the attention dropout, as a new tensor: the dropout's
     * {@code maskScale} output in training mode with {@code dropout > 0}, the softmax itself otherwise.
     */
    private Tensor attentionWeights(Tensor scores, GraphData graph) {
        Tensor alpha = edgeSoftmax(scores, graph);
        Tensor dropped;
        try {
            dropped = attentionDropout.forward(alpha);
        } catch (RuntimeException e) {
            alpha.close();
            throw e;
        }
        if (dropped != alpha) {
            alpha.close();
        }
        return dropped;
    }

    /**
     * Softmax of {@code scores [numEdges, heads]} over the edges into each target node, per head: the target's
     * maximum score is subtracted before {@code exp}, the exponentials are summed in ascending edge order, and each
     * is divided by its target's sum; a sum that is not positive gives 0, as an all {@code -inf} target does. The
     * maximum is taken from a detached handle of {@code scores}, so the shift records nothing and receives no
     * gradient (the softmax does not depend on it); the gradient reaches {@code scores} through the shifted
     * exponentials, their segment sum and the division. A target whose sum is not positive divides by 1 instead, so
     * an all {@code -inf} target passes a zero gradient back to its scores instead of the NaN of {@code 0 / 0}.
     */
    private Tensor edgeSoftmax(Tensor scores, GraphData graph) {
        int numNodes = (int) graph.numNodes();
        Tensor offsets = graph.csrOffsets();
        Tensor edges = graph.csrEdges();
        Tensor targets = graph.targetIndex();
        try (Tensor constantScores = scores.detach();
             Tensor max = TensorOps.segmentReduce(constantScores, offsets, edges, numNodes, SegmentOp.MAX, 0.0f);
             Tensor edgeMax = TensorOps.indexSelectRows(max, targets, true);
             Tensor shifted = TensorOps.sub(scores, edgeMax);
             Tensor exps = TensorOps.exp(shifted);
             Tensor sum = TensorOps.segmentReduce(exps, offsets, edges, numNodes, SegmentOp.SUM, 0.0f);
             Tensor edgeSum = TensorOps.indexSelectRows(sum, targets, true);
             Tensor zeros = Tensor.zeros(scores.backend(), scores.shape());
             Tensor ones = Tensor.ones(scores.backend(), scores.shape());
             Tensor positive = TensorOps.gt(edgeSum, zeros);
             Tensor divisor = TensorOps.where(positive, edgeSum, ones);
             Tensor ratio = TensorOps.div(exps, divisor)) {
            return TensorOps.where(positive, ratio, zeros);
        }
    }

    @Override
    protected Tensor aggregate(Tensor messages, GraphData graph) {
        return scatterAdd(messages, graph);
    }

    /**
     * Concatenated heads ({@code [numNodes, heads * outFeatures]}) or their mean ({@code [numNodes, outFeatures]}:
     * the head sum, then one division by {@code heads}), plus the bias when the layer has one.
     */
    @Override
    protected Tensor update(Tensor aggregated, Tensor nodeFeatures) {
        ensureInitialized();
        Tensor combined;
        if (concat) {
            combined = aggregated;
        } else {
            try (Tensor perHead = aggregated.reshape(aggregated.shape().dim(0), heads, outFeatures)) {
                combined = TensorOps.mean(perHead, 1);
            }
        }
        if (!useBias) {
            return combined;
        }
        try {
            return TensorOps.add(combined, bias.tensor());
        } finally {
            if (combined != aggregated) {
                combined.close();
            }
        }
    }

    @Override
    public Tensor forward(GraphData graph) {
        Objects.requireNonNull(graph, "graph cannot be null");
        return forward(graph, graph.nodeFeatures());
    }

    @Override
    public Tensor forward(GraphData graph, Tensor x) {
        ensureInitialized();
        Objects.requireNonNull(graph, "graph cannot be null");
        graph.requireNodeFeatures(x, "GATConv");
        
        if (x.shape().dim(1) != inFeatures) {
            throw new IllegalArgumentException(
                "Input feature dimension must be " + inFeatures + ", got " + x.shape().dim(1));
        }
        
        return propagate(graph, x);
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("GATConv not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() { return inFeatures; }
    public int outFeatures() { return outFeatures; }
    public int heads() { return heads; }
    public boolean isConcat() { return concat; }
    public boolean hasBias() { return useBias; }
    public float negativeSlope() { return negativeSlope; }
    public float dropout() { return dropout; }

    public Parameter weight() { ensureInitialized(); return weight; }
    public Parameter attSrc() { ensureInitialized(); return attSrc; }
    public Parameter attDst() { ensureInitialized(); return attDst; }
    public Parameter bias() {
        if (!useBias) throw new IllegalStateException("GATConv has no bias");
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "GATConv{" +
            "inFeatures=" + inFeatures +
            ", outFeatures=" + outFeatures +
            ", heads=" + heads +
            ", concat=" + concat +
            ", dropout=" + dropout +
            ", initialized=" + (weight != null) +
            '}';
    }
}
