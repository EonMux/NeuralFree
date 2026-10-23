package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * Graph convolution: each node sums the features of its incoming edges' sources (scaled by
 * {@link GraphData#symmetricNormalization()} when normalizing), then {@code @ weight + bias}. No self-loops are added.
 * The node features stay on their backend; only the graph structure is read to the host, once per
 * {@link GraphData}.
 *
 * <p><b>Autograd.</b> The forward records the clamped source gather, the multiply by the normalization column (a
 * constant), the segment sum, {@code matmul} and the bias add, so {@code weight}, {@code bias} and the node features
 * (when they require grad) receive gradients; see {@link MessagePassing}.</p>
 */
public final class GCNConv extends MessagePassing {
    private final int inFeatures;
    private final int outFeatures;
    private final boolean useBias;
    private final boolean normalize;

    private Parameter weight;
    private Parameter bias;

    public GCNConv(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true, true);
    }

    public GCNConv(int inFeatures, int outFeatures, boolean useBias, boolean normalize) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = useBias;
        this.normalize = normalize;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        float std = (float) Math.sqrt(2.0 / (inFeatures + outFeatures));
        this.weight = registerParameter("weight", normalParameter(backend, Shape.matrix(inFeatures, outFeatures),
            std, random.nextLong(), "weight"));

        if (useBias) {
            this.bias = registerParameter("bias", zeroParameter(backend, Shape.vector(outFeatures), "bias"));
        }
    }

    /**
     * With normalization every edge index must name a node of the graph.
     *
     * @throws IndexOutOfBoundsException with normalization, when an edge names a node outside the graph
     */
    @Override
    protected void requireSupportedEdges(GraphData graph) {
        if (normalize && !graph.indicesInRange()) {
            throw new IndexOutOfBoundsException("GCNConv normalization needs every edge index in [0, "
                + graph.numNodes() + "), the edge index of this graph has one outside");
        }
    }

    @Override
    protected Tensor message(Tensor srcFeatures, Tensor dstFeatures, GraphData graph) {
        return srcFeatures;
    }

    /**
     * Sum of the incoming messages in ascending edge order; with normalization each message is first scaled by the
     * edge's {@link GraphData#symmetricNormalization()} coefficient (a constant column).
     */
    @Override
    protected Tensor aggregate(Tensor messages, GraphData graph) {
        if (!normalize) {
            return scatterAdd(messages, graph);
        }
        Tensor norm = graph.symmetricNormalization();
        try (Tensor column = norm.reshape((int) graph.numEdges(), 1);
             Tensor scaled = TensorOps.mul(messages, column)) {
            return scatterAdd(scaled, graph);
        }
    }

    /** {@code aggregated @ weight}, plus the bias when the layer has one. */
    @Override
    protected Tensor update(Tensor aggregated, Tensor nodeFeatures) {
        ensureInitialized();
        Tensor result = TensorOps.matmul(aggregated, weight.tensor());
        if (!useBias) {
            return result;
        }
        try {
            return TensorOps.add(result, bias.tensor());
        } finally {
            result.close();
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
        graph.requireNodeFeatures(x, "GCNConv");

        if (x.shape().dim(1) != inFeatures) {
            throw new IllegalArgumentException(
                "Input feature dimension must be " + inFeatures + ", got " + x.shape().dim(1));
        }

        return propagate(graph, x);
    }

    private void ensureInitialized() {
        if (weight == null) {
            throw new IllegalStateException("GCNConv not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() {
        return inFeatures;
    }

    public int outFeatures() {
        return outFeatures;
    }

    public boolean hasBias() {
        return useBias;
    }

    public boolean isNormalized() {
        return normalize;
    }

    public Parameter weight() {
        ensureInitialized();
        return weight;
    }

    public Parameter bias() {
        if (!useBias) {
            throw new IllegalStateException("This GCNConv has no bias");
        }
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "GCNConv{" +
            "inFeatures=" + inFeatures +
            ", outFeatures=" + outFeatures +
            ", useBias=" + useBias +
            ", normalize=" + normalize +
            ", initialized=" + (weight != null) +
            '}';
    }
}
