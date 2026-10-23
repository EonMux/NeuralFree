package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 * GraphSAGE convolution: {@code x @ weightSelf + agg @ weightNeigh + bias}, optionally L2-normalized per node, where
 * {@code agg} is the mean (MEAN, and LSTM, which aggregates by mean) or the elementwise maximum (MAX_POOL) of the
 * incoming edges' source features. Every step is a {@link TensorOps} composition on the node features' backend.
 *
 * <p><b>Autograd.</b> The forward records every step, so {@code weightSelf}, {@code weightNeigh}, {@code bias} and the
 * node features (when they require grad) receive gradients; see {@link MessagePassing}. The MEAN aggregate passes
 * each node's gradient divided by its in-degree to every incoming message; the MAX_POOL aggregate passes it to the
 * first incoming message (in ascending edge order) equal to the maximum, and to none for a NaN maximum.</p>
 */
public final class GraphSAGEConv extends MessagePassing {
    public enum Aggregator {
        MEAN,
        MAX_POOL,
        LSTM
    }
    
    private final int inFeatures;
    private final int outFeatures;
    private final boolean useBias;
    private final Aggregator aggregator;
    private final boolean normalize;
    
    private Parameter weightSelf;
    private Parameter weightNeigh;
    private Parameter bias;

    public GraphSAGEConv(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, Aggregator.MEAN, true, true);
    }

    public GraphSAGEConv(int inFeatures, int outFeatures, Aggregator aggregator, 
                         boolean useBias, boolean normalize) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }
        
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.aggregator = Objects.requireNonNull(aggregator, "aggregator cannot be null");
        this.useBias = useBias;
        this.normalize = normalize;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        
        float stdSelf = (float) Math.sqrt(2.0 / (inFeatures + outFeatures));
        this.weightSelf = registerParameter("weight_self", normalParameter(backend,
            Shape.matrix(inFeatures, outFeatures), stdSelf, random.nextLong(), "weight_self"));
        
        float stdNeigh = (float) Math.sqrt(2.0 / (inFeatures + outFeatures));
        this.weightNeigh = registerParameter("weight_neigh", normalParameter(backend,
            Shape.matrix(inFeatures, outFeatures), stdNeigh, random.nextLong(), "weight_neigh"));
        
        if (useBias) {
            this.bias = registerParameter("bias", zeroParameter(backend, Shape.vector(outFeatures), "bias"));
        }
    }

    @Override
    protected Tensor message(Tensor srcFeatures, Tensor dstFeatures, GraphData graph) {
        return srcFeatures;
    }

    /** MEAN and LSTM: mean of the incoming messages; MAX_POOL: their elementwise maximum. Both give 0 without one. */
    @Override
    protected Tensor aggregate(Tensor messages, GraphData graph) {
        return aggregator == Aggregator.MAX_POOL ? scatterMax(messages, graph) : scatterMean(messages, graph);
    }

    /**
     * {@code x @ weightSelf + aggregated @ weightNeigh}, plus the bias, then optionally each row divided by its L2
     * norm plus {@code 1e-8}.
     */
    @Override
    protected Tensor update(Tensor aggregated, Tensor nodeFeatures) {
        ensureInitialized();
        Tensor result;
        try (Tensor self = TensorOps.matmul(nodeFeatures, weightSelf.tensor());
             Tensor neigh = TensorOps.matmul(aggregated, weightNeigh.tensor())) {
            result = TensorOps.add(self, neigh);
        }
        if (useBias) {
            result = consume(result, r -> TensorOps.add(r, bias.tensor()));
        }
        if (normalize) {
            result = consume(result, this::l2Normalize);
        }
        return result;
    }

    /** {@code step} applied to {@code input}, which is closed afterwards, also when the step fails. */
    private static Tensor consume(Tensor input, UnaryOperator<Tensor> step) {
        try {
            return step.apply(input);
        } finally {
            input.close();
        }
    }

    /** Each row divided by {@code sqrt(sum of squares) + 1e-8}: one float sqrt, add and division per row element. */
    private Tensor l2Normalize(Tensor x) {
        int numNodes = x.shape().dim(0);
        try (Tensor squares = TensorOps.mul(x, x);
             Tensor sumSquares = TensorOps.sum(squares, 1);
             Tensor norm = TensorOps.sqrt(sumSquares);
             Tensor shifted = TensorOps.add(norm, 1e-8f);
             Tensor column = shifted.reshape(numNodes, 1)) {
            return TensorOps.div(x, column);
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
        graph.requireNodeFeatures(x, "GraphSAGEConv");
        
        if (x.shape().dim(1) != inFeatures) {
            throw new IllegalArgumentException(
                "Input feature dimension must be " + inFeatures + ", got " + x.shape().dim(1));
        }
        
        return propagate(graph, x);
    }

    private void ensureInitialized() {
        if (weightSelf == null) {
            throw new IllegalStateException("GraphSAGEConv not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() { return inFeatures; }
    public int outFeatures() { return outFeatures; }
    public Aggregator aggregator() { return aggregator; }
    public boolean hasBias() { return useBias; }
    public boolean isNormalized() { return normalize; }

    public Parameter weightSelf() { ensureInitialized(); return weightSelf; }
    public Parameter weightNeigh() { ensureInitialized(); return weightNeigh; }
    public Parameter bias() {
        if (!useBias) throw new IllegalStateException("GraphSAGEConv has no bias");
        ensureInitialized();
        return bias;
    }

    @Override
    public String toString() {
        return "GraphSAGEConv{" +
            "inFeatures=" + inFeatures +
            ", outFeatures=" + outFeatures +
            ", aggregator=" + aggregator +
            ", normalize=" + normalize +
            ", initialized=" + (weightSelf != null) +
            '}';
    }
}
