package com.eonmux.neuralfree.nn.graph;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.Parameter;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * Graph isomorphism convolution: {@code mlp((1 + eps) * x + sum of the incoming edges' source features)}, with a
 * two-layer ReLU MLP and a fixed or trainable {@code eps}. Every step is a {@link TensorOps} composition on the node
 * features' backend; a trainable {@code eps} is applied as a device scalar, never read back.
 *
 * <p><b>Autograd.</b> The forward records the clamped source gather, the segment sum, {@code 1 + eps}, the scaling,
 * the sum and the MLP, so the MLP parameters, a trainable {@code eps} (through the recorded
 * {@code TensorOps.add(eps, 1)} and the broadcasting multiply) and the node features (when they require grad) receive
 * gradients; see {@link MessagePassing}.</p>
 */
public final class GINConv extends MessagePassing {
    private final int inFeatures;
    private final int outFeatures;
    private final float eps;
    private final boolean trainEps;

    private Module mlp;
    private Parameter epsParam;

    public GINConv(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, 0.0f, false);
    }

    public GINConv(int inFeatures, int outFeatures, float eps, boolean trainEps) {
        if (inFeatures <= 0) {
            throw new IllegalArgumentException("inFeatures must be positive: " + inFeatures);
        }
        if (outFeatures <= 0) {
            throw new IllegalArgumentException("outFeatures must be positive: " + outFeatures);
        }

        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.eps = eps;
        this.trainEps = trainEps;
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        if (trainEps) {
            this.epsParam = registerParameter("eps", ownedParameter(Tensor.scalar(backend, eps), "eps"));
        }

        this.mlp = new SimpleMLP(inFeatures, outFeatures, backend, random);
        registerModule("mlp", mlp);
    }

    @Override
    protected Tensor message(Tensor srcFeatures, Tensor dstFeatures, GraphData graph) {
        return srcFeatures;
    }

    @Override
    protected Tensor aggregate(Tensor messages, GraphData graph) {
        return scatterAdd(messages, graph);
    }

    /**
     * {@code mlp((1 + eps) * x + aggregated)}: {@code 1 + eps} is one float add (on the device for a trainable
     * {@code eps}), then a float multiply and a float add per element.
     */
    @Override
    protected Tensor update(Tensor aggregated, Tensor nodeFeatures) {
        ensureInitialized();
        try (Tensor scaled = scaleSelf(nodeFeatures);
             Tensor combined = TensorOps.add(scaled, aggregated)) {
            return mlp.forward(combined);
        }
    }

    private Tensor scaleSelf(Tensor nodeFeatures) {
        if (trainEps && epsParam != null) {
            try (Tensor onePlusEps = TensorOps.add(epsParam.tensor(), 1.0f)) {
                return TensorOps.mul(nodeFeatures, onePlusEps);
            }
        }
        return TensorOps.mul(nodeFeatures, 1 + eps);
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
        graph.requireNodeFeatures(x, "GINConv");

        if (x.shape().dim(1) != inFeatures) {
            throw new IllegalArgumentException(
                "Input feature dimension must be " + inFeatures + ", got " + x.shape().dim(1));
        }

        return propagate(graph, x);
    }

    private void ensureInitialized() {
        if (mlp == null) {
            throw new IllegalStateException("GINConv not initialized. Call initialize(backend) first.");
        }
    }

    public int inFeatures() { return inFeatures; }
    public int outFeatures() { return outFeatures; }
    /** The fixed {@code eps}, or the initial value of a trainable one (its current value is the "eps" parameter). */
    public float eps() { return eps; }
    public boolean isEpsTrainable() { return trainEps; }
    public Module mlp() { ensureInitialized(); return mlp; }

    @Override
    public String toString() {
        return "GINConv{" +
            "inFeatures=" + inFeatures +
            ", outFeatures=" + outFeatures +
            ", eps=" + eps +
            ", trainEps=" + trainEps +
            ", initialized=" + (mlp != null) +
            '}';
    }

    /** {@code relu(x @ weight1 + bias1) @ weight2 + bias2}, hidden width {@code max(in, out)}. */
    private static class SimpleMLP extends Module {
        private final Parameter weight1;
        private final Parameter bias1;
        private final Parameter weight2;
        private final Parameter bias2;

        SimpleMLP(int inFeatures, int outFeatures, ComputeBackend backend, Random random) {
            int hiddenDim = Math.max(inFeatures, outFeatures);

            float std1 = (float) Math.sqrt(2.0 / (inFeatures + hiddenDim));
            this.weight1 = registerParameter("weight1", normalParameter(backend,
                Shape.matrix(inFeatures, hiddenDim), std1, random.nextLong(), "weight1"));
            this.bias1 = registerParameter("bias1", zeroParameter(backend, Shape.vector(hiddenDim), "bias1"));

            float std2 = (float) Math.sqrt(2.0 / (hiddenDim + outFeatures));
            this.weight2 = registerParameter("weight2", normalParameter(backend,
                Shape.matrix(hiddenDim, outFeatures), std2, random.nextLong(), "weight2"));
            this.bias2 = registerParameter("bias2", zeroParameter(backend, Shape.vector(outFeatures), "bias2"));
        }

        @Override
        public Tensor forward(Tensor input) {
            Objects.requireNonNull(input, "input cannot be null");
            try (Tensor h1 = TensorOps.matmul(input, weight1.tensor());
                 Tensor h1b = TensorOps.add(h1, bias1.tensor());
                 Tensor hidden = TensorOps.relu(h1b);
                 Tensor h2 = TensorOps.matmul(hidden, weight2.tensor())) {
                return TensorOps.add(h2, bias2.tensor());
            }
        }
    }
}
