package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Linear;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.Objects;
import java.util.Random;

/**
 * Gated recurrent unit cell. The input-to-hidden and hidden-to-hidden layers each produce the fused gate tensor
 * {@code [batch, 3 * hiddenSize]} with the reset, update and candidate blocks in that order, and one step computes
 * <pre>
 *   r  = sigmoid(r_ih + r_hh)
 *   z  = sigmoid(z_ih + z_hh)
 *   n  = tanh(n_ih + r * n_hh)
 *   h' = z * h + (1 - z) * n
 * </pre>
 * from slice views of the fused gates, entirely through {@link TensorOps}. The slices and every operation are
 * recorded in grad mode, so the gradient of the new state reaches both layers' parameters, the input and
 * {@code hPrev}.
 */
public final class GRUCell extends Module {
    private static final int GATES = 3;

    private final int inputSize;
    private final int hiddenSize;
    private final boolean useBias;

    private final Linear ih;
    private final Linear hh;
    private boolean initialized;

    public GRUCell(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, true);
    }

    public GRUCell(int inputSize, int hiddenSize, boolean useBias) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.useBias = useBias;

        this.ih = new Linear(inputSize, GATES * hiddenSize, useBias);
        this.hh = new Linear(hiddenSize, GATES * hiddenSize, useBias);
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");

        ih.initialize(backend, random);
        registerModule("ih", ih);

        hh.initialize(backend, random);
        registerModule("hh", hh);
        initialized = true;
    }

    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * One step for {@code input} with last dimension {@code inputSize}; every other dimension is folded into the
     * batch. {@code hPrev} holds {@code batch * hiddenSize} elements (zeros when null). Returns a new
     * {@code [batch, hiddenSize]} tensor. In grad mode the result is the output of a recorded node:
     * it stays valid until a loss computed from it is back-propagated, which closes it. To carry a state past that
     * backward pass (truncated back-propagation through time), take its {@link Tensor#detach()} before the backward
     * pass.
     */
    public Tensor forward(Tensor input, Tensor hPrev) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");

        Shape inputShape = input.shape();
        if (inputShape.rank() < 1) {
            throw new IllegalArgumentException("Input must have at least 1 dimension");
        }

        int lastDim = inputShape.dim(inputShape.rank() - 1);
        if (lastDim != inputSize) {
            throw new IllegalArgumentException("Input last dimension must be " + inputSize + ", got " + lastDim);
        }

        int batchSize = Math.toIntExact(inputShape.numElements() / inputSize);
        try (Tensor ihGates = ih.forward(input);
             Tensor ihRows = ihGates.reshape(Shape.matrix(batchSize, GATES * hiddenSize));
             Tensor initial = hPrev == null ? zerosHidden(input.backend(), batchSize) : null) {
            return step(ihRows, hPrev != null ? hPrev : initial);
        }
    }

    /**
     * One step from the input-side gates {@code ihGates} ({@code [batch, 3 * hiddenSize]}, any strides) and the
     * previous hidden state ({@code batch * hiddenSize} elements). Returns a new {@code [batch, hiddenSize]} tensor;
     * the arguments are not closed.
     */
    Tensor step(Tensor ihGates, Tensor hPrev) {
        ensureInitialized();
        int batchSize = ihGates.shape().dim(0);
        try (Tensor h = RNN.Sequences.rows(hPrev, batchSize, hiddenSize, "hPrev");
             Tensor hhGates = hh.forward(h);
             Tensor rIh = gate(ihGates, 0);
             Tensor zIh = gate(ihGates, 1);
             Tensor nIh = gate(ihGates, 2);
             Tensor rHh = gate(hhGates, 0);
             Tensor zHh = gate(hhGates, 1);
             Tensor nHh = gate(hhGates, 2);
             Tensor rPre = TensorOps.add(rIh, rHh);
             Tensor r = TensorOps.sigmoid(rPre);
             Tensor zPre = TensorOps.add(zIh, zHh);
             Tensor z = TensorOps.sigmoid(zPre);
             Tensor resetHh = TensorOps.mul(r, nHh);
             Tensor nPre = TensorOps.add(nIh, resetHh);
             Tensor n = TensorOps.tanh(nPre);
             Tensor keep = TensorOps.mul(z, h);
             Tensor negZ = TensorOps.neg(z);
             Tensor oneMinusZ = TensorOps.add(negZ, 1.0f);
             Tensor update = TensorOps.mul(oneMinusZ, n)) {
            return TensorOps.add(keep, update);
        }
    }

    /** The view of gate block {@code index} of a fused {@code [batch, 3 * hiddenSize]} gate tensor. */
    private Tensor gate(Tensor gates, int index) {
        return gates.slice(1, (long) index * hiddenSize, (long) (index + 1) * hiddenSize);
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("GRUCell not initialized. Call initialize(backend) first.");
        }
    }

    public int inputSize() {
        return inputSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public boolean hasBias() {
        return useBias;
    }

    public Linear ihLayer() {
        ensureInitialized();
        return ih;
    }

    public Linear hhLayer() {
        ensureInitialized();
        return hh;
    }

    public Tensor zerosHidden(ComputeBackend backend, int batchSize) {
        return Tensor.zeros(backend, Shape.matrix(batchSize, hiddenSize));
    }

    @Override
    public String toString() {
        return "GRUCell{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", useBias=" + useBias +
            ", initialized=" + initialized +
            '}';
    }
}
