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
 * Long short-term memory cell. The input-to-hidden and hidden-to-hidden layers each produce the fused gate tensor
 * {@code [batch, 4 * hiddenSize]} with the input, forget, cell and output blocks in that order, and one step computes
 * <pre>
 *   gates = gates_ih + gates_hh
 *   i = sigmoid(gates_i), f = sigmoid(gates_f), g = tanh(gates_g), o = sigmoid(gates_o)
 *   c' = f * c + i * g
 *   h' = o * tanh(c')
 * </pre>
 * from slice views of the fused gates, entirely through {@link TensorOps}. The slices and every operation are
 * recorded in grad mode, so the gradients of the new states reach both layers' parameters, the input, {@code hPrev}
 * and {@code cPrev}.
 */
public final class LSTMCell extends Module {
    private static final int GATES = 4;

    private final int inputSize;
    private final int hiddenSize;
    private final boolean useBias;

    private final Linear ih;
    private final Linear hh;
    private boolean initialized;

    public LSTMCell(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, true);
    }

    public LSTMCell(int inputSize, int hiddenSize, boolean useBias) {
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

    /** The new hidden state of {@link #forwardWithCell(Tensor)}; the new cell state is closed. */
    @Override
    public Tensor forward(Tensor input) {
        LSTMCellOutput out = forwardWithCell(input);
        out.cell().close();
        return out.hidden();
    }

    public LSTMCellOutput forwardWithCell(Tensor input) {
        return forwardWithCell(input, null, null);
    }

    /**
     * One step for {@code input} with last dimension {@code inputSize}; every other dimension is folded into the
     * batch. {@code hPrev} and {@code cPrev} hold {@code batch * hiddenSize} elements each (zeros when null). Returns
     * new {@code [batch, hiddenSize]} hidden and cell states, which the caller owns. In grad mode both states are
     * outputs of recorded nodes: they stay valid until a loss computed from them is back-propagated, which closes
     * them. To carry a state past that backward pass (truncated back-propagation through time), take its
     * {@link Tensor#detach()} before the backward pass.
     */
    public LSTMCellOutput forwardWithCell(Tensor input, Tensor hPrev, Tensor cPrev) {
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
             Tensor initialHidden = hPrev == null ? zerosHidden(input.backend(), batchSize) : null;
             Tensor initialCell = cPrev == null ? zerosCell(input.backend(), batchSize) : null) {
            return step(ihRows, hPrev != null ? hPrev : initialHidden, cPrev != null ? cPrev : initialCell);
        }
    }

    /**
     * One step from the input-side gates {@code ihGates} ({@code [batch, 4 * hiddenSize]}, any strides) and the
     * previous hidden and cell states ({@code batch * hiddenSize} elements each). Returns new
     * {@code [batch, hiddenSize]} states; the arguments are not closed.
     */
    LSTMCellOutput step(Tensor ihGates, Tensor hPrev, Tensor cPrev) {
        ensureInitialized();
        int batchSize = ihGates.shape().dim(0);
        try (Tensor h = RNN.Sequences.rows(hPrev, batchSize, hiddenSize, "hPrev");
             Tensor c = RNN.Sequences.rows(cPrev, batchSize, hiddenSize, "cPrev");
             Tensor hhGates = hh.forward(h);
             Tensor gates = TensorOps.add(ihGates, hhGates);
             Tensor iPre = gate(gates, 0);
             Tensor fPre = gate(gates, 1);
             Tensor gPre = gate(gates, 2);
             Tensor oPre = gate(gates, 3);
             Tensor i = TensorOps.sigmoid(iPre);
             Tensor f = TensorOps.sigmoid(fPre);
             Tensor g = TensorOps.tanh(gPre);
             Tensor o = TensorOps.sigmoid(oPre);
             Tensor kept = TensorOps.mul(f, c);
             Tensor written = TensorOps.mul(i, g)) {
            Tensor cNew = TensorOps.add(kept, written);
            try (Tensor cActivated = TensorOps.tanh(cNew)) {
                return new LSTMCellOutput(TensorOps.mul(o, cActivated), cNew);
            } catch (RuntimeException e) {
                cNew.close();
                throw e;
            }
        }
    }

    /** The view of gate block {@code index} of a fused {@code [batch, 4 * hiddenSize]} gate tensor. */
    private Tensor gate(Tensor gates, int index) {
        return gates.slice(1, (long) index * hiddenSize, (long) (index + 1) * hiddenSize);
    }

    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("LSTMCell not initialized. Call initialize(backend) first.");
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

    public Tensor zerosCell(ComputeBackend backend, int batchSize) {
        return Tensor.zeros(backend, Shape.matrix(batchSize, hiddenSize));
    }

    @Override
    public String toString() {
        return "LSTMCell{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", useBias=" + useBias +
            ", initialized=" + initialized +
            '}';
    }

    /**
     * The new hidden and cell states of one step, owned by the caller; in grad mode they are valid until a loss
     * computed from them is back-propagated ({@link #forwardWithCell(Tensor, Tensor, Tensor)}).
     */
    public static final class LSTMCellOutput {
        private final Tensor hidden;
        private final Tensor cell;

        public LSTMCellOutput(Tensor hidden, Tensor cell) {
            this.hidden = Objects.requireNonNull(hidden, "Hidden cannot be null");
            this.cell = Objects.requireNonNull(cell, "Cell cannot be null");
        }

        public Tensor hidden() {
            return hidden;
        }

        public Tensor cell() {
            return cell;
        }
    }
}
