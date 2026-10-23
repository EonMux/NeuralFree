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
 * Elman recurrent cell: {@code h' = act(x W_ih^T + b_ih + h W_hh^T + b_hh)} with {@code act} tanh or ReLU, through
 * two {@link Linear} layers and {@link TensorOps}. Every operation is recorded in grad mode, so the gradient of the new
 * state reaches both layers' parameters, the input and {@code hPrev}.
 */
public final class RNNCell extends Module {
    private final int inputSize;
    private final int hiddenSize;
    private final boolean useBias;
    private final String activation;
    
    private Linear ih;
    private Linear hh;
    
    public static final String TANH = "tanh";
    public static final String RELU = "relu";

    public RNNCell(int inputSize, int hiddenSize) {
        this(inputSize, hiddenSize, true, TANH);
    }

    public RNNCell(int inputSize, int hiddenSize, boolean useBias) {
        this(inputSize, hiddenSize, useBias, TANH);
    }

    public RNNCell(int inputSize, int hiddenSize, boolean useBias, String activation) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        if (!TANH.equals(activation) && !RELU.equals(activation)) {
            throw new IllegalArgumentException("activation must be 'tanh' or 'relu', got: " + activation);
        }
        
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.useBias = useBias;
        this.activation = activation;
        
        this.ih = new Linear(inputSize, hiddenSize, useBias);
        this.hh = new Linear(hiddenSize, hiddenSize, useBias);
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
    }

    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    /**
     * One step for {@code input} with last dimension {@code inputSize}; {@code hPrev} (null for a zero state) is any
     * tensor whose last dimension is {@code hiddenSize} and whose leading dimensions broadcast against the input's.
     * Returns a new tensor of the broadcast leading dimensions and {@code hiddenSize}. In grad mode the result is the
     * output of a recorded node: it stays valid until a loss computed from it is back-propagated, which closes it. To
     * carry a state past that backward pass (truncated back-propagation through time), take its
     * {@link Tensor#detach()} before the backward pass.
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
        
        Tensor preActivation = ih.forward(input);
        try {
            if (hPrev != null) {
                Tensor sum;
                try (Tensor recurrent = hh.forward(hPrev)) {
                    sum = TensorOps.add(preActivation, recurrent);
                }
                preActivation.close();
                preActivation = sum;
            }
            return TANH.equals(activation) ? TensorOps.tanh(preActivation) : TensorOps.relu(preActivation);
        } finally {
            preActivation.close();
        }
    }

    private void ensureInitialized() {
        if (ih == null || hh == null) {
            throw new IllegalStateException("RNNCell not initialized. Call initialize(backend) first.");
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

    public String activation() {
        return activation;
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
        return "RNNCell{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", useBias=" + useBias +
            ", activation='" + activation + '\'' +
            ", initialized=" + (ih != null) +
            '}';
    }
}
