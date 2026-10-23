package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Multi-layer Elman RNN over {@code [batch, seqLen, inputSize]} input. Each layer projects the whole sequence through
 * its cells' input-to-hidden layer once, then steps through time in Java on device tensors:
 * {@code h_t = act(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)}, with {@code h_{-1} = 0} and the activation of
 * {@link RNNCell}. The hidden states of a direction are stacked into {@code [batch, seqLen, hiddenSize]}; a
 * bidirectional layer concatenates the forward and the time-reversed pass along the feature axis. Every step is a
 * recorded operation, so in grad mode the output carries the graph back to every cell parameter and to the input.
 */
public final class RNN extends Module {
    private final int inputSize;
    private final int hiddenSize;
    private final int numLayers;
    private final boolean bidirectional;
    private final float dropout;
    private final String activation;
    private final boolean useBias;

    private final List<RNNCell> forwardCells;
    private final List<RNNCell> backwardCells;
    private final List<Dropout> dropoutLayers;

    public RNN(int inputSize, int hiddenSize, int numLayers) {
        this(inputSize, hiddenSize, numLayers, false, 0.0, RNNCell.TANH, true);
    }

    public RNN(int inputSize, int hiddenSize, int numLayers, boolean bidirectional) {
        this(inputSize, hiddenSize, numLayers, bidirectional, 0.0, RNNCell.TANH, true);
    }

    public RNN(int inputSize, int hiddenSize, int numLayers, boolean bidirectional,
               double dropout, String activation, boolean useBias) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (dropout < 0.0 || dropout >= 1.0) {
            throw new IllegalArgumentException("dropout must be in [0, 1), got: " + dropout);
        }
        if (!RNNCell.TANH.equals(activation) && !RNNCell.RELU.equals(activation)) {
            throw new IllegalArgumentException("activation must be 'tanh' or 'relu', got: " + activation);
        }

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.bidirectional = bidirectional;
        this.dropout = (float) dropout;
        this.activation = activation;
        this.useBias = useBias;

        this.forwardCells = new ArrayList<>();
        this.backwardCells = bidirectional ? new ArrayList<>() : null;
        this.dropoutLayers = new ArrayList<>();
    }

    public void initialize(ComputeBackend backend) {
        initialize(backend, new Random());
    }

    public void initialize(ComputeBackend backend, Random random) {
        Objects.requireNonNull(backend, "Backend cannot be null");
        Objects.requireNonNull(random, "Random cannot be null");
        if (!forwardCells.isEmpty()) {
            throw new IllegalStateException("RNN is already initialized");
        }

        int firstLayerInputSize = inputSize;
        int subsequentInputSize = bidirectional ? 2 * hiddenSize : hiddenSize;

        for (int layer = 0; layer < numLayers; layer++) {
            int layerInputSize = (layer == 0) ? firstLayerInputSize : subsequentInputSize;

            RNNCell forwardCell = new RNNCell(layerInputSize, hiddenSize, useBias, activation);
            forwardCell.initialize(backend, random);
            forwardCells.add(forwardCell);
            registerModule("forward_" + layer, forwardCell);

            if (bidirectional) {
                RNNCell backwardCell = new RNNCell(layerInputSize, hiddenSize, useBias, activation);
                backwardCell.initialize(backend, random);
                backwardCells.add(backwardCell);
                registerModule("backward_" + layer, backwardCell);
            }
        }

        if (dropout > 0 && numLayers > 1) {
            for (int layer = 0; layer < numLayers - 1; layer++) {
                Dropout dropoutLayer = new Dropout(dropout);
                dropoutLayers.add(dropoutLayer);
                registerModule("dropout_" + layer, dropoutLayer);
            }
        }
    }

    /**
     * Runs every layer over {@code input} of shape {@code [batch, seqLen, inputSize]} and returns a new
     * {@code [batch, seqLen, numDirections * hiddenSize]} tensor. Dropout (training mode only) is applied between
     * layers. Intermediate tensors are closed; {@code input} is not.
     *
     * @throws IllegalArgumentException when {@code input} is not rank 3 with last dimension {@code inputSize}, or has
     *                                  an empty batch or sequence
     */
    @Override
    public Tensor forward(Tensor input) {
        ensureInitialized();
        Objects.requireNonNull(input, "Input cannot be null");
        int[] batchAndLength = Sequences.batchAndLength(input, inputSize, "RNN");
        int batchSize = batchAndLength[0];

        Tensor output = input;
        try {
            for (int layer = 0; layer < numLayers; layer++) {
                Tensor next = runLayer(layer, output, batchSize);
                output = Sequences.advance(output, next, input);
                if (dropout > 0 && layer < numLayers - 1) {
                    output = Sequences.advance(output, dropoutLayers.get(layer).forward(output), input);
                }
            }
            return output;
        } catch (RuntimeException e) {
            Sequences.release(output, input);
            throw e;
        }
    }

    private Tensor runLayer(int layer, Tensor layerInput, int batchSize) {
        Tensor forwardOutput = runDirection(forwardCells.get(layer), layerInput, batchSize, false);
        if (!bidirectional) {
            return forwardOutput;
        }
        Tensor backwardOutput;
        try {
            backwardOutput = runDirection(backwardCells.get(layer), layerInput, batchSize, true);
        } catch (RuntimeException e) {
            forwardOutput.close();
            throw e;
        }
        return Sequences.joinDirections(forwardOutput, backwardOutput);
    }

    /** One direction of one layer: the input projection over all steps, then the recurrence in time order. */
    private Tensor runDirection(RNNCell cell, Tensor layerInput, int batchSize, boolean reverse) {
        int seqLen = layerInput.shape().dim(1);
        Tensor[] steps = new Tensor[seqLen];
        try (Tensor projected = cell.ihLayer().forward(layerInput);
             Tensor initial = cell.zerosHidden(layerInput.backend(), batchSize)) {
            Tensor hidden = initial;
            for (int k = 0; k < seqLen; k++) {
                int t = reverse ? seqLen - 1 - k : k;
                steps[t] = step(cell, projected, t, hidden);
                hidden = steps[t];
            }
            return Sequences.stackSteps(steps);
        } catch (RuntimeException e) {
            Sequences.closeAll(steps);
            throw e;
        }
    }

    /** {@link RNNCell#forward(Tensor, Tensor)} for step {@code t} with its input projection already computed. */
    private static Tensor step(RNNCell cell, Tensor projected, int t, Tensor hidden) {
        try (Tensor inputTerm = Sequences.timeStep(projected, t);
             Tensor recurrentTerm = cell.hhLayer().forward(hidden);
             Tensor preActivation = TensorOps.add(inputTerm, recurrentTerm)) {
            return RNNCell.TANH.equals(cell.activation())
                ? TensorOps.tanh(preActivation)
                : TensorOps.relu(preActivation);
        }
    }

    private void ensureInitialized() {
        if (forwardCells.isEmpty()) {
            throw new IllegalStateException("RNN not initialized. Call initialize(backend) first.");
        }
    }

    public int inputSize() {
        return inputSize;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int numLayers() {
        return numLayers;
    }

    public boolean isBidirectional() {
        return bidirectional;
    }

    public float dropoutRate() {
        return dropout;
    }

    public int numDirections() {
        return bidirectional ? 2 : 1;
    }

    @Override
    public String toString() {
        return "RNN{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", numLayers=" + numLayers +
            ", bidirectional=" + bidirectional +
            ", dropout=" + dropout +
            ", activation='" + activation + '\'' +
            ", initialized=" + !forwardCells.isEmpty() +
            '}';
    }

    /**
     * Sequence plumbing shared by the recurrent modules of this package (RNN, GRU, LSTM, StateSpaceModel): input
     * validation, per-step views, stacking of step outputs and lifetime handling of intermediates. Everything runs
     * through {@link TensorOps} and Tensor views, so it stays on the device when the train library is available,
     * takes the CPU reference path otherwise, and records every step in grad mode: a gradient reaching the stacked
     * sequence flows back through each step to the cells' parameters and to the module input.
     *
     * <p>Closing an intermediate here only releases this handle. In grad mode the graph routes gradients by node and
     * the recorded operations keep their own handles of what their backward reads, so intermediates are closed as
     * soon as they are consumed.</p>
     */
    static final class Sequences {

        private Sequences() {
        }

        /**
         * {@code {batch, seqLen}} of a {@code [batch, seqLen, inputSize]} input.
         *
         * @throws IllegalArgumentException when the input is not rank 3, its last dimension is not {@code inputSize},
         *                                  or the batch or the sequence is empty
         */
        static int[] batchAndLength(Tensor input, int inputSize, String module) {
            Shape shape = input.shape();
            if (shape.rank() != 3) {
                throw new IllegalArgumentException(module + " expects input [batch, seqLen, inputSize] (rank 3), got "
                    + shape);
            }
            if (shape.dim(2) != inputSize) {
                throw new IllegalArgumentException(module + " input last dimension must be " + inputSize + ", got "
                    + shape.dim(2));
            }
            if (shape.dim(0) <= 0 || shape.dim(1) <= 0) {
                throw new IllegalArgumentException(module + " needs a non-empty batch and sequence, got " + shape);
            }
            return new int[]{shape.dim(0), shape.dim(1)};
        }

        /**
         * {@code t} as a new {@code [rows, cols]} tensor: a view when {@code t} is dense, otherwise a contiguous copy.
         * The caller closes the result; {@code t} stays open.
         *
         * @throws IllegalArgumentException when {@code t} does not hold {@code rows * cols} elements
         */
        static Tensor rows(Tensor t, int rows, int cols, String what) {
            Objects.requireNonNull(t, what + " cannot be null");
            if (t.numElements() != (long) rows * cols) {
                throw new IllegalArgumentException(what + " must hold " + rows + " x " + cols + " elements, got shape "
                    + t.shape());
            }
            return t.reshape(Shape.matrix(rows, cols));
        }

        /**
         * The {@code [batch, features]} view of step {@code t} of a {@code [batch, seqLen, features]} tensor (a
         * recorded slice and squeeze when {@code sequence} requires grad in grad mode).
         */
        static Tensor timeStep(Tensor sequence, int t) {
            try (Tensor slab = sequence.slice(1, t, t + 1)) {
                return slab.squeeze(1);
            }
        }

        /**
         * The {@code [batch, features]} step outputs, indexed by time, stacked into a new
         * {@code [batch, seqLen, features]} tensor by {@link TensorOps#stack}, which records one node whose backward
         * hands every step its slice of the gradient. The step tensors are closed, also when stacking fails.
         */
        static Tensor stackSteps(Tensor[] steps) {
            try {
                return TensorOps.stack(Arrays.asList(steps), 1);
            } finally {
                closeAll(steps);
            }
        }

        /**
         * {@code [forward | backward]} along the feature axis as a new tensor ({@link TensorOps#cat}, recorded);
         * closes both inputs.
         */
        static Tensor joinDirections(Tensor forward, Tensor backward) {
            try {
                return TensorOps.cat(Arrays.asList(forward, backward), 2);
            } finally {
                forward.close();
                backward.close();
            }
        }

        /** {@code next}, after closing {@code current} unless it is {@code next} itself or the caller's input. */
        static Tensor advance(Tensor current, Tensor next, Tensor callerInput) {
            if (current != next) {
                release(current, callerInput);
            }
            return next;
        }

        /** Closes {@code t} unless it is the caller's input. */
        static void release(Tensor t, Tensor callerInput) {
            if (t != null && t != callerInput) {
                t.close();
            }
        }

        /** Closes every non-null tensor; closing is idempotent. */
        static void closeAll(Tensor[] tensors) {
            for (Tensor t : tensors) {
                if (t != null) {
                    t.close();
                }
            }
        }
    }
}
