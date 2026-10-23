package com.eonmux.neuralfree.nn.recurrent;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.layers.Dropout;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Multi-layer GRU over {@code [batch, seqLen, inputSize]} input. Each layer projects the whole sequence through its
 * cells' input-to-hidden layer once (the fused {@code [batch, seqLen, 3 * hiddenSize]} gates), then steps through
 * time in Java with {@link GRUCell#step} on device tensors, starting from a zero hidden state. The hidden states of a
 * direction are stacked into {@code [batch, seqLen, hiddenSize]}; a bidirectional layer concatenates the forward and
 * the time-reversed pass along the feature axis. Every step is a recorded operation (the gates are recorded slices of
 * the fused gate tensors), so in grad mode the output carries the graph back to every cell parameter and to the input.
 */
public final class GRU extends Module {
    private final int inputSize;
    private final int hiddenSize;
    private final int numLayers;
    private final boolean bidirectional;
    private final float dropout;
    private final boolean useBias;

    private final List<GRUCell> forwardCells;
    private final List<GRUCell> backwardCells;
    private final List<Dropout> dropoutLayers;

    public GRU(int inputSize, int hiddenSize, int numLayers) {
        this(inputSize, hiddenSize, numLayers, false, 0.0f, true);
    }

    public GRU(int inputSize, int hiddenSize, int numLayers, boolean bidirectional) {
        this(inputSize, hiddenSize, numLayers, bidirectional, 0.0f, true);
    }

    public GRU(int inputSize, int hiddenSize, int numLayers, boolean bidirectional,
               float dropout, boolean useBias) {
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be positive: " + inputSize);
        }
        if (hiddenSize <= 0) {
            throw new IllegalArgumentException("hiddenSize must be positive: " + hiddenSize);
        }
        if (numLayers <= 0) {
            throw new IllegalArgumentException("numLayers must be positive: " + numLayers);
        }
        if (dropout < 0.0f || dropout >= 1.0f) {
            throw new IllegalArgumentException("dropout must be in [0, 1), got: " + dropout);
        }

        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.numLayers = numLayers;
        this.bidirectional = bidirectional;
        this.dropout = dropout;
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
            throw new IllegalStateException("GRU is already initialized");
        }

        int firstLayerInputSize = inputSize;
        int subsequentInputSize = bidirectional ? 2 * hiddenSize : hiddenSize;

        for (int layer = 0; layer < numLayers; layer++) {
            int layerInputSize = (layer == 0) ? firstLayerInputSize : subsequentInputSize;

            GRUCell forwardCell = new GRUCell(layerInputSize, hiddenSize, useBias);
            forwardCell.initialize(backend, random);
            forwardCells.add(forwardCell);
            registerModule("forward_" + layer, forwardCell);

            if (bidirectional) {
                GRUCell backwardCell = new GRUCell(layerInputSize, hiddenSize, useBias);
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
        int[] batchAndLength = RNN.Sequences.batchAndLength(input, inputSize, "GRU");
        int batchSize = batchAndLength[0];

        Tensor output = input;
        try {
            for (int layer = 0; layer < numLayers; layer++) {
                Tensor next = runLayer(layer, output, batchSize);
                output = RNN.Sequences.advance(output, next, input);
                if (dropout > 0 && layer < numLayers - 1) {
                    output = RNN.Sequences.advance(output, dropoutLayers.get(layer).forward(output), input);
                }
            }
            return output;
        } catch (RuntimeException e) {
            RNN.Sequences.release(output, input);
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
        return RNN.Sequences.joinDirections(forwardOutput, backwardOutput);
    }

    /** One direction of one layer: the fused input gates over all steps, then the recurrence in time order. */
    private Tensor runDirection(GRUCell cell, Tensor layerInput, int batchSize, boolean reverse) {
        int seqLen = layerInput.shape().dim(1);
        Tensor[] steps = new Tensor[seqLen];
        try (Tensor ihGates = cell.ihLayer().forward(layerInput);
             Tensor initial = cell.zerosHidden(layerInput.backend(), batchSize)) {
            Tensor hidden = initial;
            for (int k = 0; k < seqLen; k++) {
                int t = reverse ? seqLen - 1 - k : k;
                try (Tensor stepGates = RNN.Sequences.timeStep(ihGates, t)) {
                    steps[t] = cell.step(stepGates, hidden);
                }
                hidden = steps[t];
            }
            return RNN.Sequences.stackSteps(steps);
        } catch (RuntimeException e) {
            RNN.Sequences.closeAll(steps);
            throw e;
        }
    }

    private void ensureInitialized() {
        if (forwardCells.isEmpty()) {
            throw new IllegalStateException("GRU not initialized. Call initialize(backend) first.");
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
        return "GRU{" +
            "inputSize=" + inputSize +
            ", hiddenSize=" + hiddenSize +
            ", numLayers=" + numLayers +
            ", bidirectional=" + bidirectional +
            ", dropout=" + dropout +
            ", initialized=" + !forwardCells.isEmpty() +
            '}';
    }
}
