package com.eonmux.neuralfree.gguf.train;

import java.util.List;

/**
 * The computations a finetune run needs from a trainer: one window's forward pass and loss, one training step, and the
 * parameter arrays the optimizer steps and the writer saves. Both the host-array trainer ({@link TrainableModel}) and
 * the device trainer ({@link GpuTrainableModel}) implement it, so the runner, the optimizer and the command are the
 * same whichever backend executes the arithmetic.
 */
public interface Trainer extends AutoCloseable {

    /** The trainable geometry of the model. */
    FinetunePlan plan();

    /** The length of the window one step runs. */
    int nCtx();

    /** The model width. */
    int nEmbd();

    /** The vocabulary size. */
    int nVocab();

    /** Every trainable parameter, each tied tensor once, in plan order. */
    List<TrainableModel.Parameter> parameters();

    /** The parameter of a stored tensor name, or null. */
    TrainableModel.Parameter parameter(String name);

    /**
     * A forward pass over {@code tokens} scored against {@code labels}, without touching the gradients.
     *
     * @throws IllegalArgumentException when either array is not a valid window of this trainer
     */
    TrainableModel.StepResult forward(int[] tokens, int[] labels);

    /**
     * A training step: the forward pass, then the backward pass accumulating every parameter's gradient. The caller
     * zeroes the gradients before the step and applies the optimizer after it.
     *
     * @throws IllegalArgumentException when either array is not a valid window of this trainer
     */
    TrainableModel.StepResult step(int[] tokens, int[] labels);

    /** Zeroes every parameter's gradient. */
    void zeroGrad();

    /**
     * The whole logits table of {@code tokens}: one row per position. Runs a forward pass and leaves the trainer in
     * that state.
     */
    float[][] logits(int[] tokens);

    @Override
    void close();
}
