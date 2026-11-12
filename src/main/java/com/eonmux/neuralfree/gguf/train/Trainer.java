package com.eonmux.neuralfree.gguf.train;

import java.util.List;

/**
 * The computations a finetune run needs from a trainer: one window's forward pass and loss, one training step, and the
 * parameter arrays the optimizer steps and the writer saves. Both the host-array trainer ({@link TrainableModel}) and
 * the device trainer ({@link GpuTrainableModel}) implement it, so the runner, the optimizer and the command are the
 * same whichever backend executes the arithmetic. The parameters' arrays always read as the trainer's current values
 * and gradients, wherever the trainer keeps the authoritative copy.
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
     * The moments and arithmetic of a {@link FinetuneOptimizer} of {@code kind} over {@link #parameters()}, kept where
     * this trainer keeps its parameters. The default is the host update over the parameter arrays; the caller closes
     * the result.
     */
    default FinetuneOptimizer.Update newUpdate(FinetuneOptimizer.Kind kind, float weightDecay) {
        return FinetuneOptimizer.hostUpdate(parameters(), kind, weightDecay);
    }

    /**
     * The whole logits table of {@code tokens}: one row per position. Runs a forward pass and leaves the trainer in
     * that state.
     */
    float[][] logits(int[] tokens);

    @Override
    void close();
}
