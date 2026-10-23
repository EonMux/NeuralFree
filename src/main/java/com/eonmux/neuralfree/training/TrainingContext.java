package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.nn.loss.LossFunction;
import com.eonmux.neuralfree.optim.Optimizer;

/**
 * The training loop a {@link Callback} is attached to, as {@link Callback#onTrainBegin} hands it over: the model,
 * optimizer and loss function being trained, the configuration of the loop, and the request that ends training early.
 * Implemented by every trainer that runs callbacks ({@link Trainer}, {@code MixedPrecisionTrainer}).
 */
public interface TrainingContext {

    /** The model being trained. */
    Module model();

    /** The optimizer that updates the model's parameters. */
    Optimizer optimizer();

    /** The loss function the training step minimizes. */
    LossFunction lossFunction();

    /**
     * The configuration the loop runs with. Callbacks read the settings that concern them: {@link ModelCheckpoint}
     * its periodic interval ({@link TrainerConfig#checkpointEveryNEpochs()}), {@link EarlyStopping} the validation
     * interval ({@link TrainerConfig#evalEveryNEpochs()}).
     */
    TrainerConfig config();

    /**
     * Ends the current training run early: no further batch is fetched and no further epoch starts. The epoch in
     * progress still applies its accumulated gradients, runs its validation and reaches {@link Callback#onEpochEnd},
     * and {@link Callback#onTrainEnd} is called. The request holds until the next run starts.
     */
    void requestStop();
}
