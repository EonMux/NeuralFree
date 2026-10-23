package com.eonmux.neuralfree.training;

import java.util.List;
import java.util.Objects;

/**
 * The settings of a training loop ({@link Trainer}, {@code MixedPrecisionTrainer}): gradient clipping, gradient
 * accumulation, the validation interval, the trainer's own progress log and the checkpoint interval. Immutable; built
 * with {@link #builder()}.
 */
public final class TrainerConfig {
    private final float gradientClipNorm;
    private final float gradientClipValue;
    private final int accumulationSteps;
    private final int logEveryNSteps;
    private final int evalEveryNEpochs;
    private final int checkpointEveryNEpochs;
    private final boolean verbose;

    private TrainerConfig(Builder builder) {
        this.gradientClipNorm = builder.gradientClipNorm;
        this.gradientClipValue = builder.gradientClipValue;
        this.accumulationSteps = builder.accumulationSteps;
        this.logEveryNSteps = builder.logEveryNSteps;
        this.evalEveryNEpochs = builder.evalEveryNEpochs;
        this.checkpointEveryNEpochs = builder.checkpointEveryNEpochs;
        this.verbose = builder.verbose;
    }

    public float gradientClipNorm() {
        return gradientClipNorm;
    }

    public float gradientClipValue() {
        return gradientClipValue;
    }

    /**
     * The number of batches whose gradients are accumulated before each optimizer step. The step applies their mean:
     * at each accumulation boundary, before clipping, every gradient is multiplied by {@code 1 / n}, where {@code n}
     * is the number of batches back-propagated since the previous step (this value for a full group, fewer for the
     * last group of an epoch or a group cut short by an early stop).
     */
    public int accumulationSteps() {
        return accumulationSteps;
    }

    /**
     * The interval of the step lines of the trainer's progress log ({@link #verbose()}, {@link Trainer.ProgressLog}):
     * a line after every batch whose step count (the count {@link Callback#onBatchEnd} receives) is a multiple of this
     * value. At least 1; default 10.
     */
    public int logEveryNSteps() {
        return logEveryNSteps;
    }

    /** Validation runs after every epoch {@code e} with {@code (e + 1) % evalEveryNEpochs == 0}; default 1. */
    public int evalEveryNEpochs() {
        return evalEveryNEpochs;
    }

    /**
     * The periodic checkpoint interval, as Keras' {@code ModelCheckpoint(save_freq)} and Lightning's
     * {@code ModelCheckpoint(every_n_epochs)} set it. When positive, every {@link ModelCheckpoint} callback of the
     * trainer also saves each epoch {@code e} with {@code (e + 1) % checkpointEveryNEpochs == 0}, in addition to the
     * epochs its mode selects; the trainer refuses to be built without a {@link ModelCheckpoint} callback, or with
     * one whose own {@link ModelCheckpoint#saveEveryNEpochs()} is positive and different
     * ({@link #validateCallbacks}). 0 (the default) leaves each {@link ModelCheckpoint} to its own interval.
     */
    public int checkpointEveryNEpochs() {
        return checkpointEveryNEpochs;
    }

    /**
     * Whether the trainer writes its own progress log ({@link Trainer.ProgressLog}): through the project logger at
     * INFO, a step line (epoch, step, the batch's loss and the learning rate) every {@link #logEveryNSteps()} steps
     * and an epoch line (training loss and, when validation ran, validation loss) at the end of every epoch. The log
     * is off for a trainer whose callbacks include a {@link ProgressLogger} with {@link ProgressLogger#verbose()} set,
     * which reports the same progress. Default true.
     */
    public boolean verbose() {
        return verbose;
    }

    public boolean gradientClippingEnabled() {
        return gradientClipNorm > 0 || gradientClipValue > 0;
    }

    /**
     * Checks that {@code callbacks}, the callbacks of a trainer built with this configuration, can carry out its
     * settings: with {@link #checkpointEveryNEpochs()} positive, at least one is a {@link ModelCheckpoint}, and none
     * is a {@link ModelCheckpoint} whose own positive {@link ModelCheckpoint#saveEveryNEpochs()} differs from it.
     *
     * @throws IllegalArgumentException when a check fails; the message names the setting
     */
    public void validateCallbacks(List<? extends Callback> callbacks) {
        Objects.requireNonNull(callbacks, "Callbacks cannot be null");
        if (checkpointEveryNEpochs == 0) {
            return;
        }
        boolean checkpointing = false;
        for (Callback callback : callbacks) {
            if (callback instanceof ModelCheckpoint) {
                ((ModelCheckpoint) callback).periodicInterval(checkpointEveryNEpochs);
                checkpointing = true;
            }
        }
        if (!checkpointing) {
            throw new IllegalArgumentException("checkpointEveryNEpochs is " + checkpointEveryNEpochs
                + " but the trainer has no ModelCheckpoint callback to save the checkpoints; add a ModelCheckpoint"
                + " callback or set checkpointEveryNEpochs to 0");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public static TrainerConfig defaults() {
        return builder().build();
    }

    public static final class Builder {
        private float gradientClipNorm = 0.0f;
        private float gradientClipValue = 0.0f;
        private int accumulationSteps = 1;
        private int logEveryNSteps = 10;
        private int evalEveryNEpochs = 1;
        private int checkpointEveryNEpochs = 0;
        private boolean verbose = true;

        public Builder() {
        }

        public Builder gradientClipNorm(float gradientClipNorm) {
            if (gradientClipNorm < 0) {
                throw new IllegalArgumentException("gradientClipNorm must be non-negative, got: " + gradientClipNorm);
            }
            this.gradientClipNorm = gradientClipNorm;
            return this;
        }

        public Builder gradientClipValue(float gradientClipValue) {
            if (gradientClipValue < 0) {
                throw new IllegalArgumentException("gradientClipValue must be non-negative, got: " + gradientClipValue);
            }
            this.gradientClipValue = gradientClipValue;
            return this;
        }

        /** See {@link TrainerConfig#accumulationSteps()}; at least 1 (the default: one step per batch). */
        public Builder accumulationSteps(int accumulationSteps) {
            if (accumulationSteps < 1) {
                throw new IllegalArgumentException("accumulationSteps must be at least 1, got: " + accumulationSteps);
            }
            this.accumulationSteps = accumulationSteps;
            return this;
        }

        /** See {@link TrainerConfig#logEveryNSteps()}; at least 1. */
        public Builder logEveryNSteps(int logEveryNSteps) {
            if (logEveryNSteps < 1) {
                throw new IllegalArgumentException("logEveryNSteps must be at least 1, got: " + logEveryNSteps);
            }
            this.logEveryNSteps = logEveryNSteps;
            return this;
        }

        /** See {@link TrainerConfig#evalEveryNEpochs()}; at least 1. */
        public Builder evalEveryNEpochs(int evalEveryNEpochs) {
            if (evalEveryNEpochs < 1) {
                throw new IllegalArgumentException("evalEveryNEpochs must be at least 1, got: " + evalEveryNEpochs);
            }
            this.evalEveryNEpochs = evalEveryNEpochs;
            return this;
        }

        /** See {@link TrainerConfig#checkpointEveryNEpochs()}; non-negative, 0 for none. */
        public Builder checkpointEveryNEpochs(int checkpointEveryNEpochs) {
            if (checkpointEveryNEpochs < 0) {
                throw new IllegalArgumentException("checkpointEveryNEpochs must be non-negative, got: "
                    + checkpointEveryNEpochs);
            }
            this.checkpointEveryNEpochs = checkpointEveryNEpochs;
            return this;
        }

        /** See {@link TrainerConfig#verbose()}. */
        public Builder verbose(boolean verbose) {
            this.verbose = verbose;
            return this;
        }

        public TrainerConfig build() {
            return new TrainerConfig(this);
        }
    }

    @Override
    public String toString() {
        return "TrainerConfig{" +
            "gradientClipNorm=" + gradientClipNorm +
            ", gradientClipValue=" + gradientClipValue +
            ", accumulationSteps=" + accumulationSteps +
            ", logEveryNSteps=" + logEveryNSteps +
            ", evalEveryNEpochs=" + evalEveryNEpochs +
            ", checkpointEveryNEpochs=" + checkpointEveryNEpochs +
            ", verbose=" + verbose +
            '}';
    }
}
