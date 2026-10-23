package com.eonmux.neuralfree.training;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.nn.Module;
import com.eonmux.neuralfree.utils.LoggingUtil;
import org.apache.logging.log4j.Logger;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * Stops training when a monitored quantity has stopped improving, with the semantics of Keras'
 * {@code EarlyStopping} callback.
 *
 * <p><b>Monitored value.</b> {@value #MONITOR_VAL_LOSS} (the default) is the epoch's validation loss, {@value
 * #MONITOR_LOSS} its training loss. An epoch that ends without the monitored value (no validation ran in it) is
 * skipped: it neither improves nor counts toward the patience. When validation was due in that epoch
 * ({@link TrainerConfig#evalEveryNEpochs()}) but did not run, as for a trainer without a validation loader, a WARN is
 * logged once per run, as Keras warns that early stopping is conditioned on a metric that is not available.</p>
 *
 * <p><b>Improvement and stopping.</b> With mode {@code "min"} a value improves when {@code value + minDelta < best},
 * with {@code "max"} when {@code value - minDelta > best}; {@code best} starts at positive or negative infinity, so the
 * first finite value improves and NaN never does. Each epoch with a value that does not improve increments the wait
 * count, an improvement resets it to 0, and once it reaches {@code patience} at an epoch after the first the trainer
 * is asked to stop ({@link TrainingContext#requestStop()}): no further epoch starts.</p>
 *
 * <p><b>Restoring the best weights</b> ({@code restoreBestWeights}, Keras' {@code restore_best_weights}). The model's
 * state ({@link Module#stateDict()}: every parameter and persistent buffer, such as BatchNorm running statistics, as
 * host copies) is snapshotted at the end of every improving epoch, and at the end of the first epoch with a value
 * when that epoch does not improve. When the run ends ({@link #onTrainEnd}), stopped early or not, the snapshot is
 * loaded back into the model ({@link Module#loadStateDict(Map, boolean)}, strict), as Keras restores in
 * {@code on_train_end}; without a snapshot (no epoch had the monitored value) the model is left as it is. The
 * snapshot is released after the restore.</p>
 *
 * <p>The stop and the restore are logged at INFO through the project logger. Every run starts afresh at
 * {@link #onTrainBegin}.</p>
 */
public final class EarlyStopping implements Callback {
    /** Monitors the validation loss (the default). */
    public static final String MONITOR_VAL_LOSS = "val_loss";
    /** Monitors the training loss. */
    public static final String MONITOR_LOSS = "loss";
    public static final String MODE_MIN = "min";
    public static final String MODE_MAX = "max";

    private static final Logger LOGGER = LoggingUtil.getLogger(EarlyStopping.class);

    private final String monitor;
    private final int patience;
    private final float minDelta;
    private final String mode;
    private final boolean restoreBestWeights;

    private int wait = 0;
    private float best;
    private int bestEpoch = -1;
    private int stoppedEpoch = -1;
    private Map<String, Object> bestWeights;
    private int bestWeightsEpoch = -1;
    private boolean unavailableWarned = false;
    private TrainingContext trainer;

    /** Patience 10, as {@link #builder()} defaults otherwise. */
    public EarlyStopping() {
        this(builder());
    }

    /** {@code patience} epochs, as {@link #builder()} defaults otherwise. */
    public EarlyStopping(int patience) {
        this(builder().patience(patience));
    }

    /** The validation loss is monitored; see {@link Builder} for the settings. */
    public EarlyStopping(int patience, float minDelta, String mode, boolean restoreBestWeights) {
        this(builder().patience(patience).minDelta(minDelta).mode(mode).restoreBestWeights(restoreBestWeights));
    }

    private EarlyStopping(Builder builder) {
        if (!MONITOR_VAL_LOSS.equals(builder.monitor) && !MONITOR_LOSS.equals(builder.monitor)) {
            throw new IllegalArgumentException("Monitor must be '" + MONITOR_VAL_LOSS + "' or '" + MONITOR_LOSS
                + "', got: " + builder.monitor);
        }
        if (builder.patience < 1) {
            throw new IllegalArgumentException("Patience must be at least 1, got: " + builder.patience);
        }
        if (!(builder.minDelta >= 0) || Float.isInfinite(builder.minDelta)) {
            throw new IllegalArgumentException("minDelta must be finite and non-negative, got: " + builder.minDelta);
        }
        if (!MODE_MIN.equals(builder.mode) && !MODE_MAX.equals(builder.mode)) {
            throw new IllegalArgumentException("Mode must be 'min' or 'max', got: " + builder.mode);
        }
        this.monitor = builder.monitor;
        this.patience = builder.patience;
        this.minDelta = builder.minDelta;
        this.mode = builder.mode;
        this.restoreBestWeights = builder.restoreBestWeights;
        this.best = initialBest();
    }

    /**
     * Settings with the defaults: monitor {@value #MONITOR_VAL_LOSS}, patience 10, minDelta 0, mode "min", no
     * restore.
     */
    public static Builder builder() {
        return new Builder();
    }

    private float initialBest() {
        return MODE_MIN.equals(mode) ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
    }

    @Override
    public void onTrainBegin(TrainingContext trainer) {
        this.trainer = trainer;
        this.wait = 0;
        this.best = initialBest();
        this.bestEpoch = -1;
        this.stoppedEpoch = -1;
        this.bestWeights = null;
        this.bestWeightsEpoch = -1;
        this.unavailableWarned = false;
    }

    /**
     * Applies the rules of the class description to the epoch's monitored value.
     *
     * @throws IllegalStateException when {@code restoreBestWeights} is set and the run began without a trainer
     *                               ({@link #onTrainBegin} with null), so there is no model to snapshot
     */
    @Override
    public void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
        Float current = MONITOR_LOSS.equals(monitor) ? Float.valueOf(trainLoss) : valLoss;
        if (current == null) {
            warnUnavailable(epoch);
            return;
        }
        float value = current;
        if (improves(value)) {
            best = value;
            bestEpoch = epoch;
            wait = 0;
            if (restoreBestWeights) {
                takeSnapshot(epoch);
            }
            return;
        }
        if (restoreBestWeights && bestWeights == null) {
            takeSnapshot(epoch);
        }
        wait++;
        if (wait >= patience && epoch > 0) {
            stoppedEpoch = epoch;
            if (bestEpoch >= 0) {
                LOGGER.info("Epoch {}: early stopping; {} did not improve on {} (epoch {}) for {} epochs", epoch + 1,
                    monitor, best, bestEpoch + 1, wait);
            } else {
                LOGGER.info("Epoch {}: early stopping; {} had no improving value in {} epochs", epoch + 1, monitor,
                    wait);
            }
            if (trainer != null) {
                trainer.requestStop();
            }
        }
    }

    /**
     * Loads the snapshot of the best epoch back into the model when {@code restoreBestWeights} is set and a snapshot
     * was taken, then releases it.
     *
     * @throws IllegalArgumentException when the model's state no longer matches the snapshot
     *                                  ({@link Module#loadStateDict(Map, boolean)}, strict)
     */
    @Override
    public void onTrainEnd(TrainingResult result) {
        if (!restoreBestWeights || bestWeights == null) {
            return;
        }
        Map<String, Object> snapshot = bestWeights;
        bestWeights = null;
        LOGGER.info("Restoring model weights from the end of the best epoch: {}", bestWeightsEpoch + 1);
        trainer.model().loadStateDict(snapshot, true);
    }

    /** Whether {@code value} improves on the best so far by more than {@code minDelta}; NaN never does. */
    private boolean improves(float value) {
        if (MODE_MIN.equals(mode)) {
            return value + minDelta < best;
        }
        return value - minDelta > best;
    }

    /** Keeps a host copy of the model's state at the end of {@code epoch}, recording nothing for autograd. */
    private void takeSnapshot(int epoch) {
        if (trainer == null) {
            throw new IllegalStateException("EarlyStopping with restoreBestWeights needs the trainer's model; the run "
                + "began without a trainer (onTrainBegin with null)");
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            bestWeights = Collections.unmodifiableMap(trainer.model().stateDict());
        }
        bestWeightsEpoch = epoch;
    }

    /**
     * Logs the WARN of an unavailable monitored value once per run, when validation was due in {@code epoch} (always
     * when the run began without a trainer, whose validation interval is unknown).
     */
    private void warnUnavailable(int epoch) {
        if (unavailableWarned) {
            return;
        }
        if (trainer != null && (epoch + 1) % trainer.config().evalEveryNEpochs() != 0) {
            return;
        }
        unavailableWarned = true;
        LOGGER.warn("Early stopping conditioned on metric '{}' which is not available: no validation ran in epoch {}."
            + " Available metrics are: {}. Epochs without it neither improve nor count toward the patience; "
            + "provide a validation loader or monitor '{}'.", monitor, epoch + 1, MONITOR_LOSS, MONITOR_LOSS);
    }

    /** {@value #MONITOR_VAL_LOSS} or {@value #MONITOR_LOSS}. */
    public String monitor() {
        return monitor;
    }

    public int patience() {
        return patience;
    }

    public float minDelta() {
        return minDelta;
    }

    public String mode() {
        return mode;
    }

    public boolean restoreBestWeights() {
        return restoreBestWeights;
    }

    /** The best monitored value of the current run, or null before an epoch improved. */
    public Float bestLoss() {
        return bestEpoch >= 0 ? Float.valueOf(best) : null;
    }

    /** The epoch (from 0) of the best monitored value of the current run, or -1 before an epoch improved. */
    public int bestEpoch() {
        return bestEpoch;
    }

    /** The epoch (from 0) at whose end the current run was asked to stop, or -1 when it was not. */
    public int stoppedEpoch() {
        return stoppedEpoch;
    }

    /** The number of epochs with a monitored value since the last improvement. */
    public int waitCount() {
        return wait;
    }

    @Override
    public String name() {
        return "EarlyStopping";
    }

    @Override
    public String toString() {
        return "EarlyStopping{" +
            "monitor='" + monitor + '\'' +
            ", patience=" + patience +
            ", minDelta=" + minDelta +
            ", mode='" + mode + '\'' +
            ", restoreBestWeights=" + restoreBestWeights +
            ", wait=" + wait +
            ", bestLoss=" + bestLoss() +
            ", bestEpoch=" + bestEpoch +
            ", stoppedEpoch=" + stoppedEpoch +
            '}';
    }

    /** Settings of an {@link EarlyStopping}; {@link #build()} validates them. */
    public static final class Builder {
        private String monitor = MONITOR_VAL_LOSS;
        private int patience = 10;
        private float minDelta = 0.0f;
        private String mode = MODE_MIN;
        private boolean restoreBestWeights = false;

        private Builder() {
        }

        /** {@value EarlyStopping#MONITOR_VAL_LOSS} (the default) or {@value EarlyStopping#MONITOR_LOSS}. */
        public Builder monitor(String monitor) {
            this.monitor = Objects.requireNonNull(monitor, "Monitor cannot be null");
            return this;
        }

        /** The number of epochs without improvement after which training stops; at least 1, default 10. */
        public Builder patience(int patience) {
            this.patience = patience;
            return this;
        }

        /** The amount a value must beat the best by to count as an improvement; finite, non-negative, default 0. */
        public Builder minDelta(float minDelta) {
            this.minDelta = minDelta;
            return this;
        }

        /** {@code "min"} (the default) when lower values are better, {@code "max"} when higher values are. */
        public Builder mode(String mode) {
            this.mode = Objects.requireNonNull(mode, "Mode cannot be null");
            return this;
        }

        /** Restores the model's state of the best epoch when the run ends; default false. */
        public Builder restoreBestWeights(boolean restoreBestWeights) {
            this.restoreBestWeights = restoreBestWeights;
            return this;
        }

        /**
         * @throws IllegalArgumentException when the monitor is not {@value EarlyStopping#MONITOR_VAL_LOSS} or
         *                                  {@value EarlyStopping#MONITOR_LOSS}, the patience is below 1, minDelta is
         *                                  negative or not finite, or the mode is not {@code "min"} or {@code "max"}
         */
        public EarlyStopping build() {
            return new EarlyStopping(this);
        }
    }
}
