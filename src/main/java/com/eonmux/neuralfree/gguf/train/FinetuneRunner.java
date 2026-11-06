package com.eonmux.neuralfree.gguf.train;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * The training loop of {@code llama-finetune}: for every epoch it steps the optimizer once per training window,
 * scores the validation windows without touching the gradients, and reports the epoch's loss and accuracy.
 *
 * <p>The learning rate of epoch {@code e} (0-based) is the geometric interpolation between {@code lr0} and
 * {@code lrMin} that reaches {@code lrMin} after {@code decayEpochs} epochs,
 * {@code lr0 * (lrMin / lr0)^(e / decayEpochs)}, and stays at {@code lrMin} afterwards — llama.cpp's
 * {@code lr0}, {@code lr_min} and {@code decay_epochs}. A zero {@code decayEpochs} keeps {@code lr0} for the first
 * epoch and drops to {@code lrMin} for the rest. Windows run in corpus order, as llama.cpp trains its samples.</p>
 *
 * <p>The step of one window is one optimizer update: the gradients of the previous window are cleared, the window's
 * loss and gradients are computed by {@link TrainableModel#step}, and the weights move. Nothing is retained between
 * epochs, so the run needs one window's activations and the optimizer's moment buffers.</p>
 *
 * <p>Not thread-safe; the caller owns the model, the optimizer and the dataset.</p>
 */
public final class FinetuneRunner {

    /** The hyper-parameters of one run: llama-finetune's {@code -epochs}, {@code -lr0}, {@code -lr-min}, ... . */
    public static final class Config {
        private final int epochs;
        private final float lr0;
        private final float lrMin;
        private final float decayEpochs;
        private final float weightDecay;
        private final FinetuneOptimizer.Kind optimizer;

        /**
         * A validated configuration.
         *
         * @param epochs      the number of epochs, at least 1
         * @param lr0         the learning rate of the first epoch, positive and finite
         * @param lrMin       the floor the schedule decays to, positive, finite and not above {@code lr0}
         * @param decayEpochs the epochs the schedule takes to reach {@code lrMin}, zero or more
         * @param weightDecay the decay coefficient, zero or more
         * @param optimizer   the update rule
         * @throws IllegalArgumentException naming the flag of an invalid value
         */
        public Config(int epochs, float lr0, float lrMin, float decayEpochs, float weightDecay,
                      FinetuneOptimizer.Kind optimizer) {
            if (epochs < 1) {
                throw new IllegalArgumentException("-epochs must be at least 1, not " + epochs);
            }
            requirePositive("--learning-rate", lr0);
            requirePositive("--learning-rate-min", lrMin);
            if (lrMin > lr0) {
                throw new IllegalArgumentException("--learning-rate-min " + lrMin + " is above --learning-rate "
                        + lr0 + "; the schedule decays toward the minimum and never rises");
            }
            if (!(decayEpochs >= 0f) || !Float.isFinite(decayEpochs)) {
                throw new IllegalArgumentException("--decay-epochs must be finite and non-negative, not "
                        + decayEpochs);
            }
            if (!(weightDecay >= 0f) || !Float.isFinite(weightDecay)) {
                throw new IllegalArgumentException("--weight-decay must be finite and non-negative, not "
                        + weightDecay);
            }
            this.epochs = epochs;
            this.lr0 = lr0;
            this.lrMin = lrMin;
            this.decayEpochs = decayEpochs;
            this.weightDecay = weightDecay;
            this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        }

        /** The number of passes over the training windows. */
        public int epochs() {
            return epochs;
        }

        /** The learning rate of the first epoch. */
        public float lr0() {
            return lr0;
        }

        /** The learning rate the schedule decays to. */
        public float lrMin() {
            return lrMin;
        }

        /** The epochs the schedule takes to reach {@link #lrMin()}. */
        public float decayEpochs() {
            return decayEpochs;
        }

        /** The weight decay of every update. */
        public float weightDecay() {
            return weightDecay;
        }

        /** The update rule. */
        public FinetuneOptimizer.Kind optimizer() {
            return optimizer;
        }

        /** The learning rate of {@code epoch}, counted from zero. */
        public float learningRate(int epoch) {
            if (epoch <= 0) {
                return lr0;
            }
            if (decayEpochs <= 0f || epoch >= decayEpochs) {
                return lrMin;
            }
            double ratio = (double) lrMin / lr0;
            return (float) (lr0 * Math.pow(ratio, epoch / (double) decayEpochs));
        }
    }

    /** One epoch's outcome: its mean training loss and accuracy, the validation's, and the learning rate it used. */
    public static final class EpochStats {
        private final int epoch;
        private final float learningRate;
        private final double trainLoss;
        private final double trainAccuracy;
        private final double validationLoss;
        private final double validationAccuracy;
        private final int windows;

        EpochStats(int epoch, float learningRate, double trainLoss, double trainAccuracy, int windows,
                   double validationLoss, double validationAccuracy) {
            this.epoch = epoch;
            this.learningRate = learningRate;
            this.trainLoss = trainLoss;
            this.trainAccuracy = trainAccuracy;
            this.windows = windows;
            this.validationLoss = validationLoss;
            this.validationAccuracy = validationAccuracy;
        }

        /** The epoch, counted from 1. */
        public int epoch() {
            return epoch;
        }

        /** The learning rate the epoch's updates used. */
        public float learningRate() {
            return learningRate;
        }

        /** Mean cross entropy over the training windows, the loss before each window's update. */
        public double trainLoss() {
            return trainLoss;
        }

        /** Fraction of the training positions whose argmax was the label. */
        public double trainAccuracy() {
            return trainAccuracy;
        }

        /** Mean cross entropy over the validation windows; NaN when the split left none. */
        public double validationLoss() {
            return validationLoss;
        }

        /** Fraction of the validation positions whose argmax was the label; NaN when there are none. */
        public double validationAccuracy() {
            return validationAccuracy;
        }

        /** The training windows the epoch stepped on. */
        public int windows() {
            return windows;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "epoch %d: lr=%.6g train loss %.6f (acc %.2f%%)%s", epoch,
                    learningRate, trainLoss, 100.0 * trainAccuracy, validationText());
        }

        private String validationText() {
            if (Double.isNaN(validationLoss)) {
                return "";
            }
            return String.format(Locale.ROOT, ", validation loss %.6f (acc %.2f%%)", validationLoss,
                    100.0 * validationAccuracy);
        }
    }

    private FinetuneRunner() {
    }

    /**
     * Trains {@code model} over {@code data} with {@code config}.
     *
     * @param listener called after every epoch with that epoch's statistics; may be null
     * @return one entry per epoch, in order
     */
    public static List<EpochStats> run(Trainer model, FinetuneDataset data, Config config,
                                       Consumer<EpochStats> listener) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(config, "config");
        FinetuneOptimizer optimizer = new FinetuneOptimizer(model, config.optimizer(), config.weightDecay());
        List<EpochStats> epochs = new ArrayList<>(config.epochs());
        for (int epoch = 0; epoch < config.epochs(); epoch++) {
            EpochStats stats = trainEpoch(model, data, config, optimizer, epoch + 1);
            epochs.add(stats);
            if (listener != null) {
                listener.accept(stats);
            }
        }
        return epochs;
    }

    /** One pass over the training windows, then the validation pass. */
    private static EpochStats trainEpoch(Trainer model, FinetuneDataset data, Config config,
                                         FinetuneOptimizer optimizer, int epoch) {
        float learningRate = config.learningRate(epoch - 1);
        double loss = 0.0;
        long correct = 0;
        long positions = 0;
        List<FinetuneDataset.Window> windows = data.training();
        for (FinetuneDataset.Window window : windows) {
            model.zeroGrad();
            TrainableModel.StepResult result = model.step(window.rawTokens(), window.rawLabels());
            optimizer.step(learningRate);
            loss += result.loss();
            correct += result.correct();
            positions += result.predictions();
        }
        double validationLoss = Double.NaN;
        double validationAccuracy = Double.NaN;
        List<FinetuneDataset.Window> validation = data.validation();
        if (!validation.isEmpty()) {
            double sum = 0.0;
            long hit = 0;
            long count = 0;
            for (FinetuneDataset.Window window : validation) {
                TrainableModel.StepResult result = model.forward(window.rawTokens(), window.rawLabels());
                sum += result.loss();
                hit += result.correct();
                count += result.predictions();
            }
            validationLoss = sum / validation.size();
            validationAccuracy = count == 0 ? Double.NaN : (double) hit / count;
        }
        return new EpochStats(epoch, learningRate, loss / windows.size(),
                positions == 0 ? Double.NaN : (double) correct / positions, windows.size(), validationLoss,
                validationAccuracy);
    }

    private static void requirePositive(String flag, float value) {
        if (!(value > 0f) || !Float.isFinite(value)) {
            throw new IllegalArgumentException(flag + " must be positive and finite, not " + value);
        }
    }
}
