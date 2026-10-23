package com.eonmux.neuralfree.training;

import java.util.Map;

/**
 * Prints training progress to standard output when {@code verbose}: a banner with the model, optimizer and loss
 * function at the start of a run and a summary at its end; every {@code logEveryNEpochs}-th epoch its number when it
 * begins and its duration, training loss and validation loss when it ends; after every batch whose step count (the
 * count {@link Callback#onBatchEnd} receives) is a multiple of {@code logEveryNSteps}, the batch's position in its
 * epoch and its loss; and, with {@code logMetrics}, the validation results. Without {@code verbose} it prints nothing.
 *
 * <p>A trainer with a verbose {@code ProgressLogger} among its callbacks leaves its own progress log off
 * ({@link Trainer.ProgressLog}), so progress is reported once.</p>
 */
public final class ProgressLogger implements Callback {
    private final int logEveryNEpochs;
    private final int logEveryNSteps;
    private final boolean logMetrics;
    private final boolean verbose;

    private long epochStartTime;
    private long trainStartTime;
    private int totalEpochs;
    private int totalBatches;
    private int epochBatches;

    public ProgressLogger() {
        this(1, 10, true, true);
    }

    public ProgressLogger(int logEveryNEpochs, int logEveryNSteps, boolean logMetrics, boolean verbose) {
        if (logEveryNEpochs < 1) {
            throw new IllegalArgumentException("logEveryNEpochs must be at least 1, got: " + logEveryNEpochs);
        }
        if (logEveryNSteps < 1) {
            throw new IllegalArgumentException("logEveryNSteps must be at least 1, got: " + logEveryNSteps);
        }

        this.logEveryNEpochs = logEveryNEpochs;
        this.logEveryNSteps = logEveryNSteps;
        this.logMetrics = logMetrics;
        this.verbose = verbose;
    }

    @Override
    public void onTrainBegin(TrainingContext trainer) {
        trainStartTime = System.currentTimeMillis();
        totalEpochs = 0;
        totalBatches = 0;
        
        if (verbose) {
            System.out.println("=== Training Started ===");
            System.out.println("Model: " + trainer.model().moduleName());
            System.out.println("Optimizer: " + trainer.optimizer().getClass().getSimpleName());
            System.out.println("Loss Function: " + trainer.lossFunction().name());
            System.out.println();
        }
    }

    @Override
    public void onTrainEnd(TrainingResult result) {
        if (verbose) {
            long totalTime = System.currentTimeMillis() - trainStartTime;
            System.out.println();
            System.out.println("=== Training Completed ===");
            System.out.println("Total epochs: " + result.epochs());
            System.out.println("Total steps: " + result.totalSteps());
            System.out.println("Training time: " + formatTime(totalTime));
            System.out.println("Final train loss: " + formatFloat(result.finalTrainLoss()));
            if (result.finalValLoss() != null) {
                System.out.println("Final val loss: " + formatFloat(result.finalValLoss()));
            }
            if (result.bestValLoss() != null) {
                System.out.println("Best val loss: " + formatFloat(result.bestValLoss()) + " (epoch " + result.bestEpoch() + ")");
            }
        }
    }

    @Override
    public void onEpochBegin(int epoch, int totalEpochs) {
        this.totalEpochs = totalEpochs;
        epochBatches = 0;
        epochStartTime = System.currentTimeMillis();
        
        if (verbose && (epoch + 1) % logEveryNEpochs == 0) {
            System.out.println("Epoch " + (epoch + 1) + "/" + totalEpochs);
        }
    }

    @Override
    public void onEpochEnd(int epoch, float trainLoss, Float valLoss) {
        if (verbose && (epoch + 1) % logEveryNEpochs == 0) {
            long epochTime = System.currentTimeMillis() - epochStartTime;
            System.out.println("  Epoch " + (epoch + 1) + " completed in " + formatTime(epochTime));
            System.out.println("  Train loss: " + formatFloat(trainLoss));
            if (valLoss != null) {
                System.out.println("  Val loss: " + formatFloat(valLoss));
            }
            System.out.println();
        }
    }

    @Override
    public void onBatchBegin(int batch, int totalBatches) {
        this.totalBatches = totalBatches;
    }

    @Override
    public void onBatchEnd(int batch, float loss) {
        epochBatches++;
        if (verbose && batch % logEveryNSteps == 0) {
            int progress = totalBatches > 0 ? (int) ((long) epochBatches * 100 / totalBatches) : 0;
            System.out.println("    Step " + epochBatches + "/" + totalBatches + " [" + progress + "%] - loss: "
                + formatFloat(loss));
        }
    }

    @Override
    public void onValidationEnd(int epoch, EvalResult result) {
        if (verbose && logMetrics) {
            System.out.println("  Validation results:");
            System.out.println("    Loss: " + formatFloat(result.loss()));
            if (result.metrics() != null) {
                for (Map.Entry<String, Float> entry : result.metrics().entrySet()) {
                    System.out.println("    " + entry.getKey() + ": " + formatFloat(entry.getValue()));
                }
            }
        }
    }

    private String formatTime(long ms) {
        if (ms < 1000) {
            return ms + "ms";
        } else if (ms < 60000) {
            return String.format("%.2fs", ms / 1000.0);
        } else {
            long seconds = ms / 1000;
            long minutes = seconds / 60;
            seconds = seconds % 60;
            return minutes + "m " + seconds + "s";
        }
    }

    private String formatFloat(float value) {
        return String.format("%.6f", value);
    }

    public int logEveryNEpochs() {
        return logEveryNEpochs;
    }

    public int logEveryNSteps() {
        return logEveryNSteps;
    }

    public boolean logMetrics() {
        return logMetrics;
    }

    public boolean verbose() {
        return verbose;
    }

    @Override
    public String name() {
        return "ProgressLogger";
    }

    @Override
    public String toString() {
        return "ProgressLogger{" +
            "logEveryNEpochs=" + logEveryNEpochs +
            ", logEveryNSteps=" + logEveryNSteps +
            ", logMetrics=" + logMetrics +
            ", verbose=" + verbose +
            '}';
    }
}
