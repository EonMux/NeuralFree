package com.eonmux.neuralfree.optim;

import java.util.Locale;
import java.util.Objects;

/**
 * PyTorch's {@code ReduceLROnPlateau}: {@link #step(float)} takes the monitored metric, and once it has failed to
 * improve on the best value for more than {@code patience} calls the rate is multiplied by {@code factor}, never going
 * below {@code minLr}. The best value is tracked on every call, including during the {@code cooldown} calls after a
 * reduction, when bad calls are not counted. In {@code "rel"} threshold mode an improvement must beat the best value by
 * the fraction {@code threshold}; in {@code "abs"} mode by {@code threshold} itself.
 */
public class ReduceOnPlateauLR extends LRScheduler {
    private final String mode;
    private final float factor;
    private final int patience;
    private final float threshold;
    private final String thresholdMode;
    private final int cooldown;
    private final float minLr;

    private float bestMetric;
    private int numBadEpochs;
    private int currentCooldown;
    private float currentLr;
    private int lastEpochReduced;

    public ReduceOnPlateauLR(Optimizer optimizer, String mode, float factor, int patience,
                             float threshold, String thresholdMode, int cooldown, float minLr) {
        super(optimizer);
        Objects.requireNonNull(mode, "Mode cannot be null");
        Objects.requireNonNull(thresholdMode, "Threshold mode cannot be null");
        
        if (!"min".equalsIgnoreCase(mode) && !"max".equalsIgnoreCase(mode)) {
            throw new IllegalArgumentException("mode must be 'min' or 'max', got: " + mode);
        }
        if (!"rel".equalsIgnoreCase(thresholdMode) && !"abs".equalsIgnoreCase(thresholdMode)) {
            throw new IllegalArgumentException("threshold_mode must be 'rel' or 'abs', got: " + thresholdMode);
        }
        if (factor <= 0 || factor >= 1) {
            throw new IllegalArgumentException("factor must be in (0, 1), got: " + factor);
        }
        if (patience < 0) {
            throw new IllegalArgumentException("patience must be non-negative, got: " + patience);
        }
        if (threshold <= 0) {
            throw new IllegalArgumentException("threshold must be positive, got: " + threshold);
        }
        if (cooldown < 0) {
            throw new IllegalArgumentException("cooldown must be non-negative, got: " + cooldown);
        }
        if (minLr < 0) {
            throw new IllegalArgumentException("min_lr must be non-negative, got: " + minLr);
        }
        
        this.mode = mode.toLowerCase(Locale.ROOT);
        this.factor = factor;
        this.patience = patience;
        this.threshold = threshold;
        this.thresholdMode = thresholdMode.toLowerCase(Locale.ROOT);
        this.cooldown = cooldown;
        this.minLr = minLr;

        this.bestMetric = initialBest();
        this.numBadEpochs = 0;
        this.currentCooldown = 0;
        this.currentLr = initialLr;
        this.lastEpochReduced = 0;
    }

    public ReduceOnPlateauLR(Optimizer optimizer, String mode, float factor, int patience) {
        this(optimizer, mode, factor, patience, 1e-4f, "rel", 0, 0.0f);
    }

    public ReduceOnPlateauLR(Optimizer optimizer, String mode) {
        this(optimizer, mode, 0.1f, 10, 1e-4f, "rel", 0, 0.0f);
    }

    @Override
    public float getLR() {
        ensureOpen();
        return currentLr;
    }

    /** Records one value of the monitored metric and reduces the rate when it has plateaued. */
    public void step(float metric) {
        ensureOpen();
        stepCount++;

        if (isImprovement(metric, bestMetric)) {
            bestMetric = metric;
            numBadEpochs = 0;
        } else {
            numBadEpochs++;
        }

        if (currentCooldown > 0) {
            currentCooldown--;
            numBadEpochs = 0;
        }

        if (numBadEpochs > patience) {
            reduceLr();
            numBadEpochs = 0;
            currentCooldown = cooldown;
            lastEpochReduced = epochCount;
        }
    }

    /** The best value a metric can be compared against before any has been seen: the worst value for the mode. */
    private float initialBest() {
        return "min".equals(mode) ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
    }

    private boolean isImprovement(float metric, float best) {
        if ("min".equals(mode)) {
            if ("rel".equals(thresholdMode)) {
                return metric < best * (1 - threshold);
            } else {
                return metric < best - threshold;
            }
        } else {
            if ("rel".equals(thresholdMode)) {
                return metric > best * (1 + threshold);
            } else {
                return metric > best + threshold;
            }
        }
    }

    private void reduceLr() {
        currentLr = Math.max(currentLr * factor, minLr);
        optimizer.setLearningRate(currentLr);
    }

    public void resetMetrics() {
        bestMetric = initialBest();
        numBadEpochs = 0;
        currentCooldown = 0;
    }

    @Override
    public void reset() {
        ensureOpen();
        resetMetrics();
        currentLr = initialLr;
        lastEpochReduced = 0;
        super.reset();
    }

    public String mode() {
        return mode;
    }

    public float factor() {
        return factor;
    }

    public int patience() {
        return patience;
    }

    public float threshold() {
        return threshold;
    }

    public String thresholdMode() {
        return thresholdMode;
    }

    public int cooldown() {
        return cooldown;
    }

    public float minLr() {
        return minLr;
    }

    public float bestMetric() {
        return bestMetric;
    }

    public int numBadEpochs() {
        return numBadEpochs;
    }

    public int lastEpochReduced() {
        return lastEpochReduced;
    }

    public boolean inCooldown() {
        return currentCooldown > 0;
    }

    @Override
    public String toString() {
        if (closed) {
            return "ReduceOnPlateauLR{closed=true}";
        }
        return "ReduceOnPlateauLR{mode=" + mode + 
               ", factor=" + factor + 
               ", patience=" + patience + 
               ", currentLr=" + currentLr + 
               ", bestMetric=" + bestMetric + 
               ", numBadEpochs=" + numBadEpochs + 
               ", inCooldown=" + inCooldown() + '}';
    }
}
