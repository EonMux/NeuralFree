package com.eonmux.neuralfree.optim;

import java.util.Locale;
import java.util.Objects;

/**
 * The 1cycle policy with PyTorch's {@code OneCycleLR} schedule. The rate starts at {@code maxLr / divFactor}, rises
 * to {@code maxLr} by step {@code pctStart * totalSteps - 1}, then anneals to
 * {@code maxLr / divFactor / finalDivFactor} by step {@code totalSteps - 1}. With {@code threePhase} it instead
 * returns to the starting rate by step {@code 2 * pctStart * totalSteps - 2} and anneals to the final rate over the
 * remaining steps. The scheduler may be stepped {@code totalSteps} times; the last of those steps holds the final rate,
 * and stepping further throws.
 */
public class OneCycleLR extends LRScheduler {
    private static final String COS = "cos";
    private static final String LINEAR = "linear";

    private final float maxLr;
    private final float divFactor;
    private final float finalDivFactor;
    private final int totalSteps;
    private final float pctStart;
    private final String annealStrategy;
    private final boolean threePhase;
    private final double[] phaseEndSteps;
    private final float[] phaseStartLrs;
    private final float[] phaseEndLrs;

    public OneCycleLR(Optimizer optimizer, float maxLr, int totalSteps, float divFactor,
                      float finalDivFactor, float pctStart, String annealStrategy, boolean threePhase) {
        super(optimizer);
        if (!(maxLr > 0) || Float.isInfinite(maxLr)) {
            throw new IllegalArgumentException("max_lr must be finite and positive, got: " + maxLr);
        }
        if (totalSteps <= 0) {
            throw new IllegalArgumentException("total_steps must be positive, got: " + totalSteps);
        }
        if (!(divFactor > 0) || Float.isInfinite(divFactor)) {
            throw new IllegalArgumentException("div_factor must be finite and positive, got: " + divFactor);
        }
        if (!(finalDivFactor > 0) || Float.isInfinite(finalDivFactor)) {
            throw new IllegalArgumentException("final_div_factor must be finite and positive, got: " + finalDivFactor);
        }
        if (!(pctStart > 0 && pctStart < 1)) {
            throw new IllegalArgumentException("pct_start must be in (0, 1), got: " + pctStart);
        }
        Objects.requireNonNull(annealStrategy, "anneal_strategy cannot be null");
        String strategy = annealStrategy.toLowerCase(Locale.ROOT);
        if (!COS.equals(strategy) && !LINEAR.equals(strategy)) {
            throw new IllegalArgumentException("anneal_strategy must be 'cos' or 'linear', got: " + annealStrategy);
        }

        this.maxLr = maxLr;
        this.totalSteps = totalSteps;
        this.divFactor = divFactor;
        this.finalDivFactor = finalDivFactor;
        this.pctStart = pctStart;
        this.annealStrategy = strategy;
        this.threePhase = threePhase;

        float startLr = maxLr / divFactor;
        float minLr = startLr / finalDivFactor;
        double warmupEnd = (double) pctStart * totalSteps - 1.0;
        double lastStep = totalSteps - 1.0;
        if (threePhase) {
            this.phaseEndSteps = new double[] {warmupEnd, 2.0 * pctStart * totalSteps - 2.0, lastStep};
            this.phaseStartLrs = new float[] {startLr, maxLr, startLr};
            this.phaseEndLrs = new float[] {maxLr, startLr, minLr};
        } else {
            this.phaseEndSteps = new double[] {warmupEnd, lastStep};
            this.phaseStartLrs = new float[] {startLr, maxLr};
            this.phaseEndLrs = new float[] {maxLr, minLr};
        }

        this.initialLr = startLr;
        optimizer.setLearningRate(startLr);
    }

    public OneCycleLR(Optimizer optimizer, float maxLr, int totalSteps, float divFactor,
                      float finalDivFactor, float pctStart, String annealStrategy) {
        this(optimizer, maxLr, totalSteps, divFactor, finalDivFactor, pctStart, annealStrategy, false);
    }

    public OneCycleLR(Optimizer optimizer, float maxLr, int totalSteps) {
        this(optimizer, maxLr, totalSteps, 25.0f, 1e4f, 0.3f, COS, false);
    }

    @Override
    public float getLR() {
        ensureOpen();
        int last = phaseEndSteps.length - 1;
        int phase = 0;
        double phaseStart = 0.0;
        while (phase < last && stepCount > phaseEndSteps[phase]) {
            phaseStart = phaseEndSteps[phase];
            phase++;
        }
        double span = phaseEndSteps[phase] - phaseStart;
        double pct = span > 0 ? (stepCount - phaseStart) / span : 1.0;
        return anneal(phaseStartLrs[phase], phaseEndLrs[phase], Math.min(1.0, Math.max(0.0, pct)));
    }

    /**
     * Advances one batch.
     *
     * @throws IllegalStateException when the scheduler has already been stepped {@code totalSteps} times
     */
    @Override
    public void step() {
        ensureOpen();
        if (stepCount >= totalSteps) {
            throw new IllegalStateException("Tried to step " + (stepCount + 1)
                    + " times. The specified number of total steps is " + totalSteps);
        }
        super.step();
    }

    private float anneal(float start, float end, double pct) {
        if (LINEAR.equals(annealStrategy)) {
            return (float) (start + (end - start) * pct);
        }
        return (float) (end + (start - end) / 2.0 * (Math.cos(Math.PI * pct) + 1.0));
    }

    public float maxLr() {
        return maxLr;
    }

    public float divFactor() {
        return divFactor;
    }

    public float finalDivFactor() {
        return finalDivFactor;
    }

    public int totalSteps() {
        return totalSteps;
    }

    public float pctStart() {
        return pctStart;
    }

    public String annealStrategy() {
        return annealStrategy;
    }

    public boolean threePhase() {
        return threePhase;
    }

    public float progress() {
        return Math.min(1.0f, (float) stepCount / totalSteps);
    }

    public boolean isFinished() {
        return stepCount >= totalSteps;
    }

    @Override
    public String toString() {
        if (closed) {
            return "OneCycleLR{closed=true}";
        }
        return "OneCycleLR{maxLr=" + maxLr +
               ", totalSteps=" + totalSteps +
               ", threePhase=" + threePhase +
               ", stepCount=" + stepCount +
               ", progress=" + String.format(Locale.ROOT, "%.2f%%", progress() * 100) +
               ", currentLr=" + getLR() + '}';
    }
}
