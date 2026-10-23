package com.eonmux.neuralfree.optim;

import java.util.Objects;

public class StepLR extends LRScheduler {
    private final int stepSize;
    private final float gamma;

    public StepLR(Optimizer optimizer, int stepSize, float gamma) {
        super(optimizer);
        if (stepSize <= 0) {
            throw new IllegalArgumentException("step_size must be positive, got: " + stepSize);
        }
        if (gamma <= 0 || gamma >= 1) {
            throw new IllegalArgumentException("gamma must be in (0, 1), got: " + gamma);
        }
        this.stepSize = stepSize;
        this.gamma = gamma;
    }

    public StepLR(Optimizer optimizer, int stepSize) {
        this(optimizer, stepSize, 0.1f);
    }

    @Override
    public float getLR() {
        ensureOpen();
        int numDecays = epochCount / stepSize;
        return initialLr * (float) Math.pow(gamma, numDecays);
    }

    public int stepSize() {
        return stepSize;
    }

    public float gamma() {
        return gamma;
    }

    @Override
    public String toString() {
        if (closed) {
            return "StepLR{closed=true}";
        }
        return "StepLR{stepSize=" + stepSize + 
               ", gamma=" + gamma + 
               ", epochCount=" + epochCount + 
               ", currentLr=" + getLR() + '}';
    }
}
