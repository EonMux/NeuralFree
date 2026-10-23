package com.eonmux.neuralfree.optim;

import java.util.Objects;

/**
 * Linear warmup: after {@code k < warmupSteps} steps the rate is {@code initialLr * (k + 1) / warmupSteps}, and from
 * step {@code warmupSteps} on it is the base scheduler's rate (or {@code initialLr} without one). The base scheduler
 * receives every epoch and every step past the warmup, so its schedule starts where the warmup ends. The warmup rate
 * is applied at construction, so the first batch already runs at it.
 */
public class WarmupLR extends LRScheduler {
    private final int warmupSteps;
    private final LRScheduler baseScheduler;

    public WarmupLR(Optimizer optimizer, int warmupSteps, LRScheduler baseScheduler) {
        super(optimizer);
        if (warmupSteps < 0) {
            throw new IllegalArgumentException("warmup_steps must be non-negative, got: " + warmupSteps);
        }
        if (baseScheduler != null && baseScheduler.optimizer() != optimizer) {
            throw new IllegalArgumentException("Base scheduler must use the same optimizer");
        }
        this.warmupSteps = warmupSteps;
        this.baseScheduler = baseScheduler;
        applyLR();
    }

    public WarmupLR(Optimizer optimizer, int warmupSteps) {
        this(optimizer, warmupSteps, null);
    }

    @Override
    public float getLR() {
        ensureOpen();
        
        if (stepCount < warmupSteps) {
            float warmupFactor = (float) (stepCount + 1) / warmupSteps;
            return initialLr * warmupFactor;
        }
        
        if (baseScheduler != null) {
            return baseScheduler.getLR();
        }
        
        return initialLr;
    }

    @Override
    public void step() {
        ensureOpen();
        if (stepCount >= warmupSteps && baseScheduler != null) {
            baseScheduler.step();
        }
        super.step();
    }

    @Override
    public void stepEpoch() {
        ensureOpen();
        if (baseScheduler != null) {
            baseScheduler.stepEpoch();
        }
        super.stepEpoch();
    }

    @Override
    public void reset() {
        ensureOpen();
        if (baseScheduler != null) {
            baseScheduler.reset();
        }
        super.reset();
    }

    public int warmupSteps() {
        return warmupSteps;
    }

    public LRScheduler baseScheduler() {
        return baseScheduler;
    }

    public boolean inWarmup() {
        return stepCount < warmupSteps;
    }

    public float warmupProgress() {
        if (warmupSteps == 0) {
            return 1.0f;
        }
        return Math.min(1.0f, (float) stepCount / warmupSteps);
    }

    @Override
    public void close() {
        if (!closed) {
            if (baseScheduler != null) {
                baseScheduler.close();
            }
            super.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "WarmupLR{closed=true}";
        }
        return "WarmupLR{warmupSteps=" + warmupSteps + 
               ", stepCount=" + stepCount + 
               ", inWarmup=" + inWarmup() + 
               ", currentLr=" + getLR() + 
               ", baseScheduler=" + (baseScheduler != null ? baseScheduler.getClass().getSimpleName() : "none") + '}';
    }
}
