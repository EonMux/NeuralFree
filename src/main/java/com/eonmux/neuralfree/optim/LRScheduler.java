package com.eonmux.neuralfree.optim;

import java.util.Objects;

/**
 * Learning-rate schedule driven by two counters: {@link #step()} advances the per-batch step count and
 * {@link #stepEpoch()} the epoch count. {@link #getLR()} is the rate for the current counters, and every call that
 * moves a counter applies it to the optimizer at once, so after {@code k} steps and {@code e} epochs the optimizer
 * runs at {@code getLR()} for those counters, as with PyTorch's {@code scheduler.step()}.
 */
public abstract class LRScheduler {
    protected final Optimizer optimizer;
    protected float initialLr;
    protected int stepCount;
    protected int epochCount;
    protected volatile boolean closed = false;

    protected LRScheduler(Optimizer optimizer) {
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        this.optimizer = optimizer;
        this.initialLr = optimizer.getLearningRate();
        this.stepCount = 0;
        this.epochCount = 0;
    }

    /** The learning rate for the current step and epoch counts. */
    public abstract float getLR();

    /** Advances the step count and applies the rate for the new count. */
    public void step() {
        ensureOpen();
        stepCount++;
        applyLR();
    }

    /** Advances the epoch count and applies the rate for the new count, before the next epoch's first batch. */
    public void stepEpoch() {
        ensureOpen();
        epochCount++;
        applyLR();
    }

    /** Sets the optimizer's learning rate to {@link #getLR()}. */
    protected final void applyLR() {
        optimizer.setLearningRate(getLR());
    }

    public int stepCount() {
        ensureOpen();
        return stepCount;
    }

    public int epochCount() {
        ensureOpen();
        return epochCount;
    }

    public float initialLr() {
        return initialLr;
    }

    public Optimizer optimizer() {
        ensureOpen();
        return optimizer;
    }

    /**
     * Rewinds both counters and applies the starting rate. A subclass with state of its own restores that state
     * before calling this, so the rate applied here is computed from it.
     */
    public void reset() {
        ensureOpen();
        stepCount = 0;
        epochCount = 0;
        applyLR();
    }

    protected void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("LRScheduler has been closed");
        }
    }

    public void close() {
        if (!closed) {
            closed = true;
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "LRScheduler{closed=true}";
        }
        return getClass().getSimpleName() + 
               "{stepCount=" + stepCount + 
               ", epochCount=" + epochCount + 
               ", initialLr=" + initialLr + 
               ", currentLr=" + getLR() + '}';
    }
}
