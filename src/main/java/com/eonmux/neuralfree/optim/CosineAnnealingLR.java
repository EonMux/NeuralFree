package com.eonmux.neuralfree.optim;

import java.util.Objects;

public class CosineAnnealingLR extends LRScheduler {
    private final int tMax;
    private final float etaMin;
    private final int numCycles;

    public CosineAnnealingLR(Optimizer optimizer, int tMax, float etaMin, int numCycles) {
        super(optimizer);
        Objects.requireNonNull(optimizer, "Optimizer cannot be null");
        if (tMax <= 0) {
            throw new IllegalArgumentException("T_max must be positive, got: " + tMax);
        }
        if (etaMin < 0) {
            throw new IllegalArgumentException("eta_min must be non-negative, got: " + etaMin);
        }
        if (etaMin >= initialLr) {
            throw new IllegalArgumentException("eta_min must be less than initial learning rate");
        }
        if (numCycles <= 0) {
            throw new IllegalArgumentException("num_cycles must be positive, got: " + numCycles);
        }
        this.tMax = tMax;
        this.etaMin = etaMin;
        this.numCycles = numCycles;
    }

    public CosineAnnealingLR(Optimizer optimizer, int tMax, float etaMin) {
        this(optimizer, tMax, etaMin, 1);
    }

    public CosineAnnealingLR(Optimizer optimizer, int tMax) {
        this(optimizer, tMax, 0.0f, 1);
    }

    @Override
    public float getLR() {
        ensureOpen();
        float progress;
        if (numCycles == 1) {
            progress = (float) epochCount / tMax;
        } else {
            int cycleLength = tMax / numCycles;
            int cycleProgress = epochCount % cycleLength;
            progress = (float) cycleProgress / cycleLength;
        }
        
        float cosine = (float) Math.cos(Math.PI * progress);
        return etaMin + (initialLr - etaMin) * (1 + cosine) / 2;
    }

    public int tMax() {
        return tMax;
    }

    public float etaMin() {
        return etaMin;
    }

    public int numCycles() {
        return numCycles;
    }

    public boolean inWarmRestart() {
        int cycleLength = tMax / numCycles;
        return epochCount > 0 && epochCount % cycleLength == 0;
    }

    @Override
    public String toString() {
        if (closed) {
            return "CosineAnnealingLR{closed=true}";
        }
        return "CosineAnnealingLR{tMax=" + tMax + 
               ", etaMin=" + etaMin + 
               ", numCycles=" + numCycles + 
               ", epochCount=" + epochCount + 
               ", currentLr=" + getLR() + '}';
    }
}
