package com.eonmux.neuralfree.gguf.gpu.weights;

final class W12Rounding {
    private long moved;
    private double maxRelativeChange;

    /** Accounts one weight whose decoded value {@code before} was written as {@code after}. */
    void add(float before, float after) {
        if (after != before) {
            moved++;
            double relative = Math.abs((after - (double) before) / before);
            if (relative > maxRelativeChange) {
                maxRelativeChange = relative;
            }
        }
    }

    /** Weights the rounding moved. */
    long movedWeights() {
        return moved;
    }

    /** The largest relative change a moved weight underwent, 0 when none moved. */
    double maxRelativeChange() {
        return maxRelativeChange;
    }
}
