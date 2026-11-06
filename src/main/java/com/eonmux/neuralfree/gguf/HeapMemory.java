package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.memory.ContextFitter;

/**
 * Measures the heap available for a CPU memory plan, reserving headroom for loading and forward-pass allocations.
 * An exhausted reading is retried after collection so unreachable objects do not consume the budget.
 */
final class HeapMemory {

    /** Share of the free heap kept out of the budget for the temporaries of the load and the per-call results. */
    static final double HEADROOM_FRACTION = 0.125;
    /** Smallest headroom kept out of the budget. */
    static final long MIN_HEADROOM_BYTES = 64L << 20;

    private HeapMemory() {
    }

    /**
     * Bytes a memory plan may take from the Java heap, or {@link ContextFitter#UNLIMITED} when the heap has no
     * maximum.
     */
    static long budgetBytes() {
        Runtime rt = Runtime.getRuntime();
        long budget = budgetBytes(rt.maxMemory(), rt.totalMemory(), rt.freeMemory());
        if (budget == 0) {
            rt.gc();
            budget = budgetBytes(rt.maxMemory(), rt.totalMemory(), rt.freeMemory());
        }
        return budget;
    }

    /** {@link #budgetBytes()} over one {@link Runtime} reading. Package-visible for tests. */
    static long budgetBytes(long maxMemory, long totalMemory, long freeMemory) {
        if (maxMemory == Long.MAX_VALUE) {
            return ContextFitter.UNLIMITED;
        }
        long free = maxMemory - (totalMemory - freeMemory);
        long headroom = Math.max(MIN_HEADROOM_BYTES, (long) (free * HEADROOM_FRACTION));
        return Math.max(0L, free - headroom);
    }
}
