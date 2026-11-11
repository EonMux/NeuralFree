package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.memory.ContextFitter;

import java.util.function.LongSupplier;
import java.util.function.LongToIntFunction;

/**
 * Measures the heap available for a CPU memory plan, reserving headroom for loading and forward-pass allocations.
 * A reading that refuses or shrinks the context is taken again after a collection, so unreachable objects do not
 * consume the budget.
 */
final class HeapMemory {

    /** Share of the free heap kept out of the budget for the temporaries of the load and the per-call results. */
    static final double HEADROOM_FRACTION = 0.125;
    /** Smallest headroom kept out of the budget. */
    static final long MIN_HEADROOM_BYTES = 64L << 20;

    private HeapMemory() {
    }

    /**
     * The context {@code fit} chooses for the heap budget, where {@code fit} maps a budget in bytes to the context it
     * holds (as {@link ContextFitter#fit} does) and {@code wanted} is the context asked for: the user's, or the
     * training context when it is automatic. A reading that refuses the context, or fits less than {@code wanted}, is
     * taken again after a collection, so garbage left by earlier work (a closed model's buffers, a load's
     * temporaries) neither refuses nor shrinks a context the collected heap holds; a healthy reading collects
     * nothing.
     *
     * @throws ContextFitter.ContextBudgetException when the context does not fit the collected heap either
     */
    static int fitContext(LongToIntFunction fit, int wanted) {
        Runtime rt = Runtime.getRuntime();
        return fitContext(fit, wanted, () -> budgetBytes(rt.maxMemory(), rt.totalMemory(), rt.freeMemory()), rt::gc);
    }

    /** {@link #fitContext(LongToIntFunction, int)} over a heap {@code reading} and a {@code collect}ion. */
    static int fitContext(LongToIntFunction fit, int wanted, LongSupplier reading, Runnable collect) {
        long first = reading.getAsLong();
        int fitted = -1;
        if (first > 0) {
            try {
                fitted = fit.applyAsInt(first);
                if (fitted >= wanted || first == ContextFitter.UNLIMITED) {
                    return fitted;
                }
            } catch (ContextFitter.ContextBudgetException refused) {
                // taken again after the collection below
            }
        }
        collect.run();
        try {
            return Math.max(fitted, fit.applyAsInt(reading.getAsLong()));
        } catch (ContextFitter.ContextBudgetException e) {
            if (fitted > 0) {
                return fitted;
            }
            throw e;
        }
    }

    /**
     * Bytes a memory plan may take from a heap of one {@link Runtime} reading, or {@link ContextFitter#UNLIMITED}
     * when the heap has no maximum. Package-visible for tests.
     */
    static long budgetBytes(long maxMemory, long totalMemory, long freeMemory) {
        if (maxMemory == Long.MAX_VALUE) {
            return ContextFitter.UNLIMITED;
        }
        long free = maxMemory - (totalMemory - freeMemory);
        long headroom = Math.max(MIN_HEADROOM_BYTES, (long) (free * HEADROOM_FRACTION));
        return Math.max(0L, free - headroom);
    }
}
