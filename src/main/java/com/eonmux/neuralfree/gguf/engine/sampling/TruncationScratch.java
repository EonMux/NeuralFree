package com.eonmux.neuralfree.gguf.engine.sampling;

import java.lang.ref.SoftReference;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * The row-sized scratch a truncating stage ({@link TopPStage}, {@link TypicalStage}) fills and reads for the length
 * of one apply over a whole row: the candidates' sort keys, their float offsets, the bucketed order
 * ({@link BucketedOrder}) and the prefix heap ({@link SmallestKeys}). A row of a large vocabulary needs megabytes of
 * them, which would stay live for every open request if each stage kept its own, so an apply borrows a set from a
 * small process-wide pool and returns it when it is done: the scratch held at any time follows how many applies run
 * at once, not how many requests are open. The pool keeps at most {@link #POOLED} sets, softly reachable, so an idle
 * pool gives its memory back under heap pressure. A borrowed set belongs to its borrower alone until released.
 */
final class TruncationScratch {

    /**
     * How many idle sets the pool keeps: one per thread of the common fork-join pool (where the batch engine samples
     * a step's rows in parallel) plus the engine's own thread, and at least four.
     */
    static final int POOLED = Math.max(4, ForkJoinPool.getCommonPoolParallelism() + 1);

    private static final AtomicReferenceArray<SoftReference<TruncationScratch>> POOL =
            new AtomicReferenceArray<>(POOLED);

    /** The prefix selection of the order. */
    final SmallestKeys prefix = new SmallestKeys();
    /** The order walked bucket by bucket past the prefix. */
    final BucketedOrder buckets = new BucketedOrder();
    private long[] keys = new long[0];
    private float[] offsets = new float[0];

    private TruncationScratch() {
    }

    /** Room for {@code n} sort keys; the contents are the caller's to fill. */
    long[] keys(int n) {
        if (keys.length < n) {
            keys = new long[n];
        }
        return keys;
    }

    /** Room for {@code n} bucket offsets; the contents are the caller's to fill. */
    float[] offsets(int n) {
        if (offsets.length < n) {
            offsets = new float[n];
        }
        return offsets;
    }

    /** A set from the pool, or a new one when the pool has none idle. */
    static TruncationScratch borrow() {
        for (int i = 0; i < POOLED; i++) {
            SoftReference<TruncationScratch> ref = POOL.getAndSet(i, null);
            TruncationScratch scratch = ref == null ? null : ref.get();
            if (scratch != null) {
                return scratch;
            }
        }
        return new TruncationScratch();
    }

    /** Returns {@code scratch} to the pool, or drops it when the pool is full; the caller must not use it again. */
    static void release(TruncationScratch scratch) {
        SoftReference<TruncationScratch> ref = new SoftReference<>(scratch);
        for (int i = 0; i < POOLED; i++) {
            SoftReference<TruncationScratch> held = POOL.get(i);
            if ((held == null || held.get() == null) && POOL.compareAndSet(i, held, ref)) {
                return;
            }
        }
    }
}
