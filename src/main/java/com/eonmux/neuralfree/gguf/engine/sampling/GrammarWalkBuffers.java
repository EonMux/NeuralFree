package com.eonmux.neuralfree.gguf.engine.sampling;

import java.lang.ref.SoftReference;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * The slot arrays of a {@link GrammarRejectWalk} and the code points of a row decoded while a character is pending:
 * scratch a whole-row mask fills and reads for the length of one {@link GrammarStage#apply}, sized by the row. A row
 * of a large vocabulary needs megabytes of them, which would stay live for every constrained sequence if each stage
 * kept its own, so a mask borrows a set from a small process-wide pool and returns it when the mask is done: the
 * scratch held at any time follows how many masks run at once, not how many constrained sequences are open. The
 * pool keeps at most {@link #POOLED} sets, softly reachable, so an idle pool gives its memory back under heap
 * pressure. A borrowed set belongs to its borrower alone until released.
 */
final class GrammarWalkBuffers {

    /** How many idle sets the pool keeps. */
    static final int POOLED = 4;

    private static final AtomicReferenceArray<SoftReference<GrammarWalkBuffers>> POOL =
            new AtomicReferenceArray<>(POOLED);

    /** The slot order the walk's partitions permute. */
    int[] slots;
    int[] id;
    int[] base;
    int[] remain;
    int[] value;
    /** The code points of a row decoded from a pending partial character. */
    final GrammarPieces.Decoded rowCodePoints;

    /** A set for {@code capacity} slots. */
    GrammarWalkBuffers(int capacity) {
        allocate(capacity);
        this.rowCodePoints = new GrammarPieces.Decoded(16);
    }

    /** Grows the slot arrays to hold {@code n} slots; their contents are the caller's to fill. */
    void reserve(int n) {
        if (slots.length < n) {
            allocate(n);
        }
    }

    private void allocate(int capacity) {
        int cap = Math.max(capacity, 16);
        slots = new int[cap];
        id = new int[cap];
        base = new int[cap];
        remain = new int[cap];
        value = new int[cap];
    }

    /** A set from the pool, or a new one when the pool has none idle. */
    static GrammarWalkBuffers borrow() {
        for (int i = 0; i < POOLED; i++) {
            SoftReference<GrammarWalkBuffers> ref = POOL.getAndSet(i, null);
            GrammarWalkBuffers buffers = ref == null ? null : ref.get();
            if (buffers != null) {
                return buffers;
            }
        }
        return new GrammarWalkBuffers(16);
    }

    /** Returns {@code buffers} to the pool, or drops it when the pool is full; the caller must not use it again. */
    static void release(GrammarWalkBuffers buffers) {
        buffers.rowCodePoints.clear();
        SoftReference<GrammarWalkBuffers> ref = new SoftReference<>(buffers);
        for (int i = 0; i < POOLED; i++) {
            SoftReference<GrammarWalkBuffers> held = POOL.get(i);
            if ((held == null || held.get() == null) && POOL.compareAndSet(i, held, ref)) {
                return;
            }
        }
    }
}
