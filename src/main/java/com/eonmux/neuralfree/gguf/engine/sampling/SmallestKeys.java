package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Arrays;

/**
 * The ascending prefix of a key array without sorting it whole: {@link #select} keeps the {@code k} smallest keys in
 * a bounded max-heap and sorts only those, so a truncating stage that needs the first few entries of the order pays
 * {@code n log k} instead of {@code n log n}. Keys are distinct (their low half is the candidate index), so the
 * prefix equals the first {@code k} entries of the full ascending sort. A selection is worth it only while it is much
 * narrower than the keys ({@link #fullSort}). Not thread-safe.
 */
final class SmallestKeys {

    /** The selection width a truncating stage tries before it walks the whole order ({@link BucketedOrder}). */
    static final int FIRST = 64;

    private long[] heap = new long[0];

    /** Whether a selection of {@code k} out of {@code n} keys is better made by a full sort. */
    static boolean fullSort(int k, int n) {
        return (long) k * 4 >= n;
    }

    /**
     * The {@code k} smallest of {@code keys[0, n)}, ascending, in the first {@code k} entries of the returned
     * scratch (overwritten by the next call).
     *
     * @throws IllegalArgumentException when {@code k} is outside {@code [1, n]}
     */
    long[] select(long[] keys, int n, int k) {
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("the " + k + " smallest of " + n + " keys");
        }
        if (heap.length < k) {
            heap = new long[k];
        }
        long[] h = heap;
        int size = 0;
        for (int i = 0; i < n; i++) {
            long key = keys[i];
            if (size < k) {
                h[size] = key;
                siftUp(h, size++);
            } else if (key < h[0]) {
                h[0] = key;
                siftDown(h, k);
            }
        }
        Arrays.sort(h, 0, k);
        return h;
    }

    private static void siftUp(long[] h, int i) {
        long v = h[i];
        while (i > 0) {
            int parent = (i - 1) >>> 1;
            if (h[parent] >= v) {
                break;
            }
            h[i] = h[parent];
            i = parent;
        }
        h[i] = v;
    }

    private static void siftDown(long[] h, int size) {
        long v = h[0];
        int i = 0;
        while (true) {
            int child = 2 * i + 1;
            if (child >= size) {
                break;
            }
            if (child + 1 < size && h[child + 1] > h[child]) {
                child++;
            }
            if (h[child] <= v) {
                break;
            }
            h[i] = h[child];
            i = child;
        }
        h[i] = v;
    }
}
