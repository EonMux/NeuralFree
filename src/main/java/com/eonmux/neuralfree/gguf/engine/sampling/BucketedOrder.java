package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Arrays;

/**
 * The ascending order of a key array, produced bucket by bucket so a truncating stage sorts only the part of the
 * order it walks. {@link #prepare} scatters the keys into {@value #BUCKETS} buckets by a float offset whose order
 * agrees with the keys' (a larger key never has a smaller offset), linear over {@code [0, span)} with every larger
 * offset in the last bucket; {@link #sortBucket} then sorts one bucket's slice in place. The buckets in order,
 * each sorted, are exactly the full ascending sort of the keys, so a stage that stops its walk inside bucket
 * {@code b} has paid two passes over the keys plus the sorts of buckets {@code 0..b}, whatever the width of its
 * prefix. Keys are distinct (their low half is the candidate index). Not thread-safe.
 */
final class BucketedOrder {

    /** The buckets the offset range splits into. */
    static final int BUCKETS = 4096;

    private long[] order = new long[0];
    private final int[] start = new int[BUCKETS + 1];
    private final int[] fill = new int[BUCKETS];
    private int sortedBuckets;

    /**
     * Scatters {@code keys[0, n)} into the buckets of their {@code offsets[0, n)}: offsets are non-negative and not
     * NaN, ordered as the keys are, and bucket at {@code BUCKETS * offset / span}.
     *
     * @return the order array the buckets fill, valid up to the next call
     */
    long[] prepare(long[] keys, float[] offsets, int n, float span) {
        if (order.length < n) {
            order = new long[n];
        }
        float scale = BUCKETS / span;
        int[] count = fill;
        Arrays.fill(count, 0);
        for (int i = 0; i < n; i++) {
            count[bucket(offsets[i], scale)]++;
        }
        int at = 0;
        for (int b = 0; b < BUCKETS; b++) {
            start[b] = at;
            at += count[b];
            count[b] = start[b];
        }
        start[BUCKETS] = at;
        long[] out = order;
        for (int i = 0; i < n; i++) {
            out[count[bucket(offsets[i], scale)]++] = keys[i];
        }
        sortedBuckets = 0;
        return out;
    }

    /**
     * Sorts bucket {@code b} of the last {@link #prepare} in place; buckets are sorted in order, so {@code b} is the
     * next unsorted one.
     *
     * @return the end of the bucket's slice: the order is the full sort's from 0 to there
     */
    int sortBucket(int b) {
        if (b != sortedBuckets) {
            throw new IllegalStateException("bucket " + b + " sorted out of order, next is " + sortedBuckets);
        }
        Arrays.sort(order, start[b], start[b + 1]);
        sortedBuckets++;
        return start[b + 1];
    }

    private static int bucket(float offset, float scale) {
        float scaled = offset * scale;
        return scaled < BUCKETS - 1 ? (int) scaled : BUCKETS - 1;
    }
}
