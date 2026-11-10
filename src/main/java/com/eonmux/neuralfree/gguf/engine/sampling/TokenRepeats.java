package com.eonmux.neuralfree.gguf.engine.sampling;

import java.util.Arrays;

/**
 * DRY's per-apply table of the longest repeat each window token would extend: an open-addressing int-to-int map
 * whose slots carry the apply's generation, so starting the next apply costs no clearing and a lookup per
 * candidate costs no boxing. Not thread-safe.
 */
final class TokenRepeats {

    private int[] keys = new int[0];
    private int[] values = new int[0];
    private int[] stamps = new int[0];
    private int mask = -1;
    private int generation;
    private int size;

    /** Empties the table and sizes it for at most {@code entries} distinct tokens. */
    void reset(int entries) {
        int cap = Integer.highestOneBit(Math.max(4, entries) * 2 - 1) << 1;
        if (cap > keys.length) {
            keys = new int[cap];
            values = new int[cap];
            stamps = new int[cap];
            mask = cap - 1;
            generation = 0;
        }
        if (++generation == 0) {
            Arrays.fill(stamps, 0);
            generation = 1;
        }
        size = 0;
    }

    /** Distinct tokens recorded since the last reset. */
    int size() {
        return size;
    }

    /** Records {@code length} for {@code token} unless the token already holds a longer or equal one. */
    void putMax(int token, int length) {
        int slot = slot(token);
        while (stamps[slot] == generation) {
            if (keys[slot] == token) {
                if (values[slot] < length) {
                    values[slot] = length;
                }
                return;
            }
            slot = (slot + 1) & mask;
        }
        stamps[slot] = generation;
        keys[slot] = token;
        values[slot] = length;
        size++;
    }

    /** The length recorded for {@code token}, or -1. */
    int get(int token) {
        if (size == 0) {
            return -1;
        }
        int slot = slot(token);
        while (stamps[slot] == generation) {
            if (keys[slot] == token) {
                return values[slot];
            }
            slot = (slot + 1) & mask;
        }
        return -1;
    }

    private int slot(int token) {
        return (token * 0x9E3779B9 >>> 7) & mask;
    }
}
