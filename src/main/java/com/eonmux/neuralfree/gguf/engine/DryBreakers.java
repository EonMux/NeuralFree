package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;

/**
 * A DRY stage's restart sequences as one immutable flat table: sequence {@code i} is head token {@link #head(int)}
 * followed by {@link #tailLength(int)} tail ids starting at {@link #tailFrom(int)} of the shared tail array. A stage
 * builds its table once, and every device readout row of that stage shares it, so the prefilter neither copies the
 * sequences per token nor validates them again.
 */
public final class DryBreakers {

    /** The table of no restart sequence. */
    public static final DryBreakers EMPTY = new DryBreakers(new int[0], new int[1], new int[0], -1);

    private final int[] heads;
    /** {@code count + 1} offsets: sequence {@code i}'s tail is {@code tails[from[i], from[i + 1])}. */
    private final int[] from;
    private final int[] tails;
    private final int minId;
    private final int maxId;

    private DryBreakers(int[] heads, int[] from, int[] tails, int maxId) {
        this.heads = heads;
        this.from = from;
        this.tails = tails;
        int min = 0;
        for (int h : heads) {
            min = Math.min(min, h);
        }
        for (int t : tails) {
            min = Math.min(min, t);
        }
        this.minId = min;
        this.maxId = maxId;
    }

    /**
     * The table of {@code entries}, each the head token followed by its tail ids (copied).
     *
     * @throws IllegalArgumentException when an entry carries no head token
     */
    public static DryBreakers of(int[][] entries) {
        Objects.requireNonNull(entries, "dry breakers");
        if (entries.length == 0) {
            return EMPTY;
        }
        int[] heads = new int[entries.length];
        int[] from = new int[entries.length + 1];
        int total = 0;
        for (int i = 0; i < entries.length; i++) {
            if (entries[i] == null || entries[i].length == 0) {
                throw new IllegalArgumentException("DRY breaker " + i + " carries no head token");
            }
            total += entries[i].length - 1;
        }
        int[] tails = new int[total];
        int at = 0;
        int max = -1;
        for (int i = 0; i < entries.length; i++) {
            int[] e = entries[i];
            heads[i] = e[0];
            max = Math.max(max, e[0]);
            from[i] = at;
            for (int j = 1; j < e.length; j++) {
                tails[at++] = e[j];
                max = Math.max(max, e[j]);
            }
        }
        from[entries.length] = at;
        return new DryBreakers(heads, from, tails, max);
    }

    /** Restart sequences. */
    public int count() {
        return heads.length;
    }

    /** Tail ids across the sequences. */
    public int tailCount() {
        return tails.length;
    }

    /** Head token of sequence {@code i}. */
    public int head(int i) {
        return heads[i];
    }

    /** Offset of sequence {@code i}'s tail in the shared tail ids. */
    public int tailFrom(int i) {
        return from[i];
    }

    /** Tail ids of sequence {@code i}. */
    public int tailLength(int i) {
        return from[i + 1] - from[i];
    }

    /** Copies the shared tail ids into {@code dst} from {@code at}. */
    public void copyTails(int[] dst, int at) {
        System.arraycopy(tails, 0, dst, at, tails.length);
    }

    /** Whether every head and tail id lies in {@code [0, vocab)}. */
    public boolean inVocabulary(int vocab) {
        return minId >= 0 && maxId < vocab;
    }

    /** The sequences as one array each, the head token first (copies). */
    public int[][] entries() {
        int[][] out = new int[heads.length][];
        for (int i = 0; i < heads.length; i++) {
            int[] e = new int[1 + tailLength(i)];
            e[0] = heads[i];
            System.arraycopy(tails, from[i], e, 1, e.length - 1);
            out[i] = e;
        }
        return out;
    }
}
