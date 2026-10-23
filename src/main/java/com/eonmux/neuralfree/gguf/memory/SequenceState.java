package com.eonmux.neuralfree.gguf.memory;

import java.util.Arrays;

/**
 * Host-side record of the single sequence held in memory: the tokens whose K/V or state are stored, in position
 * order ({@code n_past} = {@link #nPast()}). Used for prefix reuse (longest common prefix between generations) and
 * rollback. This is the one mutable runtime object of the memory API; it is owned by a {@link ModelMemory} and is
 * not thread-safe.
 */
public final class SequenceState {

    private int[] tokens;
    private int nPast;
    private boolean wrapped;
    private final int contextLength;

    /** @param nCtx context length; positions at or beyond it are rejected */
    public SequenceState(int nCtx) {
        if (nCtx <= 0) {
            throw new IllegalArgumentException("sequence context must be positive, got " + nCtx);
        }
        this.contextLength = nCtx;
        this.tokens = new int[Math.min(nCtx, 1024)];
    }

    /** Number of stored positions (next position to write). */
    public int nPast() { return nPast; }

    /** Context length. */
    public int nCtx() { return contextLength; }

    /** True once any ring cache has overwritten a stored row since the last {@link #clear()}. */
    public boolean wrapped() { return wrapped; }

    /**
     * Records that a ring cache overwrote a stored row. Informational: whether positions can be dropped is decided
     * from the stored rows ({@link ModelMemory#canRollback(int)}), not from this flag.
     */
    public void markWrapped() { this.wrapped = true; }

    /** Token stored at position {@code pos}. */
    public int token(int pos) {
        if (pos < 0 || pos >= nPast) {
            throw new IndexOutOfBoundsException("position " + pos + " outside the stored range [0, " + nPast + ")");
        }
        return tokens[pos];
    }

    /** Copy of the stored tokens. */
    public int[] tokens() {
        return Arrays.copyOf(tokens, nPast);
    }

    /**
     * Appends tokens written at positions {@code [nPast, nPast + n)}.
     *
     * @throws IllegalStateException when the context would overflow
     */
    public void append(int[] newTokens, int off, int n) {
        if (off < 0 || n < 0 || off + n > newTokens.length) {
            throw new IndexOutOfBoundsException("token range [" + off + ", " + (off + n) + ") outside " + newTokens.length);
        }
        if ((long) nPast + n > contextLength) {
            throw new IllegalStateException("context full: " + nPast + " + " + n + " tokens exceed " + contextLength);
        }
        if (nPast + n > tokens.length) {
            int cap = (int) Math.min(contextLength, Math.max((long) tokens.length * 2, (long) nPast + n));
            tokens = Arrays.copyOf(tokens, cap);
        }
        System.arraycopy(newTokens, off, tokens, nPast, n);
        nPast += n;
    }

    /** Drops the last {@code n} positions. */
    public void truncate(int n) {
        if (n < 0 || n > nPast) {
            throw new IllegalArgumentException("cannot drop " + n + " of " + nPast + " stored positions");
        }
        nPast -= n;
    }

    /** Forgets every position. */
    public void clear() {
        nPast = 0;
        wrapped = false;
    }

    /** Length of the common prefix of the stored tokens and {@code prompt}. */
    public int commonPrefix(int[] prompt) {
        int n = Math.min(nPast, prompt.length);
        int i = 0;
        while (i < n && tokens[i] == prompt[i]) {
            i++;
        }
        return i;
    }

    @Override
    public String toString() {
        return "SequenceState{nPast=" + nPast + ", nCtx=" + contextLength + (wrapped ? ", wrapped" : "") + '}';
    }
}
