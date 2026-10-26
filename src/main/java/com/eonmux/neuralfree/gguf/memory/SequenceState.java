package com.eonmux.neuralfree.gguf.memory;

import java.util.Arrays;

/**
 * Host-side record of the single sequence held in memory: the tokens whose K/V or state are stored, in position
 * order ({@code n_past} = {@link #nPast()}). Used for prefix reuse (longest common prefix between generations) and
 * rollback. This is the one mutable runtime object of the memory API; it is owned by a {@link ModelMemory} and is
 * not thread-safe.
 *
 * <p>{@link #reusable()} is the contiguous-reuse frontier (PARITY §5.5.1 E-2, until PC-9): the length of the stored
 * prefix whose rows were computed with exactly the tokens at positions {@code [0, reusable)}. It never exceeds
 * {@code nPast}, is raised only by appends that extend the exact same prefix, and is lowered by every rewrite:
 * {@link #truncate}, {@link #capReusable}, and a context shift ({@link #shift}) lowers it to the kept count.
 * {@link #commonPrefix} never reports past it, so a generation can never reuse rows that were computed with a
 * different prefix. It is kept for PC-9, which turns the reuse path on; until then reusers reset the memory first.
 */
public final class SequenceState {

    private int[] tokens;
    private int nPast;
    private int reusable;
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

    /**
     * Length of the stored prefix whose rows were computed with exactly the stored tokens: the reuse frontier
     * of the contiguous path ({@code ≤ nPast}). Raised only by contiguous appends, lowered by every rewrite
     * ({@link #truncate}, {@link #capReusable}, {@link #shift}); {@link #clear} resets it to 0.
     */
    public int reusable() { return reusable; }

    /** Lowers the reuse frontier to {@code at} (a KV edit at or above it invalidates the rows after it). */
    public void capReusable(int at) {
        if (reusable > Math.max(0, at)) {
            this.reusable = Math.max(0, at);
        }
    }

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
        // PARITY §5.5.1 E-2: the frontier only rises by a contiguous append, an extension of the exact prefix.
        boolean contiguous = reusable == nPast;
        System.arraycopy(newTokens, off, tokens, nPast, n);
        nPast += n;
        if (contiguous) {
            reusable = nPast;
        }
    }

    /** Drops the last {@code n} positions. */
    public void truncate(int n) {
        if (n < 0 || n > nPast) {
            throw new IllegalArgumentException("cannot drop " + n + " of " + nPast + " stored positions");
        }
        nPast -= n;
        capReusable(nPast);
    }

    /** Forgets every position. */
    public void clear() {
        nPast = 0;
        reusable = 0;
        wrapped = false;
    }

    /**
     * Context shift on the token record: drops positions {@code [nKeep, nKeep + nDiscard)} and moves the tail down,
     * keeping the wrap flag. The reuse frontier falls to {@code nKeep}: the kept rows were computed with the
     * discarded context between them, so only the kept prefix is reusable ({@link #capReusable}).
     *
     * @see com.eonmux.neuralfree.gguf.memory.gpu.DeviceStepInputs
     */
    public void shift(int nKeep, int nDiscard) {
        if (nKeep < 0 || nDiscard <= 0 || (long) nKeep + nDiscard > nPast) {
            throw new IllegalArgumentException("context shift of keep=" + nKeep + " discard=" + nDiscard
                    + " is invalid for " + nPast + " stored positions");
        }
        System.arraycopy(tokens, nKeep + nDiscard, tokens, nKeep, nPast - nKeep - nDiscard);
        nPast -= nDiscard;
        capReusable(nKeep);
    }

    /** Length of the common prefix of the stored tokens and {@code prompt}, never past {@link #reusable()}. */
    public int commonPrefix(int[] prompt) {
        int n = Math.min(Math.min(nPast, reusable), prompt.length);
        int i = 0;
        while (i < n && tokens[i] == prompt[i]) {
            i++;
        }
        return i;
    }

    @Override
    public String toString() {
        return "SequenceState{nPast=" + nPast + ", reuse=" + reusable + ", nCtx=" + contextLength
                + (wrapped ? ", wrapped" : "") + '}';
    }
}
