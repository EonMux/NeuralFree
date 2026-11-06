package com.eonmux.neuralfree.gguf.engine;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * The n-grams lookahead decoding has observed, keyed by their first token: for every first token a ring of at most
 * {@code G} continuations of {@code N - 1} tokens, exactly llama.cpp's {@code ngram_container} of its lookahead
 * example ({@code examples/lookahead/lookahead.cpp:30-44}). The first token itself is the key and is not stored; a
 * continuation already present is not stored again, and a full ring overwrites its oldest entry.
 *
 * <p>Where llama.cpp keeps a dense {@code n_vocab * G * (N - 1)} array, this keeps a ring only for the first tokens
 * that were observed, so a large vocabulary costs nothing until it is used. Mutable, not thread-safe.
 */
public final class LookaheadNgrams {

    /** The continuations of one first token: a ring of {@code G} entries of {@code N - 1} tokens each. */
    private static final class Ring {
        final int[][] grams;
        int count;
        int head;

        Ring(int ngramSize, int maxNgrams) {
            this.grams = new int[maxNgrams][ngramSize - 1];
        }
    }

    private final int ngramSize;
    private final int maxNgrams;
    private final Map<Integer, Ring> rings = new HashMap<>();
    private long total;

    /**
     * @param ngramSize  the n-gram size {@code N}, at least 2
     * @param maxNgrams  the continuations kept per first token {@code G}, at least 1
     * @throws IllegalArgumentException for a size or count out of range
     */
    public LookaheadNgrams(int ngramSize, int maxNgrams) {
        if (ngramSize < 2) {
            throw new IllegalArgumentException("n-gram size must be at least 2, got " + ngramSize);
        }
        if (maxNgrams < 1) {
            throw new IllegalArgumentException("max n-grams must be at least 1, got " + maxNgrams);
        }
        this.ngramSize = ngramSize;
        this.maxNgrams = maxNgrams;
    }

    /** The n-gram size {@code N}. */
    public int ngramSize() {
        return ngramSize;
    }

    /** The continuations kept per first token {@code G}. */
    public int maxNgrams() {
        return maxNgrams;
    }

    /** Continuations stored for {@code firstToken}, at most {@link #maxNgrams()}. */
    public int count(int firstToken) {
        Ring ring = rings.get(firstToken);
        return ring == null ? 0 : ring.count;
    }

    /**
     * Continuation {@code index} of {@code firstToken} (a copy, {@code N - 1} tokens), in the ring's physical order.
     *
     * @throws IndexOutOfBoundsException when {@code index} is outside {@code [0, count(firstToken))}
     */
    public int[] gram(int firstToken, int index) {
        Ring ring = rings.get(firstToken);
        if (ring == null || index < 0 || index >= ring.count) {
            throw new IndexOutOfBoundsException("n-gram " + index + " of token " + firstToken + " outside [0, "
                    + count(firstToken) + ")");
        }
        return ring.grams[index].clone();
    }

    /**
     * Stores {@code continuation} under {@code firstToken}, as llama.cpp's lookahead example does: a continuation
     * already held is not stored again, and a full ring overwrites its oldest entry.
     *
     * @param firstToken   the n-gram's first token, non-negative
     * @param continuation its following {@code N - 1} tokens, non-negative
     * @return true when the continuation was stored, false when it was already present
     * @throws IllegalArgumentException for a negative token or a continuation of the wrong length
     */
    public boolean add(int firstToken, int[] continuation) {
        Objects.requireNonNull(continuation, "continuation");
        if (firstToken < 0) {
            throw new IllegalArgumentException("n-gram first token must not be negative, got " + firstToken);
        }
        if (continuation.length != ngramSize - 1) {
            throw new IllegalArgumentException("a continuation holds " + (ngramSize - 1) + " tokens, got "
                    + continuation.length);
        }
        for (int i = 0; i < continuation.length; i++) {
            if (continuation[i] < 0) {
                throw new IllegalArgumentException("continuation token " + i + " is negative: " + continuation[i]);
            }
        }
        Ring ring = rings.get(firstToken);
        if (ring == null) {
            ring = new Ring(ngramSize, maxNgrams);
            rings.put(firstToken, ring);
        }
        for (int k = 0; k < ring.count; k++) {
            if (Arrays.equals(ring.grams[k], continuation)) {
                return false;
            }
        }
        System.arraycopy(continuation, 0, ring.grams[ring.head], 0, continuation.length);
        ring.count = Math.min(maxNgrams, ring.count + 1);
        ring.head = (ring.head + 1) % maxNgrams;
        total++;
        return true;
    }

    /** The continuations stored over every first token: each stored n-gram counts once, overwritten ones too. */
    public long total() {
        return total;
    }

    /** The first tokens that hold a continuation. */
    public int distinctFirstTokens() {
        return rings.size();
    }

    @Override
    public String toString() {
        return "LookaheadNgrams{n=" + ngramSize + ", g=" + maxNgrams + ", firstTokens=" + rings.size()
                + ", total=" + total + '}';
    }
}
