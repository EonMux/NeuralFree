package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.arch.SwaType;

import java.util.Objects;

/**
 * Which cached positions a query position may attend, per llama.cpp {@code is_masked_swa} (src/llama-hparams.h:506-540)
 * plus the causal rule. For query {@code q} and key {@code k}:
 * <ul>
 *   <li>causal: masked when {@code k > q};</li>
 *   <li>STANDARD: masked when {@code q - k >= nSwa}, so the first visible key is {@code q - nSwa + 1};</li>
 *   <li>CHUNKED: masked when {@code k < (q / nSwa) * nSwa} (chunk start);</li>
 *   <li>SYMMETRIC: masked when {@code |q - k| > nSwa / 2}.</li>
 * </ul>
 * Immutable.
 */
public final class KvWindow {

    /** Full causal attention. */
    public static final KvWindow FULL_CAUSAL = new KvWindow(SwaType.NONE, 0, true);
    /** Full bidirectional attention. */
    public static final KvWindow FULL_BIDIRECTIONAL = new KvWindow(SwaType.NONE, 0, false);

    private final SwaType type;
    private final int nSwa;
    private final boolean causal;

    private KvWindow(SwaType type, int nSwa, boolean causal) {
        this.type = type;
        this.nSwa = nSwa;
        this.causal = causal;
    }

    /**
     * @throws IllegalArgumentException when a window type has {@code nSwa <= 0}
     */
    public static KvWindow of(SwaType type, int nSwa, boolean causal) {
        Objects.requireNonNull(type, "type");
        if (type == SwaType.NONE) {
            return causal ? FULL_CAUSAL : FULL_BIDIRECTIONAL;
        }
        if (nSwa <= 0) {
            throw new IllegalArgumentException(type + " window needs nSwa > 0, got " + nSwa);
        }
        return new KvWindow(type, nSwa, causal);
    }

    public SwaType type() { return type; }
    public int nSwa() { return nSwa; }
    public boolean causal() { return causal; }
    public boolean windowed() { return type != SwaType.NONE; }

    /** Smallest key position visible from query position {@code pos} (0 when unbounded). */
    public int firstVisible(int pos) {
        checkPos(pos);
        switch (type) {
            case STANDARD:
                return Math.max(0, pos - nSwa + 1);
            case CHUNKED:
                return (pos / nSwa) * nSwa;
            case SYMMETRIC:
                return Math.max(0, pos - nSwa / 2);
            default:
                return 0;
        }
    }

    /** Largest key position visible from query position {@code pos}; {@link Integer#MAX_VALUE} when unbounded. */
    public int lastVisible(int pos) {
        checkPos(pos);
        if (causal) {
            return pos;
        }
        if (type == SwaType.SYMMETRIC) {
            long last = (long) pos + nSwa / 2;
            return (int) Math.min(Integer.MAX_VALUE, last);
        }
        return Integer.MAX_VALUE;
    }

    /** Whether query position {@code q} may attend key position {@code k}. */
    public boolean visible(int q, int k) {
        checkPos(q);
        if (k < 0) {
            return false;
        }
        if (causal && k > q) {
            return false;
        }
        switch (type) {
            case STANDARD:
                return (long) q - k < nSwa;
            case CHUNKED:
                return k >= (q / nSwa) * nSwa;
            case SYMMETRIC:
                return Math.abs((long) q - k) <= nSwa / 2;
            default:
                return true;
        }
    }

    /**
     * Number of consecutive key positions that must stay cached so that the queries {@code [pos, pos + nTokens)}
     * see every visible key: {@code nSwa + nTokens - 1} for STANDARD and CHUNKED (keys from
     * {@code pos - nSwa + 1} through {@code pos + nTokens - 1}), {@code 2 * (nSwa / 2) + nTokens} for SYMMETRIC,
     * and {@link Long#MAX_VALUE} without a window.
     */
    public long retainedPositions(int nTokens) {
        if (nTokens <= 0) {
            throw new IllegalArgumentException("token count must be positive, got " + nTokens);
        }
        switch (type) {
            case STANDARD:
            case CHUNKED:
                return (long) nSwa + nTokens - 1;
            case SYMMETRIC:
                return (long) (nSwa / 2) * 2 + nTokens;
            default:
                return Long.MAX_VALUE;
        }
    }

    private static void checkPos(int pos) {
        if (pos < 0) {
            throw new IllegalArgumentException("position " + pos + " is negative");
        }
    }

    @Override
    public String toString() {
        return "KvWindow{" + type + (windowed() ? "/" + nSwa : "") + (causal ? ", causal" : ", bidirectional") + '}';
    }
}
