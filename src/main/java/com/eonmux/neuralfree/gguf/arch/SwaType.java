package com.eonmux.neuralfree.gguf.arch;

/**
 * Sliding-window attention mask type (llama.cpp {@code llama_swa_type}, src/llama-hparams.h:24).
 *
 * <p>Masking of key position {@code p0} for query position {@code p1} with window {@code n}
 * ({@code llama_hparams::is_masked_swa}): STANDARD masks when {@code p1 - p0 >= n}; CHUNKED masks when
 * {@code p0 < (p1 / n) * n}; CHUNKED_LOOKBACK masks when {@code p0 < ((p1 / n) - 1) * n}, so a query sees its
 * own aligned chunk and the one before it; SYMMETRIC masks when {@code |p1 - p0| > n / 2}.
 */
public enum SwaType {
    NONE(0),
    STANDARD(1),
    CHUNKED(2),
    SYMMETRIC(3),
    /** Aligned chunks with one chunk of look-back (RNJ-1's chunked local attention). */
    CHUNKED_LOOKBACK(4);

    private final int code;

    SwaType(int code) {
        this.code = code;
    }

    /** Numeric code shared with the kernels. */
    public int code() {
        return code;
    }
}
