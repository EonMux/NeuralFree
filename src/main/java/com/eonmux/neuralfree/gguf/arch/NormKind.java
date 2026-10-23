package com.eonmux.neuralfree.gguf.arch;

/**
 * Normalization kinds. A layer norm without a weight tensor is {@link #LAYER} with no weight; a
 * {@code (1 + w)} weight is baked into the tensor by the converters.
 */
public enum NormKind {
    /** {@code x * rsqrt(mean(x^2) + eps) * w}. */
    RMS(0),
    /** {@code (x - mean) * rsqrt(var + eps) * w + b}. */
    LAYER(1),
    /** {@code x / max(||x||, eps)} (ggml_l2_norm). */
    L2_CLAMP(2),
    /** {@code x * rsqrt(sum(x^2) + eps)} (gated delta-net L2 norm). */
    L2_GDN(3),
    /** Group normalization over channel groups. */
    GROUP(-1);

    private final int rowNormMode;

    NormKind(int rowNormMode) {
        this.rowNormMode = rowNormMode;
    }

    /** True when the kind is one of the {@code LLM_RowNorm} kernel modes. */
    public boolean isRowNorm() {
        return rowNormMode >= 0;
    }

    /**
     * The {@code mode} argument of the {@code LLM_RowNorm} kernel.
     *
     * @throws IllegalStateException for {@link #GROUP}, which has its own kernels
     */
    public int rowNormMode() {
        if (rowNormMode < 0) {
            throw new IllegalStateException(name() + " is not a row-norm mode");
        }
        return rowNormMode;
    }
}
