package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;

/**
 * FFN activation of one encoder (llama.cpp {@code ffn_op_type}, tools/mtmd/clip-model.h). The metadata keys
 * {@code clip.use_gelu} / {@code clip.use_silu} pick GELU / SILU; with neither, the default is
 * {@link #GELU_QUICK} (clip.cpp {@code load_hparams}). A gated FFN pairs the op with its
 * gated-linear-unit variant, an ungated one applies the plain unary op.
 */
public enum ClipFfnOp {

    /** {@code ggml_gelu} (tanh form) and {@code ggml_geglu_split}. */
    GELU(ActOp.GELU_TANH, GluOp.GEGLU_TANH),
    /** {@code ggml_gelu_erf} and {@code ggml_geglu_erf_split}. */
    GELU_ERF(ActOp.GELU_ERF, GluOp.GEGLU_ERF),
    /** {@code ggml_silu} and {@code ggml_swiglu_split}. */
    SILU(ActOp.SILU, GluOp.SWIGLU),
    /** {@code ggml_gelu_quick} and {@code ggml_geglu_quick_split}. */
    GELU_QUICK(ActOp.GELU_QUICK, GluOp.GEGLU_QUICK),
    /** {@code ggml_relu^2}; llama.cpp builds it ungated only. */
    RELU_SQR(ActOp.RELU_SQR, null);

    private final ActOp unary;
    private final GluOp glu;

    ClipFfnOp(ActOp unary, GluOp glu) {
        this.unary = unary;
        this.glu = glu;
    }

    /** The op of an ungated FFN ({@code ggml_gelu} family). */
    public ActOp unary() {
        return unary;
    }

    /**
     * The op of a gated FFN ({@code f(gate) * up}).
     *
     * @throws UnsupportedOperationException for {@link #RELU_SQR}, which llama.cpp builds ungated only
     */
    public GluOp glu() {
        if (glu == null) {
            throw new UnsupportedOperationException(name() + " has no gated form (clip.cpp build_ffn)");
        }
        return glu;
    }

    /**
     * The op for the {@code clip.use_gelu} / {@code clip.use_silu} keys: both true is refused, neither gives
     * {@link #GELU_QUICK} (clip.cpp {@code load_hparams}).
     */
    public static ClipFfnOp fromUseGeluSilu(boolean useGelu, boolean useSilu) {
        if (useGelu && useSilu) {
            throw new IllegalArgumentException("clip.use_gelu and clip.use_silu are both true");
        }
        if (useGelu) {
            return GELU;
        }
        if (useSilu) {
            return SILU;
        }
        return GELU_QUICK;
    }
}
