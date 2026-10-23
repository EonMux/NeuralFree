package com.eonmux.neuralfree.gguf.arch;

/**
 * RoPE variant of an architecture (llama.cpp {@code llama_rope_type}, include/llama.h; selection in
 * {@code llama_model_rope_type}, src/llama-model.cpp:3476).
 */
public enum RopeType {
    /** No rotary embedding. */
    NONE(-1),
    /** Rotates consecutive head-value pairs. */
    NORM(0),
    /** Rotates pairs offset by {@code n_rot / 2}. */
    NEOX(2),
    /** Multi-section RoPE with contiguous sections. */
    MROPE(8),
    /** Vision RoPE (two-section MRoPE over the full head). */
    VISION(24),
    /** Interleaved multi-section RoPE. */
    IMROPE(40);

    private final int ggmlMode;

    RopeType(int ggmlMode) {
        this.ggmlMode = ggmlMode;
    }

    /** The ggml rope {@code mode} argument ({@code GGML_ROPE_TYPE_*}); -1 for {@link #NONE}. */
    public int ggmlMode() {
        return ggmlMode;
    }

    /** True when positions carry four sections per token (llama.cpp {@code n_pos_per_embd() == 4}). */
    public boolean usesSections() {
        return this == MROPE || this == IMROPE;
    }
}
