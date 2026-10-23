package com.eonmux.neuralfree.gguf.arch;

/** How the FFN gate is provided (llama.cpp {@code llm_ffn_gate_type}). */
public enum FfnGating {
    /** Ungated: {@code down(act(up(x)))}. */
    SEQ,
    /** Separate gate tensor: {@code down(act(gate(x)) * up(x))}. */
    PAR,
    /** Gate and up packed in one tensor of {@code 2 * n_ff} rows, gate half first. */
    FUSED_UP
}
