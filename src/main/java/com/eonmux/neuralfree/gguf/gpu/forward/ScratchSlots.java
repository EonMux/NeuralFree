package com.eonmux.neuralfree.gguf.gpu.forward;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names of the well-known scratch slots shared by the layer executor and the blocks. Each holds {@code ubatch} rows
 * ({@code [t][width]}); the widths are fixed by {@link GpuScratch#forModel}.
 */
public final class ScratchSlots {

    /** Residual stream (width: widest stream, {@code n_embd} or {@code hc * n_embd}). */
    public static final String X = "X";
    /** Output of the current mixer or FFN branch before the residual (stream width). */
    public static final String BRANCH = "BRANCH";
    /** Normalized block input (stream width). */
    public static final String NORMED = "NORMED";
    /** Query projection, or a fused Q/K/V or gated-Q projection (widest projection). */
    public static final String Q = "Q";
    /** Key projection (widest projection). */
    public static final String K = "K";
    /** Value projection (widest projection). */
    public static final String V = "V";
    /** Attention or mixer core output before the output projection (widest projection). */
    public static final String ATTN_OUT = "ATTN_OUT";
    /** Gate and up projections of the FFN, or the selected experts' intermediates. */
    public static final String FFN_GATE_UP = "FFN_GATE_UP";
    /** FFN output before the residual, or the selected experts' outputs before their weighted sum. */
    public static final String FFN_OUT = "FFN_OUT";
    /** Logits of the output rows ({@code vocab} floats per row). */
    public static final String LOGITS = "LOGITS";
    /** Token ids then token types of the step ({@code int}, 2 per token). */
    public static final String TOKEN_IDS = "TOKEN_IDS";
    /** Embedding-stage inputs: padded input embeddings and per-layer embedding inputs. */
    public static final String INPUT_EMBD = "INPUT_EMBD";
    /** Positions of the step ({@code int}, 4 sections per token). */
    public static final String POS = "POS";

    /** Every well-known slot name (unmodifiable). */
    public static final List<String> ALL = Collections.unmodifiableList(Arrays.asList(
            X, BRANCH, NORMED, Q, K, V, ATTN_OUT, FFN_GATE_UP, FFN_OUT, LOGITS, TOKEN_IDS, INPUT_EMBD, POS));

    private ScratchSlots() {
    }

    /** True for a name in {@link #ALL}. */
    public static boolean isWellKnown(String name) {
        return ALL.contains(name);
    }
}
