package com.eonmux.neuralfree.gguf.arch;

/** Role of the post-attention norm tensor in a block. */
public enum PostNormRole {
    /** No post-attention norm. */
    NONE,
    /** Norms after both the mixer and the FFN branch, before each residual add (gemma2-style). */
    SANDWICH,
    /** {@code post_attention_norm} is the pre-FFN norm (no {@code ffn_norm} tensor). */
    PRE_FFN,
    /** No pre-norms; norms are applied to the branch outputs only (olmo2, exaone4). */
    POST_ONLY
}
