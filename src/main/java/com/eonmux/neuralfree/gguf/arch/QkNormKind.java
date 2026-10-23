package com.eonmux.neuralfree.gguf.arch;

/**
 * Normalization applied to Q and K, before RoPE unless the layer's {@code AttentionSpec.qkNormAfterRope()} says
 * otherwise.
 */
public enum QkNormKind {
    NONE,
    /**
     * RMS norm per head; the weight is one {@code head_dim} row shared by the heads, or one row per head
     * ({@code {head_dim, n_head}}, k2-horizon, models/k2-horizon.cpp:75-76).
     */
    RMS_HEAD,
    /** RMS norm over the full projection. */
    RMS_FULL,
    /** Layer norm per head. */
    LN_HEAD,
    /** RMS norm over each half of the head separately (sensenova_u1). */
    RMS_SPLIT_HALF,
    /**
     * Layer norm with optional bias over the full projection ({@code n_head * head_dim} for Q,
     * {@code n_head_kv * head_dim} for K), eps {@code f_norm_eps}: the BERT graph (jina-bert-v2, models/bert.cpp:115-129)
     * and mpt (models/mpt.cpp:106-118).
     */
    LN_FULL
}
