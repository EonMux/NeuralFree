package com.eonmux.neuralfree.gguf.arch;

/** Token mixer of a layer; each value is served by one mixer factory. */
public enum MixerKind {
    ATTENTION,
    /** deci layer with {@code n_head > 0} and {@code n_head_kv == 0}: {@code wo @ attn_norm(x)}. */
    ATTENTION_LINEAR_REPLACE,
    /** Layer without a mixer (deci {@code n_head == 0}, FFN-only hybrid layers). */
    NONE,
    MLA,
    DSA_MLA,
    /** Step5 sparse region attention: GQA with a per-head gate and an indexer over 8-token region summaries. */
    STEP5,
    MSA,
    DSV4,
    MAMBA1,
    MAMBA2,
    /** Zamba2 layer: a Mamba-2 mixer whose input also carries the output of a shared attention/MLP block. */
    ZAMBA2,
    PLAMO2_MAMBA,
    SHORTCONV,
    SHORTCONV_CENTERED,
    RWKV6,
    RWKV6_QWEN2,
    RWKV7,
    GDN,
    KDA,
    LIGHTNING,
    CROSS_ATTENTION_T5,
    /**
     * Inkling softmax attention: no RoPE, per-head Q/K RMS norms, relative-distance logits added to the scores,
     * short convolutions on the K/V streams, and the per-layer conv state that carries those streams.
     */
    INKLING
}
