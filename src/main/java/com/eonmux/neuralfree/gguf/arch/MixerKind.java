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
    MSA,
    DSV4,
    MAMBA1,
    MAMBA2,
    PLAMO2_MAMBA,
    SHORTCONV,
    SHORTCONV_CENTERED,
    RWKV6,
    RWKV6_QWEN2,
    RWKV7,
    GDN,
    KDA,
    LIGHTNING,
    CROSS_ATTENTION_T5
}
