package com.eonmux.neuralfree.gguf.arch;

/** Feed-forward block of a layer; each value is served by one FFN factory. */
public enum FfnKind {
    DENSE,
    MOE,
    /** Experts run in a latent space with down/up projections around the routed experts. */
    MOE_LATENT,
    /** Experts plus chunked experts. */
    MOE_CHUNKED,
    /** MoE added to a parallel dense FFN (arctic). */
    MOE_PARALLEL_DENSE,
    RWKV_CHANNEL_MIX,
    /** No FFN (deci {@code n_ff == 0}). */
    NONE
}
