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
    /**
     * The first physical block of a LongCat-Flash layer: its dense SwiGLU MLP <em>and</em> the shortcut-connected
     * routed experts, both fed the same normalized input. The dense output is the block's FFN output; the expert
     * output is kept in a model-wide scratch until the layer's second block adds it.
     */
    MOE_SHORTCUT,
    /**
     * The second physical block of a LongCat-Flash layer: its dense SwiGLU MLP, with the expert output the layer's
     * first block kept added to it.
     */
    SHORTCUT_ADD,
    RWKV_CHANNEL_MIX,
    /** No FFN (deci {@code n_ff == 0}). */
    NONE
}
