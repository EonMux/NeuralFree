package com.eonmux.neuralfree.gguf.arch;

/** How a block's branches are combined with the residual stream. */
public enum ResidualKind {
    /** {@code x += branch}. */
    STANDARD,
    /** Attention and FFN read the same input; both branches are added after the FFN. */
    PARALLEL,
    /** {@code x += branch * s}. */
    SCALED,
    /** {@code x = branch + s * attn_norm(x)} (minimax-01). */
    NORMED_INPUT_SCALED,
    /** Hyper-connections replace the residual adds. */
    HYPER_CONNECTIONS,
    /**
     * {@code x = x * streamScale + branch * branchScale} with per-sub-block scales from the depth of the add (beam:
     * {@code branch / sqrt(i) + x * sqrt((i - 1) / i)}, models/beam.cpp:109-113).
     */
    DEPTH_SCALED
}
