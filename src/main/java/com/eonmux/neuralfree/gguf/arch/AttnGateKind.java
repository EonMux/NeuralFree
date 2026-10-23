package com.eonmux.neuralfree.gguf.arch;

/** Output gate of the attention block. */
public enum AttnGateKind {
    NONE,
    /** Gate interleaved with Q in the Q projection, applied as sigmoid. */
    Q_INTERLEAVED,
    /** Element-wise sigmoid gate from its own projection. */
    ELEMENTWISE_SIGMOID,
    /** Element-wise softplus gate from its own projection. */
    ELEMENTWISE_SOFTPLUS,
    /** One sigmoid gate per head. */
    PER_HEAD_SIGMOID,
    /** One sigmoid gate per head with a bias. */
    PER_HEAD_SIGMOID_BIAS,
    /** One softplus gate per head, broadcast over the head's values (laguna XS.2, models/laguna.cpp:247-254). */
    PER_HEAD_SOFTPLUS,
    /**
     * Element-wise base-2 softplus gate {@code log2(1 + 2^g) = softplus(g * ln 2) / ln 2} from its own projection
     * (k2-horizon, models/k2-horizon.cpp:281-294).
     */
    SOFTPLUS_BASE2
}
