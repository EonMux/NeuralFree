package com.eonmux.neuralfree.gguf.arch;

/** Output-gate form of a KDA (Kimi delta attention) layer. */
public enum KdaGateMode {
    /** Low-rank gate projection pair. */
    LOWRANK,
    /** Full gate from {@code ssm_g}. */
    FULL_SSM_G,
    /** Full gate {@code ssm_f_a * x + dt_b} (bailingmoe3). */
    FULL_SSM_G_A
}
