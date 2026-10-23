package com.eonmux.neuralfree.gguf.arch;

/**
 * MoE router gating function. Codes are the kernel {@code gating} argument; GGUF stores
 * {@code expert_gating_func} with the llama.cpp numbering of {@code llama_expert_gating_func_type}
 * (src/llama-hparams.h:17-23), which is {@link #ggufCode()}.
 */
public enum GatingFunc {
    SOFTMAX(0, 1),
    SIGMOID(1, 2),
    /** Softmax applied to the selected router weights instead of the logits. */
    SOFTMAX_WEIGHT(2, 3),
    SQRT_SOFTPLUS(3, 4);

    /** GGUF code meaning "not set"; the architecture default applies. */
    public static final int GGUF_CODE_NONE = 0;

    private final int code;
    private final int ggufCode;

    GatingFunc(int code, int ggufCode) {
        this.code = code;
        this.ggufCode = ggufCode;
    }

    /** Kernel code. */
    public int code() {
        return code;
    }

    /** The value stored in {@code %s.expert_gating_func}. */
    public int ggufCode() {
        return ggufCode;
    }

    /**
     * Gating function for a GGUF {@code expert_gating_func} value.
     *
     * @return the function, or null for {@link #GGUF_CODE_NONE}
     * @throws IllegalArgumentException for an unknown code
     */
    public static GatingFunc fromGgufCode(long ggufCode) {
        if (ggufCode == GGUF_CODE_NONE) {
            return null;
        }
        for (GatingFunc g : values()) {
            if (g.ggufCode == ggufCode) {
                return g;
            }
        }
        throw new IllegalArgumentException("unknown expert_gating_func value " + ggufCode);
    }
}
