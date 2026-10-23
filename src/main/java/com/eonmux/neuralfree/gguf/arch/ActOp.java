package com.eonmux.neuralfree.gguf.arch;

/**
 * Element-wise activation ops. Codes are the {@code op} argument of {@code LLM_Unary} and are mirrored by
 * the C {@code #define}s in {@code kernels/llm/common/math.h}.
 *
 * <p>SOFTPLUS is {@code x > 20 ? x : log(1 + exp(x))} (llama.cpp ggml-impl.h:107). SOFTCAP is
 * {@code p0 * tanh(x * p1)} with {@code p0 = cap} and {@code p1 = scale / cap}.
 */
public enum ActOp {
    RELU(0),
    RELU_SQR(1),
    GELU_TANH(2),
    GELU_ERF(3),
    GELU_QUICK(4),
    SILU(5),
    SIGMOID(6),
    SOFTPLUS(7),
    TANH(8),
    XIELU(9),
    SCALE_BIAS(10),
    CLAMP(11),
    EXP(12),
    SQRT_SOFTPLUS(13),
    IDENTITY(14),
    SOFTCAP(15);

    private final int code;

    ActOp(int code) {
        this.code = code;
    }

    /** Kernel op code. */
    public int code() {
        return code;
    }

    /**
     * Op for a kernel code.
     *
     * @throws IllegalArgumentException when no op has that code
     */
    public static ActOp fromCode(int code) {
        for (ActOp op : values()) {
            if (op.code == code) {
                return op;
            }
        }
        throw new IllegalArgumentException("unknown activation op code " + code);
    }
}
