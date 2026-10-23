package com.eonmux.neuralfree.gguf.arch;

/**
 * Gated-linear-unit ops {@code out = f(gate) * up} and variants. Codes are the {@code op} argument of
 * {@code LLM_GLU} and are mirrored by the C {@code #define}s in {@code kernels/llm/common/math.h}.
 */
public enum GluOp {
    SWIGLU(0),
    GEGLU_TANH(1),
    GEGLU_ERF(2),
    GEGLU_QUICK(3),
    REGLU(4),
    /** gpt-oss: {@code (up + 1) * gate * sigmoid(alpha * gate)} with clamped inputs. */
    SWIGLU_OAI(5),
    /** SwiGLU with the gate and up inputs clamped to a limit. */
    SWIGLU_CLAMP(6),
    /** SwiGLU with a limit on the activated gate. */
    SWIGLU_LIMITED(7),
    /** SiTU gate (kimi-k3). */
    SITU(8),
    /** Plain product {@code gate * up}. */
    MUL(9),
    /** {@code sigmoid(gate) * up}. */
    SIGMUL(10);

    private final int code;

    GluOp(int code) {
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
    public static GluOp fromCode(int code) {
        for (GluOp op : values()) {
            if (op.code == code) {
                return op;
            }
        }
        throw new IllegalArgumentException("unknown GLU op code " + code);
    }
}
