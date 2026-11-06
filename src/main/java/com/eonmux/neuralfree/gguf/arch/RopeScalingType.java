package com.eonmux.neuralfree.gguf.arch;

/**
 * RoPE context scaling: the llama.cpp {@code llama_rope_scaling_type} names (src/llama-model.cpp:1085-1090) plus
 * {@link #DYNAMIC}, the NTK-aware scaling of vLLM's {@code dynamic} rope type.
 */
public enum RopeScalingType {
    NONE("none"),
    LINEAR("linear"),
    YARN("yarn"),
    LONGROPE("longrope"),
    /**
     * NTK-aware scaling (vLLM {@code rope_type "dynamic"}): the frequency base is multiplied by the
     * {@code nDims / (nDims - 2)} power of a scale, either {@code alpha} or the factor-and-context ratio. The base is
     * computed per layer from the normalized rotary width ({@link
     * com.eonmux.neuralfree.gguf.ops.rope.RopeMath#dynamicNtkBase} and {@link
     * com.eonmux.neuralfree.gguf.ops.rope.RopeMath#dynamicAlphaBase}).
     */
    DYNAMIC("dynamic"),
    /**
     * Fixed and mixed NTK scaling (vLLM {@code rope_type "ntk"}, rotary_embedding/ntk_scaling_rope.py): every pair's
     * frequency is divided by a factor, constant for the fixed form and growing per pair by {@code mixed_b} for the
     * mixed form. The base stays the trained one; the divisors are computed per layer
     * ({@link com.eonmux.neuralfree.gguf.ops.rope.RopeMath#ntkFactors}).
     */
    NTK("ntk");

    private final String ggufName;

    RopeScalingType(String ggufName) {
        this.ggufName = ggufName;
    }

    /** The value stored in {@code %s.rope.scaling.type}. */
    public String ggufName() {
        return ggufName;
    }

    /**
     * Scaling type for a {@code %s.rope.scaling.type} value.
     *
     * @throws IllegalArgumentException when the name is not one of none, linear, yarn, longrope, dynamic, ntk
     */
    public static RopeScalingType fromGgufName(String name) {
        for (RopeScalingType t : values()) {
            if (t.ggufName.equals(name)) {
                return t;
            }
        }
        throw new IllegalArgumentException("unknown rope scaling type '" + name + "'");
    }
}
