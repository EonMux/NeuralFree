package com.eonmux.neuralfree.gguf.arch;

/** RoPE context scaling (llama.cpp {@code llama_rope_scaling_type}, names in src/llama-model.cpp:1068). */
public enum RopeScalingType {
    NONE("none"),
    LINEAR("linear"),
    YARN("yarn"),
    LONGROPE("longrope");

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
     * @throws IllegalArgumentException when the name is not one of none, linear, yarn, longrope
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
