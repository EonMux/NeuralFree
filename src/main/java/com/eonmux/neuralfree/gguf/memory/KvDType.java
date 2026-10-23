package com.eonmux.neuralfree.gguf.memory;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * Element type of a KV cache row: the cache types llama.cpp accepts for {@code -ctk} / {@code -ctv}
 * (llama.cpp common/arg.cpp:304-314). F32, F16 and BF16 store one value per element; the block types store whole
 * 32-value ggml blocks (Q8_0 34 bytes, Q4_0 18, Q4_1 20, IQ4_NL 18, Q5_0 22, Q5_1 24), so a row width of a block type
 * must be a multiple of 32. {@link KvBlockCodec} holds the block layouts and the reference (de)quantizers.
 */
public enum KvDType {
    F32(GGMLType.F32, 0),
    F16(GGMLType.F16, 1),
    Q8_0(GGMLType.Q8_0, 2),
    BF16(GGMLType.BF16, 3),
    Q4_0(GGMLType.Q4_0, 4),
    Q4_1(GGMLType.Q4_1, 5),
    IQ4_NL(GGMLType.IQ4_NL, 6),
    Q5_0(GGMLType.Q5_0, 7),
    Q5_1(GGMLType.Q5_1, 8);

    private static final List<KvDType> ALL;

    static {
        List<KvDType> all = new ArrayList<>();
        Collections.addAll(all, values());
        ALL = Collections.unmodifiableList(all);
    }

    private final GGMLType ggmlType;
    private final int code;

    KvDType(GGMLType ggmlType, int code) {
        this.ggmlType = ggmlType;
        this.code = code;
    }

    public GGMLType ggmlType() {
        return ggmlType;
    }

    /**
     * Element-type code of the KV kernels: the {@code NF_KV_*} values of kernels/llm/common/kv_types.h and
     * shaders/llm/common/kv_types.glsl.
     */
    public int code() {
        return code;
    }

    /**
     * True for the block types ({@code ggml_is_quantized}: Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1); F32, F16 and BF16
     * are not quantized.
     */
    public boolean quantized() {
        return ggmlType.isQuantized();
    }

    /** Values per storage block (32 for the block types, 1 otherwise). */
    public int blockSize() {
        return ggmlType.getBlockSize();
    }

    /** Bytes of one storage block (one value for F32, F16 and BF16). */
    public int blockBytes() {
        return ggmlType.getTypeSize();
    }

    /** Suffix of this type's KV kernel variants, e.g. {@code Q8_0} in {@code LLM_KV_Store_Q8_0}. */
    public String kernelSuffix() {
        return name();
    }

    /** The kernel variant of {@code prefix} for this type: {@code prefix + kernelSuffix()}. */
    public String kernel(String prefix) {
        return prefix + kernelSuffix();
    }

    /**
     * Bytes of one row of {@code width} values.
     *
     * @throws IllegalArgumentException for a negative width or a block-type width that is not a multiple of 32
     */
    public long rowBytes(long width) {
        if (width < 0) {
            throw new IllegalArgumentException("KV row width " + width + " is negative");
        }
        if (width % blockSize() != 0) {
            throw new IllegalArgumentException(this + " KV rows need a width that is a multiple of " + blockSize()
                    + ", got " + width);
        }
        return ggmlType.getBytesForElements(width);
    }

    /** True when rows of this width can be stored in this type. */
    public boolean supportsWidth(long width) {
        return width >= 0 && width % blockSize() == 0;
    }

    /** Every KV type, in declaration order. */
    public static List<KvDType> all() {
        return ALL;
    }

    /** The kernel variants {@code prefix + t.kernelSuffix()} of every type, in declaration order. */
    public static List<String> kernelVariants(String prefix) {
        List<String> out = new ArrayList<>(ALL.size());
        for (KvDType t : ALL) {
            out.add(t.kernel(prefix));
        }
        return Collections.unmodifiableList(out);
    }

    /** The accepted names, lower case as llama.cpp spells them ({@code f32, f16, bf16, q8_0, ...}). */
    public static String names() {
        StringBuilder sb = new StringBuilder();
        for (KvDType t : ALL) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(t.name().toLowerCase(Locale.ROOT));
        }
        return sb.toString();
    }

    /**
     * Parses a KV type name, case-insensitive: {@code f32}, {@code f16}, {@code bf16}, {@code q8_0}, {@code q4_0},
     * {@code q4_1}, {@code iq4_nl}, {@code q5_0} or {@code q5_1} (the ggml type names of llama.cpp
     * common/arg.cpp:304-314); {@code q8} is accepted for Q8_0.
     *
     * @param property name of the property or flag being parsed, for the error message
     * @throws IllegalArgumentException for any other value
     */
    public static KvDType parse(String value, String property) {
        if (value == null) {
            throw new IllegalArgumentException(property + " is not set");
        }
        String v = value.trim().toUpperCase(Locale.ROOT);
        if (v.equals("Q8")) {
            return Q8_0;
        }
        for (KvDType t : ALL) {
            if (t.name().equals(v)) {
                return t;
            }
        }
        throw new IllegalArgumentException(property + "=" + value + " is not a supported KV cache type; use one of "
                + names());
    }
}
