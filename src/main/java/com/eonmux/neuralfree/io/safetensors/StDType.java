package com.eonmux.neuralfree.io.safetensors;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.HashMap;
import java.util.Map;

/**
 * Element dtypes of the safetensors format (safetensors {@code Dtype}, tensor.rs): the name a header stores, the
 * bits per element and the ggml type that stores the same element bytes unchanged ({@link #directType()}, null when
 * ggml has none).
 *
 * <p>Constants are declared in the reference implementation's alignment order, which is the writer's primary sort key
 * ({@link SafetensorsWriter}: descending order, then name). {@link #F8_E4M3FNUZ} and {@link #F8_E5M2FNUZ} are a
 * NeuralFree extension of the format: the reference crate does not define them, but the PyTorch binding of
 * safetensors writes them ({@code safetensors/torch.py}), so they are read as 8-bit dtypes in the 8-bit tier.</p>
 */
public enum StDType {
    /** Boolean, one byte per element. */
    BOOL("BOOL", 8, null),
    /** OCP MX FP4 E2M1, two elements per byte. */
    F4("F4", 4, null),
    /** OCP MX FP6 E2M3, packed six bits per element. */
    F6_E2M3("F6_E2M3", 6, null),
    /** OCP MX FP6 E3M2, packed six bits per element. */
    F6_E3M2("F6_E3M2", 6, null),
    /** Unsigned 8-bit integer. */
    U8("U8", 8, null),
    /** Signed 8-bit integer. */
    I8("I8", 8, GGMLType.I8),
    /** FP8 E5M2. */
    F8_E5M2("F8_E5M2", 8, null),
    /** FP8 E4M3FN (no infinities, {@code 0x7F}/{@code 0xFF} NaN); ggml's per-element F8_E4M3FN stores these bytes. */
    F8_E4M3("F8_E4M3", 8, GGMLType.F8_E4M3FN),
    /** OCP MX E8M0 scale ({@code 2^(e-127)}, {@code 0xFF} NaN). */
    F8_E8M0("F8_E8M0", 8, null),
    /** FP8 E4M3FNUZ (NeuralFree extension of the format; PyTorch {@code float8_e4m3fnuz}). */
    F8_E4M3FNUZ("F8_E4M3FNUZ", 8, null),
    /** FP8 E5M2FNUZ (NeuralFree extension of the format; PyTorch {@code float8_e5m2fnuz}). */
    F8_E5M2FNUZ("F8_E5M2FNUZ", 8, null),
    /** Signed 16-bit integer. */
    I16("I16", 16, GGMLType.I16),
    /** Unsigned 16-bit integer. */
    U16("U16", 16, null),
    /** IEEE half precision. */
    F16("F16", 16, GGMLType.F16),
    /** bfloat16. */
    BF16("BF16", 16, GGMLType.BF16),
    /** Signed 32-bit integer. */
    I32("I32", 32, GGMLType.I32),
    /** Unsigned 32-bit integer. */
    U32("U32", 32, null),
    /** IEEE single precision. */
    F32("F32", 32, GGMLType.F32),
    /** Complex number of two single-precision parts. */
    C64("C64", 64, null),
    /** IEEE double precision. */
    F64("F64", 64, GGMLType.F64),
    /** Signed 64-bit integer. */
    I64("I64", 64, GGMLType.I64),
    /** Unsigned 64-bit integer. */
    U64("U64", 64, null);

    private static final Map<String, StDType> BY_NAME = new HashMap<>();
    private static final Map<GGMLType, StDType> BY_DIRECT_TYPE = new HashMap<>();

    static {
        for (StDType t : values()) {
            BY_NAME.put(t.headerName, t);
            if (t.directType != null) {
                BY_DIRECT_TYPE.put(t.directType, t);
            }
        }
    }

    private final String headerName;
    private final int bits;
    private final GGMLType directType;

    StDType(String headerName, int bits, GGMLType directType) {
        this.headerName = headerName;
        this.bits = bits;
        this.directType = directType;
    }

    /** The name a safetensors header stores for this dtype. */
    public String headerName() {
        return headerName;
    }

    /** Bits per element: 4, 6, 8, 16, 32 or 64. */
    public int bits() {
        return bits;
    }

    /** The ggml type whose elements are these bytes unchanged, or null when ggml has none. */
    public GGMLType directType() {
        return directType;
    }

    /** The dtype stored in a header as {@code name}, or null when the name is not a known dtype. */
    public static StDType fromName(String name) {
        return name == null ? null : BY_NAME.get(name);
    }

    /** The dtype whose {@link #directType()} is {@code type}, or null when no dtype stores that ggml type. */
    public static StDType forDirectType(GGMLType type) {
        return type == null ? null : BY_DIRECT_TYPE.get(type);
    }
}
