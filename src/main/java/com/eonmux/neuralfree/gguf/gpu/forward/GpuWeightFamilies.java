package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuModelUnsupportedException;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;

import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/**
 * The weight-kernel families of the GPU forward path: for every weight type the kernel names its projection and
 * lookup rows run ({@link WeightKernels}), the {@code {T}_Hw_*} hardware-FP8 fast-path selection on CUDA sm_89+,
 * and the W12 Q-view row set an open {@link GpuOps.W12QView} scope serves. The selection depends only on the
 * weight type, the backend type and the device capabilities, so it can be checked without loading a kernel
 * library.
 */
final class GpuWeightFamilies {

/** Weight types with a Q8_1-activation integer-dot matvec ({@code {T}_MatVec_Q8}). */
private static final Set<GGMLType> INT8_MATVEC_TYPES = Collections.unmodifiableSet(EnumSet.of(
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0,
        GGMLType.MXFP4, GGMLType.NVFP4,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
        GGMLType.Q8_K, GGMLType.IQ4_NL, GGMLType.IQ4_XS, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S,
        GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ3_K, GGMLType.IQ3_KS, GGMLType.IQ1_S, GGMLType.IQ1_M));

/** Weight types with a float-input multi-column matvec ({@code {T}_MatVecN}). */
private static final Set<GGMLType> MULTI_COLUMN_TYPES = Collections.unmodifiableSet(EnumSet.of(
        GGMLType.F32, GGMLType.Q8_0, GGMLType.F16, GGMLType.BF16));

private static final Map<GGMLType, WeightKernels> FAMILIES = families();

/** The W12 row set over the nested 8-bit view of the code bytes, serving an open {@link W12QView} scope. */
static final WeightKernels W12Q_FAMILY = new WeightKernels("W12Q_MatVec", "W12Q_MatVecAdd",
        "W12Q_BatchMatMul", "W12Q_BatchMatMul_2D", null, null, null);
private static Map<GGMLType, WeightKernels> families() {
    Map<GGMLType, WeightKernels> m = new EnumMap<>(GGMLType.class);
    m.put(GGMLType.F32, new WeightKernels("Float_MatVec", "Float_MatVecAdd", "LLM_Batch_Float_MatMul",
            "LLM_Batch_Float_MatMul_2D", "F32_MatMul_Tiled", "LLM_EmbeddingLookup", "LLM_Batch_EmbeddingLookup"));
    m.put(GGMLType.Q8_0, new WeightKernels("Q8_0_MatVec", "Q8_0_MatVecAdd", "LLM_Batch_Q8_0_MatMul",
            "LLM_Batch_Q8_0_MatMul_2D", "Q8_0_MatMul_Tiled", "LLM_Q8_0_EmbeddingLookup",
            "LLM_Batch_Q8_0_EmbeddingLookup"));
    m.put(GGMLType.MXFP4, new WeightKernels("MXFP4_MatVec", "MXFP4_MatVecAdd", "MXFP4_BatchMatMul",
            "MXFP4_BatchMatMul_2D", "MXFP4_MatMul_Tiled", "LLM_MXFP4_EmbeddingLookup",
            "MXFP4_BatchEmbeddingLookup"));
    GGMLType[] standard = {
        GGMLType.F16, GGMLType.BF16,
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.IQ4_NL, GGMLType.IQ4_XS,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K,
        GGMLType.IQ3_XXS, GGMLType.IQ3_S, GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ1_S,
        GGMLType.IQ1_M,
        GGMLType.IQ3_K, GGMLType.IQ3_KS,
        GGMLType.TQ1_0, GGMLType.TQ2_0, GGMLType.NVFP4, GGMLType.Q1_0, GGMLType.Q2_0,
        GGMLType.F8_E4M3, GGMLType.F8_E5M2, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B32, GGMLType.MXFP8,
        GGMLType.MXFP6_E3M2, GGMLType.MXFP6_E2M3, GGMLType.I8_B128, GGMLType.NF4, GGMLType.FP4_BNB,
    };
    for (GGMLType t : standard) {
        m.put(t, WeightKernels.standard(t.name()));
    }
    // The integer-dot lane: the decode matvec forms that take a Q8_1 activation row, added to the
    // family the type already has (the legacy-named Q8_0 and MXFP4 families keep their batch kernels).
    for (GGMLType t : INT8_MATVEC_TYPES) {
        m.put(t, WeightKernels.int8(m.get(t)));
    }
    // The float-input multi-column lane: the {T}_MatVecN forms of the dense types, added last so
    // both a Q8_0 weight type carries the float and the integer-dot multi-column rows.
    for (GGMLType t : MULTI_COLUMN_TYPES) {
        m.put(t, WeightKernels.multiColumn(m.get(t)));
    }
    // ggml's per-element F8_E4M3FN (id 43) runs the F8_E4M3 rows: its code bytes equal the extension type's
    // blocks on rows that are whole 32-element runs (GpuWeightUploader keeps other rows off the device
    // kernels and decodes them to F32 on upload).
    m.put(GGMLType.F8_E4M3FN, WeightKernels.standard(GGMLType.F8_E4M3.name()));
    // W12: the row kernels decode the lossless 12-bit blocks in registers. A prefill wider than
    // W12_MATVEC_MAX_TOKENS decodes the tensor into its BF16 mirror and runs the BF16 family, so the type
    // has no tiled or lookup kernels of its own.
    m.put(GGMLType.W12, new WeightKernels("W12_MatVec", "W12_MatVecAdd", "W12_BatchMatMul",
            "W12_BatchMatMul_2D", null, null, null));
    return Collections.unmodifiableMap(m);
}

/** Weight types with hardware-FP8 decode rows ({@code *_Hw_*} kernels): the E4M3FN/E5M2 code types. */
private static final Set<GGMLType> HW_FP8_TYPES = Collections.unmodifiableSet(EnumSet.of(
        GGMLType.F8_E4M3, GGMLType.F8_E4M3FN, GGMLType.F8_E4M3_B128, GGMLType.MXFP8, GGMLType.F8_E5M2));

/** Lowest CUDA compute capability with the hardware FP8 conversions ({@code sm_89}: cvt.rn.f16x2.e4m3x2/e5m2x2). */
private static final int HW_FP8_COMPUTE_CAPABILITY = 89;

/**
 * Hardware-FP8 decode is selected on a device that reports the feature: CUDA sm_89 or higher
 * ({@link DeviceCapabilities#getComputeCapability()}), which converts the E4M3FN/E5M2 code pairs in hardware.
 * A missing feature, or a CPU/OpenCL/Vulkan device, selects the generic rows.
 */
private static boolean hardwareFp8Decode(ComputeBackend.BackendType backendType, DeviceCapabilities caps) {
    if (backendType == null || caps == null) {
        return false;
    }
    return backendType == ComputeBackend.BackendType.CUDA
            && caps.getComputeCapability() >= HW_FP8_COMPUTE_CAPABILITY;
}

/**
 * The {@code {T}_Hw_*} fast-path kernels of one weight type (see the class notes); these are the same
 * argument lists and decoding contract as the generic rows, with the FP8 code pairs converted in hardware on
 * CUDA sm_89+, so the results are bit-identical.
 */
private static WeightKernels hardware(String t) {
    return new WeightKernels(t + "_Hw_MatVec", t + "_Hw_MatVecAdd", t + "_Hw_BatchMatMul",
            t + "_Hw_BatchMatMul_2D", t + "_Hw_MatMul_Tiled", t + "_Hw_EmbeddingLookup",
            t + "_Hw_BatchEmbeddingLookup");
}

    /** True when GpuWeightFamilies has projection and lookup kernels for weights of this type. */
    static boolean supportsWeightType(GGMLType type) {
        return FAMILIES.containsKey(type);
    }

    /** The {@code {T}_MatVec} kernel name of a weight type ({@code Float_MatVec} for F32); null when the type has
     *  no GPU kernels. The name is the family's, so the types that share a family share the kernel rows. */
    static String matVecKernel(GGMLType type) {
        WeightKernels k = FAMILIES.get(type);
        return k == null ? null : k.matvec;
    }

    /** The single-token embedding-lookup kernel name of a weight type; null when the type has no GPU kernels. */
    static String lookupKernel(GGMLType type) {
        WeightKernels k = FAMILIES.get(type);
        return k == null ? null : k.embed;
    }

    /**
     * The weight kernels of {@code type} on a device: the generic family, or the {@code {T}_Hw_*} fast-path set
     * when the type has hardware-FP8 decode rows and the device reports the CUDA sm_89 conversions.
     *
     * <p>The selection depends only on the weight type, the backend type and the device capabilities, so it can be
     * checked without loading a kernel library.</p>
     *
     * @throws GpuModelUnsupportedException when GpuOps has no projection kernels for {@code type}
     */
    static WeightKernels weightKernels(GGMLType type, ComputeBackend.BackendType backendType,
                                       DeviceCapabilities caps) {
        WeightKernels k = FAMILIES.get(type);
        if (k == null) {
            throw new GpuModelUnsupportedException("no GPU projection kernels for weight type " + type);
        }
        return hwFamily(type, k, backendType, caps);
    }

    /** {@code generic}, or the {@code {T}_Hw_*} fast-path set when the device selects hardware FP8 decode. */
    private static WeightKernels hwFamily(GGMLType type, WeightKernels generic,
                                          ComputeBackend.BackendType backendType, DeviceCapabilities caps) {
        return HW_FP8_TYPES.contains(type) && hardwareFp8Decode(backendType, caps)
                ? hardware(familyName(type)) : generic;
    }

    /** The kernel-family name of {@code type}: the enum name, except the types that reuse another family's rows. */
    private static String familyName(GGMLType type) {
        return type == GGMLType.F8_E4M3FN ? GGMLType.F8_E4M3.name() : type.name();
    }

    /** The weight kernels of the tensor's type; throws when GpuOps has none for it. */
    static WeightKernels family(GpuWeight w, ComputeBackend.BackendType backendType, DeviceCapabilities caps) {
        WeightKernels k = FAMILIES.get(w.type());
        if (k == null) {
            throw new GpuModelUnsupportedException("tensor " + w.name() + ": no GPU kernels for weight type "
                    + w.type());
        }
        return hwFamily(w.type(), k, backendType, caps);
    }
}
