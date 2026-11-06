package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

/**
 * The Vector API CPU vec-dot engines of this package, the implementation {@link CpuSimd} loads on a Java 17+
 * JVM with the {@code jdk.incubator.vector} module resolved. It covers the legacy 4- and 5-bit types
 * ({@link SimdQ4_0}, {@link SimdQ4_1}, {@link SimdQ5_0}, {@link SimdQ5_1}), Q8_0, the K-quant super-blocks
 * ({@link SimdQ2_K}, {@link SimdQ3_K}, {@link SimdQ4_K}, {@link SimdQ5_K}, {@link SimdQ6_K}), the float types
 * ({@link SimdF32}, {@link SimdF16}, {@link SimdBf16}), the codebook types whose 16-entry table fits byte lanes
 * ({@link SimdIq4Nl}, {@link SimdIq4Xs}, {@link SimdMxfp4}), the ternary types ({@link SimdTq1_0},
 * {@link SimdTq2_0}), the new low-bit types ({@link SimdQ1_0}, {@link SimdQ2_0}) and the extension block types
 * whose vector form measured faster than the scalar kernel ({@link SimdExt}), plus the repacked-layout kernels
 * of {@link #repacked}. Every kernel returns exactly what its scalar counterpart returns, so selection changes
 * speed only.
 * <p>
 * Loading this class initializes its species fields, which links the Vector API and fails when the module is
 * not usable; {@link CpuSimd} treats any such failure as no engine. Instances are immutable and thread-safe.
 */
public final class SimdKernels implements CpuSimd.Engine {

    private static final Map<GGMLType, CpuVecDot> KERNELS = build();

    private static final Map<GGMLType, CpuVecDot> REPACKED = buildRepacked();

    private static Map<GGMLType, CpuVecDot> build() {
        Map<GGMLType, CpuVecDot> m = new EnumMap<>(GGMLType.class);
        m.put(GGMLType.Q8_0, SimdQ8_0.INSTANCE);
        m.put(GGMLType.Q4_0, SimdQ4_0.INSTANCE);
        m.put(GGMLType.Q4_1, SimdQ4_1.INSTANCE);
        m.put(GGMLType.Q5_0, SimdQ5_0.INSTANCE);
        m.put(GGMLType.Q5_1, SimdQ5_1.INSTANCE);
        m.put(GGMLType.Q2_K, SimdQ2_K.INSTANCE);
        m.put(GGMLType.Q3_K, SimdQ3_K.INSTANCE);
        m.put(GGMLType.Q4_K, SimdQ4_K.INSTANCE);
        m.put(GGMLType.Q5_K, SimdQ5_K.INSTANCE);
        m.put(GGMLType.Q6_K, SimdQ6_K.INSTANCE);
        m.put(GGMLType.F32, SimdF32.INSTANCE);
        m.put(GGMLType.F16, SimdF16.INSTANCE);
        m.put(GGMLType.BF16, SimdBf16.INSTANCE);
        m.put(GGMLType.IQ4_NL, SimdIq4Nl.INSTANCE);
        m.put(GGMLType.IQ4_XS, SimdIq4Xs.INSTANCE);
        m.put(GGMLType.MXFP4, SimdMxfp4.INSTANCE);
        m.put(GGMLType.TQ1_0, SimdTq1_0.INSTANCE);
        m.put(GGMLType.TQ2_0, SimdTq2_0.INSTANCE);
        m.put(GGMLType.Q1_0, SimdQ1_0.INSTANCE);
        m.put(GGMLType.Q2_0, SimdQ2_0.INSTANCE);
        for (GGMLType t : SimdExt.supportedTypes()) {
            m.put(t, SimdExt.forType(t));
        }
        return Collections.unmodifiableMap(m);
    }

    private static Map<GGMLType, CpuVecDot> buildRepacked() {
        Map<GGMLType, CpuVecDot> m = new EnumMap<>(GGMLType.class);
        m.put(GGMLType.Q4_0, SimdQ4_0x8.INSTANCE);
        m.put(GGMLType.Q8_0, SimdQ8_0x4.INSTANCE);
        m.put(GGMLType.Q4_K, SimdQ4_Kx8.INSTANCE);
        return Collections.unmodifiableMap(m);
    }

    /** The vec-dot of {@code type}, or null when no Vector API kernel covers it. */
    @Override
    public CpuVecDot vecDot(GGMLType type) {
        return type == null ? null : KERNELS.get(type);
    }

    /** Every weight type with a Vector API kernel (unmodifiable). */
    @Override
    public Set<GGMLType> supportedTypes() {
        return KERNELS.keySet();
    }

    /** The Vector API kernel of {@code type}'s repacked layout, or null when it has none. */
    @Override
    public CpuVecDot repacked(GGMLType type) {
        return type == null ? null : REPACKED.get(type);
    }

    /** The Vector API kernel of the repacked Q4_0x8 layout. */
    @Override
    public CpuVecDot q4_0x8() {
        return SimdQ4_0x8.INSTANCE;
    }

    /** The Vector API int8 primitives of the fused CPU paths. */
    @Override
    public CpuInt8Kernels int8() {
        return SimdInt8Kernels.INSTANCE;
    }

    /** The vector width the kernels target, for {@link CpuSimd#describe()}. */
    @Override
    public String describe() {
        return "vector/" + SimdI8.I8.vectorBitSize() + "b (jdk.incubator.vector)";
    }

    @Override
    public String toString() {
        return "SimdKernels{" + describe() + ", " + KERNELS.keySet() + ", repacked=" + REPACKED.keySet() + "}";
    }
}
