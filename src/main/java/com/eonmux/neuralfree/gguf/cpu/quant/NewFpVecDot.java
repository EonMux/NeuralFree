package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of the newer low-bit types against Q8_0 activations: NVFP4 (64-element blocks of four
 * UE4M3-scaled FP4 sub-blocks), Q1_0 (128 sign bits per half scale) and Q2_0 (64 two-bit codes per half
 * scale). Each kernel is a port of {@code ggml_vec_dot_<type>_q8_0_generic} (llama.cpp ggml/src/ggml-cpu/quants.c)
 * with the same integer sums and float accumulation order. One weight block spans several Q8_0 blocks:
 * 2 for NVFP4 and Q2_0, 4 for Q1_0.
 * <p>
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class NewFpVecDot implements CpuVecDot {

    public static final NewFpVecDot NVFP4 = new NewFpVecDot(GGMLType.NVFP4, NewFpVecDot::dotNvfp4);
    public static final NewFpVecDot Q1_0 = new NewFpVecDot(GGMLType.Q1_0, NewFpVecDot::dotQ1_0);
    public static final NewFpVecDot Q2_0 = new NewFpVecDot(GGMLType.Q2_0, NewFpVecDot::dotQ2_0);

    private static final Map<GGMLType, NewFpVecDot> BY_TYPE = index(NVFP4, Q1_0, Q2_0);

    /**
     * {@code ggml_ue4m3_to_fp32} (llama.cpp ggml-impl.h:514-527) for every scale byte, built from
     * {@link FpBlockDecoders#ue4m3ToFp32(int)} so it is bit-identical to the decoder.
     */
    private static final float[] UE4M3 = buildUe4m3();

    /** One row kernel: {@code nb} weight blocks starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(ByteBuffer b, int off, QuantizedActivations x, int nb);
    }

    private final GGMLType type;
    private final GGMLType activationType;
    private final int blockElements;
    private final RowKernel kernel;

    private NewFpVecDot(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.activationType = CpuActivationQuantizer.activationType(type);
        this.blockElements = BlockLayout.of(type).blockElements();
        this.kernel = kernel;
    }

    private static Map<GGMLType, NewFpVecDot> index(NewFpVecDot... dots) {
        Map<GGMLType, NewFpVecDot> m = new EnumMap<>(GGMLType.class);
        for (NewFpVecDot d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
    }

    private static float[] buildUe4m3() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.ue4m3ToFp32(i);
        }
        return t;
    }

    /** True when this class has a vec-dot for {@code type}. */
    public static boolean supports(GGMLType type) {
        return type != null && BY_TYPE.containsKey(type);
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return BY_TYPE.keySet();
    }

    /**
     * Returns the vec-dot of {@code type}.
     *
     * @throws IllegalArgumentException when {@code type} is not NVFP4, Q1_0 or Q2_0
     */
    public static NewFpVecDot forType(GGMLType type) {
        NewFpVecDot d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("NewFpVecDot has no kernel for ggml type " + type
                    + "; it handles " + BY_TYPE.keySet());
        }
        return d;
    }

    /** The weight type of this vec-dot. */
    public GGMLType weightType() {
        return type;
    }

    /** The activation type this vec-dot consumes (Q8_0). */
    public GGMLType activationType() {
        return activationType;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return kernel.dot(w.rowBuffer(row), w.rowOffset(row), x, (int) (w.cols() / blockElements));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final int nb = (int) (w.cols() / blockElements);
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> kernel.dot(b, off, x, nb));
    }

    private void check(CpuTensorRef w, QuantizedActivations x) {
        if (w == null || x == null) {
            throw new IllegalArgumentException("weight tensor and activations must not be null");
        }
        if (w.type() != type) {
            throw new IllegalArgumentException("the " + type + " vec-dot cannot read a " + w.type() + " tensor");
        }
        x.requireCompatible(w, activationType);
    }

    /**
     * {@code ggml_vec_dot_nvfp4_q8_0_generic} (llama.cpp ggml-cpu/quants.c:330-363): sub-block {@code s} (16
     * elements) of weight block {@code ib} pairs with half {@code s % 2} of Q8_0 block {@code 2*ib + s/2}.
     */
    private static float dotNvfp4(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int nSub = QK_NVFP4 / QK_NVFP4_SUB;
        final int subHalf = QK_NVFP4_SUB / 2;
        float sumf = 0;
        for (int ib = 0; ib < nb; ++ib, off += NVFP4_BYTES) {
            for (int s = 0; s < nSub; ++s) {
                final float d = UE4M3[b.get(off + NVFP4_D + s) & 0xFF];
                final int q8Block = 2 * ib + s / 2;
                final int o = q8Block * QK8_0 + (s % 2) * QK_NVFP4_SUB;
                final float dy = yd[q8Block];
                final int qs = off + NVFP4_QS + s * subHalf;
                int sumiLo = 0;
                int sumiHi = 0;
                for (int j = 0; j < subHalf; ++j) {
                    final int qv = b.get(qs + j) & 0xFF;
                    sumiLo += q8[o + j] * QuantTables.kvalueFp4(qv & 0xf);
                    sumiHi += q8[o + j + subHalf] * QuantTables.kvalueFp4(qv >>> 4);
                }
                sumf += dy * d * (sumiLo + sumiHi);
            }
        }
        return sumf;
    }

    /**
     * {@code ggml_vec_dot_q1_0_q8_0_generic} (llama.cpp ggml-cpu/quants.c:127-175): bit {@code e} of the 32-bit
     * little-endian word {@code qs[4k .. 4k+4)} selects {@code +q8} or {@code -q8} for element {@code e} of
     * Q8_0 block {@code 4i + k}.
     */
    private static float dotQ1_0(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int q8PerBlock = QK1_0 / QK8_0;
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++, off += Q1_0_BYTES) {
            final float d0 = fp16At(b, off + Q1_0_D);
            float sumi = 0.0f;
            for (int k = 0; k < q8PerBlock; k++) {
                final int q8Block = i * q8PerBlock + k;
                final int o = q8Block * QK8_0;
                final int bits = b.getInt(off + Q1_0_QS + 4 * k);
                int sumiBlock = 0;
                for (int e = 0; e < QK8_0; e++) {
                    final int m = ((bits >>> e) & 1) - 1;
                    sumiBlock += (q8[o + e] ^ m) - m;
                }
                sumi += yd[q8Block] * sumiBlock;
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    /**
     * {@code ggml_vec_dot_q2_0_q8_0_generic} (llama.cpp ggml-cpu/quants.c:177-223): bits {@code [2e, 2e+2)} of the
     * 64-bit little-endian word {@code qs[8k .. 8k+8)} hold the code of element {@code e} of Q8_0 block
     * {@code 2i + k}, mapped {0,1,2,3} to {-1,0,+1,+2}.
     */
    private static float dotQ2_0(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int q8PerBlock = QK2_0 / QK8_0;
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++, off += Q2_0_BYTES) {
            final float d0 = fp16At(b, off + Q2_0_D);
            float sumi = 0.0f;
            for (int k = 0; k < q8PerBlock; k++) {
                final int q8Block = i * q8PerBlock + k;
                final int o = q8Block * QK8_0;
                final long codes = b.getLong(off + Q2_0_QS + 8 * k);
                int sumiBlock = 0;
                for (int e = 0; e < QK8_0; e++) {
                    sumiBlock += (((int) (codes >>> (2 * e)) & 3) - 1) * q8[o + e];
                }
                sumi += yd[q8Block] * sumiBlock;
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "NewFpVecDot{" + type + " x " + activationType + "}";
    }
}
