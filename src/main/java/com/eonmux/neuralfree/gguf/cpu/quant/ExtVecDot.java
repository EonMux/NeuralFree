package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.quant.QuantTables;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.dotI8;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.putLong;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.floatAt;

/**
 * CPU vec-dot of the NeuralFree extension block types (F8_E4M3, F8_E5M2, F8_E4M3_B128, F8_E4M3_B32, MXFP8,
 * MXFP6_E3M2, MXFP6_E2M3, I8_B128, NF4, FP4_BNB) with
 * Q8_0 activations, in the shape of ggml's {@code ggml_vec_dot_<w>_q8_0} kernels (llama.cpp
 * ggml/src/ggml-cpu/quants.c). Each weight block is walked in 32-element runs, one per Q8_0 activation block:
 * a run contributes {@code d_w * d_x * sum_j code_j * q_j}, where {@code code_j} is the decoded element (E4M3 value,
 * codebook value or signed integer) and {@code d_w} the weight scale of the run's block (1 for F8_E4M3, whose
 * per-tensor scale is applied after the matmul). I8_B128 reduces each run in integers, exactly as the Q8_0
 * vec-dot does, and applies {@code d_w(128-block) * d_x(run)} per run.
 * <p>
 * The decoded values equal {@link com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders} element for element, so a
 * row dot equals the dot of the decoded row with the dequantized activations up to float summation order. The
 * NF4 / FP4_BNB runs accumulate eight float partial sums, lanes {@code 0..7} of the Vector API kernel, reduced
 * pairwise, so {@code SimdExt} reproduces every run's value exactly.
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class ExtVecDot implements CpuVecDot {

    /** Value of every OCP-MX FP6 E3M2 code, from {@link FpBlockDecoders#e3m2ToFp32(int)}; declared before the
     *  vec-dot constants, whose MXFP6 lambdas read it (a field initializer may not forward-reference a field). */
    private static final float[] E3M2 = buildE3m2();
    /** Value of every OCP-MX FP6 E2M3 code, from {@link FpBlockDecoders#e2m3ToFp32(int)}. */
    private static final float[] E2M3 = buildE2m3();

    public static final ExtVecDot F8_E4M3 = new ExtVecDot(GGMLType.F8_E4M3, ExtVecDot::dotF8E4m3);
    public static final ExtVecDot F8_E5M2 = new ExtVecDot(GGMLType.F8_E5M2, ExtVecDot::dotF8E5m2);
    public static final ExtVecDot F8_E4M3_B128 = new ExtVecDot(GGMLType.F8_E4M3_B128, ExtVecDot::dotF8E4m3B128);
    public static final ExtVecDot F8_E4M3_B32 = new ExtVecDot(GGMLType.F8_E4M3_B32, ExtVecDot::dotF8E4m3B32);
    public static final ExtVecDot MXFP8 = new ExtVecDot(GGMLType.MXFP8, ExtVecDot::dotMxfp8);
    public static final ExtVecDot MXFP6_E3M2 = new ExtVecDot(GGMLType.MXFP6_E3M2,
            (b, off, x, nb, aux) -> dotMxfp6(b, off, x, nb, aux, E3M2));
    public static final ExtVecDot MXFP6_E2M3 = new ExtVecDot(GGMLType.MXFP6_E2M3,
            (b, off, x, nb, aux) -> dotMxfp6(b, off, x, nb, aux, E2M3));
    public static final ExtVecDot I8_B128 = new ExtVecDot(GGMLType.I8_B128, ExtVecDot::dotI8B128);
    public static final ExtVecDot NF4 = new ExtVecDot(GGMLType.NF4, ExtVecDot::dotNf4);
    public static final ExtVecDot FP4_BNB = new ExtVecDot(GGMLType.FP4_BNB, ExtVecDot::dotFp4Bnb);

    private static final Map<GGMLType, ExtVecDot> BY_TYPE = index(F8_E4M3, F8_E5M2, F8_E4M3_B128, F8_E4M3_B32, MXFP8,
            MXFP6_E3M2, MXFP6_E2M3, I8_B128, NF4, FP4_BNB);

    /** Value of every E4M3FN code, from {@link FpBlockDecoders#e4m3ToFp32(int)} so it matches the decoder. */
    private static final float[] E4M3 = buildE4m3();
    /** Value of every E5M2 code (IEEE half high bytes), from {@link FpBlockDecoders#e5m2ToFp32(int)}. */
    private static final float[] E5M2 = buildE5m2();
    /** NF4 codebook ({@link QuantTables#nf4Code(int)}). */
    private static final float[] NF4_CODE = buildCodebook(false);
    /** FP4_BNB codebook ({@link QuantTables#fp4BnbCode(int)}). */
    private static final float[] FP4_BNB_CODE = buildCodebook(true);

    /** One row kernel: {@code nb} weight blocks starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux);
    }

    private final GGMLType type;
    private final GGMLType activationType;
    private final int blockElements;
    private final RowKernel kernel;

    private ExtVecDot(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.activationType = CpuActivationQuantizer.activationType(type);
        this.blockElements = BlockLayout.of(type).blockElements();
        this.kernel = kernel;
    }

    private static Map<GGMLType, ExtVecDot> index(ExtVecDot... dots) {
        Map<GGMLType, ExtVecDot> m = new EnumMap<>(GGMLType.class);
        for (ExtVecDot d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
    }

    private static float[] buildE4m3() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e4m3ToFp32(i);
        }
        return t;
    }

    private static float[] buildE5m2() {
        float[] t = new float[256];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e5m2ToFp32(i);
        }
        return t;
    }

    private static float[] buildE3m2() {
        float[] t = new float[64];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e3m2ToFp32(i);
        }
        return t;
    }

    private static float[] buildE2m3() {
        float[] t = new float[64];
        for (int i = 0; i < t.length; i++) {
            t[i] = FpBlockDecoders.e2m3ToFp32(i);
        }
        return t;
    }

    private static float[] buildCodebook(boolean fp4) {
        float[] t = new float[16];
        for (int i = 0; i < t.length; i++) {
            t[i] = fp4 ? QuantTables.fp4BnbCode(i) : QuantTables.nf4Code(i);
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
     * @throws IllegalArgumentException when {@code type} is not a NeuralFree extension block type
     */
    public static ExtVecDot forType(GGMLType type) {
        ExtVecDot d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("ExtVecDot has no kernel for ggml type " + type
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
        return kernel.dot(CpuVecDotRows.copyRow(w, row), 0, x, (int) (w.cols() / blockElements),
                Q8VecDot.scratch().quants);
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return kernel.dot(row, off, x, x.n / blockElements,
                Q8VecDot.scratch().quants);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
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

    /** {@code sum_j e4m3(w[qs + j]) * q8[o + j]} over one 32-element run. */
    private static float runE4m3(byte[] b, int qs, byte[] q8, int o) {
        final float[] lut = E4M3;
        float sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            sum += lut[b[qs + j] & 0xFF] * q8[o + j];
        }
        return sum;
    }

    /** F8_E4M3: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += d_x * sum}. */
    private static float dotF8E4m3(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8_E4M3_BYTES) {
            sumf += yd[ib] * runE4m3(b, off + F8_E4M3_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** {@code sum_j e5m2(w[qs + j]) * q8[o + j]} over one 32-element run. */
    private static float runE5m2(byte[] b, int qs, byte[] q8, int o) {
        final float[] lut = E5M2;
        float sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            sum += lut[b[qs + j] & 0xFF] * q8[o + j];
        }
        return sum;
    }

    /** F8_E5M2: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += d_x * sum}. */
    private static float dotF8E5m2(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8_E5M2_BYTES) {
            sumf += yd[ib] * runE5m2(b, off + F8_E5M2_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** F8_E4M3_B128: run {@code r} of block {@code ib} pairs with Q8_0 block {@code 4*ib + r}; {@code d_w * d_x}. */
    private static float dotF8E4m3B128(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int runs = QK_F8B128 / QK8_0;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8B128_BYTES) {
            final float d = floatAt(b, off + F8B128_D);
            for (int r = 0; r < runs; r++) {
                final int q8Block = ib * runs + r;
                sumf += (d * yd[q8Block]) * runE4m3(b, off + F8B128_QS + r * QK8_0, q8, q8Block * QK8_0);
            }
        }
        return sumf;
    }

    /** F8_E4M3_B32: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += (d_w * d_x) * sum}. */
    private static float dotF8E4m3B32(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8B32_BYTES) {
            final float d = floatAt(b, off + F8B32_D);
            sumf += (d * yd[ib]) * runE4m3(b, off + F8B32_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** MXFP8: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += (d_x * 2^(e-127)) * sum}. */
    private static float dotMxfp8(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += MXFP8_BYTES) {
            final float d = FpBlockDecoders.e8m0ToFp32(b[off + MXFP8_E] & 0xFF);
            sumf += (yd[ib] * d) * runE4m3(b, off + MXFP8_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** Code {@code j} of the 24 packed 6-bit-code bytes starting at {@code qs}: element {@code j} of the block. */
    private static int code6(byte[] b, int qs, int j) {
        final int b0 = qs + (j >> 2) * 3;
        final int word = (b[b0] & 0xFF) | (b[b0 + 1] & 0xFF) << 8 | (b[b0 + 2] & 0xFF) << 16;
        return (word >>> (6 * (j & 3))) & 0x3F;
    }

    /**
     * MXFP6: block {@code ib} is Q8_0 run {@code ib}; every code is finite and
     * {@code sumf += (d_x * 2^(e-127)) * sum}. Load rejects the NaN E8M0 exponent byte before the dot runs.
     */
    private static float dotMxfp6(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux, float[] lut) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += MXFP6_BYTES) {
            final float d = FpBlockDecoders.e8m0ToFp32(b[off + MXFP6_E] & 0xFF);
            float sum = 0.0f;
            final int qs = off + MXFP6_QS;
            for (int j = 0; j < QK8_0; j++) {
                sum += lut[code6(b, qs, j)] * q8[ib * QK8_0 + j];
            }
            sumf += (yd[ib] * d) * sum;
        }
        return sumf;
    }

    /**
     * I8_B128: run {@code r} of block {@code ib} pairs with Q8_0 block {@code 4*ib + r}; the run is reduced in
     * integers ({@code sumi = sum_j w.qs[j] * x.qs[j]}, the Q8_0 inner loop) and {@code sumf += sumi * (d_w * d_x)}.
     */
    private static float dotI8B128(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int runs = QK_I8B128 / QK8_0;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += I8B128_BYTES) {
            final float d = floatAt(b, off + I8B128_D);
            for (int r = 0; r < runs; r++) {
                final int q8Block = ib * runs + r;
                final int qs = off + I8B128_QS + r * QK8_0;
                putLong(aux, 0, longAt(b, qs));
                putLong(aux, 8, longAt(b, qs + 8));
                putLong(aux, 16, longAt(b, qs + 16));
                putLong(aux, 24, longAt(b, qs + 24));
                final int sumi = dotI8(aux, 0, q8, q8Block * QK8_0, QK8_0);
                sumf += sumi * (d * yd[q8Block]);
            }
        }
        return sumf;
    }

    private static float dotNf4(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        return dotBnb4(b, off, x, nb, NF4_CODE);
    }

    private static float dotFp4Bnb(byte[] b, int off, QuantizedActivations x, int nb, byte[] aux) {
        return dotBnb4(b, off, x, nb, FP4_BNB_CODE);
    }

    /**
     * NF4 / FP4_BNB: the low nibbles of block {@code ib} are elements {@code [0, 32)} and pair with Q8_0 block
     * {@code 2*ib}; the high nibbles are elements {@code [32, 64)} and pair with Q8_0 block {@code 2*ib + 1}. Each run
     * adds {@code (absmax * d_x) * sum_j code[nibble_j] * q_j}.
     */
    private static float dotBnb4(byte[] b, int off, QuantizedActivations x, int nb, float[] code) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int half = QK_BNB4 / 2;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += BNB4_BYTES) {
            final float absmax = floatAt(b, off + BNB4_D);
            final int qs = off + BNB4_QS;
            final int lo = (2 * ib) * QK8_0;
            final int hi = lo + QK8_0;
            float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
            float l4 = 0.0f, l5 = 0.0f, l6 = 0.0f, l7 = 0.0f;
            float h0 = 0.0f, h1 = 0.0f, h2 = 0.0f, h3 = 0.0f;
            float h4 = 0.0f, h5 = 0.0f, h6 = 0.0f, h7 = 0.0f;
            for (int j = 0; j < half; j += 8) {
                final int q0 = b[qs + j] & 0xFF;
                final int q1 = b[qs + j + 1] & 0xFF;
                final int q2 = b[qs + j + 2] & 0xFF;
                final int q3 = b[qs + j + 3] & 0xFF;
                final int q4 = b[qs + j + 4] & 0xFF;
                final int q5 = b[qs + j + 5] & 0xFF;
                final int q6 = b[qs + j + 6] & 0xFF;
                final int q7 = b[qs + j + 7] & 0xFF;
                l0 += code[q0 & 0x0F] * q8[lo + j];
                l1 += code[q1 & 0x0F] * q8[lo + j + 1];
                l2 += code[q2 & 0x0F] * q8[lo + j + 2];
                l3 += code[q3 & 0x0F] * q8[lo + j + 3];
                l4 += code[q4 & 0x0F] * q8[lo + j + 4];
                l5 += code[q5 & 0x0F] * q8[lo + j + 5];
                l6 += code[q6 & 0x0F] * q8[lo + j + 6];
                l7 += code[q7 & 0x0F] * q8[lo + j + 7];
                h0 += code[q0 >>> 4] * q8[hi + j];
                h1 += code[q1 >>> 4] * q8[hi + j + 1];
                h2 += code[q2 >>> 4] * q8[hi + j + 2];
                h3 += code[q3 >>> 4] * q8[hi + j + 3];
                h4 += code[q4 >>> 4] * q8[hi + j + 4];
                h5 += code[q5 >>> 4] * q8[hi + j + 5];
                h6 += code[q6 >>> 4] * q8[hi + j + 6];
                h7 += code[q7 >>> 4] * q8[hi + j + 7];
            }
            final float sumLo = ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7));
            final float sumHi = ((h0 + h1) + (h2 + h3)) + ((h4 + h5) + (h6 + h7));
            sumf += (absmax * yd[2 * ib]) * sumLo;
            sumf += (absmax * yd[2 * ib + 1]) * sumHi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "ExtVecDot{" + type + " x " + activationType + "}";
    }
}
