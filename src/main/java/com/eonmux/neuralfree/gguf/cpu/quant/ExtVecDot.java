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

import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.dotI8;
import static com.eonmux.neuralfree.gguf.cpu.quant.Q8VecDot.putLong;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;

/**
 * CPU vec-dot of the NeuralFree extension block types (F8_E4M3, F8_E5M2, F8_E4M3_B128, MXFP8, I8_B128, NF4, FP4_BNB) with
 * Q8_0 activations, in the shape of ggml's {@code ggml_vec_dot_<w>_q8_0} kernels (llama.cpp
 * ggml/src/ggml-cpu/quants.c). Each weight block is walked in 32-element runs, one per Q8_0 activation block:
 * a run contributes {@code d_w * d_x * sum_j code_j * q_j}, where {@code code_j} is the decoded element (E4M3 value,
 * codebook value or signed integer) and {@code d_w} the weight scale of the run's block (1 for F8_E4M3, whose
 * per-tensor scale is applied after the matmul). I8_B128 reduces each run in integers, exactly as the Q8_0
 * vec-dot does, and applies {@code d_w(128-block) * d_x(run)} per run.
 * <p>
 * The decoded values equal {@link com.eonmux.neuralfree.gguf.quant.CpuBlockDecoders} element for element, so a
 * row dot equals the dot of the decoded row with the dequantized activations up to float summation order.
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class ExtVecDot implements CpuVecDot {

    public static final ExtVecDot F8_E4M3 = new ExtVecDot(GGMLType.F8_E4M3, ExtVecDot::dotF8E4m3);
    public static final ExtVecDot F8_E5M2 = new ExtVecDot(GGMLType.F8_E5M2, ExtVecDot::dotF8E5m2);
    public static final ExtVecDot F8_E4M3_B128 = new ExtVecDot(GGMLType.F8_E4M3_B128, ExtVecDot::dotF8E4m3B128);
    public static final ExtVecDot MXFP8 = new ExtVecDot(GGMLType.MXFP8, ExtVecDot::dotMxfp8);
    public static final ExtVecDot I8_B128 = new ExtVecDot(GGMLType.I8_B128, ExtVecDot::dotI8B128);
    public static final ExtVecDot NF4 = new ExtVecDot(GGMLType.NF4, ExtVecDot::dotNf4);
    public static final ExtVecDot FP4_BNB = new ExtVecDot(GGMLType.FP4_BNB, ExtVecDot::dotFp4Bnb);

    private static final Map<GGMLType, ExtVecDot> BY_TYPE = index(F8_E4M3, F8_E5M2, F8_E4M3_B128, MXFP8, I8_B128, NF4, FP4_BNB);

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
        float dot(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux);
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
        return kernel.dot(w.rowBuffer(row), w.rowOffset(row), x, (int) (w.cols() / blockElements),
                Q8VecDot.scratch().quants);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        final int nb = (int) (w.cols() / blockElements);
        final byte[] aux = Q8VecDot.scratch().quants;
        CpuVecDotRows.forEachRow(w, rowStart, rowCount, y, yOff, (b, off) -> kernel.dot(b, off, x, nb, aux));
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
    private static float runE4m3(ByteBuffer b, int qs, byte[] q8, int o) {
        final float[] lut = E4M3;
        float sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            sum += lut[b.get(qs + j) & 0xFF] * q8[o + j];
        }
        return sum;
    }

    /** F8_E4M3: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += d_x * sum}. */
    private static float dotF8E4m3(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8_E4M3_BYTES) {
            sumf += yd[ib] * runE4m3(b, off + F8_E4M3_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** {@code sum_j e5m2(w[qs + j]) * q8[o + j]} over one 32-element run. */
    private static float runE5m2(ByteBuffer b, int qs, byte[] q8, int o) {
        final float[] lut = E5M2;
        float sum = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            sum += lut[b.get(qs + j) & 0xFF] * q8[o + j];
        }
        return sum;
    }

    /** F8_E5M2: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += d_x * sum}. */
    private static float dotF8E5m2(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8_E5M2_BYTES) {
            sumf += yd[ib] * runE5m2(b, off + F8_E5M2_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /** F8_E4M3_B128: run {@code r} of block {@code ib} pairs with Q8_0 block {@code 4*ib + r}; {@code d_w * d_x}. */
    private static float dotF8E4m3B128(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int runs = QK_F8B128 / QK8_0;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += F8B128_BYTES) {
            final float d = b.getFloat(off + F8B128_D);
            for (int r = 0; r < runs; r++) {
                final int q8Block = ib * runs + r;
                sumf += (d * yd[q8Block]) * runE4m3(b, off + F8B128_QS + r * QK8_0, q8, q8Block * QK8_0);
            }
        }
        return sumf;
    }

    /** MXFP8: block {@code ib} is Q8_0 run {@code ib}; {@code sumf += (d_x * 2^(e-127)) * sum}. */
    private static float dotMxfp8(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += MXFP8_BYTES) {
            final float d = FpBlockDecoders.e8m0ToFp32(b.get(off + MXFP8_E));
            sumf += (yd[ib] * d) * runE4m3(b, off + MXFP8_QS, q8, ib * QK8_0);
        }
        return sumf;
    }

    /**
     * I8_B128: run {@code r} of block {@code ib} pairs with Q8_0 block {@code 4*ib + r}; the run is reduced in
     * integers ({@code sumi = sum_j w.qs[j] * x.qs[j]}, the Q8_0 inner loop) and {@code sumf += sumi * (d_w * d_x)}.
     */
    private static float dotI8B128(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int runs = QK_I8B128 / QK8_0;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += I8B128_BYTES) {
            final float d = b.getFloat(off + I8B128_D);
            for (int r = 0; r < runs; r++) {
                final int q8Block = ib * runs + r;
                final int qs = off + I8B128_QS + r * QK8_0;
                putLong(aux, 0, b.getLong(qs));
                putLong(aux, 8, b.getLong(qs + 8));
                putLong(aux, 16, b.getLong(qs + 16));
                putLong(aux, 24, b.getLong(qs + 24));
                final int sumi = dotI8(aux, 0, q8, q8Block * QK8_0, QK8_0);
                sumf += sumi * (d * yd[q8Block]);
            }
        }
        return sumf;
    }

    private static float dotNf4(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        return dotBnb4(b, off, x, nb, NF4_CODE);
    }

    private static float dotFp4Bnb(ByteBuffer b, int off, QuantizedActivations x, int nb, byte[] aux) {
        return dotBnb4(b, off, x, nb, FP4_BNB_CODE);
    }

    /**
     * NF4 / FP4_BNB: the low nibbles of block {@code ib} are elements {@code [0, 32)} and pair with Q8_0 block
     * {@code 2*ib}; the high nibbles are elements {@code [32, 64)} and pair with Q8_0 block {@code 2*ib + 1}. Each run
     * adds {@code (absmax * d_x) * sum_j code[nibble_j] * q_j}.
     */
    private static float dotBnb4(ByteBuffer b, int off, QuantizedActivations x, int nb, float[] code) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int half = QK_BNB4 / 2;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += BNB4_BYTES) {
            final float absmax = b.getFloat(off + BNB4_D);
            final int qs = off + BNB4_QS;
            final int lo = (2 * ib) * QK8_0;
            final int hi = lo + QK8_0;
            float sumLo = 0.0f;
            float sumHi = 0.0f;
            for (int j = 0; j < half; j++) {
                final int q = b.get(qs + j) & 0xFF;
                sumLo += code[q & 0x0F] * q8[lo + j];
                sumHi += code[q >>> 4] * q8[hi + j];
            }
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
