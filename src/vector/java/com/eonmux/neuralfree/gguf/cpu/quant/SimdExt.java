package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.FloatVector;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.BNB4_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.BNB4_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.BNB4_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.I8B128_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_BNB4;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_I8B128;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.floatAt;

/**
 * Vector API vec-dot of the NeuralFree extension block types against Q8_0 activations whose vector form
 * measured faster than the scalar {@link ExtVecDot} kernel on the CPU backend: I8_B128 (int lanes through
 * {@link SimdI8}), NF4 and FP4_BNB (the eight float partial sums of each 32-element run as the lanes of one
 * float vector). Every lane product and every run sum is bit-identical to the scalar kernel.
 * <p>
 * The other extension types (F8_E4M3, F8_E5M2, F8_E4M3_B128, F8_E4M3_B32, MXFP8, MXFP6_E3M2, MXFP6_E2M3)
 * stay on the scalar kernel: their per-element codebook decode is faster than any byte-lane decode the Vector
 * API offers, and the measured vector forms were slower. Immutable, stateless and thread-safe.
 */
public final class SimdExt implements CpuVecDot {

    /** I8_B128 x Q8_0. */
    public static final SimdExt I8_B128 = new SimdExt(GGMLType.I8_B128, SimdExt::dotI8B128);
    /** NF4 x Q8_0. */
    public static final SimdExt NF4 = new SimdExt(GGMLType.NF4, SimdExt::dotNf4);
    /** FP4_BNB x Q8_0. */
    public static final SimdExt FP4_BNB = new SimdExt(GGMLType.FP4_BNB, SimdExt::dotFp4Bnb);

    private static final Map<GGMLType, SimdExt> BY_TYPE = index(I8_B128, NF4, FP4_BNB);

    /** NF4 codebook ({@link QuantTables#nf4Code(int)}). */
    private static final float[] NF4_CODE = buildCodebook(false);
    /** FP4_BNB codebook ({@link QuantTables#fp4BnbCode(int)}). */
    private static final float[] FP4_BNB_CODE = buildCodebook(true);

    /** The calling thread's eight-lane decode window of the nibble codebooks. */
    private static final ThreadLocal<float[]> VALUES = ThreadLocal.withInitial(() -> new float[8]);

    /** One row kernel: {@code nb} weight blocks starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(byte[] b, int off, QuantizedActivations x, int nb, float[] yf);
    }

    private final GGMLType type;
    private final GGMLType activationType;
    private final int blockElements;
    private final RowKernel kernel;

    private SimdExt(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.activationType = CpuActivationQuantizer.activationType(type);
        this.blockElements = BlockLayout.of(type).blockElements();
        this.kernel = kernel;
    }

    private static Map<GGMLType, SimdExt> index(SimdExt... dots) {
        Map<GGMLType, SimdExt> m = new EnumMap<>(GGMLType.class);
        for (SimdExt d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
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
     * @throws IllegalArgumentException when {@code type} is not one of the extension block types here
     */
    public static SimdExt forType(GGMLType type) {
        SimdExt d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("SimdExt has no kernel for ggml type " + type
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
                SimdFp.activations(x.qs, x.n));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return kernel.dot(row, off, x, x.n / blockElements, SimdFp.activations(x.qs, x.n));
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

    /**
     * I8_B128: run {@code r} of block {@code ib} pairs with Q8_0 block {@code 4*ib + r}; the run reduces in
     * int lanes and {@code sumf += sumi * (d_w * d_x)}.
     */
    private static float dotI8B128(byte[] b, int off, QuantizedActivations x, int nb, float[] yf) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final int runs = QK_I8B128 / QK8_0;
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += I8B128_BYTES) {
            final float d = floatAt(b, off + I8B128_D);
            for (int r = 0; r < runs; r++) {
                final int q8Block = ib * runs + r;
                final int sumi = SimdI8.dot32(b, off + I8B128_QS + r * QK8_0, q8, q8Block * QK8_0);
                sumf += sumi * (d * yd[q8Block]);
            }
        }
        return sumf;
    }

    private static float dotNf4(byte[] b, int off, QuantizedActivations x, int nb, float[] yf) {
        return dotBnb4(b, off, x, nb, yf, NF4_CODE);
    }

    private static float dotFp4Bnb(byte[] b, int off, QuantizedActivations x, int nb, float[] yf) {
        return dotBnb4(b, off, x, nb, yf, FP4_BNB_CODE);
    }

    /**
     * NF4 / FP4_BNB: the low nibbles of block {@code ib} are elements {@code [0, 32)} and pair with Q8_0 block
     * {@code 2*ib}; the high nibbles are elements {@code [32, 64)} and pair with Q8_0 block {@code 2*ib + 1}.
     * Each run adds {@code (absmax * d_x) * sum_j code[nibble_j] * q_j}, its eight partial sums in the lanes.
     */
    private static float dotBnb4(byte[] b, int off, QuantizedActivations x, int nb, float[] yf, float[] code) {
        final float[] yd = x.d;
        final int half = QK_BNB4 / 2;
        final float[] values = VALUES.get();
        float sumf = 0.0f;
        for (int ib = 0; ib < nb; ib++, off += BNB4_BYTES) {
            final float absmax = floatAt(b, off + BNB4_D);
            final int qs = off + BNB4_QS;
            final int lo = (2 * ib) * QK8_0;
            final int hi = lo + QK8_0;
            FloatVector accLo = FloatVector.zero(SimdFp.F8);
            FloatVector accHi = FloatVector.zero(SimdFp.F8);
            for (int j = 0; j < half; j += 8) {
                for (int k = 0; k < 8; k++) {
                    values[k] = code[b[qs + j + k] & 0x0F];
                }
                accLo = FloatVector.fromArray(SimdFp.F8, values, 0)
                        .mul(FloatVector.fromArray(SimdFp.F8, yf, lo + j)).add(accLo);
                for (int k = 0; k < 8; k++) {
                    values[k] = code[(b[qs + j + k] & 0xFF) >>> 4];
                }
                accHi = FloatVector.fromArray(SimdFp.F8, values, 0)
                        .mul(FloatVector.fromArray(SimdFp.F8, yf, hi + j)).add(accHi);
            }
            sumf += (absmax * yd[2 * ib]) * SimdFp.pairwise((FloatVector) accLo.check(SimdFp.F8));
            sumf += (absmax * yd[2 * ib + 1]) * SimdFp.pairwise((FloatVector) accHi.check(SimdFp.F8));
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdExt{" + type + " x " + activationType + "}";
    }
}
