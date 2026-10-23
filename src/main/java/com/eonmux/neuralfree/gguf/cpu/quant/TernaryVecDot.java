package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.*;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of the ternary BitNet types TQ1_0 and TQ2_0 against Q8_K activations, ports of
 * {@code ggml_vec_dot_tq1_0_q8_K_generic} and {@code ggml_vec_dot_tq2_0_q8_K_generic}
 * (llama.cpp ggml/src/ggml-cpu/quants.c:481-563).
 * <p>
 * ggml sums {@code (t - 1) * q8} per element with {@code t} in {0, 1, 2} (TQ1_0) or {0..3} (TQ2_0). The
 * kernels here compute the same exact integer as {@code sum(t * q8) - sum(q8)}, taking {@code sum(q8)} from
 * the Q8_K {@code bsums}, and apply the same float scaling, so results equal ggml's reference path.
 * <p>
 * Instances are immutable, thread-safe and allocate nothing per call.
 */
public final class TernaryVecDot implements CpuVecDot {

    public static final TernaryVecDot TQ1_0 = new TernaryVecDot(GGMLType.TQ1_0, TernaryVecDot::dotTq1_0);
    public static final TernaryVecDot TQ2_0 = new TernaryVecDot(GGMLType.TQ2_0, TernaryVecDot::dotTq2_0);

    private static final Map<GGMLType, TernaryVecDot> BY_TYPE = index(TQ1_0, TQ2_0);

    /** Number of 16-element {@code bsums} groups per Q8_K block. */
    private static final int BSUM_GROUPS = QK_K / 16;

    /** One row kernel: {@code nb} weight blocks starting at absolute index {@code off} of {@code b}. */
    @FunctionalInterface
    private interface RowKernel {
        float dot(ByteBuffer b, int off, QuantizedActivations x, int nb);
    }

    private final GGMLType type;
    private final GGMLType activationType;
    private final int blockElements;
    private final RowKernel kernel;

    private TernaryVecDot(GGMLType type, RowKernel kernel) {
        this.type = type;
        this.activationType = CpuActivationQuantizer.activationType(type);
        this.blockElements = BlockLayout.of(type).blockElements();
        this.kernel = kernel;
    }

    private static Map<GGMLType, TernaryVecDot> index(TernaryVecDot... dots) {
        Map<GGMLType, TernaryVecDot> m = new EnumMap<>(GGMLType.class);
        for (TernaryVecDot d : dots) {
            m.put(d.type, d);
        }
        return Collections.unmodifiableMap(m);
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
     * @throws IllegalArgumentException when {@code type} is not TQ1_0 or TQ2_0
     */
    public static TernaryVecDot forType(GGMLType type) {
        TernaryVecDot d = type == null ? null : BY_TYPE.get(type);
        if (d == null) {
            throw new IllegalArgumentException("TernaryVecDot has no kernel for ggml type " + type
                    + "; it handles " + BY_TYPE.keySet());
        }
        return d;
    }

    /** The weight type of this vec-dot. */
    public GGMLType weightType() {
        return type;
    }

    /** The activation type this vec-dot consumes (Q8_K). */
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

    /** {@code sum(q8)} over the 256 activations of Q8_K block {@code i}, from its {@code bsums}. */
    private static int blockSum(short[] bsums, int i) {
        final int base = i * BSUM_GROUPS;
        int s = 0;
        for (int g = 0; g < BSUM_GROUPS; g++) {
            s += bsums[base + g];
        }
        return s;
    }

    /**
     * Trit of a TQ1_0 byte {@code v} after multiplying by {@code pow3}: {@code q = (uint8)(v * pow3)},
     * {@code (q * 3) >> 8} (llama.cpp ggml-cpu/quants.c:503-504).
     */
    private static int trit(int v, int pow3) {
        return (((v * pow3) & 0xFF) * 3) >>> 8;
    }

    /**
     * {@code ggml_vec_dot_tq1_0_q8_K_generic} (llama.cpp ggml-cpu/quants.c:481-531). Element order inside a block:
     * the first 32 {@code qs} bytes give 5 trits each to {@code q8[l*32 + m]}, the last 16 {@code qs} bytes
     * give 5 trits each to {@code q8[160 + l*16 + m]}, and the 4 {@code qh} bytes give 4 trits each to
     * {@code q8[240 + l*4 + j]}; trit {@code l} uses multiplier {@code 3^l}.
     */
    private static float dotTq1_0(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int full = TQ1_0_QS_BYTES - TQ1_0_QS_BYTES % 32;
        float sumf = 0.0f;
        for (int i = 0, o = 0; i < nb; ++i, off += TQ1_0_BYTES, o += QK_K) {
            final int qs = off + TQ1_0_QS;
            int sum = 0;
            for (int m = 0; m < full; ++m) {
                final int v = b.get(qs + m) & 0xFF;
                final int y0 = o + m;
                sum += trit(v, 1) * q8[y0] + trit(v, 3) * q8[y0 + 32] + trit(v, 9) * q8[y0 + 64]
                        + trit(v, 27) * q8[y0 + 96] + trit(v, 81) * q8[y0 + 128];
            }
            for (int m = 0; m < TQ1_0_QS_BYTES - full; ++m) {
                final int v = b.get(qs + full + m) & 0xFF;
                final int y0 = o + full * 5 + m;
                sum += trit(v, 1) * q8[y0] + trit(v, 3) * q8[y0 + 16] + trit(v, 9) * q8[y0 + 32]
                        + trit(v, 27) * q8[y0 + 48] + trit(v, 81) * q8[y0 + 64];
            }
            final int qh = off + TQ1_0_QH;
            for (int j = 0; j < TQ1_0_QH_BYTES; ++j) {
                final int v = b.get(qh + j) & 0xFF;
                final int y0 = o + TQ1_0_QS_BYTES * 5 + j;
                sum += trit(v, 1) * q8[y0] + trit(v, 3) * q8[y0 + TQ1_0_QH_BYTES]
                        + trit(v, 9) * q8[y0 + 2 * TQ1_0_QH_BYTES] + trit(v, 27) * q8[y0 + 3 * TQ1_0_QH_BYTES];
            }
            sum -= blockSum(bsums, i);
            sumf += (float) sum * (fp16At(b, off + TQ1_0_D) * yd[i]);
        }
        return sumf;
    }

    /**
     * {@code ggml_vec_dot_tq2_0_q8_K_generic} (llama.cpp ggml-cpu/quants.c:533-563): byte {@code j + k} of each
     * 32-byte chunk {@code j} holds the 2-bit codes of {@code q8[4j + 32l + k]} at bits {@code 2l}.
     */
    private static float dotTq2_0(ByteBuffer b, int off, QuantizedActivations x, int nb) {
        final byte[] q8 = x.qs;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int qsBytes = QK_K / 4;
        float sumf = 0.0f;
        for (int i = 0, o = 0; i < nb; ++i, off += TQ2_0_BYTES, o += QK_K) {
            final int qs = off + TQ2_0_QS;
            int sumi = 0;
            for (int j = 0; j < qsBytes; j += 32) {
                final int y0 = o + j * 4;
                for (int k = 0; k < 32; ++k) {
                    final int v = b.get(qs + j + k) & 0xFF;
                    sumi += (v & 3) * q8[y0 + k] + ((v >>> 2) & 3) * q8[y0 + 32 + k]
                            + ((v >>> 4) & 3) * q8[y0 + 64 + k] + (v >>> 6) * q8[y0 + 96 + k];
                }
            }
            sumi -= blockSum(bsums, i);
            final float d = yd[i] * fp16At(b, off + TQ2_0_D);
            sumf += (float) sumi * d;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "TernaryVecDot{" + type + " x " + activationType + "}";
    }
}
