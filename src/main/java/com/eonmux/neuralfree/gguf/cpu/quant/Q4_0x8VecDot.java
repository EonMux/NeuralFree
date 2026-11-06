package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q4_0x8;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of Q4_0 weights in the interleaved {@link Q4_0x8} layout with Q8_0 activations, the row kernel of
 * llama.cpp {@code ggml_gemv_q4_0_8x8_q8_0_generic} (ggml/src/ggml-cpu/repack.cpp:843-885) folded to one row.
 * Each stored byte contributes its two nibbles as the signed quants {@code (qs << 4) >> 4} and
 * {@code (qs & 0xF0) >> 4}, dotted with the same 32 activation quants of the plain Q4_0 kernel, and the float
 * combination {@code sumi * d_x * d_y} per block follows {@link LegacyVecDot#Q4_0} exactly, so the two kernels
 * return identical values.
 * <p>
 * Instances are immutable, stateless and thread-safe; this is the scalar kernel, the reference and fallback of
 * the Vector API kernel {@code CpuSimd.q4_0x8()} returns.
 */
public final class Q4_0x8VecDot implements CpuVecDot {

    /** The Q4_0x8 vec-dot. */
    public static final Q4_0x8VecDot INSTANCE = new Q4_0x8VecDot();

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(EnumSet.of(GGMLType.Q4_0));

    private Q4_0x8VecDot() {
    }

    /** True for the single type this class handles, Q4_0 (read through the repacked layout). */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.Q4_0;
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /** The weight type this vec-dot reads, Q4_0. */
    public GGMLType weightType() {
        return GGMLType.Q4_0;
    }

    /** The activation type this vec-dot consumes, Q8_0. */
    public GGMLType activationType() {
        return GGMLType.Q8_0;
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x);
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x);
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q4_0);
        x.requireCompatible(w, GGMLType.Q8_0);
        if (w.cols() % Q4_0x8.ELEMENTS != 0) {
            throw new IllegalArgumentException("the Q4_0x8 kernel reads rows of whole 256-element groups; "
                    + w + " has " + w.cols() + " columns");
        }
    }

    /**
     * Per group of 8 blocks: {@code sumf += sumi * d[j] * d_y} for each block {@code j}, with {@code sumi} the
     * 32-element int dot of the decoded nibbles and the activation quants. The first 64 quant bytes hold
     * elements {@code 0..7} and {@code 16..23} of each block, the second 64 hold {@code 8..15} and
     * {@code 24..31}.
     */
    private static float dot(byte[] b, int off, QuantizedActivations x) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int groups = x.n / Q4_0x8.ELEMENTS;
        float sumf = 0;
        int ib = 0;
        for (int g = 0; g < groups; g++, off += Q4_0x8.BYTES) {
            final int qs = off + Q4_0x8.QS;
            for (int j = 0; j < Q4_0x8.BLOCKS; j++, ib++) {
                final int y = ib * 32;
                int sumi = 0;
                for (int i = 0; i < 8; i++) {
                    final int lo = b[qs + 8 * j + i];
                    final int hi = b[qs + 64 + 8 * j + i];
                    sumi += ((byte) (lo << 4) >> 4) * yq[y + i]
                            + (((byte) (lo & 0xF0)) >> 4) * yq[y + 16 + i]
                            + ((byte) (hi << 4) >> 4) * yq[y + 8 + i]
                            + (((byte) (hi & 0xF0)) >> 4) * yq[y + 24 + i];
                }
                sumf += sumi * fp16At(b, off + Q4_0x8.D + 2 * j) * yd[ib];
            }
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "Q4_0x8VecDot{Q4_0x8 x Q8_0}";
    }
}
