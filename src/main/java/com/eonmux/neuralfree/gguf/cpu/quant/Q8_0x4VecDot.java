package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q8_0x4;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of Q8_0 weights in the interleaved {@link Q8_0x4} layout with Q8_0 activations, the row kernel
 * of llama.cpp {@code ggml_gemv_q8_0_4x4_q8_0_generic} (ggml/src/ggml-cpu/repack.cpp:1274-1319) folded to one
 * row. Each 4-byte slice of a group holds the same four quant positions of the group's four blocks, and each
 * block's 32 integer products are combined with its own half delta, exactly as {@link Q8VecDot#Q8_0} does, so
 * the two kernels return identical values.
 * <p>
 * Instances are immutable, stateless and thread-safe; this is the scalar kernel, the reference and fallback of
 * the Vector API kernel {@code CpuSimd.repacked(GGMLType.Q8_0)} returns.
 */
public final class Q8_0x4VecDot implements CpuVecDot {

    /** The Q8_0x4 vec-dot. */
    public static final Q8_0x4VecDot INSTANCE = new Q8_0x4VecDot();

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(EnumSet.of(GGMLType.Q8_0));

    private Q8_0x4VecDot() {
    }

    /** True for the single type this class handles, Q8_0 (read through the repacked layout). */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.Q8_0;
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /** The weight type this vec-dot reads, Q8_0. */
    public GGMLType weightType() {
        return GGMLType.Q8_0;
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
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        x.requireCompatible(w, GGMLType.Q8_0);
        if (w.cols() % Q8_0x4.ELEMENTS != 0) {
            throw new IllegalArgumentException("the Q8_0x4 kernel reads rows of whole 128-element groups; "
                    + w + " has " + w.cols() + " columns");
        }
    }

    /** Per group of four blocks: the 32-element dot of each block, scaled by its own delta and the activation delta. */
    private static float dot(byte[] b, int off, QuantizedActivations x) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final int groups = x.n / Q8_0x4.ELEMENTS;
        float sumf = 0;
        for (int g = 0; g < groups; g++, off += Q8_0x4.BYTES) {
            final int qs = off + Q8_0x4.QS;
            for (int j = 0; j < Q8_0x4.BLOCKS; j++) {
                final int ib = g * Q8_0x4.BLOCKS + j;
                final int xo = ib * 32;
                int sumi = 0;
                for (int c = 0; c < 8; c++) {
                    final int at = qs + c * 16 + j * 4;
                    for (int k = 0; k < 4; k++) {
                        sumi += b[at + k] * yq[xo + c * 4 + k];
                    }
                }
                sumf += sumi * (fp16At(b, off + Q8_0x4.D + 2 * j) * yd[ib]);
            }
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "Q8_0x4VecDot{Q8_0x4 x Q8_0}";
    }
}
