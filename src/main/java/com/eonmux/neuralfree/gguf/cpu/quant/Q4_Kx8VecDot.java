package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q4_Kx8;

import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * CPU vec-dot of Q4_K weights in the interleaved {@link Q4_Kx8} layout with Q8_K activations, the row kernel
 * of llama.cpp {@code ggml_gemv_q4_K_8x8_q8_K_generic} (ggml/src/ggml-cpu/repack.cpp:958-1027) folded to one
 * row. Each super-block's quant bytes and 6-bit scales and mins are recovered from the group, its scaled
 * sub-block dots and the {@code bsums} min term are the ones of {@link KQuantVecDot#Q4_K}, and the float
 * combination follows that kernel exactly, so the two return identical values.
 * <p>
 * Instances are immutable, stateless and thread-safe; this is the scalar kernel, the reference and fallback of
 * the Vector API kernel {@code CpuSimd.repacked(GGMLType.Q4_K)} returns.
 */
public final class Q4_Kx8VecDot implements CpuVecDot {

    /** The Q4_Kx8 vec-dot. */
    public static final Q4_Kx8VecDot INSTANCE = new Q4_Kx8VecDot();

    private static final Set<GGMLType> SUPPORTED = Collections.unmodifiableSet(EnumSet.of(GGMLType.Q4_K));

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    /** Per-thread scale and min buffers of one super-block, so a row dot allocates nothing. */
    private static final ThreadLocal<int[][]> SCALES = ThreadLocal.withInitial(() -> new int[2][8]);

    private Q4_Kx8VecDot() {
    }

    /** True for the single type this class handles, Q4_K (read through the repacked layout). */
    public static boolean supports(GGMLType type) {
        return type == GGMLType.Q4_K;
    }

    /** The weight types this class handles. */
    public static Set<GGMLType> supportedTypes() {
        return SUPPORTED;
    }

    /** The weight type this vec-dot reads, Q4_K. */
    public GGMLType weightType() {
        return GGMLType.Q4_K;
    }

    /** The activation type this vec-dot consumes, Q8_K. */
    public GGMLType activationType() {
        return GGMLType.Q8_K;
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
        Q8VecDot.requireWeightType(w, GGMLType.Q4_K);
        x.requireCompatible(w, GGMLType.Q8_K);
        if (w.cols() % Q4_Kx8.ELEMENTS != 0) {
            throw new IllegalArgumentException("the Q4_Kx8 kernel reads rows of whole 2048-element groups; "
                    + w + " has " + w.cols() + " columns");
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x) {
        final byte[] yq = x.qs;
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int groups = x.n / Q4_Kx8.ELEMENTS;
        final int[] scales = SCALES.get()[0];
        final int[] mins = SCALES.get()[1];
        float sumf = 0;
        float sums = 0;
        for (int g = 0; g < groups; g++, off += Q4_Kx8.BYTES) {
            for (int j = 0; j < Q4_Kx8.BLOCKS; j++) {
                final int ib = g * Q4_Kx8.BLOCKS + j;
                final int xo = ib * QK_K;
                Q4_Kx8.scalesMins(b, off, j, scales, mins);
                int isum = 0;
                for (int jp = 0; jp < QK_K / 64; jp++) {
                    for (int e = 0; e < 4; e++) {
                        final int at = off + Q4_Kx8.QS + (4 * jp + e) * 64 + j * 8;
                        for (int l = 0; l < 8; l++) {
                            final int v = b[at + l];
                            isum += scales[2 * jp] * ((v & 0x0F) * yq[xo + 64 * jp + 8 * e + l])
                                    + scales[2 * jp + 1] * (((v >> 4) & 0x0F)
                                            * yq[xo + 64 * jp + 32 + 8 * e + l]);
                        }
                    }
                }
                int minTerm = 0;
                final int bo = ib * BSUMS_PER_BLOCK;
                for (int sb = 0; sb < 8; sb++) {
                    minTerm += mins[sb] * (bsums[bo + 2 * sb] + bsums[bo + 2 * sb + 1]);
                }
                final float d = fp16At(b, off + Q4_Kx8.D + 2 * j) * yd[ib];
                sums += d * isum;
                final float dmin = fp16At(b, off + Q4_Kx8.DMIN + 2 * j) * yd[ib];
                sumf -= dmin * minTerm;
            }
        }
        return sumf + sums;
    }

    @Override
    public String toString() {
        return "Q4_Kx8VecDot{Q4_Kx8 x Q8_K}";
    }
}
