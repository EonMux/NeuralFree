package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.cpu.repack.Q4_Kx8;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q4_K weights in the interleaved {@link Q4_Kx8} layout with Q8_K activations,
 * {@link Q4_Kx8VecDot} with each eight-byte quant slice widened to int lanes: the low nibbles multiply the
 * first, the shifted high nibbles the second 8-element activation run, each run times its 6-bit sub-block
 * scale. The scale decoding, the {@code bsums} min term and the float combination are the scalar kernel's, so
 * the two return identical values. Immutable, stateless and thread-safe.
 */
public final class SimdQ4_Kx8 implements CpuVecDot {

    /** The Q4_Kx8 vec-dot. */
    public static final SimdQ4_Kx8 INSTANCE = new SimdQ4_Kx8();

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    /** Per-thread scale and min buffers of one super-block. */
    private static final ThreadLocal<int[][]> SCALES = ThreadLocal.withInitial(() -> new int[2][8]);

    private SimdQ4_Kx8() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x, SimdI8.activations(x));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x, SimdI8.activations(x));
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

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
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
                IntVector acc = SimdI8.zero(SimdI8.I8);
                for (int jp = 0; jp < QK_K / 64; jp++) {
                    for (int e = 0; e < 4; e++) {
                        final IntVector v = SimdI8.widen(ByteVector.fromArray(SimdI8.B8,
                                b, off + Q4_Kx8.QS + (4 * jp + e) * 64 + j * 8),
                                0, SimdI8.I8);
                        final int at = xo + 64 * jp + 8 * e;
                        acc = v.and(0x0F).mul(IntVector.fromArray(SimdI8.I8, yw, at)).mul(scales[2 * jp])
                                .add(v.lanewise(VectorOperators.LSHR, 4).and(0x0F)
                                        .mul(IntVector.fromArray(SimdI8.I8, yw, at + 32))
                                        .mul(scales[2 * jp + 1]))
                                .add(acc);
                    }
                }
                final int isum = ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
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
        return "SimdQ4_Kx8{Q4_Kx8 x Q8_K}";
    }
}
