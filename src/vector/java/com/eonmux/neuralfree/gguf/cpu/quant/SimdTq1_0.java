package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QH_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ1_0_QS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of TQ1_0 weights with Q8_K activations, {@link TernaryVecDot#TQ1_0} with the ternary
 * decode in int lanes: {@code trit(v, p) = (((v * p) & 0xFF) * 3) >> 8} for the multipliers {@code 3^l}, so
 * each 16-byte group of a block decodes five trits per byte into their five activation runs. The four trailing
 * {@code qh} bytes stay scalar (four bytes do not fill a vector), and the activation block sum is subtracted
 * from the exact integer sum as {@code ggml_vec_dot_tq1_0_q8_K_generic} does, so the kernel returns what
 * {@link TernaryVecDot#TQ1_0} returns. Immutable, stateless and thread-safe.
 */
public final class SimdTq1_0 implements CpuVecDot {

    /** The TQ1_0 vec-dot. */
    public static final SimdTq1_0 INSTANCE = new SimdTq1_0();

    /** Number of 16-element {@code bsums} groups per Q8_K block. */
    private static final int BSUM_GROUPS = QK_K / 16;

    /** The five trit multipliers {@code 3^l}. */
    private static final int[] POW3 = {1, 3, 9, 27, 81};

    private SimdTq1_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.TQ1_0);
        x.requireCompatible(w, GGMLType.Q8_K);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0, o = 0; i < nb; ++i, off += TQ1_0_BYTES, o += QK_K) {
            final int qs = off + TQ1_0_QS;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int m = 0; m < TQ1_0_QS_BYTES; m += 16) {
                final boolean last = m >= 32;
                final int base = o + (last ? 160 : 0);
                final int stride = last ? 16 : 32;
                for (int h = 0; h < 2; h++) {
                    final IntVector v = (IntVector) ByteVector.fromArray(SimdI8.B8, b, qs + m + 8 * h)
                            .convertShape(VectorOperators.B2I, SimdI8.I8, 0);
                    final int y = base + m - (last ? 32 : 0) + 8 * h;
                    for (int l = 0; l < POW3.length; l++) {
                        final IntVector trit = v.mul(POW3[l]).and(0xFF).mul(3)
                                .lanewise(VectorOperators.LSHR, 8);
                        acc = acc.add(trit.mul(IntVector.fromArray(SimdI8.I8, yw, y + stride * l)));
                    }
                }
            }
            int qhSum = 0;
            final int qh = off + TQ1_0_QH;
            for (int j = 0; j < TQ1_0_QH_BYTES; ++j) {
                final int v = b[qh + j] & 0xFF;
                final int y0 = o + TQ1_0_QS_BYTES * 5 + j;
                qhSum += trit(v, 1) * yw[y0] + trit(v, 3) * yw[y0 + TQ1_0_QH_BYTES]
                        + trit(v, 9) * yw[y0 + 2 * TQ1_0_QH_BYTES]
                        + trit(v, 27) * yw[y0 + 3 * TQ1_0_QH_BYTES];
            }
            final int sum = SimdI8.sum(acc) + qhSum - blockSum(bsums, i);
            sumf += (float) sum * (fp16At(b, off + TQ1_0_D) * yd[i]);
        }
        return sumf;
    }

    /** The scalar trit of a TQ1_0 byte, matching {@link TernaryVecDot}. */
    private static int trit(int v, int pow3) {
        return (((v * pow3) & 0xFF) * 3) >>> 8;
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

    @Override
    public String toString() {
        return "SimdTq1_0{TQ1_0 x Q8_K}";
    }
}
