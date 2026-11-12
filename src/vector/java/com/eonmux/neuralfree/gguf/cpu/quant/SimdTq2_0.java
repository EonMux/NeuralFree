package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.TQ2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of TQ2_0 weights with Q8_K activations, {@link TernaryVecDot#TQ2_0} with the four two-bit
 * codes of each byte decoded in int lanes: byte {@code j + k} of a 32-byte chunk holds the codes of
 * {@code q8[4j + 32l + k]} at bits {@code 2l}, so the 32 bytes load as two 16-byte vectors and each of the
 * four bit pairs multiplies its activation run. The activation block sum is subtracted from the exact integer
 * sum exactly as {@code ggml_vec_dot_tq2_0_q8_K_generic} does, so the kernel returns what
 * {@link TernaryVecDot#TQ2_0} returns. Immutable, stateless and thread-safe.
 */
public final class SimdTq2_0 implements CpuVecDot {

    /** The TQ2_0 vec-dot. */
    public static final SimdTq2_0 INSTANCE = new SimdTq2_0();

    /** Number of 16-element {@code bsums} groups per Q8_K block. */
    private static final int BSUM_GROUPS = QK_K / 16;

    private SimdTq2_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.TQ2_0);
        x.requireCompatible(w, GGMLType.Q8_K);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final short[] bsums = x.bsums;
        final int qsBytes = QK_K / 4;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int i = 0, o = 0; i < nb; ++i, off += TQ2_0_BYTES, o += QK_K) {
            final int qs = off + TQ2_0_QS;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int j = 0; j < qsBytes; j += 32) {
                final int y0 = o + j * 4;
                for (int h = 0; h < 2; h++) {
                    final ByteVector raw = ByteVector.fromArray(SimdI8.B16, b, qs + j + 16 * h);
                    final IntVector v = (IntVector) raw.convertShape(VectorOperators.B2I, SimdI8.I8, 0);
                    final IntVector v2 = (IntVector) raw.convertShape(VectorOperators.B2I, SimdI8.I8, 1);
                    for (int l = 0; l < 4; l++) {
                        final int at = y0 + 32 * l + 16 * h;
                        acc = v.lanewise(VectorOperators.LSHR, 2 * l).and(3)
                                .mul(IntVector.fromArray(SimdI8.I8, yw, at))
                                .add(v2.lanewise(VectorOperators.LSHR, 2 * l).and(3)
                                        .mul(IntVector.fromArray(SimdI8.I8, yw, at + 8)))
                                .add(acc);
                    }
                }
            }
            final int sumi = ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD) - blockSum(bsums, i);
            final float d = yd[i] * fp16At(b, off + TQ2_0_D);
            sumf += (float) sumi * d;
        }
        return sumf;
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
        return "SimdTq2_0{TQ2_0 x Q8_K}";
    }
}
