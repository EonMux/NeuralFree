package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK5_1;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.intAt;

/**
 * Vector API vec-dot of Q5_1 weights with Q8_1 activations, {@link LegacyVecDot#Q5_1} with the unsigned
 * nibbles and fifth bits merged in int lanes (byte {@code j} holds elements {@code j} and {@code j + 16}; bit
 * {@code j} of {@code qh} is the fifth bit of element {@code j}). The float combination
 * {@code (d_x * d_y) * sumi + m_x * s_y} per block follows {@code ggml_vec_dot_q5_1_q8_1_generic}, so the
 * kernel returns exactly what {@link LegacyVecDot#Q5_1} returns. Immutable, stateless and thread-safe.
 */
public final class SimdQ5_1 implements CpuVecDot {

    /** The Q5_1 vec-dot. */
    public static final SimdQ5_1 INSTANCE = new SimdQ5_1();

    private SimdQ5_1() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q5_1);
        x.requireCompatible(w, GGMLType.Q8_1);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final float[] ys = x.s;
        final int nb = x.n / QK5_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_1_BYTES) {
            final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, off + Q5_1_QS);
            final IntVector qh = IntVector.broadcast(SimdI8.I8, intAt(b, off + Q5_1_QH));
            final int y = ib * QK5_1;
            final IntVector w0 = SimdI8.widen(q, 0, SimdI8.I8);
            final IntVector w1 = SimdI8.widen(q, 1, SimdI8.I8);
            final IntVector lo0 = w0.and(0x0F).or(SimdI8.fifthBits(qh, 0));
            final IntVector lo1 = w1.and(0x0F).or(SimdI8.fifthBits(qh, 1));
            final IntVector hi0 = w0.lanewise(VectorOperators.LSHR, 4).and(0x0F).or(SimdI8.fifthBits(qh, 2));
            final IntVector hi1 = w1.lanewise(VectorOperators.LSHR, 4).and(0x0F).or(SimdI8.fifthBits(qh, 3));
            final IntVector acc = lo0.mul(IntVector.fromArray(SimdI8.I8, yw, y))
                    .add(lo1.mul(IntVector.fromArray(SimdI8.I8, yw, y + 8)))
                    .add(hi0.mul(IntVector.fromArray(SimdI8.I8, yw, y + 16)))
                    .add(hi1.mul(IntVector.fromArray(SimdI8.I8, yw, y + 24)));
            final int sumi = SimdI8.sum(acc);
            sumf += (fp16At(b, off + Q5_1_D) * yd[ib]) * sumi + fp16At(b, off + Q5_1_M) * ys[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ5_1{Q5_1 x Q8_1}";
    }
}
