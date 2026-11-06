package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import java.nio.ByteBuffer;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK5_0;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.intAt;

/**
 * Vector API vec-dot of Q5_0 weights with Q8_0 activations, {@link LegacyVecDot#Q5_0} with the nibbles and the
 * fifth bits merged in int lanes: byte {@code j} of a block holds element {@code j} as its low nibble and
 * element {@code j + 16} as its high one, and bit {@code j} of the 32-bit {@code qh} word supplies the fifth
 * bit, so each lane forms {@code (nibble | bit << 4) - 16}. The float combination
 * {@code (d_x * d_y) * sumi} per block follows {@code ggml_vec_dot_q5_0_q8_0_generic}, so the kernel returns
 * exactly what {@link LegacyVecDot#Q5_0} returns. Immutable, stateless and thread-safe.
 */
public final class SimdQ5_0 implements CpuVecDot {

    /** The Q5_0 vec-dot. */
    public static final SimdQ5_0 INSTANCE = new SimdQ5_0();

    private SimdQ5_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q5_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final int nb = x.n / QK5_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_0_BYTES) {
            final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, off + Q5_0_QS);
            final IntVector qh = IntVector.broadcast(SimdI8.I8, intAt(b, off + Q5_0_QH));
            final int y = ib * QK5_0;
            final IntVector w0 = SimdI8.widen(q, 0, SimdI8.I8);
            final IntVector w1 = SimdI8.widen(q, 1, SimdI8.I8);
            final IntVector lo0 = w0.and(0x0F).or(SimdI8.fifthBits(qh, 0)).sub(16);
            final IntVector lo1 = w1.and(0x0F).or(SimdI8.fifthBits(qh, 1)).sub(16);
            final IntVector hi0 = w0.lanewise(VectorOperators.LSHR, 4).and(0x0F)
                    .or(SimdI8.fifthBits(qh, 2)).sub(16);
            final IntVector hi1 = w1.lanewise(VectorOperators.LSHR, 4).and(0x0F)
                    .or(SimdI8.fifthBits(qh, 3)).sub(16);
            final IntVector acc = lo0.mul(IntVector.fromArray(SimdI8.I8, yw, y))
                    .add(lo1.mul(IntVector.fromArray(SimdI8.I8, yw, y + 8)))
                    .add(hi0.mul(IntVector.fromArray(SimdI8.I8, yw, y + 16)))
                    .add(hi1.mul(IntVector.fromArray(SimdI8.I8, yw, y + 24)));
            final int sumi = SimdI8.sum(acc);
            sumf += (fp16At(b, off + Q5_0_D) * yd[ib]) * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ5_0{Q5_0 x Q8_0}";
    }
}
