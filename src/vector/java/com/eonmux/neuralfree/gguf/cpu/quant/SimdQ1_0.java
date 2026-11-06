package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q1_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK1_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.intAt;

/**
 * Vector API vec-dot of Q1_0 weights with Q8_0 activations, {@link NewFpVecDot#Q1_0} with the sign bits of a
 * 32-bit word applied to the pre-widened activation lanes: one int lane per element, the multiplier
 * {@code 2*bit - 1} built by a lane-wise shift of the word. Every lane product and every 32-element sum is
 * exact in int, and the float combination follows {@code ggml_vec_dot_q1_0_q8_0_generic}, so the kernel
 * returns what {@link NewFpVecDot#Q1_0} returns. Immutable, stateless and thread-safe.
 */
public final class SimdQ1_0 implements CpuVecDot {

    /** The Q1_0 vec-dot. */
    public static final SimdQ1_0 INSTANCE = new SimdQ1_0();

    private SimdQ1_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q1_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final int nb = x.n / QK1_0;
        final int q8PerBlock = QK1_0 / QK8_0;
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++, off += Q1_0_BYTES) {
            final float d0 = fp16At(b, off + Q1_0_D);
            float sumi = 0.0f;
            for (int k = 0; k < q8PerBlock; k++) {
                final int q8Block = i * q8PerBlock + k;
                final int o = q8Block * QK8_0;
                final int bits = intAt(b, off + Q1_0_QS + 4 * k);
                final IntVector word = IntVector.broadcast(SimdI8.I8, bits);
                IntVector acc = IntVector.zero(SimdI8.I8);
                for (int h = 0; h < QK8_0 / 8; h++) {
                    // bit e of the word selects +q8 or -q8: the multiplier is 2*bit - 1
                    final IntVector sign = word.lanewise(VectorOperators.LSHR, SimdI8.RAMP8.add(8 * h))
                            .and(1).mul(2).sub(1);
                    acc = acc.add(IntVector.fromArray(SimdI8.I8, yw, o + 8 * h).mul(sign));
                }
                sumi += yd[q8Block] * SimdI8.sum(acc);
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ1_0{Q1_0 x Q8_0}";
    }
}
