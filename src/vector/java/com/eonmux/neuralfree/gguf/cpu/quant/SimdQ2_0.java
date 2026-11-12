package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q2_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK2_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q2_0 weights with Q8_0 activations, {@link NewFpVecDot#Q2_0} with the two-bit codes of
 * each 16-bit window shifted into int lanes: the code {@code c} becomes the multiplier {@code c - 1}, and every
 * lane product and 32-element sum is exact in int. The float combination follows
 * {@code ggml_vec_dot_q2_0_q8_0_generic}, so the kernel returns what {@link NewFpVecDot#Q2_0} returns.
 * Immutable, stateless and thread-safe.
 */
public final class SimdQ2_0 implements CpuVecDot {

    /** The Q2_0 vec-dot. */
    public static final SimdQ2_0 INSTANCE = new SimdQ2_0();

    /** Shift of every int lane to its two-bit code: {@code [0, 2, ..., 14]}. */
    private static final IntVector SHIFTS = SimdI8.RAMP8.mul(2);

    private SimdQ2_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q2_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, int[] yw) {
        final float[] yd = x.d;
        final int nb = x.n / QK2_0;
        final int q8PerBlock = QK2_0 / QK8_0;
        float sumf = 0.0f;
        for (int i = 0; i < nb; i++, off += Q2_0_BYTES) {
            final float d0 = fp16At(b, off + Q2_0_D);
            float sumi = 0.0f;
            for (int k = 0; k < q8PerBlock; k++) {
                final int q8Block = i * q8PerBlock + k;
                final int o = q8Block * QK8_0;
                final long codes = longAt(b, off + Q2_0_QS + 8 * k);
                IntVector acc = IntVector.zero(SimdI8.I8);
                for (int h = 0; h < QK8_0 / 8; h++) {
                    // the 16-bit window holds eight 2-bit codes; code c maps to the multiplier c - 1
                    final int chunk = (int) (codes >>> (16 * h)) & 0xFFFF;
                    final IntVector code = IntVector.broadcast(SimdI8.I8, chunk)
                            .lanewise(VectorOperators.LSHR, SHIFTS).and(3).sub(1);
                    acc = IntVector.fromArray(SimdI8.I8, yw, o + 8 * h).mul(code).add(acc);
                }
                sumi += yd[q8Block] * ((IntVector) acc.check(SimdI8.I8)).reduceLanes(VectorOperators.ADD);
            }
            sumf += d0 * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ2_0{Q2_0 x Q8_0}";
    }
}
