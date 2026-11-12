package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_M;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_1_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_1;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q4_1 weights with Q8_1 activations, {@link LegacyVecDot#Q4_1} with the unsigned
 * nibbles in int16 lanes: the 16 quant bytes widen to 16 short lanes, the low nibbles dot elements
 * {@code 0..15} and the shifted high nibbles elements {@code 16..31}, the two product vectors add in short
 * lanes and fold pairwise into int lanes (the sign-extended low and high halves of each int lane). A nibble
 * times an int8 activation fits a short, and so does the sum of two of them. The float combination
 * {@code (d_x * d_y) * sumi + m_x * s_y} per block follows {@code ggml_vec_dot_q4_1_q8_1_generic}, so the
 * kernel returns exactly what {@link LegacyVecDot#Q4_1} returns.
 *
 * <p>The batched product ({@link #dotTiles}) unpacks each weight row once per task into float values and
 * multiplies four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#MINS}). Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_1 implements SimdBlockTiles.Kernel {

    /** The Q4_1 vec-dot. */
    public static final SimdQ4_1 INSTANCE = new SimdQ4_1();

    private SimdQ4_1() {
    }

    @Override
    public float dotRow(CpuTensorRef w, long row, QuantizedActivations x) {
        check(w, x);
        return dot(CpuVecDotRows.copyRow(w, row), 0, x, SimdI8.activationsShort(x));
    }

    @Override
    public float dotRowBytes(byte[] row, int off, QuantizedActivations x) {
        return dot(row, off, x, SimdI8.activationsShort(x));
    }

    @Override
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations x, float[] y, int yOff) {
        check(w, x);
        CpuVecDotRows.forEachRowBytes(w, rowStart, rowCount, this, x, y, yOff);
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q4_1);
        x.requireCompatible(w, GGMLType.Q8_1);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.Q4_1;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.MINS;
    }

    /**
     * The quants of one row as float values {@code 0..15} and its block scales and mins
     * ({@link SimdBlockTiles.Rows}).
     */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q4_1_BYTES) {
            LegacyVecDot.unpackBlock(GGMLType.Q4_1, b, o, q);
            SimdBlockTiles.widen(q, values, ib * QK4_1);
            d[ib] = fp16At(b, o + Q4_1_D);
            m[ib] = fp16At(b, o + Q4_1_M);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final float[] ya = x.s;
        final int nb = x.n / QK4_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q4_1_BYTES) {
            final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, off + Q4_1_QS);
            final ShortVector w = (ShortVector) q.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final ShortVector lo = w.and((short) 0x0F);
            final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F);
            final int y = ib * QK4_1;
            final IntVector p3 = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))).reinterpretShape(SimdI8.I8, 0);
            final int sumi = p3.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                    .add(p3.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
            sumf += (fp16At(b, off + Q4_1_D) * yd[ib]) * sumi + fp16At(b, off + Q4_1_M) * ya[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ4_1{Q4_1 x Q8_1}";
    }
}
