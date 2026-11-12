package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q4_0 weights with Q8_0 activations, {@link LegacyVecDot#Q4_0} with the nibble unpack
 * and the int16 product path: the 16 quant bytes widen to 16 short lanes in one instruction, the low nibbles
 * subtract 8 into elements {@code 0..15} and the shifted high nibbles into elements {@code 16..31}, each
 * multiplying the pre-widened short activation lanes in {@code vpmullw}. The two product vectors of a block add
 * in short lanes (a nibble times an int8 activation fits a short, and so does their sum, so nothing wraps) and
 * fold pairwise into int lanes (the sign-extended low and high halves of each int lane). The float
 * combination {@code sumi * d_x * d_y} per block follows {@code ggml_vec_dot_q4_0_q8_0_generic}, so the kernel returns
 * exactly what {@code LegacyVecDot.Q4_0} returns.
 *
 * <p>The batched product ({@link #dotTiles}) unpacks each weight row once per task into float values and
 * multiplies four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#SCALED}). Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_0 implements SimdBlockTiles.Kernel {

    /** The Q4_0 vec-dot. */
    public static final SimdQ4_0 INSTANCE = new SimdQ4_0();

    private SimdQ4_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q4_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.Q4_0;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.SCALED;
    }

    /** The quants of one row as float values {@code -8..7} and its block scales ({@link SimdBlockTiles.Rows}). */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q4_0_BYTES) {
            LegacyVecDot.unpackBlock(GGMLType.Q4_0, b, o, q);
            SimdBlockTiles.widen(q, values, ib * QK4_0);
            d[ib] = fp16At(b, o + Q4_0_D);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK4_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q4_0_BYTES) {
            // the 16 quant bytes widen to 16 short lanes (one instruction): byte j holds element j as its low
            // nibble and element j + 16 as its high one, and a nibble times an int8 activation fits a short,
            // so the products run in vpmullw lanes and fold pairwise into int lanes
            final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, off + Q4_0_QS);
            final ShortVector w = (ShortVector) q.convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final ShortVector lo = w.and((short) 0x0F).sub((short) 8);
            final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F).sub((short) 8);
            final int y = ib * QK4_0;
            final IntVector p = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))).reinterpretShape(SimdI8.I8, 0);
            sumf += p.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                    .add(p.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD)
                    * fp16At(b, off + Q4_0_D) * yd[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ4_0{Q4_0 x Q8_0}";
    }
}
