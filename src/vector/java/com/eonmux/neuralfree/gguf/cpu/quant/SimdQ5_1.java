package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

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
 * nibbles and fifth bits merged in int16 lanes (byte {@code j} holds elements {@code j} and {@code j + 16}; bit
 * {@code j} of {@code qh} is the fifth bit of element {@code j}), multiplied by the pre-widened short activations
 * in {@code vpmullw} lanes and folded pairwise into int lanes, as in {@link SimdQ4_1}. The float combination
 * {@code (d_x * d_y) * sumi + m_x * s_y} per block follows {@code ggml_vec_dot_q5_1_q8_1_generic}, so the
 * kernel returns exactly what {@link LegacyVecDot#Q5_1} returns.
 * The batched product ({@link #dotTiles}) unpacks each weight row once per task into float values and multiplies
 * four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#MINS}). Immutable, stateless and thread-safe.
 */
public final class SimdQ5_1 implements SimdBlockTiles.Kernel {

    /** The Q5_1 vec-dot. */
    public static final SimdQ5_1 INSTANCE = new SimdQ5_1();

    /** Lane {@code j} holds {@code 2^(15 - j)}: the factor that moves bit {@code j} of a short into its sign bit. */
    private static final ShortVector FIFTH_BIT_SHIFT = ShortVector.fromArray(SimdI8.S16, new short[]{
            (short) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
            0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001}, 0);

    private SimdQ5_1() {
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
        CpuVecDotRows.check(w, rowStart, rowCount, y, yOff);
        final short[] ys = SimdI8.activationsShort(x);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, buf);
            y[yOff + r] = dot(buf, 0, x, ys);
        }
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q5_1);
        x.requireCompatible(w, GGMLType.Q8_1);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.Q5_1;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.MINS;
    }

    /**
     * The quants of one row as float values {@code 0..31} and its block scales and mins
     * ({@link SimdBlockTiles.Rows}).
     */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q5_1_BYTES) {
            LegacyVecDot.unpackBlock(GGMLType.Q5_1, b, o, q);
            SimdBlockTiles.widen(q, values, ib * QK5_1);
            d[ib] = fp16At(b, o + Q5_1_D);
            m[ib] = fp16At(b, o + Q5_1_M);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] yw) {
        final float[] yd = x.d;
        final float[] ys = x.s;
        final int nb = x.n / QK5_1;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_1_BYTES) {
            // the 16 quant bytes widen to 16 short lanes: byte j holds element j as its low nibble and element
            // j + 16 as its high one. Bit j of a 16-bit half of qh, multiplied by 2^(15 - j) (FIFTH_BIT_SHIFT),
            // lands in the sign bit, so an arithmetic shift by 15 and a mask of 16 give the fifth bit in place.
            // A 5-bit quant times an int8 activation fits a short and so does the sum of two of them, so the
            // products run in vpmullw lanes and fold pairwise into int lanes
            final ShortVector w = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q5_1_QS)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final int qh = intAt(b, off + Q5_1_QH);
            final ShortVector lo = w.and((short) 0x0F).or(ShortVector.broadcast(SimdI8.S16, (short) qh)
                    .mul(FIFTH_BIT_SHIFT).lanewise(VectorOperators.ASHR, 15).and((short) 16));
            final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
                    .or(ShortVector.broadcast(SimdI8.S16, (short) (qh >>> 16))
                            .mul(FIFTH_BIT_SHIFT).lanewise(VectorOperators.ASHR, 15).and((short) 16));
            final int y = ib * QK5_1;
            final IntVector p = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, yw, y))
                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, yw, y + 16))).reinterpretShape(SimdI8.I8, 0);
            final int sumi = p.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                    .add(p.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
            sumf += (fp16At(b, off + Q5_1_D) * yd[ib]) * sumi + fp16At(b, off + Q5_1_M) * ys[ib];
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ5_1{Q5_1 x Q8_1}";
    }
}
