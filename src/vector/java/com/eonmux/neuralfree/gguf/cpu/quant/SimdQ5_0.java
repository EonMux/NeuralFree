package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q5_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK5_0;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.intAt;

/**
 * Vector API vec-dot of Q5_0 weights with Q8_0 activations, {@link LegacyVecDot#Q5_0} with the nibbles and the
 * fifth bits merged in int16 lanes: byte {@code j} of a block holds element {@code j} as its low nibble and
 * element {@code j + 16} as its high one, and bit {@code j} of the 32-bit {@code qh} word supplies the fifth
 * bit, so each lane forms {@code (nibble | bit << 4) - 16}; the products with the pre-widened short activations
 * run in {@code vpmullw} lanes and fold pairwise into int lanes, as in {@link SimdQ4_0}. The float combination
 * {@code (d_x * d_y) * sumi} per block follows {@code ggml_vec_dot_q5_0_q8_0_generic}, so the kernel returns
 * exactly what {@link LegacyVecDot#Q5_0} returns.
 * The batched product ({@link #dotTiles}) unpacks each weight row once per task into float values and multiplies
 * four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#DOT}). Immutable, stateless and thread-safe.
 */
public final class SimdQ5_0 implements SimdBlockTiles.Kernel {

    /** The Q5_0 vec-dot. */
    public static final SimdQ5_0 INSTANCE = new SimdQ5_0();

    /** Lane {@code j} holds {@code 2^(15 - j)}: the factor that moves bit {@code j} of a short into its sign bit. */
    private static final ShortVector FIFTH_BIT_SHIFT = ShortVector.fromArray(SimdI8.S16, new short[]{
            (short) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
            0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001}, 0);

    private SimdQ5_0() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q5_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.Q5_0;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.DOT;
    }

    /**
     * The quants of one row as float values {@code -16..15} and its block scales
     * ({@link SimdBlockTiles.Rows}).
     */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q5_0_BYTES) {
            LegacyVecDot.unpackBlock(GGMLType.Q5_0, b, o, q);
            SimdBlockTiles.widen(q, values, ib * QK5_0);
            d[ib] = fp16At(b, o + Q5_0_D);
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] yw) {
        final float[] yd = x.d;
        final int nb = x.n / QK5_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q5_0_BYTES) {
            // the 16 quant bytes widen to 16 short lanes: byte j holds element j as its low nibble and element
            // j + 16 as its high one. Bit j of a 16-bit half of qh, multiplied by 2^(15 - j) (FIFTH_BIT_SHIFT),
            // lands in the sign bit, so an arithmetic shift by 15 and a mask of 16 give the fifth bit in place.
            // A 5-bit quant times an int8 activation fits a short and so does the sum of two of them, so the
            // products run in vpmullw lanes and fold pairwise into int lanes
            final ShortVector w = (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q5_0_QS)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
            final int qh = intAt(b, off + Q5_0_QH);
            final ShortVector lo = w.and((short) 0x0F).or(ShortVector.broadcast(SimdI8.S16, (short) qh)
                    .mul(FIFTH_BIT_SHIFT).lanewise(VectorOperators.ASHR, 15).and((short) 16)).sub((short) 16);
            final ShortVector hi = w.lanewise(VectorOperators.LSHR, 4).and((short) 0x0F)
                    .or(ShortVector.broadcast(SimdI8.S16, (short) (qh >>> 16))
                            .mul(FIFTH_BIT_SHIFT).lanewise(VectorOperators.ASHR, 15).and((short) 16)).sub((short) 16);
            final int y = ib * QK5_0;
            final IntVector p = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, yw, y))
                    .add(hi.mul(ShortVector.fromArray(SimdI8.S16, yw, y + 16))).reinterpretShape(SimdI8.I8, 0);
            final int sumi = p.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                    .add(p.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
            sumf += (fp16At(b, off + Q5_0_D) * yd[ib]) * sumi;
        }
        return sumf;
    }

    @Override
    public String toString() {
        return "SimdQ5_0{Q5_0 x Q8_0}";
    }
}
