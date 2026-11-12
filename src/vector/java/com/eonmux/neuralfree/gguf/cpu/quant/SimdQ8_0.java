package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_QS;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of Q8_0 weights with Q8_0 activations, {@link Q8VecDot} with the 32-element int8 block
 * dots in int16 lanes: the two 16-byte halves of a block widen to short lanes and multiply the pre-widened
 * activation halves, and the two product vectors add in short lanes before one pairwise fold into int lanes
 * — every int8 product and every sum of two of them fits a short. The float
 * combination {@code sumi * (d_x * d_y)} per block follows {@code ggml_vec_dot_q8_0_q8_0_generic}, so the
 * kernel returns exactly what {@code Q8VecDot} returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of four weight rows by eight tokens in
 * exact float lanes ({@link SimdBlockTiles}, form {@link SimdBlockTiles.Form#DOT}), so a decoded weight block and an
 * activation block are each reused across the tile. Immutable, stateless and thread-safe.</p>
 *
 * <p>{@link #FAST} is the opt-in fast-order form of the single-token dots ({@code CpuSimd.PROP_ORDER}): each block's
 * eight int lane sums convert to float lanes and multiply-add with the block's scale product into one float
 * accumulator, and its eight lanes reduce pairwise once per row, as {@code ggml_vec_dot_q8_0_q8_0}'s AVX2 path
 * accumulates; one accumulator suffices, since the block's integer work, not the multiply-add chain, bounds the
 * loop. That drops the per-block horizontal reduction and the serial float chain of the exact order; the result
 * differs from the scalar kernel by float rounding only (every block's integer sum is still exact). Its batched
 * product is the exact one.</p>
 */
public final class SimdQ8_0 implements SimdBlockTiles.Kernel {

    /** The Q8_0 vec-dot. */
    public static final SimdQ8_0 INSTANCE = new SimdQ8_0(false);

    /** The Q8_0 vec-dot whose single-token dots sum in the fast float order. */
    public static final SimdQ8_0 FAST = new SimdQ8_0(true);

    private final boolean fastOrder;

    private SimdQ8_0(boolean fastOrder) {
        this.fastOrder = fastOrder;
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

    /** The row dot in this instance's order. */
    private float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        return fastOrder ? dotFast(b, off, x, ys) : dotExact(b, off, x, ys);
    }

    /** The single-token row loop with the widened activations and the row reader resolved once per call. */
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
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.Q8_0;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.DOT;
    }

    /** The quants of one row as float values and its block scales ({@link SimdBlockTiles.Rows}). */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q8_0_BYTES) {
            for (int k = 0; k < QK8_0; k += 8) {
                ((FloatVector) ByteVector.fromArray(SimdI8.B8, b, o + Q8_0_QS + k)
                        .convertShape(VectorOperators.B2F, SimdTokenTiles.F8, 0)).intoArray(values, ib * QK8_0 + k);
            }
            d[ib] = fp16At(b, o + Q8_0_D);
        }
    }

    private static float dotExact(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK8_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q8_0_BYTES) {
            // the two 16-byte halves of the block widen to 16 short lanes each and multiply the matching
            // pre-widened activation halves in vpmullw; every int8 product fits a short and so does the sum of
            // two of them, and the sign-extended low and high halves of each int lane are one pair of them
            final int y = ib * QK8_0;
            final ShortVector p = ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0)).mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(((ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS + 16)
                            .convertShape(VectorOperators.B2S, SimdI8.S16, 0))
                            .mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16)));
            final IntVector lanes = (IntVector) p.reinterpretShape(SimdI8.I8, 0);
            final int sumi = lanes.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                    .add(lanes.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
            sumf += sumi * (fp16At(b, off + Q8_0_D) * yd[ib]);
        }
        return sumf;
    }

    /**
     * The fast-order row dot ({@link #FAST}): block {@code ib}'s int lane sums, exact as in {@link #dotExact}, convert
     * to float lanes and multiply-add {@code d_x * d_y} into the accumulator, whose eight lanes reduce pairwise.
     */
    private static float dotFast(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK8_0;
        FloatVector acc = FloatVector.zero(SimdTokenTiles.F8);
        for (int ib = 0; ib < nb; ib++, off += Q8_0_BYTES) {
            final int y = ib * QK8_0;
            final ShortVector p = ((ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS)
                    .convertShape(VectorOperators.B2S, SimdI8.S16, 0)).mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(((ShortVector) ByteVector.fromArray(SimdI8.B16, b, off + Q8_0_QS + 16)
                            .convertShape(VectorOperators.B2S, SimdI8.S16, 0))
                            .mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16)));
            final IntVector l = (IntVector) p.reinterpretShape(SimdI8.I8, 0);
            final FloatVector s = (FloatVector) l.lanewise(VectorOperators.LSHL, 16)
                    .lanewise(VectorOperators.ASHR, 16).add(l.lanewise(VectorOperators.ASHR, 16))
                    .convertShape(VectorOperators.I2F, SimdTokenTiles.F8, 0);
            final FloatVector c = FloatVector.broadcast(SimdTokenTiles.F8, fp16At(b, off + Q8_0_D) * yd[ib]);
            acc = SimdTokenTiles.HARDWARE_FMA ? s.fma(c, acc) : s.mul(c).add(acc);
        }
        final FloatVector v = (FloatVector) acc.check(SimdTokenTiles.F8);
        return ((v.lane(0) + v.lane(1)) + (v.lane(2) + v.lane(3)))
                + ((v.lane(4) + v.lane(5)) + (v.lane(6) + v.lane(7)));
    }

    @Override
    public String toString() {
        return fastOrder ? "SimdQ8_0{Q8_0 x Q8_0, fast order}" : "SimdQ8_0{Q8_0 x Q8_0}";
    }
}
