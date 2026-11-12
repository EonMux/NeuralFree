package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_NL_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK4_NL;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;

/**
 * Vector API vec-dot of IQ4_NL weights with Q8_0 activations, {@link IQuantVecDot#IQ4_NL} with the codebook as a
 * vector lookup: the 16 quant bytes give sixteen low nibbles (elements {@code 0..15}) and sixteen shifted high
 * nibbles (elements {@code 16..31}), each selects its lane out of the {@code kvalues_iq4nl} table with
 * {@code selectFrom}, and the selected int8 values dot the pre-widened short activation lanes in
 * {@code vpmullw}, the two product vectors adding in short lanes before one pairwise fold into int lanes
 * (the sign-extended halves of each int lane). Two int8 codebook values sum without wrapping a short, and so do
 * their products with an int8 activation. The float combination {@code d_x * d_y * sumi} per block follows
 * {@code ggml_vec_dot_iq4_nl_q8_0_generic}, so the kernel returns exactly what {@link IQuantVecDot#IQ4_NL}
 * returns.
 *
 * <p>The batched product ({@link #dotTiles}) looks each weight row up once per task into float values and
 * multiplies four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#DOT}). Immutable, stateless and thread-safe.</p>
 */
public final class SimdIq4Nl implements SimdBlockTiles.Kernel {

    /** The IQ4_NL vec-dot. */
    public static final SimdIq4Nl INSTANCE = new SimdIq4Nl();

    /** {@code kvalues_iq4nl} as sixteen byte lanes: the source of {@code selectFrom}. */
    private static final ByteVector TABLE = table();

    private SimdIq4Nl() {
    }

    /** The codebook as a 16-lane byte vector. */
    private static ByteVector table() {
        final byte[] t = new byte[QK4_NL / 2];
        for (int i = 0; i < t.length; i++) {
            t[i] = (byte) QuantTables.kvalueIq4nl(i);
        }
        return ByteVector.fromArray(SimdI8.B16, t, 0);
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
        Q8VecDot.requireWeightType(w, GGMLType.IQ4_NL);
        x.requireCompatible(w, GGMLType.Q8_0);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.IQ4_NL;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.DOT;
    }

    /** The codebook values of one row as floats and its block scales ({@link SimdBlockTiles.Rows}). */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += IQ4_NL_BYTES) {
            lookup(b, o + IQ4_NL_QS, q);
            SimdBlockTiles.widen(q, values, ib * QK4_NL);
            d[ib] = fp16At(b, o + IQ4_NL_D);
        }
    }

    /** The 32 codebook values of the sixteen quant bytes at {@code b[qs..)} in element order at {@code q[0..32)}. */
    static void lookup(byte[] b, int qs, byte[] q) {
        final ByteVector v = ByteVector.fromArray(SimdI8.B16, b, qs);
        v.and((byte) 0x0F).selectFrom(TABLE).intoArray(q, 0);
        v.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE).intoArray(q, QK4_NL / 2);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK4_NL;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += IQ4_NL_BYTES) {
            final int y = ib * QK4_NL;
            final int sumi = subDot(b, off + IQ4_NL_QS, ys, y);
            sumf += (fp16At(b, off + IQ4_NL_D) * yd[ib]) * sumi;
        }
        return sumf;
    }

    /** The 32 codebook values of the sixteen quant bytes dot the thirty-two activation lanes at {@code y}. */
    private static int subDot(byte[] b, int qs, short[] ys, int y) {
        final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, qs);
        final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final IntVector p = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + QK4_NL / 2))).reinterpretShape(SimdI8.I8, 0);
        return p.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                .add(p.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public String toString() {
        return "SimdIq4Nl{IQ4_NL x Q8_0}";
    }
}
