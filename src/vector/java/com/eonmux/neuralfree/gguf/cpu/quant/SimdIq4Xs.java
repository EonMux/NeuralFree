package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.QuantTables;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_H;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.IQ4_XS_SCALES_L;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.shortAt;

/**
 * Vector API vec-dot of IQ4_XS weights with Q8_K activations, {@link IQuantVecDot#IQ4_XS} with the
 * {@code kvalues_iq4nl} codebook as a vector lookup, the same one that serves {@link SimdIq4Nl}: the sixteen
 * quant bytes of a 32-element sub-block give sixteen low nibbles (elements {@code 0..15}) and sixteen shifted
 * high nibbles (elements {@code 16..31}), each selects its lane out of the table with {@code selectFrom}, and
 * the selected int8 values dot the pre-widened short activation lanes, the two products adding in short lanes
 * before one pairwise fold into int lanes. The sub-block's signed 6-bit scale (four low bits from
 * {@code scales_l}, two more from {@code scales_h}) scales the block sum in float exactly as
 * {@code ggml_vec_dot_iq4_xs_q8_K_generic} does, so the kernel returns what {@link IQuantVecDot#IQ4_XS} returns.
 *
 * <p>Keeping the codebook lookup and the activation lanes of a sub-block in a method of their own is what makes
 * the lookup intrinsic: written inside the super-block loop, the same source fell back to the allocating
 * template. The batched product ({@link #dotTiles}) looks each weight row up once per task into float values and
 * multiplies four rows by two eight-token tiles in exact float lanes ({@link SimdBlockTiles}, form
 * {@link SimdBlockTiles.Form#SUB_SCALED}). Immutable, stateless and thread-safe.</p>
 */
public final class SimdIq4Xs implements SimdBlockTiles.Kernel {

    /** The IQ4_XS vec-dot. */
    public static final SimdIq4Xs INSTANCE = new SimdIq4Xs();

    /** Elements of one sub-block, i.e. of the activation run one lookup dots. */
    private static final int SUB = 32;

    /** {@code kvalues_iq4nl} as sixteen byte lanes: the source of {@code selectFrom}. */
    private static final ByteVector TABLE = table();

    private SimdIq4Xs() {
    }

    /** The codebook as a 16-lane byte vector. */
    private static ByteVector table() {
        final byte[] t = new byte[16];
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
        Q8VecDot.requireWeightType(w, GGMLType.IQ4_XS);
        x.requireCompatible(w, GGMLType.Q8_K);
    }

    @Override
    public GGMLType weightType() {
        return GGMLType.IQ4_XS;
    }

    @Override
    public SimdBlockTiles.Form form() {
        return SimdBlockTiles.Form.SUB_SCALED;
    }

    /**
     * The codebook values of one row as floats, each 32-element sub-block's {@code ls - 32} in {@code d} and its
     * super-block's {@code d_x} in {@code m} ({@link SimdBlockTiles.Rows}).
     */
    @Override
    public void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m) {
        final byte[] q = SimdBlockTiles.quants();
        int o = 0;
        for (int ib = 0; ib < nb; o += IQ4_XS_BYTES) {
            final float dx = fp16At(b, o + IQ4_XS_D);
            int h = shortAt(b, o + IQ4_XS_SCALES_H) & 0xFFFF;
            for (int sub = 0; sub < QK_K / SUB; sub += 2, ib += 2) {
                final int sl = b[o + IQ4_XS_SCALES_L + sub / 2] & 0xFF;
                d[ib] = ((sl & 0x0F) | ((h << 4) & 0x30)) - 32;
                d[ib + 1] = ((sl >>> 4) | ((h << 2) & 0x30)) - 32;
                m[ib] = dx;
                m[ib + 1] = dx;
                h >>>= 4;
                SimdIq4Nl.lookup(b, o + IQ4_XS_QS + 16 * sub, q);
                SimdBlockTiles.widen(q, values, ib * SUB);
                SimdIq4Nl.lookup(b, o + IQ4_XS_QS + 16 * sub + 16, q);
                SimdBlockTiles.widen(q, values, (ib + 1) * SUB);
            }
        }
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        for (int ibl = 0; ibl < nb; ibl++, off += IQ4_XS_BYTES) {
            final float d4d8 = fp16At(b, off + IQ4_XS_D) * yd[ibl];
            final int h = shortAt(b, off + IQ4_XS_SCALES_H) & 0xFFFF;
            final int o = ibl * QK_K;
            for (int sub = 0; sub < QK_K / SUB; sub++) {
                final int sl = b[off + IQ4_XS_SCALES_L + sub / 2] & 0xFF;
                final int ls = ((sub & 1) == 0 ? sl & 0x0F : sl >>> 4) | (((h >>> (2 * sub)) & 3) << 4);
                sumf += (d4d8 * (ls - 32)) * subDot(b, off + IQ4_XS_QS + 16 * sub, ys, o + sub * SUB);
            }
        }
        return sumf;
    }

    /**
     * The 32 codebook values of one sub-block's sixteen quant bytes, dotted with the activation lanes at
     * {@code y}: the lookup and the two 16-lane products of {@link SimdIq4Nl} in a method small enough for the
     * JIT to keep the {@code selectFrom} intrinsic. The row loop calls it once per iteration, as {@code SimdIq4Nl}
     * does: two calls per iteration let C2 inline both copies into the row loop, and in a share of the JVMs that ran
     * other vector code first the copies' shape changes then compile to calls into the library's Java fallback.
     */
    private static int subDot(byte[] b, int qs, short[] ys, int y) {
        final ByteVector q = ByteVector.fromArray(SimdI8.B16, b, qs);
        final ShortVector lo = (ShortVector) q.and((byte) 0x0F).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final ShortVector hi = (ShortVector) q.lanewise(VectorOperators.LSHR, 4).selectFrom(TABLE)
                .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
        final IntVector p0 = (IntVector) lo.mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16))).reinterpretShape(SimdI8.I8, 0);
        return p0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                .add(p0.lanewise(VectorOperators.ASHR, 16)).reduceLanes(VectorOperators.ADD);
    }

    @Override
    public String toString() {
        return "SimdIq4Xs{IQ4_XS x Q8_K}";
    }
}
