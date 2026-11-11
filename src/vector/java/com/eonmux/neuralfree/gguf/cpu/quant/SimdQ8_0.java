package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
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
 * ({@link SimdI8#pairSum}) — every int8 product and every sum of two of them fits a short. The float
 * combination {@code sumi * (d_x * d_y)} per block follows {@code ggml_vec_dot_q8_0_q8_0_generic}, so the
 * kernel returns exactly what {@code Q8VecDot} returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of four weight rows by eight tokens in
 * exact float lanes ({@link SimdTokenTiles}), so a decoded weight block and an activation block are each reused
 * across the tile. Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ8_0 implements CpuVecDot {

    /** The Q8_0 vec-dot. */
    public static final SimdQ8_0 INSTANCE = new SimdQ8_0();

    private SimdQ8_0() {
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
    public void dotRows(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                        float[] y, int yOff, int yStride) {
        dotTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
    }

    @Override
    public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, QuantizedActivations[] x, int nTokens,
                         float[] y, int yOff, int yStride) {
        if (nTokens == 0) {
            return;
        }
        checkTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
        dotTiles(w, rowStart, rowCount, SimdTokenTiles.layout(x, nTokens), 0, nTokens, y, yOff, yStride);
    }

    @Override
    public TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
        SimdTokenTiles.checkTokens(x, nTokens, GGMLType.Q8_0);
        return nTokens == 0 ? null : SimdTokenTiles.layout(x, nTokens);
    }

    @Override
    public int tileTokens() {
        return SimdTokenTiles.TILE_TOKENS;
    }

    /** The four rows of the float register tile. */
    @Override
    public int tileRows() {
        return SimdTokenTiles.TILE_ROWS;
    }

    /** The two eight-token tiles of the float register tile. */
    @Override
    public int tilePanelTokens() {
        return SimdTokenTiles.PAIR * SimdTokenTiles.TILE_TOKENS;
    }

    /** One float lane per laid-out activation element. */
    @Override
    public int tileLaneBytes() {
        return Float.BYTES;
    }

    @Override
    public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                         QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (!(tiles instanceof SimdTokenTiles.Layout)) {
            CpuVecDot.super.dotTiles(w, rowStart, rowCount, tiles, x, tokenStart, nTokens, y, yOff, yStride);
            return;
        }
        final SimdTokenTiles.Layout tt = (SimdTokenTiles.Layout) tiles;
        SimdTokenTiles.checkRange(tt, tokenStart, nTokens);
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        SimdTokenTiles.checkWidth(w, tt);
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        dotTiles(w, rowStart, rowCount, tt, tokenStart, nTokens, y, yOff, yStride);
    }

    /**
     * The batched product over the token tiles of {@code tiles} ({@link SimdTokenTiles.Layout}): each weight row
     * unpacks once into float values, and per 32-element block four rows by one tile form their exact block sums
     * in float lanes ({@link SimdTokenTiles#sums}), one lane per token. The float combination {@code sum +=
     * sumi * (d * yd)} then runs lane-wise in block order, which is the scalar kernel's arithmetic.
     */
    private static void dotTiles(CpuTensorRef w, long rowStart, int rowCount, SimdTokenTiles.Layout tiles,
                                 int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / QK8_0;
        final float[][] values = SimdTokenTiles.rowValues(tiles.cols);
        final float[][] scales = SimdI8.rowScales(SimdTokenTiles.TILE_ROWS, nb);
        final float[] out = SimdI8.tileAccumulators(SimdTokenTiles.TILE_ROWS * SimdTokenTiles.TILE_TOKENS);
        final float[] seg = SimdTokenTiles.segmentSums(tiles.cols, QK8_0);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdTokenTiles.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdTokenTiles.TILE_TOKENS;
        for (int r = 0; r < rowCount; r += SimdTokenTiles.TILE_ROWS) {
            final int nr = Math.min(SimdTokenTiles.TILE_ROWS, rowCount - r);
            for (int i = 0; i < SimdTokenTiles.TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                unpackRow(buf, nb, values[i], scales[i]);
            }
            for (int tile = firstTile; tile <= lastTile; ) {
                final int pair = SimdTokenTiles.sums(values, QK8_0, nb, tiles, tile, lastTile, seg);
                for (int k = 0; k < pair; k++, tile++) {
                    combine(scales, tiles, tile, k, nb, seg, out);
                    final int t0 = tile * SimdTokenTiles.TILE_TOKENS - tokenStart;
                    SimdTokenTiles.store(out, nr, Math.min(SimdTokenTiles.TILE_TOKENS, nTokens - t0), y, yOff, r,
                            t0, yStride);
                }
            }
        }
    }

    /** The quants of one row as float values and its block scales. */
    private static void unpackRow(byte[] b, int nb, float[] values, float[] d) {
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q8_0_BYTES) {
            for (int k = 0; k < QK8_0; k += 8) {
                ((FloatVector) ByteVector.fromArray(SimdI8.B8, b, o + Q8_0_QS + k)
                        .convertShape(VectorOperators.B2F, SimdTokenTiles.F8, 0)).intoArray(values, ib * QK8_0 + k);
            }
            d[ib] = fp16At(b, o + Q8_0_D);
        }
    }

    /**
     * The float combination of tile {@code tile}, member {@code k} of the pair whose block sums {@code seg} holds:
     * {@code out[r * TILE_TOKENS + ti]} receives row {@code r}'s dot with the tile's token {@code ti}.
     */
    private static void combine(float[][] d, SimdTokenTiles.Layout tiles, int tile, int k, int nb, float[] seg,
                                float[] out) {
        final int yBase = tiles.scaleBase(tile);
        final float[] d0 = d[0];
        final float[] d1 = d[1];
        final float[] d2 = d[2];
        final float[] d3 = d[3];
        FloatVector f0 = FloatVector.zero(SimdTokenTiles.F8);
        FloatVector f1 = f0;
        FloatVector f2 = f0;
        FloatVector f3 = f0;
        for (int ib = 0; ib < nb; ib++) {
            final FloatVector yd = FloatVector.fromArray(SimdTokenTiles.F8, tiles.scales,
                    yBase + ib * SimdTokenTiles.TILE_TOKENS);
            f0 = f0.add(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, 0))
                    .mul(yd.mul(d0[ib])));
            f1 = f1.add(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, 1))
                    .mul(yd.mul(d1[ib])));
            f2 = f2.add(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, 2))
                    .mul(yd.mul(d2[ib])));
            f3 = f3.add(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, 3))
                    .mul(yd.mul(d3[ib])));
        }
        f0.intoArray(out, 0);
        f1.intoArray(out, SimdTokenTiles.TILE_TOKENS);
        f2.intoArray(out, 2 * SimdTokenTiles.TILE_TOKENS);
        f3.intoArray(out, 3 * SimdTokenTiles.TILE_TOKENS);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK8_0;
        float sumf = 0;
        for (int ib = 0; ib < nb; ib++, off += Q8_0_BYTES) {
            // the two 16-byte halves of the block widen to 16 short lanes each and multiply the matching
            // pre-widened activation halves in vpmullw; every int8 product fits a short and so does the sum of
            // two of them, so the block dot is the int-lane one
            final int y = ib * QK8_0;
            final int sumi = SimdI8.sum(SimdI8.pairs(widen(b, off + Q8_0_QS).mul(ShortVector.fromArray(SimdI8.S16, ys, y))
                    .add(widen(b, off + Q8_0_QS + 16).mul(ShortVector.fromArray(SimdI8.S16, ys, y + 16)))));
            sumf += sumi * (fp16At(b, off + Q8_0_D) * yd[ib]);
        }
        return sumf;
    }

    private static ShortVector widen(byte[] b, int off) {
        return (ShortVector) ByteVector.fromArray(SimdI8.B16, b, off).convertShape(VectorOperators.B2S, SimdI8.S16, 0);
    }

    @Override
    public String toString() {
        return "SimdQ8_0{Q8_0 x Q8_0}";
    }
}
