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
 * ({@link SimdI8#pairSum}) — every int8 product and every sum of two of them fits a short. The float
 * combination {@code sumi * (d_x * d_y)} per block follows {@code ggml_vec_dot_q8_0_q8_0_generic}, so the
 * kernel returns exactly what {@code Q8VecDot} returns.
 *
 * <p>The batched product ({@link #dotTiles}) multiplies a register tile of two weight rows by eight tokens,
 * so a decoded weight block and a widened activation block are each reused across the tile. Immutable,
 * stateless and thread-safe.</p>
 */
public final class SimdQ8_0 implements CpuVecDot {

    /** The Q8_0 vec-dot. */
    public static final SimdQ8_0 INSTANCE = new SimdQ8_0();

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;


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
        dotTiles(w, rowStart, rowCount, SimdI8.tokenTiles(x, nTokens), 0, nTokens, y, yOff, yStride);
    }

    @Override
    public TileActivations prepareTiles(QuantizedActivations[] x, int nTokens) {
        SimdI8.checkTileTokens(x, nTokens, GGMLType.Q8_0);
        return nTokens == 0 ? null : SimdI8.tokenTiles(x, nTokens);
    }

    @Override
    public int tileTokens() {
        return SimdI8.TILE_TOKENS;
    }

    @Override
    public void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                         QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (!(tiles instanceof SimdI8.TokenTiles)) {
            CpuVecDot.super.dotTiles(w, rowStart, rowCount, tiles, x, tokenStart, nTokens, y, yOff, yStride);
            return;
        }
        final SimdI8.TokenTiles tt = (SimdI8.TokenTiles) tiles;
        SimdI8.checkTileRange(tt, tokenStart, nTokens);
        Q8VecDot.requireWeightType(w, GGMLType.Q8_0);
        SimdI8.checkTileWidth(w, tt);
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        dotTiles(w, rowStart, rowCount, tt, tokenStart, nTokens, y, yOff, yStride);
    }

    /**
     * The batched product over the interleaved token tiles of {@code tiles} ({@link SimdI8.TokenTiles}): each
     * weight row widens once into int16 element pairs, and per block a pair broadcast over a tile multiplies the
     * tile's interleaved pair lanes, so every int lane accumulates one token's exact block sum with no horizontal
     * reduction. The float combination {@code sum += sumi * (d * yd)} then runs lane-wise, one lane per token, in
     * block order, which is the scalar kernel's arithmetic.
     */
    private static void dotTiles(CpuTensorRef w, long rowStart, int rowCount, SimdI8.TokenTiles tiles,
                                 int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / QK8_0;
        final int[][] pairs = SimdI8.rowPairs(TILE_ROWS, tiles.cols / 2);
        final float[][] scales = SimdI8.rowScales(TILE_ROWS, nb);
        final float[] out = SimdI8.tileAccumulators(TILE_ROWS * SimdI8.TILE_TOKENS);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdI8.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdI8.TILE_TOKENS;
        for (int r = 0; r < rowCount; r += TILE_ROWS) {
            final int nr = Math.min(TILE_ROWS, rowCount - r);
            for (int i = 0; i < TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                widenRow(buf, nb, pairs[i], scales[i]);
            }
            for (int tile = firstTile; tile <= lastTile; tile++) {
                dotTile(pairs[0], pairs[1], scales[0], scales[1], tiles, tile, nb, out);
                final int t0 = tile * SimdI8.TILE_TOKENS - tokenStart;
                final int n = Math.min(SimdI8.TILE_TOKENS, nTokens - t0);
                for (int i = 0; i < nr; i++) {
                    for (int ti = 0; ti < n; ti++) {
                        y[yOff + r + i + (t0 + ti) * yStride] = out[i * SimdI8.TILE_TOKENS + ti];
                    }
                }
            }
        }
    }

    /** The quants of one row as int16 element pairs ({@code pairs[ib*16 + k]} packs elements 2k, 2k+1) and its scales. */
    private static void widenRow(byte[] b, int nb, int[] pairs, float[] d) {
        int o = 0;
        for (int ib = 0; ib < nb; ib++, o += Q8_0_BYTES) {
            widen(b, o + Q8_0_QS).reinterpretAsInts().intoArray(pairs, ib * 16);
            widen(b, o + Q8_0_QS + 16).reinterpretAsInts().intoArray(pairs, ib * 16 + 8);
            d[ib] = fp16At(b, o + Q8_0_D);
        }
    }

    /**
     * Two widened weight rows against token tile {@code tile}: {@code out[i * TILE_TOKENS + ti]} receives row
     * {@code i}'s dot with the tile's token {@code ti}.
     */
    private static void dotTile(int[] w0, int[] w1, float[] d0, float[] d1, SimdI8.TokenTiles tiles, int tile,
                                int nb, float[] out) {
        final short[] a = tiles.lanes;
        final float[] ydT = tiles.scales;
        final int aBase = tiles.laneBase(tile);
        final int yBase = tiles.scaleBase(tile);
        FloatVector f0 = FloatVector.zero(SimdFp.F8);
        FloatVector f1 = FloatVector.zero(SimdFp.F8);
        for (int ib = 0; ib < nb; ib++) {
            IntVector s0 = SimdI8.zero(SimdI8.I8);
            IntVector s1 = SimdI8.zero(SimdI8.I8);
            final int wb = ib * 16;
            final int ab = aBase + ib * QK8_0 * SimdI8.TILE_TOKENS;
            for (int k = 0; k < 16; k += 2) {
                final ShortVector a0 = ShortVector.fromArray(SimdI8.S16, a, ab + k * 16);
                final ShortVector a1 = ShortVector.fromArray(SimdI8.S16, a, ab + k * 16 + 16);
                s0 = s0.add(SimdI8.pairs(SimdI8.broadcastPair(w0[wb + k]).mul(a0)
                        .add(SimdI8.broadcastPair(w0[wb + k + 1]).mul(a1))));
                s1 = s1.add(SimdI8.pairs(SimdI8.broadcastPair(w1[wb + k]).mul(a0)
                        .add(SimdI8.broadcastPair(w1[wb + k + 1]).mul(a1))));
            }
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, ydT, yBase + ib * SimdI8.TILE_TOKENS);
            f0 = f0.add(toFloat(s0).mul(yd.mul(d0[ib])));
            f1 = f1.add(toFloat(s1).mul(yd.mul(d1[ib])));
        }
        f0.intoArray(out, 0);
        f1.intoArray(out, SimdI8.TILE_TOKENS);
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

    private static FloatVector toFloat(IntVector v) {
        return (FloatVector) v.convert(VectorOperators.I2F, 0);
    }

    @Override
    public String toString() {
        return "SimdQ8_0{Q8_0 x Q8_0}";
    }
}
