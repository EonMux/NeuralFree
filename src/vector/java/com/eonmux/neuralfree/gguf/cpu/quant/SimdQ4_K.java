package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_DMIN;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q4_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q4_K weights with Q8_K activations, {@link KQuantVecDot#Q4_K} with the nibble unpack in
 * byte lanes and the sub-block dots in int16 lanes: the 256 quants of a super-block unpack into the thread's
 * scratch row, each 32-element sub-block dots the matching activation sub-block (two 16-lane products added in
 * short lanes, folded pairwise into int lanes) and its sum joins one int accumulator scaled by the 6-bit
 * sub-block scale, so the horizontal sum runs once per token instead of once per sub-block. The scale unpack,
 * the mins sum over {@code bsums} and the float combination ({@code sums += d * isum}, {@code sumf -= dmin *
 * sumi}, result {@code sumf + sums}) follow {@code ggml_vec_dot_q4_K_q8_K_generic} exactly, so the kernel
 * returns what {@code KQuantVecDot.Q4_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) unpacks each weight row once and multiplies it with interleaved tiles
 * of eight tokens, one int lane per token.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ4_K implements CpuVecDot {

    /** The Q4_K vec-dot. */
    public static final SimdQ4_K INSTANCE = new SimdQ4_K();

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;

    /** The per-row sub-block scales of the batched kernel. */
    private static final ThreadLocal<int[][]> ROW_SCALES = new ThreadLocal<>();

    /** The per-row sub-block mins of the batched kernel. */
    private static final ThreadLocal<int[][]> ROW_MINS = new ThreadLocal<>();

    /** Number of {@code bsums} entries per Q8_K block (one per 16 quants). */
    private static final int BSUMS_PER_BLOCK = QK_K / 16;

    private SimdQ4_K() {
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
        final short[] pairs = SimdI8.bsumPairs(x);
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader rows = CpuVecDotRows.reader(w);
        for (int r = 0; r < rowCount; r++) {
            rows.copy(rowStart + r, buf);
            y[yOff + r] = dotRow(buf, 0, x, ys, pairs, sm, q);
        }
    }

    private static void check(CpuTensorRef w, QuantizedActivations x) {
        Q8VecDot.requireWeightType(w, GGMLType.Q4_K);
        x.requireCompatible(w, GGMLType.Q8_K);
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
        SimdI8.checkTileTokens(x, nTokens, GGMLType.Q8_K);
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
        Q8VecDot.requireWeightType(w, GGMLType.Q4_K);
        SimdI8.checkTileWidth(w, tt);
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        dotTiles(w, rowStart, rowCount, tt, tokenStart, nTokens, y, yOff, yStride);
    }

    /**
     * The batched product over interleaved token tiles ({@link SimdI8.TokenTiles}): each weight row unpacks once
     * into int16 element pairs with its sub-block scales and mins, and per 32-element sub-block the products of a
     * pair broadcast over a tile accumulate in short lanes (sixteen products of a 4-bit quant and an int8 quant fit
     * a short), fold once into int lanes and join the super-block sum scaled by the sub-block scale; the mins term
     * dots the tile's group pairs with the mins in int lanes. The float combination ({@code sums += (d * yd) *
     * isum}, {@code sumf -= (dmin * yd) * msum}, result {@code sumf + sums}) runs lane-wise, one lane per token,
     * which is {@code KQuantVecDot.Q4_K}'s arithmetic.
     */
    private static void dotTiles(CpuTensorRef w, long rowStart, int rowCount, SimdI8.TokenTiles tiles,
                                 int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / QK_K;
        final int[][] pairs = SimdI8.rowPairs(TILE_ROWS, tiles.cols / 2);
        final int[][] scales = SimdI8.rowInts(ROW_SCALES, TILE_ROWS, nb * 8);
        final int[][] mins = SimdI8.rowInts(ROW_MINS, TILE_ROWS, nb * 8);
        final float[][] d = SimdI8.rowScales(TILE_ROWS, 2 * nb);
        final float[] out = SimdI8.tileAccumulators(TILE_ROWS * SimdI8.TILE_TOKENS);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdI8.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdI8.TILE_TOKENS;
        for (int r = 0; r < rowCount; r += TILE_ROWS) {
            final int nr = Math.min(TILE_ROWS, rowCount - r);
            for (int i = 0; i < TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                unpackRow(buf, nb, pairs[i], scales[i], mins[i], d[i]);
            }
            for (int tile = firstTile; tile <= lastTile; tile++) {
                dotTile(pairs[0], scales[0], mins[0], d[0], pairs[1], scales[1], mins[1], d[1], tiles, tile, nb,
                        out);
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

    /**
     * One row's quants as int16 element pairs ({@code pairs[128i + k]} packs elements {@code 2k, 2k+1} of
     * super-block {@code i}), its 6-bit sub-block scales and mins, and {@code d} / {@code dmin} at {@code d[2i]} /
     * {@code d[2i + 1]}.
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d) {
        final byte[] sm = Q8VecDot.scratch().scales;
        final byte[] q = Q8VecDot.scratch().quants;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q4_K_BYTES) {
            unpack(b, o, q);
            KQuantVecDot.unpackScalesMins(b, o + Q4_K_SCALES, sm);
            for (int k = 0; k < QK_K; k += 16) {
                ((ShortVector) ByteVector.fromArray(SimdI8.B16, q, k).convertShape(VectorOperators.B2S, SimdI8.S16, 0))
                        .reinterpretAsInts().intoArray(pairs, i * (QK_K / 2) + k / 2);
            }
            for (int g = 0; g < 8; g++) {
                scales[8 * i + g] = sm[g];
                mins[8 * i + g] = sm[8 + g];
            }
            d[2 * i] = fp16At(b, o + Q4_K_D);
            d[2 * i + 1] = fp16At(b, o + Q4_K_DMIN);
        }
    }

    /**
     * Two unpacked weight rows against token tile {@code tile}: {@code out[i * TILE_TOKENS + ti]} receives row
     * {@code i}'s dot with the tile's token {@code ti}. The exact int sums of every super-block go through
     * {@code sums} first, so the integer and the float halves compile as two methods of moderate size.
     */
    private static void dotTile(int[] w0, int[] sc0, int[] m0, float[] d0, int[] w1, int[] sc1, int[] m1,
                                float[] d1, SimdI8.TokenTiles tiles, int tile, int nb, float[] out) {
        final int[] sums = SimdI8.tileSums(4 * nb * SimdI8.TILE_TOKENS);
        intSums(w0, sc0, m0, w1, sc1, m1, tiles, tile, nb, sums);
        floatSums(d0, d1, tiles, tile, nb, sums, out);
    }

    /**
     * Per super-block {@code i}, the scaled quant sums and the mins sums of rows 0 and 1, one int lane per token,
     * at {@code sums[(4i + k) * TILE_TOKENS]} for {@code k} = row 0 sum, row 0 mins, row 1 sum, row 1 mins.
     */
    private static void intSums(int[] w0, int[] sc0, int[] m0, int[] w1, int[] sc1, int[] m1,
                                SimdI8.TokenTiles tiles, int tile, int nb, int[] sums) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        final int pBase = tiles.pairBase(tile);
        for (int i = 0; i < nb; i++) {
            IntVector is0 = SimdI8.zero(SimdI8.I8);
            IntVector is1 = SimdI8.zero(SimdI8.I8);
            IntVector ms0 = SimdI8.zero(SimdI8.I8);
            IntVector ms1 = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < 8; g++) {
                final int wb = i * (QK_K / 2) + 16 * g;
                final int ab = aBase + (i * QK_K + 32 * g) * SimdI8.TILE_TOKENS;
                ShortVector p0 = ShortVector.zero(SimdI8.S16);
                ShortVector p1 = ShortVector.zero(SimdI8.S16);
                for (int k = 0; k < 16; k++) {
                    final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                    p0 = p0.add(SimdI8.broadcastPair(w0[wb + k]).mul(v));
                    p1 = p1.add(SimdI8.broadcastPair(w1[wb + k]).mul(v));
                }
                is0 = is0.add(SimdI8.pairs(p0).mul(sc0[8 * i + g]));
                is1 = is1.add(SimdI8.pairs(p1).mul(sc1[8 * i + g]));
                final IntVector gp = IntVector.fromArray(SimdI8.I8, tiles.groupPairs,
                        pBase + (8 * i + g) * SimdI8.TILE_TOKENS);
                ms0 = ms0.add(gp.mul(m0[8 * i + g]));
                ms1 = ms1.add(gp.mul(m1[8 * i + g]));
            }
            final int o = 4 * i * SimdI8.TILE_TOKENS;
            is0.intoArray(sums, o);
            ms0.intoArray(sums, o + SimdI8.TILE_TOKENS);
            is1.intoArray(sums, o + 2 * SimdI8.TILE_TOKENS);
            ms1.intoArray(sums, o + 3 * SimdI8.TILE_TOKENS);
        }
    }

    /** The float combination of {@link #intSums}' sums, super-block by super-block, lane-wise per token. */
    private static void floatSums(float[] d0, float[] d1, SimdI8.TokenTiles tiles, int tile, int nb, int[] sums,
                                  float[] out) {
        final int yBase = tiles.scaleBase(tile);
        FloatVector fs0 = FloatVector.zero(SimdFp.F8);
        FloatVector fm0 = FloatVector.zero(SimdFp.F8);
        FloatVector fs1 = FloatVector.zero(SimdFp.F8);
        FloatVector fm1 = FloatVector.zero(SimdFp.F8);
        for (int i = 0; i < nb; i++) {
            final int o = 4 * i * SimdI8.TILE_TOKENS;
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, tiles.scales, yBase + i * SimdI8.TILE_TOKENS);
            fs0 = fs0.add(yd.mul(d0[2 * i]).mul(toFloat(IntVector.fromArray(SimdI8.I8, sums, o))));
            fm0 = fm0.sub(yd.mul(d0[2 * i + 1])
                    .mul(toFloat(IntVector.fromArray(SimdI8.I8, sums, o + SimdI8.TILE_TOKENS))));
            fs1 = fs1.add(yd.mul(d1[2 * i])
                    .mul(toFloat(IntVector.fromArray(SimdI8.I8, sums, o + 2 * SimdI8.TILE_TOKENS))));
            fm1 = fm1.sub(yd.mul(d1[2 * i + 1])
                    .mul(toFloat(IntVector.fromArray(SimdI8.I8, sums, o + 3 * SimdI8.TILE_TOKENS))));
        }
        fm0.add(fs0).intoArray(out, 0);
        fm1.add(fs1).intoArray(out, SimdI8.TILE_TOKENS);
    }

    private static FloatVector toFloat(IntVector v) {
        return (FloatVector) v.convert(VectorOperators.I2F, 0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        return dotRow(b, off, x, ys, SimdI8.bsumPairs(x), Q8VecDot.scratch().scales, Q8VecDot.scratch().quants);
    }

    /**
     * One row against {@code x} with its short lanes {@code ys} and group-sum pairs {@code pairs}; {@code sm} and
     * {@code q} are the caller's scale and quant scratch.
     */
    private static float dotRow(byte[] b, int off, QuantizedActivations x, short[] ys, short[] pairs, byte[] sm,
                                byte[] q) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        float sumf = 0;
        float sums = 0;
        for (int i = 0; i < nb; i++, off += Q4_K_BYTES) {
            unpack(b, off, q);
            KQuantVecDot.unpackScalesMins(b, off + Q4_K_SCALES, sm);
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 32; g++) {
                final ShortVector lo = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final ShortVector hi = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 32 * g + 16)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 32 * g;
                acc = acc.add(SimdI8.pairSum(lo.mul(ShortVector.fromArray(SimdI8.S16, ys, yo))
                        .add(hi.mul(ShortVector.fromArray(SimdI8.S16, ys, yo + 16))),
                        SimdI8.zero(SimdI8.I8)).mul(sm[g]));
            }
            final float d = fp16At(b, off + Q4_K_D) * yd[i];
            sums += d * SimdI8.sum(acc);
            final float dmin = fp16At(b, off + Q4_K_DMIN) * yd[i];
            sumf -= dmin * SimdI8.minsDot(pairs, i * (QK_K / 32), sm);
        }
        return sumf + sums;
    }

    /**
     * Unpacks the 256 four-bit quants of one super-block into {@code q}: group {@code j} of 64 elements takes the
     * low nibbles of its 32 quant bytes as elements {@code 64j .. 64j + 31} and their high nibbles as
     * {@code 64j + 32 .. 64j + 63}, which is the order {@code dequantize_row_q4_K} lays them out in.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        for (int j = 0; j < QK_K / 64; j++) {
            final int q4 = off + Q4_K_QS + 32 * j;
            for (int t = 0; t < 4; t++) {
                final long v = longAt(b, q4 + 8 * t);
                Q8VecDot.putLong(q, 64 * j + 8 * t, v & Q8VecDot.LOW_NIBBLES);
                Q8VecDot.putLong(q, 64 * j + 32 + 8 * t, (v >>> 4) & Q8VecDot.LOW_NIBBLES);
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ4_K{Q4_K x Q8_K}";
    }
}
