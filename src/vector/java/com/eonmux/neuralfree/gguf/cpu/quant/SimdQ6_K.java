package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QH;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_QL;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q6_K_SCALES;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.longAt;

/**
 * Vector API vec-dot of Q6_K weights with Q8_K activations, {@link KQuantVecDot#Q6_K} with the unpack in byte
 * lanes and the scaled 16-element sub-block dots in int16 lanes: each 16-byte slice of {@code ql} loads with
 * its 16 {@code qh} bytes, quarter {@code q} of slice {@code t} in half {@code c} decoding to sub-block
 * {@code 8c + 2q + t} ({@code (ql quarter | qh pair << 4) - 32}). A sub-block's decodes widen to short lanes,
 * multiply the pre-widened activation lanes in {@code vpmullw}, fold pairwise into int lanes and are scaled by
 * their signed 8-bit scale there, so one int accumulator carries the 16 sub-blocks of a super-block and the
 * horizontal sum runs once per token instead of once per sub-block. The float combination
 * {@code sumf += (d_x * d_y) * isum} follows {@code ggml_vec_dot_q6_K_q8_K_generic}, so the kernel returns what
 * {@code KQuantVecDot.Q6_K} returns.
 *
 * <p>The batched product ({@link #dotTiles}) unpacks each weight row once and multiplies it with interleaved tiles
 * of eight tokens, one int lane per token.
 * Immutable, stateless and thread-safe.</p>
 */
public final class SimdQ6_K implements CpuVecDot {

    /** The Q6_K vec-dot. */
    public static final SimdQ6_K INSTANCE = new SimdQ6_K();

    /** Weight rows one register tile of the batched kernel multiplies. */
    private static final int TILE_ROWS = 2;

    /** The per-row sub-block scales of the batched kernel. */
    private static final ThreadLocal<int[][]> ROW_SCALES = new ThreadLocal<>();

    /** The two low bits of every byte. */
    private static final long LOW_PAIRS = 0x0303030303030303L;
    /** {@code 0x80 - 32} in every byte, for {@link Q8VecDot#subtractBias}. */
    private static final long BIAS_32 = 0x6060606060606060L;

    private SimdQ6_K() {
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
        Q8VecDot.requireWeightType(w, GGMLType.Q6_K);
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
        Q8VecDot.requireWeightType(w, GGMLType.Q6_K);
        SimdI8.checkTileWidth(w, tt);
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        dotTiles(w, rowStart, rowCount, tt, tokenStart, nTokens, y, yOff, yStride);
    }

    /**
     * The batched product over interleaved token tiles ({@link SimdI8.TokenTiles}): each weight row unpacks once
     * into int16 element pairs with its signed sub-block scales, and per 16-element sub-block the products of a pair
     * broadcast over a tile accumulate in short lanes, four pairs at a time (four products of a 6-bit quant and an
     * int8 quant fit a short, eight may not), fold into int lanes and join the super-block sum scaled by the
     * sub-block scale. The float combination {@code sumf += (d * yd) * isum} runs lane-wise, one lane per token,
     * which is {@code KQuantVecDot.Q6_K}'s arithmetic.
     */
    private static void dotTiles(CpuTensorRef w, long rowStart, int rowCount, SimdI8.TokenTiles tiles,
                                 int tokenStart, int nTokens, float[] y, int yOff, int yStride) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / QK_K;
        final int[][] pairs = SimdI8.rowPairs(TILE_ROWS, tiles.cols / 2);
        final int[][] scales = SimdI8.rowInts(ROW_SCALES, TILE_ROWS, nb * (QK_K / 16));
        final float[][] d = SimdI8.rowScales(TILE_ROWS, nb);
        final float[] out = SimdI8.tileAccumulators(TILE_ROWS * SimdI8.TILE_TOKENS);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdI8.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdI8.TILE_TOKENS;
        for (int r = 0; r < rowCount; r += TILE_ROWS) {
            final int nr = Math.min(TILE_ROWS, rowCount - r);
            for (int i = 0; i < TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                unpackRow(buf, nb, pairs[i], scales[i], d[i]);
            }
            for (int tile = firstTile; tile <= lastTile; tile++) {
                dotTile(pairs[0], scales[0], d[0], pairs[1], scales[1], d[1], tiles, tile, nb, out);
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
     * One row's signed quants as int16 element pairs ({@code pairs[128i + k]} packs elements {@code 2k, 2k+1} of
     * super-block {@code i}), its sixteen signed sub-block scales per super-block and its {@code d}.
     */
    private static void unpackRow(byte[] b, int nb, int[] pairs, int[] scales, float[] d) {
        final byte[] q = Q8VecDot.scratch().quants;
        int o = 0;
        for (int i = 0; i < nb; i++, o += Q6_K_BYTES) {
            unpack(b, o, q);
            for (int k = 0; k < QK_K; k += 16) {
                ((ShortVector) ByteVector.fromArray(SimdI8.B16, q, k).convertShape(VectorOperators.B2S, SimdI8.S16, 0))
                        .reinterpretAsInts().intoArray(pairs, i * (QK_K / 2) + k / 2);
            }
            for (int g = 0; g < QK_K / 16; g++) {
                scales[(QK_K / 16) * i + g] = b[o + Q6_K_SCALES + g];
            }
            d[i] = fp16At(b, o + Q6_K_D);
        }
    }

    /**
     * Two unpacked weight rows against token tile {@code tile}: {@code out[i * TILE_TOKENS + ti]} receives row
     * {@code i}'s dot with the tile's token {@code ti}.
     */
    private static void dotTile(int[] w0, int[] sc0, float[] d0, int[] w1, int[] sc1, float[] d1,
                                SimdI8.TokenTiles tiles, int tile, int nb, float[] out) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        final int yBase = tiles.scaleBase(tile);
        FloatVector f0 = FloatVector.zero(SimdFp.F8);
        FloatVector f1 = FloatVector.zero(SimdFp.F8);
        for (int i = 0; i < nb; i++) {
            IntVector is0 = SimdI8.zero(SimdI8.I8);
            IntVector is1 = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final int wb = i * (QK_K / 2) + 8 * g;
                final int ab = aBase + (i * QK_K + 16 * g) * SimdI8.TILE_TOKENS;
                IntVector g0 = SimdI8.zero(SimdI8.I8);
                IntVector g1 = SimdI8.zero(SimdI8.I8);
                for (int h = 0; h < 8; h += 4) {
                    ShortVector p0 = ShortVector.zero(SimdI8.S16);
                    ShortVector p1 = ShortVector.zero(SimdI8.S16);
                    for (int k = h; k < h + 4; k++) {
                        final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                        p0 = p0.add(SimdI8.broadcastPair(w0[wb + k]).mul(v));
                        p1 = p1.add(SimdI8.broadcastPair(w1[wb + k]).mul(v));
                    }
                    g0 = g0.add(SimdI8.pairs(p0));
                    g1 = g1.add(SimdI8.pairs(p1));
                }
                is0 = is0.add(g0.mul(sc0[(QK_K / 16) * i + g]));
                is1 = is1.add(g1.mul(sc1[(QK_K / 16) * i + g]));
            }
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, tiles.scales, yBase + i * SimdI8.TILE_TOKENS);
            f0 = f0.add(yd.mul(d0[i]).mul(toFloat(is0)));
            f1 = f1.add(yd.mul(d1[i]).mul(toFloat(is1)));
        }
        f0.intoArray(out, 0);
        f1.intoArray(out, SimdI8.TILE_TOKENS);
    }

    private static FloatVector toFloat(IntVector v) {
        return (FloatVector) v.convert(VectorOperators.I2F, 0);
    }

    private static float dot(byte[] b, int off, QuantizedActivations x, short[] ys) {
        final float[] yd = x.d;
        final int nb = x.n / QK_K;
        final byte[] q = Q8VecDot.scratch().quants;
        float sumf = 0;
        for (int i = 0; i < nb; i++, off += Q6_K_BYTES) {
            unpack(b, off, q);
            final int sc = off + Q6_K_SCALES;
            final int xo = i * QK_K;
            IntVector acc = SimdI8.zero(SimdI8.I8);
            for (int g = 0; g < QK_K / 16; g++) {
                final ShortVector v = (ShortVector) ByteVector.fromArray(SimdI8.B16, q, 16 * g)
                        .convertShape(VectorOperators.B2S, SimdI8.S16, 0);
                final int yo = xo + 16 * g;
                acc = acc.add(SimdI8.pairSum(v.mul(ShortVector.fromArray(SimdI8.S16, ys, yo)),
                        SimdI8.zero(SimdI8.I8)).mul(b[sc + g]));
            }
            sumf += fp16At(b, off + Q6_K_D) * yd[i] * SimdI8.sum(acc);
        }
        return sumf;
    }

    /**
     * Unpacks the 256 signed 6-bit quants of one super-block into {@code q}: quarter {@code q} of slice
     * {@code t} in half {@code c} is sub-block {@code 8c + 2q + t}, its value {@code (ql nibble | qh pair << 4)
     * - 32}. Eight quants at a time with the 64-bit SWAR of {@link Q8VecDot#subtractBias}: the byte-lane form
     * of the same unpack measured slower and allocated, so the SWAR form stays.
     */
    private static void unpack(byte[] b, int off, byte[] q) {
        final int ql = off + Q6_K_QL;
        final int qh = off + Q6_K_QH;
        for (int c = 0; c < QK_K / 128; c++) {
            final int a = 128 * c;
            for (int t = 0; t < 4; t++) {
                final long lo = longAt(b, ql + 64 * c + 8 * t);
                final long hi = longAt(b, ql + 64 * c + 32 + 8 * t);
                final long h = longAt(b, qh + 32 * c + 8 * t);
                Q8VecDot.putLong(q, a + 8 * t,
                        Q8VecDot.subtractBias((lo & Q8VecDot.LOW_NIBBLES) | ((h & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 32 + 8 * t,
                        Q8VecDot.subtractBias((hi & Q8VecDot.LOW_NIBBLES) | (((h >>> 2) & LOW_PAIRS) << 4), BIAS_32));
                Q8VecDot.putLong(q, a + 64 + 8 * t,
                        Q8VecDot.subtractBias(((lo >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 4) & LOW_PAIRS) << 4),
                                BIAS_32));
                Q8VecDot.putLong(q, a + 96 + 8 * t,
                        Q8VecDot.subtractBias(((hi >>> 4) & Q8VecDot.LOW_NIBBLES) | (((h >>> 6) & LOW_PAIRS) << 4),
                                BIAS_32));
            }
        }
    }

    @Override
    public String toString() {
        return "SimdQ6_K{Q6_K x Q8_K}";
    }
}
