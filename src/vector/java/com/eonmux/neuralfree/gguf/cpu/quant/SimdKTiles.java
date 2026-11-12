package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;

/**
 * The batched (prefill) products of the K-quant kernels over interleaved token tiles ({@link SimdI8.TokenTiles}):
 * each weight row unpacks once per task into int16 element pairs with its sub-block scales, mins and super-block
 * scales ({@link Rows}), and every tile of eight tokens then multiplies {@link #TILE_ROWS} unpacked rows. Per
 * sub-block the products of an element pair broadcast over a tile accumulate in short lanes, fold pairwise into int
 * lanes every {@link Form#fold} pairs (the most products whose sum still fits a short) and join the super-block sum
 * scaled by the sub-block scale. The float half runs the scalar kernel's combination lane-wise, one lane per token,
 * so every form returns the bits of its {@link KQuantVecDot} kernel. The 32-element forms hand the exact int sums of
 * each super-block to their float half through an int array, so each half compiles as a method of moderate size;
 * the 16-element forms run both halves in one method, so their sums stay in registers and never pass through the
 * array.
 * <p>
 * The vector code keeps the rules of {@link SimdI8}: vectors live only inside the methods that create them, and a
 * loop-carried vector is an argument, never a receiver, unless its species was just re-asserted.
 */
final class SimdKTiles {

    /** Weight rows one register tile multiplies. */
    static final int TILE_ROWS = 2;

    /** Int sums per super-block and tile: quant sum and mins sum of each of the two rows. */
    private static final int SUMS_PER_BLOCK = 4 * SimdI8.TILE_TOKENS;

    private static final ThreadLocal<int[][]> ROW_SCALES = new ThreadLocal<>();
    private static final ThreadLocal<int[][]> ROW_MINS = new ThreadLocal<>();

    private SimdKTiles() {
    }

    /** The sub-block structure of a K-quant type and its float combination. */
    enum Form {
        /** Q4_K: 8 sub-blocks of 32 with 6-bit scales and mins; {@code sumf - dmin terms + d terms}. */
        MINS_32(8, 16, 8),
        /** Q5_K: as {@link #MINS_32}, five-bit quants, so the short lanes fold every 8 pairs. */
        MINS_32_FOLD_8(8, 8, 8),
        /**
         * Q6_K, Q3_K and the grid-coded i-quants: 16 sub-blocks of 16 with signed scales;
         * {@code sumf += (d * d_y) * isum}.
         */
        SIGNED_16(16, 4, 0),
        /** Q2_K: 16 sub-blocks of 16 with 4-bit scales and mins; {@code sumf += dall * isum - dmin * summs}. */
        MINS_16(16, 4, 16);

        /** Sub-blocks per super-block. */
        final int subBlocks;
        /**
         * Element pairs whose products accumulate in short lanes before one fold, a literal constant in each
         * form's int method so that C2 unrolls the inner loop completely.
         */
        final int fold;
        /** Activation group sums per super-block the mins term needs: 8 pairs, 16 groups, or none. */
        final int groups;

        Form(int subBlocks, int fold, int groups) {
            this.subBlocks = subBlocks;
            this.fold = fold;
            this.groups = groups;
        }
    }

    /** Unpacks the weight rows of one K-quant type for the token-tile product. */
    @FunctionalInterface
    interface Rows {
        /**
         * Unpacks the {@code nb} super-blocks of the row at {@code b[0..)}: {@code pairs[128i + k]} packs elements
         * {@code 2k, 2k+1} of super-block {@code i} as int16 halves (low half the even element), {@code scales} and
         * {@code mins} hold {@link Form#subBlocks} values per super-block, {@code d[2i]} the super-block scale and
         * {@code d[2i + 1]} its min scale (unused by the signed forms).
         */
        void unpack(byte[] b, int nb, int[] pairs, int[] scales, int[] mins, float[] d);
    }

    /**
     * Packs the 256 unpacked int8 quants {@code q} of one super-block as int16 element pairs at
     * {@code pairs[at..at + 128)}.
     */
    static void packPairs(byte[] q, int[] pairs, int at) {
        for (int k = 0; k < QK_K; k += 16) {
            ((IntVector) ByteVector.fromArray(SimdI8.B16, q, k).convertShape(VectorOperators.B2S, SimdI8.S16, 0)
                    .reinterpretShape(SimdI8.I8, 0)).intoArray(pairs, at + k / 2);
        }
    }

    /** The span of the activation group sums {@code form}'s tiles carry ({@link SimdI8#tokenTiles}). */
    static int groupSpan(Form form) {
        return form.groups == 0 ? 0 : QK_K / form.groups;
    }

    /**
     * {@link #product} after validating a prepared product: the token range lies in the layout and starts on a
     * tile, {@code w} is of {@code type} with the layout's row length, the outputs fit, and the layout carries the
     * group sums {@code form} needs.
     *
     * @throws IllegalArgumentException  for another weight type, row length or group span, or an unaligned start
     * @throws IndexOutOfBoundsException for a token or output range outside its array
     */
    static void checkedProduct(CpuTensorRef w, GGMLType type, long rowStart, int rowCount, SimdI8.TokenTiles tiles,
                               int tokenStart, int nTokens, float[] y, int yOff, int yStride, Form form,
                               Rows unpacker) {
        checkedProduct(w, type, rowStart, rowCount, tiles, tokenStart, nTokens, y, yOff, yStride, form, unpacker, 1f);
    }

    /**
     * {@link #checkedProduct} of a kind whose scalar kernel returns {@code post * sumf} (the i-quants' {@code 0.125f}
     * and {@code 0.25f}): every result is multiplied by {@code post} once, after its super-block sum.
     */
    static void checkedProduct(CpuTensorRef w, GGMLType type, long rowStart, int rowCount, SimdI8.TokenTiles tiles,
                               int tokenStart, int nTokens, float[] y, int yOff, int yStride, Form form,
                               Rows unpacker, float post) {
        SimdI8.checkTileRange(tiles, tokenStart, nTokens);
        Q8VecDot.requireWeightType(w, type);
        SimdI8.checkTileWidth(w, tiles);
        if (form.groups != 0 && tiles.groupSpan != groupSpan(form)) {
            throw new IllegalArgumentException(type + " tiles need group sums over " + groupSpan(form)
                    + " quants, the layout has " + tiles.groupSpan);
        }
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        product(w, rowStart, rowCount, tiles, tokenStart, nTokens, y, yOff, yStride, form, unpacker, post);
    }

    /**
     * Rows {@code [rowStart, rowStart + rowCount)} of {@code w} against tokens {@code [tokenStart, tokenStart +
     * nTokens)} of {@code tiles}: row {@code r} of token {@code t} lands at {@code y[yOff + r + t * yStride]}. The
     * caller has validated the ranges; {@code tokenStart} is a multiple of the tile.
     */
    static void product(CpuTensorRef w, long rowStart, int rowCount, SimdI8.TokenTiles tiles, int tokenStart,
                        int nTokens, float[] y, int yOff, int yStride, Form form, Rows unpacker) {
        product(w, rowStart, rowCount, tiles, tokenStart, nTokens, y, yOff, yStride, form, unpacker, 1f);
    }

    /** {@link #product} whose every result is multiplied by {@code post} once, as its scalar kernel returns it. */
    static void product(CpuTensorRef w, long rowStart, int rowCount, SimdI8.TokenTiles tiles, int tokenStart,
                        int nTokens, float[] y, int yOff, int yStride, Form form, Rows unpacker, float post) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / QK_K;
        final int[][] pairs = SimdI8.rowPairs(TILE_ROWS, tiles.cols / 2);
        final int[][] scales = SimdI8.rowInts(ROW_SCALES, TILE_ROWS, nb * form.subBlocks);
        final int[][] mins = SimdI8.rowInts(ROW_MINS, TILE_ROWS, nb * form.subBlocks);
        final float[][] d = SimdI8.rowScales(TILE_ROWS, 2 * nb);
        final float[] out = SimdI8.tileAccumulators(TILE_ROWS * SimdI8.TILE_TOKENS);
        final int[] sums = SimdI8.tileSums(nb * SUMS_PER_BLOCK);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdI8.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdI8.TILE_TOKENS;
        final boolean scaled = post != 1f;
        for (int r = 0; r < rowCount; r += TILE_ROWS) {
            final int nr = Math.min(TILE_ROWS, rowCount - r);
            for (int i = 0; i < TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                unpacker.unpack(buf, nb, pairs[i], scales[i], mins[i], d[i]);
            }
            for (int tile = firstTile; tile <= lastTile; tile++) {
                tile(form, pairs, scales, mins, d, tiles, tile, nb, sums, out);
                final int t0 = tile * SimdI8.TILE_TOKENS - tokenStart;
                final int n = Math.min(SimdI8.TILE_TOKENS, nTokens - t0);
                for (int i = 0; i < nr; i++) {
                    for (int ti = 0; ti < n; ti++) {
                        final float v = out[i * SimdI8.TILE_TOKENS + ti];
                        y[yOff + r + i + (t0 + ti) * yStride] = scaled ? post * v : v;
                    }
                }
            }
        }
    }

    /** The two unpacked rows against tile {@code tile}: {@code out[i * TILE_TOKENS + ti]} for row i, token ti. */
    private static void tile(Form form, int[][] pairs, int[][] scales, int[][] mins, float[][] d,
                             SimdI8.TokenTiles tiles, int tile, int nb, int[] sums, float[] out) {
        switch (form) {
            case MINS_32:
                subBlocks32(pairs[0], scales[0], pairs[1], scales[1], tiles, tile, nb, sums);
                mins(mins[0], mins[1], tiles, tile, nb, form.groups, sums);
                minusMins(d[0], d[1], tiles, tile, nb, sums, out);
                break;
            case MINS_32_FOLD_8:
                subBlocks32Fold8(pairs[0], scales[0], pairs[1], scales[1], tiles, tile, nb, sums);
                mins(mins[0], mins[1], tiles, tile, nb, form.groups, sums);
                minusMins(d[0], d[1], tiles, tile, nb, sums, out);
                break;
            case SIGNED_16:
                signed16(pairs[0], scales[0], d[0], pairs[1], scales[1], d[1], tiles, tile, nb, out);
                break;
            case MINS_16:
                mins16(pairs[0], scales[0], mins[0], d[0], pairs[1], scales[1], mins[1], d[1], tiles, tile, nb, out);
                break;
            default:
                throw new IllegalStateException(form.name());
        }
    }

    /**
     * Per super-block {@code i}, rows 0 and 1's sums over 8 sub-blocks of 32 of the sub-block scale times the
     * sub-block's quant dot, at {@code sums[(4i + 0) * 8]} and {@code sums[(4i + 2) * 8]}, one int lane per token.
     * Sixteen products of a 4-bit quant and an int8 quant fit a short, so each sub-block folds once.
     */
    private static void subBlocks32(int[] w0, int[] sc0, int[] w1, int[] sc1, SimdI8.TokenTiles tiles, int tile,
                                    int nb, int[] sums) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        for (int i = 0; i < nb; i++) {
            IntVector is0 = IntVector.zero(SimdI8.I8);
            IntVector is1 = is0;
            for (int g = 0; g < 8; g++) {
                final int wb = i * (QK_K / 2) + 16 * g;
                final int ab = aBase + (i * QK_K + 32 * g) * SimdI8.TILE_TOKENS;
                ShortVector p0 = ShortVector.zero(SimdI8.S16);
                ShortVector p1 = p0;
                for (int k = 0; k < 16; k++) {
                    final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                    p0 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w0[wb + k]).reinterpretShape(SimdI8.S16, 0))
                            .mul(v).add(p0);
                    p1 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w1[wb + k]).reinterpretShape(SimdI8.S16, 0))
                            .mul(v).add(p1);
                }
                final IntVector l0 = (IntVector) p0.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                final IntVector l1 = (IntVector) p1.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                is0 = l0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                        .add(l0.lanewise(VectorOperators.ASHR, 16)).mul(sc0[8 * i + g]).add(is0);
                is1 = l1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                        .add(l1.lanewise(VectorOperators.ASHR, 16)).mul(sc1[8 * i + g]).add(is1);
            }
            final int o = i * SUMS_PER_BLOCK;
            ((IntVector) is0.check(SimdI8.I8)).intoArray(sums, o);
            ((IntVector) is1.check(SimdI8.I8)).intoArray(sums, o + 2 * SimdI8.TILE_TOKENS);
        }
    }

    /**
     * Per super-block {@code i}, rows 0 and 1's sums over 8 sub-blocks of 32 of the sub-block scale times the
     * sub-block's quant dot, at {@code sums[(4i + 0) * 8]} and {@code sums[(4i + 2) * 8]}, one int lane per token.
     * Eight products of a 5-bit quant and an int8 quant fit a short, so each sub-block folds twice.
     */
    private static void subBlocks32Fold8(int[] w0, int[] sc0, int[] w1, int[] sc1, SimdI8.TokenTiles tiles, int tile,
                                    int nb, int[] sums) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        for (int i = 0; i < nb; i++) {
            IntVector is0 = IntVector.zero(SimdI8.I8);
            IntVector is1 = is0;
            for (int g = 0; g < 8; g++) {
                final int wb = i * (QK_K / 2) + 16 * g;
                final int ab = aBase + (i * QK_K + 32 * g) * SimdI8.TILE_TOKENS;
                IntVector g0 = IntVector.zero(SimdI8.I8);
                IntVector g1 = g0;
                for (int h = 0; h < 16; h += 8) {
                    ShortVector p0 = ShortVector.zero(SimdI8.S16);
                    ShortVector p1 = p0;
                    for (int k = h; k < h + 8; k++) {
                        final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                        p0 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w0[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p0);
                        p1 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w1[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p1);
                    }
                    final IntVector l0 = (IntVector) p0.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    final IntVector l1 = (IntVector) p1.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    g0 = l0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l0.lanewise(VectorOperators.ASHR, 16)).add(g0);
                    g1 = l1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l1.lanewise(VectorOperators.ASHR, 16)).add(g1);
                }
                is0 = IntVector.broadcast(SimdI8.I8, sc0[8 * i + g]).mul(g0).add(is0);
                is1 = IntVector.broadcast(SimdI8.I8, sc1[8 * i + g]).mul(g1).add(is1);
            }
            final int o = i * SUMS_PER_BLOCK;
            ((IntVector) is0.check(SimdI8.I8)).intoArray(sums, o);
            ((IntVector) is1.check(SimdI8.I8)).intoArray(sums, o + 2 * SimdI8.TILE_TOKENS);
        }
    }

    /**
     * Q6_K / Q3_K in one pass: per super-block the scaled sub-block sums of 16 sub-blocks of 16, then
     * {@code sumf += (d_y * d) * isum} lane-wise per token. Four products of a 6-bit quant and an int8 quant fit a
     * short and eight may not, so each sub-block folds twice.
     */
    private static void signed16(int[] w0, int[] sc0, float[] d0, int[] w1, int[] sc1, float[] d1,
                                 SimdI8.TokenTiles tiles, int tile, int nb, float[] out) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        final int yBase = tiles.scaleBase(tile);
        FloatVector f0 = FloatVector.zero(SimdFp.F8);
        FloatVector f1 = f0;
        for (int i = 0; i < nb; i++) {
            IntVector is0 = IntVector.zero(SimdI8.I8);
            IntVector is1 = is0;
            for (int g = 0; g < QK_K / 16; g++) {
                final int wb = i * (QK_K / 2) + 8 * g;
                final int ab = aBase + (i * QK_K + 16 * g) * SimdI8.TILE_TOKENS;
                IntVector g0 = IntVector.zero(SimdI8.I8);
                IntVector g1 = g0;
                for (int h = 0; h < 8; h += 4) {
                    ShortVector p0 = ShortVector.zero(SimdI8.S16);
                    ShortVector p1 = p0;
                    for (int k = h; k < h + 4; k++) {
                        final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                        p0 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w0[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p0);
                        p1 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w1[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p1);
                    }
                    final IntVector l0 = (IntVector) p0.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    final IntVector l1 = (IntVector) p1.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    g0 = l0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l0.lanewise(VectorOperators.ASHR, 16)).add(g0);
                    g1 = l1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l1.lanewise(VectorOperators.ASHR, 16)).add(g1);
                }
                is0 = IntVector.broadcast(SimdI8.I8, sc0[16 * i + g]).mul(g0).add(is0);
                is1 = IntVector.broadcast(SimdI8.I8, sc1[16 * i + g]).mul(g1).add(is1);
            }
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, tiles.scales, yBase + i * SimdI8.TILE_TOKENS);
            f0 = ((FloatVector) f0.check(SimdFp.F8)).add(yd.mul(d0[2 * i])
                    .mul((FloatVector) is0.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0)));
            f1 = ((FloatVector) f1.check(SimdFp.F8)).add(yd.mul(d1[2 * i])
                    .mul((FloatVector) is1.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0)));
        }
        ((FloatVector) f0.check(SimdFp.F8)).intoArray(out, 0);
        ((FloatVector) f1.check(SimdFp.F8)).intoArray(out, SimdI8.TILE_TOKENS);
    }

    /**
     * Per super-block {@code i}, rows 0 and 1's mins sums {@code sum_g mins[g] * group sum g} over the tile's
     * {@code groups} activation group sums, at {@code sums[(4i + 1) * 8]} and {@code sums[(4i + 3) * 8]}.
     */
    private static void mins(int[] m0, int[] m1, SimdI8.TokenTiles tiles, int tile, int nb, int groups,
                             int[] sums) {
        final int pBase = tiles.pairBase(tile);
        for (int i = 0; i < nb; i++) {
            IntVector ms0 = IntVector.zero(SimdI8.I8);
            IntVector ms1 = ms0;
            for (int g = 0; g < groups; g++) {
                final int gp = pBase + (groups * i + g) * SimdI8.TILE_TOKENS;
                ms0 = IntVector.fromArray(SimdI8.I8, tiles.groupSums, gp).mul(m0[groups * i + g]).add(ms0);
                ms1 = IntVector.fromArray(SimdI8.I8, tiles.groupSums, gp).mul(m1[groups * i + g]).add(ms1);
            }
            final int o = i * SUMS_PER_BLOCK;
            ((IntVector) ms0.check(SimdI8.I8)).intoArray(sums, o + SimdI8.TILE_TOKENS);
            ((IntVector) ms1.check(SimdI8.I8)).intoArray(sums, o + 3 * SimdI8.TILE_TOKENS);
        }
    }

    /**
     * Q4_K / Q5_K: {@code sums += (d * d_y) * isum}, {@code sumf -= (dmin * d_y) * msum} per super-block, result
     * {@code sumf + sums}, lane-wise per token.
     */
    private static void minusMins(float[] d0, float[] d1, SimdI8.TokenTiles tiles, int tile, int nb, int[] sums,
                                  float[] out) {
        final int yBase = tiles.scaleBase(tile);
        FloatVector fs0 = FloatVector.zero(SimdFp.F8);
        FloatVector fm0 = fs0;
        FloatVector fs1 = fs0;
        FloatVector fm1 = fs0;
        for (int i = 0; i < nb; i++) {
            final int o = i * SUMS_PER_BLOCK;
            final int yo = yBase + i * SimdI8.TILE_TOKENS;
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, tiles.scales, yo);
            final FloatVector s0 = (FloatVector) IntVector.fromArray(SimdI8.I8, sums, o)
                    .convertShape(VectorOperators.I2F, SimdFp.F8, 0);
            final FloatVector m0 = (FloatVector) IntVector.fromArray(SimdI8.I8, sums, o + SimdI8.TILE_TOKENS)
                    .convertShape(VectorOperators.I2F, SimdFp.F8, 0);
            final FloatVector s1 = (FloatVector) IntVector.fromArray(SimdI8.I8, sums, o + 2 * SimdI8.TILE_TOKENS)
                    .convertShape(VectorOperators.I2F, SimdFp.F8, 0);
            final FloatVector m1 = (FloatVector) IntVector.fromArray(SimdI8.I8, sums, o + 3 * SimdI8.TILE_TOKENS)
                    .convertShape(VectorOperators.I2F, SimdFp.F8, 0);
            fs0 = ((FloatVector) fs0.check(SimdFp.F8)).add(yd.mul(d0[2 * i]).mul(s0));
            fm0 = ((FloatVector) fm0.check(SimdFp.F8)).sub(yd.mul(d0[2 * i + 1]).mul(m0));
            fs1 = ((FloatVector) fs1.check(SimdFp.F8)).add(yd.mul(d1[2 * i]).mul(s1));
            fm1 = ((FloatVector) fm1.check(SimdFp.F8)).sub(yd.mul(d1[2 * i + 1]).mul(m1));
        }
        ((FloatVector) fm0.check(SimdFp.F8)).add(fs0).intoArray(out, 0);
        ((FloatVector) fm1.check(SimdFp.F8)).add(fs1).intoArray(out, SimdI8.TILE_TOKENS);
    }

    /**
     * Q2_K in one pass: per super-block the scaled sub-block sums of 16 sub-blocks of 16 and the mins sum over the
     * sixteen 16-quant group sums, then {@code sumf += (d_y * d) * isum - (d_y * dmin) * summs} lane-wise per
     * token. Four products of a 2-bit code and an int8 quant fit a short many times over; the form shares the
     * sub-block loop of {@link #signed16}.
     */
    private static void mins16(int[] w0, int[] sc0, int[] m0, float[] d0, int[] w1, int[] sc1, int[] m1, float[] d1,
                               SimdI8.TokenTiles tiles, int tile, int nb, float[] out) {
        final short[] a = tiles.lanes;
        final int aBase = tiles.laneBase(tile);
        final int yBase = tiles.scaleBase(tile);
        final int pBase = tiles.pairBase(tile);
        FloatVector f0 = FloatVector.zero(SimdFp.F8);
        FloatVector f1 = f0;
        for (int i = 0; i < nb; i++) {
            IntVector is0 = IntVector.zero(SimdI8.I8);
            IntVector is1 = is0;
            IntVector ms0 = is0;
            IntVector ms1 = is0;
            for (int g = 0; g < QK_K / 16; g++) {
                final int wb = i * (QK_K / 2) + 8 * g;
                final int ab = aBase + (i * QK_K + 16 * g) * SimdI8.TILE_TOKENS;
                IntVector g0 = IntVector.zero(SimdI8.I8);
                IntVector g1 = g0;
                for (int h = 0; h < 8; h += 4) {
                    ShortVector p0 = ShortVector.zero(SimdI8.S16);
                    ShortVector p1 = p0;
                    for (int k = h; k < h + 4; k++) {
                        final ShortVector v = ShortVector.fromArray(SimdI8.S16, a, ab + 16 * k);
                        p0 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w0[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p0);
                        p1 = ((ShortVector) IntVector.broadcast(SimdI8.I8, w1[wb + k]).reinterpretShape(SimdI8.S16, 0))
                                .mul(v).add(p1);
                    }
                    final IntVector l0 = (IntVector) p0.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    final IntVector l1 = (IntVector) p1.check(SimdI8.S16).reinterpretShape(SimdI8.I8, 0);
                    g0 = l0.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l0.lanewise(VectorOperators.ASHR, 16)).add(g0);
                    g1 = l1.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                            .add(l1.lanewise(VectorOperators.ASHR, 16)).add(g1);
                }
                is0 = IntVector.broadcast(SimdI8.I8, sc0[16 * i + g]).mul(g0).add(is0);
                is1 = IntVector.broadcast(SimdI8.I8, sc1[16 * i + g]).mul(g1).add(is1);
                final int gp = pBase + (16 * i + g) * SimdI8.TILE_TOKENS;
                ms0 = IntVector.fromArray(SimdI8.I8, tiles.groupSums, gp).mul(m0[16 * i + g]).add(ms0);
                ms1 = IntVector.fromArray(SimdI8.I8, tiles.groupSums, gp).mul(m1[16 * i + g]).add(ms1);
            }
            final FloatVector yd = FloatVector.fromArray(SimdFp.F8, tiles.scales, yBase + i * SimdI8.TILE_TOKENS);
            f0 = ((FloatVector) f0.check(SimdFp.F8)).add(yd.mul(d0[2 * i])
                    .mul((FloatVector) is0.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0))
                    .sub(yd.mul(d0[2 * i + 1])
                            .mul((FloatVector) ms0.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0))));
            f1 = ((FloatVector) f1.check(SimdFp.F8)).add(yd.mul(d1[2 * i])
                    .mul((FloatVector) is1.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0))
                    .sub(yd.mul(d1[2 * i + 1])
                            .mul((FloatVector) ms1.check(SimdI8.I8).convertShape(VectorOperators.I2F, SimdFp.F8, 0))));
        }
        ((FloatVector) f0.check(SimdFp.F8)).intoArray(out, 0);
        ((FloatVector) f1.check(SimdFp.F8)).intoArray(out, SimdI8.TILE_TOKENS);
    }
}
