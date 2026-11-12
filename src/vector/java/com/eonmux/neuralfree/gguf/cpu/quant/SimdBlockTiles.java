package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * The batched (prefill) product of the 32-element block kernels ({@code Q8_0}, {@code Q4_0}, {@code Q4_1}, {@code
 * Q5_0}, {@code Q5_1}, {@code IQ4_NL} and {@code IQ4_XS}) over the float token tiles of {@link SimdTokenTiles}: each
 * weight row unpacks once per task into float quant values with its per-block scale terms ({@link Rows}), {@link
 * SimdTokenTiles#sums} forms the exact integer block sums of {@link SimdTokenTiles#TILE_ROWS} rows by up to {@link
 * SimdTokenTiles#PAIR} tiles in float lanes, and the float half runs the scalar kernel's per-block combination
 * lane-wise, one lane per token ({@link Form}). Every block quant is an integer of magnitude at most 128 and every
 * activation quant one of at most 128, so the block sums stay within the {@code 2^19} bound that keeps them exact, and
 * every form returns the bits of its scalar kernel.
 * <p>
 * The vector code keeps the rules of {@link SimdI8}: vectors live only inside the methods that create them, and a
 * loop-carried vector is an argument, never a receiver, unless its species was just re-asserted.
 */
final class SimdBlockTiles {

    /** Elements per block. */
    static final int BLOCK = 32;

    private static final ThreadLocal<float[][]> ROW_MINS = new ThreadLocal<>();

    private static final ThreadLocal<byte[]> QUANTS = ThreadLocal.withInitial(() -> new byte[BLOCK]);

    private SimdBlockTiles() {
    }

    /** The per-block float combination of a block type, in its scalar kernel's evaluation order. */
    enum Form {
        /** Q8_0, Q5_0, IQ4_NL: {@code sumf += sumi * (d_x * d_y)}, whose scale product commutes. */
        DOT(GGMLType.Q8_0),
        /** Q4_0: {@code sumf += sumi * d_x * d_y}, the integer sum scaled by the weight scale first. */
        SCALED(GGMLType.Q8_0),
        /** Q4_1, Q5_1: {@code sumf += (d_x * d_y) * sumi + m_x * s_y}, {@code s_y} the Q8_1 block sum term. */
        MINS(GGMLType.Q8_1),
        /**
         * IQ4_XS: per 32-element sub-block of a 256-element super-block,
         * {@code sumf += ((d_x * d_y) * (ls - 32)) * sumi}, {@code d_y} the Q8_K super-block scale.
         */
        SUB_SCALED(GGMLType.Q8_K);

        /** The activation type the form's kernels consume. */
        final GGMLType activations;

        Form(GGMLType activations) {
            this.activations = activations;
        }
    }

    /** Unpacks the weight rows of one block type for the token-tile product. */
    @FunctionalInterface
    interface Rows {
        /**
         * Unpacks the {@code nb} 32-element blocks of the row at {@code b[0..)}: {@code values[32i + j]} is element
         * {@code j} of block {@code i} as its integer quant value, {@code d[i]} and {@code m[i]} the block's scale
         * terms of the {@link Form} ({@code d_x} and {@code m_x}; for {@link Form#SUB_SCALED} {@code ls - 32} and
         * the super-block's {@code d_x}).
         */
        void unpack(byte[] b, int nb, float[] values, float[] d, float[] m);
    }

    /**
     * A 32-element block kernel whose batched product runs over the float token tiles: the kernel names its weight
     * type, its {@link Form} and its row unpack, and inherits the prepared and unprepared tile products and the
     * register-tile shape of four rows by two eight-token tiles in four-byte lanes.
     */
    interface Kernel extends CpuVecDot {

        /** The weight type the kernel reads. */
        GGMLType weightType();

        /** The per-block float combination of the kernel's scalar reference. */
        Form form();

        /** Unpacks one row for the tile product ({@link Rows}). */
        void unpackRow(byte[] b, int nb, float[] values, float[] d, float[] m);

        @Override
        default void dotRows(CpuTensorRef w, long rowStart, int rowCount, CpuVecDot.QuantizedActivations[] x,
                             int nTokens, float[] y, int yOff, int yStride) {
            dotTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
        }

        @Override
        default void dotTiles(CpuTensorRef w, long rowStart, int rowCount, CpuVecDot.QuantizedActivations[] x,
                              int nTokens, float[] y, int yOff, int yStride) {
            if (nTokens == 0) {
                return;
            }
            checkTiles(w, rowStart, rowCount, x, nTokens, y, yOff, yStride);
            product(w, rowStart, rowCount, SimdTokenTiles.layout(x, nTokens), 0, nTokens, y, yOff, yStride, form(),
                    this::unpackRow);
        }

        @Override
        default TileActivations prepareTiles(CpuVecDot.QuantizedActivations[] x, int nTokens) {
            return layout(x, nTokens, form());
        }

        @Override
        default int tileTokens() {
            return SimdTokenTiles.TILE_TOKENS;
        }

        /** The four rows of the float register tile. */
        @Override
        default int tileRows() {
            return SimdTokenTiles.TILE_ROWS;
        }

        /** The two eight-token tiles of the float register tile. */
        @Override
        default int tilePanelTokens() {
            return SimdTokenTiles.PAIR * SimdTokenTiles.TILE_TOKENS;
        }

        /** One float lane per laid-out activation element. */
        @Override
        default int tileLaneBytes() {
            return Float.BYTES;
        }

        @Override
        default void dotTiles(CpuTensorRef w, long rowStart, int rowCount, TileActivations tiles,
                              CpuVecDot.QuantizedActivations[] x, int tokenStart, int nTokens, float[] y, int yOff,
                              int yStride) {
            if (!(tiles instanceof SimdTokenTiles.Layout)) {
                CpuVecDot.super.dotTiles(w, rowStart, rowCount, tiles, x, tokenStart, nTokens, y, yOff, yStride);
                return;
            }
            checkedProduct(w, weightType(), rowStart, rowCount, (SimdTokenTiles.Layout) tiles, tokenStart, nTokens,
                    y, yOff, yStride, form(), this::unpackRow);
        }
    }

    /**
     * The calling thread's 32-byte block scratch of the row unpackers, which they fill with one block's quants in
     * element order before {@link #widen} turns them into float values.
     */
    static byte[] quants() {
        return QUANTS.get();
    }

    /** The 32 signed bytes {@code q[0..32)} as float values at {@code values[at..at + 32)}. */
    static void widen(byte[] q, float[] values, int at) {
        for (int k = 0; k < BLOCK; k += 8) {
            ((FloatVector) ByteVector.fromArray(SimdI8.B8, q, k).convertShape(VectorOperators.B2F,
                    SimdTokenTiles.F8, 0)).intoArray(values, at + k);
        }
    }

    /** The {@link SimdTokenTiles.Layout} of {@code x} after validating it holds {@code form}'s activations. */
    static SimdTokenTiles.Layout layout(CpuVecDot.QuantizedActivations[] x, int nTokens, Form form) {
        SimdTokenTiles.checkTokens(x, nTokens, form.activations);
        return nTokens == 0 ? null : SimdTokenTiles.layout(x, nTokens);
    }

    /**
     * {@link #product} after validating a prepared product: the token range lies in the layout and starts on a
     * tile, {@code w} is of {@code type} with the layout's row length, and the outputs fit.
     *
     * @throws IllegalArgumentException  for another weight type or row length, or an unaligned start
     * @throws IndexOutOfBoundsException for a token or output range outside its array
     */
    static void checkedProduct(CpuTensorRef w, GGMLType type, long rowStart, int rowCount,
                               SimdTokenTiles.Layout tiles, int tokenStart, int nTokens, float[] y, int yOff,
                               int yStride, Form form, Rows unpacker) {
        SimdTokenTiles.checkRange(tiles, tokenStart, nTokens);
        Q8VecDot.requireWeightType(w, type);
        SimdTokenTiles.checkWidth(w, tiles);
        CpuVecDotRows.checkOutputs(w, rowStart, rowCount, nTokens, y, yOff, yStride);
        product(w, rowStart, rowCount, tiles, tokenStart, nTokens, y, yOff, yStride, form, unpacker);
    }

    /**
     * Rows {@code [rowStart, rowStart + rowCount)} of {@code w} against tokens {@code [tokenStart, tokenStart +
     * nTokens)} of {@code tiles}: row {@code r} of token {@code t} lands at {@code y[yOff + r + t * yStride]}. The
     * caller has validated the ranges; {@code tokenStart} is a multiple of the tile.
     */
    static void product(CpuTensorRef w, long rowStart, int rowCount, SimdTokenTiles.Layout tiles, int tokenStart,
                        int nTokens, float[] y, int yOff, int yStride, Form form, Rows unpacker) {
        if (nTokens == 0 || rowCount == 0) {
            return;
        }
        final int nb = tiles.cols / BLOCK;
        final float[][] values = SimdTokenTiles.rowValues(tiles.cols);
        final float[][] d = SimdI8.rowScales(SimdTokenTiles.TILE_ROWS, nb);
        final float[][] m = rowMins(nb);
        final float[] out = SimdI8.tileAccumulators(SimdTokenTiles.TILE_ROWS * SimdTokenTiles.TILE_TOKENS);
        final float[] seg = SimdTokenTiles.segmentSums(tiles.cols, BLOCK);
        final byte[] buf = CpuVecDotRows.buffer(w);
        final CpuVecDotRows.RowReader reader = CpuVecDotRows.reader(w);
        final int firstTile = tokenStart / SimdTokenTiles.TILE_TOKENS;
        final int lastTile = (tokenStart + nTokens - 1) / SimdTokenTiles.TILE_TOKENS;
        for (int r = 0; r < rowCount; r += SimdTokenTiles.TILE_ROWS) {
            final int nr = Math.min(SimdTokenTiles.TILE_ROWS, rowCount - r);
            for (int i = 0; i < SimdTokenTiles.TILE_ROWS; i++) {
                reader.copy(rowStart + r + Math.min(i, nr - 1), buf);
                unpacker.unpack(buf, nb, values[i], d[i], m[i]);
            }
            for (int tile = firstTile; tile <= lastTile; ) {
                final int pair = SimdTokenTiles.sums(values, BLOCK, nb, tiles, tile, lastTile, seg);
                for (int k = 0; k < pair; k++, tile++) {
                    combine(form, d, m, tiles, tile, k, nb, seg, out);
                    final int t0 = tile * SimdTokenTiles.TILE_TOKENS - tokenStart;
                    SimdTokenTiles.store(out, nr, Math.min(SimdTokenTiles.TILE_TOKENS, nTokens - t0), y, yOff, r,
                            t0, yStride);
                }
            }
        }
    }

    /** The calling thread's per-block {@code m_x} terms of the tile's rows. */
    private static float[][] rowMins(int blocks) {
        float[][] buf = ROW_MINS.get();
        if (buf == null || buf[0].length < blocks) {
            buf = new float[SimdTokenTiles.TILE_ROWS][Math.max(blocks, 128)];
            ROW_MINS.set(buf);
        }
        return buf;
    }

    /**
     * The float combination of tile {@code tile}, member {@code k} of the pair whose block sums {@code seg} holds:
     * {@code out[r * TILE_TOKENS + ti]} receives row {@code r}'s dot with the tile's token {@code ti}.
     */
    private static void combine(Form form, float[][] d, float[][] m, SimdTokenTiles.Layout tiles, int tile, int k,
                                int nb, float[] seg, float[] out) {
        for (int r = 0; r < SimdTokenTiles.TILE_ROWS; r++) {
            final int o = r * SimdTokenTiles.TILE_TOKENS;
            switch (form) {
                case DOT:
                    dot(d[r], tiles, tile, k, r, nb, seg, out, o);
                    break;
                case SCALED:
                    scaled(d[r], tiles, tile, k, r, nb, seg, out, o);
                    break;
                case MINS:
                    mins(d[r], m[r], tiles, tile, k, r, nb, seg, out, o);
                    break;
                case SUB_SCALED:
                    subScaled(d[r], m[r], tiles, tile, k, r, nb, seg, out, o);
                    break;
                default:
                    throw new IllegalStateException(form.name());
            }
        }
    }

    /** {@link Form#DOT} of row {@code r}: {@code f += sumi * (d_y * d_x)} per block. */
    private static void dot(float[] d, SimdTokenTiles.Layout tiles, int tile, int k, int r, int nb, float[] seg,
                            float[] out, int o) {
        final int yBase = tiles.scaleBase(tile);
        FloatVector f = FloatVector.zero(SimdTokenTiles.F8);
        for (int ib = 0; ib < nb; ib++) {
            final FloatVector yd = FloatVector.fromArray(SimdTokenTiles.F8, tiles.scales,
                    yBase + ib * SimdTokenTiles.TILE_TOKENS);
            f = FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, r))
                    .mul(yd.mul(d[ib])).add(f);
        }
        ((FloatVector) f.check(SimdTokenTiles.F8)).intoArray(out, o);
    }

    /** {@link Form#SCALED} of row {@code r}: {@code f += sumi * d_x * d_y} per block. */
    private static void scaled(float[] d, SimdTokenTiles.Layout tiles, int tile, int k, int r, int nb, float[] seg,
                               float[] out, int o) {
        final int yBase = tiles.scaleBase(tile);
        FloatVector f = FloatVector.zero(SimdTokenTiles.F8);
        for (int ib = 0; ib < nb; ib++) {
            final FloatVector yd = FloatVector.fromArray(SimdTokenTiles.F8, tiles.scales,
                    yBase + ib * SimdTokenTiles.TILE_TOKENS);
            f = FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, r))
                    .mul(d[ib]).mul(yd).add(f);
        }
        ((FloatVector) f.check(SimdTokenTiles.F8)).intoArray(out, o);
    }

    /** {@link Form#MINS} of row {@code r}: {@code f += (d_y * d_x) * sumi + s_y * m_x} per block. */
    private static void mins(float[] d, float[] m, SimdTokenTiles.Layout tiles, int tile, int k, int r, int nb,
                             float[] seg, float[] out, int o) {
        final int yBase = tiles.scaleBase(tile);
        FloatVector f = FloatVector.zero(SimdTokenTiles.F8);
        for (int ib = 0; ib < nb; ib++) {
            final int yo = yBase + ib * SimdTokenTiles.TILE_TOKENS;
            final FloatVector yd = FloatVector.fromArray(SimdTokenTiles.F8, tiles.scales, yo);
            final FloatVector ys = FloatVector.fromArray(SimdTokenTiles.F8, tiles.sums, yo);
            f = yd.mul(d[ib]).mul(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, r)))
                    .add(ys.mul(m[ib])).add(f);
        }
        ((FloatVector) f.check(SimdTokenTiles.F8)).intoArray(out, o);
    }

    /**
     * {@link Form#SUB_SCALED} of row {@code r}: {@code f += ((d_y * d_x) * (ls - 32)) * sumi} per 32-element
     * sub-block, {@code d_y} the scale of the token's 256-element super-block.
     */
    private static void subScaled(float[] ls, float[] dx, SimdTokenTiles.Layout tiles, int tile, int k, int r,
                                  int nb, float[] seg, float[] out, int o) {
        final int yBase = tiles.scaleBase(tile);
        final int perSuper = tiles.blockElements / BLOCK;
        FloatVector f = FloatVector.zero(SimdTokenTiles.F8);
        for (int ib = 0; ib < nb; ib++) {
            final FloatVector yd = FloatVector.fromArray(SimdTokenTiles.F8, tiles.scales,
                    yBase + (ib / perSuper) * SimdTokenTiles.TILE_TOKENS);
            f = yd.mul(dx[ib]).mul(ls[ib])
                    .mul(FloatVector.fromArray(SimdTokenTiles.F8, seg, SimdTokenTiles.sumIndex(ib, k, r))).add(f);
        }
        ((FloatVector) f.check(SimdTokenTiles.F8)).intoArray(out, o);
    }
}
