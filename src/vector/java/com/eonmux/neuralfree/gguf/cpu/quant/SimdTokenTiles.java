package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.sun.management.HotSpotDiagnosticMXBean;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.management.ManagementFactory;

/**
 * The batched (prefill) form of the 32-element block kernels ({@link SimdBlockTiles}): activations laid out once per
 * product as tiles of {@link #TILE_TOKENS} tokens of float lanes ({@link Layout}), weight rows unpacked once per
 * task into float values ({@link #rowValues}), and the integer sums of the 32-element blocks of {@link #TILE_ROWS}
 * rows by up to {@link #PAIR} tiles computed with multiply-adds ({@link #sums}).
 * <p>
 * The float lanes compute the integer sums of the scalar kernel. Every operand is an integer, and {@link #sums}
 * accumulates each (row, token) block sum in one chain whose partial sums are integers of magnitude at most
 * {@code 32 * 128 * 128 = 2^19}, which a float represents exactly, so neither a product nor a fused multiply-add
 * rounds. The kernel then runs the scalar kernel's float combination lane-wise, one lane per token, so the results
 * are bit-identical to the scalar kernel's. The float lanes run at the fused multiply-add rate, where an int16
 * product needs a multiply, an add and a three-operation pairwise fold per sixteen products.
 * <p>
 * Because nothing rounds, a multiply followed by an add gives the same sums. The kernels use it where the JIT has no
 * fused multiply-add instruction ({@link #HARDWARE_FMA}): there {@code FloatVector.fma} falls back to
 * {@link Math#fma} per lane, which is several hundred times slower.
 */
final class SimdTokenTiles {

    /** Tokens one tile holds: one float lane per token. */
    static final int TILE_TOKENS = 8;

    /** Weight rows one register tile of the batched kernels multiplies. */
    static final int TILE_ROWS = 4;

    /** Token tiles one register tile of the batched kernels multiplies. */
    static final int PAIR = 2;

    /** The float species of one tile column: {@link #TILE_TOKENS} lanes. */
    static final VectorSpecies<Float> F8 = FloatVector.SPECIES_256;

    /**
     * Whether the JIT compiles {@code FloatVector.fma} to the hardware's fused multiply-add: HotSpot's
     * {@code UseFMA}, on by default where the processor has one. False when the option cannot be read.
     */
    static final boolean HARDWARE_FMA = hardwareFma();

    private static final ThreadLocal<float[][]> ROW_VALUES = new ThreadLocal<>();

    private static final ThreadLocal<float[]> SEGMENT = new ThreadLocal<>();


    private SimdTokenTiles() {
    }

    private static boolean hardwareFma() {
        try {
            return Boolean.parseBoolean(ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class)
                    .getVMOption("UseFMA").getValue());
        } catch (RuntimeException | LinkageError e) {
            return false;
        }
    }

    /**
     * The activations of one batched product as tiles of {@link #TILE_TOKENS} tokens: in tile {@code t}, element
     * {@code e} of token {@code ti} sits at {@code values[t * tileValues + 8e + ti]}, so one 8-lane load holds
     * element {@code e} of every token of the tile and a weight value broadcast over it multiplies all eight at
     * once. {@code scales[t * tileScales + 8b + ti]} is token {@code ti}'s block-{@code b} scale {@code d}, and
     * {@code sums} holds the Q8_1 block sum terms {@code s} at the same places (empty for other activations).
     * Tokens past the last pad with zero quants, scales and sums. Immutable once built.
     */
    static final class Layout implements CpuVecDot.TileActivations {
        final float[] values;
        final float[] scales;
        final float[] sums;
        final int tokens;
        final int count;
        final int cols;
        /** Elements per activation block: 32 for Q8_0 and Q8_1, 256 for Q8_K. */
        final int blockElements;
        final int tileValues;
        final int tileScales;

        private Layout(int tokens, int cols, int blocks, boolean withSums) {
            this.tokens = tokens;
            this.cols = cols;
            this.blockElements = cols / blocks;
            this.count = (tokens + TILE_TOKENS - 1) / TILE_TOKENS;
            this.tileValues = cols * TILE_TOKENS;
            this.tileScales = blocks * TILE_TOKENS;
            this.values = new float[Math.multiplyExact(count, tileValues)];
            this.scales = new float[Math.multiplyExact(count, tileScales)];
            this.sums = new float[withSums ? scales.length : 0];
        }

        @Override
        public int tokens() {
            return tokens;
        }

        /** Lays out token {@code t}'s quants and scales; tokens past the last keep their zeros. */
        private void put(int t, CpuVecDot.QuantizedActivations x) {
            final int base = (t / TILE_TOKENS) * tileValues + t % TILE_TOKENS;
            final byte[] qs = x.qs;
            for (int e = 0; e < cols; e++) {
                values[base + TILE_TOKENS * e] = qs[e];
            }
            final int sBase = (t / TILE_TOKENS) * tileScales + t % TILE_TOKENS;
            final float[] d = x.d;
            for (int b = 0; b < d.length; b++) {
                scales[sBase + TILE_TOKENS * b] = d[b];
            }
            if (sums.length != 0) {
                final float[] s = x.s;
                for (int b = 0; b < s.length; b++) {
                    sums[sBase + TILE_TOKENS * b] = s[b];
                }
            }
        }

        /** The first value of tile {@code tile}. */
        int valueBase(int tile) {
            return tile * tileValues;
        }

        /** The first scale of tile {@code tile}. */
        int scaleBase(int tile) {
            return tile * tileScales;
        }

    }

    /**
     * The {@link Layout} of the {@code nTokens} tokens of {@code x}, one tile per {@link CpuParallel} index: the
     * eight tokens of a tile share cache lines, so one worker writes all of them.
     */
    static Layout layout(CpuVecDot.QuantizedActivations[] x, int nTokens) {
        final Layout tiles = new Layout(nTokens, x[0].n, x[0].d.length, x[0].s != null);
        final int grain = Math.max(1, (1 << 16) / Math.max(1, x[0].n * TILE_TOKENS));
        CpuParallel.parallelFor(tiles.count, grain, (from, to) -> {
            for (int t = from * TILE_TOKENS, end = Math.min(nTokens, to * TILE_TOKENS); t < end; t++) {
                tiles.put(t, x[t]);
            }
        });
        return tiles;
    }

    /**
     * Validates the activations a batched kernel lays out: {@code nTokens} vectors of {@code type}, all of the
     * first one's length.
     *
     * @throws IllegalArgumentException naming the first token that is null, of another type or of another length
     */
    static void checkTokens(CpuVecDot.QuantizedActivations[] x, int nTokens, GGMLType type) {
        for (int t = 0; t < nTokens; t++) {
            if (x[t] == null || x[t].type != type || x[t].n != x[0].n) {
                throw new IllegalArgumentException(type + " tiles need " + type + " activations of one length,"
                        + " token " + t + " is " + x[t]);
            }
        }
    }

    /**
     * Validates a token range of a layout: it lies inside the layout and starts on a tile.
     *
     * @throws IllegalArgumentException  for a start that is not a multiple of {@link #TILE_TOKENS}
     * @throws IndexOutOfBoundsException for a range outside the layout
     */
    static void checkRange(Layout tiles, int tokenStart, int nTokens) {
        if (tokenStart < 0 || nTokens < 0 || tokenStart > tiles.tokens - nTokens) {
            throw new IndexOutOfBoundsException("tokens [" + tokenStart + ", " + ((long) tokenStart + nTokens)
                    + ") exceed the " + tiles.tokens + "-token tiles");
        }
        if (tokenStart % TILE_TOKENS != 0) {
            throw new IllegalArgumentException("token start " + tokenStart + " is not a multiple of the "
                    + TILE_TOKENS + "-token tile");
        }
    }

    /**
     * Validates a layout against weight {@code w}: the layout's row length is {@code w}'s.
     *
     * @throws IllegalArgumentException when the lengths differ
     */
    static void checkWidth(CpuTensorRef w, Layout tiles) {
        if (tiles.cols != w.cols()) {
            throw new IllegalArgumentException(w.type() + " row of " + w.cols() + " elements cannot be dotted with "
                    + tiles.cols + " activations");
        }
    }

    /** The calling thread's {@link #TILE_ROWS} unpacked weight rows of at least {@code cols} float values each. */
    static float[][] rowValues(int cols) {
        float[][] buf = ROW_VALUES.get();
        if (buf == null || buf[0].length < cols) {
            buf = new float[TILE_ROWS][Math.max(cols, 1024)];
            ROW_VALUES.set(buf);
        }
        return buf;
    }

    /**
     * The calling thread's segment sums of a tile pair over a row of {@code cols} elements cut into segments of
     * {@code len} ({@link #sums}).
     */
    static float[] segmentSums(int cols, int len) {
        final int n = cols / len * PAIR * TILE_ROWS * TILE_TOKENS;
        float[] buf = SEGMENT.get();
        if (buf == null || buf.length < n) {
            buf = new float[Math.max(n, 4096)];
            SEGMENT.set(buf);
        }
        return buf;
    }

    /**
     * Index of row {@code r}'s lanes of segment {@code sg} for tile {@code k} (0 or 1) of a pair in the output of
     * {@link #sums}.
     */
    static int sumIndex(int sg, int k, int r) {
        return ((sg * PAIR + k) * TILE_ROWS + r) * TILE_TOKENS;
    }

    /**
     * The segment sums of the four rows {@code w[0..3]} over the {@code count} segments of {@code len} elements from
     * element 0, with token tile {@code tile} of {@code tiles} and, when {@code tile < lastTile}, with the next tile
     * as well: {@code out[sumIndex(sg, k, r) + ti] = sum_j w[r][len * sg + j] * a_k[len * sg + j][ti]}, {@code a_k}
     * the elements of tile {@code tile + k}. Each (row, tile) pair accumulates its own chain of multiply-adds (fused
     * where {@link #HARDWARE_FMA}), one segment long, which the class comment's bound keeps exact. One call covers a
     * whole row, so the loop over segments stays inside one compiled method.
     *
     * @return the tiles covered: 2, or 1 for the last tile
     */
    static int sums(float[][] w, int len, int count, Layout tiles, int tile, int lastTile, float[] out) {
        if (tile < lastTile) {
            pairSums(w, len, count, tiles.values, tiles.valueBase(tile), tiles.valueBase(tile + 1), out);
            return PAIR;
        }
        singleSums(w, len, count, tiles.values, tiles.valueBase(tile), out);
        return 1;
    }

    /**
     * {@link #sums} of two tiles: eight chains, each element costing four weight broadcasts and two activation loads
     * for eight multiply-adds, which keeps the fourteen live vectors in registers.
     */
    private static void pairSums(float[][] w, int len, int count, float[] a, int aOff0, int aOff1, float[] out) {
        final float[] w0 = w[0];
        final float[] w1 = w[1];
        final float[] w2 = w[2];
        final float[] w3 = w[3];
        for (int sg = 0; sg < count; sg++) {
            FloatVector s0 = FloatVector.zero(F8);
            FloatVector s1 = s0;
            FloatVector s2 = s0;
            FloatVector s3 = s0;
            FloatVector t0 = s0;
            FloatVector t1 = s0;
            FloatVector t2 = s0;
            FloatVector t3 = s0;
            final int e0 = sg * len;
            for (int i = e0; i < e0 + len; i++) {
                final FloatVector x = FloatVector.fromArray(F8, a, aOff0 + TILE_TOKENS * i);
                final FloatVector y = FloatVector.fromArray(F8, a, aOff1 + TILE_TOKENS * i);
                final FloatVector b0 = FloatVector.broadcast(F8, w0[i]);
                s0 = HARDWARE_FMA ? b0.fma(x, s0) : b0.mul(x).add(s0);
                t0 = HARDWARE_FMA ? b0.fma(y, t0) : b0.mul(y).add(t0);
                final FloatVector b1 = FloatVector.broadcast(F8, w1[i]);
                s1 = HARDWARE_FMA ? b1.fma(x, s1) : b1.mul(x).add(s1);
                t1 = HARDWARE_FMA ? b1.fma(y, t1) : b1.mul(y).add(t1);
                final FloatVector b2 = FloatVector.broadcast(F8, w2[i]);
                s2 = HARDWARE_FMA ? b2.fma(x, s2) : b2.mul(x).add(s2);
                t2 = HARDWARE_FMA ? b2.fma(y, t2) : b2.mul(y).add(t2);
                final FloatVector b3 = FloatVector.broadcast(F8, w3[i]);
                s3 = HARDWARE_FMA ? b3.fma(x, s3) : b3.mul(x).add(s3);
                t3 = HARDWARE_FMA ? b3.fma(y, t3) : b3.mul(y).add(t3);
            }
            ((FloatVector) s0.check(F8)).intoArray(out, sumIndex(sg, 0, 0));
            ((FloatVector) s1.check(F8)).intoArray(out, sumIndex(sg, 0, 1));
            ((FloatVector) s2.check(F8)).intoArray(out, sumIndex(sg, 0, 2));
            ((FloatVector) s3.check(F8)).intoArray(out, sumIndex(sg, 0, 3));
            ((FloatVector) t0.check(F8)).intoArray(out, sumIndex(sg, 1, 0));
            ((FloatVector) t1.check(F8)).intoArray(out, sumIndex(sg, 1, 1));
            ((FloatVector) t2.check(F8)).intoArray(out, sumIndex(sg, 1, 2));
            ((FloatVector) t3.check(F8)).intoArray(out, sumIndex(sg, 1, 3));
        }
    }

    /** {@link #sums} of one tile: four chains. */
    private static void singleSums(float[][] w, int len, int count, float[] a, int aOff, float[] out) {
        final float[] w0 = w[0];
        final float[] w1 = w[1];
        final float[] w2 = w[2];
        final float[] w3 = w[3];
        for (int sg = 0; sg < count; sg++) {
            FloatVector s0 = FloatVector.zero(F8);
            FloatVector s1 = s0;
            FloatVector s2 = s0;
            FloatVector s3 = s0;
            final int e0 = sg * len;
            for (int i = e0; i < e0 + len; i++) {
                final FloatVector x = FloatVector.fromArray(F8, a, aOff + TILE_TOKENS * i);
                final FloatVector b0 = FloatVector.broadcast(F8, w0[i]);
                s0 = HARDWARE_FMA ? b0.fma(x, s0) : b0.mul(x).add(s0);
                final FloatVector b1 = FloatVector.broadcast(F8, w1[i]);
                s1 = HARDWARE_FMA ? b1.fma(x, s1) : b1.mul(x).add(s1);
                final FloatVector b2 = FloatVector.broadcast(F8, w2[i]);
                s2 = HARDWARE_FMA ? b2.fma(x, s2) : b2.mul(x).add(s2);
                final FloatVector b3 = FloatVector.broadcast(F8, w3[i]);
                s3 = HARDWARE_FMA ? b3.fma(x, s3) : b3.mul(x).add(s3);
            }
            ((FloatVector) s0.check(F8)).intoArray(out, sumIndex(sg, 0, 0));
            ((FloatVector) s1.check(F8)).intoArray(out, sumIndex(sg, 0, 1));
            ((FloatVector) s2.check(F8)).intoArray(out, sumIndex(sg, 0, 2));
            ((FloatVector) s3.check(F8)).intoArray(out, sumIndex(sg, 0, 3));
        }
    }

    /**
     * Writes rows {@code [0, nr)} of the register tile {@code out} ({@code out[8i + ti]}: row {@code i}, token
     * {@code ti}) for the tile's tokens {@code [t0, t0 + n)} of the product: row {@code r + i} of token
     * {@code t0 + ti} lands at {@code y[yOff + r + i + (t0 + ti) * yStride]}.
     */
    static void store(float[] out, int nr, int n, float[] y, int yOff, int r, int t0, int yStride) {
        for (int i = 0; i < nr; i++) {
            for (int ti = 0; ti < n; ti++) {
                y[yOff + r + i + (t0 + ti) * yStride] = out[i * TILE_TOKENS + ti];
            }
        }
    }
}
