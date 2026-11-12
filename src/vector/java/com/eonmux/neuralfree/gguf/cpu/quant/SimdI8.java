package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK_K;

/**
 * The widening int8 dot primitives of the Vector API vec-dot kernels ({@code SimdKernels} and its
 * {@code Simd*} kernels): int8 bytes widen to 32-bit lanes with {@code B2I}, multiply-add in int lanes, and
 * reduce horizontally. Every lane product and every lane sum is exact, so a block dot returns the same integer
 * as the scalar {@link Q8VecDot#dotI8} in any lane order. The species are fixed at 64, 128 and 256 bits: the JIT
 * compiles them to single instructions on AVX2 and AVX-512 hardware, while on narrower vector hardware (128-bit
 * NEON, SSE-only x86) it runs the 256-bit operations as their Java fallback, so {@link CpuSimd} keeps the scalar
 * kernels there.
 * <p>
 * Vectors load straight from the packed row arrays the row loops hand to the kernels; the int16 product path
 * ({@link #S16}, {@link #pairSum}) widens the sixteen int8 products of a byte vector into short lanes and
 * folds them pairwise into the int accumulator. Shifts run through {@code lanewise} of {@code SHL}, {@code
 * LSHR} and {@code ASHR}.
 * <p>
 * <b>Compiling the same whatever ran first.</b> C2 replaces a Vector API operation with its instruction only when
 * the receiver's exact vector class and the target species are compile-time constants. A vector that comes from
 * an operation with a constant species ({@code fromArray}, {@code broadcast}, {@code zero} of a static species, or
 * any lane-wise operation on such a vector) has an exact class; a loop-carried variable (a loop phi), a parameter
 * or a return value has only its declared type, and the operation then dispatches through the type profile of the
 * JDK's shared library code, which every other Vector API caller in the JVM also writes. Once other code (the
 * {@code FloatVector} tiles and attention, other kernels' shapes) has made such a profile polymorphic, C2 inlines
 * both targets behind guards, the method outgrows the inlining budget, the intrinsics fail and every operation
 * runs its allocating Java fallback for the life of the JVM, 2 to 15 times slower. The K-quant kernels
 * ({@code SimdQ2_K} to {@code SimdQ6_K}, {@link SimdKTiles}), the grid-coded i-quants ({@link SimdIqGrid}), the
 * block kernels ({@code SimdQ8_0}, {@code SimdQ4_0}, {@code SimdQ4_1}, {@code SimdQ5_0}, {@code SimdQ5_1},
 * {@code SimdIq4Nl}, {@code SimdIq4Xs}) and their float tiles ({@link SimdBlockTiles}, {@link SimdTokenTiles})
 * therefore keep four rules, which make their compiled form independent of the JVM's history:
 * <ol>
 *   <li>A loop-carried vector is only ever the argument of an operation whose receiver is exact
 *       ({@code acc = term.add(acc)}): the library's own species check on the argument ({@code that.check(this)})
 *       then proves its class. Where it must be the receiver (a store, a reduction, a conversion, a subtraction),
 *       {@code v.check(SPECIES)} proves it first; the check costs nothing once compiled.</li>
 *   <li>Every shape change names its target species ({@code convertShape(op, SPECIES, 0)},
 *       {@code reinterpretShape(SPECIES, 0)}), never {@code convert(op, part)} or {@code reinterpretAsInts()},
 *       whose species lookup folds only when a JDK cache entry happens to be filled.</li>
 *   <li>No method of these kernels takes or returns a vector; their helpers take arrays and return scalars, so a
 *       call the JIT does not inline never boxes a vector. The vector-valued helpers below ({@link #pairSum},
 *       {@link #mulAdd}, {@link #widen}, {@link #zero}, {@link #sum}) serve the less common kernels (MXFP4, the
 *       ternary and extension types, the repacked layouts), which {@code JitOrderProbeTest} does not cover.</li>
 *   <li>Every kernel method stays clear of C2's inlining node cutoff. C2 parses each Vector API call into a few
 *       hundred nodes before it intrinsifies it and stops inlining once a compilation passes the cutoff; which
 *       calls lie past it depends on what the profile let the compilation inline first, so a method near the
 *       cutoff loses intrinsics in some JVMs only. A loop body therefore holds one or two product paths (a shift
 *       selects the nibble or bit pair) rather than every sub-block of a group written out, and the float tiles
 *       reduce their accumulators from memory rather than by lane extraction.</li>
 * </ol>
 * {@code JitOrderProbeTest} runs each group of the covered kernels after the other vector code in child JVMs, bounds
 * their allocation and rate against the cold order, and reads the compilation log for any kernel method that
 * reached the node cutoff.
 */
final class SimdI8 {

    /** 16 byte lanes. */
    static final VectorSpecies<Byte> B16 = ByteVector.SPECIES_128;
    /** 8 byte lanes. */
    static final VectorSpecies<Byte> B8 = ByteVector.SPECIES_64;
    /** 4 int lanes: the widening partner of {@link #B16} halves. */
    static final VectorSpecies<Integer> I4 = IntVector.SPECIES_128;
    /** 8 int lanes: the widening partner of {@link #B16}. */
    static final VectorSpecies<Integer> I8 = IntVector.SPECIES_256;
    /** 16 short lanes: the widening partner of {@link #B16} used by the int16 product path. */
    static final VectorSpecies<Short> S16 = ShortVector.SPECIES_256;

    /** Lanes {@code 0..7}: the bit positions of an 8-bit group. */
    static final IntVector RAMP8 = IntVector.fromArray(I8, new int[]{0, 1, 2, 3, 4, 5, 6, 7}, 0);

    /** Lane {@code j} tests bit {@code j} of an 8-bit group: {@code 1 << j}; the fifth-bit mask of the Q5 kernels. */
    static final IntVector BIT8 = IntVector.fromArray(I8, new int[]{1, 2, 4, 8, 16, 32, 64, 128}, 0);

    /** Tokens one interleaved activation tile ({@link TokenTiles}) holds: one int lane per token. */
    static final int TILE_TOKENS = 8;

    private static final ThreadLocal<int[]> ACTS = new ThreadLocal<>();

    /** The per-block int sums the token-tile kernels hand from their integer to their float half. */
    private static final ThreadLocal<int[]> TILE_SUMS = new ThreadLocal<>();

    /** The widened weight-row element pairs of the token-tile kernels. */
    private static final ThreadLocal<int[][]> ROW_PAIRS = new ThreadLocal<>();

    /** The per-block weight-row scales of the token-tile kernels. */
    private static final ThreadLocal<float[][]> ROW_SCALES = new ThreadLocal<>();

    /** The activation vector {@link #ACTS} currently holds. */
    private static final ThreadLocal<CpuVecDot.QuantizedActivations> LAST = new ThreadLocal<>();

    /** The widened activations of every token of one batched product, one row per token. */
    private static final ThreadLocal<int[][]> ACTS_ALL = new ThreadLocal<>();

    /** The short-lane activations of the {@link #S16} product path, and the vector they hold. */
    private static final ThreadLocal<short[]> ACTS_S = new ThreadLocal<>();

    /** The activation vector {@link #ACTS_S} currently holds. */
    private static final ThreadLocal<CpuVecDot.QuantizedActivations> LAST_S = new ThreadLocal<>();

    /** The short-lane activations of every token of one batched product, one row per token. */
    private static final ThreadLocal<short[][]> ACTS_ALL_S = new ThreadLocal<>();

    /** The float accumulators of one register tile of the batched kernels: rows x tokens values. */
    private static final ThreadLocal<float[]> TILE_ACC = new ThreadLocal<>();

    /** The pair sums of the 16-quant activation sums of every token of one batched product. */
    private static final ThreadLocal<short[][]> BSUM_PAIRS = new ThreadLocal<>();

    /** The pair sums of one token's activation sums, for the single-token kernels. */
    private static final ThreadLocal<short[]> BSUM_PAIRS_ONE = new ThreadLocal<>();

    private SimdI8() {
    }

    /**
     * The calling thread's pair sums of the 16-quant activation sums ({@code bsums}) of every token, eight per
     * super-block: each pair is {@code bsums[2k] + bsums[2k+1]}, the two groups one 6-bit Q4_K / Q5_K min
     * covers, and the pair fits a short because a group sum of sixteen int8 quants does. The kernels dot the
     * eight pairs with the eight mins in int lanes, so the mins term costs one vector multiply instead of a
     * sixteen-iteration scalar loop per token and super-block.
     */
    static short[][] bsumPairs(CpuVecDot.QuantizedActivations[] x, int nTokens) {
        short[][] buf = BSUM_PAIRS.get();
        final int perBlock = QK_K / 32;
        final int blocks = x[0].bsums.length / (QK_K / 16);
        if (buf == null || buf.length < nTokens || buf[0].length < blocks * perBlock) {
            buf = new short[Math.max(nTokens, 8)][];
            for (int t = 0; t < buf.length; t++) {
                buf[t] = new short[blocks * perBlock];
            }
            BSUM_PAIRS.set(buf);
        }
        for (int t = 0; t < nTokens; t++) {
            final short[] src = x[t].bsums;
            final short[] dst = buf[t];
            for (int i = 0; i < blocks * perBlock; i++) {
                dst[i] = (short) (src[2 * i] + src[2 * i + 1]);
            }
        }
        return buf;
    }

    /**
     * The calling thread's pair sums of the 16-quant activation sums of one token, {@code 8} per super-block.
     */
    static short[] bsumPairs(CpuVecDot.QuantizedActivations x) {
        final int n = x.bsums.length / 2;
        short[] buf = BSUM_PAIRS_ONE.get();
        if (buf == null || buf.length < n) {
            buf = new short[n];
            BSUM_PAIRS_ONE.set(buf);
        }
        for (int i = 0; i < n; i++) {
            buf[i] = (short) (x.bsums[2 * i] + x.bsums[2 * i + 1]);
        }
        return buf;
    }

    /**
     * {@code sum_{k<8} pairs[po + k] * mins[k]} of one super-block, exact in int: the mins are the eight 6-bit
     * values in {@code sm[8..16)} and each covers two 16-quant group sums.
     */
    static int minsDot(short[] pairs, int po, byte[] sm) {
        final IntVector m = (IntVector) ByteVector.fromArray(B8, sm, 8).convertShape(VectorOperators.B2I, I8, 0);
        final IntVector p = (IntVector) ShortVector.fromArray(ShortVector.SPECIES_128, pairs, po)
                .convertShape(VectorOperators.S2I, I8, 0);
        return m.mul(p).reduceLanes(VectorOperators.ADD);
    }

    /**
     * The activations of one batched product as tiles of {@link #TILE_TOKENS} tokens interleaved by element pairs: in
     * tile {@code t}, pair {@code k} (elements {@code 2k, 2k+1}) of token {@code ti} sits at {@code lanes[t * tileLanes
     * + 16k + 2ti + j]}, so one 16-lane short load holds one element pair of every token of the tile and a weight pair
     * broadcast over it multiplies all eight tokens at once; once the sign-extended low and high halves of each int
     * lane are added, lane {@code ti} carries token {@code ti}'s products. {@code scales[t * tileScales + 8b + ti]} is
     * token {@code ti}'s block-{@code b} scale {@code d}; for Q8_K activations {@code groupSums[t * tilePairs + 8j +
     * ti]} is token {@code ti}'s sum over its {@code j}-th group of {@code groupSpan} quants: with a span of 32 the sum
     * of the 16-quant group sums {@code 2j} and {@code 2j + 1} ({@link #bsumPairs}), eight per super-block, with a span
     * of 16 the 16-quant group sum {@code j} itself, sixteen per super-block. Tokens past the last pad with zero
     * quants, zero scales and zero sums. Immutable once built.
     */
    static final class TokenTiles implements CpuVecDot.TileActivations {
        final short[] lanes;
        final float[] scales;
        final int[] groupSums;
        final int groupSpan;
        final int tokens;
        final int count;
        final int cols;
        final int tileLanes;
        final int tileScales;
        final int tilePairs;

        private TokenTiles(int tokens, int cols, int blocks, int groupSpan) {
            this.tokens = tokens;
            this.cols = cols;
            this.groupSpan = groupSpan;
            this.count = (tokens + TILE_TOKENS - 1) / TILE_TOKENS;
            this.tileLanes = cols * TILE_TOKENS;
            this.tileScales = blocks * TILE_TOKENS;
            this.tilePairs = groupSpan > 0 ? (cols / groupSpan) * TILE_TOKENS : 0;
            this.lanes = new short[Math.multiplyExact(count, tileLanes)];
            this.scales = new float[Math.multiplyExact(count, tileScales)];
            this.groupSums = groupSpan > 0 ? new int[Math.multiplyExact(count, tilePairs)] : null;
        }

        @Override
        public int tokens() {
            return tokens;
        }

        /** Interleaves token {@code t}'s quants and scales; tokens past the last keep their zeros. */
        private void put(int t, CpuVecDot.QuantizedActivations x) {
            final int base = (t / TILE_TOKENS) * tileLanes + 2 * (t % TILE_TOKENS);
            final int sBase = (t / TILE_TOKENS) * tileScales + t % TILE_TOKENS;
            final byte[] qs = x.qs;
            for (int k = 0; k < cols / 2; k++) {
                lanes[base + 16 * k] = qs[2 * k];
                lanes[base + 16 * k + 1] = qs[2 * k + 1];
            }
            final float[] d = x.d;
            for (int b = 0; b < d.length; b++) {
                scales[sBase + TILE_TOKENS * b] = d[b];
            }
            if (groupSums != null) {
                final short[] bs = x.bsums;
                final int pBase = (t / TILE_TOKENS) * tilePairs + t % TILE_TOKENS;
                if (groupSpan == 16) {
                    for (int j = 0; j < bs.length; j++) {
                        groupSums[pBase + TILE_TOKENS * j] = bs[j];
                    }
                } else {
                    for (int j = 0; j < bs.length / 2; j++) {
                        groupSums[pBase + TILE_TOKENS * j] = bs[2 * j] + bs[2 * j + 1];
                    }
                }
            }
        }

        /** The first group sum of tile {@code tile}. */
        int pairBase(int tile) {
            return tile * tilePairs;
        }

        /** The first lane of tile {@code tile}. */
        int laneBase(int tile) {
            return tile * tileLanes;
        }

        /** The first scale of tile {@code tile}. */
        int scaleBase(int tile) {
            return tile * tileScales;
        }
    }

    /**
     * The {@link TokenTiles} of the {@code nTokens} tokens of {@code x}, with group sums over {@code groupSpan}
     * quants (32, 16, or 0 for none; Q8_K activations only), one token per {@link CpuParallel} index (each writes
     * only its own lanes).
     */
    static TokenTiles tokenTiles(CpuVecDot.QuantizedActivations[] x, int nTokens, int groupSpan) {
        if ((groupSpan != 0 && groupSpan != 16 && groupSpan != 32) || (groupSpan != 0 && x[0].bsums == null)) {
            throw new IllegalArgumentException("group sums over " + groupSpan + " quants of " + x[0].type);
        }
        final TokenTiles tiles = new TokenTiles(nTokens, x[0].n, x[0].d.length, groupSpan);
        CpuParallel.parallelFor(nTokens, Math.max(1, (1 << 16) / Math.max(1, x[0].n)), (from, to) -> {
            for (int t = from; t < to; t++) {
                tiles.put(t, x[t]);
            }
        });
        return tiles;
    }

    /**
     * Validates a token range of a prepared layout: it lies inside the layout and starts on a tile.
     *
     * @throws IllegalArgumentException  for a start that is not a multiple of {@link #TILE_TOKENS}
     * @throws IndexOutOfBoundsException for a range outside the layout
     */
    static void checkTileRange(TokenTiles tiles, int tokenStart, int nTokens) {
        if (tokenStart < 0 || nTokens < 0 || tokenStart > tiles.tokens - nTokens) {
            throw new IndexOutOfBoundsException("tokens [" + tokenStart + ", " + ((long) tokenStart + nTokens)
                    + ") exceed the " + tiles.tokens + "-token tiles");
        }
        if (tokenStart % TILE_TOKENS != 0) {
            throw new IllegalArgumentException("token start " + tokenStart + " is not a multiple of the "
                    + TILE_TOKENS + "-token tile");
        }
    }

    /** The calling thread's {@code rows} widened weight rows of {@code pairs} int16 element pairs each. */
    static int[][] rowPairs(int rows, int pairs) {
        int[][] buf = ROW_PAIRS.get();
        if (buf == null || buf.length < rows || buf[0].length < pairs) {
            buf = new int[Math.max(rows, 4)][Math.max(pairs, 512)];
            ROW_PAIRS.set(buf);
        }
        return buf;
    }

    /**
     * Validates the activations a token-tile kernel lays out: {@code nTokens} vectors of {@code type}, all of the
     * first one's length.
     *
     * @throws IllegalArgumentException naming the first token that is null, of another type or of another length
     */
    static void checkTileTokens(CpuVecDot.QuantizedActivations[] x, int nTokens, GGMLType type) {
        for (int t = 0; t < nTokens; t++) {
            if (x[t] == null || x[t].type != type || x[t].n != x[0].n) {
                throw new IllegalArgumentException(type + " tiles need " + type + " activations of one length,"
                        + " token " + t + " is " + x[t]);
            }
        }
    }

    /**
     * Validates a prepared product against weight {@code w}: the layout's row length is {@code w}'s.
     *
     * @throws IllegalArgumentException when the lengths differ
     */
    static void checkTileWidth(CpuTensorRef w, TokenTiles tiles) {
        if (tiles.cols != w.cols()) {
            throw new IllegalArgumentException(w.type() + " row of " + w.cols() + " elements cannot be dotted with "
                    + tiles.cols + " activations");
        }
    }

    /** The calling thread's {@code rows} per-block int arrays (weight sub-block scales) of {@code n} values each. */
    static int[][] rowInts(ThreadLocal<int[][]> slot, int rows, int n) {
        int[][] buf = slot.get();
        if (buf == null || buf.length < rows || buf[0].length < n) {
            buf = new int[Math.max(rows, 4)][Math.max(n, 128)];
            slot.set(buf);
        }
        return buf;
    }

    /** The calling thread's {@code rows} per-block weight-row scale arrays of {@code blocks} values each. */
    static float[][] rowScales(int rows, int blocks) {
        float[][] buf = ROW_SCALES.get();
        if (buf == null || buf.length < rows || buf[0].length < blocks) {
            buf = new float[Math.max(rows, 4)][Math.max(blocks, 64)];
            ROW_SCALES.set(buf);
        }
        return buf;
    }

    /** The calling thread's per-block tile sums, at least {@code n} values long. */
    static int[] tileSums(int n) {
        int[] buf = TILE_SUMS.get();
        if (buf == null || buf.length < n) {
            buf = new int[Math.max(n, 1024)];
            TILE_SUMS.set(buf);
        }
        return buf;
    }

    /** The calling thread's float accumulators of one batched tile, at least {@code n} values long. */
    static float[] tileAccumulators(int n) {
        float[] buf = TILE_ACC.get();
        if (buf == null || buf.length < n) {
            buf = new float[n];
            TILE_ACC.set(buf);
        }
        return buf;
    }

    /**
     * The calling thread's widened activations: the {@code n} quants of {@code qs} sign-extended into int
     * lanes, exact, so the kernels load activation lanes with one {@code fromArray} instead of widening per
     * block.
     */
    static int[] activations(byte[] qs, int n) {
        int[] buf = ACTS.get();
        if (buf == null || buf.length < n) {
            buf = new int[Math.max(n, 1024)];
            ACTS.set(buf);
        }
        for (int i = 0; i < n; i++) {
            buf[i] = qs[i];
        }
        return buf;
    }

    /**
     * The calling thread's widened activations of {@code x}, widened only when it is not the vector widened
     * last: one quantization serves every row and every token pass of one product, so a row loop does no
     * widening work at all.
     */
    static int[] activations(CpuVecDot.QuantizedActivations x) {
        int[] buf = ACTS.get();
        if (LAST.get() == x && buf != null) {
            return buf;
        }
        buf = activations(x.qs, x.n);
        LAST.set(x);
        return buf;
    }

    /**
     * The calling thread's widened activations of the {@code nTokens} tokens of one product, one row per
     * token: the batched kernels load every token's lanes from {@code [t]} with no widening in their row loop.
     */
    static int[][] activationsAll(CpuVecDot.QuantizedActivations[] x, int nTokens) {
        final int cols = x[0].n;
        int[][] buf = ACTS_ALL.get();
        if (buf == null || buf.length < nTokens || buf[0].length < cols) {
            buf = new int[Math.max(nTokens, 8)][];
            for (int t = 0; t < buf.length; t++) {
                buf[t] = new int[cols];
            }
            ACTS_ALL.set(buf);
        }
        for (int t = 0; t < nTokens; t++) {
            final byte[] qs = x[t].qs;
            final int[] a = buf[t];
            for (int i = 0; i < cols; i++) {
                a[i] = qs[i];
            }
        }
        return buf;
    }

    /**
     * The calling thread's short-lane activations of {@code x}, widened only when it is not the one widened
     * last: the {@code S16} product path loads them with one {@code fromArray} and no conversion.
     */
    static short[] activationsShort(CpuVecDot.QuantizedActivations x) {
        short[] buf = ACTS_S.get();
        if (LAST_S.get() == x && buf != null) {
            return buf;
        }
        final int n = x.n;
        if (buf == null || buf.length < n) {
            buf = new short[Math.max(n, 1024)];
            ACTS_S.set(buf);
        }
        final byte[] qs = x.qs;
        for (int i = 0; i < n; i++) {
            buf[i] = qs[i];
        }
        LAST_S.set(x);
        return buf;
    }

    /** The calling thread's short-lane activations of the {@code nTokens} tokens, one row per token. */
    static short[][] activationsAllShort(CpuVecDot.QuantizedActivations[] x, int nTokens) {
        final int cols = x[0].n;
        short[][] buf = ACTS_ALL_S.get();
        if (buf == null || buf.length < nTokens || buf[0].length < cols) {
            buf = new short[Math.max(nTokens, 8)][];
            for (int t = 0; t < buf.length; t++) {
                buf[t] = new short[cols];
            }
            ACTS_ALL_S.set(buf);
        }
        for (int t = 0; t < nTokens; t++) {
            final byte[] qs = x[t].qs;
            final short[] a = buf[t];
            for (int i = 0; i < cols; i++) {
                a[i] = qs[i];
            }
        }
        return buf;
    }

    /**
     * {@code acc} plus the pairwise sums of the sixteen int16 products of {@code p}: the products live in the
     * low and high halves of the eight int lanes after the bitcast, so the sign-extended halves are the pairs
     * {@code p[2k]} and {@code p[2k+1]}. Exact: every product and every pair sum is an int.
     */
    static IntVector pairSum(ShortVector p, IntVector acc) {
        final IntVector lanes = (IntVector) p.reinterpretShape(I8, 0);
        return acc.add(lanes.lanewise(VectorOperators.LSHL, 16).lanewise(VectorOperators.ASHR, 16)
                .add(lanes.lanewise(VectorOperators.ASHR, 16)));
    }

    /**
     * {@code acc += sum_p widen(w, p) * widen(a, p)} over the {@code w.length() / is.length()} parts of the
     * byte vectors: the exact int widening dot of two byte vectors of equal length, accumulated into
     * {@code acc} lane by lane.
     */
    static IntVector mulAdd(ByteVector w, ByteVector a, IntVector acc, VectorSpecies<Integer> is) {
        IntVector r = acc;
        for (int p = 0, parts = w.length() / is.length(); p < parts; p++) {
            r = r.add(widen(w, p, is).mul(widen(a, p, is)));
        }
        return r;
    }

    /** Part {@code part} of the byte-to-int widening of {@code w}: every byte lane as its own int lane. */
    static IntVector widen(ByteVector w, int part, VectorSpecies<Integer> is) {
        return (IntVector) w.convertShape(VectorOperators.B2I, is, part);
    }

    /** {@code sum_i w[off + i] * y[yOff + i]} for 32 int8 weights and int8 activations, exact in int. */
    static int dot32(byte[] w, int off, byte[] y, int yOff) {
        IntVector acc = IntVector.zero(I8);
        for (int h = 0; h < 2; h++) {
            acc = mulAdd(ByteVector.fromArray(B16, w, off + 16 * h),
                    ByteVector.fromArray(B16, y, yOff + 16 * h), acc, I8);
        }
        return sum(acc);
    }

    /** The additive identity of the int widening. */
    static IntVector zero(VectorSpecies<Integer> is) {
        return IntVector.zero(is);
    }

    /** Horizontal int sum of one accumulator vector. */
    static int sum(IntVector acc) {
        return acc.reduceLanes(VectorOperators.ADD);
    }
}
