package com.eonmux.neuralfree.gguf.cpu.quant;

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
 * as the scalar {@link Q8VecDot#dotI8} in any lane order. The species are fixed at 128-bit and 256-bit, which
 * the JIT splits or widens to whatever the machine has, so one kernel serves AVX, AVX2, AVX-512 and NEON.
 * <p>
 * Vectors load straight from the packed row arrays the row loops hand to the kernels; the int16 product path
 * ({@link #S16}, {@link #pairSum}) widens the sixteen int8 products of a byte vector into short lanes and
 * folds them pairwise into the int accumulator. Shifts run through {@code lanewise} of {@code SHL}, {@code
 * LSHR} and {@code ASHR}.
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

    /** Lanes {@code 0..7}, the bit positions of one 8-byte Q5 group. */
    static final IntVector RAMP8 = IntVector.fromArray(I8, new int[]{0, 1, 2, 3, 4, 5, 6, 7}, 0);

    /** Lane {@code j} tests bit {@code j} of an 8-bit group: {@code 1 << j}. */
    private static final IntVector BIT8 = IntVector.fromArray(I8, new int[]{1, 2, 4, 8, 16, 32, 64, 128}, 0);

    /** Broadcast {@code 1}, the saturating numerator of {@link #fifthBits}. */
    private static final IntVector ONE = IntVector.broadcast(I8, 1);

    private static final ThreadLocal<int[]> ACTS = new ThreadLocal<>();

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
        return sum(m.mul(p));
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

    /**
     * The fifth bits of a Q5 block, as {@code 0} or {@code 16} per lane, from the 32-bit {@code qh} word
     * broadcast over every lane: lane {@code j} of group {@code p} tests bit {@code 8p + j}, so the group
     * shifts {@code qh} by the constant {@code 8p} and masks with {@link #BIT8}, saturates the masked power
     * of two to one and scales it to the fifth bit's place.
     * <p>
     * Every operand here is a constant or a lane-wise arithmetic, the forms the JIT intrinsifies. A shift by
     * a vector of per-lane counts ({@code lanewise(LSHR, ramp.add(8p))}), which reads more directly, is not:
     * it ran the allocating template, 5120 bytes per 2560-element row.
     */
    static IntVector fifthBits(IntVector qh, int p) {
        return qh.lanewise(VectorOperators.LSHR, 8 * p)
                .and(BIT8)
                .lanewise(VectorOperators.MIN, ONE)
                .lanewise(VectorOperators.LSHL, 4);
    }
}
