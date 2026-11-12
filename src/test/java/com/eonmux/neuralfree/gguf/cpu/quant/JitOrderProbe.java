package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.quant.BlockLayout;

import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Times one group of weight kernels ({@link #GROUPS}) in the JVM the tests launch, either first ({@code cold}) or
 * after every other Vector API kernel has run ({@code polluted}): the vector attention kernels and the single-token,
 * one-tile and batched products of every supported weight type outside the group, the order a model that mixes weight
 * types or a process that serves several models runs them in. A cold JVM waits as long as a polluted one runs the other
 * kernels, so the timed phases of JVMs started together overlap. Output lines are {@code <type>.tiles=<ns>} and
 * {@code <type>.decode=<ns>}: the best time of one token-tile product and of one single-token row loop over the
 * second half of the timed repetitions, single-threaded; and {@code <type>.tiles.bytes=<bytes>} and
 * {@code <type>.decode.bytes=<bytes>}: the most heap the calling thread allocated during one of those products. A
 * kernel C2 compiled to its vector instructions allocates nothing per product, while one whose Vector API calls
 * fell back to the library's Java code allocates a vector object per operation, megabytes per product, whatever
 * the machine's load.
 */
public final class JitOrderProbe {

    /**
     * The kernel groups the probe times, by name: the int16 K-quant tiles, the int16 tiles of the grid-coded
     * i-quants and the float block tiles.
     */
    static final Map<String, GGMLType[]> GROUPS = groups();

    private static final int ROWS = 512;
    private static final int COLS = 2560;
    private static final int TOKENS = 32;
    /** Tokens of a batch of one tile: the float kernels multiply it in a form of its own. */
    private static final int ONE_TILE = 8;
    private static final int REPS = 40;
    private static final long POLLUTE_NANOS = 1_500_000_000L;

    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    private static volatile float sink;

    private JitOrderProbe() {
    }

    private static Map<String, GGMLType[]> groups() {
        final Map<String, GGMLType[]> g = new LinkedHashMap<>();
        g.put("kquants", new GGMLType[]{GGMLType.Q4_K, GGMLType.Q6_K, GGMLType.Q5_K, GGMLType.Q3_K, GGMLType.Q2_K});
        g.put("grids", new GGMLType[]{GGMLType.IQ2_XXS, GGMLType.IQ2_XS, GGMLType.IQ2_S, GGMLType.IQ3_XXS,
                GGMLType.IQ3_S});
        g.put("blocks", new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
                GGMLType.IQ4_NL, GGMLType.IQ4_XS});
        return Collections.unmodifiableMap(g);
    }

    /** {@code args}: the order ({@code cold} or {@code polluted}) and the name of the group to time. */
    public static void main(String[] args) {
        if (args.length != 2 || !GROUPS.containsKey(args[1])) {
            throw new IllegalArgumentException("usage: JitOrderProbe cold|polluted " + GROUPS.keySet());
        }
        final GGMLType[] group = GROUPS.get(args[1]);
        if (args[0].equals("polluted")) {
            pollute(group);
        } else {
            idle();
        }
        final Product[] timed = new Product[group.length];
        for (int i = 0; i < group.length; i++) {
            timed[i] = new Product(group[i], new Random(11 + i));
        }
        final long[][] tiles = new long[group.length][REPS];
        final long[][] decode = new long[group.length][REPS];
        final long[][] tilesBytes = new long[group.length][REPS];
        final long[][] decodeBytes = new long[group.length][REPS];
        final long thread = Thread.currentThread().getId();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < timed.length; i++) {
                final long a0 = THREADS.getThreadAllocatedBytes(thread);
                tiles[i][r] = timed[i].tiles();
                final long a1 = THREADS.getThreadAllocatedBytes(thread);
                decode[i][r] = timed[i].decode();
                decodeBytes[i][r] = THREADS.getThreadAllocatedBytes(thread) - a1;
                tilesBytes[i][r] = a1 - a0;
            }
        }
        for (int i = 0; i < group.length; i++) {
            System.out.println(group[i] + ".tiles=" + Arrays.stream(tiles[i], REPS / 2, REPS).min().getAsLong());
            System.out.println(group[i] + ".decode=" + Arrays.stream(decode[i], REPS / 2, REPS).min().getAsLong());
            System.out.println(group[i] + ".tiles.bytes="
                    + Arrays.stream(tilesBytes[i], REPS / 2, REPS).max().getAsLong());
            System.out.println(group[i] + ".decode.bytes="
                    + Arrays.stream(decodeBytes[i], REPS / 2, REPS).max().getAsLong());
        }
        System.out.println("sink=" + sink);
    }

    /**
     * Waits as long as {@link #pollute} runs, so the timed phases of cold and polluted JVMs started together overlap
     * and share the machine's load.
     */
    private static void idle() {
        try {
            Thread.sleep(POLLUTE_NANOS / 1_000_000L);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted before timing", e);
        }
    }

    /**
     * Runs the other kernels: every other vector kernel's products (batched, a batch of one tile and single-token)
     * and the vector attention, for 1.5 s.
     */
    private static void pollute(GGMLType[] group) {
        final List<Product> others = new ArrayList<>();
        final List<GGMLType> timed = Arrays.asList(group);
        final Random rnd = new Random(5);
        for (GGMLType t : CpuSimd.supportedTypes()) {
            if (!timed.contains(t) && COLS % BlockLayout.of(t).blockElements() == 0) {
                others.add(new Product(t, rnd));
            }
        }
        final CpuAttnKernels attn = CpuSimd.vectorAttention();
        final int dK = 128;
        final float[] qT = SimdVecDotTest.gaussian(dK * 32, rnd);
        final float[] keys = SimdVecDotTest.gaussian(8 * dK, rnd);
        final float[] v = SimdVecDotTest.gaussian(dK, rnd);
        final float[] dots = new float[8 * 32];
        final float[] acc = new float[32 * dK];
        final long end = System.nanoTime() + POLLUTE_NANOS;
        while (System.nanoTime() < end) {
            for (Product p : others) {
                p.tiles();
                p.oneTile();
                p.decode();
            }
            for (int rep = 0; rep < 64; rep++) {
                attn.dots(qT, 32, 32, keys, 0, dK, 8, dK, dots, 0);
                attn.dots(qT, 8, 4, keys, 0, dK, 8, dK, dots, 0);
                attn.dots(qT, 32, 28, keys, 0, dK, 5, dK, dots, 0);
                for (int g = 0; g < 32; g++) {
                    attn.accumulate(v, 0, dots[g] * 1e-3f, acc, g * dK, dK);
                }
            }
        }
        sink += acc[7];
    }

    /** One weight type's batched and single-token products over fixed random weights and activations. */
    private static final class Product {
        private final CpuVecDot kernel;
        private final CpuTensorRef w;
        private final CpuVecDot.QuantizedActivations[] x;
        private final CpuVecDot.TileActivations prepared;
        private final float[] y = new float[ROWS * TOKENS];

        Product(GGMLType t, Random rnd) {
            kernel = CpuSimd.vecDot(t);
            w = CpuTensorRef.of(t, SimdVecDotTest.randomRows(t, ROWS, COLS, rnd), ROWS, COLS);
            x = new CpuVecDot.QuantizedActivations[TOKENS];
            for (int i = 0; i < TOKENS; i++) {
                x[i] = CpuActivationQuantizer.quantizeFor(t, SimdVecDotTest.gaussian(COLS, rnd), 0, COLS);
            }
            prepared = kernel.prepareTiles(x, TOKENS);
        }

        long tiles() {
            final long t0 = System.nanoTime();
            kernel.dotTiles(w, 0, ROWS, prepared, x, 0, TOKENS, y, 0, ROWS);
            final long ns = System.nanoTime() - t0;
            sink += y[ROWS + 3];
            return ns;
        }

        void oneTile() {
            kernel.dotTiles(w, 0, ROWS, prepared, x, 0, ONE_TILE, y, 0, ROWS);
            sink += y[ROWS + 3];
        }

        long decode() {
            final long t0 = System.nanoTime();
            kernel.dotRows(w, 0, ROWS, x[1], y, 0);
            final long ns = System.nanoTime() - t0;
            sink += y[5];
            return ns;
        }
    }
}
