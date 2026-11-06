package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;
import java.util.function.IntToLongFunction;
import java.util.function.IntUnaryOperator;

/**
 * The quantizer-side lattice of one grid-search IQ type, the state {@code iq2xs_init_impl} and
 * {@code iq3xs_init_impl} build in llama.cpp ggml-quants.c (:2869-3274, :3719-3918): every grid point as a
 * vector of levels ({@code kgrid_*}, a level {@code l} standing for the odd value {@code 2l+1}), the map
 * from a packed level vector to its grid index, and for every vector off the grid the list of nearest grid
 * points the encoders choose from.
 * <p>
 * The levels are recovered from the dequantization grids of {@link QuantTables}: the distinct lane values
 * of a grid, in ascending order, are its levels 0, 1, 2, ..., which reproduces the reference's
 * {@code kgrid_2bit_256}, {@code kgrid_2bit_512}, {@code kgrid_2bit_1024}, {@code kgrid_1bit_2048},
 * {@code kgrid_256} and {@code kgrid_512} tables exactly. A neighbour list holds the grid points at the
 * {@code nwant} smallest squared distances from the vector, ordered by distance then grid index, as the
 * reference's sort of {@code (d2, index)} pairs leaves them. Lists exist for the vectors the encoders can
 * form (every lane in {@code [0, maxLevel]}); the reference also fills lists for 2-bit vectors holding
 * level 3, which no encoder produces.
 * <p>
 * Immutable and thread-safe; each grid is built once, on first use.
 */
final class IqGrid {

    private final int size;
    private final int lanes;
    private final int bits;
    private final byte[] levels;
    private final int[] keys;
    private final int[] map;
    private final int[][] neighbours;

    private IqGrid(int size, int lanes, int bits, byte[] levels, int nwant) {
        this.size = size;
        this.lanes = lanes;
        this.bits = bits;
        this.levels = levels;
        this.keys = new int[size];
        this.map = new int[1 << (bits * lanes)];
        Arrays.fill(map, -1);
        for (int g = 0; g < size; g++) {
            int u = 0;
            for (int j = 0; j < lanes; j++) {
                u |= levels[g * lanes + j] << (bits * j);
            }
            if (map[u] >= 0) {
                throw new IllegalStateException("grid points " + map[u] + " and " + g + " share levels " + u);
            }
            keys[g] = u;
            map[u] = g;
        }
        this.neighbours = new int[map.length][];
        int maxLevel = maxLevel();
        for (int u = 0; u < map.length; u++) {
            if (map[u] < 0 && reachable(u, maxLevel)) {
                neighbours[u] = nearest(u, nwant);
            }
        }
    }

    /** The grid of {@code type}: IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S / IQ1_M (one shared grid), IQ3_XXS or IQ3_S. */
    static IqGrid of(GGMLType type) {
        switch (type) {
            case IQ2_XXS:
                return Iq2Xxs.GRID;
            case IQ2_XS:
                return Iq2Xs.GRID;
            case IQ2_S:
                return Iq2S.GRID;
            case IQ1_S:
            case IQ1_M:
                return Iq1.GRID;
            case IQ3_XXS:
                return Iq3Xxs.GRID;
            case IQ3_S:
                return Iq3S.GRID;
            default:
                throw new IllegalArgumentException(type + " has no quantization grid");
        }
    }

    /** Grid points. */
    int size() {
        return size;
    }

    /** Levels per grid point: 8 for the IQ2 and IQ1 grids, 4 for the IQ3 grids. */
    int lanes() {
        return lanes;
    }

    /** Bits per level in a packed vector: 2 for the IQ2 and IQ1 grids, 3 for the IQ3 grids. */
    int bits() {
        return bits;
    }

    /** Level of lane {@code j} of grid point {@code g}. */
    int level(int g, int j) {
        return levels[g * lanes + j];
    }

    /** The packed level vector of grid point {@code g}. */
    int key(int g) {
        return keys[g];
    }

    /** The grid index of packed level vector {@code u}, or -1 when {@code u} is off the grid. */
    int indexOf(int u) {
        return map[u];
    }

    /**
     * The neighbour list of the off-grid vector {@code u}, in reference order.
     *
     * @throws IllegalArgumentException when {@code u} is on the grid or holds a level above the grid's
     */
    int[] neighbours(int u) {
        int[] n = neighbours[u];
        if (n == null) {
            throw new IllegalArgumentException("level vector " + u + " has no neighbour list");
        }
        return n;
    }

    private int maxLevel() {
        int max = 0;
        for (byte l : levels) {
            max = Math.max(max, l);
        }
        return max;
    }

    private boolean reachable(int u, int maxLevel) {
        int mask = (1 << bits) - 1;
        for (int j = 0; j < lanes; j++) {
            if (((u >>> (bits * j)) & mask) > maxLevel) {
                return false;
            }
        }
        return true;
    }

    /** The grid points at the {@code nwant} smallest squared distances from {@code u}, by distance then index. */
    private int[] nearest(int u, int nwant) {
        int mask = (1 << bits) - 1;
        int[] d2 = new int[size];
        for (int g = 0; g < size; g++) {
            int d = 0;
            for (int j = 0; j < lanes; j++) {
                int diff = (2 * level(g, j) + 1) - (2 * ((u >>> (bits * j)) & mask) + 1);
                d += diff * diff;
            }
            d2[g] = d;
        }
        int[] out = new int[size];
        int n = 0;
        int previous = -1;
        for (int have = 0; have < nwant; have++) {
            int next = Integer.MAX_VALUE;
            for (int d : d2) {
                if (d > previous && d < next) {
                    next = d;
                }
            }
            if (next == Integer.MAX_VALUE) {
                break;
            }
            for (int g = 0; g < size; g++) {
                if (d2[g] == next) {
                    out[n++] = g;
                }
            }
            previous = next;
        }
        return Arrays.copyOf(out, n);
    }

    /**
     * The levels of a dequantization grid: lane {@code j} of entry {@code g} is the rank of its value among
     * the grid's distinct lane values.
     */
    private static byte[] levelsOf(long[] packed, int lanes, int laneBits, boolean signed, int expectedLevels) {
        int[] values = new int[packed.length * lanes];
        for (int g = 0; g < packed.length; g++) {
            for (int j = 0; j < lanes; j++) {
                int v = (int) ((packed[g] >>> (laneBits * j)) & ((1L << laneBits) - 1));
                values[g * lanes + j] = signed ? (byte) v : v;
            }
        }
        int[] distinct = Arrays.stream(values).distinct().sorted().toArray();
        if (distinct.length != expectedLevels) {
            throw new IllegalStateException("grid holds " + distinct.length + " lane values, expected "
                    + expectedLevels);
        }
        byte[] levels = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            levels[i] = (byte) Arrays.binarySearch(distinct, values[i]);
        }
        return levels;
    }

    private static IqGrid build64(int size, IntToLongFunction entry, boolean signed, int nwant) {
        long[] packed = new long[size];
        for (int g = 0; g < size; g++) {
            packed[g] = entry.applyAsLong(g);
        }
        return new IqGrid(size, 8, 2, levelsOf(packed, 8, 8, signed, 3), nwant);
    }

    private static IqGrid build32(int size, IntUnaryOperator entry, int nwant) {
        long[] packed = new long[size];
        for (int g = 0; g < size; g++) {
            packed[g] = entry.applyAsInt(g) & 0xFFFFFFFFL;
        }
        return new IqGrid(size, 4, 3, levelsOf(packed, 4, 8, false, 8), nwant);
    }

    private static final class Iq2Xxs {
        static final IqGrid GRID = build64(256, QuantTables::iq2xxsGrid, false, 2);
    }

    private static final class Iq2Xs {
        static final IqGrid GRID = build64(512, QuantTables::iq2xsGrid, false, 2);
    }

    private static final class Iq2S {
        static final IqGrid GRID = build64(1024, QuantTables::iq2sGrid, false, 1);
    }

    private static final class Iq1 {
        static final IqGrid GRID = build64(2048, QuantTables::iq1sGrid, true, 3);
    }

    private static final class Iq3Xxs {
        static final IqGrid GRID = build32(256, QuantTables::iq3xxsGrid, 2);
    }

    private static final class Iq3S {
        static final IqGrid GRID = build32(512, QuantTables::iq3sGrid, 3);
    }
}
