package com.eonmux.neuralfree.gguf.mm;

/**
 * The row reorders and position tables of the SAM backbone's windowed attention (models/deepseekocr.cpp
 * {@code window_partition}, {@code window_unpartition} and {@code get_rel_pos}). Grids are raster-order rows of
 * {@code n_embd} values. Static methods only.
 */
final class SamWindows {

    private SamWindows() {
    }

    /**
     * The {@code g x g} grid zero-padded at the right and bottom to a multiple of {@code win} and cut into
     * {@code win x win} windows: window {@code wy * nw + wx} ({@code nw = ceil(g / win)}) holds its rows in raster
     * order, the padding rows zero.
     *
     * @return {@code [nw * nw][win * win][n_embd]} rows
     */
    static float[] partition(float[] x, int nEmbd, int g, int win) {
        int nw = (g + win - 1) / win;
        float[] out = new float[Math.multiplyExact(nw * nw * win * win, nEmbd)];
        forEachCell(g, win, (src, dst) -> System.arraycopy(x, src * nEmbd, out, dst * nEmbd, nEmbd));
        return out;
    }

    /** The inverse of {@link #partition}: the windows back to the {@code g x g} grid, the padding dropped. */
    static float[] unpartition(float[] windows, int nEmbd, int g, int win) {
        float[] out = new float[g * g * nEmbd];
        forEachCell(g, win, (src, dst) -> System.arraycopy(windows, dst * nEmbd, out, src * nEmbd, nEmbd));
        return out;
    }

    /**
     * The source row of every window row of {@link #partition} over {@code g} grid rows: window row {@code dst}
     * reads grid row {@code src} (in raster order per window), the padding rows read the grid row count (a pad
     * row a device gather appends).
     */
    static int[] partitionIndex(int g, int win) {
        int nw = (g + win - 1) / win;
        int[] idx = new int[nw * nw * win * win];
        java.util.Arrays.fill(idx, g * g);
        forEachCell(g, win, (src, dst) -> idx[dst] = src);
        return idx;
    }

    /** The source window row of every grid row of {@link #unpartition} over {@code g} grid rows. */
    static int[] unpartitionIndex(int g, int win) {
        int nw = (g + win - 1) / win;
        int[] idx = new int[g * g];
        forEachCell(g, win, (src, dst) -> idx[src] = dst);
        return idx;
    }

    /** A grid row and its window row. */
    private interface CellMap {
        void map(int gridRow, int windowRow);
    }

    private static void forEachCell(int g, int win, CellMap m) {
        int nw = (g + win - 1) / win;
        for (int y = 0; y < g; y++) {
            for (int x = 0; x < g; x++) {
                int window = (y / win) * nw + x / win;
                m.map(y * g + x, window * win * win + (y % win) * win + x % win);
            }
        }
    }

    /**
     * The relative-position rows of an attention over {@code side} positions per axis: {@code 2 * side - 1} rows of
     * {@code d} values, row {@code q - k + side - 1} the bias of query {@code q} over key {@code k}. A table of
     * another length is resized along its rows by the half-pixel linear kernel of {@code ggml_interpolate}
     * ({@code GGML_SCALE_MODE_BILINEAR} over a height of one), computed in float as the kernel does.
     *
     * @param table {@code [length][d]} rows
     */
    static float[] relTable(float[] table, int d, int side) {
        int length = table.length / d;
        int target = 2 * side - 1;
        if (length == target) {
            return table;
        }
        float sf = (float) target / length;
        float[] out = new float[target * d];
        for (int i = 0; i < target; i++) {
            float x = (i + 0.5f) / sf - 0.5f;
            int x0 = (int) Math.floor(x);
            int x1 = Math.max(0, Math.min(x0 + 1, length - 1));
            x0 = Math.max(0, Math.min(x0, length - 1));
            float dx = Math.max(0f, Math.min(x - x0, 1f));
            for (int c = 0; c < d; c++) {
                out[i * d + c] = table[x0 * d + c] * (1 - dx) + table[x1 * d + c] * dx;
            }
        }
        return out;
    }
}
