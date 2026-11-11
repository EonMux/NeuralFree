package com.eonmux.neuralfree.gguf.cpu.quant;

import java.util.Objects;

/**
 * The row-chunk x token-panel tasks of one {@link CpuMatVec} product. A single-token product splits only its rows,
 * into the chunks a {@link com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel} loop over them would run. A batched
 * product also splits its tokens into panels whose laid-out activations hold about {@value #PANEL_ELEMENTS}
 * elements (256 KiB of the int16 kernels' two-byte lanes, 512 KiB of the Q8_0 kernel's float lanes: the kernel's
 * {@link Tiling}), so a task's activations stay in the core's L2 cache while its weight rows stream through, where
 * one task over every token re-reads all of them for each weight row; its row chunks hold at most
 * {@value #CHUNK_BYTES} weight bytes and are sized so the grid has about {@value #TASKS_PER_WORKER} tasks per
 * worker. Panels and chunks are whole multiples of the kernel's register tile, so no task splits one. Tasks are
 * numbered row-chunk major, so the consecutive tasks one worker runs reuse the same weight rows across panels.
 * Every (row, token) output is the same dot product whatever the grid, so the grid changes speed only. Immutable.
 */
final class CpuTaskGrid {

    /** Multiply-accumulates per task below which rows are not split further. */
    static final long MIN_TASK_WORK = 1L << 15;

    /** Laid-out activation elements one token panel of a batched product aims at. */
    static final long PANEL_ELEMENTS = 128L << 10;

    /** Weight bytes one row chunk of a batched product keeps at most, so it stays cached across panels. */
    static final long CHUNK_BYTES = 512L << 10;

    /** Tasks per worker a product aims at, for balance between unequal cores. */
    static final int TASKS_PER_WORKER = 4;

    /**
     * Laid-out activation bytes the prepared layout of one token slab of a batched product holds at most: a product
     * over more tokens runs slab after slab, so its layout memory does not grow with the batch.
     */
    static final long SLAB_BYTES = 32L << 20;

    /**
     * The shape a kernel's batched product imposes on the grid ({@link CpuVecDot#tileTokens},
     * {@link CpuVecDot#tileRows}, {@link CpuVecDot#tilePanelTokens}, {@link CpuVecDot#tileLaneBytes}): the token
     * alignment of its prepared layout, the rows and tokens of its register tile, and the bytes one activation
     * element takes in the layout. Immutable.
     */
    static final class Tiling {

        /** A kernel without a prepared layout: any token start, the default two-row by eight-token tile. */
        static final Tiling UNPREPARED = new Tiling(1, 2, 8, 2);

        final int align;
        final int rows;
        final int panelTokens;
        final int laneBytes;

        private Tiling(int align, int rows, int panelTokens, int laneBytes) {
            this.align = align;
            this.rows = rows;
            this.panelTokens = panelTokens;
            this.laneBytes = laneBytes;
        }

        /**
         * The tiling of a product whose layout starts on multiples of {@code align} tokens and whose register tile is
         * {@code rows} x {@code panelTokens}, with {@code laneBytes} bytes per laid-out activation element.
         *
         * @throws IllegalArgumentException for non-positive sizes
         */
        static Tiling of(int align, int rows, int panelTokens, int laneBytes) {
            if (align < 1 || rows < 1 || panelTokens < 1 || laneBytes < 1) {
                throw new IllegalArgumentException("tiling of align " + align + ", register tile " + rows + " x "
                        + panelTokens + ", " + laneBytes + " B/lane");
            }
            return new Tiling(align, rows, panelTokens, laneBytes);
        }

        /** The tiling {@code dot}'s prepared product imposes. */
        static Tiling of(CpuVecDot dot) {
            return of(dot.tileTokens(), dot.tileRows(), dot.tilePanelTokens(), dot.tileLaneBytes());
        }

        /** A tiling every product of a group satisfies: both alignments and tiles, the wider lane. */
        Tiling join(Tiling other) {
            return of(lcm(align, other.align), lcm(rows, other.rows), lcm(panelTokens, other.panelTokens),
                    Math.max(laneBytes, other.laneBytes));
        }

        @Override
        public String toString() {
            return "Tiling{align " + align + ", " + rows + " x " + panelTokens + ", " + laneBytes + " B/lane}";
        }
    }

    private final int rowCount;
    private final int nTokens;
    private final int rowChunk;
    private final int panel;
    private final int panels;
    private final int chunks;

    private CpuTaskGrid(int rowCount, int nTokens, int rowChunk, int panel) {
        this.rowCount = rowCount;
        this.nTokens = nTokens;
        this.rowChunk = rowChunk;
        this.panel = panel;
        this.panels = (nTokens + panel - 1) / panel;
        this.chunks = (rowCount + rowChunk - 1) / rowChunk;
    }

    /**
     * The grid of a product of {@code rowCount} rows of {@code cols} columns ({@code rowBytes} each) with
     * {@code nTokens} tokens over {@code workers} workers, panels and chunks shaped by {@code tiling}.
     *
     * @throws IllegalArgumentException for non-positive sizes
     */
    static CpuTaskGrid of(int rowCount, int nTokens, int cols, long rowBytes, Tiling tiling, int workers) {
        Objects.requireNonNull(tiling, "tiling");
        if (rowCount < 1 || nTokens < 1 || cols < 1 || rowBytes < 1 || workers < 1) {
            throw new IllegalArgumentException("task grid of " + rowCount + " rows x " + nTokens + " tokens x "
                    + cols + " columns (" + rowBytes + " B/row), " + tiling + ", " + workers + " workers");
        }
        long maxChunks = (long) TASKS_PER_WORKER * workers;
        if (nTokens == 1) {
            long grain = Math.max(1L, MIN_TASK_WORK / cols);
            long rows = Math.max(grain, (rowCount + maxChunks - 1) / maxChunks);
            return new CpuTaskGrid(rowCount, 1, (int) Math.min(rowCount, rows), 1);
        }
        int panel = panelTokens(cols, tiling);
        if (panel >= nTokens) {
            panel = nTokens;
        }
        int panels = (nTokens + panel - 1) / panel;
        long wantedChunks = Math.max(1L, (maxChunks + panels - 1) / panels);
        long rowsForTasks = (rowCount + wantedChunks - 1) / wantedChunks;
        long rowsForCache = Math.max(1L, CHUNK_BYTES / rowBytes);
        long rowsForWork = Math.max(1L, MIN_TASK_WORK / ((long) cols * panel));
        long rows = Math.max(rowsForWork, Math.min(rowsForTasks, rowsForCache));
        rows = Math.min(rowCount, (rows + tiling.rows - 1) / tiling.rows * tiling.rows);
        return new CpuTaskGrid(rowCount, nTokens, (int) rows, panel);
    }

    /**
     * Tokens of one slab of a batched product over {@code cols} columns shaped by {@code tiling}: whole panels whose
     * laid-out activations stay within {@value #SLAB_BYTES} bytes, at least one panel. Slab starts are then
     * multiples of the panel size, so slabbing changes neither the panels nor a value.
     *
     * @throws IllegalArgumentException for a non-positive column count
     */
    static int slabTokens(int cols, Tiling tiling) {
        Objects.requireNonNull(tiling, "tiling");
        if (cols < 1) {
            throw new IllegalArgumentException("token slab of " + cols + " columns, " + tiling);
        }
        int panel = panelTokens(cols, tiling);
        long fit = SLAB_BYTES / ((long) tiling.laneBytes * cols);
        return (int) Math.min(Integer.MAX_VALUE - panel, Math.max(panel, fit / panel * panel));
    }

    /**
     * Tokens of one panel: a multiple of both the alignment and the register tile, near {@value #PANEL_ELEMENTS}
     * laid-out elements.
     */
    private static int panelTokens(int cols, Tiling tiling) {
        int quantum = lcm(tiling.align, tiling.panelTokens);
        long fit = PANEL_ELEMENTS / cols;
        return (int) Math.max(quantum, (fit / quantum) * quantum);
    }

    /** The least common multiple of two positive token alignments. */
    static int lcm(int a, int b) {
        int x = a;
        int y = b;
        while (y != 0) {
            int r = x % y;
            x = y;
            y = r;
        }
        return Math.multiplyExact(a / x, b);
    }

    /** Number of tasks. */
    int tasks() {
        return chunks * panels;
    }

    /** First row (relative to the product's first row) of {@code task}. */
    int rowStart(int task) {
        return (task / panels) * rowChunk;
    }

    /** Rows of {@code task}. */
    int rows(int task) {
        return Math.min(rowChunk, rowCount - rowStart(task));
    }

    /** First token of {@code task}, a multiple of the panel size. */
    int tokenStart(int task) {
        return (task % panels) * panel;
    }

    /** Tokens of {@code task}. */
    int tokens(int task) {
        return Math.min(panel, nTokens - tokenStart(task));
    }

    @Override
    public String toString() {
        return "CpuTaskGrid{" + chunks + " row chunk(s) of " + rowChunk + " x " + panels + " panel(s) of " + panel
                + " token(s)}";
    }
}
