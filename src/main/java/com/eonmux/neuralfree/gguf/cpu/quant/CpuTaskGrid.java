package com.eonmux.neuralfree.gguf.cpu.quant;

/**
 * The row-chunk x token-panel tasks of one {@link CpuMatVec} product. A single-token product splits only its rows,
 * into the chunks a {@link com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel} loop over them would run. A batched
 * product also splits its tokens into panels whose widened activations (two bytes per element) stay near
 * {@value #PANEL_BYTES} bytes, so a task's activations stay in the core's L2 cache while its weight rows stream
 * through, where one task over every token re-reads all of them for each weight row; its row chunks hold at most
 * {@value #CHUNK_BYTES} weight bytes and are sized so the grid has about {@value #TASKS_PER_WORKER} tasks per
 * worker. Tasks are numbered row-chunk major, so the consecutive tasks one worker runs reuse the same weight rows
 * across panels. Every (row, token) output is the same dot product whatever the grid, so the grid changes speed only.
 * Immutable.
 */
final class CpuTaskGrid {

    /** Multiply-accumulates per task below which rows are not split further. */
    static final long MIN_TASK_WORK = 1L << 15;

    /** Widened activation bytes one token panel of a batched product aims at. */
    static final long PANEL_BYTES = 256L << 10;

    /** Weight bytes one row chunk of a batched product keeps at most, so it stays cached across panels. */
    static final long CHUNK_BYTES = 512L << 10;

    /** Token panels are whole multiples of this, the register tile of the batched kernels. */
    static final int PANEL_QUANTUM = 8;

    /** Row chunks of a batched product are whole multiples of this, the row tile of the batched kernels. */
    static final int ROW_QUANTUM = 2;

    /** Tasks per worker a product aims at, for balance between unequal cores. */
    static final int TASKS_PER_WORKER = 4;

    /**
     * Widened activation bytes (two per element) the prepared layout of one token slab of a batched product holds
     * at most: a product over more tokens runs slab after slab, so its layout memory does not grow with the batch.
     */
    static final long SLAB_BYTES = 32L << 20;

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
     * {@code nTokens} tokens over {@code workers} workers, token panels starting on multiples of {@code align}.
     *
     * @throws IllegalArgumentException for non-positive sizes
     */
    static CpuTaskGrid of(int rowCount, int nTokens, int cols, long rowBytes, int align, int workers) {
        if (rowCount < 1 || nTokens < 1 || cols < 1 || rowBytes < 1 || align < 1 || workers < 1) {
            throw new IllegalArgumentException("task grid of " + rowCount + " rows x " + nTokens + " tokens x "
                    + cols + " columns (" + rowBytes + " B/row), align " + align + ", " + workers + " workers");
        }
        long maxChunks = (long) TASKS_PER_WORKER * workers;
        if (nTokens == 1) {
            long grain = Math.max(1L, MIN_TASK_WORK / cols);
            long rows = Math.max(grain, (rowCount + maxChunks - 1) / maxChunks);
            return new CpuTaskGrid(rowCount, 1, (int) Math.min(rowCount, rows), 1);
        }
        int panel = panelTokens(cols, align);
        if (panel >= nTokens) {
            panel = nTokens;
        }
        int panels = (nTokens + panel - 1) / panel;
        long wantedChunks = Math.max(1L, (maxChunks + panels - 1) / panels);
        long rowsForTasks = (rowCount + wantedChunks - 1) / wantedChunks;
        long rowsForCache = Math.max(1L, CHUNK_BYTES / rowBytes);
        long rowsForWork = Math.max(1L, MIN_TASK_WORK / ((long) cols * panel));
        long rows = Math.max(rowsForWork, Math.min(rowsForTasks, rowsForCache));
        rows = Math.min(rowCount, (rows + ROW_QUANTUM - 1) / ROW_QUANTUM * ROW_QUANTUM);
        return new CpuTaskGrid(rowCount, nTokens, (int) rows, panel);
    }

    /**
     * Tokens of one slab of a batched product over {@code cols} columns whose panels start on multiples of
     * {@code align}: whole panels whose widened activations stay within {@value #SLAB_BYTES} bytes, at least one
     * panel. Slab starts are then multiples of the panel size, so slabbing changes neither the panels nor a value.
     *
     * @throws IllegalArgumentException for non-positive sizes
     */
    static int slabTokens(int cols, int align) {
        if (cols < 1 || align < 1) {
            throw new IllegalArgumentException("token slab of " + cols + " columns, align " + align);
        }
        int panel = panelTokens(cols, align);
        long fit = SLAB_BYTES / (2L * cols);
        return (int) Math.min(Integer.MAX_VALUE - panel, Math.max(panel, fit / panel * panel));
    }

    /** Tokens of one panel: a multiple of the alignment and the register tile near {@value #PANEL_BYTES} bytes. */
    private static int panelTokens(int cols, int align) {
        int quantum = Math.max(align, PANEL_QUANTUM);
        long fit = PANEL_BYTES / (2L * cols);
        return (int) Math.max(quantum, (fit / quantum) * quantum);
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
