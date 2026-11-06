package com.eonmux.neuralfree.gguf.quant;

/**
 * The matmul product the tapped calls of the current thread belong to, as the matmul taps of an activation
 * collection ({@code CpuMatVec.InputObserver}, {@code GpuMatmulTap}) ask about it: one llama.cpp graph op, whose
 * {@code src1} row count and kind ({@code MUL_MAT} or {@code MUL_MAT_ID}) decide llama.cpp's imatrix ask.
 *
 * <p>A tapped call is by default its own product: a dense product of the call's input rows, or an indirect
 * (routed expert) product. Code that runs one dense product as several calls (one per token, one per (token, head)
 * pair of a stacked per-head weight) opens a scope with {@link #dense(int)} around them, so each call is asked as the
 * one dense product of the scope's rows. Scopes are per thread and nest; the innermost one applies.</p>
 */
public final class TapProduct implements AutoCloseable {

    private static final ThreadLocal<TapProduct> CURRENT = new ThreadLocal<>();

    private final int rows;
    private final TapProduct outer;
    private boolean closed;

    private TapProduct(int rows, TapProduct outer) {
        this.rows = rows;
        this.outer = outer;
    }

    /**
     * Opens the scope of one dense product of {@code rows} input rows on this thread, to be closed by the same
     * thread (try-with-resources).
     *
     * @throws IllegalArgumentException when {@code rows} is below 1
     */
    public static TapProduct dense(int rows) {
        if (rows < 1) {
            throw new IllegalArgumentException("a dense product has at least one input row, got " + rows);
        }
        TapProduct p = new TapProduct(rows, CURRENT.get());
        CURRENT.set(p);
        return p;
    }

    /** The input rows a tapped call of {@code callRows} rows is asked with: the open scope's, else its own. */
    public static int rows(int callRows) {
        TapProduct p = CURRENT.get();
        return p == null ? callRows : p.rows;
    }

    /** Whether a tapped call is asked as an indirect product: never inside a dense scope, else as the call says. */
    public static boolean indirect(boolean callIndirect) {
        return callIndirect && CURRENT.get() == null;
    }

    /**
     * Closes the scope, restoring the enclosing one.
     *
     * @throws IllegalStateException when a scope opened inside this one is still open
     */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        if (CURRENT.get() != this) {
            throw new IllegalStateException("tap product scopes must close innermost first");
        }
        closed = true;
        if (outer == null) {
            CURRENT.remove();
        } else {
            CURRENT.set(outer);
        }
    }

    @Override
    public String toString() {
        return "TapProduct{dense, " + rows + " rows}";
    }
}
