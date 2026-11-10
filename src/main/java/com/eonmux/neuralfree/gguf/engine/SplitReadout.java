package com.eonmux.neuralfree.gguf.engine;

import java.util.Objects;

/**
 * One forward call's output rows read two ways at once ({@link ForwardBackend#forwardBatchSplit}): the top candidates
 * of every output row, and the whole logits of the rows the caller flagged. A batch step whose sampling pieces mostly
 * fit a device cone reads only their candidates, while a piece that needs its whole rows (a grammar, a logprobs
 * capture, a verify round's walk) still gets them from the same forward. Immutable.
 */
public final class SplitReadout {

    private final RowTopK top;
    private final float[][] rows;

    /**
     * @param top  the candidates of every output row, in batch order
     * @param rows per output row its whole logits when the call flagged it, else null (copied shallowly)
     * @throws IllegalArgumentException when the two cover different row counts
     */
    public SplitReadout(RowTopK top, float[][] rows) {
        this.top = Objects.requireNonNull(top, "top");
        Objects.requireNonNull(rows, "rows");
        if (rows.length != top.rows()) {
            throw new IllegalArgumentException("the readout holds " + top.rows() + " candidate rows and "
                    + rows.length + " row slots");
        }
        this.rows = rows.clone();
    }

    /** The candidates of every output row, in batch order. */
    public RowTopK top() {
        return top;
    }

    /** The whole logits of output row {@code i} when the call flagged it, else null. */
    public float[] row(int i) {
        if (i < 0 || i >= rows.length) {
            throw new IndexOutOfBoundsException("output row " + i + " outside [0, " + rows.length + ")");
        }
        return rows[i];
    }

    /** The output rows the readout covers. */
    public int rows() {
        return rows.length;
    }
}
