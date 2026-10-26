package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.memory.StepLayout;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * An immutable unit of forward work built by {@link BatchSplitter#split} from a {@link ForwardBatch}: the batch
 * row indices it runs, ordered by sequence then position, and the {@link StepLayout} the mixers and extras read.
 * The rows of one sequence stay consecutive (its positions are consecutive by the batch rules), so each ubatch
 * holds one segment per sequence and {@code layout.tokStart(seg)} marks where it starts.
 */
public final class UBatch {

    private final ForwardBatch batch;
    private final int[] rows; // batch row indices, ordered by sequence then position
    private final StepLayout layout;

    private UBatch(ForwardBatch batch, int[] rows, StepLayout layout) {
        this.batch = Objects.requireNonNull(batch, "batch");
        this.rows = Objects.requireNonNull(rows, "rows").clone();
        this.layout = Objects.requireNonNull(layout, "layout");
    }

    /** A ubatch of {@code rows} (already ordered by sequence then position) out of {@code batch}. */
    static UBatch of(ForwardBatch batch, int[] rows, boolean paged) {
        int n = rows.length;
        if (n <= 0) {
            throw new IllegalArgumentException("a ubatch runs at least one row");
        }
        int[] seq = batch.seqIds();
        int[] pos = batch.positions();
        for (int i = 1; i < n; i++) {
            if (seq[rows[i]] < seq[rows[i - 1]]
                    || (seq[rows[i]] == seq[rows[i - 1]] && pos[rows[i]] <= pos[rows[i - 1]])) {
                throw new IllegalArgumentException("ubatch rows are ordered by sequence then position: row "
                        + rows[i] + " (" + seq[rows[i]] + ',' + pos[rows[i]] + ") after row " + rows[i - 1]
                        + " (" + seq[rows[i - 1]] + ',' + pos[rows[i - 1]] + ")");
            }
        }
        int[] tokSeq = new int[n];
        int[] tokStart = new int[n];
        int[] len = new int[n];
        int[] pos0 = new int[n];
        int nSeg = 0;
        int[] outputLocal = new int[0];
        {
            int prevSeq = -1;
            int prevPos = -1;
            int out = 0;
            int[] outputs = batch.outputRows();
            for (int i = 0; i < n; i++) {
                if (seq[rows[i]] != prevSeq) {
                    prevSeq = seq[rows[i]];
                    prevPos = pos[rows[i]] - 1;
                    tokSeq[nSeg] = prevSeq;
                    tokStart[nSeg] = i;
                    len[nSeg] = 1;
                    pos0[nSeg] = pos[rows[i]];
                    nSeg++;
                } else {
                    if (pos[rows[i]] != prevPos + 1) {
                        throw new IllegalArgumentException("sequence " + prevSeq + " holed in the ubatch: position "
                                + pos[rows[i]] + " after " + prevPos);
                    }
                    len[nSeg - 1]++;
                }
                prevPos = pos[rows[i]];
                while (out < outputs.length && outputs[out] <= rows[i]) {
                    if (outputs[out] == rows[i]) {
                        int[] grown = new int[outputLocal.length + 1];
                        System.arraycopy(outputLocal, 0, grown, 0, outputLocal.length);
                        grown[outputLocal.length] = i;
                        outputLocal = grown;
                    }
                    out++;
                }
            }
        }
        return new UBatch(batch, rows, StepLayout.of(n, Arrays.copyOf(tokSeq, nSeg),
                Arrays.copyOf(tokStart, nSeg), Arrays.copyOf(len, nSeg), Arrays.copyOf(pos0, nSeg),
                outputLocal, paged));
    }

    /** The batch this ubatch was built from. */
    public ForwardBatch batch() {
        return batch;
    }

    /** Rows of the ubatch (the same count as {@link #nTokens()}). */
    public int nTokens() {
        return rows.length;
    }

    /** Batch row indices of the ubatch, ordered by sequence then position. */
    public int[] batchRows() {
        return rows.clone();
    }

    /** Batch row index of ubatch row {@code row}. */
    public int batchRow(int row) {
        return rows[row];
    }

    /** Token ids of the ubatch rows, in ubatch order. */
    public int[] tokens() {
        int[] all = batch.tokens();
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = all[rows[i]];
        }
        return out;
    }

    /** Positions of the ubatch rows, in ubatch order. */
    public int[] positions() {
        int[] all = batch.positions();
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = all[rows[i]];
        }
        return out;
    }

    /** Sequence ids of the ubatch rows, in ubatch order. */
    public int[] seqIds() {
        int[] all = batch.seqIds();
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            out[i] = all[rows[i]];
        }
        return out;
    }

    /** The {@link StepLayout} of the ubatch: segments, positions and output rows. */
    public StepLayout layout() {
        return layout;
    }

    @Override
    public String toString() {
        return "UBatch" + layout + " rows=" + Arrays.toString(rows);
    }
}
