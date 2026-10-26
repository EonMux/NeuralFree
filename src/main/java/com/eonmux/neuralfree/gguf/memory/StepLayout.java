package com.eonmux.neuralfree.gguf.memory;

import java.util.Arrays;
import java.util.Objects;

/**
 * The addressing layout of one forward step over a {@link com.eonmux.neuralfree.gguf.engine.ForwardBatch}: its
 * rows grouped into segments of one sequence each, ordered by sequence then position, with the first position and
 * the output rows of the step. The mixers, their extras and the forward engine read it instead of the single
 * {@code (pos0, nTokens)} pair of the contiguous path.
 *
 * <p>Immutable and validated; arrays are copied at construction and accessors return the values of one segment
 * only. A step over the single held sequence produces {@link #contiguousSingle()} == true, the delegation
 * condition of the {@code batch} defaults: it runs today's {@code decode} / {@code prefill} unchanged.</p>
 */
public final class StepLayout {

    private final int nTokens;
    private final int[] seq;      // seq[seg]
    private final int[] tokStart; // first ubatch row of seg
    private final int[] len;      // rows of seg
    private final int[] pos0;     // first position of seg
    private final int[] outputRows; // ubatch rows whose logits (or other output products) are produced
    private final int uniformSegLen; // the segment length all segments share, -1 when they differ
    private final boolean paged;

    private StepLayout(int nTokens, int[] seq, int[] tokStart, int[] len, int[] pos0, int[] outputRows,
            int uniformSegLen, boolean paged) {
        this.nTokens = nTokens;
        this.seq = seq;
        this.tokStart = tokStart;
        this.len = len;
        this.pos0 = pos0;
        this.outputRows = outputRows;
        this.uniformSegLen = uniformSegLen;
        this.paged = paged;
    }

    /**
     * Builds the layout of one step from its segments.
     *
     * @param nTokens    rows of the step, the sum of {@code len}
     * @param seq        sequence id of each segment (ascending, one segment per sequence in the step)
     * @param tokStart   first row of each segment; {@code tokStart[0]} is 0 and a segment starts where the
     *                   previous one ends
     * @param len        rows of each segment
     * @param pos0       first position of each segment
     * @param outputRows rows of the step (ubatch-local) whose logits are produced, ascending
     * @param pagedMemory whether the memory of the model is paged ({@code neuralfree.parallel > 1} or
     *                    {@code neuralfree.kv.paged}): a single-segment step over paged memory is still not
     *                    the contiguous path
     * @throws IllegalArgumentException when a count or range breaks the layout rules
     */
    public static StepLayout of(int nTokens, int[] seq, int[] tokStart, int[] len, int[] pos0,
            int[] outputRows, boolean pagedMemory) {
        Objects.requireNonNull(seq, "seq");
        Objects.requireNonNull(tokStart, "tokStart");
        Objects.requireNonNull(len, "len");
        Objects.requireNonNull(pos0, "pos0");
        Objects.requireNonNull(outputRows, "outputRows");
        int nSeg = seq.length;
        if (nTokens < 0) {
            throw new IllegalArgumentException("a step holds non-negative token counts, got " + nTokens);
        }
        if (nSeg == 0) {
            throw new IllegalArgumentException("a step has at least one segment");
        }
        if (tokStart.length != nSeg || len.length != nSeg || pos0.length != nSeg) {
            throw new IllegalArgumentException("seq, tokStart, len and pos0 share the segment count " + nSeg);
        }
        int start = 0;
        for (int s = 0; s < nSeg; s++) {
            if (seq[s] < 0) {
                throw new IllegalArgumentException("segment " + s + " has sequence id " + seq[s]);
            }
            if (s > 0 && seq[s] <= seq[s - 1]) {
                throw new IllegalArgumentException("segments are ordered by sequence id ascending, " + seq[s]
                        + " after " + seq[s - 1]);
            }
            if (tokStart[s] != start) {
                throw new IllegalArgumentException("segment " + s + " starts at row " + tokStart[s] + ", expected "
                        + start);
            }
            if (len[s] <= 0) {
                throw new IllegalArgumentException("segment " + s + " holds " + len[s] + " rows");
            }
            if (pos0[s] < 0) {
                throw new IllegalArgumentException("segment " + s + " starts at position " + pos0[s]);
            }
            start += len[s];
        }
        if (start != nTokens) {
            throw new IllegalArgumentException("segments cover " + start + " rows, the step holds " + nTokens);
        }
        int uniform = len[0];
        for (int s = 1; s < nSeg; s++) {
            if (len[s] != uniform) {
                uniform = -1;
                break;
            }
        }
        int prevOutput = -1;
        for (int row : outputRows) {
            if (row < 0 || row >= nTokens) {
                throw new IllegalArgumentException("output row " + row + " outside the step of " + nTokens);
            }
            if (row <= prevOutput) {
                throw new IllegalArgumentException("output rows ascend, " + row + " after " + prevOutput);
            }
            prevOutput = row;
        }
        return new StepLayout(nTokens, seq.clone(), tokStart.clone(), len.clone(), pos0.clone(),
                outputRows.clone(), uniform, pagedMemory);
    }

    /** Rows of the step. */
    public int nTokens() {
        return nTokens;
    }

    /** Segments of the step, one per sequence. */
    public int nSeg() {
        return seq.length;
    }

    /** Sequence id of segment {@code seg}.
     *
     * @throws IndexOutOfBoundsException when {@code seg} is outside the segments
     */
    public int seq(int seg) {
        return seq[seg];
    }

    /** First row (ubatch-local token index) of segment {@code seg}. */
    public int tokStart(int seg) {
        return tokStart[seg];
    }

    /** Rows of segment {@code seg}. */
    public int len(int seg) {
        return len[seg];
    }

    /** First position of segment {@code seg} ({@code pos0}); rows of the segment sit at {@code pos0 .. +len}). */
    public int pos0(int seg) {
        return pos0[seg];
    }

    /** Ubatch-local rows whose output products are produced, ascending. */
    public int[] outputRows() {
        return outputRows.clone();
    }

    /**
     * The segment length every segment shares, or -1 when they differ ({@code n_seq_tokens} of llama.cpp
     * {@code llama_ubatch} when {@link #uniformSegLen()} is not -1).
     */
    public int uniformSegLen() {
        return uniformSegLen;
    }

    /** Whether the memory of this step is paged: rows are addressed through page tables, not one position each. */
    public boolean paged() {
        return paged;
    }

    /**
     * True when the step is a single segment over un-paged memory: the delegation condition. Only then the
     * default {@code batch} methods run today's {@code decode} / {@code prefill} with
     * {@code pos0 = pos0(0)} and {@code nTokens = nTokens()}.
     */
    public boolean contiguousSingle() {
        return nSeg() == 1 && !paged;
    }

    @Override
    public String toString() {
        return "StepLayout{nTokens=" + nTokens + ", nSeg=" + seq.length + ", uniformSegLen=" + uniformSegLen
                + ", paged=" + paged + ", seq=" + Arrays.toString(seq) + ", len=" + Arrays.toString(len)
                + ", pos0=" + Arrays.toString(pos0) + '}';
    }
}
