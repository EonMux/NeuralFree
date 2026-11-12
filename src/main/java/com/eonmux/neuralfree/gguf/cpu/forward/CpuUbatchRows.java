package com.eonmux.neuralfree.gguf.cpu.forward;

import com.eonmux.neuralfree.gguf.engine.UBatch;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

/** A paged ubatch's view of its forward batch's per-row data: tree bits, rope sections and token records. */
final class CpuUbatchRows {

    private CpuUbatchRows() {
    }

    /** The {@code rows}' entries named by {@code at}, in {@code at}'s order. */
    static long[] gather(long[] rows, int[] at) {
        long[] out = new long[at.length];
        for (int i = 0; i < at.length; i++) {
            out[i] = rows[at[i]];
        }
        return out;
    }

    /**
     * Extends each of the step's sequences' token record with its segment's rows (ubatch row order): their ids
     * ({@code embeddingToken} for an embedding row) and stored positions.
     */
    static void appendRows(PagedSequences paged, PagedStep step,
                           int[] tokens, int[] positions, int embeddingToken) {
        for (int s = 0; s < step.nSeg(); s++) {
            paged.appendRows(step.segSeq(s), tokens, positions, step.segStart(s), step.segLen(s), embeddingToken);
        }
    }

    /** The batch's rope sections restricted to the ubatch's rows, or null when the batch carries none. */
    static int[][] sliceSections(int[][] sections, UBatch u) {
        if (sections == null) {
            return null;
        }
        int[] rows = u.batchRows();
        int[][] out = new int[sections.length][];
        for (int s = 0; s < sections.length; s++) {
            out[s] = new int[rows.length];
            for (int r = 0; r < rows.length; r++) {
                out[s][r] = sections[s][rows[r]];
            }
        }
        return out;
    }
}
