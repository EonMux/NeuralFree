package com.eonmux.neuralfree.gguf.cpu.forward;

import java.util.Arrays;

/**
 * Output-row pruning of the CPU stack's last layer (llama.cpp {@code inp_out_ids}), the host twin of the GPU
 * engine's {@code LastLayerRows}: when a step's stack output is read at a few rows only, the last layer's attention
 * part still runs over every row, since its K/V rows and every query's keys are needed, but its FFN part runs over
 * the read rows alone, moved to the front of the residual stream ({@link CpuLayerExecutor#forwardRows}). A step
 * whose output no reader consumes (a prompt chunk before the last) keeps only the last layer's cache writes.
 *
 * <p>The read rows are step-local and strictly ascending; read row {@code rows[i]} of the stack output then sits at
 * row {@code i}.</p>
 */
final class CpuOutputRows {

    /** No row read: the step keeps only the last layer's cache writes. */
    static final int[] NONE = new int[0];

    private CpuOutputRows() {
    }

    /**
     * The distinct step-local rows, ascending, of the call rows {@code rows} that fall in the step
     * {@code [off, off + n)}.
     */
    static int[] read(int[] rows, int off, int n) {
        return Arrays.stream(rows).filter(r -> r >= off && r < off + n).map(r -> r - off).distinct().sorted()
                .toArray();
    }

    /** Whether {@code rows} is a strictly ascending set of fewer than {@code n} rows of an {@code n}-row step. */
    static boolean subset(int n, int[] rows) {
        if (rows == null || rows.length >= n) {
            return false;
        }
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] < 0 || rows[i] >= n || (i > 0 && rows[i] <= rows[i - 1])) {
                return false;
            }
        }
        return true;
    }

    /**
     * Moves rows {@code rows} of {@code x} (rows of {@code width} floats from index 0) to rows {@code 0 ..}; the set
     * must be a {@link #subset}. Row {@code rows[i]} is at or after its destination {@code i} and every later source
     * lies past it, so no copy reads a row an earlier copy wrote.
     */
    static void compact(float[] x, int width, int[] rows) {
        for (int i = 0; i < rows.length; i++) {
            if (rows[i] != i) {
                System.arraycopy(x, rows[i] * width, x, i * width, width);
            }
        }
    }

    /**
     * The stack-output row of step-local row {@code local}: the row itself when the step ran every row
     * ({@code read} null), else its index in the read rows.
     */
    static int stackRow(int[] read, int local) {
        return read == null ? local : Arrays.binarySearch(read, local);
    }
}
