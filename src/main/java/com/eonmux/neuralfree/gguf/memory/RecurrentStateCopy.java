package com.eonmux.neuralfree.gguf.memory;

import java.util.Objects;

/**
 * A copy of the recurrent state of one sequence: the sequence's row of every state tensor of every layer, in the
 * memory's allocation order ({@link ModelMemory#saveSequenceState}). It is the recurrent half of a context
 * checkpoint (llama.cpp's {@code common_prompt_checkpoint} target state, vLLM's Mamba "align" cached state): a
 * recurrent state cannot be rewound, so a prompt prefix is reusable only where such a copy was taken.
 * {@link ModelMemory#restoreSequenceState} writes it into any sequence slot of the same memory. Immutable.
 */
public final class RecurrentStateCopy {

    /** The copy of a memory without recurrent state: no rows. */
    public static final RecurrentStateCopy EMPTY = new RecurrentStateCopy(new float[0][]);

    private final float[][] rows;
    private final long bytes;

    private RecurrentStateCopy(float[][] rows) {
        this.rows = rows;
        long b = 0;
        for (float[] r : rows) {
            b += (long) r.length * Float.BYTES;
        }
        this.bytes = b;
    }

    /**
     * A copy holding {@code rows} (copied): one state row per state tensor, in the memory's allocation order.
     *
     * @throws NullPointerException for a null array or row
     */
    public static RecurrentStateCopy of(float[][] rows) {
        Objects.requireNonNull(rows, "rows");
        float[][] copy = new float[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            copy[i] = Objects.requireNonNull(rows[i], "state row " + i).clone();
        }
        return new RecurrentStateCopy(copy);
    }

    /** State rows the copy holds (one per state tensor). */
    public int rows() {
        return rows.length;
    }

    /** Floats of state row {@code i}. */
    public int floats(int i) {
        return rows[Objects.checkIndex(i, rows.length)].length;
    }

    /**
     * Writes state row {@code i} into {@code dst} from {@code offset}.
     *
     * @throws IndexOutOfBoundsException when {@code i} is outside the rows or the row does not fit {@code dst}
     */
    public void copyInto(int i, float[] dst, int offset) {
        float[] row = rows[Objects.checkIndex(i, rows.length)];
        System.arraycopy(row, 0, dst, offset, row.length);
    }

    /** Bytes the copy holds. */
    public long bytes() {
        return bytes;
    }

    @Override
    public String toString() {
        return "RecurrentStateCopy[" + rows.length + " rows, " + bytes + " bytes]";
    }
}
