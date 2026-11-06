package com.eonmux.neuralfree.gguf.ops.flow.dflash;

/**
 * Top candidates of one block's rows with their probabilities: the readback a device-scorched readout hands its
 * scorer instead of the block's whole logits. Row {@code r} holds {@code k} candidates in rank order (larger
 * probability first, the lower token id on ties), {@code p} after the row's softmax. Immutable.
 */
public final class DflashCandidates {

    private final int rows;
    private final int k;
    private final int[] ids;
    private final float[] probs;

    /**
     * @throws IllegalArgumentException for a non-positive row or candidate count or a short array
     */
    public DflashCandidates(int rows, int k, int[] ids, float[] probs) {
        if (rows < 1 || k < 1) {
            throw new IllegalArgumentException("candidates need positive rows and width, got " + rows + "/" + k);
        }
        if (ids.length < rows * k || probs.length < rows * k) {
            throw new IllegalArgumentException("candidates of " + rows + " rows x " + k + " need "
                    + (rows * k) + " entries, got " + ids.length + "/" + probs.length);
        }
        this.rows = rows;
        this.k = k;
        this.ids = ids.clone();
        this.probs = probs.clone();
    }

    /** Rows of candidates. */
    public int rows() {
        return rows;
    }

    /** Candidates per row. */
    public int k() {
        return k;
    }

    /** Token id of candidate {@code j} of row {@code row}. */
    public int id(int row, int j) {
        return ids[row * k + j];
    }

    /** Probability of candidate {@code j} of row {@code row}, from the row's softmax. */
    public float probability(int row, int j) {
        return probs[row * k + j];
    }
}
