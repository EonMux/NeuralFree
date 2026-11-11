package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuOps;

import java.util.Arrays;
import java.util.Objects;

/**
 * The top candidates of each of a forward call's evaluated rows, in rank order: larger logit first, the lower token
 * id on ties, exactly {@code CpuOps.topK} ranks. A truncated sampler chain samples over these candidates exactly
 * what it samples over the whole row (llama.cpp backend sampling {@code -bs} over a device top-k readback), so a
 * generation whose chain keeps its candidates inside the top cone reads {@code k} values per row back instead of
 * the vocabulary. Immutable.
 */
public final class RowTopK {

    /** Most candidates one device readback of a sampled row carries. */
    public static final int MAX_DEVICE_CANDIDATES = 64;

    private final int k;
    private final int[] ids;
    private final float[] logits;

    private RowTopK(int k, int[] ids, float[] logits) {
        this.k = k;
        this.ids = ids;
        this.logits = logits;
    }

    /**
     * The top candidates of every row of {@code rows}, selected on the host; rows shorter than {@code k} shrink it
     * for every row (a vocabulary below the cone width keeps all of its tokens).
     *
     * @throws IllegalArgumentException for an empty row list or a non-positive {@code k}
     */
    public static RowTopK of(float[][] rows, int k) {
        Objects.requireNonNull(rows, "rows");
        if (rows.length == 0) {
            throw new IllegalArgumentException("top-k of no rows");
        }
        if (k < 1) {
            throw new IllegalArgumentException("top-k needs at least one candidate, got " + k);
        }
        int width = k;
        for (float[] row : rows) {
            Objects.requireNonNull(row, "row");
            width = Math.min(width, row.length);
        }
        if (width < 1) {
            throw new IllegalArgumentException("top-k of empty rows");
        }
        int[] ids = new int[rows.length * width];
        float[] vals = new float[rows.length * width];
        int[] top = new int[width];
        for (int r = 0; r < rows.length; r++) {
            CpuOps.topK(rows[r], 0, rows[r].length, width, top, 0);
            for (int j = 0; j < width; j++) {
                ids[r * width + j] = top[j];
                vals[r * width + j] = rows[r][top[j]];
            }
        }
        return new RowTopK(width, ids, vals);
    }

    /** The candidates of a device readback: {@code rows} rows of {@code k} packed ids and logits, in rank order. */
    public static RowTopK packed(int k, int rows, int[] ids, float[] logits) {
        if (k < 1 || rows < 1) {
            throw new IllegalArgumentException("top-k readback of " + rows + " rows of " + k + " candidates");
        }
        if (ids.length < rows * k || logits.length < rows * k) {
            throw new IllegalArgumentException("top-k readback of " + rows + " rows of " + k + " candidates needs "
                    + (rows * k) + " entries, got " + ids.length + "/" + logits.length);
        }
        return new RowTopK(k, ids.clone(), logits.clone());
    }

    /** Candidates per row. */
    public int k() {
        return k;
    }

    /** Evaluated rows. */
    public int rows() {
        return ids.length / k;
    }

    /** Token id of candidate {@code j} of row {@code row}. */
    public int id(int row, int j) {
        return ids[row * k + j];
    }

    /** Logit of candidate {@code j} of row {@code row}. */
    public float logit(int row, int j) {
        return logits[row * k + j];
    }

    /** The candidates of row {@code row}: {@code k} ids and their logits, in rank order. */
    public int[] ids(int row) {
        return Arrays.copyOfRange(ids, row * k, (row + 1) * k);
    }

    /** The logits of row {@code row}'s candidates, in rank order. */
    public float[] logits(int row) {
        return Arrays.copyOfRange(logits, row * k, (row + 1) * k);
    }

    @Override
    public String toString() {
        return "RowTopK{rows=" + rows() + ", k=" + k + "}";
    }
}
