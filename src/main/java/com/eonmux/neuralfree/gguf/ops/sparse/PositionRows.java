package com.eonmux.neuralfree.gguf.ops.sparse;

import java.util.Objects;

/**
 * One sequence's position &rarr; cache row map, the addressing of the MSA and QSA kernels ({@link CpuMsa},
 * {@link CpuQsa}): {@link #row(int)} is the cache row holding position {@code p} of the sequence, or -1 when the
 * position is not cached. A contiguous cache maps {@code p} to row {@code p % kvRows} when that row holds {@code p}
 * ({@link #ring}); a paged memory maps it to the cell the sequence's page table names (llama.cpp
 * {@code llama_kv_cache_msa_context::set_input_pos_slot}, src/llama-kv-cache-msa.cpp:303-345). Reads must be safe
 * from several threads at once.
 */
@FunctionalInterface
public interface PositionRows {

    /** The cache row holding position {@code p}, or -1 when the sequence has no cached row at {@code p}. */
    int row(int p);

    /**
     * Rows of a contiguous cache: position {@code p} lives in row {@code p % kvRows} and is present when
     * {@code kvPos[p % kvRows] == p} ({@code kvPos} is the KV group's row &rarr; position map, -1 for an empty row).
     *
     * @throws IllegalArgumentException when {@code kvRows < 1}
     */
    static PositionRows ring(int kvRows, int[] kvPos) {
        Objects.requireNonNull(kvPos, "kvPos");
        if (kvRows < 1) {
            throw new IllegalArgumentException("a contiguous cache needs at least one row, got " + kvRows);
        }
        return p -> p >= 0 && kvPos[p % kvRows] == p ? p % kvRows : -1;
    }

    /**
     * Rows of an explicit map: {@code rowOfPos[p]} for {@code p < nPos}, -1 beyond.
     *
     * @throws IllegalArgumentException when {@code nPos} is outside {@code [0, rowOfPos.length]}
     */
    static PositionRows of(int[] rowOfPos, int nPos) {
        Objects.requireNonNull(rowOfPos, "rowOfPos");
        if (nPos < 0 || nPos > rowOfPos.length) {
            throw new IllegalArgumentException("position map of " + nPos + " positions exceeds its " + rowOfPos.length
                    + " entries");
        }
        return p -> p >= 0 && p < nPos ? rowOfPos[p] : -1;
    }
}
