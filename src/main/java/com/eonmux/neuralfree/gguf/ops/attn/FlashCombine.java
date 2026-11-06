package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;

import java.util.Objects;

/**
 * The split-K merge of one layer's flash launches ({@code LLM_FlashAttn_Combine}): folds the partial rows a split
 * launch left into the output rows. The grid is re-set only when the row count changes. Not thread-safe.
 */
final class FlashCombine {

    private final PreparedDispatch dispatch;
    private int rows;

    /** @param rows the row count {@code dispatch}'s grid was prepared for */
    FlashCombine(PreparedDispatch dispatch, int rows) {
        this.dispatch = Objects.requireNonNull(dispatch, "dispatch");
        this.rows = rows;
    }

    /** The merge launch whose arguments a step re-sets. */
    PreparedDispatch dispatch() {
        return dispatch;
    }

    /** Merges {@code nSplit} partials of {@code rows} head rows into the output at float offset {@code outOff}. */
    void run(int rows, int nSplit, int outOff) {
        dispatch.setInt(AttnKernels.ARG_FC_N_ROWS, rows);
        dispatch.setInt(AttnKernels.ARG_FC_N_SPLIT, nSplit);
        dispatch.setInt(AttnKernels.ARG_FC_OUT_OFF, outOff);
        if (rows != this.rows) {
            dispatch.setGlobal(FlashAttnTiling.combineGlobal(rows));
            this.rows = rows;
        }
        dispatch.dispatch();
    }
}
