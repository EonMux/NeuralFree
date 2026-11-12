package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.engine.lora.AdapterRows;

/**
 * Output-row pruning of a stack's last layer (llama.cpp {@code inp_out_ids}): when a step's stack output is read at
 * a few rows only, the last layer's attention part still runs over every row, since its K/V rows and every query's
 * keys are needed, but the FFN part runs over the read rows alone, moved to the front of the residual stream. A
 * step whose output no reader consumes (a prompt chunk before the last) keeps only the last layer's cache writes.
 *
 * <p>The rows are step-local, strictly ascending and fewer than the step's rows; read row {@code rows[i]} of the
 * stack output then sits at row {@code i}. The compaction copies contiguous runs of rows forward in place, one copy
 * per run, so a set whose run would overlap its own destination, or that takes more than {@link #MAX_RUNS}
 * copies, is not pruned.</p>
 */
final class LastLayerRows {

    /** Most row copies a compaction may take; generation reads one row and a verify step all of them. */
    static final int MAX_RUNS = 4;

    private LastLayerRows() {
    }

    /**
     * Whether {@code l} may run its post-attention part over a subset of the step's rows: every block after the
     * mixer residual acts on each row alone and nothing reads the other rows afterwards. Hyper-connection and
     * parallel residuals carry more than the stream row, a discarded layer output and a shared FFN norm read
     * full-step slots, the extras, cross attention and the adaptive norm take step-wide inputs, and an FFN that
     * keeps state must see every row.
     */
    static boolean layerPrunable(LayerBlocks l) {
        return l.extras.isEmpty()
                && l.kind != ResidualKind.PARALLEL && l.kind != ResidualKind.HYPER_CONNECTIONS
                && l.crossMixer == null && !l.discard && !l.sharedFfnNorm && l.ada == null
                && (l.ffn == null || l.ffn.rowLocal());
    }

    /**
     * Whether no matmul tap is installed: a collection tool's tap ({@link GpuMatmulTap}, or {@link CpuMatVec}'s
     * global observer for host-run products) records the inputs of every product over every row, as llama.cpp's
     * imatrix run computes every row's output.
     */
    static boolean matmulTapsIdle() {
        return GpuMatmulTap.global() == null && CpuMatVec.globalObserver() == null;
    }

    /**
     * Whether {@code rows} is a strictly ascending set of fewer than {@code nTokens} rows of a {@code nTokens}-row
     * step whose compaction to the front takes at most {@link #MAX_RUNS} non-overlapping copies.
     */
    static boolean compactable(int nTokens, int[] rows) {
        if (rows == null || rows.length >= nTokens) {
            return false;
        }
        int runs = 0;
        int i = 0;
        while (i < rows.length) {
            if (rows[i] < 0 || rows[i] >= nTokens || (i > 0 && rows[i] <= rows[i - 1])) {
                return false;
            }
            int len = runLength(rows, i);
            if (rows[i] != i) {
                if (rows[i] < i + len) {
                    return false;
                }
                runs++;
            }
            i += len;
        }
        return runs <= MAX_RUNS;
    }

    /**
     * Moves stream rows {@code rows} of {@code x} (rows of {@code width} floats) to rows {@code 0 ..}; the set must
     * be {@link #compactable}. Each source run lies at or after its destination and past every earlier destination,
     * so the copies never read a row an earlier copy wrote.
     */
    static void compact(GpuOps ops, ScratchSlot x, int width, int[] rows) {
        int i = 0;
        while (i < rows.length) {
            int len = runLength(rows, i);
            if (rows[i] != i) {
                ops.copy(x.buffer(), x.offset() + rows[i] * width, x.buffer(), x.offset() + i * width, len * width);
            }
            i += len;
        }
    }

    /**
     * Narrows a mixed adapter table to the compacted rows: row {@code i} of the following products is step row
     * {@code rows[i]}, so it takes that row's slot. A uniform table already serves any rows.
     */
    static void selectAdapterRows(AdapterRows table, int[] rows) {
        if (table.uniform() >= 0) {
            return;
        }
        int[] slots = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            slots[i] = table.slot(rows[i]);
        }
        table.set(slots);
    }

    /** The length of the run of consecutive rows starting at index {@code i}. */
    private static int runLength(int[] rows, int i) {
        int len = 1;
        while (i + len < rows.length && rows[i + len] == rows[i] + len) {
            len++;
        }
        return len;
    }
}
