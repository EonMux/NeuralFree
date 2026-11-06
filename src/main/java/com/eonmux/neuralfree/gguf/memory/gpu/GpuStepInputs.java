package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;

/**
 * Per-step device inputs shared by every layer: the addressing contract of the KV and recurrent caches.
 * {@link #upload(int, int)} writes each buffer once per ubatch with a single transfer:
 * <ul>
 *   <li>{@link #kvPos(int)}: {@code int[rows]} per KV group, the position stored in each row or -1 when empty;
 *       updated for the rows the step writes. Attention derives visibility (causal, SWA, chunked, symmetric) from
 *       it.</li>
 *   <li>{@link #cellIdx(int)}: {@code int[nTokens]} per KV group, {@code cell_idx[t] = rowOf(pos0 + t)}, the row each
 *       token's K/V is stored in.</li>
 *   <li>{@link #qPos()}: {@code int[nTokens]} query positions {@code pos0 + t}.</li>
 *   <li>{@link #ropePos()}: {@code int[4 * nTokens]} RoPE positions laid out {@code pos[t + k * nTokens]}; section
 *       {@code k} of a text token per {@code RopeSpec.textPositions()}; single-section types read only section 0,
 *       which holds {@code pos0 + t}.</li>
 *   <li>{@link #stateRows()}: {@code int[1] = {0}}, the recurrent-state row of the single sequence.</li>
 * </ul>
 * Buffers are sized for {@code ubatch} tokens and stay valid until {@code close()} of the owning memory.
 */
public interface GpuStepInputs {

    /** Number of KV groups. */
    int groups();

    /** {@code kv_pos[rows]} of a KV group. */
    GpuBuffer kvPos(int group);

    /** {@code cell_idx[nTokens]} of a KV group for the current step. */
    GpuBuffer cellIdx(int group);

    /** Query positions of the current step. */
    GpuBuffer qPos();

    /** Four-section RoPE positions of the current step. */
    GpuBuffer ropePos();

    /** Recurrent-state row indices ({@code {0}}). */
    GpuBuffer stateRows();

    /** First position of the current step. */
    int pos0();

    /** Tokens in the current step. */
    int nTokens();

    /**
     * Writes the inputs for tokens at positions {@code [pos0, pos0 + nTokens)} and marks their rows in every group's
     * {@code kv_pos}.
     *
     * @throws IllegalArgumentException when {@code nTokens} is outside {@code [1, ubatch]} or a position is outside a
     *                                  non-ring group
     */
    void upload(int pos0, int nTokens);

    /**
     * Host position of row {@code row} of the current step: {@code pos0() + row} on the contiguous form, the row's
     * own position on a paged step (the rows of a media chunk share one).
     *
     * @throws IndexOutOfBoundsException when {@code row} is outside the current step
     */
    default int rowPosition(int row) {
        if (row < 0 || row >= nTokens()) {
            throw new IndexOutOfBoundsException("row " + row + " is outside the step's " + nTokens() + " rows");
        }
        return pos0() + row;
    }

    /**
     * The paged step the current forward pass runs, or null between steps and on the contiguous form. A mixer on a
     * paged memory reads it to run the batched step through the page tables.
     */
    default PagedStep currentPagedStep() {
        return null;
    }

    /**
     * Whether the current paged step's rows attend each other without the causal mask (a media chunk's batch,
     * llama.cpp {@code llama_set_causal_attn(false)}); false between steps and on the contiguous form. The rows
     * still read the sequence's stored keys.
     */
    default boolean stepNonCausal() {
        return false;
    }

    /**
     * Page table of the current paged step for a KV group, the step's segments compacted to launch order
     * (segment {@code s} at row {@code s * maxPages()}), or null on the contiguous form.
     */
    default GpuBuffer pageTable(int group) {
        return null;
    }

    /** Slots each launch sequence holds in a KV group ({@code seq_kv_len}), or null on the contiguous form. */
    default GpuBuffer seqKvLen(int group) {
        return null;
    }

    /** Query position of each launch sequence ({@code q_pos_seq}), or null on the contiguous form. */
    default GpuBuffer qPosSeq() {
        return null;
    }

    /** Positions of the segment whose paged prefill launch the caller enqueues next, or null when contiguous. */
    default GpuBuffer segmentPositions() {
        return null;
    }

    /** {@code log2} page size of the paged step, or -1 on the contiguous form. */
    default int pageShift() {
        return -1;
    }

    /** Page-table entries per sequence ({@code PagedStep.maxPages()}), or 0 on the contiguous form. */
    default int maxPages() {
        return 0;
    }

    /** Sequences the paged memory holds ({@code PagedSequences.maxSequences()}), or 1 when contiguous. */
    default int pagedMaxSequences() {
        return 1;
    }

    /**
     * Writes the positions of one multi-token segment into {@link #segmentPositions()}, so the paged prefill launch
     * the caller enqueues next reads them at {@code [0, len)}.
     *
     * @throws UnsupportedOperationException on the contiguous form
     */
    default void uploadSegmentPositions(int pos0, int len) {
        throw new UnsupportedOperationException("the contiguous step inputs hold no segment positions");
    }

    /**
     * The tree-attention mask of the current step, or null between steps and outside a tree verify step:
     * {@code long} per step row, bit {@code i} of row {@code r} set when fresh row {@code r} sees fresh row
     * {@code i} ({@link com.eonmux.neuralfree.gguf.engine.ForwardBatch#treeMask}). The last {@link #nTokens()}
     * slots of the step's key range are the fresh rows the mask covers.
     */
    default GpuBuffer treeMask() {
        return null;
    }

    /**
     * Begins a tree verify step over the rows the last {@link #upload(int, int)} or paged step laid out:
     * uploads {@code mask} (one {@code long} per current step row) and, when the rows run at attention positions
     * other than their store positions ({@code ropeSections} of one section per row, the depth positions of the
     * tree), rewrites the query and RoPE position inputs with them. Kernels read the mask through
     * {@link #treeMask()} while the step lasts.
     *
     * @param ropeSections one position section per step row (the rows' depth positions), or null when the rows run
     *                     at their store positions (no rope, or depth equals store position)
     * @throws IllegalArgumentException when {@code mask} does not hold exactly {@link #nTokens()} rows or a
     *                                  section list does not match
     */
    default void beginTreeStep(long[] mask, int[][] ropeSections) {
        throw new UnsupportedOperationException(this.getClass().getSimpleName() + " holds no tree step state");
    }

    /** Ends the tree verify step; the next {@link #upload(int, int)} or paged step rewrites every input. */
    default void endTreeStep() {
    }

    /**
     * Writes the tree step's rope positions of one multi-token segment — the rows {@code [segStart, segStart+len)}
     * of the depth positions {@link #beginTreeStep} laid out — into {@link #segmentPositions()} at {@code [0, len)},
     * so the paged prefill launch the caller enqueues next reads them launch-local. A tree step's queries run at
     * their depth positions, not at the segment's store positions
     * ({@link #uploadSegmentPositions} serves the plain steps).
     *
     * @throws UnsupportedOperationException on the contiguous form
     */
    default void uploadSegmentTreePositions(int segStart, int len) {
        throw new UnsupportedOperationException("the contiguous step inputs hold no segment positions");
    }

    /**
     * Writes the tree mask relativized to one multi-token segment — the bits of the step's rows
     * {@code [segStart, segStart+len)} shifted down by {@code segStart} — into {@link #treeMask()} at
     * {@code [0, len)}, so the paged prefill launch the caller enqueues next reads the segment's rows launch-local
     * (the kernel indexes the mask without a row offset). A masked batched step carries several sequences' pieces,
     * one segment each, and every segment's launch uploads its own slice.
     *
     * @throws UnsupportedOperationException on the contiguous form
     */
    default void uploadTreeMaskSegment(int segStart, int len) {
        throw new UnsupportedOperationException("the contiguous step inputs hold no tree mask segments");
    }

    /**
     * Reserves the per-sequence addressing buffers of a KV group ({@link #sequenceKvPos}), once per group; a mixer
     * of a paged memory calls it at bind. No-op on the contiguous form.
     */
    default void reserveSequenceAddressing(int group) {
    }

    /**
     * {@code kv_pos} of a KV group as paged sequence {@code seq} of the current step sees it: {@code int[rows]}
     * holding the position of every cell in the sequence's page table and -1 for every other cell, the sequence
     * mask llama.cpp's KV mask applies ({@code cells.seq_has}, llama-kv-cache.cpp:1664-1710). A kernel scoring all
     * cells against it sees the sequence's own rows only. Built and uploaded on the first request of a step for the
     * pair; null on the contiguous form.
     *
     * @throws IllegalStateException when the group's addressing was not reserved or no paged step is current
     */
    default GpuBuffer sequenceKvPos(int group, int seq) {
        return null;
    }
}
