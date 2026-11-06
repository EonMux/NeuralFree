package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

/**
 * The attention mixer of a prefill context-parallel split: the DualChunkSwap prefill the group's coordinator
 * drives. The coordinator has packed every rank's chunk-pair rows and uploaded the packed step inputs before the
 * trunk, and this mixer's prefill runs the layer's exchange over the ranks' parts:
 *
 * <ol>
 *   <li>every rank's part prepares its packed rows — the projections, rope and cache stores of its own chunk
 *       pair; a peer's input rows are the coordinator-packed normed rows of its trunk;</li>
 *   <li>the query all-gather: every rank's packed query rows land in every rank's gather buffer at their global
 *       row offsets, with the true positions;</li>
 *   <li>the cache scatter: a chunk owner re-runs its own stores with rebound destinations (the shadow store) to
 *       write the cache rows another rank's block holds into shadow buffers, and the coordinator moves the
 *       contiguous segments into that rank's cache rows;</li>
 *   <li>every rank attends over the whole gathered query against its own cache block under the causal mask; the
 *       rank taps merge the partitions into rank 0's attention buffer in global row order, and the coordinator
 *       scatters the merged rows into every rank's buffer in that rank's packed order;</li>
 *   <li>rank 0 finishes its chunk-pair rows; the peers' trunks finish theirs through {@link PcpPeerMixer}.</li>
 * </ol>
 *
 * <p>Over a paged memory the coordinator carries the whole batched step (a paged layout with a multi-row segment):
 * every rank's part prepares the packed rows of its translated page tables, the gather, cache scatter and merge run
 * over the whole step's rows, and every rank's flash dispatches its own page tables through {@code
 * attendPagedStep}. A single-row segment (a decode row of a mixed step) replicates onto every rank's packed rows
 * and its partition keeps the decode-context-parallel merge. A pure-decode paged batch never reaches the
 * coordinator: it keeps {@link CpAttentionMixer}'s protocol.</p>
 */
final class PcpAttentionMixer extends CpAttentionMixer {

    PcpAttentionMixer(TensorParallelGroup group, LayerSpec layer) {
        super(group, layer);
    }

    @Override
    protected void pcpPrefill(int pos0, int n0, ScratchSlot in, ScratchSlot out) {
        requireBound();
        PcpPrefillCoordinator coord = group.pcpCoordinator();
        if (coord.batchedStep()) {
            pcpBatched(coord, pos0, n0, in, out);
            return;
        }
        PcpChunkSwap swap = coord.swap();
        int n = coord.stepTokens();
        if (swap.ownedCount(0) != n0 || pos0 != coord.stepPos0()) {
            throw new IllegalStateException("layer " + describe() + ": the coordinator packed a "
                    + swap.ownedCount(0) + "-row window at " + coord.stepPos0() + " for rank 0, but the step"
                    + " carries " + n0 + " rows at " + pos0);
        }
        this.stepRows = n * nHead;
        this.stepSkip = 0;
        preparePackedRows(coord, swap, pos0, n0, in);
        coord.gatherQueries(parts, layer.attention().qRopeDim());
        coord.scatterCacheRows(parts);
        attendGatheredRows(coord, n);
        coord.scatterAttention(this, nHead * headDimV);
        parts.get(0).finishStep(n0, out);
    }

    /**
     * The batched paged step: the coordinator packed every rank's DualChunkSwap rows over its translated page
     * tables, and the exchange runs over the whole step's rows — every rank's part stores the rows its pages own,
     * the flashes dispatch the rank's page tables, and the log-sum-exp merge makes the gathered whole exact.
     */
    private void pcpBatched(PcpPrefillCoordinator coord, int pos0, int n0, ScratchSlot in, ScratchSlot out) {
        int n = coord.stepTokens();
        if (coord.packedRows(0) != n0 || pos0 != coord.packedPos0(0)) {
            throw new IllegalStateException("layer " + describe() + ": the coordinator packed a "
                    + coord.packedRows(0) + "-row window at " + coord.packedPos0(0) + " for rank 0, but the step"
                    + " carries " + n0 + " rows at " + pos0);
        }
        this.stepRows = n * nHead;
        this.stepSkip = 0;
        prepareBatchedRows(coord, n0, in);
        coord.gatherBatchedQueries(parts, layer.attention().qRopeDim());
        coord.scatterBatchedCacheRows(parts);
        attendBatchedRows(coord);
        coord.scatterBatchedAttention(this, nHead * headDimV);
        parts.get(0).finishStep(n0, out);
    }

    /** Phase 1 of the batched step: every rank's part projects, ropes and stores its own packed rows. */
    private void prepareBatchedRows(PcpPrefillCoordinator coord, int n0, ScratchSlot in) {
        parts.get(0).prepareStep(coord.packedPos0(0), n0, in);
        for (int rank = 1; rank < parts.size(); rank++) {
            int rows = coord.packedRows(rank);
            if (rows > 0) {
                parts.get(rank).prepareStep(coord.packedPos0(rank), rows, coord.normedSlot(rank));
            }
        }
    }

    /** Phase 4 of the batched step: every rank flashes over its own pages; a rank without pages yields the empty
     * partition the merge weights 0. The peers first, rank 0 merges. */
    private void attendBatchedRows(PcpPrefillCoordinator coord) {
        for (int rank = parts.size() - 1; rank >= 1; rank--) {
            parts.get(rank).attendPagedStep(coord.rankStep(rank), coord.queryGather(rank));
        }
        parts.get(0).attendPagedStep(coord.rankStep(0), coord.queryGather(0));
    }

    /** Phase 1: every rank's part projects, ropes and stores its own packed chunk-pair rows. */
    private void preparePackedRows(PcpPrefillCoordinator coord, PcpChunkSwap swap, int pos0, int n0,
                                   ScratchSlot in) {
        parts.get(0).prepareStep(pos0, n0, in);
        for (int rank = 1; rank < parts.size(); rank++) {
            int rows = swap.ownedCount(rank);
            if (rows > 0) {
                parts.get(rank).prepareStep(coord.packedPos0(rank), rows, coord.normedSlot(rank));
            }
        }
    }

    /** Phase 4: every rank flashes the gathered query over its cache block; the peers first, rank 0 merges. */
    private void attendGatheredRows(PcpPrefillCoordinator coord, int n) {
        for (int rank = parts.size() - 1; rank >= 1; rank--) {
            parts.get(rank).attendStep(coord.stepPos0(), n, coord.queryGather(rank), coord.queryPos(rank));
        }
        parts.get(0).attendStep(coord.stepPos0(), n, coord.queryGather(0), coord.queryPos(0));
    }
}
