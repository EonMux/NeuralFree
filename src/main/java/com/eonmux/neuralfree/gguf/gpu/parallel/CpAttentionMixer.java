package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.ops.attn.AttentionTap;
import com.eonmux.neuralfree.gguf.ops.attn.AttnKernels;
import com.eonmux.neuralfree.gguf.ops.attn.GpuAttentionMixer;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The attention mixer of a context-parallel split (vLLM's decode context parallel, {@code
 * decode_context_parallel_size}): the context positions are divided between the ranks, every rank keeps every head,
 * and each rank attends over its own context block. The rank partials are merged into the one softmax over the
 * concatenated blocks with the log-sum-exp {@code LLM_AttnCpMerge} (vLLM {@code cp_lse_ag_out_ar}), so the layer
 * output equals the single-device attention.
 *
 * <p>Per step every rank's {@link GpuAttentionMixer} runs over its own cache shard, whose rows hold only that rank's
 * context block of the sequence; its {@link AttentionTap} copies the raw head outputs and the merged log-sum-exp
 * ({@code LLM_FlashAttn_Lse}) into rank 0's gather buffers, and rank 0's tap merges them and replaces its attention
 * output before the layer's output stages run. The peers run first so their partitions are ready when rank 0 merges.
 * Rank 0's output is the layer's branch output; a peer stops after the tap, so its discarded output projection is
 * never computed. Every rank keeps the whole attention weights because it computes every head over its context
 * block, but its KV cache holds only its block's rows.</p>
 *
 * <p>A rank whose block starts after the step's last position stores no row and holds no key any query row sees
 * ({@code GpuFlashAttention} clamps the visible keys to the shard), so its partition is empty: zero rows with a
 * {@code -INF} log-sum-exp, which the merge weights with 0. The rank still runs every step, and the merge covers
 * every rank over the same rows.</p>
 *
 * <p>A paged multi-sequence batch runs through the same merge: every rank's memory holds the translated
 * {@code PagedStep} of its own page shard (the position-based page interleaving of {@code CpPageShard}), its
 * flash attention dispatches the rank's page tables, and the tap merges the per-rank partials over the step's
 * rows as in the contiguous step.</p>
 *
 * <p>With a prefill context-parallel size above 1 (vLLM's {@code prefill_context_parallel_size}) the query split
 * is vLLM's DualChunkSwap: while the coordinator ({@code PcpPrefillCoordinator}) runs a step, every prefill
 * routes to {@code pcpPrefill} whatever row count a rank's packed window carries, and the {@link
 * PcpAttentionMixer} subclass replaces the prefill with the exchange and the per-rank flashes over the gathered
 * rows. The decode, KV-only and batched steps keep this class's protocol.</p>
 */
class CpAttentionMixer implements GpuMixer {

    protected final TensorParallelGroup group;
    protected final LayerSpec layer;
    protected final TpAttentionPlan plan;
    protected final List<GpuAttentionMixer> parts = new ArrayList<>();

    protected GpuForwardContext ctx;
    protected boolean bound;
    /** Rank 0 buffers holding every rank's normalized head outputs and log-sum-exp, rank-major. */
    protected GpuBuffer gatherOut;
    protected GpuBuffer gatherLse;
    protected PreparedDispatch merge;
    protected int nHead;
    protected int headDimV;
    protected int ubatch;
    /** The running step's whole head row count ({@code nTokens * nHead}) and the current part's skipped rows. */
    protected int stepRows;
    protected int stepSkip;

    CpAttentionMixer(TensorParallelGroup group, LayerSpec layer) {
        this.group = Objects.requireNonNull(group, "group");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.plan = group.attentionPlan();
        if (!plan.contextParallel()) {
            throw new IllegalArgumentException("layer " + layer.il() + " is not a context-parallel split");
        }
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (bound) {
            throw new IllegalStateException("layer " + describe() + " is already bound");
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.nHead = layer.attention().nHead();
        this.headDimV = layer.attention().headDimV();
        this.ubatch = group.rankContext(0).scratch().ubatch();
        int ranks = group.rankCount();
        long outFloats = (long) ranks * ubatch * nHead * headDimV;
        long lseFloats = (long) ranks * ubatch * nHead;
        this.gatherOut = group.rankBackend(0).allocateBuffer(outFloats * Float.BYTES, false);
        this.gatherLse = group.rankBackend(0).allocateBuffer(lseFloats * Float.BYTES, false);
        this.merge = group.rankOps(0).prepare(AttnKernels.CP_MERGE, new long[]{128L}, new long[]{128L},
                new Object[]{nHead, headDimV, ranks, gatherOut, gatherLse, gatherOut});
        try {
            for (int rank = 0; rank < ranks; rank++) {
                GpuAttentionMixer part = new GpuAttentionMixer();
                part.setAttentionTap(new RankTap(rank));
                part.bind(group.rankContext(rank), layer);
                parts.add(part);
            }
        } catch (RuntimeException e) {
            try {
                release();
            } catch (RuntimeException closeFailure) {
                e.addSuppressed(closeFailure);
            }
            throw e;
        }
        bound = true;
    }

    @Override
    public void decode(int pos, ScratchSlot in, ScratchSlot out) {
        forward(pos, 1, in, out);
    }

    @Override
    public void prefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        forward(pos0, nTokens, in, out);
    }

    private void forward(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        requireBound();
        if (group.pcpStepRunning()) {
            /* The coordinator packed every rank's DualChunkSwap rows and uploaded the packed step inputs before
             * the trunk; this protocol's window upload would overwrite them. A rank's packed window can carry a
             * single row of a multi-row step, so the routing cannot key on the row count. */
            pcpPrefill(pos0, nTokens, in, out);
            return;
        }
        group.uploadStepInputs(pos0, nTokens);
        int width = streamWidth();
        long floats = (long) nTokens * width;
        this.stepRows = nTokens * nHead;
        /* The peers first: rank 0's tap merges their partitions into its own attention output. */
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerIn = group.inputSlot(rank);
            ScratchSlot peerOut = group.partialSlot(rank);
            peerIn.require(floats);
            peerOut.require(floats);
            int skip = group.prefillSplitOffset(rank, pos0, nTokens);
            if (skip >= nTokens) {
                /* The rank's whole block lies above the step: its partition is empty for every row. */
                fillEmptyPartition(rank);
                continue;
            }
            this.stepSkip = skip;
            RankExchange.replay(ctx.queue(), in.buffer(), in.offset() + (long) skip * width,
                    group.rankQueue(rank), peerIn.buffer(), (long) (nTokens - skip) * width);
            if (nTokens == 1) {
                parts.get(rank).decode(pos0, peerIn, peerOut);
            } else {
                parts.get(rank).prefill(pos0 + skip, nTokens - skip, peerIn, peerOut);
            }
        }
        this.stepSkip = 0;
        if (nTokens == 1) {
            parts.get(0).decode(pos0, in, out);
        } else {
            parts.get(0).prefill(pos0, nTokens, in, out);
        }
    }

    /**
     * The prefill of a prefill context-parallel step: the coordinator has packed every rank's DualChunkSwap rows
     * and uploaded the packed step inputs. The base class does not run it; the {@link PcpAttentionMixer} overrides.
     */
    protected void pcpPrefill(int pos0, int nTokens, ScratchSlot in, ScratchSlot out) {
        throw new IllegalStateException("layer " + describe() + ": the prefill context-parallel size is "
                + group.pcpSize() + " but this split has no DualChunkSwap prefill");
    }

    /** Zeroes one rank's whole gather slot with an empty partition ({@code -INF} log-sum-exp). */
    private void fillEmptyPartition(int rank) {
        ComputeQueue root = group.rankQueue(0);
        long rows = stepRows;
        root.enqueueFillBuffer(gatherOut, (long) rank * rows * headDimV * Float.BYTES,
                rows * (long) headDimV * Float.BYTES, 0);
        root.enqueueFillBuffer(gatherLse, (long) rank * rows * Float.BYTES, rows * Float.BYTES,
                Float.floatToRawIntBits(Float.NEGATIVE_INFINITY));
    }

    @Override
    public void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        requireBound();
        group.uploadStepInputs(pos0, nTokens);
        int width = streamWidth();
        long floats = (long) nTokens * width;
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerIn = group.inputSlot(rank);
            peerIn.require(floats);
            int skip = group.prefillSplitOffset(rank, pos0, nTokens);
            if (skip >= nTokens) {
                continue;
            }
            RankExchange.replay(ctx.queue(), in.buffer(), in.offset() + (long) skip * width,
                    group.rankQueue(rank), peerIn.buffer(), (long) (nTokens - skip) * width);
            parts.get(rank).prefillKV(pos0 + skip, nTokens - skip, peerIn);
        }
        parts.get(0).prefillKV(pos0, nTokens, in);
    }

    @Override
    public String kvRefusal() {
        return null;
    }

    @Override
    public String batchRefusal() {
        return null;
    }

    /**
     * One batched step: the peers run their page shards' batched pipelines first (their taps copy the partitions
     * into rank 0's gather buffers), then rank 0 runs its own and its tap merges. The step inputs are already
     * uploaded ({@code TensorParallelGroup.beginPagedStep} translated and installed each rank's step), so unlike
     * {@link #forward} there is no position upload here.
     */
    @Override
    public void batch(com.eonmux.neuralfree.gguf.memory.StepLayout s, ScratchSlot in, ScratchSlot out) {
        requireBound();
        Objects.requireNonNull(s, "s");
        if (s.contiguousSingle()) {
            if (s.nTokens() == 1) {
                decode(s.pos0(0), in, out);
            } else {
                prefill(s.pos0(0), s.nTokens(), in, out);
            }
            return;
        }
        int width = streamWidth();
        long floats = (long) s.nTokens() * width;
        this.stepRows = s.nTokens() * nHead;
        this.stepSkip = 0;
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerIn = group.inputSlot(rank);
            ScratchSlot peerOut = group.partialSlot(rank);
            peerIn.require(floats);
            peerOut.require(floats);
            RankExchange.replay(ctx.queue(), in.buffer(), in.offset(), group.rankQueue(rank), peerIn.buffer(),
                    floats);
            parts.get(rank).batch(s, peerIn, peerOut);
        }
        parts.get(0).batch(s, in, out);
    }

    @Override
    public void shiftContext(int nKeep, int nDiscard) {
        requireBound();
        String refusal = shiftRefusal();
        if (refusal != null) {
            throw new UnsupportedOperationException("context shift unsupported: " + refusal);
        }
        for (GpuAttentionMixer part : parts) {
            part.shiftContext(nKeep, nDiscard);
        }
    }

    @Override
    public String shiftRefusal() {
        if (!bound) {
            return "layer " + describe() + " is not bound";
        }
        return "a context split shards the sequence between the rank caches, so a shift that moves positions across a"
                + " block boundary cannot move the rows; disable the context shift";
    }

    protected int streamWidth() {
        return layer.residual().streamWidth();
    }

    /** The bound part mixer of a rank (the coordinator's exchange and the peer finish path read them). */
    GpuAttentionMixer part(int rank) {
        return parts.get(rank);
    }

    protected void requireBound() {
        if (!bound) {
            throw new IllegalStateException("layer " + describe() + " is not bound");
        }
    }

    protected String describe() {
        return layer.stack() + "." + layer.il();
    }

    private void release() {
        RuntimeException failure = null;
        List<GpuAttentionMixer> closed = new ArrayList<>(parts);
        parts.clear();
        for (GpuAttentionMixer part : closed) {
            try {
                part.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : suppress(failure, e);
            }
        }
        failure = close(merge, failure);
        merge = null;
        failure = close(gatherOut, failure);
        gatherOut = null;
        failure = close(gatherLse, failure);
        gatherLse = null;
        ctx = null;
        bound = false;
        if (failure != null) {
            throw failure;
        }
    }

    private static RuntimeException suppress(RuntimeException first, RuntimeException next) {
        first.addSuppressed(next);
        return first;
    }

    private static RuntimeException close(AutoCloseable closeable, RuntimeException failure) {
        if (closeable == null) {
            return failure;
        }
        try {
            closeable.close();
            return failure;
        } catch (Exception e) {
            RuntimeException r = e instanceof RuntimeException ? (RuntimeException) e
                    : new IllegalStateException("closing " + closeable + " failed", e);
            return failure == null ? r : suppress(failure, r);
        }
    }

    @Override
    public void close() {
        if (!bound) {
            return;
        }
        release();
    }

    /** Copies one rank's raw attention partition to rank 0; rank 0's copy also merges every partition. */
    private final class RankTap implements AttentionTap {

        private final int rank;

        RankTap(int rank) {
            this.rank = rank;
        }

        @Override
        public boolean ownsOutput() {
            /* Only rank 0 hands the merged attention to the layer's output stages; a peer's output projection is
             * discarded, so it is never computed. */
            return rank == 0;
        }

        @Override
        public void afterAttention(int pos0, int nTokens, GpuBuffer out, GpuBuffer lse, int heads, int dV) {
            int rows = nTokens * heads;
            long outOffset = (long) rank * stepRows * dV + (long) stepSkip * heads * dV;
            long lseOffset = (long) rank * stepRows + (long) stepSkip * heads;
            ComputeQueue root = group.rankQueue(0);
            if (stepSkip > 0) {
                /* The rows this rank skipped above its block carry an empty partition, which the merge weights 0. */
                root.enqueueFillBuffer(gatherOut, (long) rank * stepRows * dV * Float.BYTES,
                        (long) stepSkip * heads * dV * Float.BYTES, 0);
                root.enqueueFillBuffer(gatherLse, (long) rank * stepRows * Float.BYTES,
                        (long) stepSkip * heads * Float.BYTES, Float.floatToRawIntBits(Float.NEGATIVE_INFINITY));
            }
            if (rank == 0) {
                /* The root's own partition is a device copy on the queue that later enqueues the merge. */
                root.enqueueCopyBuffer(out, 0L, gatherOut, outOffset * Float.BYTES,
                        (long) rows * dV * Float.BYTES);
                root.enqueueCopyBuffer(lse, 0L, gatherLse, lseOffset * Float.BYTES,
                        (long) rows * Float.BYTES);
            } else {
                RankExchange.replay(group.rankQueue(rank), out, 0L, root, gatherOut, outOffset,
                        (long) rows * dV);
                RankExchange.replay(group.rankQueue(rank), lse, 0L, root, gatherLse, lseOffset, rows);
                return;
            }
            merge.setInt(AttnKernels.ARG_CM_N_ROWS, stepRows);
            merge.setInt(AttnKernels.ARG_CM_D_V, dV);
            merge.setInt(AttnKernels.ARG_CM_N_RANK, parts.size());
            merge.setBuffer(AttnKernels.ARG_CM_BUF_OUT, out);
            merge.setGlobal((long) stepRows * 128L);
            merge.dispatch();
        }
    }
}
