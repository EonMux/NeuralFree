package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * GPU dual-chunk attention of one layer (Qwen2.5-1M, vLLM's dual-chunk flash-attention backend): the query's chunk
 * ({@code intra}, causal), the preceding chunk ({@code succ}, non-causal) and every earlier chunk ({@code inter},
 * non-causal) are scored by three flash launches, each over its own query rotation plane and key range, into its own
 * {@code (o, m, l)} partial, then folded into one softmax by {@link AttnKernels#DUAL_CHUNK_MERGE}. The dense stages
 * run {@code LLM_FlashAttn_*} (the tensor-core variant when the plan chose one); a paged cache runs
 * {@code LLM_FlashAttnPaged_*} per segment. The Java twin is {@link CpuDualChunkAttention}; the query rotations, the
 * key rotation and the sequence-length scaling are bound by the mixer.
 *
 * <p>Each stage launch reuses the flash prefill kernel with {@code n_split = 1} and {@code direct_out = 0}, so it
 * leaves one {@code (o, m, l)} partial per (row, head). The merge kernel takes the three buffers, resolves
 * {@code m = max(m_s)} and {@code out = sum_s o_s e^(m_s - m) / sum_s l_s e^(m_s - m)}. A stage with no visible key
 * launches over an empty key range and contributes nothing.
 *
 * <p>The {@code kv_pos} visibility of the flash kernels uses the cached keys' original positions, so the intra stage
 * masks causally over the chunk and the succ/inter stages run without the causal bit. Not thread-safe.
 */
final class GpuDualChunkAttention {

    private static final int STAGES = 3;

    private final FlashAttnTiling prefillPlan;
    private final PreparedDispatch[] stages;
    /** The paged stages, or null when the cache is contiguous. */
    private final PreparedDispatch[] pagedStages;
    private final PagedFlashGeometry pg;
    private final PreparedDispatch merge;
    private final ComputeQueue queue;
    private final GpuBuffer origPos;
    private final int[] hostPositions;
    private final int ubatch;
    private final int nHead;
    private final int qDim;
    private final int outDim;
    private final int planePitch;
    private final int chunkLen;
    private final int causalFlags;

    private GpuDualChunkAttention(int chunkLen, FlashAttnTiling prefillPlan, PreparedDispatch[] stages,
                                  PreparedDispatch[] pagedStages, PagedFlashGeometry pg, PreparedDispatch merge,
                                  ComputeQueue queue, GpuBuffer origPos, int ubatch, int nHead, int qDim, int outDim,
                                  int planePitch, int causalFlags) {
        this.prefillPlan = prefillPlan;
        this.stages = stages;
        this.pagedStages = pagedStages;
        this.pg = pg;
        this.merge = merge;
        this.queue = queue;
        this.origPos = origPos;
        this.hostPositions = new int[ubatch];
        this.ubatch = ubatch;
        this.nHead = nHead;
        this.qDim = qDim;
        this.outDim = outDim;
        this.planePitch = planePitch;
        this.chunkLen = chunkLen;
        this.causalFlags = causalFlags;
    }

    /**
     * Prepares the three stage launches, their partial buffers and the merge.
     *
     * @param args       the layer's flash arguments; its Q buffer must hold the five rotation planes at
     *                   {@code s * planePitch}
     * @param pg         the page geometry of a paged cache, or null
     * @param origPos    an {@code int[ubatch]} buffer the stage launches read the chunk's query positions from
     * @param planePitch floats between query planes ({@code ubatch * qDim})
     * @throws UnsupportedOperationException when the cache is paged and the plan has no paged prefill kernel
     */
    static GpuDualChunkAttention prepare(GpuMixerResources res, ComputeQueue queue, FlashPlan plan,
                                         FlashKernelArgs args, PagedFlashGeometry pg, int ubatch, int planePitch,
                                         GpuBuffer origPos) {
        Objects.requireNonNull(res, "res");
        Objects.requireNonNull(queue, "queue");
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(args, "args");
        Objects.requireNonNull(origPos, "origPos");
        AttentionSpec a = args.attention();
        DualChunkSpec dc = a.dualChunk();
        if (dc == null) {
            throw new IllegalArgumentException("dual-chunk attention needs a dual-chunk attention spec");
        }
        if (!a.usesMemory()) {
            throw new IllegalArgumentException("dual-chunk attention needs a KV cache");
        }
        FlashKeys keys = args.keys();
        FlashAttnTiling p = plan.prefill();
        int dV = a.headDimV();
        long partialFloats = (long) ubatch * a.nHead() * (dV + 2L);
        if (partialFloats > Integer.MAX_VALUE / Float.BYTES) {
            throw new IllegalArgumentException("dual-chunk partials of " + partialFloats + " floats exceed one buffer");
        }
        boolean mma = plan.mmaKernel() != null;
        String denseKernel = mma ? plan.mmaKernel() : AttnKernels.flashAttn(keys.typeK(), keys.typeV());
        String pagedKernel = pg != null ? plan.pagedPrefillKernel() : null;
        if (pg != null && pagedKernel == null) {
            throw new UnsupportedOperationException("dual-chunk attention over paged memory needs the paged prefill"
                    + " kernel of this attention plan, which is unavailable on this backend");
        }
        PreparedDispatch[] stages = new PreparedDispatch[STAGES];
        PreparedDispatch[] paged = pg != null ? new PreparedDispatch[STAGES] : null;
        GpuBuffer[] partials = new GpuBuffer[STAGES];
        for (int s = 0; s < STAGES; s++) {
            partials[s] = res.allocate(partialFloats * Float.BYTES);
            FlashKernelArgs stageArgs = args.withPartial(partials[s]);
            Object[] denseArgs = mma ? stageArgs.mmaPrefill(p, ubatch) : stageArgs.tiledPrefill(p, ubatch);
            /* The stage masks against the chunk's own positions, uploaded before each launch (not the step's). */
            denseArgs[AttnKernels.ARG_FA_BUF_Q_POS] = origPos;
            stages[s] = res.prepare(denseKernel, p.prefillGlobal(1, 1), FlashAttnTiling.prefillLocal(), denseArgs);
            if (paged != null) {
                Object[] pagedArgs = stageArgs.pagedPrefill(p, pg, mma);
                pagedArgs[AttnKernels.ARG_FAP_BUF_Q_POS] = origPos;
                paged[s] = res.prepare(pagedKernel, p.prefillGlobal(1, 1), FlashAttnTiling.prefillLocal(), pagedArgs);
            }
        }
        Object[] mergeArgs = new Object[AttnKernels.DCM_ARG_COUNT];
        mergeArgs[AttnKernels.ARG_DCM_N_ROWS] = 1;
        mergeArgs[AttnKernels.ARG_DCM_N_HEAD] = a.nHead();
        mergeArgs[AttnKernels.ARG_DCM_D_V] = dV;
        mergeArgs[AttnKernels.ARG_DCM_OUT_ROW_STRIDE] = a.outDim();
        mergeArgs[AttnKernels.ARG_DCM_OUT_OFF] = 0;
        mergeArgs[AttnKernels.ARG_DCM_BUF_P0] = partials[0];
        mergeArgs[AttnKernels.ARG_DCM_BUF_P1] = partials[1];
        mergeArgs[AttnKernels.ARG_DCM_BUF_P2] = partials[2];
        mergeArgs[AttnKernels.ARG_DCM_BUF_OUT] = args.out();
        PreparedDispatch merge = res.prepare(AttnKernels.DUAL_CHUNK_MERGE, FlashAttnTiling.combineGlobal(1),
                FlashAttnTiling.combineLocal(), mergeArgs);
        return new GpuDualChunkAttention(dc.chunkLength(), p, stages, paged, pg, merge, queue, origPos, ubatch,
                a.nHead(), a.qDim(), a.outDim(), planePitch, args.flags());
    }

    /** The host positions of the step's rows, filled by the mixer before a step runs; index = ubatch row. */
    int[] hostPositions() {
        return hostPositions;
    }

    /**
     * Dual-chunk attention of {@code n} query rows at step rows {@code [firstRow, firstRow + n)} over the first
     * {@code keys} cache rows: the chunk loop of vLLM's prefill (one chunk of queries at a time), or its decode
     * grouping when {@code n == 1}. {@link #hostPositions()} holds the rows' original positions.
     */
    void runDense(int firstRow, int n, int keys) {
        if (keys < n) {
            throw new IllegalStateException("dual-chunk attention over " + keys + " cache rows cannot serve " + n
                    + " query rows");
        }
        int first = keys - n;
        int begin = first;
        while (begin < keys) {
            int prev = begin / chunkLen * chunkLen;
            int end = Math.min(prev + chunkLen, keys);
            int qb = begin - first;
            int len = end - first - qb;
            int row = firstRow + qb;
            writePositions(row, len);
            launch(stages[0], false, 0, row, len, prev, end, causalFlags, 0);
            launch(stages[1], false, 1, row, len, Math.max(0, prev - chunkLen), prev,
                    causalFlags & ~AttnKernels.FLAG_CAUSAL, 0);
            launch(stages[2], false, 2, row, len, 0, Math.max(0, prev - chunkLen),
                    causalFlags & ~AttnKernels.FLAG_CAUSAL, 0);
            mergeRows(row, len);
            begin = end;
        }
    }

    /**
     * Dual-chunk attention of a paged step's segments, each through its sequence's page table and slot count
     * ({@code PagedStep.seqSlots}); segment rows are {@code [segStart, segStart + segLen)} of the step and of
     * {@link #hostPositions()}.
     *
     * @throws UnsupportedOperationException when the cache is not paged
     */
    void runPaged(PagedStep step, int group) {
        if (pagedStages == null) {
            throw new UnsupportedOperationException("dual-chunk attention runs a paged step without a paged cache");
        }
        for (int seg = 0; seg < step.nSeg(); seg++) {
            int start = step.segStart(seg);
            int n = step.segLen(seg);
            int keys = step.seqSlot(group, step.segSeq(seg));
            if (keys < n) {
                throw new IllegalStateException("dual-chunk attention over " + keys + " cache slots cannot serve " + n
                        + " query rows of sequence " + step.segSeq(seg));
            }
            int first = keys - n;
            int ptOff = seg * pg.maxPages();
            int begin = first;
            while (begin < keys) {
                int prev = begin / chunkLen * chunkLen;
                int end = Math.min(prev + chunkLen, keys);
                int qb = begin - first;
                int len = end - first - qb;
                int row = start + qb;
                writePositions(row, len);
                launch(pagedStages[0], true, 0, row, len, prev, end, causalFlags, ptOff);
                launch(pagedStages[1], true, 1, row, len, Math.max(0, prev - chunkLen), prev,
                        causalFlags & ~AttnKernels.FLAG_CAUSAL, ptOff);
                launch(pagedStages[2], true, 2, row, len, 0, Math.max(0, prev - chunkLen),
                        causalFlags & ~AttnKernels.FLAG_CAUSAL, ptOff);
                mergeRows(row, len);
                begin = end;
            }
        }
    }

    /**
     * One stage launch: {@code n} rows from step row {@code row}, reading rotation plane {@code stage} and cache
     * slots {@code [keyStart, keyEnd)}.
     */
    private void launch(PreparedDispatch d, boolean paged, int stage, int row, int n, int keyStart, int keyEnd,
                        int flags, int ptOff) {
        d.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
        d.setInt(AttnKernels.ARG_FA_N_SPLIT, 1);
        d.setInt(AttnKernels.ARG_FA_SPLIT_LEN, Math.max(1, keyEnd - keyStart));
        d.setInt(AttnKernels.ARG_FA_KV_START, keyStart);
        d.setInt(AttnKernels.ARG_FA_KV_END, keyEnd);
        d.setInt(AttnKernels.ARG_FA_FLAGS, flags);
        d.setInt(AttnKernels.ARG_FA_Q_OFF, stage * planePitch + row * qDim);
        d.setInt(AttnKernels.ARG_FA_OUT_OFF, 0);
        if (paged) {
            d.setInt(AttnKernels.ARG_FAP_PT_OFF, ptOff);
        }
        d.setGlobal(prefillPlan.prefillGlobal(n, 1));
        d.dispatch();
    }

    /** The chunk's original positions, read by the stage launches as {@code q_pos[0..n)}. */
    private void writePositions(int row, int n) {
        if (row < 0 || n < 1 || row + n > ubatch) {
            throw new IllegalStateException("dual-chunk stage rows [" + row + ", " + (row + n) + ") exceed the "
                    + ubatch + "-row step");
        }
        ByteBuffer bb = ByteBuffer.allocate(n * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int t = 0; t < n; t++) {
            bb.putInt(hostPositions[row + t]);
        }
        GpuEvent e = queue.enqueueWriteBuffer(origPos, 0, (long) n * Integer.BYTES, bb.array());
        if (e != null) {
            e.close();
        }
    }

    /** The merge of the three stage partials into the output rows of the current chunk. */
    private void mergeRows(int row, int n) {
        merge.setInt(AttnKernels.ARG_DCM_N_ROWS, n);
        merge.setInt(AttnKernels.ARG_DCM_OUT_OFF, row * outDim);
        merge.setGlobal(FlashAttnTiling.combineGlobal(n * nHead));
        merge.dispatch();
    }
}
