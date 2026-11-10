package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

import java.util.Objects;

/**
 * The paged flash launches of one layer over a paged cache, addressed through the step's page tables
 * ({@link PagedFlashGeometry}): one {@code LLM_FlashDecodePaged_*} launch over the step's decode rows when every
 * segment carries one row (the continuous-batching decode) — on CUDA with an F16 or BF16 cache the grouped-query
 * {@code LLM_FlashDecodePagedGqa_*} ({@link FlashPlan#pagedGqaKernel}), which reads each K and V row once for the
 * whole query-head group and splits the keys over the launch's sequences — else one {@code LLM_FlashAttnPaged_*} (or
 * {@code LLM_FlashAttnMmaPaged_F16}) per segment, or the {@link PagedRowFallback} when the prefill plan has no
 * paged prefill kernel. A non-causal step ({@link GpuStepInputs#stepNonCausal()}, a media chunk) lifts the causal
 * mask of its launches; the window rule stays. With batch invariance on, every segment always runs the per-row
 * fallback, so each row's split comes from its own sequence's key count. With a log-sum-exp capture bound
 * ({@code lse}, a context-parallel merge's capture), every launch keeps its partials combined and writes the
 * per-row log-sum-exp like the contiguous step. Not thread-safe.
 */
final class GpuPagedFlashAttention {

    private final FlashAttnTiling decodePlan;
    private final FlashAttnTiling prefillPlan;
    private final PreparedDispatch decode;
    /**
     * The grouped-query paged decode taking every paged decode launch in {@link #decode}'s place; null where it does
     * not serve the caches, and with batch invariance on (its split follows the launch's sequence count).
     */
    private final PreparedDispatch gqaDecode;
    /** The paged prefill; null when segments run the per-row fallback. */
    private final PreparedDispatch prefill;
    /** The per-row fallback; null when the paged prefill exists. */
    private final PagedRowFallback fallback;
    private final FlashCombine combine;
    private final StaticSinkPartial sinkPartial;
    /** The merged log-sum-exp dispatch; null for a plain attention that writes normalized output directly. */
    private final PreparedDispatch lse;
    private final PagedFlashGeometry pg;
    private final GpuStepInputs steps;
    private final long partialCapacity;
    private final int flags;
    private final int nHead;
    private final int nHeadKv;
    private final int qDim;
    private final int outDim;
    private final int dV;
    /** The paged decode-step patcher, made on first use ({@link #patcher()}). */
    private PagedDecodePatcher patcher;

    private GpuPagedFlashAttention(FlashPlan plan, FlashKernelArgs args, PreparedDispatch decode,
                                   PreparedDispatch gqaDecode, PreparedDispatch prefill, PagedRowFallback fallback, FlashCombine combine,
                                   StaticSinkPartial sinkPartial, PreparedDispatch lse, PagedFlashGeometry pg,
                                   GpuStepInputs steps) {
        AttentionSpec a = args.attention();
        this.decodePlan = plan.paged();
        this.prefillPlan = plan.prefill();
        this.decode = decode;
        this.gqaDecode = gqaDecode;
        this.prefill = prefill;
        this.fallback = fallback;
        this.combine = combine;
        this.sinkPartial = sinkPartial;
        this.lse = lse;
        this.pg = pg;
        this.steps = Objects.requireNonNull(steps, "steps");
        this.partialCapacity = plan.partialCapacity();
        this.flags = args.flags();
        this.nHead = a.nHead();
        this.nHeadKv = a.nHeadKv();
        this.qDim = a.qDim();
        this.outDim = a.outDim();
        this.dV = a.headDimV();
    }

    /**
     * Prepares the paged decode (and the grouped-query paged decode of {@link FlashPlan#pagedGqaKernel} with batch
     * invariance off), the paged prefill of {@link FlashPlan#pagedPrefillKernel} or, without one, the per-row
     * fallback's buffers.
     */
    static GpuPagedFlashAttention prepare(GpuMixerResources res, ComputeQueue queue, FlashPlan plan,
                                          FlashKernelArgs args, PagedFlashGeometry pg, GpuStepInputs steps,
                                          FlashCombine combine, StaticSinkPartial sinkPartial,
                                          PreparedDispatch lse) {
        PreparedDispatch decode = res.prepare(plan.pagedDecodeKernel(), plan.paged().decodePagedGlobal(1, 1),
                plan.paged().decodeLocalSize(), args.pagedDecode(plan.paged(), pg));
        boolean invariant = BatchInvariance.enabled();
        String gqaKernel = invariant ? null : plan.pagedGqaKernel();
        PreparedDispatch gqaDecode = gqaKernel != null
                ? res.prepare(gqaKernel, FlashAttnTiling.decodeGqaGlobal(1, args.attention().nHeadKv(), 1),
                        FlashAttnTiling.decodeLocal(), args.pagedDecode(plan.paged(), pg))
                : null;
        String prefillKernel = invariant ? null : plan.pagedPrefillKernel();
        PreparedDispatch prefill = prefillKernel != null
                ? res.prepare(prefillKernel, plan.prefill().prefillGlobal(1, 1), plan.prefill().prefillLocalSize(),
                        args.pagedPrefill(plan.prefill(), pg,
                                plan.prefill().plan() == FlashAttnTiling.Plan.MMA))
                : null;
        AttentionSpec a = args.attention();
        PagedRowFallback fallback = prefill == null
                ? PagedRowFallback.allocate(res, queue, pg, a.qDim(), a.outDim()) : null;
        return new GpuPagedFlashAttention(plan, args, decode, gqaDecode, prefill, fallback, combine, sinkPartial, lse, pg,
                steps);
    }

    /** The attention of paged step {@code step} over KV group {@code group}. */
    void run(PagedStep step, int group) {
        int nSeg = step.nSeg();
        if (steps.treeMask() != null) {
            if (BatchInvariance.enabled()) {
                throw new UnsupportedOperationException("a tree verify step needs the paged prefill kernel, and"
                        + " batch invariance (neuralfree.batch-invariant) runs every segment through the per-row"
                        + " fallback");
            }
            if (prefill == null) {
                throw new UnsupportedOperationException("a tree verify step needs the paged prefill kernel, and"
                        + " this layer's plan runs every segment through the per-row fallback");
            }
        }
        if (BatchInvariance.enabled()) {
            /* One launch per segment's row, sized from that segment's own key count: a batch's largest sequence
             * then cannot change another sequence's split, which is what the single-row step uses. */
            for (int seg = 0; seg < nSeg; seg++) {
                fallback.run(this, step, seg, step.seqSlot(group, step.segSeq(seg)));
            }
            return;
        }
        boolean allSingle = true;
        for (int i = 0; i < nSeg; i++) {
            allSingle &= step.segLen(i) == 1;
        }
        if (allSingle) {
            decodeAll(step, group, nSeg);
            return;
        }
        for (int seg = 0; seg < nSeg; seg++) {
            prefillSegment(step, group, seg);
        }
    }

    /** One paged decode launch over the step's {@code nSeq} one-row segments (launch order = segment order). */
    private void decodeAll(PagedStep step, int group, int nSeq) {
        int maxKeys = 0;
        for (int i = 0; i < nSeq; i++) {
            maxKeys = Math.max(maxKeys, step.seqSlot(group, step.segSeq(i)));
        }
        bindLaunchRows(pg.pageTable(), pg.seqKvLen(), pg.qPosSeq(), qDim, outDim);
        launchDecode(nSeq, maxKeys, 0, 0);
    }

    /** One segment's paged attention: the paged prefill, or the per-row paged decode fallback. */
    private void prefillSegment(PagedStep step, int group, int seg) {
        int len = step.segLen(seg);
        int tokStart = step.segStart(seg);
        int keys = step.seqSlot(group, step.segSeq(seg));
        if (steps.treeMask() != null) {
            // a tree step's segment launches run at their depth positions under their own ancestor bits, both
            // launch-local ([0, len) of the segment positions and mask buffers)
            steps.uploadSegmentTreePositions(tokStart, len);
            steps.uploadTreeMaskSegment(tokStart, len);
        } else {
            steps.uploadSegmentPositions(step.segPos0(seg), len);
        }
        if (prefill == null) {
            fallback.run(this, step, seg, keys);
            return;
        }
        /* Reserve the sink split's rows so the cached splits never fill the whole partial buffer. */
        long splitsCapacity = sinkPartial == null ? partialCapacity
                : Math.max(0L, partialCapacity - (long) len * nHead * (dV + 2L));
        FlashAttnTiling.Split sp = prefillPlan.prefillSplit(len, keys, splitsCapacity);
        boolean direct = sp.direct() && sinkPartial == null && lse == null;
        /* The sink partial takes the split after the cached-key ones, so the cached partials are laid out for one
         * more split than they launch (the extra grid group is never launched). */
        int layoutSplits = sp.count() + (sinkPartial != null ? 1 : 0);
        GpuBuffer tree = steps.treeMask();
        boolean treeStep = tree != null;
        boolean mma = prefillPlan.plan() == FlashAttnTiling.Plan.MMA;
        if (treeStep) {
            /* The launch's fresh rows are the segment's last len: the mask holds the segment's bits launch-local
             * (uploadTreeMaskSegment), bit 0 its first row. */
            prefill.setBuffer(mma ? AttnKernels.ARG_FAP_MMA_BUF_TREE_MASK : AttnKernels.ARG_FAP_BUF_TREE_MASK, tree);
        }
        prefill.setInt(AttnKernels.ARG_FA_N_TOKENS, len);
        prefill.setInt(AttnKernels.ARG_FA_N_SPLIT, layoutSplits);
        prefill.setInt(AttnKernels.ARG_FA_SPLIT_LEN, sp.length());
        prefill.setInt(AttnKernels.ARG_FA_KV_END, keys);
        prefill.setInt(AttnKernels.ARG_FA_FLAGS,
                treeStep ? AttnKernels.treeFlags(pagedFlags(direct), len) : pagedFlags(direct));
        prefill.setInt(AttnKernels.ARG_FA_Q_OFF, tokStart * qDim);
        prefill.setInt(AttnKernels.ARG_FA_OUT_OFF, tokStart * outDim);
        prefill.setInt(AttnKernels.ARG_FAP_PT_OFF, seg * pg.maxPages());
        prefill.setGlobal(prefillPlan.prefillGlobal(len, sp.count()));
        prefill.dispatch();
        if (!direct) {
            int splits = sp.count();
            if (sinkPartial != null) {
                splits = sinkPartial.run(len * nHead, tokStart * qDim, qDim, 0, 1, 0, pg.segPos(), sp.count());
            }
            combine.run(len * nHead, splits, tokStart * outDim);
            if (lse != null) {
                writeLse(len * nHead, splits, tokStart * nHead);
            }
        }
    }

    /**
     * Binds the gathered query rows of a prefill context-parallel batched step (see
     * {@link GpuFlashAttention#bindPagedStepQueries}): the paged prefill and the paged decode launch read
     * {@code q} instead of the packed query rows the mixer projected. The caller restores the mixer's own buffer
     * with {@link #restoreStepQueries} before the next step.
     */
    void bindStepQueries(GpuBuffer q) {
        if (prefill != null) {
            prefill.setBuffer(AttnKernels.ARG_FAP_BUF_Q, q);
        }
        for (PreparedDispatch d : decodeDispatches()) {
            d.setBuffer(AttnKernels.ARG_FDP_BUF_Q, q);
        }
    }

    /** Restores the query buffer a {@link #bindStepQueries} replaced. */
    void restoreStepQueries(GpuBuffer q) {
        if (prefill != null) {
            prefill.setBuffer(AttnKernels.ARG_FAP_BUF_Q, q);
        }
        for (PreparedDispatch d : decodeDispatches()) {
            d.setBuffer(AttnKernels.ARG_FDP_BUF_Q, q);
        }
    }

    /**
     * Points the paged decode's launch rows at a page table, its slot counts and query positions, with the Q and
     * output rows of launch row {@code s} at {@code s * qSeqStride} and {@code s * outSeqStride} past the offsets.
     */
    void bindLaunchRows(GpuBuffer pageTable, GpuBuffer seqKvLen, GpuBuffer qPosSeq, int qSeqStride,
                        int outSeqStride) {
        for (PreparedDispatch d : decodeDispatches()) {
            d.setBuffer(AttnKernels.ARG_FDP_BUF_PAGE_TABLE, pageTable);
            d.setBuffer(AttnKernels.ARG_FDP_BUF_SEQ_KV_LEN, seqKvLen);
            d.setBuffer(AttnKernels.ARG_FDP_BUF_Q_POS_SEQ, qPosSeq);
            d.setInt(AttnKernels.ARG_FDP_Q_SEQ_STRIDE, qSeqStride);
            d.setInt(AttnKernels.ARG_FDP_OUT_SEQ_STRIDE, outSeqStride);
        }
    }

    /**
     * The decode-step patcher of this attention's plain paged decode launch (one row per sequence,
     * {@link #run}'s every-segment-single-row form): claims this layer's paged decode kernel (the grouped-query one
     * where it serves) and its merges in a captured graph and patches them from the step's key count, replacing the
     * body {@link #launchDecode} re-runs on a replay. Null with batch invariance on, where every row launches alone.
     */
    PagedDecodePatcher patcher() {
        if (BatchInvariance.enabled()) {
            return null;
        }
        if (patcher == null) {
            boolean grouped = gqaDecode != null;
            patcher = new PagedDecodePatcher(decodePlan, grouped, nHead, nHeadKv, sinkPartial != null, lse != null,
                    grouped ? gqaDecode : decode, combine.dispatch(), lse,
                    sinkPartial != null ? sinkPartial.dispatch() : null);
        }
        return patcher;
    }

    /** The paged decode dispatches whose launch-row bindings move together: the plain one and the grouped one. */
    private PreparedDispatch[] decodeDispatches() {
        return gqaDecode == null ? new PreparedDispatch[]{decode} : new PreparedDispatch[]{decode, gqaDecode};
    }

    /**
     * One paged decode launch over the first {@code nSeq} launch rows, split for {@code keys} keys, reading Q from
     * float offset {@code qOff} and writing (or combining into) the output from {@code outOff}.
     */
    void launchDecode(int nSeq, int keys, int qOff, int outOff) {
        launchDecode(nSeq, keys, qOff, outOff, 0, true);
    }

    /** A launch whose {@code nSeq} rows all read the one query position {@code fixedPos} (the per-row fallback). */
    void launchDecodeAt(int nSeq, int keys, int qOff, int outOff, int fixedPos) {
        launchDecode(nSeq, keys, qOff, outOff, fixedPos, false);
    }

    private void launchDecode(int nSeq, int keys, int qOff, int outOff, int fixedPos, boolean perSequencePositions) {
        boolean grouped = gqaDecode != null;
        FlashAttnTiling.Split sp = grouped ? decodePlan.decodeGqaSplit(keys, nSeq) : decodePlan.decodeSplit(keys);
        PreparedDispatch d = grouped ? gqaDecode : decode;
        boolean direct = decodeWritesDirect(sp, grouped && decodePlan.decodeGqaSplitsEarly(nSeq), sinkPartial != null,
                lse != null);
        int layoutSplits = sp.count() + (sinkPartial != null ? 1 : 0);
        d.setInt(AttnKernels.ARG_FD_N_SPLIT, layoutSplits);
        d.setInt(AttnKernels.ARG_FD_SPLIT_LEN, sp.length());
        d.setInt(AttnKernels.ARG_FD_FLAGS, pagedFlags(direct));
        d.setInt(AttnKernels.ARG_FDP_N_SEQ, nSeq);
        d.setInt(AttnKernels.ARG_FD_Q_OFF, qOff);
        d.setInt(AttnKernels.ARG_FD_OUT_OFF, outOff);
        d.setGlobal(grouped ? FlashAttnTiling.decodeGqaGlobal(nSeq, nHeadKv, sp.count())
                : decodePlan.decodePagedGlobal(nSeq, sp.count()));
        d.dispatch();
        if (!direct) {
            int splits = sp.count();
            if (sinkPartial != null) {
                splits = perSequencePositions
                        ? sinkPartial.run(nSeq * nHead, qOff, qDim, 0, 1, 0, pg.qPosSeq(), sp.count())
                        : sinkPartial.run(nSeq * nHead, qOff, qDim, 0, 0, fixedPos, null, sp.count());
            }
            combine.run(nSeq * nHead, splits, outOff);
            if (lse != null) {
                writeLse(nSeq * nHead, splits, (qOff / qDim) * nHead);
            }
        }
    }

    /**
     * Whether a paged decode launch of split {@code sp} writes its normalized output itself instead of partials the
     * merge combines: one split, no static sink partial and no log-sum-exp capture, unless the launch merges its one
     * split too ({@code mergesOneSplit}). The grouped-query kernel does where its plan splits the keys as soon as
     * they outgrow one split ({@link FlashAttnTiling#decodeGqaSplitsEarly}, at 33 keys): the merge of one partial
     * reproduces the direct output bit for bit (it rescales the partial by exp(0) = 1 and divides by the same sum),
     * so a decode trunk keeps one node sequence from its first key on and a captured step graph serves the steps
     * past that boundary instead of being recaptured there. Where the sequences alone fill the device, the plan
     * keeps one split up to {@link FlashAttnTiling#GQA_MAX_SPLIT} keys, so the launch writes directly and every
     * step there is spared the combine launch; a trunk that outgrows that split is recaptured once.
     * {@link PagedDecodePatcher} derives the same rule.
     */
    static boolean decodeWritesDirect(FlashAttnTiling.Split sp, boolean mergesOneSplit, boolean hasSink,
                                      boolean hasLse) {
        return sp.direct() && !mergesOneSplit && !hasSink && !hasLse;
    }

    /** Writes the merged log-sum-exp of {@code splits} partials of {@code rows} rows at {@code offset}. */
    private void writeLse(int rows, int splits, int offset) {
        lse.setInt(AttnKernels.ARG_FC_N_ROWS, rows);
        lse.setInt(AttnKernels.ARG_FC_N_SPLIT, splits);
        lse.setInt(AttnKernels.ARG_FC_OUT_OFF, offset);
        lse.setGlobal(FlashAttnTiling.combineGlobal(rows));
        lse.dispatch();
    }

    /**
     * The packed flags of a paged launch: a non-causal step drops {@link AttnKernels#FLAG_CAUSAL}, and
     * {@code direct} adds {@link AttnKernels#FLAG_DIRECT}.
     */
    private int pagedFlags(boolean direct) {
        int f = steps.stepNonCausal() ? flags & ~AttnKernels.FLAG_CAUSAL : flags;
        return direct ? f | AttnKernels.FLAG_DIRECT : f;
    }
}
