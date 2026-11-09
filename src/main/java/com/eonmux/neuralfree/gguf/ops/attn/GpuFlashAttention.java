package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.BatchInvariance;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuStepInputs;
import com.eonmux.neuralfree.gguf.memory.paged.PagedStep;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;

/**
 * Flash attention of one softmax-attention layer over its {@link FlashKeys}: the decode kernel per query row, the
 * prefill kernel of the {@link FlashPlan} (tiled {@code LLM_FlashAttn_*} or the tensor-core
 * {@code LLM_FlashAttnMma*}), the per-row decode fallback of a {@link FlashAttnTiling.Plan#PER_ROW_DECODE} plan,
 * and the split-K {@link FlashCombine}. Over a paged cache it also runs the batched step through
 * {@link GpuPagedFlashAttention}. Its dispatches belong to the mixer's {@link GpuMixerResources}. Not thread-safe.
 */
final class GpuFlashAttention {

    private final FlashAttnTiling decodePlan;
    private final FlashAttnTiling prefillPlan;
    private final PreparedDispatch decode;
    /** The tiled or mma prefill; null when the prefill runs the decode kernel per row. */
    private final PreparedDispatch prefill;
    private final FlashCombine combine;
    /** The paged launches; null when the keys are not a paged cache. */
    private final GpuPagedFlashAttention paged;
    private final boolean cached;
    private final boolean causal;
    private final int keyRows;
    private final int keyRowBase;
    private final int nHead;
    private final int qDim;
    private final int outDim;
    private final int dV;
    private final long partialCapacity;
    private final int flags;
    /** The learned static sink-token partial, or null when the layer has no sink tokens. */
    private final StaticSinkPartial sinkPartial;
    /** The merged log-sum-exp dispatch; null for a plain attention that writes normalized output directly. */
    private final PreparedDispatch lse;
    /** The key and query buffers of this layer's decode kernel, the identity of its graph dispatches. */
    private final GpuBuffer k;
    private final GpuBuffer q;
    /** True when the plan's tensor-core prefill serves the one-row step faster than the decode kernel. */
    private final boolean routesDecodeThroughPrefill;
    /** True when the one-row route follows the step's live key count (an mma prefill and a dense fast-path plan). */
    private final boolean routesDecodeByDepth;
    /** The decode-step patcher; created on first use. */
    private FlashDecodePatcher patcher;
    /** The multi-row step patcher; created on first use, null while the plan runs the decode kernel per row. */
    private FlashPrefillPatcher prefillPatcher;
    private int decodeSplits = 1;

    private GpuFlashAttention(FlashPlan plan, FlashKernelArgs args, PreparedDispatch decode,
                              PreparedDispatch prefill, FlashCombine combine, GpuPagedFlashAttention paged,
                              StaticSinkPartial sinkPartial, PreparedDispatch lse) {
        AttentionSpec a = args.attention();
        this.decodePlan = plan.decode();
        this.prefillPlan = plan.prefill();
        this.decode = decode;
        this.prefill = prefill;
        this.combine = combine;
        this.paged = paged;
        this.cached = args.keys().cached();
        this.causal = a.causal();
        this.keyRows = args.keys().rows();
        this.keyRowBase = args.keys().rowBase();
        this.nHead = a.nHead();
        this.qDim = a.qDim();
        this.outDim = a.outDim();
        this.dV = a.headDimV();
        this.partialCapacity = plan.partialCapacity();
        this.flags = args.flags();
        this.sinkPartial = sinkPartial;
        this.lse = lse;
        this.k = args.keys().k();
        this.q = args.q();
        this.routesDecodeThroughPrefill = plan.routesDecodeThroughPrefill();
        this.routesDecodeByDepth = plan.routesDecodeByDepth();
    }

    /**
     * True when this step's one-row attention runs the tensor-core prefill launch: the plan-level route
     * ({@link FlashPlan#routesDecodeThroughPrefill}) plus, on a dense fast-path plan with an mma kernel, every
     * step over at most {@link FlashAttnTiling#WG} live keys — the workgroup bound at which the decode plan stops
     * splitting, where the paired benches put the prefill route ahead.
     */
    boolean routesDecodeThroughPrefill(int liveKeys) {
        return routesDecodeThroughPrefill || (routesDecodeByDepth && liveKeys <= FlashAttnTiling.WG);
    }

    /** True when the one-row route follows the step's live key count, so either kernel may run a replayed step. */
    boolean routesDecodeByDepth() {
        return routesDecodeByDepth;
    }

    /**
     * The decode-step patcher of this attention's group: claims this layer's decode kernel and its
     * merges in a captured graph and patches them from the step position, replacing the body
     * {@link #decodeRow} re-runs on a replay.
     */
    FlashDecodePatcher patcher() {
        if (patcher == null) {
            patcher = new FlashDecodePatcher(decodePlan, cached, keyRowBase, keyRows, nHead, flags,
                    sinkPartial != null, lse != null, k, q, decode, combine.dispatch(), lse,
                    sinkPartial != null ? sinkPartial.dispatch() : null);
        }
        return patcher;
    }

    /**
     * The multi-row step patcher of this attention's group, or null when the prefill runs the decode kernel
     * per row ({@link FlashAttnTiling.Plan#PER_ROW_DECODE}): claims this layer's prefill kernel and its
     * merges in a captured graph and patches them from the step position and the captured row count,
     * replacing the body {@link #prefill} re-runs on a replay.
     */
    FlashPrefillPatcher prefillPatcher() {
        if (prefill == null) {
            return null;
        }
        if (prefillPatcher == null) {
            prefillPatcher = new FlashPrefillPatcher(prefillPlan, cached, keyRowBase, keyRows, nHead,
                    dV, partialCapacity, flags, sinkPartial != null, lse != null, k, q, prefill,
                    combine.dispatch(), lse, sinkPartial != null ? sinkPartial.dispatch() : null);
        }
        return prefillPatcher;
    }

    /**
     * Prepares the launches of {@code plan} with the arguments {@code args}: the decode, the prefill unless the plan
     * runs per row, the combine and, when {@code pg} is not null, the paged launches.
     *
     * @param pg        the page geometry of a paged cache, or null
     * @param steps     the step inputs the paged launches upload segment positions to; unused without {@code pg}
     * @param lseBuffer the per-row log-sum-exp output a context-parallel merge reads, or null for a plain attention
     *                  that writes its normalized output directly whenever the split allows it
     */
    static GpuFlashAttention prepare(GpuMixerResources res, ComputeQueue queue, FlashPlan plan,
                                     FlashKernelArgs args, PagedFlashGeometry pg, GpuStepInputs steps, int ubatch,
                                     GpuBuffer lseBuffer) {
        FlashKeys keys = args.keys();
        int nHead = args.attention().nHead();
        PreparedDispatch decode = res.prepare(AttnKernels.flashDecode(keys.typeK(), keys.typeV()),
                plan.decode().decodeGlobal(1), FlashAttnTiling.decodeLocal(), args.decode(plan.decode()));
        PreparedDispatch prefill = BatchInvariance.enabled() ? null : preparePrefill(res, plan, args, ubatch);
        FlashCombine combine = new FlashCombine(res.prepare(AttnKernels.FLASH_COMBINE,
                FlashAttnTiling.combineGlobal(nHead), FlashAttnTiling.combineLocal(), args.combine()), nHead);
        PreparedDispatch lse = lseBuffer == null ? null
                : res.prepare(AttnKernels.FLASH_LSE, FlashAttnTiling.combineGlobal(nHead),
                        FlashAttnTiling.combineLocal(), args.lse(lseBuffer));
        StaticSinkPartial sinkPartial = args.attention().sinkTokens() > 0
                ? StaticSinkPartial.prepare(res, args.attention(), args.q(), args.qPos(), args.staticSinkK(),
                        args.staticSinkV(), args.partial())
                : null;
        GpuPagedFlashAttention paged = pg != null
                ? GpuPagedFlashAttention.prepare(res, queue, plan, args, pg, steps, combine, sinkPartial, lse) : null;
        return new GpuFlashAttention(plan, args, decode, prefill, combine, paged, sinkPartial, lse);
    }

    /** The learned static sink-token partial, or null without sink tokens. */
    StaticSinkPartial sinkPartial() {
        return sinkPartial;
    }

    /** True when the plan's tensor-core prefill serves the one-row step faster than the decode kernel. */
    boolean routesDecodeThroughPrefill() {
        return routesDecodeThroughPrefill;
    }

    /** The prefill dispatch of a TILED or MMA plan over {@code ubatch} rows; null for a per-row plan. */
    private static PreparedDispatch preparePrefill(GpuMixerResources res, FlashPlan plan, FlashKernelArgs args,
                                                   int ubatch) {
        FlashAttnTiling p = plan.prefill();
        FlashKeys keys = args.keys();
        if (p.plan() == FlashAttnTiling.Plan.TILED) {
            return res.prepare(AttnKernels.flashAttn(keys.typeK(), keys.typeV()), p.prefillGlobal(ubatch, 1),
                    FlashAttnTiling.prefillLocal(), args.tiledPrefill(p, ubatch));
        }
        if (p.plan() == FlashAttnTiling.Plan.MMA) {
            return res.prepare(plan.mmaKernel(), p.prefillGlobal(ubatch, 1), FlashAttnTiling.prefillLocal(),
                    args.mmaPrefill(p, ubatch));
        }
        return null;
    }

    /** One query row at position {@code pos}: the cache rows up to it, or the single batch row. */
    void decode(int pos) {
        decode(pos, null);
    }

    /**
     * {@link #decode(int)} under a tree verify step's mask: the row is the step's one fresh row, so it sees only
     * the fresh slot its mask word names (bit 0) ahead of the ordinary rules.
     */
    void decode(int pos, GpuBuffer treeMask) {
        decodeRow(pos, 0, cached ? cachedKeys(pos + 1) : 1, treeMask);
    }

    /**
     * Binds the query rows of a prefill context-parallel attention: {@code q} holds the whole step's rows in the
     * query's global row order and {@code qPos} their positions, so the prefill launch reads rows this rank does
     * not hold packed. The per-row decode fallback reads {@code q} through the same binding and takes each row's
     * position as the scalar it always does. The caller restores the mixer's own buffers with
     * {@link #restoreStepQueries} before the next step.
     */
    void bindStepQueries(GpuBuffer q, GpuBuffer qPos) {
        if (prefill != null) {
            prefill.setBuffer(AttnKernels.ARG_FA_BUF_Q, q);
            prefill.setBuffer(AttnKernels.ARG_FA_BUF_Q_POS, qPos);
        }
        decode.setBuffer(AttnKernels.ARG_FD_BUF_Q, q);
    }

    /** Restores the query buffers a {@link #bindStepQueries} replaced. */
    void restoreStepQueries(GpuBuffer q, GpuBuffer qPos) {
        if (prefill != null) {
            prefill.setBuffer(AttnKernels.ARG_FA_BUF_Q, q);
            prefill.setBuffer(AttnKernels.ARG_FA_BUF_Q_POS, qPos);
        }
        decode.setBuffer(AttnKernels.ARG_FD_BUF_Q, q);
    }

    /**
     * Binds the gathered query rows of a paged batched step: the paged prefill and the paged decode read
     * {@code q} at their per-segment offsets instead of the packed rows this mixer projected. Their query
     * positions stay the prepared buffers — the segment positions the launches upload per segment and the
     * per-sequence query positions, which carry the true positions of the step's segments. The caller restores
     * the mixer's own buffer with {@link #restorePagedStepQueries} before the next step.
     *
     * @throws UnsupportedOperationException when the keys are not a paged cache
     */
    void bindPagedStepQueries(GpuBuffer q) {
        if (paged == null) {
            throw new UnsupportedOperationException("the paged step queries bind only over a paged cache");
        }
        paged.bindStepQueries(q);
    }

    /** Restores the query buffer a {@link #bindPagedStepQueries} replaced. */
    void restorePagedStepQueries(GpuBuffer q) {
        if (paged != null) {
            paged.restoreStepQueries(q);
        }
    }

    /**
     * Cache rows a query at {@code upTo - 1} can see of this rank's block: the rows holding positions
     * {@code [rowBase, upTo)}, clamped to the shard. A whole cache has {@code rowBase 0}.
     */
    private int cachedKeys(int upTo) {
        long n = (long) upTo - keyRowBase;
        return (int) Math.max(0L, Math.min((long) keyRows, n));
    }

    /**
     * {@code n} query rows from {@code pos0}: one prefill launch (split while the call is small, then combined), or
     * one decode launch per row, each admitting the keys up to its own position under a causal cache. Batch
     * invariance always takes the per-row decode launches: a row's split then depends on its own live key count
     * only, exactly as the single-row decode step's does.
     */
    void prefill(int pos0, int n) {
        prefill(pos0, n, null);
    }

    /**
     * {@code n} query rows from {@code pos0} under the step's tree mask, or the plain attention with a null
     * {@code treeMask}: one prefill launch (split while the call is small, then combined), or one decode launch per
     * row, each admitting the keys up to its own position under a causal cache. Batch invariance always takes the
     * per-row decode launches: a row's split then depends on its own live key count only, exactly as the single-row
     * decode step's does.
     *
     * @param treeMask the step's tree mask over the {@code n} rows (the step's fresh rows are its last {@code n}
     *                 slots), or null outside a tree verify step
     * @throws UnsupportedOperationException when a tree mask is set but this layer's plan cannot take one (a
     *                                       per-row plan, batch invariance, a context-parallel output or a sharded
     *                                       cache whose live rows stop short of the step's own)
     */
    void prefill(int pos0, int n, GpuBuffer treeMask) {
        int nKeys = cached ? cachedKeys(pos0 + n) : n;
        if (treeMask != null) {
            String refusal = prefillTreeRefusal(pos0, n, nKeys);
            if (refusal != null) {
                throw new UnsupportedOperationException(refusal);
            }
        }
        if (prefill != null) {
            /* Reserve the sink split's rows so the cached splits never fill the whole partial buffer. */
            long splitsCapacity = sinkPartial == null ? partialCapacity
                    : Math.max(0L, partialCapacity - (long) n * nHead * (dV + 2L));
            FlashAttnTiling.Split s = prefillPlan.prefillSplit(n, nKeys, splitsCapacity);
            boolean direct = s.direct() && sinkPartial == null && lse == null;
            /* The sink partial takes the split after the cached-key ones, so the cached partials are laid out for
             * one more split than they launch (the extra grid group is never launched). */
            int layoutSplits = s.count() + (sinkPartial != null ? 1 : 0);
            boolean tree = treeMask != null;
            boolean mma = prefillPlan.plan() == FlashAttnTiling.Plan.MMA;
            if (tree) {
                prefill.setBuffer(mma ? AttnKernels.ARG_FA_MMA_BUF_TREE_MASK : AttnKernels.ARG_FA_BUF_TREE_MASK,
                        treeMask);
            }
            prefill.setInt(AttnKernels.ARG_FA_N_TOKENS, n);
            prefill.setInt(AttnKernels.ARG_FA_N_SPLIT, layoutSplits);
            prefill.setInt(AttnKernels.ARG_FA_SPLIT_LEN, s.length());
            prefill.setInt(AttnKernels.ARG_FA_KV_END, nKeys);
            prefill.setInt(AttnKernels.ARG_FA_FLAGS, stepFlags(flags, direct, tree, n));
            prefill.setGlobal(prefillPlan.prefillGlobal(n, s.count()));
            prefill.dispatch();
            if (!direct) {
                int splits = s.count();
                if (sinkPartial != null) {
                    splits = sinkPartial.run(n * nHead, 0, qDim, 0, 1, 0, null, s.count());
                }
                combine.run(n * nHead, splits, 0);
                if (lse != null) {
                    writeLse(n * nHead, splits, 0);
                }
            }
            return;
        }
        for (int t = 0; t < n; t++) {
            int qPos = pos0 + t;
            int keys = cached && causal ? cachedKeys(qPos + 1) : nKeys;
            decodeRow(qPos, t, keys, treeMask);
        }
    }

    /** Why the tree-masked prefill cannot run, or null when it can. */
    private String prefillTreeRefusal(int pos0, int n, int nKeys) {
        if (prefill == null) {
            return "a tree verify step needs the tiled or tensor-core prefill kernel, and this layer's plan runs"
                    + " the decode kernel per row";
        }
        if (lse != null) {
            return "a tree verify step runs one whole sequence's cache, not the sharded cache of a context-parallel"
                    + " attention";
        }
        if (nKeys != pos0 + n) {
            return "a tree verify step's fresh rows are the last slots of the key range, but this layer's cache"
                    + " holds " + nKeys + " live rows for positions [" + pos0 + ", " + (pos0 + n) + ")";
        }
        return null;
    }

    /**
     * The launch's {@code flags}: the layer's base bits with the split's direct bit and, for a tree step, the tree
     * bit with the step's fresh-row count packed in bits 5-31 ({@link AttnKernels#treeFlags}).
     */
    private static int stepFlags(int base, boolean direct, boolean tree, int nFresh) {
        int f = base | (direct ? AttnKernels.FLAG_DIRECT : 0);
        return tree ? AttnKernels.treeFlags(f, nFresh) : f;
    }

    /**
     * The batched step's attention over the paged step {@code step}, whose KV group is {@code group}
     * ({@link GpuPagedFlashAttention#run}); with a log-sum-exp capture bound, every launch keeps its split
     * partials combined and writes the per-row log-sum-exp like the contiguous step.
     *
     * @throws UnsupportedOperationException when the keys are not a paged cache
     */
    void batched(PagedStep step, int group) {
        if (paged == null) {
            throw new UnsupportedOperationException("flash attention runs a batched step without a paged cache");
        }
        paged.run(step, group);
    }

    /**
     * One query row of the decode kernel. Split count and length follow the live key count
     * ({@link FlashAttnTiling#decodeSplit}); the grid is re-set only when the count changes. {@code treeMask} is
     * the step's tree mask (the row reads its word at {@code tree_mask[0]}, the step's one fresh row), or null.
     */
    private void decodeRow(int qPos, int row, int nKeys, GpuBuffer treeMask) {
        FlashAttnTiling.Split s = decodePlan.decodeSplit(nKeys);
        int count = s.count();
        boolean direct = s.direct() && sinkPartial == null && lse == null;
        int layoutSplits = count + (sinkPartial != null ? 1 : 0);
        boolean tree = treeMask != null;
        if (tree) {
            decode.setBuffer(AttnKernels.ARG_FD_BUF_TREE_MASK, treeMask);
        }
        decode.setInt(AttnKernels.ARG_FD_N_SPLIT, layoutSplits);
        decode.setInt(AttnKernels.ARG_FD_SPLIT_LEN, s.length());
        decode.setInt(AttnKernels.ARG_FD_KV_END, nKeys);
        decode.setInt(AttnKernels.ARG_FD_Q_POS, qPos);
        decode.setInt(AttnKernels.ARG_FD_FLAGS, stepFlags(flags, direct, tree, 1));
        decode.setInt(AttnKernels.ARG_FD_Q_OFF, row * qDim);
        decode.setInt(AttnKernels.ARG_FD_OUT_OFF, direct ? row * outDim : 0);
        if (count != decodeSplits) {
            decode.setGlobal(decodePlan.decodeGlobal(count));
            decodeSplits = count;
        }
        decode.dispatch();
        if (!direct) {
            int splits = count;
            if (sinkPartial != null) {
                /* One query row: its position is the scalar the decode launch received. */
                splits = sinkPartial.run(nHead, row * qDim, qDim, 0, 0, qPos, null, count);
            }
            combine.run(nHead, splits, row * outDim);
            if (lse != null) {
                writeLse(nHead, splits, row * nHead);
            }
        }
    }

    /** Writes the merged log-sum-exp of {@code splits} partials of {@code rows} rows at {@code offset}. */
    private void writeLse(int rows, int splits, int offset) {
        lse.setInt(AttnKernels.ARG_FC_N_ROWS, rows);
        lse.setInt(AttnKernels.ARG_FC_N_SPLIT, splits);
        lse.setInt(AttnKernels.ARG_FC_OUT_OFF, offset);
        lse.setGlobal(FlashAttnTiling.combineGlobal(rows));
        lse.dispatch();
    }
}
