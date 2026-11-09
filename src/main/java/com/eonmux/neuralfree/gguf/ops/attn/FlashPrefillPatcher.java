package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;

import java.util.Arrays;

/**
 * The step patcher of one {@link GpuFlashAttention}'s prefill group: from the step's start position alone it
 * derives the values {@link GpuFlashAttention#prefill} re-sets on its kernels every step — the key count
 * ({@code kv_end}) of the cache view, the split count and length and grid the tiling picks from it, the direct
 * flag, and the split counts of the merge launches — and hands them to a captured graph's patch sink, so a
 * replay trunk (the one-row decode step of an MMA plan and the multi-row steps of every plan) replays without
 * re-running the dispatch body. The step's row count is not derivable: the patcher records it from the capture
 * it claims ({@link #accepts} reads the captured {@code n_tokens}), so one plan per capture serves the row
 * count that capture holds and a step of another count rebuilds the plan.
 *
 * <p>The captured dispatches of one group are claimed by identity and order: the prefill kernel is this layer's
 * when it reads this layer's Q and K rows over the whole query block (never a tree step's, a context-parallel
 * binding's or another layer's), and the merge launches are the ones that follow it immediately, in the order
 * {@code prefill} enqueues them — the static sink partial when the layer has one, then the combine, then the
 * log-sum-exp merge. A capture that breaks the order (a foreign kernel inside the group) claims nothing beyond
 * the prefill node, and a step whose merge pattern would leave the capture is refused, so the caller runs the
 * body. {@link #verify(int)} checks the same values against the kernels' current bindings after a body replay,
 * which is how the plan earns the right to skip the body. Not thread-safe.
 */
final class FlashPrefillPatcher implements GpuStepPatcher {

    private final FlashAttnTiling prefillPlan;
    private final boolean cached;
    private final int keyRowBase;
    private final int keyRows;
    private final int nHead;
    private final int dV;
    private final long partialCapacity;
    private final int flags;
    private final boolean hasSink;
    private final boolean hasLse;
    private final GpuBuffer k;
    private final GpuBuffer q;
    private final PreparedDispatch prefill;
    private final PreparedDispatch combine;
    private final PreparedDispatch lse;
    private final PreparedDispatch sink;

    /** The captured dispatch positions this patcher claimed, {@code -1} for a role the capture lacks. */
    private int prefillIdx = -1;
    private int combineIdx = -1;
    private int lseIdx = -1;
    private int sinkIdx = -1;
    /** Merge roles claimed so far, in {@code prefill}'s enqueue order. */
    private int groupClaimed;
    /** True once a capture placed a kernel inside this group that is not one of its merges. */
    private boolean groupBroken;
    /** The row count of the capture this patcher claimed, {@code 0} before the first claim. */
    private int capturedTokens;
    /** The split count of the step the plan was captured at, {@code -1} before the first patch. */
    private int capturedCount = -1;

    FlashPrefillPatcher(FlashAttnTiling prefillPlan, boolean cached, int keyRowBase, int keyRows,
                        int nHead, int dV, long partialCapacity, int flags, boolean hasSink,
                        boolean hasLse, GpuBuffer k, GpuBuffer q, PreparedDispatch prefill, PreparedDispatch combine,
                        PreparedDispatch lse, PreparedDispatch sink) {
        this.prefillPlan = prefillPlan;
        this.cached = cached;
        this.keyRowBase = keyRowBase;
        this.keyRows = keyRows;
        this.nHead = nHead;
        this.dV = dV;
        this.partialCapacity = partialCapacity;
        this.flags = flags;
        this.hasSink = hasSink;
        this.hasLse = hasLse;
        this.k = k;
        this.q = q;
        this.prefill = prefill;
        this.combine = combine;
        this.lse = lse;
        this.sink = sink;
    }

    @Override
    public void resetClaims() {
        prefillIdx = -1;
        combineIdx = -1;
        lseIdx = -1;
        sinkIdx = -1;
        groupClaimed = 0;
        groupBroken = false;
        capturedTokens = 0;
        capturedCount = -1;
    }

    @Override
    public boolean accepts(GpuGraphDispatch d) {
        if (prefillIdx < 0) {
            if (!isGroupPrefill(d)) {
                return false;
            }
            prefillIdx = d.index();
            groupClaimed = 0;
            groupBroken = false;
            capturedTokens = intArg(d.args(), AttnKernels.ARG_FA_N_TOKENS);
            return true;
        }
        if (groupBroken || d.index() != prefillIdx + 1 + groupClaimed) {
            return false;
        }
        String role = nextMergeRole();
        if (!d.kernelName().equals(role)) {
            groupBroken = true;
            return false;
        }
        claimMerge(d.index());
        return true;
    }

    @Override
    public boolean patch(int pos, int recordIndex, GpuGraphPatchSink sink) {
        Step step = stepOf(pos);
        if (!patternMatches(step)) {
            return false;
        }
        if (capturedCount < 0) {
            capturedCount = step.count;
        }
        if (step.count != capturedCount) {
            /* A step whose split layout differs from the capture's would patch a split count the captured
             * launches never ran; the merge pattern and every derived length belong to that layout, so the
             * group refuses and the caller recaptures at the new one. */
            return false;
        }
        if (recordIndex == prefillIdx) {
            sink.patchKernel(recordIndex, prefillPlan.prefillGlobal(capturedTokens, step.count), new int[]{
                            AttnKernels.ARG_FA_N_TOKENS, AttnKernels.ARG_FA_N_SPLIT, AttnKernels.ARG_FA_SPLIT_LEN,
                            AttnKernels.ARG_FA_KV_END, AttnKernels.ARG_FA_FLAGS},
                    new Object[]{capturedTokens, step.layoutSplits, step.splitLen, step.nKeys, step.flags});
            return true;
        }
        if (recordIndex == combineIdx || recordIndex == lseIdx) {
            sink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_FC_N_SPLIT},
                    new Object[]{step.layoutSplits});
            return true;
        }
        if (recordIndex == sinkIdx) {
            sink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_SS_N_SPLIT,
                            AttnKernels.ARG_SS_SPLIT_INDEX},
                    new Object[]{step.layoutSplits, step.count});
            return true;
        }
        return false;
    }

    /** True when this patcher claimed the prefill dispatch of the current capture. */
    boolean claimed() {
        return prefillIdx >= 0;
    }

    @Override
    public boolean verify(int pos) {
        if (prefillIdx < 0) {
            return false;
        }
        Step step = stepOf(pos);
        if (!patternMatches(step)) {
            return false;
        }
        boolean ok = intOf(prefill.argValue(AttnKernels.ARG_FA_N_TOKENS)) == capturedTokens
                && intOf(prefill.argValue(AttnKernels.ARG_FA_N_SPLIT)) == step.layoutSplits
                && intOf(prefill.argValue(AttnKernels.ARG_FA_SPLIT_LEN)) == step.splitLen
                && intOf(prefill.argValue(AttnKernels.ARG_FA_KV_END)) == step.nKeys
                && intOf(prefill.argValue(AttnKernels.ARG_FA_FLAGS)) == step.flags
                && Arrays.equals(prefill.global(), prefillPlan.prefillGlobal(capturedTokens, step.count));
        if (combineIdx >= 0) {
            ok &= intOf(combine.argValue(AttnKernels.ARG_FC_N_ROWS)) == capturedTokens * nHead
                    && intOf(combine.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && Arrays.equals(combine.global(), FlashAttnTiling.combineGlobal(capturedTokens * nHead));
        }
        if (lseIdx >= 0) {
            ok &= intOf(lse.argValue(AttnKernels.ARG_FC_N_ROWS)) == capturedTokens * nHead
                    && intOf(lse.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && Arrays.equals(lse.global(), FlashAttnTiling.combineGlobal(capturedTokens * nHead));
        }
        if (sinkIdx >= 0) {
            long rows = (long) capturedTokens * nHead;
            ok &= intOf(sink.argValue(AttnKernels.ARG_SS_N_ROWS)) == (int) rows
                    && intOf(sink.argValue(AttnKernels.ARG_SS_N_SPLIT)) == step.layoutSplits
                    && intOf(sink.argValue(AttnKernels.ARG_SS_SPLIT_INDEX)) == step.count
                    && Arrays.equals(sink.global(), new long[]{
                            (rows + FlashAttnTiling.WG - 1) / FlashAttnTiling.WG * FlashAttnTiling.WG});
        }
        return ok;
    }

    /** True when the captured merges are exactly the ones a direct step of this layer skips. */
    private boolean patternMatches(Step step) {
        return (combineIdx >= 0) == step.wantsCombine && (sinkIdx >= 0) == hasSink && (lseIdx >= 0) == hasLse;
    }

    /**
     * True when {@code d} is this layer's prefill kernel over the whole query block, no tree mask: the kernel
     * name is the layer's prefill kernel, the Q and K buffers are this layer's, and the captured row count and
     * flag bits say a plain causal step of one row (an MMA plan's decode step) or more.
     */
    private boolean isGroupPrefill(GpuGraphDispatch d) {
        Object[] args = d.args();
        return d.kernelName().equals(prefill.kernelName())
                && args.length > AttnKernels.ARG_FA_BUF_K
                && args[AttnKernels.ARG_FA_BUF_K] == k
                && args[AttnKernels.ARG_FA_BUF_Q] == q
                && intArg(args, AttnKernels.ARG_FA_N_TOKENS) >= 1
                && intArg(args, AttnKernels.ARG_FA_KV_START) == 0
                && intArg(args, AttnKernels.ARG_FA_Q_OFF) == 0
                && intArg(args, AttnKernels.ARG_FA_OUT_OFF) == 0
                && AttnKernels.flagFresh(intArg(args, AttnKernels.ARG_FA_FLAGS)) == 0
                && (intArg(args, AttnKernels.ARG_FA_FLAGS) & AttnKernels.FLAG_TREE) == 0;
    }

    /** The name of the group's next merge role, or null when every role of the layer is claimed. */
    private String nextMergeRole() {
        if (hasSink && sinkIdx < 0) {
            return AttnKernels.STATIC_SINK_PARTIAL;
        }
        if (combineIdx < 0) {
            return AttnKernels.FLASH_COMBINE;
        }
        if (hasLse && lseIdx < 0) {
            return AttnKernels.FLASH_LSE;
        }
        return null;
    }

    private void claimMerge(int index) {
        if (hasSink && sinkIdx < 0) {
            sinkIdx = index;
        } else if (combineIdx < 0) {
            combineIdx = index;
        } else {
            lseIdx = index;
        }
        groupClaimed++;
    }

    /** The values {@link GpuFlashAttention#prefill} sets for one multi-row step at {@code pos0}. */
    private Step stepOf(int pos0) {
        int nKeys = cached ? cachedKeys(pos0 + capturedTokens) : capturedTokens;
        FlashAttnTiling.Split s = prefillPlan.prefillSplit(capturedTokens, nKeys, splitsCapacity());
        boolean direct = s.direct() && !hasSink && !hasLse;
        return new Step(nKeys, s.count(), s.length(), s.count() + (hasSink ? 1 : 0),
                flags | (direct ? AttnKernels.FLAG_DIRECT : 0), direct, !direct);
    }

    /** The partial rows the cached-key splits may fill: the capacity a sink partial's split reserves from. */
    private long splitsCapacity() {
        return hasSink ? Math.max(0L, partialCapacity - (long) capturedTokens * nHead * (dV + 2L)) : partialCapacity;
    }

    /** Cache rows a step ending at {@code upTo - 1} can see of this rank's block: {@code [rowBase, upTo)}. */
    private int cachedKeys(int upTo) {
        long n = (long) upTo - keyRowBase;
        return (int) Math.max(0L, Math.min((long) keyRows, n));
    }

    private static int intArg(Object[] args, int i) {
        return (Integer) args[i];
    }

    private static int intOf(Object value) {
        return (Integer) value;
    }

    /** One step's derived prefill values. */
    private static final class Step {
        final int nKeys;
        final int count;
        final int splitLen;
        final int layoutSplits;
        final int flags;
        final boolean direct;
        final boolean wantsCombine;

        Step(int nKeys, int count, int splitLen, int layoutSplits, int flags, boolean direct, boolean wantsCombine) {
            this.nKeys = nKeys;
            this.count = count;
            this.splitLen = splitLen;
            this.layoutSplits = layoutSplits;
            this.flags = flags;
            this.direct = direct;
            this.wantsCombine = wantsCombine;
        }
    }
}
