package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;

import java.util.Arrays;

/**
 * The paged decode-step patcher of one {@link GpuPagedFlashAttention}: for the plain paged decode step (one row per
 * sequence, every row in one launch) it derives, from the step's key count alone, the values the launch re-sets
 * every step — the split count and length and the grid the tiling picks for the captured sequence count, the direct
 * flag, the static sink partial's split count and index, and the split count of the merge and log-sum-exp
 * launches — and hands them to a captured graph's patch sink, so the paged decode trunk replays without re-running
 * the dispatch body. The step's key count is the replay position: the largest live key count of the step's
 * sequences, which every decode step of the same sequences advances by one. Every other input of the launch (the
 * page tables, the per-sequence key counts and query positions) lives in device buffers the step uploads outside the
 * trunk.
 *
 * <p>The captured dispatches are claimed by identity and order: the decode kernel is this layer's when it is the
 * kernel the paged decode launches (the grouped-query one where it serves), reads this layer's K buffer, starts at
 * query and output offset 0 and carries no tree mask; the merge launches are the ones that follow it immediately in
 * the order the launch enqueues them — the static sink partial when the layer has one, then the combine, then the
 * log-sum-exp merge. The captured sequence count fixes the launch geometry. A step whose merge pattern would leave
 * the capture (a direct launch against a captured split one, or the reverse; on the grouped-query kernel only where
 * its sequences keep one split up to its longest split, since it merges at every split count elsewhere:
 * {@link GpuPagedFlashAttention#decodeWritesDirect}) is refused, so the caller recaptures;
 * a step that only changes the split count and length patches the grid with them. {@link #verify(int)} checks the
 * same values against the kernels' current bindings after a body replay, which is how the plan earns the right to
 * skip the body. Not thread-safe.</p>
 */
final class PagedDecodePatcher implements GpuStepPatcher {

    private final FlashAttnTiling decodePlan;
    private final boolean grouped;
    private final int nHead;
    private final int nHeadKv;
    private final boolean hasSink;
    private final boolean hasLse;
    private final Object k;
    private final PreparedDispatch decode;
    private final PreparedDispatch combine;
    private final PreparedDispatch lse;
    private final PreparedDispatch sink;

    /** The captured dispatch positions this patcher claimed, {@code -1} for a role the capture lacks. */
    private int decodeIdx = -1;
    private int combineIdx = -1;
    private int lseIdx = -1;
    private int sinkIdx = -1;
    /** Merge roles claimed so far, in the launch's order. */
    private int groupClaimed;
    /** True once a capture placed a kernel inside this group that is not one of its merges. */
    private boolean groupBroken;
    /** The sequence count of the captured decode launch. */
    private int nSeq;
    /** The captured decode flags without {@link AttnKernels#FLAG_DIRECT}. */
    private int baseFlags;

    /**
     * @param grouped true when {@code decode} is the grouped-query kernel ({@link FlashAttnTiling#decodeGqaSplit}),
     *                false for the plain paged decode ({@link FlashAttnTiling#decodeSplit})
     */
    PagedDecodePatcher(FlashAttnTiling decodePlan, boolean grouped, int nHead, int nHeadKv, boolean hasSink,
                       boolean hasLse, PreparedDispatch decode, PreparedDispatch combine, PreparedDispatch lse,
                       PreparedDispatch sink) {
        this.decodePlan = decodePlan;
        this.grouped = grouped;
        this.nHead = nHead;
        this.nHeadKv = nHeadKv;
        this.hasSink = hasSink;
        this.hasLse = hasLse;
        this.k = decode.argValue(AttnKernels.ARG_FDP_BUF_K);
        this.decode = decode;
        this.combine = combine;
        this.lse = lse;
        this.sink = sink;
    }

    @Override
    public void resetClaims() {
        decodeIdx = -1;
        combineIdx = -1;
        lseIdx = -1;
        sinkIdx = -1;
        groupClaimed = 0;
        groupBroken = false;
        nSeq = 0;
        baseFlags = 0;
    }

    @Override
    public boolean accepts(GpuGraphDispatch d) {
        if (decodeIdx < 0) {
            if (!isGroupDecode(d)) {
                return false;
            }
            decodeIdx = d.index();
            nSeq = intArg(d.args(), AttnKernels.ARG_FDP_N_SEQ);
            baseFlags = intArg(d.args(), AttnKernels.ARG_FD_FLAGS) & ~AttnKernels.FLAG_DIRECT;
            groupClaimed = 0;
            groupBroken = false;
            return true;
        }
        if (groupBroken || d.index() != decodeIdx + 1 + groupClaimed) {
            return false;
        }
        String role = nextMergeRole();
        if (role == null || !d.kernelName().equals(role)) {
            groupBroken = true;
            return false;
        }
        claimMerge(d.index());
        return true;
    }

    @Override
    public boolean patch(int pos, int recordIndex, GpuGraphPatchSink patchSink) {
        Step step = stepOf(pos);
        if (step == null || !patternMatches(step)) {
            return false;
        }
        if (recordIndex == decodeIdx) {
            patchSink.patchKernel(recordIndex, decodeGlobal(step.count), new int[]{
                    AttnKernels.ARG_FD_N_SPLIT, AttnKernels.ARG_FD_SPLIT_LEN, AttnKernels.ARG_FD_FLAGS},
                    new Object[]{step.layoutSplits, step.splitLen, step.flags});
            return true;
        }
        if (recordIndex == combineIdx || recordIndex == lseIdx) {
            patchSink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_FC_N_SPLIT},
                    new Object[]{step.layoutSplits});
            return true;
        }
        if (recordIndex == sinkIdx) {
            patchSink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_SS_N_SPLIT,
                    AttnKernels.ARG_SS_SPLIT_INDEX}, new Object[]{step.layoutSplits, step.count});
            return true;
        }
        return false;
    }

    @Override
    public boolean verify(int pos) {
        if (decodeIdx < 0) {
            return false;
        }
        Step step = stepOf(pos);
        if (step == null || !patternMatches(step)) {
            return false;
        }
        int rows = nSeq * nHead;
        boolean ok = intOf(decode.argValue(AttnKernels.ARG_FD_N_SPLIT)) == step.layoutSplits
                && intOf(decode.argValue(AttnKernels.ARG_FD_SPLIT_LEN)) == step.splitLen
                && intOf(decode.argValue(AttnKernels.ARG_FD_FLAGS)) == step.flags
                && intOf(decode.argValue(AttnKernels.ARG_FDP_N_SEQ)) == nSeq
                && Arrays.equals(decode.global(), decodeGlobal(step.count));
        if (combineIdx >= 0) {
            ok &= intOf(combine.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && intOf(combine.argValue(AttnKernels.ARG_FC_N_ROWS)) == rows
                    && Arrays.equals(combine.global(), FlashAttnTiling.combineGlobal(rows));
        }
        if (lseIdx >= 0) {
            ok &= intOf(lse.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && intOf(lse.argValue(AttnKernels.ARG_FC_N_ROWS)) == rows
                    && Arrays.equals(lse.global(), FlashAttnTiling.combineGlobal(rows));
        }
        if (sinkIdx >= 0) {
            ok &= intOf(sink.argValue(AttnKernels.ARG_SS_N_SPLIT)) == step.layoutSplits
                    && intOf(sink.argValue(AttnKernels.ARG_SS_SPLIT_INDEX)) == step.count
                    && intOf(sink.argValue(AttnKernels.ARG_SS_N_ROWS)) == rows;
        }
        return ok;
    }

    /** The decode grid of {@code count} splits over the captured sequence count. */
    private long[] decodeGlobal(int count) {
        return grouped ? FlashAttnTiling.decodeGqaGlobal(nSeq, nHeadKv, count)
                : decodePlan.decodePagedGlobal(nSeq, count);
    }

    /** True when the captured merges are exactly the ones the step's launch enqueues. */
    private boolean patternMatches(Step step) {
        return (combineIdx >= 0) == step.wantsCombine && (sinkIdx >= 0) == (hasSink && step.wantsCombine)
                && (lseIdx >= 0) == (hasLse && step.wantsCombine);
    }

    /** True when {@code d} is this layer's paged decode of the step's rows from offset 0, no tree mask. */
    private boolean isGroupDecode(GpuGraphDispatch d) {
        Object[] args = d.args();
        return d.kernelName().equals(decode.kernelName())
                && args.length > AttnKernels.ARG_FDP_BUF_K
                && args[AttnKernels.ARG_FDP_BUF_K] == k
                && intArg(args, AttnKernels.ARG_FD_Q_OFF) == 0
                && intArg(args, AttnKernels.ARG_FD_OUT_OFF) == 0
                && (intArg(args, AttnKernels.ARG_FD_FLAGS) & AttnKernels.FLAG_TREE) == 0;
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

    /**
     * The values the paged decode launch sets for a step of {@code nKeys} keys over the captured sequences, or null
     * when the key count is outside the plan's range.
     */
    private Step stepOf(int nKeys) {
        if (nSeq <= 0 || nKeys < 0 || nKeys > decodePlan.keyCapacity()) {
            return null;
        }
        FlashAttnTiling.Split s;
        try {
            s = grouped ? decodePlan.decodeGqaSplit(nKeys, nSeq) : decodePlan.decodeSplit(nKeys);
        } catch (IllegalStateException e) {
            return null;
        }
        boolean direct = GpuPagedFlashAttention.decodeWritesDirect(s, grouped && decodePlan.decodeGqaSplitsEarly(nSeq),
                hasSink, hasLse);
        return new Step(s.count(), s.length(), s.count() + (hasSink ? 1 : 0),
                baseFlags | (direct ? AttnKernels.FLAG_DIRECT : 0), !direct);
    }

    private static int intArg(Object[] args, int i) {
        return (Integer) args[i];
    }

    private static int intOf(Object value) {
        return (Integer) value;
    }

    /** One step's derived launch values. */
    private static final class Step {
        final int count;
        final int splitLen;
        final int layoutSplits;
        final int flags;
        final boolean wantsCombine;

        Step(int count, int splitLen, int layoutSplits, int flags, boolean wantsCombine) {
            this.count = count;
            this.splitLen = splitLen;
            this.layoutSplits = layoutSplits;
            this.flags = flags;
            this.wantsCombine = wantsCombine;
        }
    }
}
