package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphDispatch;
import com.eonmux.neuralfree.acceleration.backend.GpuGraphPatchSink;
import com.eonmux.neuralfree.acceleration.backend.GpuStepPatcher;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;

import java.util.Arrays;

/**
 * The decode-step patcher of one {@link GpuFlashAttention}: from the step position alone it derives the
 * values {@link GpuFlashAttention#decodeRow} re-sets on its kernels every step — the key count
 * ({@code kv_end}) of the cache view, the split count and length and grid the tiling picks from it, the
 * query position and the direct flag, the static sink partial's split index and fixed position, and the
 * split count of the merge and log-sum-exp launches — and hands them to a captured graph's patch sink,
 * so the decode trunk replays without re-running the dispatch body.
 *
 * <p>The captured dispatches of one group are claimed by identity and order: the decode kernel is this
 * layer's when it reads this layer's Q and K rows of one query row at offset 0 (never a tree step's or
 * another row's), and the merge launches are the ones that follow it immediately, in the order
 * {@code decodeRow} enqueues them — the static sink partial when the layer has one, then the combine,
 * then the log-sum-exp merge. A capture that breaks the order (a foreign kernel inside the group)
 * claims nothing beyond the decode node, and a step whose merge pattern would leave the capture is
 * refused, so the caller runs the body. {@link #verify(int)} checks the same values against the
 * kernels' current bindings after a body replay, which is how the plan earns the right to skip the
 * body. Not thread-safe.
 */
final class FlashDecodePatcher implements GpuStepPatcher {

    private final FlashAttnTiling decodePlan;
    private final boolean cached;
    private final int keyRowBase;
    private final int keyRows;
    private final int nHead;
    private final int flags;
    private final boolean hasSink;
    private final boolean hasLse;
    private final GpuBuffer k;
    private final GpuBuffer q;
    private final PreparedDispatch decode;
    private final PreparedDispatch combine;
    private final PreparedDispatch lse;
    private final PreparedDispatch sink;

    /** The captured dispatch positions this patcher claimed, {@code -1} for a role the capture lacks. */
    private int decodeIdx = -1;
    private int combineIdx = -1;
    private int lseIdx = -1;
    private int sinkIdx = -1;
    /** Merge roles claimed so far, in {@code decodeRow}'s order. */
    private int groupClaimed;
    /** True once a capture placed a kernel inside this group that is not one of its merges. */
    private boolean groupBroken;
    /** The split count of the step the plan was captured at, {@code -1} before the first patch. */
    private int capturedCount = -1;

    FlashDecodePatcher(FlashAttnTiling decodePlan, boolean cached, int keyRowBase, int keyRows, int nHead,
                       int flags, boolean hasSink, boolean hasLse, GpuBuffer k, GpuBuffer q,
                       PreparedDispatch decode, PreparedDispatch combine, PreparedDispatch lse,
                       PreparedDispatch sink) {
        this.decodePlan = decodePlan;
        this.cached = cached;
        this.keyRowBase = keyRowBase;
        this.keyRows = keyRows;
        this.nHead = nHead;
        this.flags = flags;
        this.hasSink = hasSink;
        this.hasLse = hasLse;
        this.k = k;
        this.q = q;
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
        capturedCount = -1;
    }

    @Override
    public boolean accepts(GpuGraphDispatch d) {
        if (decodeIdx < 0) {
            if (!isGroupDecode(d)) {
                return false;
            }
            decodeIdx = d.index();
            groupClaimed = 0;
            groupBroken = false;
            return true;
        }
        if (groupBroken || d.index() != decodeIdx + 1 + groupClaimed) {
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
            /* A step whose split layout differs from the capture's would patch a split count the
             * captured launches never ran; the merge pattern and every derived length belong to
             * that layout, so the group refuses and the caller recaptures at the new one. */
            return false;
        }
        if (recordIndex == decodeIdx) {
            sink.patchKernel(recordIndex, decodePlan.decodeGlobal(step.count), new int[]{
                    AttnKernels.ARG_FD_N_SPLIT, AttnKernels.ARG_FD_SPLIT_LEN, AttnKernels.ARG_FD_KV_END,
                    AttnKernels.ARG_FD_Q_POS, AttnKernels.ARG_FD_FLAGS},
                    new Object[]{step.layoutSplits, step.splitLen, step.nKeys, pos, step.flags});
            return true;
        }
        if (recordIndex == combineIdx || recordIndex == lseIdx) {
            sink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_FC_N_SPLIT},
                    new Object[]{step.layoutSplits});
            return true;
        }
        if (recordIndex == sinkIdx) {
            sink.patchKernel(recordIndex, null, new int[]{AttnKernels.ARG_SS_FIXED_POS,
                    AttnKernels.ARG_SS_N_SPLIT, AttnKernels.ARG_SS_SPLIT_INDEX},
                    new Object[]{pos, step.layoutSplits, step.count});
            return true;
        }
        return false;
    }

    /** True when this patcher claimed the decode dispatch of the current capture. */
    boolean claimed() {
        return decodeIdx >= 0;
    }

    @Override
    public boolean verify(int pos) {
        if (decodeIdx < 0) {
            return false;
        }
        Step step = stepOf(pos);
        if (!patternMatches(step)) {
            return false;
        }
        boolean ok = intOf(decode.argValue(AttnKernels.ARG_FD_N_SPLIT)) == step.layoutSplits
                && intOf(decode.argValue(AttnKernels.ARG_FD_SPLIT_LEN)) == step.splitLen
                && intOf(decode.argValue(AttnKernels.ARG_FD_KV_END)) == step.nKeys
                && intOf(decode.argValue(AttnKernels.ARG_FD_Q_POS)) == pos
                && intOf(decode.argValue(AttnKernels.ARG_FD_FLAGS)) == step.flags
                && Arrays.equals(decode.global(), decodePlan.decodeGlobal(step.count));
        if (combineIdx >= 0) {
            ok &= intOf(combine.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && Arrays.equals(combine.global(), FlashAttnTiling.combineGlobal(nHead));
        }
        if (lseIdx >= 0) {
            ok &= intOf(lse.argValue(AttnKernels.ARG_FC_N_SPLIT)) == step.layoutSplits
                    && Arrays.equals(lse.global(), FlashAttnTiling.combineGlobal(nHead));
        }
        if (sinkIdx >= 0) {
            ok &= intOf(sink.argValue(AttnKernels.ARG_SS_FIXED_POS)) == pos
                    && intOf(sink.argValue(AttnKernels.ARG_SS_N_SPLIT)) == step.layoutSplits
                    && intOf(sink.argValue(AttnKernels.ARG_SS_SPLIT_INDEX)) == step.count
                    && Arrays.equals(sink.global(), FlashAttnTiling.combineGlobal(nHead));
        }
        return ok;
    }

    /** True when the captured merges are exactly the ones a direct step of this layer skips. */
    private boolean patternMatches(Step step) {
        return (combineIdx >= 0) == step.wantsCombine && (sinkIdx >= 0) == hasSink && (lseIdx >= 0) == hasLse;
    }

    /** True when {@code d} is this layer's decode kernel of one query row at offset 0, no tree mask. */
    private boolean isGroupDecode(GpuGraphDispatch d) {
        Object[] args = d.args();
        return d.kernelName().startsWith(AttnKernels.FLASH_DECODE_PREFIX)
                && args.length > AttnKernels.ARG_FD_BUF_K
                && args[AttnKernels.ARG_FD_BUF_K] == k
                && args[AttnKernels.ARG_FD_BUF_Q] == q
                && intArg(args, AttnKernels.ARG_FD_Q_OFF) == 0
                && intArg(args, AttnKernels.ARG_FD_OUT_OFF) == 0
                && intArg(args, AttnKernels.ARG_FD_KV_START) == 0
                && AttnKernels.flagFresh(intArg(args, AttnKernels.ARG_FD_FLAGS)) == 0
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

    /** The values {@link GpuFlashAttention#decodeRow} sets for one query row at {@code pos}. */
    private Step stepOf(int pos) {
        int nKeys = cached ? cachedKeys(pos + 1) : 1;
        FlashAttnTiling.Split s = decodePlan.decodeSplit(nKeys);
        boolean direct = s.direct() && !hasSink && !hasLse;
        return new Step(nKeys, s.count(), s.length(), s.count() + (hasSink ? 1 : 0),
                flags | (direct ? AttnKernels.FLAG_DIRECT : 0), direct, !direct);
    }

    /** Cache rows a query at {@code upTo - 1} can see: the rows of positions {@code [rowBase, upTo)}. */
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

    /** One step's derived decode values. */
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
