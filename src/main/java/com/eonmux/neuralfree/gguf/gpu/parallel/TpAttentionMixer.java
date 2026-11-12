package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.block.GpuMixer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuAllReduce;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.memory.StepLayout;
import com.eonmux.neuralfree.gguf.ops.attn.GpuAttentionMixer;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The attention mixer of one tensor-parallel layer: every rank's {@link GpuAttentionMixer} over the rank's head shard
 * and the rank's own KV cache. Every peer rank first receives one replay of the main device's normalized input in its
 * own buffer ({@link RankExchange#replayAll}) and its heads are enqueued there; then rank 0 computes its heads from
 * the input into the branch output slot while the peers run, and a reduction ({@link GpuAllReduce#reduceToRoot}) adds
 * the partial branch outputs into rank 0's slot in rank order, so the layer output is the sum of the rank blocks
 * exactly as vLLM's row-parallel attention output projection sums them.
 *
 * <p>Each rank's mixer binds against a copy of the layer spec whose head counts are the rank's, so its projection
 * shapes, scratch rows, RoPE geometry and KV cache rows are shard-local. Every rank's cache is written by that rank's
 * own store kernel from the same positions, so the per-rank caches stay in lockstep with the main memory's sequence.
 * A paged multi-sequence step runs {@link #batch}: every rank's mixer steps over the layout's segments through its own
 * memory's current {@link com.eonmux.neuralfree.gguf.memory.paged.PagedStep}, whose page tables and cell rows the step
 * mirror installed on every rank cache, and the rank partials all-reduce exactly as the contiguous step's do.</p>
 */
final class TpAttentionMixer implements GpuMixer {

    private final TensorParallelGroup group;
    private final LayerSpec layer;
    private final TpAttentionPlan.Layer split;
    private final List<GpuAttentionMixer> parts = new ArrayList<>();
    private GpuForwardContext ctx;
    private boolean bound;

    TpAttentionMixer(TensorParallelGroup group, LayerSpec layer, TpAttentionPlan.Layer split) {
        this.group = Objects.requireNonNull(group, "group");
        this.layer = Objects.requireNonNull(layer, "layer");
        this.split = Objects.requireNonNull(split, "split");
    }

    @Override
    public void bind(GpuForwardContext ctx, LayerSpec layer) {
        if (bound) {
            throw new IllegalStateException("layer " + describe() + " is already bound");
        }
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        for (int rank = 0; rank < group.rankCount(); rank++) {
            GpuAttentionMixer part = new GpuAttentionMixer();
            part.bind(group.rankContext(rank), shardLayer(layer, rank));
            parts.add(part);
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
        group.uploadStepInputs(pos0, nTokens);
        replayToPeers(in, nTokens, true);
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerIn = group.inputSlot(rank);
            ScratchSlot peerOut = group.partialSlot(rank);
            if (nTokens == 1) {
                parts.get(rank).decode(pos0, peerIn, peerOut);
            } else {
                parts.get(rank).prefill(pos0, nTokens, peerIn, peerOut);
            }
        }
        if (nTokens == 1) {
            parts.get(0).decode(pos0, in, out);
        } else {
            parts.get(0).prefill(pos0, nTokens, in, out);
        }
        sumPartials(out, nTokens);
    }

    /**
     * One paged multi-sequence step: every rank's mixer runs the layout's segments through its own memory's current
     * {@code PagedStep} over its shard of the heads, and the partial branch outputs all-reduce into rank 0's slot.
     * A contiguous layout (a single sequence over un-paged memory) takes the contiguous {@link #forward} path, whose
     * rank step inputs the mirror uploaded.
     */
    @Override
    public void batch(StepLayout s, ScratchSlot in, ScratchSlot out) {
        requireBound();
        Objects.requireNonNull(s, "s");
        if (s.contiguousSingle()) {
            forward(s.pos0(0), s.nTokens(), in, out);
            return;
        }
        int nTokens = s.nTokens();
        replayToPeers(in, nTokens, true);
        for (int rank = 1; rank < parts.size(); rank++) {
            parts.get(rank).batch(s, group.inputSlot(rank), group.partialSlot(rank));
        }
        parts.get(0).batch(s, in, out);
        sumPartials(out, nTokens);
    }

    /** Sends {@code in}'s rows to every peer rank's input slot ({@link RankExchange#replayToPeers}). */
    private void replayToPeers(ScratchSlot in, int nTokens, boolean partials) {
        long floats = (long) nTokens * streamWidth();
        RankExchange.replayToPeers(group, parts.size(), ctx.queue(), in, floats, partials ? floats : 0L);
    }

    /** Adds every rank's partial branch output into rank 0's {@code out}; the peers' partials are left as they were. */
    private void sumPartials(ScratchSlot out, int nTokens) {
        int width = streamWidth();
        List<GpuAllReduce.Participant> participants = new ArrayList<>(parts.size());
        participants.add(new GpuAllReduce.Participant(ctx.queue(), out.buffer(), out.offset()));
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerOut = group.partialSlot(rank);
            participants.add(new GpuAllReduce.Participant(group.rankQueue(rank), peerOut.buffer(), peerOut.offset()));
        }
        GpuAllReduce.reduceToRoot(ctx.ops(), group.allReduceTemporary(), participants, nTokens * width);
    }

    /** Null when every rank's mixer can step over a paged layout, or the unbound reason. */
    @Override
    public String batchRefusal() {
        return bound ? null : "layer " + describe() + " is not bound";
    }

    @Override
    public void prefillKV(int pos0, int nTokens, ScratchSlot in) {
        requireBound();
        group.uploadStepInputs(pos0, nTokens);
        replayToPeers(in, nTokens, false);
        for (int rank = 1; rank < parts.size(); rank++) {
            parts.get(rank).prefillKV(pos0, nTokens, group.inputSlot(rank));
        }
        parts.get(0).prefillKV(pos0, nTokens, in);
    }

    @Override
    public String kvRefusal() {
        return null;
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

    /**
     * Null when the layer can shift: every rank's mixer can re-rotate its shard cache. The contiguous shift's paged
     * refusal is the group's ({@link TensorParallelBackend#shiftContext}), so a per-sequence shift over paged rank
     * caches is not blocked here. Otherwise the first rank's refusal, by name.
     */
    @Override
    public String shiftRefusal() {
        if (!bound) {
            return "layer " + describe() + " is not bound";
        }
        String refusal;
        for (int rank = 0; rank < parts.size(); rank++) {
            refusal = parts.get(rank).shiftRefusal();
            if (refusal != null) {
                return "rank" + rank + ": " + refusal;
            }
        }
        return null;
    }

    /** The layer's spec with this rank's heads. */
    private LayerSpec shardLayer(LayerSpec layer, int rank) {
        return LayerSpec.builder(layer).attention(split.shardSpec(layer.attention(), rank)).build();
    }

    private int streamWidth() {
        return layer.residual().streamWidth();
    }

    private void requireBound() {
        if (!bound) {
            throw new IllegalStateException("layer " + describe() + " is not bound");
        }
    }

    private String describe() {
        return layer.stack() + "." + layer.il();
    }

    @Override
    public void close() {
        List<GpuAttentionMixer> closed = new ArrayList<>(parts);
        parts.clear();
        for (GpuAttentionMixer part : closed) {
            part.close();
        }
        ctx = null;
        bound = false;
    }
}
