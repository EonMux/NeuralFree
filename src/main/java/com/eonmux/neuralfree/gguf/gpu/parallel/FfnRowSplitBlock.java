package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuAllReduce;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.ops.act.GpuDenseFfn;
import com.eonmux.neuralfree.gguf.spec.FfnSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The feed-forward block of one split layer: every rank's {@link GpuDenseFfn} over the rank's shard weights, driven
 * from the model's step. Every peer rank first receives one replay of the normalized FFN input in its own buffer and
 * its partial is enqueued there; rank 0 then computes its partial straight into the output slot of the model device
 * while the peers run, and a reduction ({@link GpuAllReduce#reduceToRoot}) adds the partials into the output slot,
 * which then holds the whole
 * {@code down(act(gate x, up x))} of the layer as the whole-device block would have written it.
 *
 * <p>Lifecycle and per-step rules as {@link GpuFfnBlock}. A block binds once, over the per-rank contexts of its
 * {@link TensorParallelGroup}; each rank's FFN binds against a shard-local copy of the layer spec whose {@code n_ff}
 * is the shard width, so the projection shapes and scratch rows the dense FFN derives are shard-local too.</p>
 */
final class FfnRowSplitBlock implements GpuFfnBlock {

    private final TensorParallelGroup group;
    private final LayerSpec layer;
    private final FfnRowSplitPlan.Layer split;
    private final List<GpuDenseFfn> parts = new ArrayList<>();
    private GpuForwardContext ctx;
    private boolean bound;

    FfnRowSplitBlock(TensorParallelGroup group, LayerSpec layer, FfnRowSplitPlan.Layer split) {
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
            GpuDenseFfn part = new GpuDenseFfn();
            part.bind(group.rankContext(rank), LayerSpec.builder(layer).ffn(shardFfn(layer, split.shard(rank))).build());
            parts.add(part);
        }
        bound = true;
    }

    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (!bound) {
            throw new IllegalStateException("layer " + describe() + " is not bound");
        }
        RankExchange.replayToPeers(group, parts.size(), ctx.queue(), in, (long) nTokens * split.nIn(),
                (long) nTokens * split.nOut());
        for (int rank = 1; rank < parts.size(); rank++) {
            parts.get(rank).forward(nTokens, group.inputSlot(rank), group.partialSlot(rank));
        }
        parts.get(0).forward(nTokens, in, out);
        List<GpuAllReduce.Participant> participants = new ArrayList<>(parts.size());
        participants.add(new GpuAllReduce.Participant(ctx.queue(), out.buffer(), out.offset()));
        for (int rank = 1; rank < parts.size(); rank++) {
            ScratchSlot peerOut = group.partialSlot(rank);
            participants.add(new GpuAllReduce.Participant(group.rankQueue(rank), peerOut.buffer(), peerOut.offset()));
        }
        GpuAllReduce.reduceToRoot(ctx.ops(), group.allReduceTemporary(), participants, nTokens * split.nOut());
    }


    /** The layer's spec with a shard-local {@code n_ff}, so the dense FFN binds over the shard shapes. */
    static com.eonmux.neuralfree.gguf.spec.FfnSpec shardFfn(LayerSpec layer, int shard) {
        com.eonmux.neuralfree.gguf.spec.FfnSpec ffn = layer.ffn();
        com.eonmux.neuralfree.gguf.spec.FfnSpec.Builder shardFfn =
                com.eonmux.neuralfree.gguf.spec.FfnSpec.builder(ffn.role()).nFf(shard);
        if (ffn.usesGlu()) {
            shardFfn.glu(ffn.gating(), ffn.glu());
        } else {
            shardFfn.act(ffn.gated(), ffn.act());
        }
        shardFfn.params(ffn.params()).biases(ffn.biases());
        return shardFfn.build();
    }

    private String describe() {
        return layer.stack() + "." + layer.il();
    }

    @Override
    public void close() {
        List<GpuDenseFfn> closed = new ArrayList<>(parts);
        parts.clear();
        for (GpuDenseFfn part : closed) {
            part.close();
        }
        ctx = null;
        bound = false;
    }
}
