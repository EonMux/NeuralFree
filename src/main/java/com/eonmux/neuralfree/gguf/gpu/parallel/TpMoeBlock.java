package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.block.GpuFfnBlock;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuAllReduce;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardContext;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlot;
import com.eonmux.neuralfree.gguf.ops.moe.GpuMoeFfn;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The expert block of one split layer: every rank's {@link GpuMoeFfn} over the rank's intermediate-width shard of
 * every expert and of the shared expert, or, under expert parallel, over the rank's own whole experts with the
 * routed pairs of the other ranks' experts zeroed. Every peer first receives one replay of the normalized FFN input
 * and its partial is enqueued there; rank 0 then computes its partial into the output slot while the peers run, and
 * a reduction into rank 0 ({@link GpuAllReduce#reduceToRoot}) sums them. The router
 * weights and the routing stay whole on every rank, so each rank selects the same experts, and the sum of the
 * partials is the whole block's output.
 */
final class TpMoeBlock implements GpuFfnBlock {

    private final TensorParallelGroup group;
    private final LayerSpec layer;
    private final TpMoePlan.Layer split;
    private final List<GpuMoeFfn> parts = new ArrayList<>();
    private GpuForwardContext ctx;
    private boolean bound;

    TpMoeBlock(TensorParallelGroup group, LayerSpec layer, TpMoePlan.Layer split) {
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
            LayerSpec shardLayer = LayerSpec.builder(layer)
                    .moe(split.shardSpec(layer.moe(), rank)).build();
            GpuMoeFfn part = new GpuMoeFfn(layer.ffnKind(), shardLayer);
            part.bind(group.rankContext(rank), shardLayer);
            parts.add(part);
        }
        bound = true;
    }

    @Override
    public void forward(int nTokens, ScratchSlot in, ScratchSlot out) {
        if (!bound) {
            throw new IllegalStateException("layer " + describe() + " is not bound");
        }
        int width = layer.residual().streamWidth();
        long floats = (long) nTokens * width;
        RankExchange.replayToPeers(group, parts.size(), ctx.queue(), in, floats, floats);
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
        GpuAllReduce.reduceToRoot(ctx.ops(), group.allReduceTemporary(), participants, nTokens * width);
    }

    private String describe() {
        return layer.stack() + "." + layer.il();
    }

    @Override
    public void close() {
        List<GpuMoeFfn> closed = new ArrayList<>(parts);
        parts.clear();
        for (GpuMoeFfn part : closed) {
            part.close();
        }
        ctx = null;
        bound = false;
    }
}
