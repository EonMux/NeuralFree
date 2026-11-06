package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;

/**
 * Builds the rank-local {@link ModelSpec} of a tensor-parallel split: every layer whose attention or dense FFN the
 * split covers carries the rank's head and neuron counts, so the rank's blocks derive shard-local projection shapes,
 * scratch rows and KV-cache geometry. A layer outside both plans keeps its whole-device spec. The embedding, the
 * output head, the schedule and the hparams are the model's own: the embedding and the output head are not
 * vocab-parallel and stay whole on the model device.
 */
final class TpShardSpec {

    private TpShardSpec() {
    }

    /**
     * The spec of {@code rank}.
     *
     * @param ffn  the feed-forward split, or null when that half is not split
     * @param attn the attention split, or null in a feed-forward-only (row) split
     */
    static ModelSpec of(ModelSpec spec, FfnRowSplitPlan ffn, TpAttentionPlan attn, int rank) {
        ModelSpec.Builder b = ModelSpec.builder(spec.hparams(), spec.tensors())
                .schedule(spec.schedule())
                .embedding(spec.embedding())
                .head(spec.head())
                .memoryKind(spec.memoryKind());
        for (LayerSpec l : spec.layers()) {
            b.layer(shard(l, ffn, attn, rank));
        }
        for (LayerSpec l : spec.nextnLayers()) {
            b.nextnLayer(shard(l, ffn, attn, rank));
        }
        for (LayerSpec l : spec.encoderLayers()) {
            b.encoderLayer(shard(l, ffn, attn, rank));
        }
        return b.build();
    }

    private static LayerSpec shard(LayerSpec layer, FfnRowSplitPlan ffn, TpAttentionPlan attn, int rank) {
        LayerSpec.Builder b = LayerSpec.builder(layer);
        if (attn != null) {
            TpAttentionPlan.Layer split = attn.layerOf(layer);
            if (split != null) {
                b.attention(split.shardSpec(layer.attention(), rank));
            }
        }
        if (ffn != null) {
            FfnRowSplitPlan.Layer split = ffn.layerOf(layer);
            if (split != null) {
                b.ffn(FfnRowSplitBlock.shardFfn(layer, split.shard(rank)));
            }
        }
        return b.build();
    }
}
