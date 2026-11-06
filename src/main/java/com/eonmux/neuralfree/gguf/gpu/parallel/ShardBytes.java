package com.eonmux.neuralfree.gguf.gpu.parallel;

import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Device bytes one rank's shards of a tensor-parallel split take, computed from the uploader's tensor plans without
 * allocating anything. The group uses it for the per-rank VRAM fit check that runs before any shard is uploaded, and
 * the uploader uses {@link #splitShards} to account the main device's exact share of every split tensor instead of an
 * average over the ranks.
 */
public final class ShardBytes {

    private ShardBytes() {
    }

    /** Device bytes of {@code rank}'s dense feed-forward shards. */
    public static long ffn(FfnRowSplitPlan plan, int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        for (FfnRowSplitPlan.Layer layer : plan.layers()) {
            bytes += ffn(layer, rank, uploader);
        }
        return bytes;
    }

    /** Device bytes of {@code rank}'s routed and shared expert shards. */
    public static long moe(TpMoePlan plan, int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        for (TpMoePlan.Layer layer : plan.layers()) {
            for (Map.Entry<String, TpMoePlan.Shard> entry : layer.tensors().entrySet()) {
                bytes += moe(layer, entry.getKey(), entry.getValue(), rank, uploader);
            }
        }
        return bytes;
    }

    /** Device bytes of the whole MoE tensors every rank holds beside its shards (router, scales, latent). */
    public static long moeSupport(TpMoePlan plan, int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        for (TpMoePlan.Layer layer : plan.layers()) {
            for (String name : layer.support()) {
                GpuWeightUploader.TensorPlan p = uploader.plan(name);
                if (p == null) {
                    continue;
                }
                if (layer.expertParallel() && isExpertScale(layer, name)) {
                    // the per-expert scales are indexed by physical slot, so a redundant bank holds more values
                    bytes += (long) layer.physicalExperts() * Float.BYTES;
                } else {
                    bytes += p.deviceBytes();
                }
            }
        }
        return bytes;
    }

    /** True when {@code name} is one of a layer's per-expert scale vectors. */
    private static boolean isExpertScale(TpMoePlan.Layer layer, String name) {
        return name.equals(layer.tensor(TensorNames.scale(TensorNames.FFN_GATE_EXPS)))
                || name.equals(layer.tensor(TensorNames.scale(TensorNames.FFN_UP_EXPS)))
                || name.equals(layer.tensor(TensorNames.scale(TensorNames.FFN_DOWN_EXPS)));
    }

    /** Device bytes of one MoE tensor on {@code rank}. */
    public static long moe(TpMoePlan.Layer layer, String suffix, TpMoePlan.Shard kind, int rank,
                           GpuWeightUploader uploader) {
        GpuWeightUploader.TensorPlan p = uploader.plan(layer.tensor(suffix));
        long rows = p.ne().length < 2 ? p.info().numElements() : p.ne()[1];
        long rowBytes = p.ne().length < 2 ? Float.BYTES : p.deviceRowBytes();
        switch (kind) {
            case EXPERT_ROWS:
                return p.ne()[2] * layer.shard() * rowBytes;
            case EXPERT_FUSED:
                return p.ne()[2] * 2L * layer.shard() * rowBytes;
            case EXPERT_DOWN:
                return p.ne()[2] * p.ne()[1] * p.deviceType().getBytesForElements(layer.shard());
            case EXPERT_SLICE:
                return p.ne()[1] * layer.expertCount() * p.deviceRowBytes();
            case SHARED_ROWS:
                return layer.sharedShard() * rowBytes;
            case SHARED_FUSED:
                return 2L * layer.sharedShard() * rowBytes;
            case SHARED_DOWN:
                return p.deviceType().getBytesForElements(layer.sharedShard()) * p.ne()[1];
            case SHARED_HIDDEN_VECTOR:
                return layer.sharedShard() * Float.BYTES;
            case SHARED_OUTPUT_VECTOR:
                return rank == 0 ? p.deviceBytes() : 0L;
            default:
                throw new IllegalStateException("unknown expert shard " + kind + " of " + p.name());
        }
    }

    /** Device bytes of {@code rank}'s attention shards. */
    public static long attention(TpAttentionPlan plan, int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        for (TpAttentionPlan.Layer layer : plan.layers()) {
            bytes += attention(layer, plan.ranks(), rank, uploader);
        }
        return bytes;
    }

    /**
     * Device bytes of every split tensor on {@code rank}, keyed by full tensor name. The uploader skips these tensors
     * when it uploads the whole model, so its placement accounting takes each tensor's entry here instead of the whole
     * tensor's bytes; an entry may be zero (a scalar read on the host, or a tensor only rank 0 holds when this is not
     * rank 0).
     */
    public static Map<String, Long> splitShards(FfnRowSplitPlan ffn, TpAttentionPlan attention, int rank,
                                                GpuWeightUploader uploader) {
        return splitShards(ffn, attention, null, rank, uploader);
    }

    /**
     * Device bytes of every split tensor on {@code rank}, keyed by full tensor name, including the vocabulary shards
     * of a {@link TpVocabPlan}. The uploader skips these tensors when it uploads the whole model, so its placement
     * accounting takes each tensor's entry here instead of the whole tensor's bytes; an entry may be zero (a scalar
     * read on the host, or a tensor only rank 0 holds when this is not rank 0).
     */
    public static Map<String, Long> splitShards(FfnRowSplitPlan ffn, TpAttentionPlan attention, TpVocabPlan vocab,
                                                int rank, GpuWeightUploader uploader) {
        return splitShards(ffn, attention, vocab, null, rank, uploader);
    }

    /**
     * Device bytes of every split tensor on {@code rank}, keyed by full tensor name, including the vocabulary shards
     * of a {@link TpVocabPlan} and the expert shards of a {@link TpMoePlan}. The uploader skips these tensors when it
     * uploads the whole model, so its placement accounting takes each tensor's entry here instead of the whole
     * tensor's bytes; an entry may be zero (a scalar read on the host, or a tensor only rank 0 holds when this is not
     * rank 0).
     */
    public static Map<String, Long> splitShards(FfnRowSplitPlan ffn, TpAttentionPlan attention, TpVocabPlan vocab,
                                                TpMoePlan moe, int rank, GpuWeightUploader uploader) {
        Map<String, Long> out = new LinkedHashMap<>();
        if (ffn != null) {
            for (FfnRowSplitPlan.Layer layer : ffn.layers()) {
                put(out, layer.tensor(layer.up()), ffnTensor(layer, layer.up(), rank, uploader));
                if (layer.gate() != null) {
                    put(out, layer.tensor(layer.gate()), ffnTensor(layer, layer.gate(), rank, uploader));
                }
                put(out, layer.tensor(layer.down()), ffnTensor(layer, layer.down(), rank, uploader));
                put(out, tensor(layer, layer.upBias()), vector(layer.upBias(), layer.shard(rank), uploader));
                put(out, tensor(layer, layer.gateBias()), vector(layer.gateBias(), layer.shard(rank), uploader));
                put(out, tensor(layer, layer.actScales()), vector(layer.actScales(), layer.shard(rank), uploader));
                put(out, tensor(layer, layer.downBias()),
                        rank == 0 && layer.downBias() != null
                                ? uploader.plan(layer.tensor(layer.downBias())).deviceBytes() : 0L);
                put(out, tensor(layer, layer.downScale()), 0L);
            }
        }
        if (attention != null) {
            for (TpAttentionPlan.Layer layer : attention.layers()) {
                for (Map.Entry<String, TpAttentionPlan.Shard> entry : layer.tensors().entrySet()) {
                    put(out, layer.tensor(entry.getKey()),
                            attentionTensor(layer, entry.getKey(), entry.getValue(), attention.ranks(), rank,
                                    uploader));
                }
            }
        }
        if (vocab != null) {
            put(out, vocab.tokenTensor(), vocab.tokenBytes(rank));
            put(out, vocab.headTensor(), vocab.headBytes(rank));
            put(out, vocab.headBias(), vocab.headBiasBytes(rank));
        }
        if (moe != null) {
            for (TpMoePlan.Layer layer : moe.layers()) {
                for (Map.Entry<String, TpMoePlan.Shard> entry : layer.tensors().entrySet()) {
                    put(out, layer.tensor(entry.getKey()), moe(layer, entry.getKey(), entry.getValue(), rank,
                            uploader));
                }
            }
        }
        return out;
    }

    /** Device bytes of one dense feed-forward layer's shards on {@code rank}. */
    private static long ffn(FfnRowSplitPlan.Layer layer, int rank, GpuWeightUploader uploader) {
        long bytes = 0;
        for (String suffix : new String[]{layer.up(), layer.gate(), layer.down()}) {
            if (suffix != null) {
                bytes += ffnTensor(layer, suffix, rank, uploader);
            }
        }
        for (String suffix : new String[]{layer.upBias(), layer.gateBias(), layer.actScales()}) {
            bytes += vector(suffix, layer.shard(rank), uploader);
        }
        if (rank == 0 && layer.downBias() != null) {
            bytes += uploader.plan(layer.tensor(layer.downBias())).deviceBytes();
        }
        return bytes;
    }

    /** Device bytes of one feed-forward projection or bias the split shards. */
    private static long ffnTensor(FfnRowSplitPlan.Layer layer, String suffix, int rank,
                                  GpuWeightUploader uploader) {
        GpuWeightUploader.TensorPlan p = uploader.plan(layer.tensor(suffix));
        long shard = layer.shard(rank);
        if (suffix.equals(layer.up())) {
            return shard * p.deviceRowBytes() * (layer.fusedUp() ? 2 : 1);
        }
        if (suffix.equals(layer.gate())) {
            return shard * p.deviceRowBytes();
        }
        if (suffix.equals(layer.down())) {
            return p.deviceType().getBytesForElements(shard) * layer.nOut();
        }
        return vector(suffix, shard, uploader);
    }

    /** Device bytes of {@code count} floats of a vector tensor, or 0 without the tensor. */
    private static long vector(String suffix, long count, GpuWeightUploader uploader) {
        return suffix == null ? 0 : count * Float.BYTES;
    }

    /** Device bytes of one attention layer's shards on {@code rank}. */
    private static long attention(TpAttentionPlan.Layer layer, int ranks, int rank,
                                  GpuWeightUploader uploader) {
        long bytes = 0;
        for (Map.Entry<String, TpAttentionPlan.Shard> entry : layer.tensors().entrySet()) {
            bytes += attentionTensor(layer, entry.getKey(), entry.getValue(), ranks, rank, uploader);
        }
        return bytes;
    }

    /** Device bytes of one attention tensor on {@code rank}. */
    private static long attentionTensor(TpAttentionPlan.Layer layer, String suffix, TpAttentionPlan.Shard kind,
                                        int ranks, int rank, GpuWeightUploader uploader) {
        GpuWeightUploader.TensorPlan p = uploader.plan(layer.tensor(suffix));
        long rows = p.ne().length < 2 ? p.info().numElements() : p.ne()[1];
        long rowBytes = p.ne().length < 2 ? Float.BYTES : p.deviceRowBytes();
        if (kind == TpAttentionPlan.Shard.ROWS) {
            return rows / ranks * rowBytes;
        }
        if (kind == TpAttentionPlan.Shard.FUSED_ROWS) {
            return rows * rowBytes / ranks;
        }
        if (kind == TpAttentionPlan.Shard.NORM_HEADS || kind == TpAttentionPlan.Shard.PER_HEAD) {
            return p.info().numElements() * Float.BYTES / ranks;
        }
        if (kind == TpAttentionPlan.Shard.OUT_COLUMNS) {
            return p.deviceType().getBytesForElements(layer.outColumns()) * p.ne()[1];
        }
        if (kind == TpAttentionPlan.Shard.OUT_BIAS) {
            return rank == 0 ? p.deviceBytes() : 0L;
        }
        return rank == 0 || kind == TpAttentionPlan.Shard.WHOLE ? p.deviceBytes() : 0L;
    }

    private static void put(Map<String, Long> out, String name, long bytes) {
        if (name != null) {
            out.put(name, bytes);
        }
    }

    /** Full name of a feed-forward companion suffix, or null without the tensor. */
    private static String tensor(FfnRowSplitPlan.Layer layer, String suffix) {
        return suffix == null ? null : layer.tensor(suffix);
    }
}
