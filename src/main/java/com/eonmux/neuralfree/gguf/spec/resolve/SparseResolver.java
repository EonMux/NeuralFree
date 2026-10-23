package com.eonmux.neuralfree.gguf.spec.resolve;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.spec.DsaSpec;
import com.eonmux.neuralfree.gguf.spec.Dsv4Spec;
import com.eonmux.neuralfree.gguf.spec.MsaSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.util.Objects;

/**
 * Resolves the sparse-attention facets of a layer: the DeepSeek lightning indexer ({@link DsaSpec}, including the QSA
 * pooled indexer of glm5-next / qwen4exp), MiniMax-M3 block-sparse attention ({@link MsaSpec}) and DeepSeek-V4
 * compressed attention ({@link Dsv4Spec}). Each accessor returns null for a layer without that facet.
 *
 * <p>Indexer rope: deepseek32, dots3note rotate the leading {@code n_rot} dims with NEOX; glm-dsa uses NORM
 * (glm-dsa.cpp:261-263); hy-v4 and deepseek4 rotate the trailing {@code n_rot} dims with the model rope type
 * (hy-v4.cpp:361-374); qwen4exp ropes the leading dims with the model's sectioned rope (qwen4exp.cpp:797-800);
 * glm5-next indexes without rope. The Hadamard rotation applies where the indexer key cache sets
 * {@code n_embd_head_k_full = indexer_head_size} (llama-kv-cache-dsa.cpp:45, llama-kv-cache.cpp:325-335). Immutable.
 */
public final class SparseResolver {

    private final ModelHparams hp;
    private final TensorCatalog tc;

    public SparseResolver(ModelHparams hp, TensorCatalog tc) {
        this.hp = Objects.requireNonNull(hp, "hp");
        this.tc = Objects.requireNonNull(tc, "tc");
    }

    /**
     * Lightning indexer of layer {@code il}, or null when the layer selects no top-k cells. deepseek4 indexers are
     * part of {@link #dsv4(int)}. The deepseek32 and glm-dsa NextN blocks ({@code il >= n_layer}) run dense MLA over a
     * plain KV cache (llama-model.cpp:2830-2850; deepseek32.cpp:486, 552; glm-dsa.cpp:526, 592).
     */
    public DsaSpec dsa(int il) {
        checkLayer(il);
        switch (hp.arch()) {
            case DEEPSEEK32:
                if (il >= hp.nLayer()) {
                    return null;
                }
                return indexer(true, il, RopeType.NEOX, ropeDims(), false, true).build();
            case GLM_DSA: {
                if (il >= hp.nLayer()) {
                    return null;
                }
                boolean full = hp.isIndexerFull(il);
                return indexer(full, sourceLayer(il, false), RopeType.NORM, ropeDims(), false, true).build();
            }
            case DOTS3NOTE:
                if (il >= hp.nLayer() || hp.isSwa(il)) {
                    return null;                                       // dots3note.cpp:75, 208
                }
                return indexer(true, il, RopeType.NEOX, ropeDims(), false, true).build();
            case HY_V4: {
                if (hp.indexerTopK() <= 0) {
                    return null;                                       // hy-v4.cpp:45-50
                }
                boolean full = hp.isIndexerFull(il);
                return indexer(full, sourceLayer(il, false), hp.ropeType(), ropeDims(), true, false).build();
            }
            case GLM5_NEXT: {
                if (hp.isRecurrent(il)) {
                    return null;
                }
                boolean full = il >= hp.nLayer() || hp.isIndexerFull(il);    // glm5-next.cpp:143
                return indexer(full, sourceLayer(il, true), RopeType.NONE, 0, false, false)
                        .kpool(hp.indexerKpool(), hp.indexerKpoolRow(), hp.indexerKpoolByOrder(),
                                hp.indexerKpoolSelectTail())
                        .build();
            }
            case QWEN4EXP: {
                if (hp.compressRatio(il) <= 0 || hp.indexerKpool() <= 0) {
                    return null;                                       // qwen4exp.cpp:64-75, 434
                }
                return indexer(true, il, hp.ropeType(), ropeDims(), false, false)
                        .kpool(hp.indexerKpool(), hp.indexerKpoolRow(), hp.indexerKpoolByOrder(),
                                hp.indexerKpoolSelectTail())
                        .weightScale(1.0f / (float) Math.sqrt((float) hp.indexerHeadSize()))   // qwen4exp.cpp:807
                        .build();
            }
            default:
                return null;
        }
    }

    /** MiniMax-M3 block-sparse attention of layer {@code il}, or null for other architectures. */
    public MsaSpec msa(int il) {
        checkLayer(il);
        if (hp.arch() != Arch.MINIMAX_M3) {
            return null;
        }
        return MsaSpec.builder()
                .idxHeads(hp.indexerNHead())
                .idxHeadDim(hp.indexerHeadSize())
                .idxRopeDim(ropeDims())                                  // minimax-m3.cpp:350-352
                .blockSize(hp.indexerBlockSize())
                .localBlocks(hp.indexerLocalBlocks())
                .topKBlocks(hp.indexerTopK())                          // minimax-m3.cpp:29
                .sparse(il >= hp.nLayerDenseLead())
                .kqScale(1.0f / (float) Math.sqrt((float) hp.headK(0)))    // minimax-m3.cpp:211, 386
                .build();
    }

    /**
     * DeepSeek-V4 attention of layer {@code il}, or null for other architectures: compress ratio 0 (sliding window
     * only), 4 (CSA with a lightning indexer) or 128 (HCA), grouped low-rank output and optional sinks.
     */
    public Dsv4Spec dsv4(int il) {
        checkLayer(il);
        if (hp.arch() != Arch.DEEPSEEK4) {
            return null;
        }
        int ratio = hp.compressRatio(il);
        int headDim = hp.headK(0);                             // deepseek4.cpp:897: n_embd_head_k()
        Dsv4Spec.Builder b = Dsv4Spec.builder()
                .nHead(hp.nHead(il))
                .compressRatio(ratio)
                .window(hp.nSwa())
                .headDim(headDim)
                .ropeDim(ropeDims())
                .qLoraRank(hp.nLoraQ())
                .oGroupCount(hp.dsv4OGroupCount())
                .oLoraRank(hp.dsv4OLoraRank())
                .compressRopeBase(hp.dsv4CompressRopeBase())
                .sinks(tc.hasLayer(il, TensorNames.weight(TensorNames.ATTN_SINKS)))
                .kqScale(1.0f / (float) Math.sqrt((float) headDim));
        if (ratio == Dsv4Spec.RATIO_CSA) {
            b.indexer(indexer(true, il, hp.ropeType(), ropeDims(), true, true).build());   // deepseek4.cpp:590-624
        }
        return b.build();
    }

    /** Indexer builder with the model's head count, head size and top-k and the default weight scale. */
    private DsaSpec.Builder indexer(boolean full, int source, RopeType rope, int ropeDim, boolean ropeTail,
                                    boolean hadamard) {
        return DsaSpec.builder()
                .indexerHeads(hp.indexerNHead())
                .indexerHeadDim(hp.indexerHeadSize())
                .indexerRopeDim(ropeDim)
                .ropeType(rope)
                .ropeTail(ropeTail)
                .topK(hp.indexerTopK())
                .full(full)
                .sourceLayer(source)
                .hadamard(hadamard)
                .defaultWeightScale();
    }

    /**
     * Nearest full-indexer layer at or below {@code il} whose top-k a shared layer reuses (glm-dsa.cpp:254,
     * hy-v4.cpp:439-440).
     *
     * @param nextnFull NextN blocks always score (glm5-next.cpp:143)
     * @throws IllegalStateException when no full layer precedes {@code il}
     */
    private int sourceLayer(int il, boolean nextnFull) {
        for (int l = il; l >= 0; l--) {
            boolean attention = hp.arch() != Arch.GLM5_NEXT || !hp.isRecurrent(l);
            if (attention && ((nextnFull && l >= hp.nLayer()) || hp.isIndexerFull(l))) {
                return l;
            }
        }
        throw new IllegalStateException(hp.arch().ggufName() + ": layer " + il
                + " reuses the indexer top-k but no earlier layer computes it");
    }

    /**
     * Rotary dims the sparse-attention graphs read from {@code hparams.n_rot()}: the value of layer 0, which is the
     * SWA rotary count when layer 0 is a sliding-window layer (llama-hparams.cpp:110-116).
     */
    private int ropeDims() {
        return hp.nRot(0);
    }

    private void checkLayer(int il) {
        if (il < 0 || il >= hp.nLayerAll()) {
            throw new IndexOutOfBoundsException("layer " + il + " outside [0, " + hp.nLayerAll() + ")");
        }
    }
}
