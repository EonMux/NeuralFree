package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.overlayFlags;
import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.rmsEps;

/**
 * Hparams hook of the Step5 text decoder (vLLM {@code vllm/models/step5/nvidia/model.py},
 * {@code Step5Model} and {@code common/config.py} {@code Step5SparseConfig}). The dense GQA attention shares one
 * geometry across the trunk; full-attention layers additionally carry the weighted-ReLU sparse indexer whose proxy
 * head is 256 wide and whose rotary part is {@link GgufKeys#ATTENTION_INDEXER_ROPE_DIM} wide. The checkpoint's
 * {@code sparse_config} section is validated by the converter and stored as the indexer keys, and the arbitrary
 * {@code moe_layers_enum} / {@code moe_layer_list} becomes {@link GgufKeys#MOE_LAYER_MASK}.
 *
 * <p>The sparse indexer's fixed options (weighted ReLU scoring, RMS query norm, LayerNorm key norm, NeoX indexer
 * RoPE, region block size 8, {@code csa_block_compress}) are the only ones vLLM supports; the converter refuses the
 * others, so no metadata records them.</p>
 */
public final class Step5Hooks {

    /** {@code Step5SparseConfig.proxy_dim}. */
    public static final int PROXY_DIM = 256;
    /** {@code Step5SparseConfig.region_block_size}. */
    public static final int REGION_BLOCK_SIZE = 8;
    /** {@code _STEP5_SPARSE_TOPK_MAX}. */
    public static final int TOPK_MAX = 1024;
    /** Supported indexer query heads per provider group ({@code _SUPPORTED_INDEXER_Q_HEADS_PER_PROVIDER}). */
    private static final int[] INDEXER_Q_HEADS_PER_GROUP = {2, 4};
    /** Supported attention query heads per KV group (vLLM {@code _validate_step5_dsa_parallel_geometry}). */
    private static final int[] Q_HEADS_PER_KV = {4, 8, 16};

    private Step5Hooks() {
    }

    /**
     * Reads the Step5 keys after the generic metadata and validates the DSA geometry.
     *
     * @throws GgufFormatException when a key is missing or the sparse geometry is unsupported
     */
    public static void step5(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);
        int indexerHeads = kv.u32Required(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT);
        int proxyDim = kv.u32Required(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH);
        int topK = kv.u32Required(GgufKeys.ATTENTION_INDEXER_TOP_K);
        int blockSize = kv.u32Required(GgufKeys.ATTENTION_INDEXER_BLOCK_SIZE);
        int ropeDim = kv.u32Required(GgufKeys.ATTENTION_INDEXER_ROPE_DIM);
        int groups = kv.u32Required(GgufKeys.ATTENTION_INDEXER_GROUP_COUNT);
        b.indexerNHead(indexerHeads);
        b.indexerHeadSize(proxyDim);
        b.indexerTopK(topK);
        b.indexerBlockSize(blockSize);
        b.indexerRopeDim(ropeDim);
        b.indexerGroupCount(groups);
        b.fp32Residual(kv.bool(GgufKeys.FP32_RESIDUAL_CONNECTION, false));
        b.zeroCenteredNorms(kv.bool(GgufKeys.ZERO_CENTERED_NORMS, true));

        HparamsSupport.readExpertFfLength(b, kv, b.nExpert() > 0);
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));
        if (b.nExpert() > 0) {
            if (b.gating() == GatingFunc.SOFTMAX) {
                b.gating(GatingFunc.SIGMOID);
            }
            if (b.nExpertUsed(0) <= 0) {
                throw new GgufFormatException("step5: the routed experts need a positive expert_used_count");
            }
            if (b.nFfShexp() <= 0) {
                throw new GgufFormatException("step5: the routed experts need a positive"
                        + " expert_shared_feed_forward_length");
            }
        }
        b.moeLayers(moeMask(b, kv));

        if (kv.has(GgufKeys.ATTENTION_SLIDING_WINDOW)) {
            int window = kv.u32(GgufKeys.ATTENTION_SLIDING_WINDOW, 0);
            if (window > 0) {
                b.nSwa(window);
                b.swaType(SwaType.STANDARD);
                b.isSwaArray(overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                        b.isSwaArray(), true));
            }
        }
        validate(b);
    }

    /** The per-block MoE flags: the stored mask, else every trunk block after the dense lead when experts exist. */
    private static boolean[] moeMask(ModelHparamsBuilder b, GgufKv kv) {
        if (kv.has(GgufKeys.MOE_LAYER_MASK)) {
            boolean[] mask = kv.boolArray(GgufKeys.MOE_LAYER_MASK);
            if (mask.length != b.nLayerAll()) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.MOE_LAYER_MASK) + "' holds "
                        + mask.length + " flags for " + b.nLayerAll() + " blocks");
            }
            return mask;
        }
        boolean[] mask = new boolean[b.nLayerAll()];
        if (b.nExpert() > 0) {
            int denseLead = Math.max(1, b.nLayerDenseLead());
            for (int il = 0; il < b.nLayerAll(); il++) {
                mask[il] = il >= denseLead;
            }
        }
        return mask;
    }

    /** Refuses the sparse geometries vLLM cannot run, naming the offending key. */
    private static void validate(ModelHparamsBuilder b) {
        if (b.indexerHeadSize() != PROXY_DIM) {
            throw new GgufFormatException("step5: the sparse indexer proxy head must be " + PROXY_DIM + " wide, got "
                    + b.indexerHeadSize());
        }
        if (b.indexerBlockSize() != REGION_BLOCK_SIZE) {
            throw new GgufFormatException("step5: the sparse indexer region block must be " + REGION_BLOCK_SIZE
                    + ", got " + b.indexerBlockSize());
        }
        if (b.indexerTopK() <= 0 || b.indexerTopK() > TOPK_MAX) {
            throw new GgufFormatException("step5: the sparse indexer top-k must be in [1, " + TOPK_MAX + "], got "
                    + b.indexerTopK());
        }
        if (b.indexerRopeDim() <= 0 || b.indexerRopeDim() > PROXY_DIM || b.indexerRopeDim() % 2 != 0) {
            throw new GgufFormatException("step5: the sparse indexer rope dim must be positive, even and no larger"
                    + " than " + PROXY_DIM + ", got " + b.indexerRopeDim());
        }
        int groups = b.indexerGroupCount();
        if (groups <= 0) {
            throw new GgufFormatException("step5: the sparse indexer group count must be positive, got " + groups);
        }
        if (b.indexerNHead() % groups != 0
                || !contains(INDEXER_Q_HEADS_PER_GROUP, b.indexerNHead() / groups)) {
            throw new GgufFormatException("step5: sparse indexer head_count " + b.indexerNHead() + " over group count "
                    + groups + " must be 2 or 4 query heads per group");
        }
        for (int il = 0; il < b.nLayerAll(); il++) {
            int kvHeads = b.nHeadKv(il);
            if (kvHeads != groups) {
                throw new GgufFormatException("step5: block " + il + " has " + kvHeads + " key/value heads but the"
                        + " sparse indexer provider groups are " + groups + "; Step5 requires them to align");
            }
            int qHeads = b.nHead(il);
            if (qHeads % kvHeads != 0 || !contains(Q_HEADS_PER_KV, qHeads / kvHeads)) {
                throw new GgufFormatException("step5: block " + il + " has " + qHeads + " query heads over "
                        + kvHeads + " key/value heads; Step5 requires 4, 8 or 16 query heads per group");
            }
            if (b.nRotFull() <= 0 || b.nRotFull() > b.headKFull() || b.nRotFull() % 2 != 0) {
                throw new GgufFormatException("step5: block " + il + " rotary dim " + b.nRotFull()
                        + " must be positive, even and no larger than head dim " + b.headKFull());
            }
        }
    }

    private static boolean contains(int[] values, int v) {
        for (int x : values) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }
}
