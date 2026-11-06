package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.gating;
import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.requiredGating;
import static com.eonmux.neuralfree.gguf.hparams.hooks.MoeHooksSupport.rmsEps;

/**
 * Hparams hooks for the multi-head latent attention (MLA), DeepSeek sparse attention (DSA), DeepSeek-V4 and
 * MiniMax sparse attention architectures, referenced from {@link com.eonmux.neuralfree.gguf.arch.ArchTableMoeMla}.
 * Each hook transcribes {@code load_arch_hparams} of llama.cpp src/models/&lt;file&gt;.cpp (cited
 * {@code models/<file>.cpp:line}); conventions as in {@link MoeHooks}.
 */
public final class MoeHooksMla {

    /** Leading layers of the GLM-5.2 default indexer layout that own a full indexer (models/glm-dsa.cpp:5-27). */
    private static final int GLM52_INDEXER_HEAD = 2;
    /** Number of entries of the GLM-5.2 default indexer layout (2 + 19 groups of 4). */
    private static final int GLM52_INDEXER_LEN = 78;
    /** Context length from which glm-dsa files carry the GLM-5.2 indexer layout (models/glm-dsa.cpp:60). */
    private static final int GLM52_MIN_CTX = 1048576;

    /** minicpm3 {@code scale_depth} (models/minicpm3.cpp:67). */
    private static final float MINICPM3_SCALE_DEPTH = 1.4f;
    /** minicpm3 {@code n_embd_base} (models/minicpm3.cpp:65). */
    private static final int MINICPM3_N_EMBD_BASE = 256;

    private MoeHooksMla() {
    }

    /**
     * models/minicpm3.cpp:3-6. The graph's fixed scales (:65-67) become hparams: both residual branches are scaled by
     * {@code scale_depth / sqrt(n_layer)} (:197-199, 219-220) and the normed hidden state by
     * {@code n_embd_base / n_embd} before the LM head (:241-242), which is the logit multiplier here.
     */
    public static void minicpm3(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :5
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :6
        if (b.nLayer() <= 0 || b.nEmbd() <= 0) {
            throw new GgufFormatException("minicpm3: block_count and embedding_length must be positive, got "
                    + b.nLayer() + " and " + b.nEmbd());
        }
        b.residualScale(MINICPM3_SCALE_DEPTH / (float) Math.sqrt((float) b.nLayer()));   // :197
        b.logitScale((float) MINICPM3_N_EMBD_BASE / (float) b.nEmbd());                   // :241
    }

    /**
     * models/deepseek2.cpp:3-44, shared by mistral4 (llama_model_mistral4 derives from llama_model_deepseek2).
     * Lite variants (27 or 26 blocks, or 48 blocks with a 128256 vocabulary) have no Q LoRA. Absent gating
     * is sigmoid for GLM 4.7 Lite (47/48 blocks, vocabulary 154880) and softmax otherwise. The stored YaRN
     * log multiplier has the converter's 0.1 factor removed.
     */
    public static void deepseek2(ModelHparamsBuilder b, GgufKv kv) {
        int nLayer = b.nLayer();
        int nVocab = b.nVocab();                                                          // :4-5
        boolean isLite = nLayer == 27 || nLayer == 26 || (nLayer == 48 && nVocab == 128256); // :8
        rmsEps(b, kv);                                                                    // :10
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :11
        if (!isLite) {                                                                    // :12
            b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                     // :13
        }
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :15
        readMlaHeads(b, kv, false);                                                       // :16-17
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :18
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :19
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :20
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :21
        boolean glm47Lite = (nLayer == 47 || nLayer == 48) && nVocab == 154880;           // :26
        b.gating(gating(kv, glm47Lite ? GatingFunc.SIGMOID : GatingFunc.SOFTMAX));       // :22-31
        readYarnLogMul(b, kv);                                                            // :34-38
        b.attnTempScale(kv.f32(GgufKeys.ATTENTION_TEMPERATURE_SCALE, b.attnTempScale())); // :41
        b.attnTempFloorScale(kv.u32(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, b.attnTempFloorScale())); // :42
        b.attnTempOffset(0.0f);                                                           // :44
        requireQkNope(b, "deepseek2");                                                    // tensors :65-71
        MoeHooksSupport.requireExpertsAfterLead(b, "deepseek2", b.nLayerAll());           // tensors :128-133
    }

    /**
     * vLLM's {@code AXK1ForCausalLM} (AXK1.py): the deepseek2 MLA and routed-expert hparams with the llama 4 query
     * temperature ({@code attention.temperature_scale}/{@code _length}, offset 0), an optional Q LoRA rank (0 when
     * the checkpoint has no {@code q_a_proj}) and sigmoid routing by default, vLLM's AXK1 scoring. The sparse layers
     * are the contiguous block after the dense lead; their extra post-MLP norm arrives as {@code post_ffw_norm}.
     */
    public static void axk1(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));
        b.nLoraQ(kv.u32(GgufKeys.ATTENTION_Q_LORA_RANK, 0));
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));
        readMlaHeads(b, kv, false);
        HparamsSupport.readExpertFfLength(b, kv, true);
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm()));
        b.gating(gating(kv, GatingFunc.SIGMOID));
        readYarnLogMul(b, kv);
        b.attnTempScale(kv.f32(GgufKeys.ATTENTION_TEMPERATURE_SCALE, b.attnTempScale()));
        b.attnTempFloorScale(kv.u32(GgufKeys.ATTENTION_TEMPERATURE_LENGTH, b.attnTempFloorScale()));
        b.attnTempOffset(0.0f);
        requireQkNope(b, "axk1");
        MoeHooksSupport.requireExpertsAfterLead(b, "axk1", b.nLayerAll());
    }

    /** models/deepseek2ocr.cpp:3-15 (the deepseek2 graph without MLA). */
    public static void deepseek2ocr(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :5
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :6
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :7
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :8
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :9
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :10
        b.gating(gating(kv, GatingFunc.SOFTMAX));                                         // :11-15
        MoeHooksSupport.requireExpertsAfterLead(b, "deepseek2-ocr", b.nLayer());          // tensors :58-63
    }

    /** models/deepseek32.cpp:6-38: MLA plus the DSA lightning indexer; layer-norm eps 1e-6 for the indexer. */
    public static void deepseek32(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :7, :23
        rmsEps(b, kv);                                                                    // :8
        b.normEps(1e-6f);                                                                 // :9
        HparamsSupport.readRopeSections(b, kv, false);                                    // :10
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :13, :24
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :14
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :15
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :16
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :19
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :20
        readMlaHeads(b, kv, false);                                                       // :21-22
        readIndexer(b, kv);                                                               // :27-29
        b.gating(requiredGating(kv));                                                     // :32
        readYarnLogMul(b, kv);                                                            // :34-38
        requireMla(b, "deepseek32");                                                      // tensors :53-56
        MoeHooksSupport.requireExpertsAfterLead(b, "deepseek32", b.nLayerAll());          // tensors :117-122
    }

    /**
     * models/deepseek4.cpp:19-74. Every trunk layer is a sliding-window layer ({@code set_swa_pattern(0)}) and
     * the NextN blocks are marked SWA too; image-span tokens attend bidirectionally ({@code SWA_FULL}).
     * The shared-expert SwiGLU clamp falls back to the routed-expert clamp. The loader's probe that drops
     * the NextN blocks when {@code blk.<n_layer>.nextn.eh_proj.weight} is absent (:20-26) needs the tensor
     * index and is applied by the model loader, not here.
     */
    public static void deepseek4(ModelHparamsBuilder b, GgufKv kv) {
        deepseek4Core(b, kv, false);
    }

    /**
     * models/deepseek41.cpp:27-132: the DeepSeek-V4 hparams plus the V4.1 cross-layer sharing and Engram layers.
     * Compressed layers use ratios 2 (selected) and 1 (dense); each reads the latest KV source at or before it
     * ({@code attention.kv_source_layers}) and the latest indexer source ({@code attention.indexer.source_layers}),
     * which must be at or after the KV source and share its ratio. The optional candidate selection names an index
     * source plus its block size and block count. Engram layers hash n-grams through one shared token table. No
     * hyper-connection head: the final stream collapses with the last FFN's pre-mix.
     */
    public static void deepseek41(ModelHparamsBuilder b, GgufKv kv) {
        deepseek4Core(b, kv, true);
        b.nEmbdOutImpl(b.nEmbd());                                                        // :30
        if (b.indexerTopK() <= 0) {
            throw new GgufFormatException("DeepSeek-V4.1 requires '" + kv.key(GgufKeys.ATTENTION_INDEXER_TOP_K)
                    + "' > 0, got " + b.indexerTopK());
        }
        readDsv41Sources(b, kv);                                                          // :35-86
        readEngram(b, kv);                                                                // :88-131
    }

    /** The DeepSeek-V4 hparams shared by {@link #deepseek4} and {@link #deepseek41}. */
    private static void deepseek4Core(ModelHparamsBuilder b, GgufKv kv, boolean v41) {
        rmsEps(b, kv);                                                                    // :28
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :29
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :30
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :32
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :33
        b.expertWeightsScale(kv.f32Required(GgufKeys.EXPERT_WEIGHTS_SCALE));              // :34
        b.expertWeightsNorm(MoeHooksSupport.boolRequired(kv, GgufKeys.EXPERT_WEIGHTS_NORM)); // :35
        float[] clampExp = MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_EXP, true); // :36
        b.swigluClampExpArray(clampExp);
        float[] clampShexp = MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_SHEXP, false); // :37
        b.swigluClampShexpArray(clampShexp != null ? clampShexp : clampExp);              // :38
        readIndexer(b, kv);                                                               // :41-43
        b.dsv4OGroupCount(kv.u32Required(GgufKeys.ATTENTION_OUTPUT_GROUP_COUNT));         // :45
        b.dsv4OLoraRank(kv.u32Required(GgufKeys.ATTENTION_OUTPUT_LORA_RANK));             // :46
        b.dsv4CompressRopeBase(kv.f32Required(GgufKeys.ATTENTION_COMPRESS_ROPE_FREQ_BASE)); // :47
        b.dsv4HcMult(kv.u32Required(GgufKeys.HYPER_CONNECTION_COUNT));                    // :48
        b.dsv4HcSinkhornIters(kv.u32Required(GgufKeys.HYPER_CONNECTION_SINKHORN_ITERATIONS)); // :49
        b.dsv4HcEps(kv.f32Required(GgufKeys.HYPER_CONNECTION_EPSILON));                   // :50
        b.dsv4HashLayerCount(kv.u32Required(GgufKeys.HASH_LAYER_COUNT));                  // :51
        b.nEmbdOutImpl(b.dsv4HcMult() * b.nEmbd());                                       // :53
        readCompressRatios(b, kv, v41);                                                   // :55-61
        GatingFunc g = requiredGating(kv);                                                // :63
        if (g != GatingFunc.SQRT_SOFTPLUS) {                                              // :64-66
            throw new GgufFormatException("deepseek4: expected sqrtsoftplus MoE scoring, got " + g);
        }
        b.gating(g);
        b.swaType(SwaType.STANDARD);                                                      // :67
        HparamsSupport.setSwaPattern(b, 0, false);                                        // :68
        b.nonCausalType(ModelHparams.NON_CAUSAL_SWA_FULL);                                // :71
        for (int il = b.nLayer(); il < b.nLayerAll(); il++) {                             // :72-74
            b.isSwa(il, true);
        }
    }

    /**
     * DeepSeek-V4.1 compressed-cache sharing and candidate selection (models/deepseek41.cpp:35-86).
     */
    private static void readDsv41Sources(ModelHparamsBuilder b, GgufKv kv) {
        int[] kvSrc = requiredU32Array(kv, GgufKeys.ATTENTION_KV_SOURCE_LAYERS);
        int[] idxSrc = requiredU32Array(kv, GgufKeys.ATTENTION_INDEXER_SOURCE_LAYERS);
        int nLayer = b.nLayer();
        for (int il : kvSrc) {
            if (il < 0 || il >= nLayer) {
                throw new GgufFormatException("deepseek41: attention.kv_source_layers lists layer " + il
                        + ", outside [0, " + nLayer + ")");
            }
        }
        for (int il : idxSrc) {
            if (il < 0 || il >= nLayer) {
                throw new GgufFormatException("deepseek41: attention.indexer.source_layers lists layer " + il
                        + ", outside [0, " + nLayer + ")");
            }
        }
        int lastKv = -1;
        int lastIdx = -1;
        for (int il = 0; il < nLayer; il++) {
            int ratio = b.compressRatio(il);
            boolean isKv = contains(kvSrc, il);
            boolean isIdx = contains(idxSrc, il);
            if ((isKv || isIdx) && ratio == 0) {
                throw new GgufFormatException("deepseek41: layer " + il + " is listed in '"
                        + kv.key(GgufKeys.ATTENTION_KV_SOURCE_LAYERS) + "' or '"
                        + kv.key(GgufKeys.ATTENTION_INDEXER_SOURCE_LAYERS) + "' but has compress ratio 0");
            }
            lastKv = isKv ? il : lastKv;
            lastIdx = isIdx ? il : lastIdx;
            if (ratio == 0) {
                continue;
            }
            if (lastKv < 0 || lastIdx < lastKv || b.compressRatio(lastKv) != ratio
                    || b.compressRatio(lastIdx) != ratio) {
                throw new GgufFormatException("deepseek41: layer " + il + " has compress ratio " + ratio
                        + " without a matching source in '" + kv.key(GgufKeys.ATTENTION_KV_SOURCE_LAYERS) + "' and '"
                        + kv.key(GgufKeys.ATTENTION_INDEXER_SOURCE_LAYERS) + "' (kv " + lastKv + ", indexer "
                        + lastIdx + ")");
            }
            b.dsv41KvSource(il, lastKv);
            b.dsv41TopkSource(il, lastIdx);
        }

        int cand = kv.u32(GgufKeys.ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, -1);
        if (cand >= 0) {
            int topKBlocks = kv.u32Required(GgufKeys.ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS);
            int blockSize = kv.u32Required(GgufKeys.ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE);
            if (cand >= nLayer || b.dsv41TopkSource(cand) != cand || topKBlocks <= 0 || blockSize <= 0) {
                throw new GgufFormatException("deepseek41: invalid '"
                        + kv.key(GgufKeys.ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER) + "' (the source must be an"
                        + " index source and '" + kv.key(GgufKeys.ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS) + "'/'"
                        + kv.key(GgufKeys.ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE) + "' must be > 0; got source "
                        + cand + " x " + topKBlocks + "/" + blockSize + ")");
            }
            for (int il = cand + 1; il < nLayer; il++) {
                if (b.dsv41TopkSource(il) == il && b.compressRatio(il) != b.compressRatio(cand)) {
                    throw new GgufFormatException("deepseek41: candidate consumer layer " + il + " must share the"
                            + " compress ratio of '" + kv.key(GgufKeys.ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER)
                            + "' layer " + cand);
                }
            }
            b.dsv41CandidateSourceLayer(cand);
            b.dsv41CandidateTopKBlocks(topKBlocks);
            b.dsv41CandidateBlockSize(blockSize);
        }
    }

    /**
     * DeepSeek-V4.1 Engram metadata (models/deepseek41.cpp:88-131): {@code ple.layers} names the Engram layers in
     * order; each layer owns one set of {@code ngram} hash multipliers and {@code (ngram - 1) * heads_per_ngram}
     * hash heads, stored back to back across the layers.
     */
    private static void readEngram(ModelHparamsBuilder b, GgufKv kv) {
        int nPle = Math.max(0, kv.arrayLength(GgufKeys.PLE_LAYERS));
        if (nPle == 0) {
            return;
        }
        int[] layers = requiredU32Array(kv, GgufKeys.PLE_LAYERS);
        int nGram = kv.u32Required(GgufKeys.PLE_NGRAM_SIZE);
        int perGram = kv.u32Required(GgufKeys.PLE_HEADS_PER_NGRAM);
        int headDim = kv.u32Required(GgufKeys.EMBEDDING_LENGTH_PER_LAYER);
        long[] mult = requiredU64Array(kv, GgufKeys.PLE_LAYER_MULTIPLIERS);
        long[] offsets = requiredU64Array(kv, GgufKeys.PLE_HEAD_OFFSETS);
        long[] vocab = requiredU64Array(kv, GgufKeys.PLE_HEAD_VOCAB_SIZES);
        int padId = kv.u32Required(GgufKeys.ENGRAM_PAD_TOKEN_ID);
        int[] tokenMap = requiredU32Array(kv, GgufKeys.ENGRAM_TOKEN_MAP);
        long nColsLong = nGram >= 2 ? (long) (nGram - 1) * perGram : 0;
        if (nColsLong <= 0 || nColsLong > Integer.MAX_VALUE || headDim <= 0
                || mult.length != (long) nPle * nGram
                || offsets.length != nPle * nColsLong || vocab.length != nPle * nColsLong
                || padId < 0 || padId >= tokenMap.length) {
            throw new GgufFormatException("deepseek41: Engram metadata in '" + kv.key(GgufKeys.PLE_LAYERS) + "', '"
                    + kv.key(GgufKeys.PLE_NGRAM_SIZE) + "', '" + kv.key(GgufKeys.PLE_HEADS_PER_NGRAM) + "', '"
                    + kv.key(GgufKeys.EMBEDDING_LENGTH_PER_LAYER) + "', '" + kv.key(GgufKeys.PLE_LAYER_MULTIPLIERS)
                    + "', '" + kv.key(GgufKeys.PLE_HEAD_OFFSETS) + "', '" + kv.key(GgufKeys.PLE_HEAD_VOCAB_SIZES)
                    + "', '" + kv.key(GgufKeys.ENGRAM_PAD_TOKEN_ID) + "' and '" + kv.key(GgufKeys.ENGRAM_TOKEN_MAP)
                    + "' is inconsistent (layers " + nPle + ", ngram " + nGram + ", heads/ngram " + perGram
                    + ", head dim " + headDim + ", multipliers " + mult.length + ", offsets " + offsets.length
                    + ", vocab " + vocab.length + ", pad " + padId + " of " + tokenMap.length + ")");
        }
        if ((long) nColsLong * headDim != b.nEmbd()) {
            throw new GgufFormatException("deepseek41: " + nColsLong + " Engram heads of " + headDim
                    + " do not span n_embd " + b.nEmbd());
        }
        int nCols = (int) nColsLong;
        for (int e = 0; e < nPle; e++) {
            int il = layers[e];
            if (il < 0 || il >= b.nLayer() || (e > 0 && il <= layers[e - 1])) {
                throw new GgufFormatException("deepseek41: '" + kv.key(GgufKeys.PLE_LAYERS) + "' lists Engram layer "
                    + il + " outside [0, " + b.nLayer() + ") or out of order");
            }
            b.isPleLayer(il, true);
        }
        long heads = (long) nPle * nCols;
        for (long c = 0; c < heads; c++) {
            if (vocab[(int) c] < 2 || offsets[(int) c] < 0 || vocab[(int) c] > Integer.MAX_VALUE
                    || offsets[(int) c] > Integer.MAX_VALUE
                    || vocab[(int) c] > Integer.MAX_VALUE - offsets[(int) c]) {
                throw new GgufFormatException("deepseek41: Engram head " + c + " bucket range does not fit the"
                        + " int32 row index");
            }
        }
        b.pleNgramSize(nGram);
        b.pleHeadsPerNgram(perGram);
        b.pleNHeads(nCols);
        b.pleHeadDim(headDim);
        b.nEmbdPerLayer(headDim);
        b.pleLayerMultipliers(mult);
        b.pleHeadOffsets(offsets);
        b.pleHeadVocabSizes(vocab);
        b.engramPadTokenId(padId);
        b.engramTokenMap(tokenMap);
    }

    private static boolean contains(int[] values, int v) {
        for (int value : values) {
            if (value == v) {
                return true;
            }
        }
        return false;
    }

    private static int[] requiredU32Array(GgufKv kv, String keyFmt) {
        int[] v = kv.u32Array(keyFmt);
        if (v == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(keyFmt) + "' is missing");
        }
        return v;
    }

    private static long[] requiredU64Array(GgufKv kv, String keyFmt) {
        long[] v = kv.u64Array(keyFmt);
        if (v == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(keyFmt) + "' is missing");
        }
        return v;
    }

    /** models/glm4-moe.cpp:3-18. MoE layers are {@code il >= n_layer_dense_lead}, NextN blocks included (:70). */
    public static void glm4Moe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :4
        rmsEps(b, kv);                                                                    // :5
        HparamsSupport.readRopeSections(b, kv, false);                                    // :6
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :9
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :10
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :11
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :12
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :15-18
        MoeHooksSupport.requireExperts(b, "glm4moe");                                     // tensors :36-37
    }

    /**
     * models/glm-dsa.cpp:29-66. Files with a training context below 2^20 (GLM 5, 5.1) give every layer a full
     * indexer; later files default to the GLM-5.2 layout (full at layers 0, 1 and every fourth layer from
     * 2), and {@code attention.indexer_types} over the trunk layers overrides either.
     */
    public static void glmDsa(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :30, :45
        rmsEps(b, kv);                                                                    // :31
        HparamsSupport.readRopeSections(b, kv, false);                                    // :32
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :35, :46
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :36
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :37
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :38
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :41
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :42
        readMlaHeads(b, kv, false);                                                       // :43-44
        readIndexer(b, kv);                                                               // :49-51
        b.gating(gating(kv, GatingFunc.SIGMOID));                                         // :54-57
        boolean pre52 = b.nCtxTrain() < GLM52_MIN_CTX;                                    // :60
        boolean[] full = MoeHooksSupport.layersWhere(b, il -> pre52 || il < GLM52_INDEXER_HEAD
                || (il < GLM52_INDEXER_LEN && (il - GLM52_INDEXER_HEAD) % 4 == 0));       // :61-65
        b.isIndexerFullArray(MoeHooksSupport.keyOrArrFlags(kv, GgufKeys.ATTENTION_INDEXER_TYPES,
                b.nLayer(), full, false));                                                // :66
        requireMla(b, "glm-dsa");                                                         // tensors :82-85
        MoeHooksSupport.requireExpertsAfterLead(b, "glm-dsa", b.nLayerAll());             // tensors :147-152
    }

    /** models/plm.cpp:3-5. */
    public static void plm(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :4
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :5
    }

    /**
     * models/hy-v4.cpp:24-65. Only routed experts are clamped; the shared-expert clamp stays 0. With
     * a DSA top-k the indexer key norm uses the RMS eps, layer 0 must own a full indexer, and the indexer
     * key length must exceed the RoPE dimension. The graph passes the loaded gating, which has no default.
     */
    public static void hyV4(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :25
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :26
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :27
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :28
        readMlaHeads(b, kv, true);                                                        // :29-30
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :31
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :32
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :33
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :34
        b.gating(requiredGating(kv));                                                     // :35
        float[] clampExp = MoeHooksSupport.keyOrArrF32(b, kv, GgufKeys.SWIGLU_CLAMP_EXP, false); // :39
        if (clampExp != null) {
            b.swigluClampExpArray(clampExp);
        }
        b.dsv4HcMult(kv.u32Required(GgufKeys.HYPER_CONNECTION_COUNT));                    // :41
        b.dsv4HcEps(kv.f32Required(GgufKeys.HYPER_CONNECTION_EPSILON));                   // :42
        b.hcMagnitude(kv.f32Required(GgufKeys.HYPER_CONNECTION_MAGNITUDE));               // :43
        b.indexerNHead(kv.u32(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, b.indexerNHead()));  // :46
        b.indexerHeadSize(kv.u32(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, b.indexerHeadSize())); // :47
        b.indexerTopK(kv.u32(GgufKeys.ATTENTION_INDEXER_TOP_K, b.indexerTopK()));         // :48
        if (b.indexerTopK() > 0) {                                                        // :50
            b.normEps(b.normRmsEps());                                                    // :53
            if (b.indexerNHead() == 0 || b.indexerHeadSize() <= b.nRotFull()) {           // :55-57
                throw new GgufFormatException("hy_v4: bad indexer head count / key length");
            }
            b.isIndexerFullArray(MoeHooksSupport.keyOrArrFlags(kv, GgufKeys.ATTENTION_INDEXER_TYPES,
                    b.nLayer(), b.isIndexerFullArray(), false));                          // :59
            if (!b.isIndexerFullArray()[0]) {                                             // :60-62
                throw new GgufFormatException("hy_v4: layer 0 must own an indexer, nothing precedes it to share");
            }
        }
        requireMla(b, "hy_v4");                                                           // :65
        requireQkNope(b, "hy_v4");                                                        // tensors :73-77
        MoeHooksSupport.requireExpertsAfterLead(b, "hy_v4", b.nLayer());                  // tensors :139-144
    }

    /**
     * models/dots3note.cpp:8-40: MLA with separate sliding-window LoRA/head sizes, an explicit SWA layer
     * array and per-layer DSA indexer types. NextN blocks and the leading dense blocks use a dense FFN
     * (tensors :117), so MoE layers are the trunk layers at or after the dense lead.
     */
    public static void dots3note(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :9
        b.normEps(1e-6f);                                                                 // :10
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :13
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :14
        b.nLayerDenseLead(kv.u32Required(GgufKeys.LEADING_DENSE_BLOCK_COUNT));            // :15
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :16
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :17
        b.gating(requiredGating(kv));                                                     // :18
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));                         // :21
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :22
        readMlaHeads(b, kv, true);                                                        // :23-24
        b.nLoraKvSwa(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK_SWA));                // :27
        b.headKMlaSwa(kv.u32Required(GgufKeys.ATTENTION_KEY_LENGTH_MLA_SWA));             // :28
        b.headVMlaSwa(kv.u32Required(GgufKeys.ATTENTION_VALUE_LENGTH_MLA_SWA));           // :29
        b.swaType(SwaType.STANDARD);                                                      // :31
        b.nSwa(kv.u32Required(GgufKeys.ATTENTION_SLIDING_WINDOW));                        // :32
        b.ropeFreqBaseSwa(kv.f32Required(GgufKeys.ROPE_FREQ_BASE_SWA));                   // :33
        b.isSwaArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN,
                b.isSwaArray(), true));                                                   // :34
        readIndexer(b, kv);                                                               // :37-39
        b.isIndexerFullArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_INDEXER_TYPES,
                b.isIndexerFullArray(), true));                                           // :40
        requireMla(b, "dots3note");                                                       // tensors :52-54
        MoeHooksSupport.requireExpertsAfterLead(b, "dots3note", b.nLayer());              // tensors :122-124
        final int nLayer = b.nLayer();
        final int lead = b.nLayerDenseLead();
        final boolean experts = b.nExpert() > 0;
        b.moeLayers(MoeHooksSupport.layersWhere(b, il -> experts && il < nLayer && il >= lead)); // tensors :117
    }

    /**
     * models/mistral-large4.cpp:6-56. MLA with the DSA lightning indexer on the full-indexer layers (the
     * {@code attention.indexer_types} array or all layers, layer 0 always full), SSMax attention temperature
     * ({@code s * log(pos + 1) + b} per head), learned LeRoPE factors and per-layer rope patterns
     * ({@code attention.rope_pattern}), layers without their own KV reading the cache of
     * {@code attention.kv_source_map[il]}. The indexer layer norms carry biases and use the LayerNorm epsilon
     * (:9). The Q LoRA is optional; MoE layers follow the leading dense blocks.
     */
    public static void mistralLarge4(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :7
        rmsEps(b, kv);                                                                    // :8
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));                      // :9
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :12
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :13
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :14
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :15
        b.gating(requiredGating(kv));                                                     // :16
        b.nLoraQ(kv.u32(GgufKeys.ATTENTION_Q_LORA_RANK, b.nLoraQ()));                     // :19
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));                       // :20
        readMlaHeads(b, kv, true);                                                        // :21-22
        readIndexer(b, kv);                                                               // :25-27
        boolean[] full = MoeHooksSupport.keyOrArrFlags(kv, GgufKeys.ATTENTION_INDEXER_TYPES,   // :29-30
                b.nLayer(), allTrue(b.nLayer()), false);
        if (!full[0]) {                                                                   // :31-33
            throw new GgufFormatException("mistral-large4: the first layer must have a full indexer");
        }
        b.isIndexerFullArray(full);
        b.ropeEnabledArray(MoeHooksSupport.overlayFlags(kv, GgufKeys.ATTENTION_ROPE_PATTERN,     // :35
                b.ropeEnabledArray(), false));
        readKvSourceMap(b, kv);                                                           // :36-42
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, 0.0f));                   // :44-45
        readYarnLogMul(b, kv);                                                            // :46
        b.attnTempFloorScale(1);                                                          // :48-51
        b.attnTempScale(1.0f);
        b.attnTempOffset(0.0f);
        requireMla(b, "mistral-large4");                                                  // tensors :61-63
    }

    /** Every layer flags true, the default of the mistral-large4 indexer types and rope pattern. */
    private static boolean[] allTrue(int n) {
        boolean[] all = new boolean[n];
        java.util.Arrays.fill(all, true);
        return all;
    }

    /**
     * The optional {@code attention.kv_source_map}: an entry holds the layer whose KV cache layer {@code il}
     * reads, an absent or self entry means the layer owns its cache ({@code has_kv}). The source must not be
     * later, must own a cache and must share the layer's rope pattern (mistral-large4.cpp:37-42).
     */
    private static void readKvSourceMap(ModelHparamsBuilder b, GgufKv kv) {
        int[] src = kv.keyOrArr(GgufKeys.ATTENTION_KV_SOURCE_MAP, b.nLayer(), -1);
        int[] owners = new int[b.nLayerAll()];
        for (int il = 0; il < owners.length; il++) {
            owners[il] = il;
        }
        boolean mapped = false;
        for (int il = 0; il < b.nLayer(); il++) {
            if (src[il] < 0 || src[il] == il) {
                continue;
            }
            int source = src[il];
            if (source > il) {
                throw new GgufFormatException("mistral-large4: invalid KV source layer " + source + " for layer "
                        + il + "; a source cannot be later");
            }
            if (source >= b.nLayer() || owners[source] != source) {
                throw new GgufFormatException("mistral-large4: invalid KV source layer " + source + " for layer "
                        + il + "; the source must own its own cache");
            }
            if (b.ropeEnabled(source) != b.ropeEnabled(il)) {
                throw new GgufFormatException("mistral-large4: invalid KV source layer " + source + " for layer "
                        + il + "; the source must share the layer's rope pattern");
            }
            owners[il] = source;
            mapped = true;
        }
        if (mapped) {
            b.kvOwnerLayers(owners);
        }
    }

    /** models/minimax-m3.cpp:13-33: MoE with the block-sparse MiniMax indexer (positive block size required). */
    public static void minimaxM3(ModelHparamsBuilder b, GgufKv kv) {        rmsEps(b, kv);                                                                    // :14
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead())); // :15
        HparamsSupport.readExpertFfLength(b, kv, true);                                   // :16
        b.nExpertShared(kv.u32Required(GgufKeys.EXPERT_SHARED_COUNT));                    // :17
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale())); // :18
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, b.expertWeightsNorm())); // :19
        b.gating(requiredGating(kv));                                                     // :20
        readIndexer(b, kv);                                                               // :21-23
        b.indexerBlockSize(kv.u32Required(GgufKeys.ATTENTION_INDEXER_BLOCK_SIZE));        // :24
        b.indexerLocalBlocks(kv.u32Required(GgufKeys.ATTENTION_INDEXER_LOCAL_BLOCKS));    // :25
        MoeHooksSupport.requirePositive(kv, GgufKeys.ATTENTION_INDEXER_BLOCK_SIZE, b.indexerBlockSize()); // :33
    }

    // ---- shared readers ----

    /**
     * vLLM's longcat_flash.py: MLA attention with Q and KV LoRA, a dense SwiGLU FFN on every physical block and the
     * shortcut-connected routed experts on every first (even) block. The routed-expert pool holds
     * {@code expert_zero_count} identity slots after the real experts. The n-gram variant also reads the fused
     * input-embedding hash parameters.
     */
    public static void longcatFlash(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);
        b.nLoraQ(kv.u32Required(GgufKeys.ATTENTION_Q_LORA_RANK));
        b.nLoraKv(kv.u32Required(GgufKeys.ATTENTION_KV_LORA_RANK));
        readMlaHeads(b, kv, true);
        b.nExpertZero(kv.u32(GgufKeys.EXPERT_ZERO_COUNT, 0));
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, b.expertWeightsScale()));
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, false));
        b.gating(GatingFunc.SOFTMAX);
        readYarnLogMul(b, kv);
        // the trunk holds two physical blocks per logical layer; a NextN block (block_count - nLayer) is appended
        int physical = b.nLayer();
        if (physical % 2 != 0) {
            throw new GgufFormatException("longcat-flash: block_count " + physical + " is not two blocks per"
                    + " logical layer");
        }
        if (b.nExpert() <= 0) {
            throw new GgufFormatException("longcat-flash: expert_count must be positive");
        }
        requireMla(b, "longcat-flash");
        requireQkNope(b, "longcat-flash");
        if (b.arch() == Arch.LONGCAT_FLASH_NGRAM) {
            readNgram(b, kv);
        }
    }

    /** The n-gram input embedding's hash parameters ({@code ngram_vocab_size_ratio}, {@code emb_*} of the config). */
    private static void readNgram(ModelHparamsBuilder b, GgufKv kv) {
        b.ngramVocabSize(kv.u32Required(GgufKeys.NGRAM_BASE_TABLE_SIZE));
        b.ngramNeighborCount(kv.u32Required(GgufKeys.NGRAM_NEIGHBOR_COUNT));
        b.ngramSplitCount(kv.u32Required(GgufKeys.NGRAM_SPLIT_COUNT));
        b.ngramEosTokenId((int) kv.u32Required(GgufKeys.NGRAM_EOS_TOKEN_ID));
        if (b.ngramNeighborCount() < 2 || b.ngramSplitCount() < 1) {
            throw new GgufFormatException("longcat-flash-ngram: ngram.neighbor_count must be at least 2 and"
                    + " ngram.split_count at least 1, got " + b.ngramNeighborCount() + " and " + b.ngramSplitCount());
        }
    }

    /** {@code attention.key_length_mla} / {@code value_length_mla} into the MLA head sizes. */
    private static void readMlaHeads(ModelHparamsBuilder b, GgufKv kv, boolean required) {
        if (required) {
            b.headKMlaImpl(kv.u32Required(GgufKeys.ATTENTION_KEY_LENGTH_MLA));
            b.headVMlaImpl(kv.u32Required(GgufKeys.ATTENTION_VALUE_LENGTH_MLA));
        } else {
            b.headKMlaImpl(kv.u32(GgufKeys.ATTENTION_KEY_LENGTH_MLA, b.headKMlaImpl()));
            b.headVMlaImpl(kv.u32(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, b.headVMlaImpl()));
        }
    }

    /** Required indexer head count, key length and top-k. */
    private static void readIndexer(ModelHparamsBuilder b, GgufKv kv) {
        b.indexerNHead(kv.u32Required(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT));
        b.indexerHeadSize(kv.u32Required(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH));
        b.indexerTopK(kv.u32Required(GgufKeys.ATTENTION_INDEXER_TOP_K));
    }

    /** Optional YaRN log multiplier, divided by 0.1 to cancel the converter's factor ([TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]). */
    private static void readYarnLogMul(ModelHparamsBuilder b, GgufKv kv) {
        if (kv.has(GgufKeys.ROPE_SCALING_YARN_LOG_MUL)) {
            b.yarnLogMul(kv.f32(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, 0f) / 0.1f);
        }
    }

    /**
     * {@code attention.compress_ratios}: at least {@code n_layer_all} entries and at most
     * {@link ModelHparams#MAX_LAYERS} (models/deepseek4.cpp:55-61); the first {@code n_layer_all} are used. A
     * DeepSeek-V4.1 file uses ratios 0, 2 (selected) and 1 (dense) instead of 0, 4 and 128.
     */
    private static void readCompressRatios(ModelHparamsBuilder b, GgufKv kv, boolean v41) {
        String key = GgufKeys.ATTENTION_COMPRESS_RATIOS;
        int[] ratios = kv.u32Array(key);
        if (ratios == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        if (ratios.length < b.nLayerAll()) {
            throw new GgufFormatException("deepseek4: compress_ratios is shorter than block_count");
        }
        if (ratios.length > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("deepseek4: compress_ratios has " + ratios.length + " entries, more than "
                    + ModelHparams.MAX_LAYERS);
        }
        int[] perLayer = new int[b.nLayerAll()];
        System.arraycopy(ratios, 0, perLayer, 0, perLayer.length);
        for (int ratio : perLayer) {                                                      // tensors :133-153
            boolean ok = v41 ? ratio == 0 || ratio == 2 || ratio == 1
                    : ratio == 0 || ratio == 4 || ratio == 128;
            if (!ok) {
                throw new GgufFormatException("DeepSeek-V4" + (v41 ? ".1" : "") + " loader only supports compression"
                        + " ratios " + (v41 ? "0, 2 and 1" : "0, 4 and 128") + " in '" + kv.key(key) + "', got "
                        + ratio);
            }
        }
        b.compressRatioArray(perLayer);
    }

    /**
     * The loaders' {@code n_embd_head_qk_nope = n_embd_head_k_mla() - n_rot() >= 1} assertion: the key head
     * keeps at least one non-rotary dimension. {@code n_embd_head_k_mla()} is the MLA key length when both MLA
     * lengths are set ({@code is_mla()}) and the plain key length otherwise.
     */
    private static void requireQkNope(ModelHparamsBuilder b, String arch) {
        boolean mla = b.headKMlaImpl() != 0 && b.headVMlaImpl() != 0;
        int headK = mla ? b.headKMlaImpl() : b.headKFull();
        if (headK - b.nRotFull() < 1) {
            throw new GgufFormatException(arch + ": key head length " + headK + " leaves no non-rotary dimension"
                    + " beside the " + b.nRotFull() + " rotary ones");
        }
    }

    /** The loaders' MLA requirement ({@code hparams.is_mla()}): both MLA head sizes set. */
    private static void requireMla(ModelHparamsBuilder b, String arch) {
        if (b.headKMlaImpl() == 0 || b.headVMlaImpl() == 0) {
            throw new GgufFormatException(arch + ": MLA key and value lengths are required");
        }
    }
}