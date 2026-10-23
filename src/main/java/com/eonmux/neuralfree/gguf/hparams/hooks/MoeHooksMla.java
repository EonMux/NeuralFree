package com.eonmux.neuralfree.gguf.hparams.hooks;

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
        readCompressRatios(b, kv);                                                        // :55-61
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

    /** models/minimax-m3.cpp:13-33: MoE with the block-sparse MiniMax indexer (positive block size required). */
    public static void minimaxM3(ModelHparamsBuilder b, GgufKv kv) {
        rmsEps(b, kv);                                                                    // :14
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
     * {@link ModelHparams#MAX_LAYERS} (models/deepseek4.cpp:55-61); the first {@code n_layer_all} are used.
     */
    private static void readCompressRatios(ModelHparamsBuilder b, GgufKv kv) {
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
            if (ratio != 0 && ratio != 4 && ratio != 128) {
                throw new GgufFormatException("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128, got "
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
