package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Hooks of the encoder architectures: the BERT family, modern-bert, neo-bert, eurobert, the EmbeddingGemma
 * models and T5. Each transcribes {@code llama_model_<arch>::load_arch_hparams}; line references are to
 * llama.cpp src/models/.
 */
public final class SpecialHooksEncoder {

    private static final Logger LOGGER = LogManager.getLogger(SpecialHooksEncoder.class);

    /** Question types of a modern-bert decision model: choice, score, noul (modern-bert.cpp:4). */
    private static final int MODERN_BERT_DECISION_TYPES = 3;

    /** Names {@code llm_ffn_op_type_from_string} maps (llama-model.cpp:1109-1123). */
    private static final Set<String> FFN_OP_NAMES = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "gelu", "gelu_python", "gelu_pytorch_tanh", "gelu_new", "gelu_fast", "gelu_accurate", "gelu_python_tanh",
            "geglu", "silu", "swish", "swiglu", "relu", "reglu")));

    public static final ArchHparamsHook BERT = SpecialHooksEncoder::bert;
    public static final ArchHparamsHook NOMIC_BERT = SpecialHooksEncoder::nomicBert;
    public static final ArchHparamsHook NOMIC_BERT_MOE = SpecialHooksEncoder::nomicBertMoe;
    public static final ArchHparamsHook JINA_BERT_V2 = SpecialHooksEncoder::jinaBertV2;
    public static final ArchHparamsHook JINA_BERT_V3 = SpecialHooksEncoder::jinaBertV3;
    public static final ArchHparamsHook MODERN_BERT = SpecialHooksEncoder::modernBert;
    public static final ArchHparamsHook NEO_BERT = SpecialHooksEncoder::rmsEpsOnly;
    public static final ArchHparamsHook EUROBERT = SpecialHooksEncoder::rmsEpsOnly;
    public static final ArchHparamsHook GEMMA_EMBEDDING = SpecialHooksEncoder::gemmaEmbedding;
    public static final ArchHparamsHook GEMMA_EMBEDDING2 = SpecialHooksEncoder::gemmaEmbedding2;
    public static final ArchHparamsHook T5 = SpecialHooksEncoder::t5;
    public static final ArchHparamsHook T5ENCODER = SpecialHooksEncoder::t5Encoder;

    private SpecialHooksEncoder() {
    }

    /** bert.cpp:3-21; token types from bert.cpp:26-28. */
    private static void bert(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // bert.cpp:4
        SpecialHooks.requireTokenTypes(b, kv);                                     // bert.cpp:26-28
    }

    /** nomic-bert.cpp:3-11; token types from nomic-bert.cpp:16-18. */
    private static void nomicBert(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // nomic-bert.cpp:4
        SpecialHooks.requireTokenTypes(b, kv);                                     // nomic-bert.cpp:16-18
    }

    /**
     * nomic-bert-moe.cpp:3-12; token types from nomic-bert-moe.cpp:17-19. Layer {@code il} is an expert layer
     * when {@code moe_every_n_layers > 0 && il % moe_every_n_layers == 1} (nomic-bert-moe.cpp:35, bert.cpp:171).
     * The bert graph routes with softmax and unnormalized weights (bert.cpp:173-183).
     */
    private static void nomicBertMoe(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // nomic-bert-moe.cpp:4
        b.moeEveryN(kv.u32(GgufKeys.MOE_EVERY_N_LAYERS, 0));                      // nomic-bert-moe.cpp:5
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(false);                    // bert.cpp:180-182
        SpecialHooks.requireTokenTypes(b, kv);                                     // nomic-bert-moe.cpp:17-19
    }

    /** jina-bert-v2.cpp:3-12: ALiBi with a fixed maximum bias of 8. */
    private static void jinaBertV2(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // jina-bert-v2.cpp:4
        b.maxAlibiBias(8.0f);                                                       // jina-bert-v2.cpp:5
    }

    /** jina-bert-v3.cpp:3-11; token types from jina-bert-v3.cpp:16-18. */
    private static void jinaBertV3(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // jina-bert-v3.cpp:4
        SpecialHooks.requireTokenTypes(b, kv);                                     // jina-bert-v3.cpp:16-18
    }

    /** neo-bert.cpp:3-9 and eurobert.cpp:3-9: only the RMS norm epsilon. */
    private static void rmsEpsOnly(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // neo-bert.cpp:4, eurobert.cpp:4
    }

    /**
     * modern-bert.cpp:6-59. Symmetric SWA with every third layer dense (dense first) when
     * {@code sliding_window > 0}; the FFN activation defaults to GEGLU-erf and {@code hidden_activation}
     * overrides it; the classifier defaults to mean pooling and GELU-erf; {@code decision.block_count} turns
     * the last blocks into a full-attention decision head with three outputs (one score per question type).
     */
    private static void modernBert(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readSlidingWindow(b, kv, false);                            // modern-bert.cpp:7
        if (b.nSwa() > 0) {                                                        // modern-bert.cpp:8
            b.swaType(SwaType.SYMMETRIC);                                          // modern-bert.cpp:9
            HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());       // modern-bert.cpp:10
            HparamsSupport.loadSwaPattern(b, kv, 3, true);                         // modern-bert.cpp:11
        } else {
            b.swaType(SwaType.NONE);                                               // modern-bert.cpp:13
        }

        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));             // modern-bert.cpp:16

        String hiddenAct = kv.str(GgufKeys.HIDDEN_ACT, null);                      // modern-bert.cpp:20-24
        if (hiddenAct != null) {
            if (FFN_OP_NAMES.contains(hiddenAct)) {
                b.hiddenActivation(hiddenAct);
            } else {
                LOGGER.warn("{}: unknown {} '{}', using the GEGLU-erf FFN as llama.cpp does",
                        b.arch().ggufName(), kv.key(GgufKeys.HIDDEN_ACT), hiddenAct);
            }
        }

        if (b.poolingCls() == null) {                                              // modern-bert.cpp:27-29
            b.poolingCls(PoolingKind.MEAN);
        }
        if (!kv.has(GgufKeys.CLASSIFIER_ACTIVATION)) {                            // modern-bert.cpp:32-35
            b.classifierActivation(ActOp.GELU_ERF);
        }

        b.nLayerDecision(kv.u32(GgufKeys.DECISION_BLOCK_COUNT, 0));               // modern-bert.cpp:37
        if (b.nLayerDecision() > 0) {
            if (b.nLayerDecision() >= b.nLayer()) {                                // modern-bert.cpp:39-41
                throw SpecialHooks.invalid(b, "invalid number of decision blocks " + b.nLayerDecision()
                        + " for " + b.nLayer() + " layers");
            }
            for (int il = b.nLayer() - b.nLayerDecision(); il < b.nLayer(); il++) {  // modern-bert.cpp:43-45
                b.isSwa(il, false);
            }
            b.nEmbdOutImpl(MODERN_BERT_DECISION_TYPES);                            // modern-bert.cpp:47
            int types = SpecialHooks.tokenTypeCount(kv);                           // modern-bert.cpp:120-122
            if (types != MODERN_BERT_DECISION_TYPES) {
                throw SpecialHooks.invalid(b, "decision model must have one token type per question type ("
                        + MODERN_BERT_DECISION_TYPES + "), got " + types);
            }
        }
    }

    /**
     * gemma-embedding.cpp:3-28: symmetric SWA with five SWA layers per dense layer, non-causal, optional
     * {@code dense_2}/{@code dense_3} projections, attention scale {@code 1/sqrt(head_k)}.
     */
    private static void gemmaEmbedding(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.SYMMETRIC);                                              // gemma-embedding.cpp:4
        HparamsSupport.loadSwaPattern(b, kv, 6, false);                            // gemma-embedding.cpp:5
        b.causal(false);                                                           // gemma-embedding.cpp:7
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());           // gemma-embedding.cpp:9
        HparamsSupport.readSlidingWindow(b, kv, true);                             // gemma-embedding.cpp:10
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // gemma-embedding.cpp:11
        b.dense2FeatIn(kv.u32(GgufKeys.DENSE_2_FEAT_IN, 0));                       // gemma-embedding.cpp:14-17
        b.dense2FeatOut(kv.u32(GgufKeys.DENSE_2_FEAT_OUT, 0));
        b.dense3FeatIn(kv.u32(GgufKeys.DENSE_3_FEAT_IN, 0));
        b.dense3FeatOut(kv.u32(GgufKeys.DENSE_3_FEAT_OUT, 0));
        if (b.dense2FeatIn() != 0 && b.dense2FeatIn() != b.nEmbd()) {            // gemma-embedding.cpp:19
            throw SpecialHooks.invalid(b, "dense_2_feat_in " + b.dense2FeatIn() + " must equal n_embd " + b.nEmbd());
        }
        if (b.dense3FeatOut() != 0 && b.dense3FeatOut() != b.nEmbd()) {          // gemma-embedding.cpp:20
            throw SpecialHooks.invalid(b, "dense_3_feat_out " + b.dense3FeatOut() + " must equal n_embd " + b.nEmbd());
        }
        int headK = SpecialHooks.headK0(b);                                        // gemma-embedding.cpp:26
        if (headK <= 0) {
            throw SpecialHooks.invalid(b, "key head size must be positive, got " + headK);
        }
        b.attentionScale((float) (1.0 / Math.sqrt(headK)));
    }

    /**
     * gemma-embedding2.cpp:3-19: as gemma-embedding with attention scale 1 and per-layer embeddings; equal key
     * and value head sizes from gemma-embedding2.cpp:27-32.
     */
    private static void gemmaEmbedding2(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.SYMMETRIC);                                              // gemma-embedding2.cpp:4
        HparamsSupport.loadSwaPattern(b, kv, 6, false);                            // gemma-embedding2.cpp:5
        b.causal(false);                                                           // gemma-embedding2.cpp:7
        b.attentionScale(1.0f);                                                    // gemma-embedding2.cpp:8
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());           // gemma-embedding2.cpp:10
        HparamsSupport.readSlidingWindow(b, kv, true);                             // gemma-embedding2.cpp:11
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // gemma-embedding2.cpp:12
        b.nEmbdPerLayer(kv.u32Required(GgufKeys.EMBEDDING_LENGTH_PER_LAYER));     // gemma-embedding2.cpp:13
        SpecialHooks.requireEqualHeadSizes(b);                                     // gemma-embedding2.cpp:27-32
    }

    /**
     * t5.cpp:3-36: relative position buckets, the decoder start token and the decoder block count, which
     * defaults to the encoder block count.
     */
    private static void t5(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // t5.cpp:4
        b.nRelAttnBuckets(kv.u32Required(GgufKeys.ATTENTION_RELATIVE_BUCKETS_COUNT)); // t5.cpp:5
        if (kv.has(GgufKeys.DECODER_START_TOKEN_ID)) {                             // t5.cpp:7-10
            b.decStartTokenId(kv.u32Required(GgufKeys.DECODER_START_TOKEN_ID));
        }
        b.decNLayer(kv.u32(GgufKeys.DECODER_BLOCK_COUNT, b.nLayer()));            // t5.cpp:12-13
    }

    /** t5encoder.cpp:3-7. */
    private static void t5Encoder(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));      // t5encoder.cpp:4
        b.nRelAttnBuckets(kv.u32Required(GgufKeys.ATTENTION_RELATIVE_BUCKETS_COUNT)); // t5encoder.cpp:5
    }
}
