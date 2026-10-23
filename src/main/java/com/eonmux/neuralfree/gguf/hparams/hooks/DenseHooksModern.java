package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.RopeScalingType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;

/**
 * Hparams hooks of the dense architectures with multi-section RoPE, scaled residual/logit heads, NoPE
 * layers, xIELU activations, embedded LoRA adapters or attention temperature: the Qwen2-VL/Qwen3 family,
 * SenseNova-U1, MiniCPM, GLM-4, Granite and Granite-Switch, ERNIE-4.5/PaddleOCR, Hunyuan, SmolLM3, Apertus
 * and Mistral-3. Each hook transcribes one llama.cpp {@code load_arch_hparams} (src/models/&lt;arch&gt;.cpp).
 */
public final class DenseHooksModern {

    private static final Logger LOGGER = LogManager.getLogger(DenseHooksModern.class);

    /** RMS epsilon of a Qwen2-VL file that stores none: the Qwen2 RMSNorm default. */
    private static final float QWEN2VL_DEFAULT_RMS_EPS = 1e-6f;
    /** MiniCPM embedding scale of GGUFs without the key (minicpm.cpp:5). */
    private static final float MINICPM_DEFAULT_EMBEDDING_SCALE = 12.0f;
    /** MiniCPM {@code dim_model_base}: the logit divisor defaults to {@code n_embd / 256}. */
    private static final float MINICPM_DIM_MODEL_BASE = 256.0f;
    /** SmolLM3 skips RoPE on every 4th layer (smollm3.cpp:5). */
    private static final int SMOLLM3_NO_ROPE_STEP = 4;
    /** Upper bound of the Granite-Switch adapter count and LoRA rank (granite-switch.cpp:30-35). */
    private static final int GRANITE_SWITCH_MAX_DIM = 4096;
    /** Granite-Switch router gain when the key is absent (models.h:1837). */
    private static final float GRANITE_SWITCH_DEFAULT_ROUTER_GAIN = 15.0f;

    private DenseHooksModern() {
    }

    // ---- Qwen family ----

    /**
     * qwen2vl.cpp:3-5 reads only the required M-RoPE sections (:4). llama.cpp never reads the RMS epsilon for
     * this architecture; the optional key is read here, defaulting to the Qwen2 value 1e-6.
     */
    public static void qwen2vl(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readRopeSections(b, kv, true);
        b.normRmsEps(kv.f32(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, QWEN2VL_DEFAULT_RMS_EPS));
    }

    /**
     * sensenova-u1.cpp:28-39: RMS epsilon (:29), required spatial RoPE base (:31) and equal Q/K/V head sizes
     * divisible by four (:32-34). The graph rotates text with NEOX over the temporal half of each head
     * ({@code n_dims = head/2}, freq_scale 1, ext_factor 0, attn_factor 1; :96-104); the spatial halves see
     * position 0 for text and are identities. That rotation is recorded as the rotary width and neutral
     * scaling, so the architecture runs as NEOX rather than M-RoPE.
     */
    public static void sensenovaU1(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.ropeFreqBaseSpatial(kv.f32Required(GgufKeys.ROPE_FREQ_BASE_SPATIAL));
        int headK = b.headKFull();
        if (headK % 4 != 0 || b.headVFull() != headK) {
            throw new GgufFormatException("sensenova_u1: requires equal Q/K/V head sizes divisible by four, got key "
                    + headK + " and value " + b.headVFull());
        }
        b.nRotFull(headK / 2);
        b.nRotSwa(headK / 2);
        b.ropeFreqScale(1.0f);
        b.ropeScalingType(RopeScalingType.NONE);
        b.ropeAttnFactor(1.0f);
    }

    /**
     * qwen3moe.cpp:3-12: optional per-layer expert FFN size (:4) and RMS epsilon (:5). The tensor loader
     * rejects files without experts (:42-47); the check runs here so it fails before tensors are touched. The graph
     * routes with softmax and renormalizes the selected weights ({@code build_moe_ffn(..., norm_w = true, ...,
     * LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)}, :136-146).
     */
    public static void qwen3moe(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readExpertFfLength(b, kv, false);
        DenseHooks.readRmsEps(b, kv);
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);
        if (b.nExpert() == 0) {
            throw new GgufFormatException("qwen3moe: n_expert must be > 0");
        }
        if (Arrays.stream(b.nExpertUsedArray()).max().orElse(0) == 0) {
            throw new GgufFormatException("qwen3moe: n_expert_used must be > 0");
        }
    }

    /** qwen3vl.cpp:3-14: optional deepstack layer count (:4), required IMRoPE sections (:5), RMS epsilon (:6). */
    public static void qwen3vl(ModelHparamsBuilder b, GgufKv kv) {
        b.nDeepstackLayers(kv.u32(GgufKeys.NUM_DEEPSTACK_LAYERS, b.nDeepstackLayers()));
        HparamsSupport.readRopeSections(b, kv, true);
        DenseHooks.readRmsEps(b, kv);
    }

    // ---- scaled heads ----

    /**
     * minicpm.cpp:3-24: embedding scale 12 and residual scale {@code 1.4/sqrt(n_layer)} as defaults (:5-6),
     * RMS epsilon (:9), optional overrides (:12-14), RoPE always on (:17). The logit scale divides the logits
     * (MiniCPM uses the Granite graph, models.h:1885); when absent it defaults to {@code n_embd/256}, the
     * Hugging Face {@code hidden_size / dim_model_base}; llama.cpp's {@code 256/n_embd} at :7 inverts the scale
     * under division. Expert layers (MiniCPM-MoE) route with softmax and renormalize the selected weights, as
     * the shared Granite graph hard-codes ({@code build_moe_ffn(..., norm_w = true, ...,
     * LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)}, granite.cpp:277-287); no GGUF key controls it.
     */
    public static void minicpm(ModelHparamsBuilder b, GgufKv kv) {
        b.embeddingScale(MINICPM_DEFAULT_EMBEDDING_SCALE);
        b.residualScale(1.4f / (float) Math.sqrt(b.nLayer()));
        DenseHooks.readRmsEps(b, kv);
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));
        b.residualScale(kv.f32(GgufKeys.RESIDUAL_SCALE, b.residualScale()));
        if (kv.has(GgufKeys.LOGIT_SCALE)) {
            b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
        } else {
            float scale = b.nEmbd() != 0 ? b.nEmbd() / MINICPM_DIM_MODEL_BASE : 1.0f;
            b.logitScale(scale);
            LOGGER.info("minicpm: {} is absent; dividing logits by n_embd/{} = {}", kv.key(GgufKeys.LOGIT_SCALE),
                    (int) MINICPM_DIM_MODEL_BASE, scale);
        }
        b.ropeFinetuned(true);
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // granite.cpp:277-287
    }

    /**
     * granite.cpp:5-57: RMS epsilon (:6), required logit scale (:7, divided), optional residual, embedding and
     * attention scales (:8-10), the optional Granite-4 Vision deepstack mapping with its unique-index count
     * and bound check (:13-31), {@code rope.scaling.finetuned} defaulting to true as the per-layer RoPE switch
     * (:34-37) and the optional shared-expert FFN size (:56). Expert layers route with softmax and renormalize
     * the selected weights, hard-coded by the graph (granite.cpp:277-287).
     */
    public static void granite(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
        readGraniteScales(b, kv);
        readDeepstackMapping(b, kv);
        applyRopeFinetunedSwitch(b, kv);
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);                             // granite.cpp:277-287
    }

    /**
     * granite-switch.cpp:5-72: the Granite keys (:6-15, :23), the required adapter count and LoRA rank with
     * the optional router gain (:25-35), the required activate/substitute token id arrays of one entry per
     * adapter (:39-46), and one extra single-head, FFN-less router layer appended after the trunk as the
     * NextN slot (:61-71). The substitute ids are range-checked against the vocabulary as the tensor loader
     * does (:85-91).
     */
    public static void graniteSwitch(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
        readGraniteScales(b, kv);
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));

        int nAdapters = kv.u32Required(GgufKeys.ADAPTER_COUNT);
        int loraRank = kv.u32Required(GgufKeys.ADAPTER_LORA_RANK);
        if (nAdapters > GRANITE_SWITCH_MAX_DIM) {
            throw new GgufFormatException("graniteswitch: invalid adapter count " + nAdapters);
        }
        if (loraRank > GRANITE_SWITCH_MAX_DIM) {
            throw new GgufFormatException("graniteswitch: invalid lora rank " + loraRank);
        }
        int[] activate = requiredI32Array(kv, GgufKeys.ADAPTER_TOKEN_IDS_ACTIVATE);
        int[] substitute = requiredI32Array(kv, GgufKeys.ADAPTER_TOKEN_IDS_SUBSTITUTE);
        if (activate.length != nAdapters || substitute.length != nAdapters) {
            throw new GgufFormatException("graniteswitch: adapter token id arrays (" + activate.length
                    + " activate, " + substitute.length + " substitute) do not match adapter count " + nAdapters);
        }
        int nVocab = b.nVocab();
        for (int sub : substitute) {
            if (nVocab > 0 && (sub < 0 || sub >= nVocab)) {
                throw new GgufFormatException("graniteswitch: substitute token id " + sub + " out of range [0, "
                        + nVocab + ")");
            }
        }
        b.adapterCount(nAdapters);
        b.adapterLoraRank(loraRank);
        b.adapterRouterGain(kv.f32(GgufKeys.ADAPTER_ROUTER_GAIN, GRANITE_SWITCH_DEFAULT_ROUTER_GAIN));
        b.adapterTokenIdsActivate(activate);
        b.adapterTokenIdsSubstitute(substitute);

        int nReal = b.nLayer();
        if (nReal >= ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("graniteswitch: block count " + nReal + " exceeds " + ModelHparams.MAX_LAYERS);
        }
        b.layerCounts(nReal + 1, 1);
        b.routerLayer(nReal);
        b.nHead(nReal, 1);
        b.nHeadKv(nReal, 1);
        b.nFf(nReal, 0);
        applyRopeFinetunedSwitch(b, kv);
    }

    // ---- multi-section RoPE ----

    /** glm4.cpp:3-13: RMS epsilon (:4) and the optional M-RoPE sections (:5) that switch GLM-4V to MROPE. */
    public static void glm4(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        HparamsSupport.readRopeSections(b, kv, false);
    }

    /**
     * ernie4-5.cpp:3-21, dense branch: the optional M-RoPE sections used by PaddleOCR (:5) and RMS epsilon
     * (:7). PaddleOCR reuses this loader (models.h:2143-2145).
     */
    public static void ernie45(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.readRopeSections(b, kv, false);
        DenseHooks.readRmsEps(b, kv);
    }

    /**
     * hunyuan-vl.cpp:3-21 (also hunyuan-dense, models.h:2260-2262): RMS epsilon (:4), optional M-RoPE
     * sections (:5) and the XDRoPE NTK-aware base {@code theta * alpha^(dim/(dim-2))} when
     * {@code rope.scaling.alpha > 0} (:8-12).
     */
    public static void hunyuanVl(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        HparamsSupport.readRopeSections(b, kv, false);
        float alpha = b.ropeScalingAlpha();
        if (alpha > 0.0f) {
            int dim = b.headKFull();
            if (dim <= 2) {
                throw new GgufFormatException(b.arch().ggufName() + ": XDRoPE needs a head size above 2, got " + dim);
            }
            float exponent = (float) dim / (float) (dim - 2);
            b.ropeFreqBase(b.ropeFreqBase() * (float) Math.pow(alpha, exponent));
        }
    }

    // ---- NoPE layers, activations, attention temperature ----

    /**
     * smollm3.cpp:3-11: RMS epsilon (:4) and a NoPE layer every 4th layer (:5); the graph skips RoPE when
     * {@code (il + 1) % 4 == 0} (:69), recorded per layer.
     */
    public static void smollm3(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.noRopeLayerStep(SMOLLM3_NO_ROPE_STEP);
        for (int il = 0; il < b.nLayer(); il++) {
            b.ropeEnabled(il, (il + 1) % SMOLLM3_NO_ROPE_STEP != 0);
        }
    }

    /** apertus.cpp:3-15: RMS epsilon (:4) and the four required per-layer xIELU parameters (:6-9). */
    public static void apertus(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.xieluAlphaNArray(requiredTrunkF32(b, kv, GgufKeys.XIELU_ALPHA_N));
        b.xieluAlphaPArray(requiredTrunkF32(b, kv, GgufKeys.XIELU_ALPHA_P));
        b.xieluBetaArray(requiredTrunkF32(b, kv, GgufKeys.XIELU_BETA));
        b.xieluEpsArray(requiredTrunkF32(b, kv, GgufKeys.XIELU_EPS));
    }

    /**
     * mistral3.cpp:3-29: RMS epsilon (:4), optional attention temperature scale (:5), optional YaRN beta and
     * log-multiplier keys (:7-9), optional realtime delay (:11), temperature offset 0 (:13) and, when the
     * temperature is on, the floor scale {@code n_ctx_orig_yarn}, which must be non-zero (:16-21). MoE layers route
     * with softmax and renormalize the selected weights ({@code build_moe_ffn(..., norm_w = true, ...,
     * LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX)}, :253-262).
     */
    public static void mistral3(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        b.gating(GatingFunc.SOFTMAX).expertWeightsNorm(true);
        b.attnTempScale(kv.f32(GgufKeys.ATTENTION_TEMPERATURE_SCALE, b.attnTempScale()));
        b.yarnBetaFast(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, b.yarnBetaFast()));
        b.yarnBetaSlow(kv.f32(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, b.yarnBetaSlow()));
        b.yarnLogMul(kv.f32(GgufKeys.ROPE_SCALING_YARN_LOG_MUL, b.yarnLogMul()));
        b.nTokensDelay(kv.u32(GgufKeys.REALTIME_DELAY_TOKENS, b.nTokensDelay()));
        b.attnTempOffset(0.0f);
        if (b.attnTempScale() != 0.0f) {
            b.attnTempFloorScale(b.nCtxOrigYarn());
            if (b.attnTempFloorScale() == 0) {
                throw new GgufFormatException("mistral3: invalid n_ctx_orig_yarn for attention temperature scaling");
            }
        }
    }

    // ---- helpers ----

    /** Optional Granite residual, embedding and attention scales (granite.cpp:8-10, granite-switch.cpp:8-10). */
    private static void readGraniteScales(ModelHparamsBuilder b, GgufKv kv) {
        b.residualScale(kv.f32(GgufKeys.RESIDUAL_SCALE, b.residualScale()));
        b.embeddingScale(kv.f32(GgufKeys.EMBEDDING_SCALE, b.embeddingScale()));
        b.attentionScale(kv.f32(GgufKeys.ATTENTION_SCALE, b.attentionScale()));
    }

    /**
     * Granite uses {@code rope.scaling.finetuned} as the RoPE switch: default true, copied to every layer's
     * RoPE flag (granite.cpp:34-37, granite-switch.cpp:12-15).
     */
    private static void applyRopeFinetunedSwitch(ModelHparamsBuilder b, GgufKv kv) {
        boolean ropeFinetuned = kv.bool(GgufKeys.ROPE_SCALING_FINETUNED, true);
        b.ropeFinetuned(ropeFinetuned);
        boolean[] rope = new boolean[b.nLayerAll()];
        Arrays.fill(rope, ropeFinetuned);
        b.ropeEnabledArray(rope);
    }

    /**
     * granite.cpp:13-31: the optional {@code deepstack_mapping} array (-1 = no deepstack input); the deepstack
     * layer count is the number of distinct non-negative indices, and no index may exceed that count.
     */
    private static void readDeepstackMapping(ModelHparamsBuilder b, GgufKv kv) {
        int[] mapping = kv.i32Array(GgufKeys.DEEPSTACK_MAPPING);
        Set<Integer> unique = new TreeSet<>();
        if (mapping != null) {
            if (mapping.length > ModelHparams.MAX_LAYERS) {
                throw new GgufFormatException("metadata key '" + kv.key(GgufKeys.DEEPSTACK_MAPPING) + "' has "
                        + mapping.length + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS);
            }
            int[] perLayer = new int[b.nLayerAll()];
            Arrays.fill(perLayer, -1);
            System.arraycopy(mapping, 0, perLayer, 0, Math.min(mapping.length, perLayer.length));
            b.deepstackMappingArray(perLayer);
            for (int v : mapping) {
                if (v >= 0) {
                    unique.add(v);
                }
            }
        }
        b.nDeepstackLayers(unique.size());
        for (int v : unique) {
            if (v > unique.size()) {
                throw new GgufFormatException("Invalid deepstack index: " + v + " > " + unique.size());
            }
        }
    }

    private static int[] requiredI32Array(GgufKv kv, String key) {
        int[] values = kv.i32Array(key);
        if (values == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        return values;
    }

    /**
     * Required per-layer float key over the trunk ({@code get_key_or_arr(key, arr, n_layer())}), stored in a
     * block-sized array whose NextN entries stay 0.
     */
    private static float[] requiredTrunkF32(ModelHparamsBuilder b, GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        float[] trunk = kv.keyOrArrF32(key, b.nLayer(), 0f);
        float[] all = new float[b.nLayerAll()];
        System.arraycopy(trunk, 0, all, 0, trunk.length);
        return all;
    }
}
