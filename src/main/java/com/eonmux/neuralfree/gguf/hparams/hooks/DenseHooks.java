package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * {@link com.eonmux.neuralfree.gguf.hparams.ArchHparamsHook} implementations for the plain decoder attention
 * architectures, referenced from {@link com.eonmux.neuralfree.gguf.arch.ArchTableDense}. This class holds the
 * epsilon-only hooks, the LayerNorm-era and early RMSNorm families and the shared readers; the sliding-window
 * families are in {@link DenseHooksSwa} and the scaled, multi-section RoPE and adapter families in
 * {@link DenseHooksModern}.
 *
 * <p>Each hook transcribes one llama.cpp {@code llama_model_<arch>::load_arch_hparams}
 * (src/models/&lt;arch&gt;.cpp); the comments cite its lines. {@code ml.get_key(k, v)} without the
 * {@code false} argument is a required key.
 */
public final class DenseHooks {

    private static final Logger LOGGER = LogManager.getLogger(DenseHooks.class);

    /** ALiBi slope bias that llama.cpp hard-codes for baichuan-13B, refact and bloom. */
    private static final float FIXED_ALIBI_BIAS = 8.0f;
    /** Block count of baichuan-7B, the only size whose graph applies RoPE (baichuan.cpp:6, 58, 78-92). */
    private static final int BAICHUAN_7B_LAYERS = 32;
    /** Block count of baichuan-13B (baichuan.cpp:7). */
    private static final int BAICHUAN_13B_LAYERS = 40;

    private DenseHooks() {
    }

    // ---- shared readers ----

    /** Required {@code attention.layer_norm_rms_epsilon}. */
    static void readRmsEps(ModelHparamsBuilder b, GgufKv kv) {
        b.normRmsEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS));
    }

    /** Required {@code attention.layer_norm_epsilon}. */
    static void readLayerNormEps(ModelHparamsBuilder b, GgufKv kv) {
        b.normEps(kv.f32Required(GgufKeys.ATTENTION_LAYERNORM_EPS));
    }

    /**
     * Required boolean key.
     *
     * @throws GgufFormatException when the key is missing or not a boolean
     */
    static boolean boolRequired(GgufKv kv, String key) {
        if (!kv.has(key)) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        return kv.bool(key, false);
    }

    // ---- epsilon-only architectures ----

    /**
     * Architectures whose loader reads only the required RMS norm epsilon: deci.cpp:4, qwen.cpp:4,
     * qwen2.cpp:4, qwen3.cpp:4, plamo.cpp:4, internlm2.cpp:4, gemma.cpp:4, xverse.cpp:4, openelm.cpp:4,
     * chatglm.cpp:4, bitnet.cpp:4, exaone.cpp:4, arcee.cpp:4, seed-oss.cpp:4, cogvlm.cpp:4,
     * pangu-embed.cpp:4, maincoder.cpp:4.
     */
    public static void rmsEps(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
    }

    /**
     * Architectures whose loader reads only the required LayerNorm epsilon: falcon.cpp:4, gpt2.cpp:4,
     * starcoder.cpp:4, stablelm.cpp:4, phi2.cpp:4, codeshell.cpp:4, orion.cpp:4, starcoder2.cpp:4,
     * jais2.cpp:4, nemotron.cpp:4.
     */
    public static void layerNormEps(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
    }

    // ---- LayerNorm-era families ----

    /** gptneox.cpp:3-48: LayerNorm epsilon (:4) and the required parallel-residual flag (:5). */
    public static void gptneox(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
        b.useParallelResidual(boolRequired(kv, GgufKeys.USE_PARALLEL_RESIDUAL));
    }

    /** mpt.cpp:3-13: LayerNorm epsilon (:4), optional KQV clamp (:5) and optional ALiBi bias (:6). */
    public static void mpt(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
        b.clampKqv(kv.f32(GgufKeys.ATTENTION_CLAMP_KQV, b.clampKqv()));
        b.maxAlibiBias(kv.f32(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, b.maxAlibiBias()));
    }

    /**
     * baichuan.cpp:3-15: RMS epsilon (:4); the 13B variant (40 blocks, :7) uses ALiBi with bias 8 (:11-14). Only
     * the 7B variant (32 blocks) rotates Q and K: the graph builds positions for 7B alone (:58) and the 13B and
     * unknown-size variants skip the rope (:78-92), so every layer of those is a NoPE layer.
     */
    public static void baichuan(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        if (b.nLayer() == BAICHUAN_13B_LAYERS) {
            b.maxAlibiBias(FIXED_ALIBI_BIAS);
        }
        if (b.nLayer() != BAICHUAN_7B_LAYERS) {
            b.ropeEnabledArray(new boolean[b.nLayerAll()]);
        }
    }

    /** refact.cpp:3-13: RMS epsilon (:4) and ALiBi with bias 8 (:12). */
    public static void refact(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        b.maxAlibiBias(FIXED_ALIBI_BIAS);
    }

    /** bloom.cpp:3-19: LayerNorm epsilon (:4) and ALiBi with bias 8 (:18). */
    public static void bloom(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
        b.maxAlibiBias(FIXED_ALIBI_BIAS);
    }

    /** olmo.cpp:3-13: LayerNorm epsilon (:4) and optional KQV clamp (:5). */
    public static void olmo(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
        b.clampKqv(kv.f32(GgufKeys.ATTENTION_CLAMP_KQV, b.clampKqv()));
    }

    /** jais.cpp:3-13: LayerNorm epsilon (:4) and optional ALiBi bias (:5). */
    public static void jais(ModelHparamsBuilder b, GgufKv kv) {
        readLayerNormEps(b, kv);
        b.maxAlibiBias(kv.f32(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, b.maxAlibiBias()));
    }

    /** command-r.cpp:3-11: optional logit scale (:4, multiplied) and LayerNorm epsilon (:5). */
    public static void commandR(ModelHparamsBuilder b, GgufKv kv) {
        b.logitScale(kv.f32(GgufKeys.LOGIT_SCALE, b.logitScale()));
        readLayerNormEps(b, kv);
    }

    /**
     * chameleon.cpp:4-14: RMS epsilon (:5), the fixed torch-default epsilon 1e-5 of the Q/K LayerNorms (:6)
     * and the optional swin-norm placement (:7).
     */
    public static void chameleon(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        b.normEps(1e-5f);
        b.swinNorm(kv.bool(GgufKeys.SWIN_NORM, b.swinNorm()));
    }

    /** talkie.cpp:3-11: RMS epsilon (:4) and the required logit scale (:5, multiplied). */
    public static void talkie(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
    }

    // ---- Phi / PLaMo ----

    /**
     * phi3.cpp:3-25: RMS epsilon (:4). Sliding-window attention stays disabled: a positive
     * {@code attention.sliding_window} is read (:13) and then discarded with a warning (:15-24).
     */
    public static void phi3(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        boolean found = HparamsSupport.readSlidingWindow(b, kv, false);
        if (found && b.nSwa() > 0) {
            LOGGER.warn("phi3: sliding-window attention ({} tokens) is disabled, as in llama.cpp; results may be "
                    + "suboptimal for some models", b.nSwa());
            HparamsSupport.disableSwa(b);
            return;
        }
        b.swaType(SwaType.NONE);
    }

    /**
     * plamo3.cpp:3-18: RMS epsilon (:4); a positive {@code attention.sliding_window} (:5-6) enables standard
     * SWA with the optional SWA RoPE base (:8) and pattern period 8 (:9), otherwise SWA is off (:11).
     */
    public static void plamo3(ModelHparamsBuilder b, GgufKv kv) {
        readRmsEps(b, kv);
        boolean found = HparamsSupport.readSlidingWindow(b, kv, false);
        if (found && b.nSwa() > 0) {
            b.swaType(SwaType.STANDARD);
            HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
            HparamsSupport.loadSwaPattern(b, kv, 8, false);
        } else {
            b.swaType(SwaType.NONE);
        }
    }
}
