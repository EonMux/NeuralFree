package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hooks of the dense architectures with sliding-window attention: Gemma 2/3/3n, Command-R7B
 * (cohere2), OLMo-2/3, Muse Glimmer, EXAONE-4 and Spark-2.5. Each hook transcribes one llama.cpp
 * {@code load_arch_hparams} (src/models/&lt;arch&gt;.cpp) and keeps its exact SWA condition: unconditional for
 * gemma2, gemma3n, cohere2, muse-glimmer and spark2_5; on a positive {@code attention.sliding_window} for
 * gemma3 and olmo2; on a 64-block trunk for exaone4.
 */
public final class DenseHooksSwa {

    /** Gemma-2 default window (gemma2.cpp:5). */
    private static final int GEMMA2_DEFAULT_WINDOW = 4096;
    /** Block count of Gemma-2 27B (gemma2.cpp:20). */
    private static final int GEMMA2_27B_LAYERS = 46;
    /** Block count of Gemma-3 27B (gemma3.cpp:24). */
    private static final int GEMMA3_27B_LAYERS = 62;
    /** Gemma-3n layers that own a KV cache; later layers reuse them (gemma3n.cpp:7). */
    private static final int GEMMA3N_KV_LAYERS = 20;
    /** Block count of EXAONE-4 32B, the only variant with sliding windows (exaone4.cpp:4). */
    private static final int EXAONE4_32B_LAYERS = 64;
    /** EXAONE-4 32B window (exaone4.cpp:6). */
    private static final int EXAONE4_WINDOW = 4096;
    /** Epsilon of Muse Glimmer's post-attention and post-FFN norms (muse-glimmer.cpp:62-63). */
    private static final float MUSE_GLIMMER_POST_NORM_EPS = 1e-8f;

    private DenseHooksSwa() {
    }

    /**
     * gemma2.cpp:3-28: unconditional standard SWA with window 4096 and period 2 (:4-6), attention logit
     * softcapping on (:7), SWA RoPE base/scale copied from the full ones then the optional SWA base (:8-11),
     * optional window (:12), RMS epsilon (:13), optional softcap values (:14-15) and the KQ scale
     * {@code 1/sqrt(n_embd/n_head)} for 27B, {@code 1/sqrt(head_k)} otherwise (:24-27).
     */
    public static void gemma2(ModelHparamsBuilder b, GgufKv kv) {
        HparamsSupport.enableSwa(b, SwaType.STANDARD, GEMMA2_DEFAULT_WINDOW);
        HparamsSupport.loadSwaPattern(b, kv, 2, false);
        b.attnSoftcapEnabled(true);
        copyRopeToSwa(b);
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        HparamsSupport.readSlidingWindow(b, kv, false);
        DenseHooks.readRmsEps(b, kv);
        b.attnSoftcap(kv.f32(GgufKeys.ATTN_LOGIT_SOFTCAPPING, b.attnSoftcap()));
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, b.finalSoftcap()));
        b.attentionScale(gemmaQueryScale(b, GEMMA2_27B_LAYERS));
    }

    /**
     * gemma3.cpp:3-32: standard SWA with period 6 and the optional SWA RoPE base when the window is positive
     * (:4-9), otherwise none (:11); final softcapping off unless the key is present (:14-15); RMS epsilon
     * (:16); KQ scale as gemma2 with the 62-block 27B variant (:28-31).
     */
    public static void gemma3(ModelHparamsBuilder b, GgufKv kv) {
        boolean found = HparamsSupport.readSlidingWindow(b, kv, false);
        if (found && b.nSwa() > 0) {
            b.swaType(SwaType.STANDARD);
            HparamsSupport.loadSwaPattern(b, kv, 6, false);
            HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        } else {
            b.swaType(SwaType.NONE);
        }
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, 0.0f));
        DenseHooks.readRmsEps(b, kv);
        b.attentionScale(gemmaQueryScale(b, GEMMA3_27B_LAYERS));
    }

    /**
     * gemma3n.cpp:3-19: unconditional standard SWA with period 5 (:4-5), KV caches owned by the first 20
     * layers (:7), KQ scale 1 (:8), optional SWA RoPE base (:10), required window (:11) and RMS epsilon
     * (:12).
     */
    public static void gemma3n(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.STANDARD);
        HparamsSupport.loadSwaPattern(b, kv, 5, false);
        b.nLayerKvFromStart(GEMMA3N_KV_LAYERS);
        b.attentionScale(1.0f);
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        HparamsSupport.readSlidingWindow(b, kv, true);
        DenseHooks.readRmsEps(b, kv);
    }

    /**
     * cohere2.cpp:3-19: unconditional standard SWA with period 4 (:4-5), SWA RoPE base/scale copied then the
     * optional SWA base (:7-10), required window (:11), required logit scale (:12, multiplied) and LayerNorm
     * epsilon (:13).
     */
    public static void cohere2(ModelHparamsBuilder b, GgufKv kv) {
        b.swaType(SwaType.STANDARD);
        HparamsSupport.loadSwaPattern(b, kv, 4, false);
        copyRopeToSwa(b);
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        HparamsSupport.readSlidingWindow(b, kv, true);
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
        DenseHooks.readLayerNormEps(b, kv);
    }

    /**
     * olmo2.cpp:3-25: RMS epsilon (:4); a positive window (:6-7) enables standard SWA with period 4 (:8-9),
     * the full RoPE base with SWA scale 1 and the optional SWA base (:11-13) (OLMo-3); otherwise none (:15).
     */
    public static void olmo2(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        boolean found = HparamsSupport.readSlidingWindow(b, kv, false);
        if (found && b.nSwa() > 0) {
            b.swaType(SwaType.STANDARD);
            HparamsSupport.loadSwaPattern(b, kv, 4, false);
            b.ropeFreqBaseSwa(b.ropeFreqBase());
            b.ropeFreqScaleSwa(1.0f);
            HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        } else {
            b.swaType(SwaType.NONE);
        }
    }

    /**
     * muse-glimmer.cpp:3-19: RMS epsilon (:4), required window (:5), optional final softcap (:6), required
     * logit scale (:7, multiplied), SWA RoPE base copied then optional (:9-10), unconditional standard SWA
     * with period 4 (:12-13). The graph's fixed post-norm epsilon 1e-8 (:62-63) is recorded as
     * {@code postNormEps}, and RoPE runs on the SWA layers only (:87-88), recorded per layer.
     */
    public static void museGlimmer(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        HparamsSupport.readSlidingWindow(b, kv, true);
        b.finalSoftcap(kv.f32(GgufKeys.FINAL_LOGIT_SOFTCAPPING, b.finalSoftcap()));
        b.logitScale(kv.f32Required(GgufKeys.LOGIT_SCALE));
        b.ropeFreqBaseSwa(b.ropeFreqBase());
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        b.swaType(SwaType.STANDARD);
        HparamsSupport.loadSwaPattern(b, kv, 4, false);
        b.postNormEps(MUSE_GLIMMER_POST_NORM_EPS);
        ropeOnSwaLayersOnly(b);
    }

    /**
     * exaone4.cpp:3-22: only the 64-block 32B variant (:4) uses standard SWA with window 4096 and period 4
     * (:5-7) and SWA RoPE base/scale copied then the optional SWA base (:9-11); the optional window (:14) and
     * RMS epsilon (:15) follow. With SWA the graph applies RoPE on SWA layers only (exaone4.cpp:111), recorded
     * per layer.
     */
    public static void exaone4(ModelHparamsBuilder b, GgufKv kv) {
        if (b.nLayer() == EXAONE4_32B_LAYERS) {
            HparamsSupport.enableSwa(b, SwaType.STANDARD, EXAONE4_WINDOW);
            HparamsSupport.loadSwaPattern(b, kv, 4, false);
            copyRopeToSwa(b);
            HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
        }
        HparamsSupport.readSlidingWindow(b, kv, false);
        DenseHooks.readRmsEps(b, kv);
        if (b.swaType() != SwaType.NONE) {
            ropeOnSwaLayersOnly(b);
        }
    }

    /**
     * spark2-5.cpp:3-18: RMS epsilon (:4), required window (:5), standard SWA (:7) with the per-layer flags
     * taken from the required {@code attention.sliding_window_pattern} array (:8), SWA RoPE base/scale copied
     * then the optional SWA base (:10-12).
     */
    public static void spark25(ModelHparamsBuilder b, GgufKv kv) {
        DenseHooks.readRmsEps(b, kv);
        HparamsSupport.readSlidingWindow(b, kv, true);
        b.swaType(SwaType.STANDARD);
        b.isSwaArray(requiredSwaFlags(b, kv));
        copyRopeToSwa(b);
        HparamsSupport.readRopeFreqBaseSwa(b, kv, b.ropeFreqBaseSwa());
    }

    // ---- helpers ----

    /** {@code rope_freq_{base,scale}_train_swa = rope_freq_{base,scale}_train}. */
    private static void copyRopeToSwa(ModelHparamsBuilder b) {
        b.ropeFreqBaseSwa(b.ropeFreqBase());
        b.ropeFreqScaleSwa(b.ropeFreqScale());
    }

    /**
     * Gemma query scale in single precision: {@code 1/sqrt(n_embd / n_head(0))} (integer division) for the
     * 27B variant, otherwise {@code 1/sqrt(n_embd_head_k(0))}, the key head size of layer 0, which is the SWA
     * head size when layer 0 is a sliding-window layer (llama-hparams.cpp:140-146).
     */
    private static float gemmaQueryScale(ModelHparamsBuilder b, int layers27b) {
        if (b.nLayer() == layers27b) {
            int nHead0 = b.nHead(0);
            if (nHead0 <= 0) {
                throw new GgufFormatException(b.arch().ggufName() + ": layer 0 has no attention heads");
            }
            return 1.0f / (float) Math.sqrt(b.nEmbd() / nHead0);
        }
        int headK0 = b.isSwa(0) ? b.headKSwa() : b.headKFull();
        if (headK0 <= 0) {
            throw new GgufFormatException(b.arch().ggufName() + ": layer 0 has no key head size");
        }
        return 1.0f / (float) Math.sqrt(headK0);
    }

    /** Enables RoPE exactly on the SWA layers. */
    private static void ropeOnSwaLayersOnly(ModelHparamsBuilder b) {
        b.ropeEnabledArray(b.isSwaArray());
    }

    /**
     * The required per-layer SWA flags array, as llama.cpp {@code get_arr} into {@code is_swa_impl}: at most
     * {@link ModelHparams#MAX_LAYERS} entries, missing trailing entries mean dense layers, entries past the
     * block count are unused.
     */
    private static boolean[] requiredSwaFlags(ModelHparamsBuilder b, GgufKv kv) {
        String key = GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN;
        boolean[] flags = kv.boolArray(key);
        if (flags == null) {
            throw new GgufFormatException("required metadata key '" + kv.key(key) + "' is missing");
        }
        if (flags.length > ModelHparams.MAX_LAYERS) {
            throw new GgufFormatException("metadata key '" + kv.key(key) + "' has " + flags.length
                    + " entries, more than the maximum of " + ModelHparams.MAX_LAYERS + " layers");
        }
        boolean[] swa = new boolean[b.nLayerAll()];
        System.arraycopy(flags, 0, swa, 0, Math.min(flags.length, swa.length));
        return swa;
    }
}
