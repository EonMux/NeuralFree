package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.ModelHparams;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;

/**
 * Hparams hook and lightning-attention decay slopes of minimax-01. Lightning (linear, recurrent) layers
 * alternate with softmax attention layers per {@code attention.recurrent_layers} or
 * {@code full_attention_interval} (default 8). Both residual branches add the RMS-normed block input scaled
 * by {@code residual_scale} ({@link com.eonmux.neuralfree.gguf.arch.ResidualKind#NORMED_INPUT_SCALED};
 * minimax-01.cpp:246-249, 428, 440, 455).
 */
public final class HybridHooksLightning {

    /** Default {@code full_attention_interval} (minimax-01.cpp:13). */
    static final int DEFAULT_FULL_ATTENTION_INTERVAL = 8;

    private HybridHooksLightning() {
    }

    /**
     * minimax-01.cpp:4-18. The lightning state is one {@code head_k x head_k} matrix per head, so the
     * linear-attention head size is the full K head size (minimax-01.cpp:9). Routing is softmax with
     * normalized weights regardless of metadata (minimax-01.cpp:449-451). The per-layer slope scale divides
     * by {@code n_layer - 1} (minimax-01.cpp:288), so a model with lightning layers needs two blocks.
     */
    public static void minimax01(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);                                        // minimax-01.cpp:5
        b.residualScale(kv.f32Required(GgufKeys.RESIDUAL_SCALE));             // minimax-01.cpp:6
        b.headLinearAttn(b.headKFull());                                      // minimax-01.cpp:9
        HybridHooks.readRecurrentLayers(b, kv, DEFAULT_FULL_ATTENTION_INTERVAL);   // minimax-01.cpp:12-18
        b.gating(GatingFunc.SOFTMAX);                                         // minimax-01.cpp:451
        b.expertWeightsNorm(true);                                            // minimax-01.cpp:449
        HybridHooks.requireEqualHeadSizes(b, "minimax-01.cpp:195");
        if (b.nLayer() < 2) {
            for (int il = 0; il < b.nLayer(); il++) {
                if (b.isRecurrent(il)) {
                    throw HybridHooks.invalid(b, "lightning attention needs at least two blocks for its per-layer"
                            + " slope scale 1 - il / (n_layer - 1), got " + b.nLayer());
                }
            }
        }
    }

    /**
     * Base decay slope of every lightning head: {@code start * ratio^h} with
     * {@code start = ratio = 2^(-2^(-(log2(n_head) - 3)))}, {@code n_head = hparams.n_head()}
     * (minimax-01.cpp:74, 98-103).
     *
     * @return one slope per head of layer 0
     */
    public static float[] lightningSlopes(ModelHparams hp) {
        int nHead = hp.nHead(0);
        if (nHead <= 0) {
            throw new IllegalArgumentException(hp.arch().ggufName() + ": lightning attention needs a positive head count");
        }
        float log2Heads = (float) (Math.log(nHead) / Math.log(2.0));
        float start = (float) Math.pow(2.0, -Math.pow(2.0, -(log2Heads - 3.0f)));
        float[] slopes = new float[nHead];
        for (int h = 0; h < nHead; h++) {
            slopes[h] = start * (float) Math.pow(start, h);
        }
        return slopes;
    }

    /**
     * Per-layer factor applied to the base slopes: {@code 1 - il / (n_layer - 1) + 1e-5}, evaluated in double
     * precision as the reference does (minimax-01.cpp:288).
     *
     * @throws IllegalArgumentException when the model has fewer than two blocks or {@code il} is not a trunk layer
     */
    public static float lightningSlopeScale(ModelHparams hp, int il) {
        int nLayer = hp.nLayer();
        if (nLayer < 2 || il < 0 || il >= nLayer) {
            throw new IllegalArgumentException(hp.arch().ggufName() + ": no lightning slope scale for layer " + il
                    + " of " + nLayer);
        }
        return (float) (1.0 - 1.0 * il / (nLayer - 1) + 1e-5);
    }
}
