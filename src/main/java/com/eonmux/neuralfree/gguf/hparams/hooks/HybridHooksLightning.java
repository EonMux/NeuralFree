package com.eonmux.neuralfree.gguf.hparams.hooks;

import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.HparamsSupport;
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
        b.ssmLinearSilu(true);                                                // minimax-01.cpp:293-299
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
     * Bailing MoE V2.5's lightning layers (vLLM {@code BailingMoELinearAttention}). The MLA full-attention layers
     * carry the compressed KV latent ({@code kv_lora_rank} plus the rotary dims), the KDA-free lightning layers a
     * per-head state. A layer without KV heads is a lightning layer; the others are MLA layers
     * (vLLM {@code is_linear_layer}). The lightning head size is the attention key length, the shared-expert width
     * falls back to {@code n_ff_exp * num_shared_experts} as vLLM's does, and the router normalizes its weights
     * unless the file says otherwise ({@code norm_expert_prob}). The SiLU, Q/K norm, Q scale, output-norm group
     * count and rotary width of the lightning mixer come from the {@code ssm.*} keys the converter writes.
     */
    public static void bailingV25(ModelHparamsBuilder b, GgufKv kv) {
        HybridHooks.readRmsEps(b, kv);
        HybridHooks.readMla(b, kv);
        b.nLoraQ(kv.u32(GgufKeys.ATTENTION_Q_LORA_RANK, b.nLoraQ()));
        int head = b.headKFull();
        if (head <= 0) {
            throw HybridHooks.invalid(b, "lightning attention needs a positive attention.key_length, got " + head);
        }
        b.headLinearAttn(head);
        HybridHooks.markRecurrentByKvHeads(b);
        HparamsSupport.readExpertFfLength(b, kv, true);
        b.nFfShexp(kv.u32(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, b.nFfShexp()));
        b.nExpertShared(kv.u32(GgufKeys.EXPERT_SHARED_COUNT, 1));
        b.nLayerDenseLead(kv.u32(GgufKeys.LEADING_DENSE_BLOCK_COUNT, b.nLayerDenseLead()));
        b.expertWeightsScale(kv.f32(GgufKeys.EXPERT_WEIGHTS_SCALE, 1f));
        b.expertWeightsNorm(kv.bool(GgufKeys.EXPERT_WEIGHTS_NORM, true));
        b.gating(HybridHooks.requiredGating(kv));
        b.ssmGroupNormSize(kv.u32(GgufKeys.SSM_GROUP_NORM_SIZE, 1));
        b.ssmLinearSilu(kv.bool(GgufKeys.SSM_LINEAR_SILU, false));
        b.ssmQkNorm(kv.bool(GgufKeys.SSM_QK_NORM, false));
        b.ssmQScale(kv.bool(GgufKeys.SSM_Q_SCALE, false));
        b.ssmRopeDims(kv.u32(GgufKeys.SSM_ROPE_DIMENSION_COUNT, 0));
        if (b.nFfShexp() == 0) {
            long shexp = (long) b.nFfExp(0) * Math.max(1, b.nExpertShared());
            if (shexp > Integer.MAX_VALUE) {
                throw HybridHooks.invalid(b, "shared expert FFN size " + shexp + " overflows");
            }
            b.nFfShexp((int) shexp);
        }
        validateLightningKnobs(b);
    }

    /** Checks the lightning mixer parameters the converter records. */
    private static void validateLightningKnobs(ModelHparamsBuilder b) {
        int width = b.headLinearAttn() * b.nHead(0);
        if (b.nHead(0) > 0 && (b.ssmGroupNormSize() <= 0 || width % b.ssmGroupNormSize() != 0)) {
            throw HybridHooks.invalid(b, "ssm.group_norm_size " + b.ssmGroupNormSize()
                    + " must be positive and divide the lightning output width " + width);
        }
        int rot = b.ssmRopeDims();
        if (rot < 0 || rot % 2 != 0 || rot > b.headLinearAttn()) {
            throw HybridHooks.invalid(b, "ssm.rope_dimension_count " + rot + " must be even and inside the"
                    + " lightning head " + b.headLinearAttn());
        }
    }

    /**
     * Base decay slope of every lightning head of Bailing MoE V2.5, vLLM's {@code get_slopes}
     * ({@code MiniMaxText01LinearAttention._build_slope_tensor}): {@code start * start^h} with
     * {@code start = 2^(-2^(-(log2(n_head) - 3)))} for a power-of-two head count, and the concatenation of the
     * closest power of two's slopes with every second slope of twice that count otherwise.
     *
     * @return one slope per head of layer 0
     */
    public static float[] bailingSlopes(ModelHparams hp) {
        int nHead = hp.nHead(0);
        if (nHead <= 0) {
            throw new IllegalArgumentException(hp.arch().ggufName() + ": lightning attention needs a positive head count");
        }
        return vllmSlopes(nHead);
    }

    private static float[] vllmSlopes(int n) {
        if ((n & (n - 1)) == 0) {
            return powerOfTwoSlopes(n);
        }
        int closest = Integer.highestOneBit(n);
        float[] lower = powerOfTwoSlopes(closest);
        float[] doubled = vllmSlopes(2 * closest);
        float[] out = new float[n];
        System.arraycopy(lower, 0, out, 0, closest);
        for (int i = 0; i < n - closest; i++) {
            out[closest + i] = doubled[2 * i];
        }
        return out;
    }

    private static float[] powerOfTwoSlopes(int n) {
        double log2 = Math.log(n) / Math.log(2.0);
        double start = Math.pow(2.0, -Math.pow(2.0, -(log2 - 3.0)));
        float[] slopes = new float[n];
        for (int h = 0; h < n; h++) {
            slopes[h] = (float) (start * Math.pow(start, h));
        }
        return slopes;
    }

    /**
     * Per-layer factor of Bailing MoE V2.5's base slopes: {@code 1 - il / (num_hidden_layers - 1) + 1e-5}, or
     * {@code 1 + 1e-5} for a single-block model (vLLM {@code BailingMoELinearAttention.__init__}).
     *
     * @throws IllegalArgumentException when {@code il} is not a trunk layer
     */
    public static float bailingSlopeScale(ModelHparams hp, int il) {
        int nLayer = hp.nLayer();
        if (nLayer < 1 || il < 0 || il >= nLayer) {
            throw new IllegalArgumentException(hp.arch().ggufName() + ": no lightning slope scale for layer " + il
                    + " of " + nLayer);
        }
        if (nLayer == 1) {
            return (float) (1.0 + 1e-5);
        }
        return (float) (1.0 - 1.0 * il / (nLayer - 1) + 1e-5);
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
