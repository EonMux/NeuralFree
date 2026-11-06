package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * The AliceAI converter (vLLM {@code models/alice_ai.py}, checkpoint {@code yandex/AliceAI-Foundation-80B-A3B-Base}):
 * a pre-norm hybrid whose {@code linear_attention} layers run Kimi delta attention with the low-rank decay and output
 * gates, whose {@code full_attention} layers run Qwen3-Next gated attention, and whose FFN is a sigmoid-routed MoE with
 * a gated shared expert. Every layer is wrapped in the cross-layer attention-residual arithmetic grouped by
 * {@code block_attn_res_block_size}.
 *
 * <p>The converter writes the layout the runtime mixers already read: the KDA {@code q/k/v/b/f_a/f_b/g_a/g_b}
 * projections under the {@code ssm_*} names, the separate Q/K/V convolutions, {@code a_log_bias} as {@code -exp(.)},
 * the interleaved Q gate left in {@code attn_q}, the Gemma (zero-centered) norms shifted by one, the fused expert
 * {@code gate_up_proj}/{@code down_proj} split into the gate/up/down expert stacks, the gated shared expert, and the
 * attention-residual scores as the fused {@code <x>_res_norm * <x>_res_proj} products. The checkpoint's
 * {@code mtp.*} block is dropped, as vLLM ignores it.
 */
public final class HfAliceAiHooks {

    private static final String LAYERS = "model.layers.";
    private static final String LINEAR_PREFIX = "linear_attn.";
    private static final String WEIGHT = ".weight";
    private static final String RES_STATE = HfAliceAiHooks.class.getName() + ".resParts";
    private static final String MTP = "mtp.";

    /** The two checkpoint halves of one fused attention-residual score. */
    private enum ResKind {
        ATTN(true, "ATTN_RES_SCORE", ".attn_res_norm_weight", ".attn_res_proj.weight"),
        FFN(true, "FFN_RES_SCORE", ".mlp_res_norm_weight", ".mlp_res_proj.weight"),
        OUTPUT(false, "OUTPUT_RES_SCORE", ".attnres_final.res_norm_weight", ".attnres_final.res_proj.weight");

        private final boolean perLayer;
        private final String tensor;
        private final String normSuffix;
        private final String projSuffix;

        ResKind(boolean perLayer, String tensor, String normSuffix, String projSuffix) {
            this.perLayer = perLayer;
            this.tensor = tensor;
            this.normSuffix = normSuffix;
            this.projSuffix = projSuffix;
        }

        /** True when either half of this group ends the name. */
        boolean isHalf(String hfName) {
            return hfName.endsWith(normSuffix) || hfName.endsWith(projSuffix);
        }

        boolean isNormHalf(String hfName) {
            return hfName.endsWith(normSuffix);
        }

        String key(String hfName) {
            int bid = HfLoadContext.bidOf(hfName);
            return bid < 0 || !perLayer ? name() : name() + "." + bid;
        }

        String ggufName(HfLoadContext ctx, String hfName) {
            return ctx.names().ggufName(tensor, perLayer ? HfHookSupport.layerOf(hfName) : -1, WEIGHT);
        }
    }

    private HfAliceAiHooks() {
    }

    /** A resolver to {@link Arch#ALICE_AI}. */
    public static HfArchResolver arch() {
        return HfArchResolver.of(Arch.ALICE_AI);
    }

    // ------------------------------------------------------------------ metadata

    /**
     * Writes the AliceAI dimensions and refuses the graph variants the layout cannot carry. The per-layer attention
     * head counts differ between the KDA layers ({@code linear_num_key_heads}) and the attention layers
     * ({@code num_attention_heads}), so they are written as arrays.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        Geometry g = geometry(cfg);
        List<Boolean> recurrent = recurrentLayers(cfg, ctx.blockCount());

        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, g.rotaryDim);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, headCounts(recurrent, g));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvHeadCounts(recurrent, g));
        kv.set(GgufKeys.SSM_CONV_KERNEL, g.convKernel);
        kv.set(GgufKeys.KDA_HEAD_DIM, g.keyHeadDim);
        kv.set(GgufKeys.ATTN_RES_BLOCK_SIZE, g.attnResBlock);
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, g.expertFf);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, g.sharedFf);
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, 1);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, 0);
        // AliceAISparseMoeBlock passes neither a routed scaling factor nor a shared-expert count to FusedMoE.
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, 1.0);
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "norm_topk_prob"));
    }

    /** The validated configuration values the layout is built from. */
    private static final class Geometry {
        final int attnHeads;
        final int attnKvHeads;
        final int headDim;
        final int rotaryDim;
        final int kdaHeads;
        final int keyHeadDim;
        final int convKernel;
        final int expertFf;
        final int sharedFf;
        final int attnResBlock;

        Geometry(int attnHeads, int attnKvHeads, int headDim, int rotaryDim, int kdaHeads, int keyHeadDim,
                 int convKernel, int expertFf, int sharedFf, int attnResBlock) {
            this.attnHeads = attnHeads;
            this.attnKvHeads = attnKvHeads;
            this.headDim = headDim;
            this.rotaryDim = rotaryDim;
            this.kdaHeads = kdaHeads;
            this.keyHeadDim = keyHeadDim;
            this.convKernel = convKernel;
            this.expertFf = expertFf;
            this.sharedFf = sharedFf;
            this.attnResBlock = attnResBlock;
        }
    }

    /**
     * Reads and validates the dimensions, naming the first unsupported field.
     *
     * @throws GgufFormatException for an unsupported activation, KDA negative eigenvalues, router contract, expert
     *                             geometry or a rotary width the engine cannot apply
     */
    private static Geometry geometry(HfConfig cfg) {
        if (!"silu".equals(cfg.findStringOr("silu", "hidden_act"))) {
            throw invalid(cfg, "hidden_act", "only silu is supported");
        }
        if (cfg.findBool(false, "kda_allow_negative_eigenvalues")) {
            throw invalid(cfg, "kda_allow_negative_eigenvalues", "only the softplus decay gate (false) is supported");
        }
        if (!"sigmoid".equals(cfg.findStringOr("sigmoid", "router_score_function"))) {
            throw invalid(cfg, "router_score_function", "only sigmoid routing is supported");
        }
        if (!cfg.findBool(true, "router_bias_correction")) {
            throw invalid(cfg, "router_bias_correction", "the selection bias is required");
        }
        int attnHeads = positive(cfg, "num_attention_heads");
        int attnKvHeads = cfg.findInt(attnHeads, "num_key_value_heads");
        if (attnKvHeads <= 0 || attnHeads % attnKvHeads != 0) {
            throw invalid(cfg, "num_key_value_heads", "must divide the positive num_attention_heads");
        }
        int headDim = cfg.findInt("head_dim").orElse(Math.floorDiv(positive(cfg, "hidden_size"), attnHeads));
        if (headDim <= 0) {
            throw invalid(cfg, "head_dim", "must be positive");
        }
        int kdaHeads = positive(cfg, "linear_num_key_heads");
        int kdaValueHeads = positive(cfg, "linear_num_value_heads");
        int keyHeadDim = positive(cfg, "linear_key_head_dim");
        int valueHeadDim = positive(cfg, "linear_value_head_dim");
        if (kdaHeads != kdaValueHeads) {
            throw invalid(cfg, "linear_num_value_heads", "must equal linear_num_key_heads for the KDA mixer");
        }
        if (keyHeadDim != valueHeadDim) {
            throw invalid(cfg, "linear_value_head_dim", "must equal linear_key_head_dim for the KDA mixer");
        }
        int convKernel = cfg.findInt(4, "linear_conv_kernel_dim");
        if (convKernel < 2) {
            throw invalid(cfg, "linear_conv_kernel_dim", "must be at least 2");
        }
        int expertFf = positive(cfg, "moe_intermediate_size");
        int sharedFf = positive(cfg, "shared_expert_intermediate_size");
        int experts = positive(cfg, "num_experts");
        int used = positive(cfg, "num_experts_per_tok");
        if (used > experts) {
            throw invalid(cfg, "num_experts_per_tok", "must not exceed num_experts");
        }
        if (cfg.findInt(0, "routed_expert_hidden_size") > 0) {
            throw invalid(cfg, "routed_expert_hidden_size", "latent routed experts are not supported");
        }
        int attnResBlock = positive(cfg, "block_attn_res_block_size");
        double factor = cfg.rope().global().partialRotaryFactor().orElse(1.0);
        if (!(factor > 0.0) || factor > 1.0) {
            throw invalid(cfg, "partial_rotary_factor", "must be in (0, 1]");
        }
        int rotaryDim = (int) (headDim * factor);
        if (rotaryDim <= 0 || rotaryDim % 2 != 0) {
            throw invalid(cfg, "partial_rotary_factor", "rotary width " + rotaryDim + " must be even and positive");
        }
        return new Geometry(attnHeads, attnKvHeads, headDim, rotaryDim, kdaHeads, keyHeadDim, convKernel, expertFf,
                sharedFf, attnResBlock);
    }

    /** {@code layer_types}: {@code linear_attention} is recurrent, {@code full_attention} is not. */
    private static List<Boolean> recurrentLayers(HfConfig cfg, int blockCount) {
        int nLayer = positive(cfg, "num_hidden_layers");
        Optional<Object> raw = cfg.find("layer_types");
        List<?> values;
        if (!raw.isPresent()) {
            // AliceAIConfig's default: full attention on every fourth layer (configuration_alice_ai.py)
            List<String> defaults = new ArrayList<>(nLayer);
            for (int il = 0; il < nLayer; il++) {
                defaults.add((il + 1) % 4 == 0 ? "full_attention" : "linear_attention");
            }
            values = defaults;
        } else if (raw.get() instanceof List) {
            values = (List<?>) raw.get();
        } else {
            throw invalid(cfg, "layer_types", "must be an array of one entry per hidden layer");
        }
        if (values.size() != nLayer) {
            throw invalid(cfg, "layer_types", "has " + values.size() + " entries, expected num_hidden_layers ("
                    + nLayer + ")");
        }
        if (blockCount < nLayer) {
            throw invalid(cfg, "num_hidden_layers", "the checkpoint index holds fewer blocks than the configuration");
        }
        List<Boolean> out = new ArrayList<>(blockCount);
        for (Object v : values) {
            if ("linear_attention".equals(v)) {
                out.add(true);
            } else if ("full_attention".equals(v)) {
                out.add(false);
            } else {
                throw invalid(cfg, "layer_types", "holds '" + v + "'; expected linear_attention or full_attention");
            }
        }
        for (int i = nLayer; i < blockCount; i++) {
            out.add(false);
        }
        return out;
    }

    /** The per-layer query head count: the KDA heads on recurrent layers, the attention heads elsewhere. */
    private static List<Integer> headCounts(List<Boolean> recurrent, Geometry g) {
        List<Integer> out = new ArrayList<>(recurrent.size());
        for (Boolean r : recurrent) {
            out.add(r ? g.kdaHeads : g.attnHeads);
        }
        return out;
    }

    /** The per-layer KV head count: zero on the KDA layers, {@code num_key_value_heads} on the attention layers. */
    private static List<Integer> kvHeadCounts(List<Boolean> recurrent, Geometry g) {
        List<Integer> out = new ArrayList<>(recurrent.size());
        for (Boolean r : recurrent) {
            out.add(r ? 0 : g.attnKvHeads);
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * Plans one checkpoint tensor under the AliceAI layout.
     *
     * @throws GgufFormatException when a tensor is not part of the layout, or a per-expert checkpoint naming is used
     *                             instead of the fused {@code gate_up_proj}/{@code down_proj}
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.startsWith(MTP)) {
            out.drop(hfName, "AliceAI ignores the MTP block, as vLLM does");
            return;
        }
        if (residual(ctx, hfName, src, out)) {
            return;
        }
        switch (hfName) {
            case "model.embed_tokens.weight":
                out.add(TensorNames.weight(TensorNames.TOKEN_EMBD), src);
                return;
            case "model.norm.weight":
                out.add(TensorNames.weight(TensorNames.OUTPUT_NORM), normPlusOne(src));
                return;
            case "lm_head.weight":
                out.add(TensorNames.weight(TensorNames.OUTPUT), src);
                return;
            default:
                break;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid < 0 || !hfName.startsWith(LAYERS)) {
            throw new GgufFormatException("AliceAI does not support tensor '" + hfName + "'");
        }
        if (hfName.contains(".mlp.experts.")) {
            experts(hfName, bid, src, out);
            return;
        }
        out.add(TensorNames.layerName(bid, layerSuffix(hfName)), transform(ctx.config(), hfName, src));
    }

    /** The full GGUF suffix ({@code attn_q.weight}, {@code ssm_a}, {@code ssm_dt.bias}) of a layer tensor. */
    private static String layerSuffix(String hfName) {
        String rest = hfName.substring(hfName.indexOf(LAYERS) + LAYERS.length());
        int dot = rest.indexOf('.');
        if (dot < 0) {
            throw new GgufFormatException("AliceAI does not support tensor '" + hfName + "'");
        }
        String base = rest.substring(dot + 1);
        if (base.endsWith(WEIGHT)) {
            base = base.substring(0, base.length() - WEIGHT.length());
        }
        if (base.startsWith(LINEAR_PREFIX)) {
            return linearSuffix(base.substring(LINEAR_PREFIX.length()), hfName);
        }
        return attentionOrMoeSuffix(base, hfName);
    }

    /** The GGUF suffix of a KDA tensor ({@code base} is the name after {@code linear_attn.}). */
    private static String linearSuffix(String base, String hfName) {
        switch (base) {
            case "q_proj": return TensorNames.weight(TensorNames.ATTN_Q);
            case "k_proj": return TensorNames.weight(TensorNames.ATTN_K);
            case "v_proj": return TensorNames.weight(TensorNames.ATTN_V);
            case "o_proj": return TensorNames.weight(TensorNames.ATTN_OUT);
            case "f_a_proj": return TensorNames.weight(TensorNames.SSM_F_A);
            case "f_b_proj": return TensorNames.weight(TensorNames.SSM_F_B);
            case "b_proj": return TensorNames.weight(TensorNames.SSM_BETA);
            case "g_a_proj": return TensorNames.weight(TensorNames.SSM_G_A);
            case "g_b_proj": return TensorNames.weight(TensorNames.SSM_G_B);
            case "o_norm": return TensorNames.weight(TensorNames.SSM_NORM);
            case "a_log_bias": return TensorNames.SSM_A_NOSCAN;
            case "dt_bias": return TensorNames.bias(TensorNames.SSM_DT);
            case "q_conv1d": return TensorNames.weight(TensorNames.SSM_CONV1D_Q);
            case "k_conv1d": return TensorNames.weight(TensorNames.SSM_CONV1D_K);
            case "v_conv1d": return TensorNames.weight(TensorNames.SSM_CONV1D_V);
            default: throw new GgufFormatException("AliceAI does not support linear-attention tensor '" + hfName + "'");
        }
    }

    /** The GGUF suffix of an attention, norm or shared-expert tensor. */
    private static String attentionOrMoeSuffix(String base, String hfName) {
        switch (base) {
            case "input_layernorm": return TensorNames.weight(TensorNames.ATTN_NORM);
            case "post_attention_layernorm": return TensorNames.weight(TensorNames.FFN_NORM);
            case "self_attn.q_proj": return TensorNames.weight(TensorNames.ATTN_Q);
            case "self_attn.k_proj": return TensorNames.weight(TensorNames.ATTN_K);
            case "self_attn.v_proj": return TensorNames.weight(TensorNames.ATTN_V);
            case "self_attn.o_proj": return TensorNames.weight(TensorNames.ATTN_OUT);
            case "self_attn.q_norm": return TensorNames.weight(TensorNames.ATTN_Q_NORM);
            case "self_attn.k_norm": return TensorNames.weight(TensorNames.ATTN_K_NORM);
            case "mlp.gate": return TensorNames.weight(TensorNames.FFN_GATE_INP);
            // the name filter spells the router bias e_score_correction.bias (filters.tsv)
            case "mlp.gate.e_score_correction_bias":
            case "mlp.gate.e_score_correction.bias": return TensorNames.bias(TensorNames.FFN_EXP_PROBS_B);
            case "mlp.shared_expert_gate": return TensorNames.weight(TensorNames.FFN_GATE_INP_SHEXP);
            case "mlp.shared_expert.gate_proj": return TensorNames.weight(TensorNames.FFN_GATE_SHEXP);
            case "mlp.shared_expert.up_proj": return TensorNames.weight(TensorNames.FFN_UP_SHEXP);
            case "mlp.shared_expert.down_proj": return TensorNames.weight(TensorNames.FFN_DOWN_SHEXP);
            default: throw new GgufFormatException("AliceAI does not support tensor '" + hfName + "'");
        }
    }

    /**
     * The value transform of a layer tensor: zero-centered norms gain one, {@code a_log_bias} becomes {@code -exp(.)}
     * and the KDA convolutions are reshaped to the depthwise layout.
     */
    private static RowSource transform(HfConfig cfg, String hfName, RowSource src) {
        if (hfName.endsWith(".a_log_bias")) {
            return aLog(cfg, hfName, src);
        }
        if (hfName.contains(LINEAR_PREFIX) && hfName.endsWith("_conv1d.weight")) {
            return HfKdaHooks.convWeight(src, hfName);
        }
        if (hfName.endsWith(".input_layernorm.weight") || hfName.endsWith(".post_attention_layernorm.weight")
                || hfName.endsWith(".self_attn.q_norm.weight") || hfName.endsWith(".self_attn.k_norm.weight")) {
            return normPlusOne(src);
        }
        return src;
    }

    /**
     * {@code a_log_bias -> -exp(A_log)}: the KDA decay factor per key head. vLLM loads the same tensor into its
     * {@code A_log}, either as one value per head or as the legacy {@code (1, 1, heads, 1)} shape
     * (kimi_gdn_linear_attn.py:47-67), and its KDA kernel applies {@code -exp(A_log)}.
     *
     * @throws GgufFormatException when the tensor does not hold one value per {@code linear_num_key_heads}
     */
    private static RowSource aLog(HfConfig cfg, String hfName, RowSource src) {
        int nHead = cfg.requireInt("linear_num_key_heads");
        long[] shape = src.shape();
        if (shape.length == 4 && shape[0] == 1 && shape[1] == 1 && shape[3] == 1) {
            if (shape[2] != nHead) {
                throw aLogShape(hfName, shape, nHead);
            }
            return ElementTransforms.negExp(ElementTransforms.reshape(src, nHead));
        }
        if (shape.length != 1 || shape[0] != nHead) {
            throw aLogShape(hfName, shape, nHead);
        }
        return ElementTransforms.negExp(src);
    }

    private static GgufFormatException aLogShape(String hfName, long[] shape, int nHead) {
        return new GgufFormatException("AliceAI tensor '" + hfName + "' has shape " + java.util.Arrays.toString(shape)
                + "; a_log_bias must hold one value per linear_num_key_heads (" + nHead + ")");
    }

    /**
     * The fused routed experts: {@code gate_up_proj [E, 2F, D]} splits at dim {@code -2} into the gate and up
     * stacks, {@code down_proj [E, D, F]} is written as stored (the vLLM FusedMoE fused-checkpoint layout).
     */
    private static void experts(String hfName, int bid, RowSource src, TensorSink out) {
        if (hfName.endsWith(".mlp.experts.gate_up_proj") || hfName.endsWith(".mlp.experts.gate_up_proj.weight")) {
            long[] shape = src.shape();
            if (shape.length < 3 || shape[shape.length - 2] % 2 != 0) {
                throw new GgufFormatException("AliceAI layer " + bid + ": mlp.experts.gate_up_proj shape "
                        + java.util.Arrays.toString(shape) + " is not [experts, 2 * n_ff, n_embd]");
            }
            RowSource[] gateUp = ExpertTransforms.splitExperts(src, -2, 2);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE_EXPS)), gateUp[0]);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_UP_EXPS)), gateUp[1]);
            return;
        }
        if (hfName.endsWith(".mlp.experts.down_proj") || hfName.endsWith(".mlp.experts.down_proj.weight")) {
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_DOWN_EXPS)), src);
            return;
        }
        throw new GgufFormatException("AliceAI requires fused mlp.experts.gate_up_proj and mlp.experts.down_proj"
                + " tensors; '" + hfName + "' is not supported");
    }

    /** Whether {@code hfName} is one half of a fused attention-residual score, planning it when the pair completes. */
    private static boolean residual(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        for (ResKind kind : ResKind.values()) {
            if (!kind.isHalf(hfName)) {
                continue;
            }
            Parts parts = ctx.state(RES_STATE, Parts.class, Parts::new);
            String key = kind.key(hfName);
            Half other = parts.pending.remove(key);
            if (other == null) {
                parts.pending.put(key, new Half(src, kind.isNormHalf(hfName)));
                return true;
            }
            RowSource norm = other.norm ? other.src : src;
            RowSource proj = other.norm ? src : other.src;
            long width = norm.shape().length == 0 ? 1 : norm.shape()[norm.shape().length - 1];
            RowSource flat = ElementTransforms.reshape(proj, width);
            ctx.log("AliceAI: fused residual score " + kind.name() + " -> " + kind.ggufName(ctx, hfName));
            out.add(kind.ggufName(ctx, hfName), HfKdaHooksB.ResidualProduct.of(norm, flat));
            return true;
        }
        return false;
    }

    /** Every fused attention-residual score completed. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        Parts parts = ctx.state(RES_STATE, Parts.class, Parts::new);
        if (!parts.pending.isEmpty()) {
            throw new GgufFormatException("AliceAI: unpaired attention-residual tensors " + parts.pending.keySet()
                    + ": every residual norm needs its projection");
        }
    }

    /** The pending halves of one load's attention-residual groups. */
    private static final class Parts {
        final Map<String, Half> pending = new LinkedHashMap<>();
    }

    /** A buffered residual half. */
    private static final class Half {
        final RowSource src;
        final boolean norm;

        Half(RowSource src, boolean norm) {
            this.src = src;
            this.norm = norm;
        }
    }

    // ------------------------------------------------------------------ vocabulary

    /**
     * The AliceAI tokenizer: the {@code tokenizer.model} SentencePiece vocabulary (the checkpoint's
     * {@code LlamaTokenizer}), or the checkpoint's GPT-2 BPE files when it ships those instead.
     */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        return HfVocabWriters.canSentencePiece(in) ? HfVocabWriters.sentencePiece(in, pre)
                : HfVocabWriters.gpt2(in, pre);
    }

    // ------------------------------------------------------------------ small helpers

    private static RowSource normPlusOne(RowSource src) {
        return HfHookSupport.normPlusOne(src);
    }

    private static int positive(HfConfig cfg, String key) {
        int value = cfg.requireInt(key);
        if (value <= 0) {
            throw invalid(cfg, key, "must be positive");
        }
        return value;
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
