package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face conversion of Bailing MoE V2.5 (vLLM {@code bailing_moe_linear.py}), a hybrid of lightning-attention
 * layers and MLA full-attention layers. The layers alternate by {@code layer_group_size}: a layer whose index is not
 * the last of its group runs lightning attention, the group's last layer runs MLA. The routed experts are either
 * dense (a leading {@code first_k_dense_replace} layers) or MoE with optional shared experts.
 *
 * <p>Lightning layers carry vLLM's fused {@code query_key_value} in block {@code [q | k | v]} order, which the
 * lightning kernel reads head-interleaved ({@code [h][q | k | v]}), so the rows are regrouped. The per-head Q/K
 * norms (when {@code use_qk_norm}) and the gated output norm {@code g_norm} are carried as they are. MLA layers keep
 * the DeepSeek-style projections: the fused {@code fused_qkv_a_proj} splits into {@code attn_q_a} and
 * {@code attn_kv_a_mqa}, and the {@code kv_b_proj} splits into the absorbed {@code attn_k_b}/{@code attn_v_b}.
 * Routed experts stack per layer, the shared expert maps by its own name, and the router bias gains its
 * {@code .bias} tail. MTP tensors are dropped: vLLM serves the trunk only.
 *
 * <p>Refused by name: biased projections, MTP layers asked for explicitly, an unknown score function, the
 * score-function/bias mismatches vLLM asserts, grouped top-k without the correction bias (where vLLM scores groups
 * by their largest expert and NeuralFree by their two largest), a {@code use_mla_nope} checkpoint and a rotary
 * scaling other than default, linear or yarn. Vocabulary: GPT-2 BPE.
 */
public final class HfBailingV25Hooks {

    /** {@code experts.<x>.<part>} parts, stacked in vLLM's checkpoint order. */
    private static final List<String> EXPERT_PARTS =
            Collections.unmodifiableList(java.util.Arrays.asList("down_proj", "gate_proj", "up_proj"));

    private static final String EXPERTS_STATE = HfBailingV25Hooks.class.getName() + ".experts";
    private static final Pattern LAYER = Pattern.compile("^(?:model\\.)?layers\\.(\\d+)\\.(.*)$");
    private static final Pattern EXPERT = Pattern.compile(
            "^(?:model\\.)?layers\\.(\\d+)\\.mlp\\.experts\\.(\\d{1,9})\\.([a-z0-9_]+)\\.weight$");

    private HfBailingV25Hooks() {
    }

    /** A resolver to {@link Arch#BAILINGMOE_V25}. */
    public static HfArchResolver arch() {
        return HfArchResolver.of(Arch.BAILINGMOE_V25);
    }

    /**
     * The filter: MTP layers ({@code model.mtp.*} and layers past {@code num_hidden_layers}) and the non-persistent
     * rotary buffers are dropped, since the served model is the trunk only.
     *
     * @throws GgufFormatException when the configuration has no {@code num_hidden_layers}
     */
    public static HfNameFilter filter(HfLoadContext ctx) {
        int nLayer = ctx.config().requireInt("num_hidden_layers");
        return (name, base) -> {
            String t = base.apply(name);
            if (t == null) {
                return null;
            }
            if (t.startsWith("model.mtp") || t.startsWith("mtp.") || t.endsWith(".rotary_emb.inv_freq")) {
                return null;
            }
            Matcher m = LAYER.matcher(t);
            if (m.matches() && Integer.parseInt(m.group(1)) >= nLayer) {
                return null;
            }
            return t;
        };
    }

    // ------------------------------------------------------------------ metadata

    /**
     * The dimensions and routing of the trunk. The lightning head is {@code head_dim} (the linear-attention head
     * size, which is also the MLA key length source in vLLM), the rotary width of the lightning layers is
     * {@code head_dim * partial_rotary_factor}, and the MLA rotary width is the full {@code qk_rope_head_dim}.
     * The shared experts use {@code moe_shared_expert_intermediate_size} (else {@code moe_intermediate_size}) times
     * {@code num_shared_experts}, as vLLM's {@code BailingMoeV25} does.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        rejectUnsupported(cfg);
        if (ctx.options().includeMtp() && cfg.findInt(0, "num_nextn_predict_layers") > 0) {
            throw invalid(cfg, "num_nextn_predict_layers", "the served Bailing MoE V2.5 graph is the trunk only;"
                    + " MTP layers are dropped");
        }
        int nLayer = cfg.requireInt("num_hidden_layers");
        int hidden = cfg.requireInt("hidden_size");
        int heads = cfg.requireInt("num_attention_heads");
        int headDim = headDim(cfg, hidden, heads);
        int nope = positive(cfg, cfg.requireInt("qk_nope_head_dim"), "qk_nope_head_dim");
        int rope = positive(cfg, cfg.requireInt("qk_rope_head_dim"), "qk_rope_head_dim");
        int vHead = positive(cfg, cfg.requireInt("v_head_dim"), "v_head_dim");
        int kvLora = positive(cfg, cfg.requireInt("kv_lora_rank"), "kv_lora_rank");
        int group = cfg.findInt(1, "layer_group_size");
        int nExperts = cfg.requireInt("num_experts", "n_routed_experts");
        int topK = positive(cfg, cfg.requireInt("num_experts_per_tok"), "num_experts_per_tok");
        if (topK > nExperts) {
            throw invalid(cfg, "num_experts_per_tok", "must not exceed num_experts (" + nExperts + ")");
        }
        int nShared = cfg.findInt(1, "num_shared_experts");
        if (nShared < 0) {
            throw invalid(cfg, "num_shared_experts", "must not be negative");
        }
        int moeFf = positive(cfg, cfg.requireInt("moe_intermediate_size"), "moe_intermediate_size");
        long sharedFf = (long) cfg.findInt(moeFf, "moe_shared_expert_intermediate_size") * Math.max(1, nShared);
        if (sharedFf > Integer.MAX_VALUE) {
            throw invalid(cfg, "moe_shared_expert_intermediate_size", "overflows");
        }
        int firstK = cfg.requireInt("first_k_dense_replace");
        if (firstK < 0 || firstK > nLayer) {
            throw invalid(cfg, "first_k_dense_replace", "must be inside [0, num_hidden_layers] (" + nLayer + ")");
        }
        GatingFunc gating = gating(cfg);
        int[] groups = expertGroups(cfg, nExperts);
        if (groups[0] > 0 && !cfg.findBool(false, "moe_router_enable_expert_bias")) {
            throw invalid(cfg, "n_group", "grouped top-k needs the correction bias: without it vLLM scores each"
                    + " group by its largest expert while NeuralFree sums its two largest");
        }
        int linearRope = linearRotaryDims(cfg, headDim);
        List<Boolean> recurrent = recurrentLayers(nLayer, group);

        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvHeadCounts(recurrent));
        kv.set(GgufKeys.ATTENTION_RECURRENT_LAYERS, recurrent);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH_MLA, nope + rope);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_MLA, vHead);
        kv.set(GgufKeys.ATTENTION_KV_LORA_RANK, kvLora);
        OptionalInt qLora = cfg.findInt("q_lora_rank");
        if (qLora.isPresent()) {
            kv.set(GgufKeys.ATTENTION_Q_LORA_RANK, positive(cfg, qLora.getAsInt(), "q_lora_rank"));
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rope);
        kv.set(GgufKeys.SSM_ROPE_DIMENSION_COUNT, linearRope);
        kv.set(GgufKeys.SSM_GROUP_NORM_SIZE, groupNormSize(cfg, headDim * heads));
        kv.set(GgufKeys.SSM_LINEAR_SILU, cfg.findBool(false, "use_linear_silu", "linear_silu"));
        kv.set(GgufKeys.SSM_QK_NORM, cfg.findBool(false, "use_qk_norm"));
        kv.set(GgufKeys.SSM_Q_SCALE, true);
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, moeFf);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, (int) sharedFf);
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, nShared);
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, firstK);
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, (float) cfg.findFloat(1.0, "routed_scaling_factor"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(true, "norm_topk_prob"));
        kv.set(GgufKeys.EXPERT_GATING_FUNC, gating.ggufCode());
        if (groups[0] > 0) {
            kv.set(GgufKeys.EXPERT_GROUP_COUNT, groups[0]);
            kv.set(GgufKeys.EXPERT_GROUP_USED_COUNT, groups[1]);
        }
    }

    /**
     * Refuses the graph variants the layout cannot carry, each naming its field: biased projections, a
     * {@code use_mla_nope} checkpoint and a rotary scaling other than default, linear or yarn.
     */
    private static void rejectUnsupported(HfConfig cfg) {
        if (cfg.findBool(false, "use_bias") || cfg.findBool(false, "use_qkv_bias")) {
            throw invalid(cfg, "use_bias", "the layout carries no Q/K/V, attention-output or FFN bias");
        }
        if (cfg.findBool(false, "use_mla_nope")) {
            throw invalid(cfg, "use_mla_nope", "the MLA layers rotate qk_rope_head_dim and the layout needs the"
                    + " absorbed rope split");
        }
        String type = cfg.rope().global().type().orElse("default");
        if (!"default".equals(type) && !"linear".equals(type) && !"yarn".equals(type)) {
            throw invalid(cfg, "rope_parameters.rope_type", "only default, linear and yarn rotary scaling are"
                    + " supported, got " + type);
        }
    }

    /**
     * The vLLM router's score function. An absent key means softmax (vLLM's default), which also accepts the
     * correction bias for expert selection, as its {@code topk_softmax} kernel does; an explicit softmax refuses
     * the bias and an explicit sigmoid requires it (vLLM's {@code BailingMoeV25} assertion).
     */
    private static GatingFunc gating(HfConfig cfg) {
        Optional<String> explicit = cfg.findString("score_function", "scoring_func");
        boolean bias = cfg.findBool(false, "moe_router_enable_expert_bias");
        String score = explicit.orElse("softmax");
        switch (score) {
            case "softmax":
                if (explicit.isPresent() && bias) {
                    throw invalid(cfg, "moe_router_enable_expert_bias", "an explicit softmax router takes no"
                            + " correction bias (vLLM's BailingMoeV25 assertion)");
                }
                return GatingFunc.SOFTMAX;
            case "sigmoid":
                if (!bias) {
                    throw invalid(cfg, "moe_router_enable_expert_bias", "a sigmoid router needs its correction bias"
                            + " (vLLM's BailingMoeV25 assertion)");
                }
                return GatingFunc.SIGMOID;
            default:
                throw invalid(cfg, "score_function", "must be softmax or sigmoid, got " + score);
        }
    }

    /**
     * The grouped top-k counts: {@code {n_group, topk_group}} when the configuration sets both, {@code {0, 0}}
     * otherwise (vLLM uses grouped top-k only when both are present).
     */
    private static int[] expertGroups(HfConfig cfg, int nExperts) {
        OptionalInt nGroup = cfg.findInt("n_group");
        OptionalInt topkGroup = cfg.findInt("topk_group");
        if (nGroup.isPresent() != topkGroup.isPresent()) {
            throw invalid(cfg, "n_group", "and topk_group must be given together for grouped top-k routing");
        }
        if (!nGroup.isPresent()) {
            return new int[]{0, 0};
        }
        int g = nGroup.getAsInt();
        int used = topkGroup.getAsInt();
        if (g <= 0 || nExperts % g != 0) {
            throw invalid(cfg, "n_group", "must be a positive divisor of num_experts (" + nExperts + "), got " + g);
        }
        if (used <= 0 || used > g) {
            throw invalid(cfg, "topk_group", "must be inside [1, n_group] (" + g + "), got " + used);
        }
        return new int[]{g, used};
    }

    /** The lightning output-norm group count: {@code group_norm_size}, which must divide the inner width. */
    private static int groupNormSize(HfConfig cfg, int inner) {
        int groups = cfg.findInt(1, "group_norm_size");
        if (groups <= 0 || inner % groups != 0) {
            throw invalid(cfg, "group_norm_size", "must be positive and divide head_dim * num_attention_heads ("
                    + inner + "), got " + groups);
        }
        return groups;
    }

    /**
     * The rotary width of the lightning layers: {@code head_dim * partial_rotary_factor}, which must be even and
     * inside the head, or 0 when {@code linear_rope} is off.
     */
    private static int linearRotaryDims(HfConfig cfg, int headDim) {
        if (!cfg.findBool(true, "linear_rope")) {
            return 0;
        }
        OptionalDouble top = cfg.findFloat("partial_rotary_factor");
        double partial = top.isPresent() ? top.getAsDouble()
                : cfg.rope().global().partialRotaryFactor().orElse(1.0);
        double dims = headDim * partial;
        int rot = (int) dims;
        if (partial <= 0 || partial > 1 || rot != dims || rot % 2 != 0 || rot > headDim) {
            throw invalid(cfg, "partial_rotary_factor", "gives " + dims + " rotary dims in a head of " + headDim
                    + "; it must be positive, at most 1, and produce an even whole number of dims");
        }
        return rot;
    }

    /** {@code head_dim} when the configuration carries it, else {@code hidden_size // num_attention_heads}. */
    private static int headDim(HfConfig cfg, int hidden, int heads) {
        if (heads <= 0) {
            throw invalid(cfg, "num_attention_heads", "must be positive, got " + heads);
        }
        OptionalInt headDim = cfg.findInt("head_dim");
        if (headDim.isPresent()) {
            return positive(cfg, headDim.getAsInt(), "head_dim");
        }
        if (hidden <= 0 || hidden % heads != 0) {
            throw invalid(cfg, "num_attention_heads", "must divide the positive hidden_size (" + hidden + ")");
        }
        return hidden / heads;
    }

    /**
     * vLLM's {@code is_linear_layer}: a layer with index {@code i} runs lightning attention when
     * {@code layer_group_size > 0 && (i + 1) % layer_group_size != 0}, MLA otherwise.
     */
    static List<Boolean> recurrentLayers(int nLayer, int group) {
        List<Boolean> out = new ArrayList<>(nLayer);
        for (int i = 0; i < nLayer; i++) {
            out.add(group > 0 && (i + 1) % group != 0);
        }
        return out;
    }

    /** The per-layer KV head count: the MLA layers keep one KV head, the lightning layers none. */
    private static List<Integer> kvHeadCounts(List<Boolean> recurrent) {
        List<Integer> out = new ArrayList<>(recurrent.size());
        for (Boolean r : recurrent) {
            out.add(r ? 0 : 1);
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * Plans one checkpoint tensor.
     *
     * @throws GgufFormatException when the tensor is not part of the layout or its shape does not fit
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if ("model.embed_tokens.weight".equals(hfName)) {
            out.add(TensorNames.weight(TensorNames.TOKEN_EMBD), src);
            return;
        }
        if ("model.norm.weight".equals(hfName)) {
            out.add(TensorNames.weight(TensorNames.OUTPUT_NORM), src);
            return;
        }
        if ("lm_head.weight".equals(hfName)) {
            out.add(TensorNames.weight(TensorNames.OUTPUT), src);
            return;
        }
        Matcher m = LAYER.matcher(hfName);
        if (!m.matches()) {
            throw new GgufFormatException("Bailing MoE V2.5 does not support tensor '" + hfName + "'");
        }
        int bid = Integer.parseInt(m.group(1));
        String rest = m.group(2);
        if (rest.startsWith("self_attn.") || rest.startsWith("attention.")) {
            attention(ctx, hfName, bid, rest.substring(rest.indexOf('.') + 1), src, out);
            return;
        }
        if (rest.startsWith("mlp.")) {
            mlp(ctx, hfName, bid, rest.substring("mlp.".length()), src, out);
            return;
        }
        if ("input_layernorm.weight".equals(rest)) {
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_NORM)), src);
            return;
        }
        if ("post_attention_layernorm.weight".equals(rest)) {
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_NORM)), src);
            return;
        }
        throw new GgufFormatException("Bailing MoE V2.5 does not support tensor '" + hfName + "'");
    }

    /** One attention tensor: the lightning layer's own projections or the MLA layer's absorbed projections. */
    private static void attention(HfLoadContext ctx, String hfName, int bid, String part, RowSource src,
                                  TensorSink out) {
        HfConfig cfg = ctx.config();
        if ("query_key_value.weight".equals(part)) {
            requireRecurrent(cfg, hfName, bid);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_QKV)), fusedQkv(cfg, hfName, src));
            return;
        }
        if ("g_proj.weight".equals(part)) {
            requireRecurrent(cfg, hfName, bid);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_GATE)), src);
            return;
        }
        if ("g_norm.weight".equals(part)) {
            requireRecurrent(cfg, hfName, bid);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_NORM_2)), src);
            return;
        }
        if ("query_layernorm.weight".equals(part) || "q_norm.weight".equals(part)) {
            requireRecurrent(cfg, hfName, bid);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_NORM)), src);
            return;
        }
        if ("key_layernorm.weight".equals(part) || "k_norm.weight".equals(part)) {
            requireRecurrent(cfg, hfName, bid);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K_NORM)), src);
            return;
        }
        if ("dense.weight".equals(part) || "o_proj.weight".equals(part)) {
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_OUT)), src);
            return;
        }
        requireMla(cfg, hfName, bid);
        switch (part) {
            case "q_a_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_A)), src);
                return;
            case "q_a_layernorm.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_A_NORM)), src);
                return;
            case "q_b_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_B)), src);
                return;
            case "q_proj.weight":
                if (qLoraRank(cfg).isPresent()) {
                    throw new GgufFormatException("tensor '" + hfName + "': the layer runs the low-rank query"
                            + " (q_lora_rank), so it has fused_qkv_a_proj instead of q_proj");
                }
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q)), src);
                return;
            case "kv_a_proj_with_mqa.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_KV_A_MQA)), src);
                return;
            case "kv_a_layernorm.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_KV_A_NORM)), src);
                return;
            case "fused_qkv_a_proj.weight":
                splitQkvA(cfg, hfName, bid, src, out);
                return;
            case "kv_b_proj.weight":
                splitKvB(cfg, hfName, bid, src, out);
                return;
            default:
                throw new GgufFormatException("Bailing MoE V2.5 does not support attention tensor '" + hfName + "'");
        }
    }

    /** One MLP tensor: the dense FFN, the router with its bias, the routed experts or the shared expert. */
    private static void mlp(HfLoadContext ctx, String hfName, int bid, String part, RowSource src, TensorSink out) {
        HfConfig cfg = ctx.config();
        switch (part) {
            case "gate.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE_INP)), src);
                return;
            case "gate.expert_bias":
            case "gate.e_score_correction_bias":
            case "e_score_correction_bias":
            case "experts.expert_bias":
            case "experts.e_score_correction_bias":
                out.add(TensorNames.layerName(bid, TensorNames.bias(TensorNames.FFN_EXP_PROBS_B)), src);
                return;
            case "shared_experts.gate_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE_SHEXP)), src);
                return;
            case "shared_experts.up_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_UP_SHEXP)), src);
                return;
            case "shared_experts.down_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_DOWN_SHEXP)), src);
                return;
            case "shared_experts.gate_up_proj.weight":
                splitGateUp(hfName, bid, true, src, out);
                return;
            case "gate_proj.weight":
            case "up_proj.weight":
            case "down_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(
                        part.startsWith("gate") ? TensorNames.FFN_GATE
                                : part.startsWith("up") ? TensorNames.FFN_UP : TensorNames.FFN_DOWN)), src);
                return;
            case "gate_up_proj.weight":
                splitGateUp(hfName, bid, false, src, out);
                return;
            default:
                break;
        }
        Matcher e = EXPERT.matcher(hfName);
        if (e.matches()) {
            if (bid < cfg.requireInt("first_k_dense_replace")) {
                throw new GgufFormatException("tensor '" + hfName + "': layer " + bid + " is a dense layer"
                        + " (first_k_dense_replace) and has no routed experts");
            }
            experts(ctx).offer(ctx, bid, Integer.parseInt(e.group(2)), e.group(3), src, out);
            return;
        }
        throw new GgufFormatException("Bailing MoE V2.5 does not support MLP tensor '" + hfName + "'");
    }

    /**
     * The fused {@code query_key_value} rows regrouped from vLLM's block {@code [q | k | v]} into the
     * head-interleaved {@code [head][q | k | v]} layout the lightning kernel reads: unit {@code 3h + t} (t = q, k,
     * v) comes from unit {@code t * n_head + h}.
     */
    private static RowSource fusedQkv(HfConfig cfg, String hfName, RowSource src) {
        int heads = cfg.requireInt("num_attention_heads");
        int headDim = headDim(cfg, cfg.requireInt("hidden_size"), heads);
        long rows = 3L * headDim * heads;
        long[] shape = src.shape();
        if (shape.length != 2 || shape[0] != rows) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape "
                    + java.util.Arrays.toString(shape) + "; query_key_value must have " + rows + " rows (3 x "
                    + heads + " heads of " + headDim + ")");
        }
        int[] order = new int[3 * heads];
        for (int u = 0; u < order.length; u++) {
            order[u] = (u % 3) * heads + u / 3;
        }
        return RowTransforms.permuteRowGroups(src, 0, headDim, order);
    }

    /** {@code fused_qkv_a_proj} ({@code [q_lora + kv_lora + rope, hidden]}) into {@code q_a} and {@code kv_a_mqa}. */
    private static void splitQkvA(HfConfig cfg, String hfName, int bid, RowSource src, TensorSink out) {
        OptionalInt qLora = qLoraRank(cfg);
        if (!qLora.isPresent()) {
            throw new GgufFormatException("tensor '" + hfName + "': fused_qkv_a_proj needs q_lora_rank");
        }
        int kvRows = cfg.requireInt("kv_lora_rank") + cfg.requireInt("qk_rope_head_dim");
        long[] shape = src.shape();
        if (shape.length != 2 || shape[0] != (long) qLora.getAsInt() + kvRows) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + java.util.Arrays.toString(shape)
                    + "; fused_qkv_a_proj must have q_lora_rank + kv_lora_rank + qk_rope_head_dim = "
                    + ((long) qLora.getAsInt() + kvRows) + " rows");
        }
        RowSource[] parts = RowTransforms.splitRows(src, qLora.getAsInt(), kvRows);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_A)), parts[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_KV_A_MQA)), parts[1]);
    }

    /** {@code kv_b_proj} ({@code [n_head * (nope + v), kv_lora]}) into the absorbed {@code attn_k_b}/{@code attn_v_b}. */
    private static void splitKvB(HfConfig cfg, String hfName, int bid, RowSource src, TensorSink out) {
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + java.util.Arrays.toString(shape)
                    + "; kv_b_proj must have rank 2");
        }
        int nHead = cfg.requireInt("num_attention_heads");
        RowSource[] kvB;
        try {
            kvB = QkvLayouts.mlaKvB(src, nHead, cfg.requireInt("qk_nope_head_dim"),
                    cfg.requireInt("v_head_dim"), Math.toIntExact(shape[1]));
        } catch (IllegalArgumentException | ArithmeticException e) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into " + nHead + " heads of "
                    + cfg.requireInt("qk_nope_head_dim") + " key and " + cfg.requireInt("v_head_dim")
                    + " value rows: " + e.getMessage(), e);
        }
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K_B)), kvB[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V_B)), kvB[1]);
    }

    /** A packed {@code gate_up_proj} ({@code [2 * ff, hidden]}) into its gate and up halves. */
    private static void splitGateUp(String hfName, int bid, boolean shared, RowSource src, TensorSink out) {
        long[] shape = src.shape();
        if (shape.length != 2 || shape[0] % 2 != 0) {
            throw new GgufFormatException("tensor '" + hfName + "' has shape " + java.util.Arrays.toString(shape)
                    + "; gate_up_proj must hold two equal halves of rows");
        }
        RowSource[] parts = RowTransforms.splitRows(src, shape[0] / 2, shape[0] / 2);
        String gate = shared ? TensorNames.FFN_GATE_SHEXP : TensorNames.FFN_GATE;
        String up = shared ? TensorNames.FFN_UP_SHEXP : TensorNames.FFN_UP;
        out.add(TensorNames.layerName(bid, TensorNames.weight(gate)), parts[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(up)), parts[1]);
    }

    private static void requireRecurrent(HfConfig cfg, String hfName, int bid) {
        if (!recurrentLayers(cfg.requireInt("num_hidden_layers"), cfg.findInt(1, "layer_group_size")).get(bid)) {
            throw new GgufFormatException("tensor '" + hfName + "': layer " + bid + " runs MLA attention"
                    + " (layer_group_size), not lightning attention");
        }
    }

    private static void requireMla(HfConfig cfg, String hfName, int bid) {
        if (recurrentLayers(cfg.requireInt("num_hidden_layers"), cfg.findInt(1, "layer_group_size")).get(bid)) {
            throw new GgufFormatException("tensor '" + hfName + "': layer " + bid + " runs lightning attention"
                    + " (layer_group_size), not MLA");
        }
    }

    private static OptionalInt qLoraRank(HfConfig cfg) {
        OptionalInt qLora = cfg.findInt("q_lora_rank");
        if (qLora.isPresent() && qLora.getAsInt() <= 0) {
            throw invalid(cfg, "q_lora_rank", "must be positive when present, got " + qLora.getAsInt());
        }
        return qLora;
    }

    /** {@code BailingMoE V2.5}'s routed experts: one stacked tensor per part and layer, in expert order. */
    private static Experts experts(HfLoadContext ctx) {
        return ctx.state(EXPERTS_STATE, Experts.class, Experts::new);
    }

    /** Every offered expert group must be complete. */
    public static void extra(HfLoadContext ctx, TensorSink out) {
        experts(ctx).requireComplete();
    }

    /** A tensor is F32 when the router or its correction bias needs full precision (vLLM's router dtype). */
    public static GGMLType forceType(String ggufName, int nDims, ElementFormat source,
                                     ModelSourceOptions.OutType outType) {
        return ggufName.contains(".ffn_gate_inp.") || ggufName.contains(".exp_probs_b.") ? GGMLType.F32 : null;
    }

    /** {@code BailingMoeV25ForCausalLM} uses the GPT-2 byte-level BPE vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ experts

    /** Per-layer expert buffers of one load. */
    private static final class Experts {
        private final Map<Integer, Group> groups = new LinkedHashMap<>();

        synchronized void offer(HfLoadContext ctx, int bid, int xid, String part, RowSource src, TensorSink out) {
            int index = EXPERT_PARTS.indexOf(part);
            if (index < 0) {
                throw new GgufFormatException("Bailing MoE V2.5 does not support expert tensor part '" + part
                        + "' of layer " + bid);
            }
            int nExperts = ctx.config().requireInt("num_experts", "n_routed_experts");
            if (xid >= nExperts) {
                throw new GgufFormatException("Bailing MoE V2.5 layer " + bid + ": expert " + xid
                        + " is outside [0, " + nExperts + ")");
            }
            Group g = groups.computeIfAbsent(bid, k -> new Group(nExperts));
            if (g.parts[index][xid] != null) {
                throw new GgufFormatException("Bailing MoE V2.5 layer " + bid + ": expert " + xid + " has two "
                        + part + " weights");
            }
            g.parts[index][xid] = src;
            g.seen++;
            if (g.seen == 3 * nExperts) {
                emit(bid, g, out);
                groups.remove(bid);
            }
        }

        /** The GGUF expert tensor of each {@link #EXPERT_PARTS} entry, in the same order. */
        private static final String[] NAMES = {TensorNames.FFN_DOWN_EXPS, TensorNames.FFN_GATE_EXPS,
                TensorNames.FFN_UP_EXPS};

        private static void emit(int bid, Group g, TensorSink out) {
            for (int i = 0; i < NAMES.length; i++) {
                out.add(TensorNames.layerName(bid, TensorNames.weight(NAMES[i])),
                        ExpertTransforms.stack(java.util.Arrays.asList(g.parts[i])));
            }
        }

        synchronized void requireComplete() {
            if (!groups.isEmpty()) {
                Integer bid = Collections.min(groups.keySet());
                throw new GgufFormatException("experts of layer " + bid + " were not all planned: expected "
                        + groups.get(bid).nExperts + " experts of " + EXPERT_PARTS);
            }
        }

        /** The members of one layer: part-major, then expert. */
        private static final class Group {
            final RowSource[][] parts = new RowSource[EXPERT_PARTS.size()][];
            final int nExperts;
            int seen;

            Group(int nExperts) {
                this.nExperts = nExperts;
                for (int i = 0; i < parts.length; i++) {
                    parts[i] = new RowSource[nExperts];
                }
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private static int positive(HfConfig cfg, int v, String field) {
        if (v <= 0) {
            throw invalid(cfg, field, "must be positive, got " + v);
        }
        return v;
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
