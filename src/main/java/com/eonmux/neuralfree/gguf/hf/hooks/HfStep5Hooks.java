package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.hooks.Step5Hooks;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hooks of vLLM's Step5 text decoder ({@code vllm/models/step5/nvidia/model.py}, {@code common/config.py} and
 * {@code nvidia/mtp.py}). The checkpoint stores separate Q/K/V, output and per-head gate projections beside the
 * sparse indexer's query, key, region-weight and head-weight projections and its RMS query / LayerNorm key norms.
 * Every RMS norm weight is centered on zero, so the converter stores {@code 1 + weight}; the indexer key LayerNorm is
 * stored unchanged. The checkpoint's {@code sparse_config} section becomes the indexer metadata and is validated
 * against the only geometry vLLM can run, and the arbitrary {@code moe_layers_enum} / {@code moe_layer_list} becomes
 * a per-block mask. NextN/MTP blocks at {@code num_hidden_layers ..} carry their fusion projections, their norms and
 * their shared head beside a decoder body whose tensors sit under {@code .mtp_block.}.
 */
public final class HfStep5Hooks {

    private static final Pattern SPEC_LAYER = Pattern.compile(
            "^(?:model\\.)?layers\\.(\\d+)\\.(?:mtp_block\\.)?(.*)$");
    private static final float SWIGLU_LIMIT = 7.0f;

    private HfStep5Hooks() {
    }

    /**
     * The Step5 text architecture. A checkpoint without the {@code sparse_config} section is a dense-only variant
     * vLLM itself refuses, and the sparse section's unsupported options are refused by name here.
     */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.STEP5;
    }

    /**
     * {@code Step5Model.__init__}'s block count: the trunk plus {@code num_nextn_predict_layers} MTP blocks, the
     * trunk alone when the load excludes them.
     */
    public static int blocks(HfLoadContext ctx) {
        int trunk = ctx.config().blockCount();
        int nextn = HfQwenHooks.includeMtp(ctx) ? ctx.config().findInt(0, "num_nextn_predict_layers") : 0;
        if (nextn > 0) {
            for (int il : moeIndices(ctx.config())) {
                if (il >= trunk) {
                    throw new GgufFormatException(ctx.config().source() + ": Step5 MTP block " + il
                            + " is a routed-expert layer; vLLM's Step5MTP supports dense draft layers only");
                }
            }
        }
        return trunk + nextn;
    }

    /**
     * The configuration values Step5 reads under llama.cpp's names: one key/value head per indexer provider group, the
     * routed expert counts, and the required {@code zero_centered}, GQA, sigmoid-router and {@code silu} switches.
     *
     * @throws GgufFormatException for a configuration vLLM cannot run
     */
    public static HfConfig fixup(HfConfig cfg) {
        Step5Sparse sparse = Step5Sparse.read(cfg);
        if (!cfg.findBool(true, "zero_centered")) {
            throw new GgufFormatException(cfg.source() + ": 'zero_centered' must be true; Step5 RMS norms are"
                    + " centered on zero");
        }
        String attn = cfg.findStringOr("GQA", "att_impl_type");
        if (!"GQA".equalsIgnoreCase(attn)) {
            throw new GgufFormatException(cfg.source() + ": 'att_impl_type' = '" + attn
                    + "' is not supported: Step5 attention is GQA");
        }
        if (cfg.findBool(false, "attention_bias", "qkv_bias")) {
            throw new GgufFormatException(cfg.source() + ": 'attention_bias' is not supported: Step5 attention has"
                    + " no input biases");
        }
        String act = cfg.findStringOr("silu", "hidden_act", "activation_function");
        if (!"silu".equalsIgnoreCase(act)) {
            throw new GgufFormatException(cfg.source() + ": 'hidden_act' = '" + act
                    + "' is not supported: Step5 uses the gated SwiGLU feed-forward");
        }
        refuseList(cfg, "rope_theta", "per-layer rope_theta lists are not supported");
        Object yarn = cfg.find("yarn_only_types").orElse(null);
        if (yarn instanceof List && !((List<?>) yarn).isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": 'yarn_only_types' is not supported");
        }
        Object other = cfg.find("attention_other_setting").orElse(null);
        if (other != null) {
            throw new GgufFormatException(cfg.source() + ": 'attention_other_setting' (per-layer head widths) is"
                    + " not supported");
        }
        Object swaHeads = cfg.find("swa_num_attention_heads").orElse(null);
        if (swaHeads instanceof Number
                && ((Number) swaHeads).intValue() != cfg.requireInt("num_attention_heads")) {
            throw new GgufFormatException(cfg.source() + ": 'swa_num_attention_heads' differing from"
                    + " 'num_attention_heads' is not supported");
        }
        Object useRope = cfg.find("use_rope_layers").orElse(null);
        if (useRope instanceof List) {
            for (Object v : (List<?>) useRope) {
                if (!truthy(v)) {
                    throw new GgufFormatException(cfg.source() + ": 'use_rope_layers' with a NoPE layer is not"
                            + " supported");
                }
            }
        }
        int nextn = cfg.findInt(0, "num_nextn_predict_layers");
        int layers = cfg.blockCount();
        int groups = cfg.findInt(0, "num_attention_groups", "num_key_value_heads");
        if (groups <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'num_attention_groups' must be positive");
        }
        if (groups != sparse.indexTpSize) {
            throw new GgufFormatException(cfg.source() + ": 'num_attention_groups' is " + groups
                    + ", but sparse_config.index_tp_size is " + sparse.indexTpSize
                    + "; Step5 requires the indexer provider groups to equal the key/value heads");
        }
        Map<String, Object> derived = new LinkedHashMap<>();
        derived.put("num_key_value_heads", (long) groups);
        if (!cfg.find("n_routed_experts").isPresent() && cfg.find("moe_num_experts").isPresent()) {
            derived.put("n_routed_experts", cfg.find("moe_num_experts").get());
        }
        if (!cfg.find("num_experts_per_tok").isPresent() && cfg.find("moe_top_k").isPresent()) {
            derived.put("num_experts_per_tok", cfg.find("moe_top_k").get());
        }
        validateMoeLayers(cfg, layers + nextn);
        return cfg.withFlat(derived);
    }

    private static void validateMoeLayers(HfConfig cfg, int total) {
        for (int il : moeIndices(cfg)) {
            if (il < 0 || il >= total) {
                throw new GgufFormatException(cfg.source() + ": 'moe_layer_list' names block " + il
                        + " outside [0, " + total + ")");
            }
        }
    }

    /** The routed-expert block indices: {@code moe_layers_enum}, else {@code moe_layer_list}, else every block but the first. */
    private static List<Integer> moeIndices(HfConfig cfg) {
        List<Integer> enumIdx = parseLayerList(cfg, "moe_layers_enum");
        List<Integer> listIdx = parseLayerList(cfg, "moe_layer_list");
        if (enumIdx != null && listIdx != null && !enumIdx.equals(listIdx)) {
            throw new GgufFormatException(cfg.source() + ": 'moe_layers_enum' and 'moe_layer_list' describe"
                    + " different layers");
        }
        List<Integer> chosen = enumIdx != null ? enumIdx : listIdx;
        if (chosen == null) {
            chosen = new ArrayList<>();
            for (int i = 1; i < cfg.blockCount(); i++) {
                chosen.add(i);
            }
        }
        return chosen;
    }

    private static List<Integer> parseLayerList(HfConfig cfg, String name) {
        Object raw = cfg.find(name).orElse(null);
        if (raw == null) {
            return null;
        }
        List<Integer> out = new ArrayList<>();
        if (raw instanceof String) {
            for (String part : ((String) raw).split(",")) {
                if (!part.trim().isEmpty()) {
                    out.add(Integer.parseInt(part.trim()));
                }
            }
        } else if (raw instanceof List) {
            for (Object v : (List<?>) raw) {
                if (!(v instanceof Number)) {
                    throw new GgufFormatException(cfg.source() + ": '" + name + "' must hold integers");
                }
                out.add(((Number) v).intValue());
            }
        } else if (raw instanceof Number) {
            out.add(((Number) raw).intValue());
        } else {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a list of layer indices");
        }
        return out;
    }

    private static void refuseList(HfConfig cfg, String name, String reason) {
        Object v = cfg.find(name).orElse(null);
        if (v instanceof List) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "': " + reason);
        }
    }

    private static boolean truthy(Object v) {
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        return v != null;
    }

    /**
     * The generic metadata plus the Step5 keys: per-block head counts and KV heads, key/value width, the partial
     * rotary width, the sliding window and its pattern, the sparse indexer geometry, the MoE counts and routing, the
     * SwiGLU clamps, the NextN block count, the FP32 residual switch and the per-block MoE mask.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int blocks = ctx.blockCount();
        Step5Sparse sparse = Step5Sparse.read(cfg);
        List<String> types = layerTypes(cfg, blocks);
        int headDim = requireHeadDim(cfg);
        int heads = cfg.requireInt("num_attention_heads");
        int kvHeads = cfg.findInt(1, "num_attention_groups", "num_key_value_heads");
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, Collections.nCopies(blocks, heads));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, Collections.nCopies(blocks, kvHeads));
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rotaryDim(cfg, headDim));
        if (cfg.findBool(false, "fp32_residual_connection")) {
            kv.set(GgufKeys.FP32_RESIDUAL_CONNECTION, true);
        }
        kv.set(GgufKeys.ZERO_CENTERED_NORMS, true);
        kv.set(GgufKeys.ATTENTION_INDEXER_HEAD_COUNT, sparse.indexerHeads);
        kv.set(GgufKeys.ATTENTION_INDEXER_KEY_LENGTH, Step5Hooks.PROXY_DIM);
        kv.set(GgufKeys.ATTENTION_INDEXER_TOP_K, sparse.topK);
        kv.set(GgufKeys.ATTENTION_INDEXER_BLOCK_SIZE, Step5Hooks.REGION_BLOCK_SIZE);
        kv.set(GgufKeys.ATTENTION_INDEXER_ROPE_DIM, sparse.indexerRopeDim);
        kv.set(GgufKeys.ATTENTION_INDEXER_GROUP_COUNT, sparse.indexTpSize);
        writeSlidingWindow(cfg, kv, types);
        writeMoe(ctx, kv, blocks);
        writeClamps(cfg, kv, blocks);
        int nextn = HfQwenHooks.includeMtp(ctx) ? cfg.findInt(0, "num_nextn_predict_layers") : 0;
        if (nextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn);
        }
    }

    /** {@code layer_types_with_mtp}, else {@code layer_types}, else full attention on every block. */
    private static List<String> layerTypes(HfConfig cfg, int blocks) {
        Object raw = cfg.find("layer_types_with_mtp").orElseGet(() -> cfg.find("layer_types").orElse(null));
        boolean sliding = cfg.find("sliding_window").isPresent();
        List<String> out = new ArrayList<>(blocks);
        for (int i = 0; i < blocks; i++) {
            Object v = raw instanceof List && i < ((List<?>) raw).size() ? ((List<?>) raw).get(i) : null;
            String type = v == null ? "full_attention" : String.valueOf(v);
            if (!"full_attention".equals(type) && !"sliding_attention".equals(type)) {
                throw new GgufFormatException(cfg.source() + ": 'layer_types[" + i + "]' = '" + type
                        + "' is not supported: Step5 uses full_attention and sliding_attention");
            }
            if ("sliding_attention".equals(type) && !sliding) {
                throw new GgufFormatException(cfg.source() + ": layer " + i + " is sliding_attention but"
                        + " 'sliding_window' is absent");
            }
            out.add(type);
        }
        return out;
    }

    private static int requireHeadDim(HfConfig cfg) {
        int headDim = cfg.findInt(0, "head_dim");
        if (headDim <= 0) {
            headDim = cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads");
        }
        if (headDim <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'head_dim' must be positive");
        }
        return headDim;
    }

    /** The partial rotary width: {@code head_dim * partial_rotary_factor}, uniform over the blocks. */
    private static int rotaryDim(HfConfig cfg, int headDim) {
        Object raw = cfg.find("partial_rotary_factors").orElse(null);
        double factor = cfg.findFloat(1.0, "partial_rotary_factor");
        if (raw instanceof List && !((List<?>) raw).isEmpty()) {
            double first = ((Number) ((List<?>) raw).get(0)).doubleValue();
            for (Object v : (List<?>) raw) {
                if (!(v instanceof Number) || Math.abs(((Number) v).doubleValue() - first) > 1e-9) {
                    throw new GgufFormatException(cfg.source() + ": non-uniform 'partial_rotary_factors' are not"
                            + " supported");
                }
            }
            factor = first;
        }
        double dim = headDim * factor;
        int rotary = (int) Math.round(dim);
        if (rotary <= 0 || rotary % 2 != 0 || rotary > headDim) {
            throw new GgufFormatException(cfg.source() + ": partial_rotary_factor " + factor + " of head_dim "
                    + headDim + " gives the invalid rotary dim " + rotary);
        }
        return rotary;
    }

    private static void writeSlidingWindow(HfConfig cfg, HfMetadataBuilder kv, List<String> types) {
        boolean any = false;
        List<Boolean> pattern = new ArrayList<>(types.size());
        for (String t : types) {
            boolean sliding = "sliding_attention".equals(t);
            pattern.add(sliding);
            any |= sliding;
        }
        if (!any) {
            return;
        }
        int window = cfg.findInt(0, "sliding_window");
        if (window <= 0) {
            throw new GgufFormatException(cfg.source() + ": sliding_attention layers need a positive"
                    + " 'sliding_window'");
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window);
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    }

    /** The MoE counts, routing and per-block mask; a dense-only checkpoint writes none. */
    private static void writeMoe(HfLoadContext ctx, HfMetadataBuilder kv, int blocks) {
        HfConfig cfg = ctx.config();
        int experts = cfg.findInt(0, "moe_num_experts", "n_routed_experts");
        List<Boolean> mask = new ArrayList<>(blocks);
        if (experts <= 0) {
            for (int i = 0; i < blocks; i++) {
                mask.add(false);
            }
            kv.set(GgufKeys.MOE_LAYER_MASK, mask);
            return;
        }
        if (!cfg.findBool(true, "use_moe_router_bias")) {
            throw new GgufFormatException(cfg.source() + ": 'use_moe_router_bias' must be true; Step5's router"
                    + " corrects its selection with a bias");
        }
        if (!cfg.findBool(true, "need_fp32_gate")) {
            throw new GgufFormatException(cfg.source() + ": 'need_fp32_gate' must be true");
        }
        String router = cfg.findStringOr("sigmoid", "moe_router_activation");
        if (!"sigmoid".equalsIgnoreCase(router)) {
            throw new GgufFormatException(cfg.source() + ": 'moe_router_activation' = '" + router
                    + "' is not supported: Step5 routes with sigmoid scores");
        }
        kv.set(GgufKeys.EXPERT_COUNT, experts);
        kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_top_k", "num_experts_per_tok"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, 1);
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, cfg.requireInt("share_expert_dim"));
        Object scale = cfg.find("moe_router_scaling_factor").orElse(null);
        if (scale instanceof Number) {
            kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, ((Number) scale).doubleValue());
        }
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_expert_weight"));
        List<Integer> chosen = parseLayerList(cfg, "moe_layers_enum");
        if (chosen == null) {
            chosen = parseLayerList(cfg, "moe_layer_list");
        }
        for (int i = 0; i < blocks; i++) {
            boolean moe = chosen != null ? chosen.contains(i)
                    : (i >= 1 && i < cfg.blockCount());
            mask.add(moe);
        }
        kv.set(GgufKeys.MOE_LAYER_MASK, mask);
    }

    /** The per-block SwiGLU clamps; vLLM's fused MoE accepts only 0 and 7.0. */
    private static void writeClamps(HfConfig cfg, HfMetadataBuilder kv, int blocks) {
        List<Double> shared = clampList(cfg, "swiglu_limits_shared", blocks);
        List<Double> routed = clampList(cfg, "swiglu_limits", blocks);
        if (shared != null) {
            kv.set(GgufKeys.SWIGLU_CLAMP_SHEXP, shared);
        }
        if (routed != null) {
            kv.set(GgufKeys.SWIGLU_CLAMP_EXP, routed);
        }
    }

    private static List<Double> clampList(HfConfig cfg, String name, int blocks) {
        Object raw = cfg.find(name).orElse(null);
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + name + "' must be a per-layer list");
        }
        List<Double> out = new ArrayList<>(blocks);
        for (int i = 0; i < blocks; i++) {
            Object v = i < ((List<?>) raw).size() ? ((List<?>) raw).get(i) : null;
            double d = v instanceof Number ? ((Number) v).doubleValue() : 0.0;
            if (d != 0.0 && d != SWIGLU_LIMIT) {
                throw new GgufFormatException(cfg.source() + ": '" + name + "[" + i + "]' = " + d
                        + " is not supported: Step5 uses 0 or 7.0");
            }
            out.add(d);
        }
        return out;
    }

    /**
     * The class filter: text-only names drop the vision tower and its projector, flat {@code layers.} names take the
     * {@code model.} prefix the MTP loader requests, and {@code shared_head.output} is the checkpoint's spelling of
     * {@code shared_head.head}.
     */
    public static final HfNameFilter STEP5_FILTER = (name, base) -> {
        if (name.startsWith("vision_model.") || name.startsWith("vit_large_projector.")
                || name.startsWith("multi_modal_projector.")) {
            return null;
        }
        String n = name;
        if (n.startsWith("layers.")) {
            n = "model." + n;
        }
        if (n.endsWith("self_attn.ssmax_s")) {
            return null;
        }
        n = n.replace("shared_head.output", "shared_head.head");
        return base.apply(n);
    };

    /**
     * {@code Step5Model.load_weights} and {@code Step5MTP.load_weights}: the checkpoint's separate projections, the
     * per-head Q/K norms, the sparse indexer tensors and both FFN shapes map onto the NF layout; every RMS norm weight
     * gains its zero-centered {@code +1}. The MTP block's {@code .mtp_block.} infix is dropped so its decoder body
     * takes the same names as a trunk block.
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        Matcher m = SPEC_LAYER.matcher(hfName);
        String base = hfName;
        if (m.matches()) {
            int bid = Integer.parseInt(m.group(1));
            String rest = m.group(2);
            if (mapLayer(ctx, hfName, bid, rest, src, out)) {
                return;
            }
            base = "model.layers." + bid + "." + rest;
        }
        RowSource data = src;
        if ((base.endsWith("norm.weight") || base.endsWith(".norm.weight"))
                && !base.endsWith("sparse_indexer_k_norm.weight")) {
            data = ElementTransforms.addScalar(src, 1.0f);
        }
        HfTensorHook.DEFAULT.apply(ctx, base, data, out);
    }

    /** Maps one block tensor; false when the name is not a Step5 block tensor. */
    private static boolean mapLayer(HfLoadContext ctx, String hfName, int bid, String rest, RowSource src,
                                    TensorSink out) {
        String mapped = layerTensor(ctx, bid, rest);
        if (mapped == null) {
            return false;
        }
        RowSource data = needsZeroCentered(rest) ? ElementTransforms.addScalar(src, 1.0f) : src;
        out.add(mapped, data);
        return true;
    }

    /** The NF name of the block tensor {@code rest} of block {@code bid}, or null when it is not a Step5 tensor. */
    private static String layerTensor(HfLoadContext ctx, int bid, String rest) {
        String name = rest;
        String prefix = "blk." + bid + ".";
        if (name.equals("input_layernorm.weight")) {
            return prefix + TensorNames.ATTN_NORM + ".weight";
        }
        if (name.equals("post_attention_layernorm.weight")) {
            return prefix + TensorNames.FFN_NORM + ".weight";
        }
        if (name.startsWith("self_attn.")) {
            return attentionTensor(ctx, bid, name.substring("self_attn.".length()));
        }
        if (name.startsWith("mlp.")) {
            String mlp = name.substring("mlp.".length());
            if (mlp.equals("gate_up_proj.weight")) {
                throw new GgufFormatException("tensor '" + name + "' is a fused gate/up projection; Step5"
                        + " checkpoints store separate gate_proj and up_proj tensors");
            }
            return mlpTensors(bid, mlp);
        }
        if (name.startsWith("share_expert.")) {
            return shareExpertTensors(bid, name.substring("share_expert.".length()));
        }
        if (name.startsWith("moe.")) {
            return moeTensor(bid, name.substring("moe.".length()));
        }
        if (name.startsWith("mtp_block.")) {
            return layerTensor(ctx, bid, name.substring("mtp_block.".length()));
        }
        return nextnTensor(bid, name);
    }

    private static String attentionTensor(HfLoadContext ctx, int bid, String name) {
        String prefix = "blk." + bid + ".";
        if (name.equals("q_proj.weight")) {
            return prefix + TensorNames.ATTN_Q + ".weight";
        }
        if (name.equals("k_proj.weight")) {
            return prefix + TensorNames.ATTN_K + ".weight";
        }
        if (name.equals("v_proj.weight")) {
            return prefix + TensorNames.ATTN_V + ".weight";
        }
        if (name.equals("o_proj.weight")) {
            return prefix + TensorNames.ATTN_OUT + ".weight";
        }
        if (name.equals("q_norm.weight")) {
            return prefix + TensorNames.ATTN_Q_NORM + ".weight";
        }
        if (name.equals("k_norm.weight")) {
            return prefix + TensorNames.ATTN_K_NORM + ".weight";
        }
        if (name.equals("g_proj.weight")) {
            return prefix + TensorNames.ATTN_GATE + ".weight";
        }
        if (name.equals("qkv_proj.weight") || name.equals("qkvg_proj.weight")) {
            throw new GgufFormatException("tensor 'self_attn." + name + "' is a fused Q/K/V projection; Step5"
                    + " checkpoints store separate q_proj, k_proj and v_proj tensors");
        }
        if (name.equals("qkv_indexer_proj.weight")) {
            throw new GgufFormatException("tensor 'self_attn.qkv_indexer_proj.weight' is a pre-fused"
                    + " Q/K/V/indexer projection, which vLLM's Step5 loader refuses");
        }
        if (name.equals("sparse_indexer_q.weight")) {
            return prefix + TensorNames.INDEXER_ATTN_Q_B + ".weight";
        }
        if (name.equals("sparse_indexer_k.weight")) {
            return prefix + TensorNames.INDEXER_ATTN_K + ".weight";
        }
        if (name.equals("sparse_indexer_z.weight")) {
            return prefix + TensorNames.INDEXER_Z + ".weight";
        }
        if (name.equals("sparse_indexer_w.weight")) {
            return prefix + TensorNames.INDEXER_PROJ + ".weight";
        }
        if (name.equals("sparse_indexer_q_norm.weight")) {
            return prefix + TensorNames.INDEXER_Q_NORM + ".weight";
        }
        if (name.equals("sparse_indexer_k_norm.weight")) {
            return prefix + TensorNames.INDEXER_K_NORM + ".weight";
        }
        if (name.equals("sparse_indexer_k_norm.bias")) {
            return prefix + TensorNames.INDEXER_K_NORM + ".bias";
        }
        return null;
    }

    private static String mlpTensors(int bid, String name) {
        String prefix = "blk." + bid + ".";
        if (name.equals("gate_proj.weight")) {
            return prefix + TensorNames.FFN_GATE + ".weight";
        }
        if (name.equals("up_proj.weight")) {
            return prefix + TensorNames.FFN_UP + ".weight";
        }
        if (name.equals("down_proj.weight")) {
            return prefix + TensorNames.FFN_DOWN + ".weight";
        }
        return null;
    }

    private static String moeTensor(int bid, String name) {
        String prefix = "blk." + bid + ".";
        if (name.equals("gate.weight")) {
            return prefix + TensorNames.FFN_GATE_INP + ".weight";
        }
        if (name.equals("router_bias") || name.equals("gate.e_score_correction_bias")) {
            return prefix + TensorNames.FFN_EXP_PROBS_B + ".bias";
        }
        if (name.startsWith("share_expert.")) {
            return shareExpertTensors(bid, name.substring("share_expert.".length()));
        }
        if (name.startsWith("experts.")) {
            name = name.substring("experts.".length());
        }
        if (name.equals("gate_proj.weight")) {
            return prefix + TensorNames.FFN_GATE_EXPS + ".weight";
        }
        if (name.equals("up_proj.weight")) {
            return prefix + TensorNames.FFN_UP_EXPS + ".weight";
        }
        if (name.equals("down_proj.weight")) {
            return prefix + TensorNames.FFN_DOWN_EXPS + ".weight";
        }
        return null;
    }

    /** A shared expert's dense SwiGLU projection under its {@code _shexp} name. */
    private static String shareExpertTensors(int bid, String name) {
        String out = mlpTensors(bid, name);
        if (out == null) {
            return null;
        }
        if (out.endsWith(TensorNames.FFN_GATE + ".weight")) {
            return out.replace(TensorNames.FFN_GATE + ".weight", TensorNames.FFN_GATE_SHEXP + ".weight");
        }
        if (out.endsWith(TensorNames.FFN_UP + ".weight")) {
            return out.replace(TensorNames.FFN_UP + ".weight", TensorNames.FFN_UP_SHEXP + ".weight");
        }
        return out.replace(TensorNames.FFN_DOWN + ".weight", TensorNames.FFN_DOWN_SHEXP + ".weight");
    }

    private static String nextnTensor(int bid, String name) {
        String prefix = "blk." + bid + ".";
        if (name.equals("eh_proj.weight")) {
            return prefix + TensorNames.NEXTN_EH_PROJ + ".weight";
        }
        if (name.equals("enorm.weight")) {
            return prefix + TensorNames.NEXTN_ENORM + ".weight";
        }
        if (name.equals("hnorm.weight")) {
            return prefix + TensorNames.NEXTN_HNORM + ".weight";
        }
        if (name.equals("shared_head.norm.weight")) {
            return prefix + TensorNames.NEXTN_SHARED_HEAD_NORM + ".weight";
        }
        if (name.equals("shared_head.head.weight")) {
            return prefix + TensorNames.NEXTN_SHARED_HEAD_HEAD + ".weight";
        }
        return null;
    }

    /** True for the RMS norm weights Step5 centers on zero (the indexer key LayerNorm is not one). */
    private static boolean needsZeroCentered(String rest) {
        if (!rest.endsWith("norm.weight")) {
            return false;
        }
        return !rest.equals("sparse_indexer_k_norm.weight");
    }

    /**
     * The validated {@code sparse_config} section: vLLM's {@code get_step5_sparse_config} rules. Immutable.
     */
    static final class Step5Sparse {
        final int indexerHeads;
        final int indexerRopeDim;
        final int indexTpSize;
        final int topK;

        private Step5Sparse(int indexerHeads, int indexerRopeDim, int indexTpSize, int topK) {
            this.indexerHeads = indexerHeads;
            this.indexerRopeDim = indexerRopeDim;
            this.indexTpSize = indexTpSize;
            this.topK = topK;
        }

        static Step5Sparse read(HfConfig cfg) {
            if (cfg.find("step5_sparse_config").isPresent() || cfg.find("step3p5_sparse_config").isPresent()) {
                throw new GgufFormatException(cfg.source() + ": Step5 reads the sparse layout from"
                        + " 'sparse_config'; 'step5_sparse_config' and 'step3p5_sparse_config' are not supported");
            }
            Object raw = cfg.find("sparse_config").orElse(null);
            if (raw == null) {
                throw new GgufFormatException(cfg.source() + ": Step5 requires a checkpoint with a 'sparse_config'"
                        + " section; dense-only checkpoints are not supported");
            }
            if (!(raw instanceof Map)) {
                throw new GgufFormatException(cfg.source() + ": 'sparse_config' must be an object");
            }
            Map<?, ?> m = (Map<?, ?>) raw;
            if (m.get("softmax_variant") != null) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.softmax_variant is not supported;"
                        + " use sparse_indexer_softmax_variant");
            }
            Object enabled = m.get("enabled");
            if (enabled != null && !(enabled instanceof Boolean)) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.enabled must be a bool");
            }
            if (Boolean.FALSE.equals(enabled)) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.enabled is false; Step5 has no"
                        + " dense-only execution");
            }
            String variant = str(m.get("sparse_indexer_softmax_variant"));
            if (variant != null && !Arrays.asList("softmax", "standard", "ssmax").contains(variant.toLowerCase())) {
                throw new GgufFormatException(cfg.source() + ": sparse_indexer_softmax_variant '" + variant
                        + "' is not supported");
            }
            fixed(m, cfg, "sparse_indexer_use_rope", true);
            fixed(m, cfg, "sparse_indexer_q_norm_type", "rmsnorm");
            fixed(m, cfg, "sparse_indexer_k_norm_type", "layernorm");
            fixed(m, cfg, "sparse_indexer_csa_z_norm_type", "none");
            fixed(m, cfg, "compression_method", "csa_block_compress");
            fixed(m, cfg, "attention_impl", "sparse_gqa");
            fixed(m, cfg, "sparse_indexer_ssmax_s_granularity", "q_head");
            int proxy = intOr(m, "proxy_dim", 256);
            if (proxy != Step5Hooks.PROXY_DIM) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.proxy_dim must be "
                        + Step5Hooks.PROXY_DIM + ", got " + proxy);
            }
            int block = intOr(m, "region_block_size", Step5Hooks.REGION_BLOCK_SIZE);
            if (block != Step5Hooks.REGION_BLOCK_SIZE) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.region_block_size must be "
                        + Step5Hooks.REGION_BLOCK_SIZE + ", got " + block);
            }
            int topK = intOr(m, "topk", 512);
            if (topK < 1 || topK > Step5Hooks.TOPK_MAX) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.topk must be in [1, "
                        + Step5Hooks.TOPK_MAX + "], got " + topK);
            }
            int kHeads = intOr(m, "sparse_indexer_num_k_heads", 1);
            if (kHeads != 1) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.sparse_indexer_num_k_heads must be 1,"
                        + " got " + kHeads);
            }
            int groups = m.get("index_tp_size") != null ? intOr(m, "index_tp_size", 0)
                    : intOr(m, "num_provider_groups", 4);
            if (groups <= 0) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.index_tp_size must be positive");
            }
            int heads = intOr(m, "sparse_indexer_num_heads", 16);
            if (heads <= 0 || heads % groups != 0 || !(heads / groups == 2 || heads / groups == 4)) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.sparse_indexer_num_heads " + heads
                        + " over index_tp_size " + groups + " must be 2 or 4 query heads per group");
            }
            int rope = intOr(m, "sparse_indexer_rope_dim", 32);
            if (rope <= 0 || rope > proxy || rope % 2 != 0) {
                throw new GgufFormatException(cfg.source() + ": sparse_config.sparse_indexer_rope_dim must be"
                        + " positive, even and no larger than proxy_dim, got " + rope);
            }
            Object apply = m.get("apply_to_layer_types");
            if (apply != null) {
                List<String> types = new ArrayList<>();
                if (apply instanceof List) {
                    for (Object t : (List<?>) apply) {
                        types.add(String.valueOf(t).trim().toLowerCase());
                    }
                } else {
                    types.add(String.valueOf(apply).trim().toLowerCase());
                }
                if (!types.equals(Collections.singletonList("full_attention"))) {
                    throw new GgufFormatException(cfg.source() + ": sparse_config.apply_to_layer_types must be"
                            + " ['full_attention'], got " + types);
                }
            }
            return new Step5Sparse(heads, rope, groups, topK);
        }

        private static void fixed(Map<?, ?> m, HfConfig cfg, String name, Object expected) {
            Object actual = m.get(name);
            if (actual == null || "".equals(actual)) {
                return;
            }
            if (expected instanceof Boolean) {
                if (!(actual instanceof Boolean)) {
                    throw new GgufFormatException(cfg.source() + ": sparse_config." + name + " must be a bool, got "
                            + actual);
                }
                if (!actual.equals(expected)) {
                    throw new GgufFormatException(cfg.source() + ": sparse_config." + name + " must be " + expected
                            + ", got " + actual);
                }
                return;
            }
            String normalized = String.valueOf(actual).trim().toLowerCase();
            if (!normalized.equals(expected)) {
                throw new GgufFormatException(cfg.source() + ": sparse_config." + name + " must be " + expected
                        + ", got " + actual);
            }
        }

        private static int intOr(Map<?, ?> m, String name, int dflt) {
            Object v = m.get(name);
            if (v == null || "".equals(v)) {
                return dflt;
            }
            if (!(v instanceof Number)) {
                throw new GgufFormatException("sparse_config." + name + " must be an integer");
            }
            return ((Number) v).intValue();
        }

        private static String str(Object v) {
            return v instanceof String ? (String) v : null;
        }
    }
}
