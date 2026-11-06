package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
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
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Hugging Face OLMo Hybrid conversion (vLLM's {@code olmo_hybrid.py}): OLMo2 post-norm attention layers alternate
 * with Gated DeltaNet linear-attention layers. The checkpoint keeps separate Q/K/V projections and Q/K/V convolutions
 * for the linear attention; the NeuralFree layout fuses them the way vLLM does ({@code in_proj_qkvg} without the
 * output gate, {@code conv1d} over {@code [q | k | v]}), converts {@code A_log} to {@code -exp(A_log)}, and records
 * the delta-rule update-rate multiplier and output-norm epsilon the Hugging Face layer hard-codes.
 *
 * <p>Released checkpoints carry {@code rope_parameters: null} (or an object without {@code rope_theta}) and rotate
 * nothing; the layout then writes an all-false {@code attention.rope_pattern} and a zero rotary width. The vocabulary
 * is the family's GPT-2 byte-level BPE.
 */
public final class HfOlmoHybridHooks {

    /** The delta-rule output norm epsilon of HF's {@code OlmoHybridGatedDeltaNet} (FLA's 1e-5). */
    static final float DELTA_NORM_EPS = 1e-5f;
    /** {@code linear_allow_neg_eigval} doubles the sigmoid update rate. */
    static final float NEG_EIGVAL_BETA_SCALE = 2f;
    /** The full-attention stride of the default {@code layer_types} (configuration_olmo_hybrid.py post_init). */
    private static final int DEFAULT_FULL_ATTENTION_STRIDE = 4;

    private static final String LAYERS = "model.layers.";
    private static final String LINEAR = ".linear_attn.";
    private static final String QKV_STATE = HfOlmoHybridHooks.class.getName() + ".qkv";
    private static final String CONV_STATE = HfOlmoHybridHooks.class.getName() + ".conv";

    private HfOlmoHybridHooks() {
    }

    // ------------------------------------------------------------------ metadata

    /**
     * Writes the OLMo Hybrid dimensions and refuses the graph variants the layout cannot carry: a biased attention,
     * a non-silu activation, a disabled output gate, a non-default RoPE scaling or partial rotary width, and an
     * unknown {@code layer_types} entry.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!"silu".equals(cfg.findStringOr("silu", "hidden_act"))) {
            throw invalid(cfg, "hidden_act", "only silu is supported");
        }
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "the layout carries no Q/K/V projection bias");
        }
        if (!cfg.findBool(true, "linear_use_gate")) {
            throw invalid(cfg, "linear_use_gate", "the linear attention needs its output gate");
        }
        int width = cfg.requireInt("hidden_size");
        int heads = cfg.requireInt("num_attention_heads");
        if (width <= 0 || heads <= 0 || width % heads != 0) {
            throw invalid(cfg, "num_attention_heads", "must divide the positive hidden_size");
        }
        int headDim = cfg.findInt("head_dim").orElse(width / heads);
        if (headDim <= 0) {
            throw invalid(cfg, "head_dim", "must be positive");
        }
        int nKv = cfg.findInt(heads, "num_key_value_heads");
        int nK = cfg.requireInt("linear_num_key_heads");
        int nV = cfg.requireInt("linear_num_value_heads");
        int headK = cfg.requireInt("linear_key_head_dim");
        int headV = cfg.requireInt("linear_value_head_dim");
        int convKernel = cfg.findInt(4, "linear_conv_kernel_dim");
        if (nK <= 0 || nV <= 0 || nV % nK != 0) {
            throw invalid(cfg, "linear_num_value_heads", "must be a positive multiple of linear_num_key_heads");
        }
        if (headK <= 0 || headV <= 0) {
            throw invalid(cfg, "linear_key_head_dim", "linear key and value head dimensions must be positive");
        }
        if (convKernel <= 1) {
            throw invalid(cfg, "linear_conv_kernel_dim", "must be greater than 1");
        }
        List<Boolean> recurrent = recurrentLayers(cfg, ctx.blockCount());
        // Checked before the generic writer: a scaled rotary embedding has no layout here, and its required
        // original_max_position_embeddings would otherwise be reported first.
        boolean rope = attentionRope(cfg);

        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, kvHeadCounts(recurrent, nKv));
        kv.set(GgufKeys.ATTENTION_RECURRENT_LAYERS, recurrent);
        kv.set(GgufKeys.SSM_CONV_KERNEL, convKernel);
        kv.set(GgufKeys.SSM_STATE_SIZE, headK);
        kv.set(GgufKeys.SSM_GROUP_COUNT, nK);
        kv.set(GgufKeys.SSM_TIME_STEP_RANK, nV);
        long inner = (long) headV * nV;
        if (inner > 0xFFFF_FFFFL) {
            throw invalid(cfg, "linear_value_head_dim", "the linear-attention inner size does not fit UINT32");
        }
        kv.set(GgufKeys.SSM_INNER_SIZE, inner);
        kv.set(GgufKeys.SSM_NORM_EPS, DELTA_NORM_EPS);
        kv.set(GgufKeys.SSM_BETA_SCALE,
                cfg.findBool(true, "linear_allow_neg_eigval") ? NEG_EIGVAL_BETA_SCALE : 1f);
        if (!rope) {
            kv.set(GgufKeys.ROPE_DIMENSION_COUNT, 0);
            List<Boolean> none = new ArrayList<>(ctx.blockCount());
            for (int i = 0; i < ctx.blockCount(); i++) {
                none.add(false);
            }
            kv.set(GgufKeys.ATTENTION_ROPE_PATTERN, none);
        }
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            throw invalid(cfg, "intermediate_size", "the feed-forward width is required");
        }
    }

    /**
     * True when the attention layers rotate: the checkpoint sets {@code rope_parameters} to an object holding a
     * {@code rope_theta}. Only the default full-head rotation is carried.
     *
     * @throws GgufFormatException for a scaled rotary embedding or a partial rotary width
     */
    private static boolean attentionRope(HfConfig cfg) {
        Optional<Object> raw = cfg.find("rope_parameters");
        if (!raw.isPresent() || !(raw.get() instanceof Map)
                || ((Map<?, ?>) raw.get()).get("rope_theta") == null) {
            return false;
        }
        String type = cfg.rope().global().type().orElse("default");
        if (!"default".equals(type)) {
            throw invalid(cfg, "rope_parameters.rope_type", "only the default rotary embedding is supported, got "
                    + type);
        }
        Optional<Object> partial = cfg.find("partial_rotary_factor");
        if (partial.isPresent() && partial.get() instanceof Number
                && ((Number) partial.get()).doubleValue() != 1.0) {
            throw invalid(cfg, "partial_rotary_factor", "the graph rotates the whole head");
        }
        return true;
    }

    /**
     * The recurrent flags of the trunk: {@code linear_attention} is recurrent, {@code full_attention} is not. The
     * configuration class defaults to a full-attention layer every fourth layer, and the last layer when the model is
     * shorter than one stride.
     */
    static List<Boolean> recurrentLayers(HfConfig cfg, int blockCount) {
        int nLayer = cfg.requireInt("num_hidden_layers");
        List<String> types = layerTypes(cfg, nLayer);
        if (blockCount < nLayer) {
            throw invalid(cfg, "num_hidden_layers", "the checkpoint index holds fewer blocks than the configuration");
        }
        List<Boolean> out = new ArrayList<>(blockCount);
        for (int i = 0; i < nLayer; i++) {
            String t = types.get(i);
            if ("linear_attention".equals(t)) {
                out.add(true);
            } else if ("full_attention".equals(t)) {
                out.add(false);
            } else {
                throw invalid(cfg, "layer_types", "layer " + i + " is '" + t + "'; expected linear_attention or"
                        + " full_attention");
            }
        }
        for (int i = nLayer; i < blockCount; i++) {
            out.add(false);
        }
        return out;
    }

    /** {@code layer_types}, or the configuration class's default pattern when the checkpoint omits it. */
    private static List<String> layerTypes(HfConfig cfg, int nLayer) {
        Optional<Object> raw = cfg.find("layer_types");
        if (raw.isPresent()) {
            if (!(raw.get() instanceof List)) {
                throw invalid(cfg, "layer_types", "must be an array");
            }
            List<?> values = (List<?>) raw.get();
            if (values.size() != nLayer) {
                throw invalid(cfg, "layer_types", "has " + values.size() + " entries, expected num_hidden_layers ("
                        + nLayer + ")");
            }
            List<String> out = new ArrayList<>(nLayer);
            for (Object v : values) {
                if (!(v instanceof String)) {
                    throw invalid(cfg, "layer_types", "holds a non-string entry " + v);
                }
                out.add((String) v);
            }
            return out;
        }
        List<String> out = new ArrayList<>(nLayer);
        boolean hasAttention = false;
        for (int i = 0; i < nLayer; i++) {
            if (i % DEFAULT_FULL_ATTENTION_STRIDE == DEFAULT_FULL_ATTENTION_STRIDE - 1) {
                out.add("full_attention");
                hasAttention = true;
            } else {
                out.add("linear_attention");
            }
        }
        if (!hasAttention && nLayer > 0) {
            out.set(nLayer - 1, "full_attention");
        }
        return out;
    }

    /** The per-layer KV head count: {@code nKv} on attention layers, 0 on the recurrent ones. */
    private static List<Integer> kvHeadCounts(List<Boolean> recurrent, int nKv) {
        List<Integer> out = new ArrayList<>(recurrent.size());
        for (Boolean r : recurrent) {
            out.add(r ? 0 : nKv);
        }
        return out;
    }

    // ------------------------------------------------------------------ tensors

    /**
     * Plans one checkpoint tensor under the OLMo Hybrid layout. The three linear-attention Q/K/V projections and the
     * three Q/K/V convolutions are buffered per layer and emitted fused in {@code [q | k | v]} order once all three
     * have been seen.
     *
     * @throws GgufFormatException when a tensor is not part of the layout
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
        int bid = HfLoadContext.bidOf(hfName);
        if (bid < 0 || !hfName.startsWith(LAYERS)) {
            throw new GgufFormatException("OLMo Hybrid does not support tensor '" + hfName + "'");
        }
        if (hfName.contains(LINEAR)) {
            linearTensor(ctx, hfName, bid, src, out);
            return;
        }
        out.add(TensorNames.layerName(bid, suffix(hfName)), src);
    }

    /** One linear-attention tensor of layer {@code bid}. */
    private static void linearTensor(HfLoadContext ctx, String hfName, int bid, RowSource src, TensorSink out)
            throws IOException {
        for (String part : new String[]{"q_proj", "k_proj", "v_proj"}) {
            if (hfName.endsWith(LINEAR + part + ".weight")) {
                fusion(ctx, QKV_STATE).offer(bid, part.substring(0, 1), src, out,
                        TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_QKV)));
                return;
            }
        }
        for (String part : new String[]{"q_conv1d", "k_conv1d", "v_conv1d"}) {
            if (hfName.endsWith(LINEAR + part + ".weight")) {
                fusion(ctx, CONV_STATE).offer(bid, part.substring(0, 1), ElementTransforms.squeezeConv(src), out,
                        TensorNames.layerName(bid, TensorNames.weight(TensorNames.SSM_CONV1D)));
                return;
            }
        }
        String suffix = linearSuffix(hfName);
        if (TensorNames.SSM_A_NOSCAN.equals(suffix)) {
            out.add(TensorNames.layerName(bid, suffix), ElementTransforms.negExp(src));
            return;
        }
        out.add(TensorNames.layerName(bid, suffix), src);
    }

    /** The GGUF suffix of a linear-attention tensor, mapping the HF names the layout renames. */
    private static String linearSuffix(String hfName) {
        String rest = hfName.substring(hfName.indexOf(LINEAR) + LINEAR.length());
        if ("A_log".equals(rest)) {
            return TensorNames.SSM_A_NOSCAN;
        }
        if ("dt_bias".equals(rest)) {
            return TensorNames.bias(TensorNames.SSM_DT);
        }
        String part = rest.endsWith(".weight") ? "weight" : rest.endsWith(".bias") ? "bias" : null;
        if (part == null) {
            throw new GgufFormatException("OLMo Hybrid does not support linear-attention tensor '" + hfName + "'");
        }
        String base = rest.substring(0, rest.length() - part.length() - 1);
        switch (base) {
            case "g_proj":
                return TensorNames.weight(TensorNames.ATTN_GATE);
            case "o_proj":
                return TensorNames.weight(TensorNames.SSM_OUT);
            case "b_proj":
                return TensorNames.weight(TensorNames.SSM_BETA);
            case "a_proj":
                return TensorNames.weight(TensorNames.SSM_ALPHA);
            case "o_norm":
                return TensorNames.weight(TensorNames.SSM_NORM);
            default:
                throw new GgufFormatException("OLMo Hybrid does not support linear-attention tensor '" + hfName + "'");
        }
    }

    /** The GGUF suffix of an attention or FFN tensor. */
    private static String suffix(String hfName) {
        String rest = hfName.substring(hfName.indexOf(LAYERS) + LAYERS.length());
        int dot = rest.indexOf('.');
        if (dot < 0) {
            throw new GgufFormatException("OLMo Hybrid does not support tensor '" + hfName + "'");
        }
        String base = rest.substring(dot + 1);
        if (base.endsWith(".weight")) {
            base = base.substring(0, base.length() - ".weight".length());
        } else if (base.endsWith(".bias")) {
            base = base.substring(0, base.length() - ".bias".length());
        } else {
            throw new GgufFormatException("OLMo Hybrid does not support tensor '" + hfName + "'");
        }
        switch (base) {
            case "input_layernorm": return TensorNames.weight(TensorNames.ATTN_NORM);
            case "post_attention_layernorm": return TensorNames.weight(TensorNames.ATTN_POST_NORM);
            case "post_feedforward_layernorm": return TensorNames.weight(TensorNames.FFN_POST_NORM);
            case "self_attn.q_proj": return TensorNames.weight(TensorNames.ATTN_Q);
            case "self_attn.k_proj": return TensorNames.weight(TensorNames.ATTN_K);
            case "self_attn.v_proj": return TensorNames.weight(TensorNames.ATTN_V);
            case "self_attn.o_proj": return TensorNames.weight(TensorNames.ATTN_OUT);
            case "self_attn.q_norm": return TensorNames.weight(TensorNames.ATTN_Q_NORM);
            case "self_attn.k_norm": return TensorNames.weight(TensorNames.ATTN_K_NORM);
            case "mlp.gate_proj": return TensorNames.weight(TensorNames.FFN_GATE);
            case "mlp.up_proj": return TensorNames.weight(TensorNames.FFN_UP);
            case "mlp.down_proj": return TensorNames.weight(TensorNames.FFN_DOWN);
            default: throw new GgufFormatException("OLMo Hybrid does not support tensor '" + hfName + "'");
        }
    }

    /** Checks that every buffered Q/K/V part of the load was fused. */
    public static void requireComplete(HfLoadContext ctx, TensorSink out) {
        fusion(ctx, QKV_STATE).requireComplete("linear-attention Q/K/V projections");
        fusion(ctx, CONV_STATE).requireComplete("linear-attention Q/K/V convolutions");
    }

    /** {@code GPT-2} byte-level BPE: the OLMo tokenizer. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ fusion

    private static Fusion fusion(HfLoadContext ctx, String state) {
        return ctx.state(state, Fusion.class, Fusion::new);
    }

    /** Per-layer {@code [q | k | v]} buffers of one projection family; one instance per load. */
    private static final class Fusion {
        private final Map<Integer, RowSource[]> pending = new LinkedHashMap<>();

        synchronized void offer(int bid, String part, RowSource src, TensorSink out, String target) {
            RowSource[] parts = pending.get(bid);
            if (parts == null) {
                parts = new RowSource[3];
                pending.put(bid, parts);
            }
            int index = "q".equals(part) ? 0 : "k".equals(part) ? 1 : 2;
            if (parts[index] != null) {
                throw new GgufFormatException("OLMo Hybrid: layer " + bid + " has two " + part + " projections");
            }
            parts[index] = src;
            if (parts[0] != null && parts[1] != null && parts[2] != null) {
                out.add(target, RowTransforms.concatRows(Arrays.asList(parts[0], parts[1], parts[2])));
                pending.remove(bid);
            }
        }

        synchronized void requireComplete(String what) {
            if (!pending.isEmpty()) {
                throw new GgufFormatException("OLMo Hybrid " + what + " of layer "
                        + Collections.min(pending.keySet()) + " are incomplete");
            }
        }
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
