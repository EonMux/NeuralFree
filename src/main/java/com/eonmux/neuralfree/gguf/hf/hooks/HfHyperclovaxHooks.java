package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * Hugging Face HyperCLOVAX conversion (vLLM's {@code hyperclovax.py}): a Llama decoder with the MuP embedding,
 * residual, attention and logit multipliers and the peri-layer post norm of each sub-layer output.
 */
public final class HfHyperclovaxHooks {
    private static final Set<String> ROPE_KEYS = new HashSet<>(Arrays.asList(
            "type", "rope_type", "rope_theta", "partial_rotary_factor"));
    private static final Pattern LAYER_TENSOR = Pattern.compile("blk\\.\\d+\\.(?:"
            + "attn_norm\\.weight|post_attention_norm\\.weight|ffn_norm\\.weight|post_ffw_norm\\.weight|"
            + "attn_[qkvo]+\\.(?:weight|bias)|attn_output\\.(?:weight|bias)|"
            + "ffn_(?:gate|up|down)\\.(?:weight|bias))");
    private static final float DEFAULT_RMS_EPS = 1e-6f;
    private static final String LAYERS = "model.layers.";

    private HfHyperclovaxHooks() {
    }

    /** Writes the HyperCLOVAX dimensions, the MuP scales and refuses the graph variants the layout cannot carry. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!"silu".equals(cfg.findStringOr("silu", "hidden_act"))) {
            throw invalid(cfg, "hidden_act", "only silu is supported");
        }
        if (!cfg.findBool(true, "use_post_norm")) {
            throw invalid(cfg, "use_post_norm", "the post norm after each sub-layer is required");
        }
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "the layout carries no Q/K/V projection bias");
        }
        validateRope(cfg);
        int width = cfg.requireInt("hidden_size");
        int heads = cfg.requireInt("num_attention_heads");
        if (width <= 0 || heads <= 0 || width % heads != 0) {
            throw invalid(cfg, "num_attention_heads", "must divide the positive hidden_size");
        }
        int headDim = cfg.findInt("head_dim").orElse(width / heads);
        if (headDim <= 0) {
            throw invalid(cfg, "head_dim", "must be positive");
        }
        HfGenericMetadata.apply(ctx, kv);
        HfLlamaHooks.llamaKv(ctx, kv, true);
        double attention = cfg.findFloat(Math.pow(headDim, -0.5), "attention_multiplier");
        if (!(attention > 0.0)) {
            throw invalid(cfg, "attention_multiplier", "must be positive");
        }
        kv.set(GgufKeys.ATTENTION_SCALE, attention);
        kv.set(GgufKeys.EMBEDDING_SCALE, multiplier(cfg, 1.0f, "embedding_multiplier"));
        kv.set(GgufKeys.RESIDUAL_SCALE, multiplier(cfg, 1.0f, "residual_multiplier"));
        // vLLM multiplies the logits by logits_scaling and, when the configuration carries one, by logit_scale.
        float logits = multiplier(cfg, 1.0f, "logits_scaling");
        kv.set(GgufKeys.LOGIT_SCALE, logits * multiplier(cfg, 1.0f, "logit_scale"));
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, Math.multiplyExact(width, 4));
        }
        if (!kv.has(GgufKeys.CONTEXT_LENGTH)) {
            kv.set(GgufKeys.CONTEXT_LENGTH, 2048);
        }
        if (!kv.has(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS)) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, DEFAULT_RMS_EPS);
        }
    }

    /** A MuP multiplier; an absent or null value means {@code dflt} (the checkpoint's neutral value). */
    private static float multiplier(HfConfig cfg, float dflt, String name) {
        OptionalDouble v = cfg.findFloat(name);
        if (!v.isPresent()) {
            return dflt;
        }
        float d = (float) v.getAsDouble();
        if (!(d > 0f) || Float.isInfinite(d)) {
            throw invalid(cfg, name, "must be a positive finite number");
        }
        return d;
    }

    private static void validateRope(HfConfig cfg) {
        if (cfg.rope().nested()) {
            throw invalid(cfg, "rope_parameters", "per-layer parameters are unsupported");
        }
        String type = cfg.rope().global().type().orElse("default");
        if (!"default".equals(type)) {
            throw invalid(cfg, "rope_parameters.rope_type", "unsupported value " + type);
        }
        for (String key : cfg.rope().global().raw().keySet()) {
            if (!ROPE_KEYS.contains(key)) {
                throw invalid(cfg, "rope_parameters." + key, "unsupported parameter");
            }
        }
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }

    /** Discards the causal-mask buffers stored by older checkpoints. */
    public static String filter(String name, UnaryOperator<String> base) {
        if (name.endsWith(".attn.bias") || name.endsWith(".attn.masked_bias")) {
            return null;
        }
        return base.apply(name);
    }

    /** Maps the HyperCLOVAX weights directly; the NEOX rotary needs no projection permutation. */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        out.add(ggufName(ctx, hfName), src);
    }

    /**
     * The GGUF name of {@code hfName}.
     *
     * @throws GgufFormatException when the checkpoint carries a tensor the layout does not define
     */
    static String ggufName(HfLoadContext ctx, String hfName) {
        if ("model.embed_tokens.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.TOKEN_EMBD);
        }
        if ("model.norm.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.OUTPUT_NORM);
        }
        if ("lm_head.weight".equals(hfName)) {
            return TensorNames.weight(TensorNames.OUTPUT);
        }
        int bid = HfLoadContext.bidOf(hfName);
        String suffix = layerSuffix(hfName);
        if (bid < 0 || suffix == null) {
            throw new GgufFormatException("HyperCLOVAX does not support tensor '" + hfName + "'");
        }
        String mapped = TensorNames.layerName(bid, suffix);
        if (!LAYER_TENSOR.matcher(mapped).matches()) {
            throw new GgufFormatException("HyperCLOVAX does not support tensor '" + hfName + "'");
        }
        return mapped;
    }

    /** The GGUF suffix of a block tensor, or null for a name outside the block layout. */
    private static String layerSuffix(String hfName) {
        if (!hfName.startsWith(LAYERS)) {
            return null;
        }
        int start = LAYERS.length();
        int dot = hfName.indexOf('.', start);
        if (dot <= start) {
            return null;
        }
        for (int i = start; i < dot; i++) {
            if (!Character.isDigit(hfName.charAt(i))) {
                return null;
            }
        }
        String rest = hfName.substring(dot + 1);
        String part = rest.endsWith(".weight") ? "weight" : rest.endsWith(".bias") ? "bias" : null;
        if (part == null) {
            return null;
        }
        String base = rest.substring(0, rest.length() - part.length() - 1);
        String tensor = tensorOf(base);
        return tensor == null ? null : tensor + "." + part;
    }

    private static String tensorOf(String base) {
        switch (base) {
            case "input_layernorm": return TensorNames.ATTN_NORM;
            case "post_attention_layernorm": return TensorNames.FFN_NORM;
            case "post_norm1": return TensorNames.ATTN_POST_NORM;
            case "post_norm2": return TensorNames.FFN_POST_NORM;
            case "self_attn.q_proj": return TensorNames.ATTN_Q;
            case "self_attn.k_proj": return TensorNames.ATTN_K;
            case "self_attn.v_proj": return TensorNames.ATTN_V;
            case "self_attn.o_proj": return TensorNames.ATTN_OUT;
            case "mlp.gate_proj": return TensorNames.FFN_GATE;
            case "mlp.up_proj": return TensorNames.FFN_UP;
            case "mlp.down_proj": return TensorNames.FFN_DOWN;
            default: return null;
        }
    }
}
