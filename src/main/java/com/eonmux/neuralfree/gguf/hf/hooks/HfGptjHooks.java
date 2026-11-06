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

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.UnaryOperator;
import java.util.regex.Pattern;

/**
 * Hugging Face GPT-J conversion (vLLM's {@code gpt_j.py}, which has no llama.cpp counterpart): a LayerNorm decoder
 * with separate, bias-free q/k/v projections, adjacent-pair rotary over the first {@code rotary_dim} dims of each
 * head, a parallel residual and a biased GELU MLP and output head.
 */
public final class HfGptjHooks {
    private static final Set<String> ROPE_KEYS = new HashSet<>(Arrays.asList(
            "type", "rope_type", "rope_theta", "partial_rotary_factor"));
    private static final Pattern LAYER_TENSOR = Pattern.compile("blk\\.\\d+\\.(?:"
            + "attn_norm\\.(?:weight|bias)|attn_[qkv]\\.weight|attn_output\\.weight|"
            + "ffn_(?:up|down)\\.(?:weight|bias))");

    private HfGptjHooks() { }

    /** Writes GPT-J dimensions and rejects unsupported graph variants. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!"gelu_new".equals(cfg.findStringOr("gelu_new", "activation_function"))) {
            throw invalid(cfg, "activation_function", "only gelu_new is supported");
        }
        if (cfg.findBool(false, "tie_word_embeddings")) {
            throw invalid(cfg, "tie_word_embeddings", "GPT-J requires an untied output head");
        }
        if (!cfg.findBool(true, "rotary")) {
            throw invalid(cfg, "rotary", "GPT-J requires rotary attention");
        }
        validateRope(cfg);
        int width = cfg.requireInt("n_embd", "hidden_size");
        int heads = cfg.requireInt("n_head", "num_attention_heads");
        if (width <= 0 || heads <= 0 || width % heads != 0) {
            throw invalid(cfg, "n_head", "must divide the positive n_embd");
        }
        int headDim = width / heads;
        int rotary = cfg.findInt("rotary_dim").orElseGet(() -> cfg.rope().global().partialRotaryFactor().isPresent()
                ? (int) Math.round(cfg.rope().global().partialRotaryFactor().getAsDouble() * headDim)
                : 64);
        if (rotary <= 0 || rotary % 2 != 0 || rotary > headDim) {
            throw invalid(cfg, "rotary_dim", "must be positive, even, and no larger than the attention head");
        }
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, rotary);
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, Math.multiplyExact(width, 4));
        }
        if (!kv.has(GgufKeys.CONTEXT_LENGTH)) {
            kv.set(GgufKeys.CONTEXT_LENGTH, 2048);
        }
        if (!kv.has(GgufKeys.ATTENTION_LAYERNORM_EPS)) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, 1e-5);
        }
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

    /** Maps the GPT-J weights directly; the adjacent-pair rotary is applied to q and k, so no projection is permuted. */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(name)) {
            return;
        }
        String mapped = ctx.names().mapOrThrow(name);
        if (!mapped.equals("token_embd.weight") && !mapped.equals("output.weight")
                && !mapped.equals("output.bias") && !mapped.equals("output_norm.weight")
                && !mapped.equals("output_norm.bias") && !LAYER_TENSOR.matcher(mapped).matches()) {
            throw new GgufFormatException("GPT-J does not support tensor '" + name + "'");
        }
        out.add(mapped, src);
    }
}
