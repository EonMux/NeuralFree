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
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face OPT conversion (vLLM's {@code opt.py}, which has no llama.cpp counterpart): a LayerNorm decoder with
 * learned position embeddings whose sequence positions are shifted by the two padding rows, separate biased q/k/v
 * projections, an ungated ReLU MLP, an optional input and output projection when {@code word_embed_proj_dim} differs
 * from {@code hidden_size}, and block norms placed before or after the residual adds by
 * {@code do_layer_norm_before} (the 350M checkpoints are the post-LN variant).
 *
 * <p>The post-LN variant writes its two block norms as {@code attn_output_norm} and {@code layer_output_norm} so the
 * GGUF layout states where they apply; the pre-LN variant writes {@code attn_norm} and {@code ffn_norm}. The tied
 * embedding flag needs no special case: a missing {@code lm_head} ties the head to {@code token_embd}, and a stored
 * one carries the same weights.</p>
 */
public final class HfOptHooks {
    private static final Pattern MODEL_NAME = Pattern.compile("(?:model\\.)?decoder\\."
            + "(embed_tokens|embed_positions|project_in|project_out|final_layer_norm)\\.(weight|bias)");
    private static final Pattern LAYER_NAME = Pattern.compile("(?:model\\.)?decoder\\.layers\\.(\\d+)\\.(.+?)"
            + "\\.(weight|bias)");
    private static final String OUTPUT_HEAD = "lm_head.weight";
    /** Older checkpoints store the causal-mask buffers beside the attention weights. */
    private static final Pattern MASK_BUFFER = Pattern.compile("(?:model\\.)?decoder\\.layers\\.\\d+\\.self_attn\\."
            + "(?:bias|masked_bias)");

    private HfOptHooks() {
    }

    /** Writes the OPT dimensions and refuses the graph variants the layout cannot carry. */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        if (!"relu".equals(cfg.findStringOr("relu", "activation_function"))) {
            throw invalid(cfg, "activation_function", "only relu is supported");
        }
        if (!cfg.findBool(true, "layer_norm_elementwise_affine")) {
            throw invalid(cfg, "layer_norm_elementwise_affine", "OPT requires affine LayerNorms");
        }
        // do_layer_norm_before is a boolean; findBool rejects another type by field name.
        boolean postLn = !cfg.findBool(true, "do_layer_norm_before");
        boolean removeFinal = cfg.findBool(false, "_remove_final_layer_norm");
        int hidden = cfg.requireInt("hidden_size", "n_embd");
        int word = cfg.findInt(hidden, "word_embed_proj_dim");
        int heads = cfg.requireInt("num_attention_heads", "n_head");
        int ffn = cfg.requireInt("ffn_dim", "intermediate_size");
        if (hidden <= 0 || heads <= 0 || hidden % heads != 0) {
            throw invalid(cfg, "num_attention_heads", "must divide the positive hidden_size");
        }
        if (word <= 0) {
            throw invalid(cfg, "word_embed_proj_dim", "must be positive");
        }
        if (ffn <= 0) {
            throw invalid(cfg, "ffn_dim", "must be positive");
        }
        checkFinalNorm(cfg, plan, postLn, removeFinal);
        checkProjections(cfg, plan, word, hidden);
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, ffn);
        }
        if (!kv.has(GgufKeys.CONTEXT_LENGTH)) {
            kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_position_embeddings"));
        }
        if (!kv.has(GgufKeys.ATTENTION_LAYERNORM_EPS)) {
            kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS,
                    (float) cfg.findFloat(1e-5, "layer_norm_eps", "layer_norm_epsilon"));
        }
    }

    private static boolean postLayerNorm(HfConfig cfg) {
        return !cfg.findBool(true, "do_layer_norm_before");
    }

    /**
     * The decoder's final LayerNorm exists exactly when the graph is pre-LN and the checkpoint was not converted with
     * {@code _remove_final_layer_norm} (vLLM opt.py:247-262); another combination would apply a norm the graph it
     * claims to run does not have, or leave a required one missing.
     */
    private static void checkFinalNorm(HfConfig cfg, HfPlanView plan, boolean postLn, boolean removeFinal) {
        boolean expected = !postLn && !removeFinal;
        boolean stored = plan.has("output_norm.weight");
        if (expected && !stored) {
            throw invalid(cfg, "final_layer_norm", "a pre-LN checkpoint without _remove_final_layer_norm must store"
                    + " decoder.final_layer_norm");
        }
        if (!expected && stored) {
            throw invalid(cfg, "final_layer_norm", "the checkpoint stores decoder.final_layer_norm but"
                    + (postLn ? " do_layer_norm_before is false" : " _remove_final_layer_norm is true"));
        }
    }

    /**
     * {@code project_in} and {@code project_out} exist exactly when {@code word_embed_proj_dim} differs from
     * {@code hidden_size} (vLLM opt.py:224-238).
     */
    private static void checkProjections(HfConfig cfg, HfPlanView plan, int word, int hidden) {
        boolean expected = word != hidden;
        boolean stored = plan.has("project_in.weight") || plan.has("project_out.weight");
        if (expected && !(plan.has("project_in.weight") && plan.has("project_out.weight"))) {
            throw invalid(cfg, "word_embed_proj_dim", "differs from hidden_size so the checkpoint must store"
                    + " project_in and project_out");
        }
        if (!expected && stored) {
            throw invalid(cfg, "word_embed_proj_dim", "equals hidden_size so the checkpoint must not store"
                    + " project_in or project_out");
        }
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }

    /** Discards the causal-mask buffers stored by older checkpoints. */
    public static String filter(String name, UnaryOperator<String> base) {
        if (MASK_BUFFER.matcher(name).matches()) {
            return null;
        }
        return base.apply(name);
    }

    /** Maps the OPT weights to NeuralFree's {@code opt} layout. */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(name)) {
            return;
        }
        out.add(mapOrThrow(ctx.config(), name, postLayerNorm(ctx.config())), src);
    }

    /**
     * The GGUF name of the OPT tensor {@code name}.
     *
     * @throws GgufFormatException naming the tensor when the checkpoint carries one the layout does not define
     */
    private static String mapOrThrow(HfConfig cfg, String name, boolean postLn) {
        if (OUTPUT_HEAD.equals(name)) {
            return "output.weight";
        }
        Matcher model = MODEL_NAME.matcher(name);
        if (model.matches()) {
            switch (model.group(1)) {
                case "embed_tokens":
                    return "token_embd." + model.group(2);
                case "embed_positions":
                    return "position_embd." + model.group(2);
                case "project_in":
                    return "project_in." + model.group(2);
                case "project_out":
                    return "project_out." + model.group(2);
                default:
                    return "output_norm." + model.group(2);
            }
        }
        Matcher layer = LAYER_NAME.matcher(name);
        if (layer.matches()) {
            String mapped = layerSuffix(layer.group(2), postLn);
            if (mapped != null) {
                return "blk." + layer.group(1) + "." + mapped + "." + layer.group(3);
            }
        }
        throw new GgufFormatException(cfg.source() + ": OPT does not support tensor '" + name + "'");
    }

    /** The block tensor base name of an OPT layer part, or null for a part the layout does not define. */
    private static String layerSuffix(String part, boolean postLn) {
        switch (part) {
            case "self_attn.q_proj":
                return "attn_q";
            case "self_attn.k_proj":
                return "attn_k";
            case "self_attn.v_proj":
                return "attn_v";
            case "self_attn.out_proj":
                return "attn_output";
            case "self_attn_layer_norm":
                return postLn ? "attn_output_norm" : "attn_norm";
            case "final_layer_norm":
                return postLn ? "layer_output_norm" : "ffn_norm";
            case "fc1":
                return "ffn_up";
            case "fc2":
                return "ffn_down";
            default:
                return null;
        }
    }
}
