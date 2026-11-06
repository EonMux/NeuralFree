package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Hugging Face RNJ-1 conversion (vLLM's {@code rnj1.py}): the Gemma 3 block with per-head Q/K norms, sandwich norms,
 * a gated tanh-GELU MLP and the {@code sqrt(hidden_size)} token scale, plus chunked local attention. Each layer named
 * {@code chunked_attention} sees its aligned chunk of {@code sliding_window / 2} positions and the chunk before it;
 * every other layer attends the whole context. The layer tensors keep the Gemma 3 checkpoint names, so the tensor
 * step is {@link HfGemmaHooks}' (the norm offset included).
 */
public final class HfRnj1Hooks {

    /** The per-layer geometry of one checkpoint. Immutable. */
    static final class Layout {
        final boolean[] chunked;
        final int chunkSize;
        final int headDim;
        final float attentionScale;
        final float ropeBase;
        final float ropeBaseSwa;
        final int contextLength;

        Layout(boolean[] chunked, int chunkSize, int headDim, float attentionScale, float ropeBase,
               float ropeBaseSwa, int contextLength) {
            this.chunked = chunked;
            this.chunkSize = chunkSize;
            this.headDim = headDim;
            this.attentionScale = attentionScale;
            this.ropeBase = ropeBase;
            this.ropeBaseSwa = ropeBaseSwa;
            this.contextLength = contextLength;
        }

        boolean anyChunked() {
            for (boolean c : chunked) {
                if (c) {
                    return true;
                }
            }
            return false;
        }
    }

    private HfRnj1Hooks() {
    }

    /** RNJ-1 has no llama.cpp converter, so it carries its own GGUF architecture name. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.RNJ1;
    }

    /**
     * Writes the RNJ-1 metadata: the generic text-model keys, the head width, the per-layer-type RoPE bases, the
     * attention scale, the final logit soft cap, and the chunk size with its per-layer switches.
     *
     * @throws GgufFormatException when the activation, the layer types, the RoPE parameters or the attention
     *                             configuration is one the graph cannot carry
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        Layout l = layout(ctx);
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.CONTEXT_LENGTH, l.contextLength);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, l.headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, l.headDim);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, l.headDim);
        kv.set(GgufKeys.ROPE_FREQ_BASE, l.ropeBase);
        kv.set(GgufKeys.ROPE_FREQ_BASE_SWA, l.ropeBaseSwa);
        kv.set(GgufKeys.ATTENTION_SCALE, l.attentionScale);
        float softcap = (float) ctx.config().findFloat(0.0, "final_logit_softcapping");
        if (softcap != 0f) {
            kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, softcap);
        }
        if (l.anyChunked()) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, l.chunkSize);
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, l.chunked);
    }

    /** The Gemma 3 tensor step: the norm offset and the out-of-vocabulary embedding rows. */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        HfGemmaHooks.tensors(ctx, hfName, src, out);
    }

    /** The Gemma 3 vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfGemmaHooks.vocab(ctx, pre);
    }

    /** The parsed geometry of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(HfRnj1Hooks.class.getName() + ".layout", Layout.class,
                () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates the RNJ-1 configuration.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    static Layout parse(HfConfig cfg, int blocks) {
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "biased attention projections are unsupported");
        }
        checkActivation(cfg, "hidden_act");
        checkActivation(cfg, "hidden_activation");
        if (cfg.find("attn_logit_softcapping").isPresent()) {
            throw invalid(cfg, "attn_logit_softcapping", "attention logit soft-capping is unsupported");
        }

        Object layerTypesRaw = cfg.find("layer_types").orElse(null);
        if (!(layerTypesRaw instanceof List)) {
            throw invalid(cfg, "layer_types", "is required and must be an array");
        }
        List<?> layerTypes = (List<?>) layerTypesRaw;
        if (layerTypes.size() < blocks) {
            throw invalid(cfg, "layer_types", "has " + layerTypes.size() + " entries, fewer than the " + blocks
                    + " layers");
        }
        boolean[] chunked = new boolean[blocks];
        int nChunked = 0;
        for (int il = 0; il < blocks; il++) {
            Object type = layerTypes.get(il);
            if ("chunked_attention".equals(type)) {
                chunked[il] = true;
                nChunked++;
            } else if (!"full_attention".equals(type)) {
                throw invalid(cfg, "layer_types", "layer " + il + " is '" + type
                        + "'; only 'full_attention' and 'chunked_attention' are supported");
            }
        }
        int chunkSize = 0;
        if (nChunked > 0) {
            int window = cfg.requireInt("sliding_window");
            if (window < 2) {
                throw invalid(cfg, "sliding_window", "must be at least 2 for one chunk of look-back, got " + window);
            }
            chunkSize = window / 2;
        }

        checkRope(cfg, cfg.rope().global(), "rope_parameters");
        Optional<HfRope.Params> chunkRope = cfg.rope().layer("chunked_attention");
        if (!chunkRope.isPresent()) {
            chunkRope = cfg.rope().layer("sliding_attention");
        }
        if (chunkRope.isPresent()) {
            checkRope(cfg, chunkRope.get(), "rope_parameters.chunked_attention");
        }
        float ropeBase = (float) cfg.rope().global().theta().orElse(1_000_000.0);
        float ropeBaseSwa = chunkRope.isPresent() ? (float) chunkRope.get().theta().orElse(ropeBase) : ropeBase;
        if (!(ropeBase > 0f) || !(ropeBaseSwa > 0f)) {
            throw invalid(cfg, "rope_parameters", "the rotary bases must be positive, got " + ropeBase + " and "
                    + ropeBaseSwa);
        }

        int nEmbd = cfg.requireInt("hidden_size");
        int nHead = cfg.requireInt("num_attention_heads");
        if (nHead <= 0 || nEmbd % nHead != 0) {
            throw invalid(cfg, "num_attention_heads", nHead + " heads do not divide hidden_size " + nEmbd);
        }
        int headDim = cfg.findInt(nEmbd / nHead, "head_dim");
        if (headDim <= 0) {
            throw invalid(cfg, "head_dim", "must be positive, got " + headDim);
        }
        double queryScale = cfg.findFloat(headDim, "query_pre_attn_scalar");
        if (!(queryScale > 0) || Double.isInfinite(queryScale)) {
            throw invalid(cfg, "query_pre_attn_scalar", "must be positive and finite, got " + queryScale);
        }
        int context = cfg.findInt(131_072, "max_position_embeddings");
        return new Layout(chunked, chunkSize, headDim, (float) (1.0 / Math.sqrt(queryScale)), ropeBase, ropeBaseSwa,
                context);
    }

    private static void checkActivation(HfConfig cfg, String key) {
        Optional<String> act = cfg.findString(key);
        if (act.isPresent() && !"gelu_pytorch_tanh".equals(act.get())) {
            throw invalid(cfg, key, "must be 'gelu_pytorch_tanh', got '" + act.get() + "'");
        }
    }

    /** Rejects a RoPE variant the NEOX rotation over the whole head cannot express. */
    private static void checkRope(HfConfig cfg, HfRope.Params params, String field) {
        Optional<String> type = params.type();
        if (type.isPresent() && !"default".equals(type.get().toLowerCase(Locale.ROOT))) {
            throw invalid(cfg, field, "scaling type '" + type.get() + "' is unsupported");
        }
        OptionalDouble factor = params.factor();
        if (factor.isPresent() && factor.getAsDouble() != 1.0) {
            throw invalid(cfg, field, "scaling factor " + factor.getAsDouble() + " is unsupported");
        }
        OptionalDouble partial = params.partialRotaryFactor();
        if (partial.isPresent() && partial.getAsDouble() != 1.0) {
            throw invalid(cfg, field, "partial rotation " + partial.getAsDouble() + " is unsupported");
        }
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
