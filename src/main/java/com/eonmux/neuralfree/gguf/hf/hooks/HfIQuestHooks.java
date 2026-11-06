package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;

/**
 * Hugging Face IQuestLoopCoder conversion (vLLM's {@code iquest_loopcoder.py}). The trunk is a Llama block (RMS
 * pre-norms, GQA with RoPE, SwiGLU FFN, residual adds) run {@code loop_num} times. The first pass writes the global
 * KV caches; every later pass computes the same projections and mixes a read-only attention over the first pass's
 * cache with a sliding-window attention over its own per-pass cache, using a per-head gate
 * {@code sigmoid(gate_proj(q_head) + bias_head)}. The final RMS norm precedes the head.
 */
public final class HfIQuestHooks {

    /** vLLM's defaults for the loop configuration ({@code iquest_loopcoder.py:104-108}). */
    private static final int DEFAULT_LOOPS = 2;
    private static final int DEFAULT_WINDOW = 64;
    private static final double DEFAULT_ROPE_BASE = 10_000.0;
    private static final String LAYERS = "layers.";
    private static final String GATES = "gate_projections.";
    private static final String MODEL = "model.";
    private static final String WEIGHT = ".weight";
    private static final String LAYOUT_STATE = HfIQuestHooks.class.getName() + ".layout";

    private HfIQuestHooks() {
    }

    /** The geometry of one checkpoint. Immutable. */
    static final class Layout {
        final int blocks;
        final int loops;
        final int window;
        final int hidden;
        final int nHead;
        final int nKv;
        final int headDim;
        final int nFf;
        final int vocab;
        final int context;
        final float attentionScale;
        final float ropeBase;
        final float eps;

        Layout(int blocks, int loops, int window, int hidden, int nHead, int nKv, int headDim, int nFf, int vocab,
               int context, float attentionScale, float ropeBase, float eps) {
            this.blocks = blocks;
            this.loops = loops;
            this.window = window;
            this.hidden = hidden;
            this.nHead = nHead;
            this.nKv = nKv;
            this.headDim = headDim;
            this.nFf = nFf;
            this.vocab = vocab;
            this.context = context;
            this.attentionScale = attentionScale;
            this.ropeBase = ropeBase;
            this.eps = eps;
        }

        int qSize() {
            return nHead * headDim;
        }

        int kvSize() {
            return nKv * headDim;
        }
    }

    /** IQuestLoopCoder has no llama.cpp converter, so it carries its own GGUF architecture name. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.IQUEST_LOOPCODER;
    }

    /**
     * Writes the IQuestLoopCoder metadata: the generic text-model keys, the head geometry, the RMS epsilon, the
     * repeat geometry of the loop passes and the dual-attention window.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        Layout l = layout(ctx);
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.CONTEXT_LENGTH, l.context);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, l.nHead);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, l.nKv);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, l.headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, l.headDim);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, l.headDim);
        kv.set(GgufKeys.ROPE_FREQ_BASE, l.ropeBase);
        kv.set(GgufKeys.ATTENTION_SCALE, l.attentionScale);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, l.eps);
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, l.nFf);
        }
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_LOOPS, GGUFValueType.UINT32, l.loops);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_INPUT, GGUFValueType.UINT32, 0);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_RECURRENT, GGUFValueType.UINT32, l.blocks);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_OUTPUT, GGUFValueType.UINT32, 0);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_JUSTNORM, GGUFValueType.BOOL, Boolean.FALSE);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_DUAL_ATTENTION, GGUFValueType.BOOL, Boolean.TRUE);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_LOCAL_WINDOW, GGUFValueType.UINT32, l.window);
    }

    /** The IQuestLoopCoder vocabulary; the checkpoint carries a standard Llama-style tokenizer. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfLlamaHooks.vocab(ctx, pre);
    }

    /**
     * Maps the IQuestLoopCoder weights: the token embedding, the optional untied head and output norm, the Llama
     * block tensors and the per-layer gate projection of the mixed attention.
     *
     * @throws GgufFormatException when the checkpoint carries a tensor the graph does not define or a projection
     *                             has the wrong shape
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName.startsWith(MODEL) ? hfName.substring(MODEL.length()) : hfName;
        if ("embed_tokens.weight".equals(name)) {
            out.add(TensorNames.weight(TensorNames.TOKEN_EMBD), src);
            return;
        }
        if ("norm.weight".equals(name)) {
            out.add(TensorNames.weight(TensorNames.OUTPUT_NORM), src);
            return;
        }
        if ("lm_head.weight".equals(name)) {
            out.add(TensorNames.weight(TensorNames.OUTPUT), src);
            return;
        }
        Layout l = layout(ctx);
        int gid = gateLayer(name);
        if (gid >= 0) {
            gate(l, gid, name, src, out, hfName);
            return;
        }
        int bid = layerOf(name);
        if (bid < 0) {
            throw new GgufFormatException("IQuestLoopCoder does not support tensor '" + hfName + "'");
        }
        if (bid >= l.blocks) {
            throw new GgufFormatException("IQuestLoopCoder does not support tensor '" + hfName + "': layer " + bid
                    + " is outside the " + l.blocks + " configured layers");
        }
        String part = name.substring(layerPrefixLength(name));
        switch (part) {
            case "input_layernorm.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_NORM)), src);
                return;
            case "post_attention_layernorm.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_NORM)), src);
                return;
            case "self_attn.q_proj.weight":
                checkRows(name, src, l.qSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q)), src);
                return;
            case "self_attn.k_proj.weight":
                checkRows(name, src, l.kvSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K)), src);
                return;
            case "self_attn.v_proj.weight":
                checkRows(name, src, l.kvSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V)), src);
                return;
            case "self_attn.o_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_OUT)), src);
                return;
            case "self_attn.qkv_proj.weight": {
                RowSource[] parts = RowTransforms.splitRows(src, l.qSize(), l.kvSize(), l.kvSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q)), parts[0]);
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K)), parts[1]);
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V)), parts[2]);
                return;
            }
            case "mlp.gate_proj.weight":
                checkRows(name, src, l.nFf);
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE)), src);
                return;
            case "mlp.up_proj.weight":
                checkRows(name, src, l.nFf);
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_UP)), src);
                return;
            case "mlp.down_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_DOWN)), src);
                return;
            default:
                throw new GgufFormatException("IQuestLoopCoder does not support tensor '" + hfName + "'");
        }
    }

    /** The per-layer gate projection: a {@code [n_head][head_dim]} weight and its per-head bias. */
    private static void gate(Layout l, int bid, String name, RowSource src, TensorSink out, String hfName) {
        if (bid >= l.blocks) {
            throw new GgufFormatException("IQuestLoopCoder does not support tensor '" + hfName + "': gate layer " + bid
                    + " is outside the " + l.blocks + " configured layers");
        }
        if (name.endsWith(WEIGHT)) {
            checkRows(name, src, l.nHead);
            if (src.shape().length < 2 || src.shape()[1] != l.headDim) {
                throw new GgufFormatException("IQuestLoopCoder tensor '" + name + "' must be [" + l.nHead + ", "
                        + l.headDim + "]");
            }
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_LOOP_GATE)), src);
            return;
        }
        if (name.endsWith(".bias")) {
            checkElements(name, src, l.nHead);
            out.add(TensorNames.layerName(bid, TensorNames.bias(TensorNames.ATTN_LOOP_GATE)), src);
            return;
        }
        throw new GgufFormatException("IQuestLoopCoder does not support tensor '" + hfName + "'");
    }

    /** The config block number of a layer tensor ({@code [model.]layers.<n>.<...>}), or -1. */
    private static int layerOf(String name) {
        return leadingIndex(name, LAYERS);
    }

    /** The config block number of a gate tensor ({@code [model.]gate_projections.<n>.<...>}), or -1. */
    private static int gateLayer(String name) {
        return leadingIndex(name, GATES);
    }

    private static int leadingIndex(String name, String prefix) {
        if (!name.startsWith(prefix)) {
            return -1;
        }
        int start = prefix.length();
        int dot = name.indexOf('.', start);
        if (dot <= start) {
            return -1;
        }
        for (int i = start; i < dot; i++) {
            if (!Character.isDigit(name.charAt(i))) {
                return -1;
            }
        }
        try {
            return Integer.parseInt(name.substring(start, dot));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    private static int layerPrefixLength(String name) {
        return name.indexOf('.', LAYERS.length()) + 1;
    }

    private static void checkElements(String name, RowSource src, long expected) {
        long n = 1;
        for (long d : src.shape()) {
            n *= d;
        }
        if (n != expected) {
            throw new GgufFormatException("IQuestLoopCoder tensor '" + name + "' has " + n + " values, expected "
                    + expected);
        }
    }

    private static void checkRows(String name, RowSource src, int expected) {
        if (src.shape().length < 2 || src.shape()[0] != expected) {
            throw new GgufFormatException("IQuestLoopCoder tensor '" + name + "' has "
                    + (src.shape().length == 0 ? "no" : Long.toString(src.shape()[0])) + " rows, expected " + expected);
        }
    }

    /** The parsed geometry of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(LAYOUT_STATE, Layout.class, () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates the IQuestLoopCoder configuration.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    static Layout parse(HfConfig cfg, int blocks) {
        String act = cfg.findStringOr("silu", "hidden_act");
        if (!"silu".equals(act)) {
            throw invalid(cfg, "hidden_act", "must be 'silu', got '" + act + "'");
        }
        if (!cfg.findBool(true, "is_causal")) {
            throw invalid(cfg, "is_causal", "IQuestLoopCoder runs causal decoder attention only");
        }
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "IQuestLoopCoder requires bias-free attention projections");
        }
        if (cfg.findBool(false, "mlp_bias")) {
            throw invalid(cfg, "mlp_bias", "IQuestLoopCoder requires bias-free MLP projections");
        }
        int hidden = cfg.requireInt("hidden_size");
        int nHead = cfg.requireInt("num_attention_heads");
        int nKv = cfg.findInt(nHead, "num_key_value_heads");
        int headDim = cfg.findInt(hidden / Math.max(nHead, 1), "head_dim");
        int nFf = cfg.requireInt("intermediate_size");
        int vocab = cfg.requireInt("vocab_size");
        if (hidden <= 0 || nHead <= 0 || nKv <= 0 || headDim <= 0 || nFf <= 0 || vocab <= 0) {
            throw invalid(cfg, "hidden_size", "the model dimensions and the head width must be positive");
        }
        if (nHead % nKv != 0) {
            throw invalid(cfg, "num_key_value_heads", nKv + " does not divide " + nHead + " attention heads");
        }
        if ((long) nHead * headDim != hidden) {
            throw invalid(cfg, "head_dim", "the head width " + headDim + " x " + nHead + " heads does not cover the "
                    + hidden + " hidden values");
        }
        int loops = cfg.findInt(DEFAULT_LOOPS, "loop_num");
        if (loops < 1) {
            throw invalid(cfg, "loop_num", "must be at least 1, got " + loops);
        }
        int window = cfg.findInt(DEFAULT_WINDOW, "loop_window_size");
        if (window <= 0) {
            throw invalid(cfg, "loop_window_size", "must be positive, got " + window);
        }
        int context = cfg.findInt(8192, "max_position_embeddings");
        if (context <= 0) {
            throw invalid(cfg, "max_position_embeddings", "must be positive, got " + context);
        }
        float eps = (float) cfg.findFloat(1e-6, "rms_norm_eps");
        if (!(eps > 0f) || Float.isInfinite(eps)) {
            throw invalid(cfg, "rms_norm_eps", "must be a positive finite number, got " + eps);
        }
        return new Layout(blocks, loops, window, hidden, nHead, nKv, headDim, nFf, vocab, context,
                (float) (1.0 / Math.sqrt(headDim)), ropeBase(cfg), eps);
    }

    /** The global rotary base, refusing the scaling variants the graph does not carry. */
    private static float ropeBase(HfConfig cfg) {
        HfRope.Params p = cfg.rope().global();
        Optional<String> type = p.type();
        if (type.isPresent() && !"default".equals(type.get().toLowerCase(Locale.ROOT))) {
            throw invalid(cfg, "rope_parameters.rope_type", "scaling type '" + type.get() + "' is unsupported");
        }
        OptionalDouble factor = p.factor();
        if (factor.isPresent() && factor.getAsDouble() != 1.0) {
            throw invalid(cfg, "rope_parameters.factor", "RoPE scaling is unsupported");
        }
        OptionalDouble partial = p.partialRotaryFactor();
        if (partial.isPresent() && partial.getAsDouble() != 1.0) {
            throw invalid(cfg, "rope_parameters.partial_rotary_factor", "partial rotation is unsupported");
        }
        double theta = p.theta().orElse(DEFAULT_ROPE_BASE);
        if (!(theta > 0) || Double.isInfinite(theta)) {
            throw invalid(cfg, "rope_parameters.rope_theta", "must be positive and finite, got " + theta);
        }
        return (float) theta;
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
