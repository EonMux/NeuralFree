package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Hugging Face Dory conversion (vLLM's {@code dory.py}). The normalized latent recurrent transformer runs a
 * {@code [n_input, n_input + n_recurrent)} block {@code n_recurrent_loops} times with one KV cache per pass and
 * combines the passes with nGPT justnorm. The physical blocks alternate attention-only and MLP-only layers, every
 * layer output joins the stream through a slerp residual, and the Q/K, attention-output and FFN normalizations are
 * weightless justnorms.
 *
 * <p>The learned nGPT scales are folded into the weights the graph reads, so the runtime kernels only apply the
 * normalizations: {@code sqk} becomes the {@code attn_q_norm} weight {@code (sqk * sqrt(head_dim) / init)^2},
 * {@code s_gate} scales {@code attn_gate} by {@code (s_gate * sqrt(g0) / init)^2}, {@code suv_gate}/{@code suv_up}
 * scale the FFN projections by {@code suv * sqrt(hidden)}, {@code lm_head.logit_scale} becomes the
 * {@code output_logit_scale} vector divided by the initializer range, and the slerp alphas keep their raw values
 * (the graph applies {@code |alpha * ngpt_alpha_init / init|} at run time).</p>
 */
public final class HfDoryHooks {

    /** The nGPT justnorm epsilon ({@code dory.py:31}). */
    private static final float JUSTNORM_EPS = 1e-6f;
    private static final String LAYERS = "layers.";
    private static final String BACKBONE = "backbone.";
    private static final String WEIGHT = ".weight";
    private static final String LAYOUT_STATE = HfDoryHooks.class.getName() + ".layout";
    private static final String SLIDING = "sliding_attention";
    private static final String FULL = "full_attention";
    private static final String MLP = "mlp";

    private HfDoryHooks() {
    }

    /** The geometry of one checkpoint. Immutable. */
    static final class Layout {
        final int blocks;
        final boolean[] mlp;
        final boolean[] sliding;
        final int nInput;
        final int nRecurrent;
        final int nOutput;
        final int nLoops;
        final int hidden;
        final int nHead;
        final int nKv;
        final int headDim;
        final int nFf;
        final int vocab;
        final int context;
        final int window;
        final float attentionScale;
        final float ropeBase;
        final float ropeBaseSwa;
        final float slerpAlphaScale;
        final float qkFactor;
        final float sGateFactor;
        final float suvFactor;
        final float logitScaleDivisor;

        Layout(int blocks, boolean[] mlp, boolean[] sliding, int nInput, int nRecurrent, int nOutput, int nLoops,
               int hidden, int nHead, int nKv, int headDim, int nFf, int vocab, int context, int window,
               float attentionScale, float ropeBase, float ropeBaseSwa, float slerpAlphaScale, float qkFactor,
               float sGateFactor, float suvFactor, float logitScaleDivisor) {
            this.blocks = blocks;
            this.mlp = mlp;
            this.sliding = sliding;
            this.nInput = nInput;
            this.nRecurrent = nRecurrent;
            this.nOutput = nOutput;
            this.nLoops = nLoops;
            this.hidden = hidden;
            this.nHead = nHead;
            this.nKv = nKv;
            this.headDim = headDim;
            this.nFf = nFf;
            this.vocab = vocab;
            this.context = context;
            this.window = window;
            this.attentionScale = attentionScale;
            this.ropeBase = ropeBase;
            this.ropeBaseSwa = ropeBaseSwa;
            this.slerpAlphaScale = slerpAlphaScale;
            this.qkFactor = qkFactor;
            this.sGateFactor = sGateFactor;
            this.suvFactor = suvFactor;
            this.logitScaleDivisor = logitScaleDivisor;
        }

        int qSize() {
            return nHead * headDim;
        }

        int kvSize() {
            return nKv * headDim;
        }

        boolean anySliding() {
            for (boolean s : sliding) {
                if (s) {
                    return true;
                }
            }
            return false;
        }
    }

    /** Dory has no llama.cpp converter, so it carries its own GGUF architecture name. */
    public static com.eonmux.neuralfree.gguf.arch.Arch arch(HfConfig cfg, HfTensorIndex index) {
        return com.eonmux.neuralfree.gguf.arch.Arch.DORY;
    }

    /**
     * Writes the Dory metadata: the generic text-model keys, the head geometry, the per-type RoPE bases, the sliding
     * window and its per-layer pattern, the MLP-only block pattern, the repeat geometry and the nGPT scales.
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
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT_SWA, l.headDim);
        kv.set(GgufKeys.ROPE_FREQ_BASE, l.ropeBase);
        kv.set(GgufKeys.ROPE_FREQ_BASE_SWA, l.ropeBaseSwa);
        kv.set(GgufKeys.ATTENTION_SCALE, l.attentionScale);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, JUSTNORM_EPS);
        if (!kv.has(GgufKeys.FEED_FORWARD_LENGTH)) {
            kv.set(GgufKeys.FEED_FORWARD_LENGTH, l.nFf);
        }
        if (l.anySliding()) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, l.window);
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, l.sliding);
        kv.setTyped(GgufKeys.MLP_LAYER_PATTERN, GGUFValueType.ARRAY, GGUFValueType.BOOL, l.mlp);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_LOOPS, GGUFValueType.UINT32, l.nLoops);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_INPUT, GGUFValueType.UINT32, l.nInput);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_RECURRENT, GGUFValueType.UINT32, l.nRecurrent);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_OUTPUT, GGUFValueType.UINT32, l.nOutput);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_JUSTNORM, GGUFValueType.BOOL, Boolean.TRUE);
        kv.setTyped(GgufKeys.RECURRENT_BLOCK_JUSTNORM_EPS, GGUFValueType.FLOAT32, JUSTNORM_EPS);
        kv.setTyped(GgufKeys.SLERP_ALPHA_SCALE, GGUFValueType.FLOAT32, l.slerpAlphaScale);
    }

    /** The Dory vocabulary; the checkpoint carries a standard Llama-style tokenizer. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfLlamaHooks.vocab(ctx, pre);
    }

    /**
     * Maps the Dory weights, folding the learned nGPT scales into the tensors the graph reads.
     *
     * @throws GgufFormatException when the checkpoint carries a tensor the graph does not define, or a scale vector
     *                             has the wrong width
     */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName.startsWith(BACKBONE) ? hfName.substring(BACKBONE.length()) : hfName;
        if ("embeddings.weight".equals(name)) {
            out.add(TensorNames.weight(TensorNames.TOKEN_EMBD), src);
            return;
        }
        if ("lm_head.weight".equals(name)) {
            out.add(TensorNames.weight(TensorNames.OUTPUT), src);
            return;
        }
        if ("lm_head.logit_scale".equals(name)) {
            Layout l = layout(ctx);
            checkElements(name, src, l.vocab);
            out.add(TensorNames.weight(TensorNames.OUTPUT_LOGIT_SCALE),
                    ElementTransforms.mulScalar(src, l.logitScaleDivisor));
            return;
        }
        int bid = layerOf(name);
        if (bid < 0) {
            throw new GgufFormatException("Dory does not support tensor '" + hfName + "'");
        }
        Layout l = layout(ctx);
        if (bid >= l.blocks) {
            throw new GgufFormatException("Dory does not support tensor '" + hfName + "': layer " + bid
                    + " is outside the " + l.blocks + " configured layers");
        }
        String part = name.substring(layerPrefixLength(name));
        if ("attn_alpha".equals(part)) {
            checkElements(name, src, l.hidden);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_SLERP_ALPHA)), src);
            return;
        }
        if ("mlp_alpha".equals(part)) {
            checkElements(name, src, l.hidden);
            out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_SLERP_ALPHA)), src);
            return;
        }
        if (!part.startsWith("mixer.")) {
            throw new GgufFormatException("Dory does not support tensor '" + hfName + "'");
        }
        String mixer = part.substring("mixer.".length());
        switch (mixer) {
            case "sqk": {
                float[] sqk = floats(name, src, l.qSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q_NORM)),
                        RopeTensors.fromFloats(new long[]{l.qSize()}, qkFactors(sqk, l)));
                return;
            }
            case "s_gate":
                floats(name, src, l.qSize());
                return;
            case "suv_gate":
            case "suv_up":
                floats(name, src, l.nFf);
                return;
            case "qkv_proj.weight": {
                qkv(ctx, bid, l, src, out, TensorNames.ATTN_Q, TensorNames.ATTN_K, TensorNames.ATTN_V);
                return;
            }
            case "q_proj.weight":
                checkRows(name, src, l.qSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_Q)), src);
                return;
            case "k_proj.weight":
                checkRows(name, src, l.kvSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_K)), src);
                return;
            case "v_proj.weight":
                checkRows(name, src, l.kvSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_V)), src);
                return;
            case "o_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_OUT)), src);
                return;
            case "g_proj.weight":
                checkRows(name, src, l.qSize());
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.ATTN_GATE)),
                        scaled(src, gateFactors(layerVector(ctx, bid, "s_gate", l.qSize()), l), name));
                return;
            case "gate_up_proj.weight": {
                gateUp(ctx, bid, l, src, out);
                return;
            }
            case "gate_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE)),
                        scaled(src, suvFactors(layerVector(ctx, bid, "suv_gate", l.nFf), l), name));
                return;
            case "up_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_UP)),
                        scaled(src, suvFactors(layerVector(ctx, bid, "suv_up", l.nFf), l), name));
                return;
            case "down_proj.weight":
                out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_DOWN)), src);
                return;
            default:
                throw new GgufFormatException("Dory does not support tensor '" + hfName + "'");
        }
    }

    /** Splits a packed QKV projection into its Q, K and V rows, scaling Q by the nGPT query factor. */
    private static void qkv(HfLoadContext ctx, int bid, Layout l, RowSource src, TensorSink out, String q, String k,
                            String v) {
        RowSource[] parts = RowTransforms.splitRows(src, l.qSize(), l.kvSize(), l.kvSize());
        out.add(TensorNames.layerName(bid, TensorNames.weight(q)), parts[0]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(k)), parts[1]);
        out.add(TensorNames.layerName(bid, TensorNames.weight(v)), parts[2]);
    }

    /** Splits a packed gate/up projection into the FFN gate and up rows, scaled by their nGPT factors. */
    private static void gateUp(HfLoadContext ctx, int bid, Layout l, RowSource src, TensorSink out) {
        RowSource[] parts = RowTransforms.splitRows(src, l.nFf, l.nFf);
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_GATE)),
                scaled(parts[0], suvFactors(layerVector(ctx, bid, "suv_gate", l.nFf), l), "gate_up_proj.weight"));
        out.add(TensorNames.layerName(bid, TensorNames.weight(TensorNames.FFN_UP)),
                scaled(parts[1], suvFactors(layerVector(ctx, bid, "suv_up", l.nFf), l), "gate_up_proj.weight"));
    }

    private static RowSource scaled(RowSource src, float[] factors, String name) {
        if (src.shape().length < 2 || src.shape()[0] != factors.length) {
            throw new GgufFormatException("Dory tensor '" + name + "' has " + describeRows(src) + " rows but its"
                    + " scale has " + factors.length + " values");
        }
        return ElementTransforms.mulRowsBy(src, factors);
    }

    private static String describeRows(RowSource src) {
        return src.shape().length == 0 ? "no" : Long.toString(src.shape()[0]);
    }

    /** {@code (sqk * sqrt(head_dim) / initializer_range)^2}. */
    private static float[] qkFactors(float[] sqk, Layout l) {
        float[] f = new float[sqk.length];
        for (int i = 0; i < f.length; i++) {
            float v = sqk[i] * l.qkFactor;
            f[i] = v * v;
        }
        return f;
    }

    /** {@code (s_gate * sqrt(g0) / initializer_range)^2}. */
    private static float[] gateFactors(float[] gate, Layout l) {
        float[] f = new float[gate.length];
        for (int i = 0; i < f.length; i++) {
            float v = gate[i] * l.sGateFactor;
            f[i] = v * v;
        }
        return f;
    }

    /** {@code suv * sqrt(hidden_size)}. */
    private static float[] suvFactors(float[] suv, Layout l) {
        float[] f = new float[suv.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = suv[i] * l.suvFactor;
        }
        return f;
    }

    /** The values of a model-level scale tensor, read from the checkpoint index. */
    private static float[] vector(HfLoadContext ctx, String name, int expected) {
        RowSource src = ctx.index().get(name);
        if (src == null) {
            throw new GgufFormatException("Dory tensor '" + name + "' is missing");
        }
        return floats(name, src, expected);
    }

    /** The values of a per-layer scale tensor of block {@code bid}, read from the checkpoint index. */
    private static float[] layerVector(HfLoadContext ctx, int bid, String tensor, int expected) {
        return vector(ctx, BACKBONE + LAYERS + bid + ".mixer." + tensor, expected);
    }

    /** The config block number of a layer tensor ({@code [backbone.]layers.<n>.<...>}), or -1. */
    private static int layerOf(String name) {
        if (!name.startsWith(LAYERS)) {
            return -1;
        }
        int start = LAYERS.length();
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

    /** The values of a 1-D scale tensor, checked against the width it must have. */
    private static float[] floats(String name, RowSource src, int expected) {
        checkElements(name, src, expected);
        try {
            return HfHookSupport.floats(src, name);
        } catch (IOException e) {
            throw new GgufFormatException("Dory tensor '" + name + "' cannot be read: " + e.getMessage());
        }
    }

    private static void checkElements(String name, RowSource src, int expected) {
        long n = elements(src);
        if (n != expected) {
            throw new GgufFormatException("Dory tensor '" + name + "' has " + n + " values, expected " + expected);
        }
    }

    private static long elements(RowSource src) {
        long n = 1;
        for (long d : src.shape()) {
            n *= d;
        }
        return n;
    }

    private static void checkRows(String name, RowSource src, int expected) {
        if (src.shape().length < 2 || src.shape()[0] != expected) {
            throw new GgufFormatException("Dory tensor '" + name + "' has " + describeRows(src) + " rows, expected "
                    + expected);
        }
    }

    /** The parsed geometry of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(LAYOUT_STATE, Layout.class, () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates the Dory configuration.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    static Layout parse(HfConfig cfg, int blocks) {
        String act = cfg.findStringOr("silu", "hidden_act");
        if (!"silu".equals(act)) {
            throw invalid(cfg, "hidden_act", "must be 'silu', got '" + act + "'");
        }
        if (cfg.findBool(false, "tie_word_embeddings")) {
            throw invalid(cfg, "tie_word_embeddings", "Dory requires untied embeddings");
        }
        if (cfg.findBool(false, "attention_bias")) {
            throw invalid(cfg, "attention_bias", "Dory requires bias-free attention projections");
        }
        if (cfg.findBool(false, "mlp_bias")) {
            throw invalid(cfg, "mlp_bias", "Dory requires bias-free MLP projections");
        }
        String cacheMode = cfg.findStringOr("per_loop", "recurrent_kv_cache_mode").toLowerCase(Locale.ROOT);
        if ("last_loop".equals(cacheMode)) {
            throw invalid(cfg, "recurrent_kv_cache_mode", "'last_loop' reuses one cache across the loop passes;"
                    + " NeuralFree keeps one cache per pass and supports only 'per_loop'");
        }
        if (!"per_loop".equals(cacheMode)) {
            throw invalid(cfg, "recurrent_kv_cache_mode", "must be 'per_loop', got '" + cacheMode + "'");
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

        boolean[] mlp = new boolean[blocks];
        boolean[] sliding = new boolean[blocks];
        String[] types = layerTypes(cfg, blocks);
        for (int i = 0; i < blocks; i++) {
            switch (types[i]) {
                case MLP:
                    mlp[i] = true;
                    break;
                case SLIDING:
                    sliding[i] = true;
                    break;
                case FULL:
                    break;
                default:
                    throw invalid(cfg, "layer_types", "layer " + i + " is '" + types[i]
                            + "'; only 'sliding_attention', 'full_attention' and 'mlp' are supported");
            }
        }
        int window = 0;
        if (any(sliding)) {
            window = cfg.findInt(0, "sliding_window");
            if (window <= 0) {
                throw invalid(cfg, "sliding_window", "must be positive for a sliding_attention layer");
            }
        }
        if (cfg.rope().nested()) {
            if (any(sliding) && !cfg.rope().layer(SLIDING).isPresent()) {
                throw invalid(cfg, "rope_parameters", "must define the sliding_attention rotary parameters");
            }
            if (anyFull(mlp, sliding) && !cfg.rope().layer(FULL).isPresent()) {
                throw invalid(cfg, "rope_parameters", "must define the full_attention rotary parameters");
            }
        }

        int nInput = cfg.findInt(0, "n_input_layers");
        int nOutput = cfg.findInt(0, "n_output_layers");
        int nRecurrent = cfg.findInt(blocks - nInput - nOutput, "n_recurrent_layers");
        int nLoops = cfg.findInt(1, "n_recurrent_loops");
        if (nInput < 0 || nOutput < 0 || nRecurrent < 1 || nLoops < 1
                || (long) nInput + nRecurrent + nOutput != blocks) {
            throw invalid(cfg, "n_recurrent_layers", "the input, recurrent and output layers (" + nInput + " + "
                    + nRecurrent + " + " + nOutput + ") must be non-negative, sum to " + blocks
                    + " and leave a non-empty recurrent block, with n_recurrent_loops >= 1");
        }

        float init = (float) cfg.findFloat(0.02, "initializer_range");
        if (!(init > 0f) || Float.isInfinite(init)) {
            throw invalid(cfg, "initializer_range", "must be a positive finite number, got " + init);
        }
        float alphaInit = (float) cfg.findFloat(0.05, "ngpt_alpha_init");
        if (!(alphaInit > 0f) || Float.isInfinite(alphaInit)) {
            throw invalid(cfg, "ngpt_alpha_init", "must be a positive finite number, got " + alphaInit);
        }
        float g0 = (float) cfg.findFloat(Math.sqrt(hidden), "attention_output_gate_logit_scale_init");
        if (!(g0 > 0f) || Float.isInfinite(g0)) {
            throw invalid(cfg, "attention_output_gate_logit_scale_init", "must be a positive finite number, got " + g0);
        }

        float ropeBase = ropeBase(cfg, FULL, 10_000.0);
        float ropeBaseSwa = ropeBase(cfg, SLIDING, 10_000.0);
        int context = cfg.findInt(8192, "max_position_embeddings");
        if (context <= 0) {
            throw invalid(cfg, "max_position_embeddings", "must be positive, got " + context);
        }
        return new Layout(blocks, mlp, sliding, nInput, nRecurrent, nOutput, nLoops, hidden, nHead, nKv, headDim, nFf,
                vocab, context, window, (float) (1.0 / Math.sqrt(headDim)), ropeBase, ropeBaseSwa, alphaInit / init,
                (float) (Math.sqrt(headDim) / init), (float) (Math.sqrt(g0) / init), (float) Math.sqrt(hidden),
                1.0f / init);
    }

    /** The configured layer types, or the configuration's default pattern when the key is absent. */
    private static String[] layerTypes(HfConfig cfg, int blocks) {
        Object raw = cfg.find("layer_types").orElse(null);
        String[] out = new String[blocks];
        if (raw == null) {
            for (int i = 0; i < blocks; i++) {
                out[i] = i % 2 == 1 ? MLP : i % 8 == 6 ? FULL : SLIDING;
            }
            return out;
        }
        if (!(raw instanceof List)) {
            throw invalid(cfg, "layer_types", "must be an array of layer types");
        }
        List<?> list = (List<?>) raw;
        if (list.size() != blocks) {
            throw invalid(cfg, "layer_types", "has " + list.size() + " entries, expected " + blocks);
        }
        for (int i = 0; i < blocks; i++) {
            Object t = list.get(i);
            if (!(t instanceof String)) {
                throw invalid(cfg, "layer_types", "layer " + i + " is not a string");
            }
            out[i] = (String) t;
        }
        return out;
    }

    /** The rotary base of one attention type: its {@code rope_parameters} entry, else the default. */
    private static float ropeBase(HfConfig cfg, String type, double dflt) {
        Optional<HfRope.Params> params = cfg.rope().layer(type);
        if (!params.isPresent()) {
            return (float) dflt;
        }
        HfRope.Params p = params.get();
        Optional<String> ropeType = p.type();
        if (ropeType.isPresent() && !"default".equals(ropeType.get().toLowerCase(Locale.ROOT))) {
            throw invalid(cfg, "rope_parameters." + type + ".rope_type", "scaling type '" + ropeType.get()
                    + "' is unsupported");
        }
        if (p.factor().isPresent() && p.factor().getAsDouble() != 1.0) {
            throw invalid(cfg, "rope_parameters." + type + ".factor", "RoPE scaling is unsupported");
        }
        if (p.partialRotaryFactor().isPresent() && p.partialRotaryFactor().getAsDouble() != 1.0) {
            throw invalid(cfg, "rope_parameters." + type + ".partial_rotary_factor",
                    "partial rotation is unsupported");
        }
        double theta = p.theta().orElse(dflt);
        if (!(theta > 0) || Double.isInfinite(theta)) {
            throw invalid(cfg, "rope_parameters." + type + ".rope_theta", "must be positive and finite, got " + theta);
        }
        return (float) theta;
    }

    private static boolean any(boolean[] flags) {
        for (boolean f : flags) {
            if (f) {
                return true;
            }
        }
        return false;
    }

    /** True when some block is a full-attention block. */
    private static boolean anyFull(boolean[] mlp, boolean[] sliding) {
        for (int i = 0; i < mlp.length; i++) {
            if (!mlp[i] && !sliding[i]) {
                return true;
            }
        }
        return false;
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
