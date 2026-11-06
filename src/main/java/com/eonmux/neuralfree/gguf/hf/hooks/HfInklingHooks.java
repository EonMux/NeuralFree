package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Hugging Face Inkling conversion (vLLM's {@code InklingForCausalLM}, which has no llama.cpp converter): a NoPE
 * decoder whose local layers attend a sliding window, whose attention adds per-query relative logits (a query-side
 * {@code attn_rel} projection widened by a shared {@code [d_rel, rel_extent]} table) and whose K/V rows pass through
 * depthwise short convolutions, with short convolutions on the attention-output and MLP-output streams and a SwiGLU
 * MLP.
 *
 * <p>The checkpoint fuses the query, key, value and relative projections into one {@code attn.qkvr} product whose
 * rows are {@code [Q | K | V | R]}, and stores the MLP's gate and up rows interleaved ({@code gate0, up0, gate1,
 * up1, ...}); the conversion splits both into the arrays the graph reads. Short-convolution kernels are stored
 * {@code [dim, 1, W]} and are squeezed to {@code [dim, W]}.
 */
public final class HfInklingHooks {

    private static final Pattern MODEL_NAME = Pattern.compile("(?:language_model\\.)?model\\."
            + "(embed_tokens|embed_norm|norm)\\.weight");
    private static final Pattern LAYER_NAME = Pattern.compile("(?:language_model\\.)?model\\.layers\\.(\\d+)\\.(.+)");
    /** The multimodal towers of {@code InklingForConditionalGeneration}; the text backbone is converted alone. */
    private static final Pattern TOWER = Pattern.compile("(?:model\\.)?(?:visual|audio|multi_modal_projector)\\..*");

    /** The per-layer and model geometry of one checkpoint. Immutable. */
    static final class Layout {
        final int nLayer;
        final int nEmbd;
        final int dMlp;
        final int[] nHead;
        final int[] nHeadKv;
        final int[] keyDim;
        final int[] valueDim;
        final boolean[] local;
        final int window;
        final int dRel;
        final int relExtent;
        final int convWidth;
        final int context;
        final int vocab;
        final int paddedVocab;
        final float softcap;
        final float mup;
        final int logFloor;
        final float logAlpha;

        Layout(int nLayer, int nEmbd, int dMlp, int[] nHead, int[] nHeadKv, int[] keyDim, int[] valueDim,
               boolean[] local, int window, int dRel, int relExtent, int convWidth, int context, int vocab,
               int paddedVocab, float softcap, float mup, int logFloor, float logAlpha) {
            this.nLayer = nLayer;
            this.nEmbd = nEmbd;
            this.dMlp = dMlp;
            this.nHead = nHead;
            this.nHeadKv = nHeadKv;
            this.keyDim = keyDim;
            this.valueDim = valueDim;
            this.local = local;
            this.window = window;
            this.dRel = dRel;
            this.relExtent = relExtent;
            this.convWidth = convWidth;
            this.context = context;
            this.vocab = vocab;
            this.paddedVocab = paddedVocab;
            this.softcap = softcap;
            this.mup = mup;
            this.logFloor = logFloor;
            this.logAlpha = logAlpha;
        }

        boolean anyLocal() {
            for (boolean l : local) {
                if (l) {
                    return true;
                }
            }
            return false;
        }
    }

    private HfInklingHooks() {
    }

    /** Inkling has no llama.cpp converter, so it carries its own GGUF architecture name. */
    public static Arch arch(HfConfig cfg, HfTensorIndex index) {
        return Arch.INKLING;
    }

    /**
     * Writes the Inkling metadata: the generic text keys, the per-layer head counts and head dims (local layers may
     * carry their own), the sliding window and its layer switches, the relative table and convolution geometry, the
     * log scaling and the output processors.
     *
     * @throws GgufFormatException when the checkpoint uses a feature the graph does not carry (MoE layers, MTP heads,
     *                             another activation, an inconsistent head geometry)
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        Layout l = layout(ctx);
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.CONTEXT_LENGTH, l.context);
        // The tokenizer and the sampling range cover the real vocabulary; the padded rows are masked at the head.
        kv.set(GgufKeys.VOCAB_SIZE, l.vocab);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, l.nHead);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, l.nHeadKv);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, l.keyDim[fullLayer(l)]);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, l.valueDim[fullLayer(l)]);
        if (l.anyLocal()) {
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH_SWA, l.keyDim[localLayer(l)]);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH_SWA, l.valueDim[localLayer(l)]);
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, l.window);
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, l.local);
        kv.set(GgufKeys.INKLING_RELATIVE_EXTENT, l.relExtent);
        kv.set(GgufKeys.INKLING_RELATIVE_DIM, l.dRel);
        kv.set(GgufKeys.INKLING_SCONV_KERNEL, l.convWidth);
        if (l.logFloor > 0) {
            kv.set(GgufKeys.INKLING_LOG_SCALING_FLOOR, l.logFloor);
        }
        kv.set(GgufKeys.INKLING_LOG_SCALING_ALPHA, l.logAlpha);
        if (l.softcap > 0f) {
            kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, l.softcap);
        }
        if (l.mup > 0f) {
            kv.set(GgufKeys.LOGIT_SCALE, l.mup);
        }
        HfConfig cfg = ctx.config();
        checkPresence(cfg, plan, "use_embed_norm", cfg.findBool(false, "use_embed_norm"),
                "token_embd_norm.weight");
        checkPresence(cfg, plan, "q_bias", cfg.findBool(false, "q_bias"), "blk.0.attn_q.bias");
        checkPresence(cfg, plan, "o_bias", cfg.findBool(false, "o_bias"), "blk.0.attn_output.bias");
        checkPresence(cfg, plan, "use_global_scale", cfg.findBool(false, "use_global_scale"),
                "blk.0.ffn_global_scale.weight");
    }

    /**
     * Requires the checkpoint's optional tensors exactly where the configuration asks for them; a stored tensor the
     * graph would not read, or a missing one it would, changes the forward silently.
     */
    private static void checkPresence(HfConfig cfg, HfPlanView plan, String field, boolean expected, String tensor) {
        boolean present = plan.has(tensor);
        if (present != expected) {
            throw invalid(cfg, field, "is " + expected + " but the checkpoint " + (present ? "stores" : "has no")
                    + " '" + tensor + "'");
        }
    }

    /** The Inkling vocabulary ({@code _set_vocab_gpt2} through the shared SentencePiece/BPE writers). */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfLlamaHooks.vocab(ctx, pre);
    }

    /** Drops the multimodal towers, which the text graph does not read. */
    public static String filter(String name, UnaryOperator<String> base) {
        if (TOWER.matcher(name).matches()) {
            return null;
        }
        return base.apply(name);
    }

    /**
     * Maps the Inkling weights to NeuralFree's {@code inkling} layout, splitting the fused {@code attn.qkvr} rows and
     * the interleaved {@code mlp.gate_up_proj} rows.
     *
     * @throws GgufFormatException naming the tensor when the checkpoint carries one the layout does not define
     */
    public static void tensors(HfLoadContext ctx, String name, RowSource src, TensorSink out) throws IOException {
        if (out.isConsumed(name)) {
            return;
        }
        Layout l = layout(ctx);
        if (name.equals("lm_head.weight")) {
            requireShape(ctx, name, src, l.paddedVocab, l.nEmbd);
            out.add("output.weight", src);
            return;
        }
        Matcher model = MODEL_NAME.matcher(name);
        if (model.matches()) {
            if (model.group(1).equals("embed_tokens")) {
                requireShape(ctx, name, src, l.paddedVocab, l.nEmbd);
            } else {
                requireShape(ctx, name, src, l.nEmbd);
            }
            out.add(modelBase(model.group(1)) + ".weight", src);
            return;
        }
        Matcher layer = LAYER_NAME.matcher(name);
        if (!layer.matches()) {
            throw invalid(ctx.config(), "tensor", "Inkling does not support tensor '" + name + "'");
        }
        int il = Integer.parseInt(layer.group(1));
        if (il < 0 || il >= l.nLayer) {
            throw invalid(ctx.config(), "tensor", "Inkling has " + l.nLayer + " layers; '" + name + "' names layer "
                    + il);
        }
        layerTensor(ctx, l, il, layer.group(2), src, out);
    }

    private static String modelBase(String part) {
        switch (part) {
            case "embed_tokens":
                return "token_embd";
            case "embed_norm":
                return "token_embd_norm";
            default:
                return "output_norm";
        }
    }

    /** One layer tensor: its GGUF name, or its split parts, from the source rows. */
    private static void layerTensor(HfLoadContext ctx, Layout l, int il, String part, RowSource src, TensorSink out) {
        String blk = "blk." + il + ".";
        Geometry g = geometry(l, il);
        switch (part) {
            case "attn_norm.weight":
                requireShape(ctx, part, src, l.nEmbd);
                out.add(blk + "attn_norm.weight", src);
                return;
            case "mlp_norm.weight":
                requireShape(ctx, part, src, l.nEmbd);
                out.add(blk + "ffn_norm.weight", src);
                return;
            case "attn.qkvr.weight":
                requireShape(ctx, part, src, g.fusedRows, l.nEmbd);
                addQueryRows(l, il, src, out, blk, true);
                return;
            case "attn.qkvr.bias":
                requireShape(ctx, part, src, g.fusedRows);
                addQueryRows(l, il, src, out, blk, false);
                return;
            case "attn.wo_ud.weight":
                requireShape(ctx, part, src, l.nEmbd, g.qDim);
                out.add(blk + "attn_output.weight", src);
                return;
            case "attn.wo_ud.bias":
                requireShape(ctx, part, src, l.nEmbd);
                out.add(blk + "attn_output.bias", src);
                return;
            case "attn.rel_logits_proj.proj":
                requireShape(ctx, part, src, l.dRel, g.relExtent);
                out.add(blk + "attn_rel_proj.weight", src);
                return;
            case "attn.q_norm.weight":
                requireShape(ctx, part, src, g.headDim);
                out.add(blk + "attn_q_norm.weight", src);
                return;
            case "attn.k_norm.weight":
                requireShape(ctx, part, src, g.headDim);
                out.add(blk + "attn_k_norm.weight", src);
                return;
            case "attn.k_sconv.weight":
                out.add(blk + "attn_sconv_k.weight", convSource(ctx, src));
                return;
            case "attn.v_sconv.weight":
                out.add(blk + "attn_sconv_v.weight", convSource(ctx, src));
                return;
            case "attn_sconv.weight":
                out.add(blk + "attn_sconv.weight", convSource(ctx, src));
                return;
            case "mlp_sconv.weight":
                out.add(blk + "ffn_sconv.weight", convSource(ctx, src));
                return;
            case "mlp.gate_up_proj.weight":
                requireShape(ctx, part, src, 2 * l.dMlp, l.nEmbd);
                out.add(blk + "ffn_gate.weight", RowTransforms.interleaveRows(src, 2, 0));
                out.add(blk + "ffn_up.weight", RowTransforms.interleaveRows(src, 2, 1));
                return;
            case "mlp.down_proj.weight":
                requireShape(ctx, part, src, l.nEmbd, l.dMlp);
                out.add(blk + "ffn_down.weight", src);
                return;
            case "mlp.global_scale":
                out.add(blk + "ffn_global_scale.weight", src);
                return;
            default:
                throw invalid(ctx.config(), "tensor", "Inkling layer " + il + " has no layout for '" + part
                        + "' (tensor '" + blk + part + "')");
        }
    }

    /**
     * Requires the source's shape, so a checkpoint whose fused projections or padded vocabulary disagree with its
     * configuration fails by name instead of producing a mis-sized layer.
     *
     * @throws GgufFormatException naming the tensor, its shape and the expected one
     */
    private static void requireShape(HfLoadContext ctx, String name, RowSource src, long... expected) {
        long[] shape = src.shape();
        if (java.util.Arrays.equals(shape, expected)) {
            return;
        }
        throw invalid(ctx.config(), "tensor", "'" + name + "' has shape " + Arrays.toString(shape) + ", expected "
                + Arrays.toString(expected));
    }

    /**
     * Splits the fused {@code [Q | K | V | R]} rows (or the fused bias vector) of layer {@code il} into
     * {@code attn_q}, {@code attn_k}, {@code attn_v} and {@code attn_rel}, with that layer's head geometry.
     */
    private static void addQueryRows(Layout l, int il, RowSource src, TensorSink out, String blk, boolean matrix) {
        long q = (long) l.nHead[il] * l.keyDim[il];
        long kv = (long) l.nHeadKv[il] * l.keyDim[il];
        long r = (long) l.nHead[il] * l.dRel;
        String weight = matrix ? ".weight" : ".bias";
        out.add(blk + "attn_q" + weight, RowTransforms.sliceRows(src, 0, q));
        out.add(blk + "attn_k" + weight, RowTransforms.sliceRows(src, q, kv));
        out.add(blk + "attn_v" + weight, RowTransforms.sliceRows(src, q + kv, kv));
        out.add(blk + "attn_rel" + weight, RowTransforms.sliceRows(src, q + 2 * kv, r));
    }

    /** A {@code [dim, 1, W]} depthwise convolution kernel squeezed to {@code [dim, W]}. */
    private static RowSource convSource(HfLoadContext ctx, RowSource src) {
        RowSource s = ElementTransforms.squeezeConv(src);
        if (s.shape().length != 2) {
            throw invalid(ctx.config(), "tensor", "an Inkling short-convolution kernel must squeeze to two"
                    + " dimensions, got " + java.util.Arrays.toString(s.shape()));
        }
        return s;
    }

    /** The attention geometry of layer {@code il}: its head counts and dims and the fused projection's row count. */
    private static final class Geometry {
        final int qDim;
        final int kDim;
        final int rDim;
        final int headDim;
        final int relExtent;
        final int fusedRows;

        Geometry(Layout l, int il) {
            this.headDim = l.keyDim[il];
            this.qDim = l.nHead[il] * headDim;
            this.kDim = l.nHeadKv[il] * headDim;
            this.rDim = l.nHead[il] * l.dRel;
            this.relExtent = l.local[il] ? l.window : l.relExtent;
            this.fusedRows = qDim + 2 * kDim + rDim;
        }
    }

    private static Geometry geometry(Layout l, int il) {
        return new Geometry(l, il);
    }

    /** The first local layer, whose head dims the {@code *_swa} keys carry. */
    private static int localLayer(Layout l) {
        for (int il = 0; il < l.nLayer; il++) {
            if (l.local[il]) {
                return il;
            }
        }
        throw new IllegalStateException("no local layer in a layout that reports one");
    }

    /** The first full-attention layer, whose head dims the {@code attention.key_length} keys carry. */
    private static int fullLayer(Layout l) {
        for (int il = 0; il < l.nLayer; il++) {
            if (!l.local[il]) {
                return il;
            }
        }
        return 0;
    }

    /** The parsed geometry of {@code ctx}, built once per load. */
    static Layout layout(HfLoadContext ctx) {
        return ctx.state(HfInklingHooks.class.getName() + ".layout", Layout.class,
                () -> parse(ctx.config(), ctx.blockCount()));
    }

    /**
     * Parses and validates the Inkling configuration.
     *
     * @throws GgufFormatException naming the unsupported field
     */
    static Layout parse(HfConfig cfg, int blocks) {
        if (blocks <= 0) {
            throw invalid(cfg, "num_hidden_layers", "must be positive, got " + blocks);
        }
        String act = cfg.findStringOr("silu", "hidden_act", "hidden_activation");
        if (!"silu".equals(act) && !"swish".equals(act)) {
            throw invalid(cfg, "hidden_act", "must be 'silu', got '" + act + "'");
        }
        int routed = cfg.findInt(0, "n_routed_experts");
        int denseLayers = cfg.findInt(0, "dense_mlp_idx");
        if (routed > 0 && denseLayers < blocks) {
            throw invalid(cfg, "dense_mlp_idx", "Inkling mixture-of-experts layers are unsupported; every layer must"
                    + " use the dense MLP (" + routed + " routed experts from layer " + denseLayers + ", "
                    + blocks + " layers)");
        }
        if (cfg.findInt(0, "num_nextn_predict_layers") != 0) {
            throw invalid(cfg, "num_nextn_predict_layers", "Inkling multi-token-prediction heads are unsupported");
        }
        if (cfg.findBool(false, "chain_hidden_post_norm")) {
            throw invalid(cfg, "chain_hidden_post_norm", "the MTP chain norm is unsupported");
        }
        if (cfg.findBool(false, "tie_word_embeddings")) {
            throw invalid(cfg, "tie_word_embeddings", "Inkling's LM head must be stored (untied)");
        }
        int nEmbd = cfg.requireInt("hidden_size");
        int dMlp = cfg.findInt(cfg.requireInt("intermediate_size"), "dense_intermediate_size");
        if (nEmbd <= 0 || dMlp <= 0) {
            throw invalid(cfg, "hidden_size", "hidden_size and dense_intermediate_size must be positive, got "
                    + nEmbd + "/" + dMlp);
        }
        int nHead = cfg.requireInt("num_attention_heads");
        int nHeadKv = cfg.findInt(nHead, "num_key_value_heads");
        int headDim = cfg.findInt(nEmbd / Math.max(1, nHead), "head_dim");
        int valueDim = cfg.findInt(headDim, "v_head_dim");
        int swaHead = cfg.findInt(nHead, "swa_num_attention_heads");
        int swaHeadKv = cfg.findInt(nHeadKv, "swa_num_key_value_heads");
        int swaHeadDim = cfg.findInt(headDim, "swa_head_dim");
        int swaValueDim = cfg.findInt(swaHeadDim, "swa_v_head_dim");
        List<Integer> localIds = cfg.findIntList(Collections.emptyList(), "local_layer_ids");
        boolean[] local = new boolean[blocks];
        for (Integer id : localIds) {
            if (id == null || id < 0 || id >= blocks) {
                throw invalid(cfg, "local_layer_ids", "layer " + id + " is outside [0, " + blocks + ")");
            }
            local[id] = true;
        }
        int window = cfg.findInt(0, "sliding_window_size");
        if (localIds.size() > 0 && window < 1) {
            throw invalid(cfg, "sliding_window_size", "local layers need a positive window, got " + window);
        }
        int[] heads = new int[blocks];
        int[] kvHeads = new int[blocks];
        int[] keyDims = new int[blocks];
        int[] valueDims = new int[blocks];
        for (int il = 0; il < blocks; il++) {
            heads[il] = local[il] ? swaHead : nHead;
            kvHeads[il] = local[il] ? swaHeadKv : nHeadKv;
            keyDims[il] = local[il] ? swaHeadDim : headDim;
            valueDims[il] = local[il] ? swaValueDim : valueDim;
            if (heads[il] <= 0 || kvHeads[il] <= 0 || heads[il] % kvHeads[il] != 0) {
                throw invalid(cfg, "num_attention_heads", "layer " + il + " has " + heads[il]
                        + " query heads, not a positive multiple of its " + kvHeads[il] + " KV heads");
            }
            if (keyDims[il] <= 0 || valueDims[il] <= 0 || keyDims[il] != valueDims[il]) {
                throw invalid(cfg, "head_dim", "layer " + il + " has K/V head dims " + keyDims[il] + "/"
                        + valueDims[il] + "; the graph normalizes K and V per head over one width");
            }
        }
        int dRel = cfg.findInt(16, "d_rel");
        int relExtent = cfg.findInt(1024, "rel_extent");
        if (dRel <= 0 || relExtent <= 0) {
            throw invalid(cfg, "d_rel", "d_rel and rel_extent must be positive, got " + dRel + "/" + relExtent);
        }
        int convWidth = cfg.findInt(4, "sconv_kernel_size", "conv_kernel_size");
        if (convWidth < 2) {
            throw invalid(cfg, "sconv_kernel_size", "must be at least 2, got " + convWidth);
        }
        int logFloor = cfg.findInt(0, "log_scaling_n_floor");
        if (logFloor < 0) {
            throw invalid(cfg, "log_scaling_n_floor", "must be non-negative, got " + logFloor);
        }
        double logAlpha = cfg.findFloat(0.1, "log_scaling_alpha");
        if (!Double.isFinite(logAlpha)) {
            throw invalid(cfg, "log_scaling_alpha", "must be finite, got " + logAlpha);
        }
        double softcap = cfg.findFloat(0.0, "final_logit_softcapping");
        if (softcap < 0) {
            throw invalid(cfg, "final_logit_softcapping", "must be non-negative, got " + softcap);
        }
        double mup = cfg.findFloat(0.0, "logits_mup_width_multiplier");
        if (mup < 0) {
            throw invalid(cfg, "logits_mup_width_multiplier", "must be non-negative, got " + mup);
        }
        if (mup > 0 && softcap > 0) {
            throw invalid(cfg, "logits_mup_width_multiplier", "the muP width divisor and the final logit soft cap"
                    + " are exclusive in vLLM's Inkling logits processor");
        }
        double eps = cfg.findFloat(1e-6, "rms_norm_eps");
        if (!(eps > 0)) {
            throw invalid(cfg, "rms_norm_eps", "must be positive, got " + eps);
        }
        int context = cfg.findInt(131_072, "max_position_embeddings");
        int vocab = cfg.requireInt("vocab_size");
        int padded = cfg.findInt(vocab, "padded_vocab_size");
        if (padded < vocab) {
            throw invalid(cfg, "padded_vocab_size", "is smaller than vocab_size " + vocab + ", got " + padded);
        }
        return new Layout(blocks, nEmbd, dMlp, heads, kvHeads, keyDims, valueDims, local, window, dRel, relExtent,
                convWidth, context, vocab, padded, (float) softcap, (float) mup, logFloor, (float) logAlpha);
    }

    private static GgufFormatException invalid(HfConfig cfg, String field, String message) {
        return new GgufFormatException(cfg.source() + ": '" + field + "': " + message);
    }
}
