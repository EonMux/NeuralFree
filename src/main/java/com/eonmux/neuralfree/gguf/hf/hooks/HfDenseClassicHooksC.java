package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.hooks.HfDenseClassicHooksA;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.QkvLayouts;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Hooks of the llama.cpp classic-dense converter classes, part C: {@code FalconModel}
 * (LC/conversion/falcon.py:15-59), {@code GPT2Model} and {@code RuGPT3XLModel} (gpt2.py:15-80),
 * {@code StarCoderModel} (starcoder.py:8-21), {@code GPTNeoXModel} (gptneox.py:17-64),
 * {@code RefactModel} (refact.py:13-67), {@code Glm4Model} (glm.py:19-81) and {@code GlmOCRModel} (glm.py:91-112).
 *
 * <p>Every method is a public static hook reference that the table composes and subclasses reuse where the converter
 * calls {@code super()}.</p>
 */
public final class HfDenseClassicHooksC {

    private HfDenseClassicHooksC() {
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                + "' is missing"));
    }

    // ------------------------------------------------------------------ Falcon

    /**
     * {@code FalconModel.set_gguf_parameters} (falcon.py:19-36), hand-set: the fixed 2048 context (not in
     * config.json), the {@code jploski} layout marker of the QKV tensor transform, the width and 4x FFN, the head
     * count aliases {@code num_attention_heads|n_head} over the KV aliases {@code num_kv_heads|n_head_kv|1}, and the
     * {@code layer_norm_epsilon} epsilon.
     */
    public static void falconMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int heads = cfg.findInt(cfg.requireInt("n_head"), "num_attention_heads", "n_head");
        kv.set(GgufKeys.CONTEXT_LENGTH, 2048);
        kv.set(GgufKeys.TENSOR_DATA_LAYOUT, "jploski");
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.findInt(1, "num_kv_heads", "n_head_kv"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_epsilon"));
    }

    /**
     * {@code FalconModel.modify_tensors}' QKV part (falcon.py:48-57): the fused {@code query_key_value} tensor of
     * the grouped layout {@code [nKv, nHead/nKv + 2, headDim, hidden]} is re-ordered to {@code [Q; K; V]} and keeps
     * its GGUF mapping; every other tensor passes on.
     *
     * @throws GgufFormatException when the fused tensor does not hold the grouped heads of the head width
     */
    public static void falconTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName) || !hfName.contains("query_key_value")) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.findInt(cfg.requireInt("n_head"), "num_attention_heads", "n_head");
        int kvHeads = cfg.findInt(1, "num_kv_heads", "n_head_kv");
        if (kvHeads <= 0) {
            kvHeads = 1;
        }
        int headDim = cfg.requireInt("hidden_size") / heads;
        RowSource regrouped = QkvLayouts.regroupQkv(src, QkvLayouts.Layout.GROUPED, heads, kvHeads, headDim);
        out.add(ctx.names().mapOrThrow(hfName), regrouped);
    }

    // ------------------------------------------------------------------ GPT-2

    /** The Conv1D name tails whose logical shape decides the transpose of gpt2.py:34-36. */
    private static final String[] CONV1D_WEIGHTS = {".c_attn.weight", ".c_proj.weight", ".c_fc.weight"};

    /**
     * {@code GPT2Model.set_gguf_parameters} (gpt2.py:18-24), hand-set: the {@code n_ctx} context, the
     * {@code n_embd} width and 4x FFN, the {@code n_head} head count, and the {@code layer_norm_epsilon} epsilon.
     */
    public static void gpt2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmb = cfg.requireInt("n_embd");
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("n_ctx"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmb);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * nEmb);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_head"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_epsilon"));
    }

    /**
     * {@code GPT2Model.modify_tensors} (gpt2.py:25-43): the attention mask buffers are dropped (the comment's intent
     * of gpt2.py:27-29, consumed downstream by the generic skip list), the Conv1D weights are transposed last-two
     * only when their logical shape is {@code [in, out]} against the configured widths (so a Quantization-Rewrite-1
     * tensor already in {@code [N, K]} is not touched again), and every other tensor maps directly.
     *
     * @throws GgufFormatException when a Conv1D weight matches no configured width
     */
    public static void gpt2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".attn.masked_bias") || hfName.endsWith(".attn.bias")) {
            out.drop(hfName, "the attention mask buffer is not part of the GGUF view (gpt2.py:27-29)");
            return;
        }
        RowSource data = src;
        for (String tail : CONV1D_WEIGHTS) {
            if (hfName.endsWith(tail)) {
                data = transposeConv1d(ctx, src, tail);
                break;
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /**
     * {@code data_torch.transpose(1, 0)} of gpt2.py:34-36 when the stored shape is the logical {@code [in, out]}:
     * {@code c_attn} and {@code c_fc} take their input width as {@code n_embd}, the MLP {@code c_proj} as the
     * inner width {@code n_inner}; a tensor whose outer dimension already matches the output width is left as
     * produced by the quantization rewrite.
     */
    private static RowSource transposeConv1d(HfLoadContext ctx, RowSource src, String tail) {
        HfConfig cfg = ctx.config();
        int nEmb = cfg.requireInt("n_embd");
        int inner = cfg.findInt(4 * nEmb, "n_inner", "n_embd");
        long[] shape = src.shape();
        boolean weightTail = ".weight".equals(tail.substring(tail.length() - ".weight".length()));
        if (!weightTail || shape.length < 2) {
            return src;
        }
        boolean proj = tail.startsWith(".c_proj");
        long dIn = proj ? inner : nEmb;
        long d0 = shape[0];
        if (d0 == dIn && d0 != shape[1]) {
            return ExpertTransforms.transposeLast2(src);
        }
        if (d0 == shape[1] && d0 == dIn && !proj) {
            throw new GgufFormatException("Conv1D tensor '" + src.describe() + "' of " + Arrays.toString(shape)
                    + " matches no configured width for '" + tail + "'");
        }
        return src;
    }

    /**
     * The Conv1D transpose decision of {@code gpt2Tensors} shared with the Jais part (jais.py:104-109 picks its
     * Conv1D name tails around {@code gpt2.py}'s shape rule): the transpose decision of gpt2.py:34-36 for the tail
     * that matches, the source unchanged otherwise.
     */
    static RowSource gpt2TransposeDecision(HfLoadContext ctx, RowSource src, String name) {
        for (String tail : CONV1D_WEIGHTS) {
            if (name.endsWith(tail)) {
                return transposeConv1d(ctx, src, tail);
            }
        }
        return src;
    }

    // ------------------------------------------------------------------ RuGPT3XL

    /** The per-load Q/K/V buffers of {@code RuGPT3XLModel.modify_tensors} (gpt2.py:46-72). */
    private static final String QKV_STATE = HfDenseClassicHooksC.class.getName() + ".rugptQkv";

    /**
     * {@code RuGPT3XLModel.modify_tensors} (gpt2.py:46-72): the separate {@code q_proj}, {@code k_proj} and
     * {@code v_proj} tensors of a layer and suffix are fused into {@code attn_qkv.<suffix>} when all three were
     * seen, and buffered until then; every other tensor passes on.
     */
    public static void rugptTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        boolean isQ = hfName.contains(".self_attn.q_proj.");
        boolean isK = !isQ && hfName.contains(".self_attn.k_proj.");
        boolean isV = !isQ && !isK && hfName.contains(".self_attn.v_proj.");
        if (!isQ && !isK && !isV) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        String part = isQ ? "q" : isK ? "k" : "v";
        String tail = hfName.endsWith(".weight") ? "weight" : "bias";
        int bid = HfHookSupport.layerOf(hfName);
        Map<Integer, Map<String, RowSource>> state = qkvState(ctx);
        Map<String, RowSource> layer = state.computeIfAbsent(bid, b -> new LinkedHashMap<>());
        layer.put(part + "." + tail, src);
        if (layer.containsKey("q." + tail) && layer.containsKey("k." + tail) && layer.containsKey("v." + tail)) {
            RowSource q = layer.remove("q.assoc.".replace("assoc", tail));
            RowSource k = layer.remove("k." + tail);
            RowSource v = layer.remove("v." + tail);
            out.add(ctx.names().ggufName("ATTN_QKV", bid, "." + tail),
                    RowTransforms.concatRows(Arrays.asList(q, k, v)));
        }
    }

    private static Map<Integer, Map<String, RowSource>> qkvState(HfLoadContext ctx) {
        return ctx.state(QKV_STATE, Map.class, HashMap::new);
    }

    /**
     * The {@code prepare_tensors} check of {@code RuGPT3XLModel} (gpt2.py:74-80): no buffered Q/K/V part may be left
     * after the checkpoint was planned.
     */
    public static void rugptExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        Map<Integer, Map<String, RowSource>> state = qkvState(ctx);
        for (Map.Entry<Integer, Map<String, RowSource>> e : state.entrySet()) {
            if (!e.getValue().isEmpty()) {
                throw new GgufFormatException("unprocessed Q/K/V parts of layer " + e.getKey() + ": "
                        + e.getValue().keySet());
            }
        }
    }

    // ------------------------------------------------------------------ StarCoder

    /**
     * {@code StarCoderModel.set_gguf_parameters} (starcoder.py:11-17), hand-set: the {@code n_positions} context,
     * the {@code n_embd} width and 4x FFN, the {@code n_head} head count over one KV head, and the
     * {@code layer_norm_epsilon} epsilon.
     */
    public static void starcoderMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmb = cfg.requireInt("n_embd");
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("n_positions"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmb);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * nEmb);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_head"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_epsilon"));
    }

    // ------------------------------------------------------------------ GPT-NeoX

    /**
     * {@code GPTNeoXModel.set_gguf_parameters} (gptneox.py:20-34), hand-set: the context, width, block count, the
     * partial-rotary RoPE width, head count, parallel residual and epsilon.
     */
    public static void gptneoxMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int nEmb = cfg.requireInt("hidden_size");
        double pct = cfg.rope().global().partialRotaryFactor().orElse(1.0);
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_position_embeddings"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmb);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (pct * (nEmb / heads)));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.USE_PARALLEL_RESIDUAL, cfg.findBool(true, "use_parallel_residual"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_eps"));
    }

    /**
     * {@code GPTNeoXModel.modify_tensors} (gptneox.py:36-63): like {@link HfDenseClassicHooksA
     * #interleavedQkvTensors}, the Bloom-style per-head interleaved {@code query_key_value} weight or bias becomes
     * {@code [Q; K; V]}; every other tensor passes on.
     */
    public static void gptneoxTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName) || !hfName.contains("." + HfDenseClassicHooksA.QKV_TAIL + ".")) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.findInt(cfg.requireInt("num_attention_heads"), "n_head", "num_attention_heads");
        int hidden = cfg.requireInt("hidden_size");
        RowSource regrouped = QkvLayouts.regroupQkv(src, QkvLayouts.Layout.PER_HEAD_INTERLEAVED, heads, heads,
                hidden / heads);
        out.add(ctx.names().mapOrThrow(hfName), regrouped);
    }

    // ------------------------------------------------------------------ Refact

    /** The Refact intermediate width: {@code multiple_of=256}, 2/3 of {@code 4 * n_embd} (refact.py:37-40). */
    private static int refactFfn(HfConfig cfg) {
        int hidden = cfg.requireInt("n_embd");
        int inner = 4 * hidden;
        int upon = (int) (2.0 * inner / 3);
        int multiple = 256;
        return multiple * ((upon + multiple - 1) / multiple);
    }

    /**
     * {@code RefactModel.set_gguf_parameters} (refact.py:42-52), hand-set: the Alibi position embedding carries no
     * RoPE; the context, width, the rounded FFN width, one KV head and the RMS epsilon.
     */
    public static void refactMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmb = cfg.requireInt("n_embd");
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("n_positions"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmb);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, refactFfn(cfg));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_head"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 1);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, requireDouble(cfg, "layer_norm_epsilon"));
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
    }

    /**
     * {@code RefactModel.set_vocab}'s special-tokens step (refact.py:20-28): the FIM prefix, suffix and middle ids
     * are fixed and the chat template stays unset; the eot id of the tokenizer still applies.
     */
    public static Map<String, Object> refactVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfDenseClassicHooksA.gpt2Vocab(ctx, pre));
        Map<String, Object> out = new LinkedHashMap<>();
        out.put(GgufKeys.TOKENIZER_FIM_PRE_ID, 1);
        out.put(GgufKeys.TOKENIZER_FIM_SUF_ID, 3);
        out.put(GgufKeys.TOKENIZER_FIM_MID_ID, 2);
        for (Map.Entry<String, Object> e : md.entrySet()) {
            if (e.getKey().equals(GgufKeys.TOKENIZER_CHAT_TEMPLATE)) {
                continue;
            }
            if (!out.containsKey(e.getKey())) {
                out.put(e.getKey(), e.getValue());
            }
        }
        if (!out.containsKey(GgufKeys.TOKENIZER_EOT_ID) && md.containsKey(GgufKeys.TOKENIZER_EOT_ID)) {
            out.put(GgufKeys.TOKENIZER_EOT_ID, md.get(GgufKeys.TOKENIZER_EOT_ID));
        }
        return Collections.unmodifiableMap(out);
    }

    /**
     * {@code RefactModel.modify_tensors} (refact.py:55-66): block tensors of the listed names are split with the
     * configured widths; every other tensor passes on.
     *
     * @throws GgufFormatException when a split tensor is not the configured width
     */
    public static void refactTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0) {
            if (hfName.contains(".attn.kv.weight")) {
                HfConfig c = cfg;
                int headDim = c.requireInt("n_embd") / c.requireInt("n_head");
                long kvWidth = 1L * headDim;
                if (src.shape()[0] < 2 * kvWidth) {
                    throw new GgufFormatException("refact attn.kv '" + hfName + "' is shorter than 2 kv heads");
                }
                out.add(ctx.names().ggufName("ATTN_K", bid, ".weight"), RowTransforms.sliceRows(src, 0, kvWidth));
                out.add(ctx.names().ggufName("ATTN_V", bid, ".weight"),
                        RowTransforms.sliceRows(src, kvWidth, src.shape()[0] - kvWidth));
                return;
            }
            if (hfName.contains(".attn.q.weight")) {
                out.add(ctx.names().ggufName("ATTN_Q", bid, ".weight"), src);
                return;
            }
            if (hfName.contains(".mlp.gate_up_proj.weight")) {
                int ff = refactFfn(cfg);
                if (src.shape()[0] != 2L * ff) {
                    throw new GgufFormatException("refact gate_up_proj '" + hfName + "' must hold 2 * " + ff
                            + " rows, got " + src.shape()[0]);
                }
                out.add(ctx.names().ggufName("FFN_GATE", bid, ".weight"), RowTransforms.sliceRows(src, 0, ff));
                out.add(ctx.names().ggufName("FFN_UP", bid, ".weight"),
                        RowTransforms.sliceRows(src, ff, ff));
                return;
            }
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ GLM4 and GLM-OCR

    /** Shape-filter of the {@code assistant}-carried decisions. */
    public static final HfNameFilter GLM_FILTER = HfNameFilter.INHERITED;

    /**
     * {@code Glm4Model.set_gguf_parameters} (glm.py:46-49): the inherited keys, then the RoPE width
     * {@code head_dim|hidden_size // num_attention_heads} times the partial rotary factor (0.5).
     */
    public static void glm4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int ropeDim = cfg.findInt(cfg.requireInt("hidden_size") / heads, "head_dim");
        double partial = cfg.rope().global().partialRotaryFactor().orElse(0.5);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (ropeDim * partial));
    }

    /**
     * {@code Glm4Model.modify_tensors} (glm.py:51-64): when the configuration carries {@code mrope_section}, q/k
     * weights and biases take the normal-to-NeoX permutation within the rotary part (glm.py:54-73) so the M-RoPE
     * kernel reads them; every other tensor passes on.
     */
    public static void glm4Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (!cfg.rope().raw().containsKey("mrope_section")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, glm4MropePermute(ctx, hfName, src), out);
    }

    private static RowSource glm4MropePermute(HfLoadContext ctx, String hfName, RowSource src) {
        boolean q = hfName.endsWith("q_proj.weight") || hfName.endsWith("q_proj.bias");
        boolean k = !q && (hfName.endsWith("k_proj.weight") || hfName.endsWith("k_proj.bias"));
        if (!q && !k) {
            return src;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int kvHeads = cfg.requireInt("num_key_value_heads");
        int headDim = cfg.findInt(cfg.requireInt("hidden_size") / heads, "head_dim");
        double partial = cfg.rope().global().partialRotaryFactor().orElse(0.5);
        int rotDim = (int) (headDim * partial);
        if (rotDim % 2 != 0) {
            throw new GgufFormatException(cfg.source() + ": the GLM-4 rotary width " + rotDim + " must be even");
        }
        if (q) {
            return RowTransforms.normalToNeoxPartial(src, heads, headDim, rotDim);
        }
        long d0 = src.shape().length == 0 ? 0 : src.shape()[0];
        long kvRows = (long) kvHeads * headDim;
        if (d0 % headDim != 0 || d0 / headDim != kvHeads) {
            if (d0 == ctx.config().requireInt("hidden_size")) {
                return RowTransforms.normalToNeoxPartial(src, heads, headDim, rotDim);
            }
        }
        return RowTransforms.normalToNeoxPartial(src, kvHeads, headDim, rotDim);
    }

    /**
     * {@code GlmOCRModel.set_gguf_parameters}' NextN keys (glm.py:109-111): the shared GLM-4 keys, then
     * {@code num_nextn_predict_layers} when present.
     */
    public static void glmOcrMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        glm4Meta(ctx, kv, plan);
        int nNextn = ctx.config().findInt(0, "num_nextn_predict_layers");
        if (nNextn > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nNextn);
        }
    }
}
