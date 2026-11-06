package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * The masked-diffusion and recurrent-depth hooks: LC/conversion/dream.py
 * {@code DreamModel} (:13-73), LC/conversion/llada.py {@code LLaDAModel} (:15-118) and — for the Qwen-MoE tensor
 * steps the converter copies from {@code Qwen2MoeModel} — {@code LLaDAMoEModel}; LC/conversion/qwen.py
 * {@code RND1Model} (:449-461) and LC/conversion/nanbeige.py {@code NanbeigeModel} on the Llama hooks;
 * LC/conversion/hrm_text.py {@code HrmTextModel} (:15-79).
 *
 * <p>The diffusion models run bidirectionally, so every entry writes {@code attention.causal} false; the audio
 * decoder shares {@link #passThrough} because its checkpoint tensors map as they are.</p>
 */
public final class HfDiffusionHooks {

    private HfDiffusionHooks() {
    }

    // ------------------------------------------------------------------ shared steps

    /** {@code ModelBase.modify_tensors} with the consumed check the table entries carry. */
    public static void passThrough(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** The diffusion mask token of the configuration, or null. */
    private static Integer maskTokenId(HfLoadContext ctx) {
        return ctx.config().findInt("mask_token_id").isPresent() ? ctx.config().findInt("mask_token_id").getAsInt()
                : null;
    }

    // ------------------------------------------------------------------ DreamModel

    /**
     * {@code DreamModel.set_gguf_parameters} (dream.py:41-52): the generic keys, the optional pooling, the
     * bidirectional attention and the mask token.
     */
    public static void dreamMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfQwenHooks.pooling(ctx, kv);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        Integer mask = maskTokenId(ctx);
        if (mask != null) {
            kv.set(GgufKeys.TOKENIZER_MASK_ID, mask);
        }
    }

    /**
     * {@code DreamModel.set_vocab} (dream.py:58-63): the sentencepiece tokenizer of the checkpoint, else GPT-2 BPE.
     */
    public static Map<String, Object> dreamVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        if (HfVocabWriters.canSentencePiece(in)) {
            return HfVocabWriters.sentencePiece(in, pre);
        }
        return HfVocabWriters.gpt2(in, pre);
    }

    // ------------------------------------------------------------------ LLaDAModel

    /**
     * {@code LLaDAModel.set_gguf_parameters} (llada.py:89-108): the generic keys, the pooling, then the model's own
     * dimension keys read through its {@code d_model}/{@code mlp_hidden_size} configuration names, the context
     * length of {@code max_sequence_length} and the bidirectional non-shifted diffusion.
     */
    public static void lladaMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfQwenHooks.pooling(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.CONTEXT_LENGTH, (int) cfg.findLong(4096, "max_sequence_length"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, (int) cfg.findLong(4096, "d_model"));
        int heads = cfg.requireInt("num_attention_heads", "n_heads");
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, (int) cfg.findLong(12288, "mlp_hidden_size"));
        Optional<Integer> headDim = cfg.findInt("head_dim").isPresent() ? Optional.of(cfg.findInt("head_dim").getAsInt())
                : Optional.empty();
        int ropeDim = headDim.orElseGet(() -> cfg.findInt("hidden_size", "d_model").orElse(-1) / heads);
        if (ropeDim <= 0) {
            throw new GgufFormatException(cfg.source() + ": 'head_dim' or hidden_size/n_heads is required");
        }
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        kv.set(GgufKeys.DIFFUSION_SHIFT_LOGITS, false);
    }

    /**
     * {@code LLaDAModel.modify_tensors} (llada.py:110-118): the undo permute of the Q/K projections and their
     * biases ({@code undo_permute = True}), then the direct name mapping of the base model.
     */
    public static void lladaTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = HfLlamaHooks.permuteQk(ctx.config(), hfName, src);
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /** {@code LLaDAModel.set_vocab} (llada.py:84-88): GPT-2 BPE with an explicit BOS. */
    public static Map<String, Object> lladaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_ADD_BOS, Boolean.TRUE);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ LLaDAMoEModel

    /**
     * {@code LLaDAMoEModel.set_gguf_parameters} (llada.py:121-132): the generic keys, the expert width of
     * {@code expert_intermediate_size}, the bidirectional non-shifted diffusion and the hard-coded mask token
     * 156895 (llada.py:128).
     */
    public static void lladaMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        cfg.findInt("expert_intermediate_size")
                .ifPresent(v -> kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, v));
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        kv.set(GgufKeys.DIFFUSION_SHIFT_LOGITS, false);
        kv.set(GgufKeys.TOKENIZER_MASK_ID, 156895);
    }

    // ------------------------------------------------------------------ RND1Model

    /**
     * {@code RND1Model.set_gguf_parameters} (qwen.py:452-460): the Qwen2Moe keys, the bidirectional attention and
     * the mask token of the configuration.
     */
    public static void rnd1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);
        Integer mask = maskTokenId(ctx);
        if (mask != null) {
            kv.set(GgufKeys.TOKENIZER_MASK_ID, mask);
        }
    }

    // ------------------------------------------------------------------ HrmTextModel

    /** The per-stack block count of the HRM text model: {@code num_layers_per_stack} else {@code num_hidden_layers}. */
    private static int layersPerStack(HfConfig cfg) {
        int stack = cfg.findInt("num_layers_per_stack").orElseGet(() -> cfg.requireInt("num_hidden_layers"));
        if (stack <= 0) {
            throw new GgufFormatException(cfg.source() + ": the HRM stack count must be positive, got " + stack);
        }
        return stack;
    }

    /**
     * {@code HrmTextModel.__init__} (hrm_text.py:26-33): the expanded cache-slot block count
     * {@code layers_per_stack * H_cycles * (L_cycles + 1)}; the checkpoint file holds only the
     * {@code 2 * layers_per_stack} physical blocks (hrm_text.py:35).
     */
    public static int hrmTextBlockCount(HfConfig cfg) {
        int stack = layersPerStack(cfg);
        int h = requireCycles(cfg, "H_cycles");
        int l = requireCycles(cfg, "L_cycles");
        long total = (long) stack * h * (l + 1L);
        if (total > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": block count " + total + " is outside [0, "
                    + HfConfig.MAX_LAYERS + ")");
        }
        return (int) total;
    }

    private static int requireCycles(HfConfig cfg, String where) {
        int v = cfg.findInt(where).orElseGet(() -> cfg.findInt(where.toLowerCase()).orElse(0));
        if (v <= 0) {
            throw new GgufFormatException(cfg.source() + ": the HRM '" + where + "' count must be positive, got " + v);
        }
        return v;
    }

    /** A required float of {@code hparams} ({@code hparams[name]} fails the load when absent). */
    private static double requireDouble(HfConfig cfg, String name) {
        return cfg.findFloat(name).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key '" + name + "' is missing"));
    }

    /**
     * {@code HrmTextModel.set_gguf_parameters} (hrm_text.py:37-46): the generic keys, the rope dimension of the
     * head size, the embedding scale and the hrm passes.
     */
    public static void hrmTextMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, HfDraftHooksA.headDimOf(cfg));
        kv.set(GgufKeys.EMBEDDING_SCALE, requireDouble(cfg, "embedding_scale"));
        kv.set(GgufKeys.HRM_LAYERS_PER_STACK, layersPerStack(cfg));
        kv.set(GgufKeys.HRM_H_CYCLES, requireCycles(cfg, "H_cycles"));
        kv.set(GgufKeys.HRM_L_CYCLES, requireCycles(cfg, "L_cycles"));
        kv.set(GgufKeys.HRM_PREFIX_LM, cfg.findBool(false, "prefix_lm"));
    }

    /**
     * {@code HrmTextModel.modify_tensors} (hrm_text.py:48-79): the embedding, the output head and
     * {@code model.z_L_init} map directly; every {@code model.{L,H}_module.layers.<i>.<rest>} tensor is re-numbered
     * ({@code H} behind the {@code L} stack); the fused {@code attn.gqkv_proj} splits into gate/q/k/v chunk 4 and
     * {@code mlp.gate_up_proj} into gate/up chunk 2; the rest keeps the converter's {@code attn.} →
     * {@code self_attn.} rewrite and the parent mapping.
     */
    public static void hrmTextTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.equals("model.embed_tokens.weight") || hfName.equals("lm_head.weight")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        if (hfName.equals("model.z_L_init")) {
            out.add(ctx.names().ggufName("HRM_Z_L_INIT", -1, ""), src);
            return;
        }
        java.util.regex.Matcher m = HRM_LAYER.matcher(hfName);
        if (!m.matches()) {
            throw new GgufFormatException("can not map tensor: " + hfName);
        }
        boolean high = "H".equals(m.group(1));
        int i = Integer.parseInt(m.group(2));
        String rest = m.group(3);
        int layerIdx = i + (high ? layersPerStack(ctx.config()) : 0);
        if (layerIdx < 0 || layerIdx >= ctx.blockCount()) {
            throw new GgufFormatException("tensor '" + hfName + "': layer " + layerIdx
                    + " is outside the " + ctx.blockCount() + " physical blocks");
        }
        if (rest.equals("attn.gqkv_proj.weight")) {
            chunk(ctx, out, layerIdx, src, 4, "ATTN_GATE", "ATTN_Q", "ATTN_K", "ATTN_V");
            return;
        }
        if (rest.equals("mlp.gate_up_proj.weight")) {
            chunk(ctx, out, layerIdx, src, 2, "FFN_GATE", "FFN_UP");
            return;
        }
        String name = rest.startsWith("attn.") ? "self_attn." + rest.substring("attn.".length()) : rest;
        HfTensorHook.DEFAULT.apply(ctx, "model.layers." + layerIdx + "." + name, src, out);
    }

    /** Splits {@code src} into {@code names.length} parts along dim 0 and maps them to GGUF block tensors. */
    private static void chunk(HfLoadContext ctx, TensorSink out, int bid, RowSource src, int parts, String... names)
            throws IOException {
        for (int i = 0; i < names.length; i++) {
            out.add(ctx.names().ggufName(names[i], bid, ".weight"), RowTransforms.chunkRows(src, parts, i));
        }
    }

    private static final java.util.regex.Pattern HRM_LAYER =
            java.util.regex.Pattern.compile("model\\.([LH])_module\\.layers\\.(\\d+)\\.(.+)");

    /** {@code HrmTextModel.set_vocab} (hrm_text.py:32-33): the GPT-2 vocabulary. */
    public static Map<String, Object> hrmTextVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    // ------------------------------------------------------------------ NanbeigeModel

    /**
     * {@code NanbeigeModel.set_gguf_parameters} (nanbeige.py:10-24) over the Llama parent: the loop passes and the
     * loop-final norm flag.
     */
    public static void nanbeigeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfLlamaHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        long loops = cfg.findLong(1, "num_loops");
        kv.set(GgufKeys.NUM_LOOPS, (int) Math.max(1, loops));
        kv.set(GgufKeys.SKIP_LOOP_FINAL_NORM, cfg.findBool(false, "skip_loop_final_norm"));
    }
}
