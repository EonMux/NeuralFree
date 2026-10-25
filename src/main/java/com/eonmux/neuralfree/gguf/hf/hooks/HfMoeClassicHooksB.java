package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
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
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.function.UnaryOperator;

/**
 * Hooks of the classic MoE converter classes, part B: {@code DeepseekModel} (LC/conversion/deepseek.py:141-222),
 * the Bailing MoE classes and {@code SarvamMoEModel} (bailingmoe.py:15-219), {@code Dots1Model} (dots1.py:15-33)
 * and {@code Ernie4_5MoeModel} (ernie.py:78-157). The shared collector helpers, the GPT-2 vocabulary and the rope
 * width default live in {@link HfMoeClassicHooksA}.
 */
public final class HfMoeClassicHooksB {

    /**
     * The filter of {@code BailingMoeV2Model} (bailingmoe.py:138-141, and inherited by Sarvam): the router bias
     * {@code .expert_bias} gains its {@code .bias} suffix so it maps.
     */
    public static final HfNameFilter BAILING_V2_FILTER =
            (name, base) -> base.apply(name.replace(".expert_bias", ".expert_bias.bias"));

    private static final String WEIGHT = ".weight";
    private static final String DEEPSEEK_STATE = HfMoeClassicHooksB.class.getName() + ".deepseek";
    private static final String BAILING_STATE = HfMoeClassicHooksB.class.getName() + ".bailing";
    private static final String DOTS1_STATE = HfMoeClassicHooksB.class.getName() + ".dots1";
    private static final String ERNIE_STATE = HfMoeClassicHooksB.class.getName() + ".ernie";

    private HfMoeClassicHooksB() {
    }

    // ------------------------------------------------------------------ DeepseekModel (v1)

    /**
     * {@code DeepseekModel.set_gguf_parameters} (deepseek.py:150-163) after the generic metadata: the RoPE dimension
     * count, the explicit NONE scaling, the leading dense block count, the vocabulary size, the expert feed-forward
     * length, the fixed 1.0 expert weight scale and the routed and shared expert counts.
     */
    public static void deepseekMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, HfMoeClassicHooksA.ropeWidth(cfg));
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, 1.0);
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("n_routed_experts"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("n_shared_experts"));
    }

    /**
     * {@code DeepseekModel.modify_tensors} (deepseek.py:175-213): the Llama Q/K permutation, then the stacking of
     * the per-expert {@code mlp.experts.<x>.down|gate|up_proj} of a layer under the merged same-prefix names
     * ({@link #deepseekExtra} fails on leftovers).
     */
    public static void deepseekTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = HfLlamaHooks.permuteQk(ctx.config(), hfName, src);
        if (hfName.contains("mlp.experts")) {
            HfConfig cfg = ctx.config();
            HfHookSupport.ExpertCollector experts = HfMoeClassicHooksA.stackedExperts(ctx, DEEPSEEK_STATE,
                    cfg.requireInt("n_routed_experts"), HfMoeClassicHooksA.GATE_UP_DOWN);
            if (!experts.offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not a Deepseek expert weight"
                        + " (mlp.experts.<x>.down_proj|gate_proj|up_proj)");
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), data);
    }

    /** {@code DeepseekModel.prepare_tensors} (deepseek.py:215-222): leftover experts are invalid. */
    public static void deepseekExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, DEEPSEEK_STATE, "mlp.experts");
    }

    /**
     * {@code DeepseekModel.set_vocab} (deepseek.py:144-148): the SentencePiece vocabulary, or GPT-2 BPE when the
     * checkpoint has no SentencePiece model.
     */
    public static Map<String, Object> deepseekVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        if (files.file(HfTokenizerFiles.TOKENIZER_MODEL) != null) {
            return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
        }
        return HfMoeClassicHooksA.gpt2Vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ BailingMoe family

    /** {@code BailingMoeModel.set_vocab} (bailingmoe.py:18-19, 122-123): the GPT-2 BPE vocabulary. */
    public static Map<String, Object> bailingMoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfMoeClassicHooksA.gpt2Vocab(ctx, pre);
    }

    /**
     * {@code BailingMoeV2Model.__init__} (bailingmoe.py:118): the extended block count
     * {@code num_hidden_layers + num_nextn_predict_layers} when the checkpoint counts NextN layers.
     */
    public static int nextnPredictLayers(HfConfig cfg) {
        OptionalInt n = cfg.findInt("num_nextn_predict_layers");
        return n.isPresent() && n.getAsInt() > 0 ? n.getAsInt() : 0;
    }

    /** {@code BailingMoeModel.set_gguf_parameters} (bailingmoe.py:21-33) after the generic metadata. */
    public static void bailingMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, HfMoeClassicHooksA.ropeWidth(cfg));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, 1.0);
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("num_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
    }

    /**
     * {@code BailingMoeModel.modify_tensors} (bailingmoe.py:45-99): {@code attention.dense.weight} maps to
     * {@code ATTN_OUT}; the fused {@code query_key_value.weight} splits into q, k and v row groups
     * {@code [n_head * head_dim, n_kv * head_dim, n_kv * head_dim]} with q and k through the llama.cpp permutation;
     * per-expert tensors stack under the merged names; the output weight is L2 normalized along dim 0 when
     * {@code norm_head}.
     */
    public static void bailingMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int bid = HfHookSupport.layerOf(hfName);
        if (hfName.endsWith("attention.dense.weight")) {
            out.add(ctx.names().ggufName("ATTN_OUT", bid, WEIGHT), src);
            return;
        }
        if (hfName.endsWith("query_key_value.weight")) {
            splitQkv(ctx, cfg, hfName, src, bid, out);
            return;
        }
        if (hfName.contains("mlp.experts")) {
            HfHookSupport.ExpertCollector experts = HfMoeClassicHooksA.stackedExperts(ctx, BAILING_STATE,
                    cfg.requireInt("num_local_experts", "num_experts"), HfMoeClassicHooksA.GATE_UP_DOWN);
            if (!experts.offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not a BailingMoe expert weight"
                        + " (mlp.experts.<x>.down_proj|gate_proj|up_proj)");
            }
            return;
        }
        boolean normalize = cfg.findBool(false, "norm_head")
                && (hfName.equals("output.weight") || hfName.equals("lm_head.weight"));
        out.add(ctx.names().mapOrThrow(hfName), normalize ? normalizeOutput(hfName, src) : src);
    }

    /** The fused-QKV branch (bailingmoe.py:57-62) of {@link #bailingMoeTensors}. */
    private static void splitQkv(HfLoadContext ctx, HfConfig cfg, String hfName, RowSource src, int bid,
                                 TensorSink out) {
        int heads = cfg.requireInt("num_attention_heads");
        int kvHeads = cfg.findInt(heads, "num_key_value_heads");
        if (kvHeads <= 0 || heads % kvHeads != 0) {
            throw new GgufFormatException(cfg.source() + ": 'num_key_value_heads' must divide"
                    + " 'num_attention_heads'");
        }
        int hd = HfMoeClassicHooksA.ropeWidth(cfg);
        long rows = product(src.shape()) / (src.shape().length > 1 ? src.shape()[src.shape().length - 1] : 1);
        if (rows != (long) (heads + 2L * kvHeads) * hd) {
            throw new GgufFormatException("tensor '" + hfName + "' does not split into [" + heads + " * " + hd
                    + ", " + kvHeads + " * " + hd + ", " + kvHeads + " * " + hd + "]");
        }
        RowSource q = RowTransforms.permuteQk(RowTransforms.sliceRows(src, 0, (long) heads * hd), heads, heads);
        RowSource k = RowTransforms.permuteQk(
                RowTransforms.sliceRows(src, (long) heads * hd, (long) kvHeads * hd), heads, kvHeads);
        RowSource v = RowTransforms.sliceRows(src, (long) (heads + kvHeads) * hd, (long) kvHeads * hd);
        out.add(ctx.names().ggufName("ATTN_Q", bid, WEIGHT), q);
        out.add(ctx.names().ggufName("ATTN_K", bid, WEIGHT), k);
        out.add(ctx.names().ggufName("ATTN_V", bid, WEIGHT), v);
    }

    /** {@code BailingMoeModel.prepare_tensors} (bailingmoe.py:101-108): leftover experts are invalid. */
    public static void bailingMoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, BAILING_STATE, "mlp.experts");
    }

    /**
     * {@code BailingMoeModel.modify_tensors}' {@code norm_head} branch (bailingmoe.py:95-97) in F32: each column of
     * the 2-D output weight is divided by its L2 norm over dim 0 plus {@code 1e-7}; the norms are read once and the
     * column factors applied row by row.
     */
    private static RowSource normalizeOutput(String hfName, RowSource src) throws IOException {
        long[] shape = src.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("tensor '" + hfName + "': the output weight must be 2-D, got "
                    + Arrays.toString(shape));
        }
        int cols = (int) shape[1];
        int rows = (int) shape[0];
        if (cols < 1 || cols > HfPlanView.MAX_READ_ELEMENTS) {
            throw new GgufFormatException("tensor '" + hfName + "': " + cols + " output columns do not fit the"
                    + " [1, " + HfPlanView.MAX_READ_ELEMENTS + "] read window");
        }
        RowSource values = ElementTransforms.convert(src, GGMLType.F32);
        ByteBuffer b = ByteBuffer.allocate(rows * cols * 4).order(ByteOrder.LITTLE_ENDIAN);
        readF32Rows(values, rows, b);
        b.flip();
        float[] factors = new float[cols];
        for (int c = 0; c < cols; c++) {
            double sum = 0;
            for (int r = 0; r < rows; r++) {
                double v = b.getFloat((r * cols + c) * 4);
                sum += v * v;
            }
            factors[c] = (float) (1.0 / (Math.sqrt(sum) + 1e-7));
        }
        return ElementTransforms.mulColsBy(src, factors);
    }

    private static void readF32Rows(RowSource src, int rows, ByteBuffer b) throws IOException {
        int rowBytes = (int) src.rowBytes();
        int chunk = Math.max(1, (1 << 20) / Math.max(1, rowBytes));
        for (int r = 0; r < rows; r += chunk) {
            src.readRows(r, Math.min(chunk, rows - r), b);
        }
    }

    private static long product(long[] shape) {
        long n = 1;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    /** {@code BailingMoeV2Model.set_gguf_parameters} (bailingmoe.py:125-141) after the generic metadata. */
    public static void bailingMoeV2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (HfMoeClassicHooksA.ropeWidth(cfg) * partialRotaryFactor(cfg)));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, cfg.requireInt("moe_intermediate_size"));
        int sharedFf = cfg.findInt("moe_shared_expert_intermediate_size")
                .orElseGet(() -> cfg.requireInt("moe_intermediate_size") * cfg.requireInt("num_shared_experts"));
        kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, sharedFf);
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0f, "routed_scaling_factor"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("num_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
        OptionalInt nextn = cfg.findInt("num_nextn_predict_layers");
        if (nextn.isPresent()) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, nextn.getAsInt());
        }
    }

    /** The partial-rotary factor of {@code rope_parameters}, default 0.5 (bailingmoe.py:131). */
    private static double partialRotaryFactor(HfConfig cfg) {
        Object v = cfg.rope().global().raw().get("partial_rotary_factor");
        if (v == null) {
            return 0.5;
        }
        if (!(v instanceof Number) || v instanceof Boolean) {
            throw new GgufFormatException(cfg.source() + ": rope_parameters.partial_rotary_factor must be a number"
                    + " value");
        }
        return ((Number) v).doubleValue();
    }

    /**
     * {@code BailingMoeV2Model.modify_tensors} (bailingmoe.py:146-181): per-expert tensors stack under the merged
     * names; the router bias arrived with its {@code .bias} suffix from the filter; everything else maps directly.
     * The V2 checkpoints neither fuse QKV rows, invert the output nor normalize the bias (Sarvam alone does, see
     * {@link #sarvamMoeTensors}).
     */
    public static void bailingMoeV2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("mlp.experts")) {
            HfHookSupport.ExpertCollector experts = HfMoeClassicHooksA.stackedExperts(ctx, BAILING_STATE,
                    ctx.config().requireInt("num_local_experts", "num_experts"), HfMoeClassicHooksA.GATE_UP_DOWN);
            if (!experts.offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not a BailingMoeV2 expert weight"
                        + " (mlp.experts.<x>.down_proj|gate_proj|up_proj)");
            }
            return;
        }
        out.add(ctx.names().mapOrThrow(hfName), src);
    }

    /** {@code BailingMoeV2Model.prepare_tensors} (bailingmoe.py:183-190): leftover experts are invalid. */
    public static void bailingMoeV2Extra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, BAILING_STATE, "mlp.experts");
    }

    /**
     * {@code SarvamMoEModel.set_gguf_parameters} (bailingmoe.py:201-207): {@link #bailingMoeV2Meta}, then the full
     * rotary dimension count (no partial factor) written over the V2 value.
     */
    public static void sarvamMoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        bailingMoeV2Meta(ctx, kv, plan);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, HfMoeClassicHooksA.ropeWidth(ctx.config()));
    }

    /**
     * {@code SarvamMoEModel.filter_tensors}' data change (bailingmoe.py:210-219), applied in the tensors hook since
     * the name filter carries names only: the renamed router bias {@code .expert_bias.bias} is zero-mean normalized.
     * Everything else is {@link #bailingMoeV2Tensors}.
     */
    public static void sarvamMoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.endsWith(".expert_bias.bias")) {
            out.add(ctx.names().mapOrThrow(hfName), zeroMean(hfName, src));
            return;
        }
        bailingMoeV2Tensors(ctx, hfName, src, out);
    }

    /** {@code t - t.mean()} of the 1-D router bias in F32 (bailingmoe.py:216-218). */
    private static RowSource zeroMean(String hfName, RowSource src) throws IOException {
        long[] shape = src.shape();
        long n = product(shape);
        if (n <= 0 || n > HfPlanView.MAX_READ_ELEMENTS) {
            throw new GgufFormatException("tensor '" + hfName + "': the zero-mean normalization reads the "
                    + n + "-element bias in memory");
        }
        ByteBuffer b = ByteBuffer.allocate((int) n * 4).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, (int) src.rows(), b);
        b.flip();
        int count = b.remaining() / 4;
        double sum = 0;
        for (int i = 0; i < count; i++) {
            sum += b.getFloat();
        }
        return ElementTransforms.affine(src, 1.0f, (float) (-sum / count));
    }

    // ------------------------------------------------------------------ Dots1Model

    /**
     * {@code Dots1Model.__init__} (dots1.py:19-20): the alias {@code num_experts} receives
     * {@code n_routed_experts}, as the Qwen2Moe helpers read only those two.
     *
     * @throws GgufFormatException when {@code n_routed_experts} is absent
     */
    public static HfConfig dots1Fixup(HfConfig cfg) {
        Map<String, Object> inject = new LinkedHashMap<>();
        inject.put("num_experts", cfg.requireInt("n_routed_experts"));
        return cfg.withFlat(inject);
    }

    /**
     * {@code Dots1Model.set_gguf_parameters} (dots1.py:22-27) after {@code Qwen2MoeModel.set_gguf_parameters}:
     * the leading dense count, the shared-expert count, the routed weight scale and the weight norm flag.
     */
    public static void dots1Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfQwenMoeHooks.qwen2MoeMeta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("first_k_dense_replace"));
        kv.set(GgufKeys.EXPERT_SHARED_COUNT, cfg.requireInt("n_shared_experts"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_SCALE, cfg.findFloat(1.0f, "routed_scaling_factor"));
        kv.set(GgufKeys.EXPERT_WEIGHTS_NORM, cfg.findBool(false, "norm_topk_prob"));
    }

    /**
     * {@code Dots1Model.modify_tensors} (dots1.py:29-33): the shared experts map by their own names first, then
     * everything else through {@code Qwen2MoeModel.modify_tensors}.
     */
    public static void dots1Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("shared_experts")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
            return;
        }
        HfQwenMoeHooks.qwen2MoeTensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Ernie4_5MoeModel

    /** The mtp tensor skip of {@code Ernie4_5MoeModel.filter_tensors} (ernie.py:100-121), then the Ernie fixup. */
    public static String ernie45MoeFilter(String name, UnaryOperator<String> base) {
        if (name.startsWith("model.mtp_block.") || name.startsWith("model.mtp_emb_norm.")
                || name.startsWith("model.mtp_hidden_norm.") || name.startsWith("model.mtp_linear_proj.")) {
            return null;
        }
        return HfBaseDenseHooks.ernie45Filter(name, base);
    }

    /**
     * {@code Ernie4_5MoeModel.set_gguf_parameters} (ernie.py:86-97) after {@code Ernie4_5Model.set_gguf_parameters}:
     * the expert count and used count, the interleave step, the leading dense index, the expert feed-forward length,
     * the shared-expert count and the shared width divided by {@code num_key_value_heads} as the converter writes it
     * (ernie.py:96-97, reproduced as written).
     */
    public static void ernie45MoeMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.EXPERT_COUNT, cfg.requireInt("moe_num_experts"));
        kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_k"));
        kv.set(GgufKeys.INTERLEAVE_MOE_LAYER_STEP, cfg.requireInt("moe_layer_interval"));
        kv.set(GgufKeys.LEADING_DENSE_BLOCK_COUNT, cfg.requireInt("moe_layer_start_index"));
        OptionalInt moeFf = cfg.findInt("moe_intermediate_size");
        if (moeFf.isPresent()) {
            kv.set(GgufKeys.EXPERT_FEED_FORWARD_LENGTH, moeFf.getAsInt());
        }
        OptionalInt sharedCount = cfg.findInt("moe_num_shared_experts");
        OptionalInt sharedFf = cfg.findInt("intermediate_size");
        OptionalInt kvHeads = cfg.findInt("num_key_value_heads");
        if (sharedCount.isPresent() && sharedCount.getAsInt() > 0 && sharedFf.isPresent() && kvHeads.isPresent()) {
            kv.set(GgufKeys.EXPERT_SHARED_FEED_FORWARD_LENGTH, sharedFf.getAsInt() / kvHeads.getAsInt());
        }
    }

    /**
     * {@code Ernie4_5MoeModel.modify_tensors} (ernie.py:123-148): per-expert {@code mlp.experts.<x>.<part>} stack
     * under the merged same-prefix names, in the converter's gate/up/down order; everything else goes through the
     * base {@code ModelBase.modify_tensors}.
     */
    public static void ernie45MoeTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("mlp.experts")) {
            List<String> parts = List.of("gate_proj", "up_proj", "down_proj");
            if (!HfMoeClassicHooksA.stackedExperts(ctx, ERNIE_STATE, ctx.config().requireInt("moe_num_experts"),
                    parts).offer(ctx, hfName, out)) {
                throw new GgufFormatException("tensor '" + hfName + "' is not an Ernie4.5-MoE expert weight"
                        + " (mlp.experts.<x>.gate_proj|up_proj|down_proj)");
            }
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code Ernie4_5Model.set_vocab} (ernie.py:22-30): SentencePiece, then {@code add_prefix_space} honored. */
    public static Map<String, Object> ernie45MoeVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfBaseDenseHooks.ernie45Vocab(ctx, pre);
    }

    /** {@code Ernie4_5MoeModel.prepare_tensors} (ernie.py:150-157): leftover experts are invalid. */
    public static void ernie45MoeExtra(HfLoadContext ctx, TensorSink out) {
        HfMoeClassicHooksA.requireOrphans(ctx, ERNIE_STATE, "mlp.experts");
    }
}
