package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;

/**
 * Hooks of the llama.cpp classic-dense converter classes of Appendix A table HK-D1, part D: {@code Jais2Model}
 * (LC/conversion/jais.py:16-34), {@code JaisModel} (jais.py:36-107), {@code StableLMModel}
 * (stablelm.py:15-97), {@code MPTModel} (mpt.py:13-48), {@code MaincoderModel} (maincoder.py:8-13),
 * {@code MiniCPMModel} (minicpm.py:18-62), {@code NemotronModel} (nemotron.py:366-402) and
 * {@code MuseGlimmerModel} (muse_glimmer.py:28-95).
 *
 * <p>Every method is a public static hook reference that the table composes and subclasses reuse where the converter
 * calls {@code super()}.</p>
 */
public final class HfDenseClassicHooksD {

    private HfDenseClassicHooksD() {
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                + "' is missing"));
    }

    // ------------------------------------------------------------------ Jais2

    /**
     * {@code Jais2Model.set_gguf_parameters} (jais.py:19-23): the inherited keys, then the RoPE width
     * {@code head_dim | hidden_size / num_attention_heads}.
     */
    public static void jais2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                cfg.findInt(cfg.requireInt("hidden_size") / heads, "head_dim"));
    }

    // ------------------------------------------------------------------ Jais

    /** The Jais stored norm weight (output width) of an embedding-scaled projection (jais.py:101-106). */
    static final String TOKEN_EMBD_SUFFIX = "token_embd.weight";
    /** The output head the MuP width scale is applied to (jais.py:103-104). */
    static final String OUTPUT_SUFFIX = "output.weight";

    /**
     * {@code Jais2Model} uses the base class defaults; {@code JaisModel} only re-maps the FIM-free special-token
     * output ids. The GPT-2 vocabulary of {@code JaisModel.set_vocab} (jais.py:66-67).
     */
    public static Map<String, Object> jaisVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfDenseClassicHooksA.gpt2Vocab(ctx, pre);
    }

    /** The class filter of {@code JaisModel} (jais.py:77-83): the causal-attention bias is dropped. */
    public static final HfNameFilter JAIS_FILTER = (name, base) ->
            name.endsWith(".attn.bias") ? null : base.apply(name);

    /**
     * {@code JaisModel.set_gguf_parameters} (jais.py:69-75): hand-set from the GPT-2 style names, with no RoPE of
     * its own (the model uses ALiBi).
     */
    public static void jaisMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("n_positions"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("n_embd"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("n_inner"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_head"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "layer_norm_epsilon"));
    }

    /**
     * {@code JaisModel.modify_tensors} (jais.py:85-106): the relative ALiBi slopes carry
     * {@code max_alibi_bias} into the metadata through the sink (∶ref jais.py:89-97); the Conv1D weights transpose
     * last-two by the shape rule of gpt2.py; the embeddings and the output carry their MuP scales (T8's fast
     * path on block formats); everything else maps directly.
     *
     * @throws GgufFormatException when the slopes tensor is too small or the scales are absent
     */
    public static void jaisTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (hfName.endsWith("relative_pe.slopes")) {
            writeMaxAlibiBias(cfg, hfName, src, out);
            return;
        }
        RowSource data = src;
        if (hfName.endsWith(".c_attn.weight") || hfName.endsWith(".c_proj.weight")
                || hfName.endsWith(".c_fc.weight") || hfName.endsWith(".c_fc2.weight")) {
            data = HfDenseClassicHooksC.gpt2TransposeDecision(ctx, data, hfName);
        }
        if (hfName.endsWith(TOKEN_EMBD_SUFFIX)) {
            data = ElementTransforms.mulScalar(data, (float) muPScale(cfg, ctx, TOKEN_EMBD_SUFFIX));
        } else if (hfName.endsWith(OUTPUT_SUFFIX)) {
            data = ElementTransforms.mulScalar(data, (float) muPWidthScale(cfg, ctx));
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /**
     * {@code JaisModel.prepare_tensors}' {@code add_max_alibi_bias} (jais.py:100-106, 89-96): the first slope
     * value of the ALiBi tensor decides the negative head-count-scaled bias, written on the sink so the model view
     * sees it. The number of heads is rounded down to a power of two.
     */
    private static void writeMaxAlibiBias(HfConfig cfg, String hfName, RowSource src, TensorSink out)
            throws IOException {
        int heads = cfg.requireInt("n_head");
        int log2heads = 31 - Integer.numberOfLeadingZeros(heads);
        int nearest = 1 << log2heads;
        if (src.rows() < 1) {
            throw new GgufFormatException("the relative_pe.slopes tensor '" + hfName + "' is empty");
        }
        ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        src.readRows(0, 1, b);
        float first = b.getFloat(0);
        if (first > 0) {
            throw new GgufFormatException("the first ALiBi slope of '" + hfName + "' (" + first
                    + ") must be negative");
        }
        double bias = -Math.round(-Math.log(Math.abs(first)) / Math.log(2.0)) * nearest
                / Math.log(1.0) > 0 ? Math.round(Math.log(Math.abs(first)) / Math.log(2.0)) * nearest : 0.0;
        out.addKv(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, GGUFValueType.FLOAT32,
                -Math.round(Math.log(Math.abs(first)) / Math.log(2.0)) * nearest);
    }

    /** The MuP embedding scale of {@code JaisModel.__init__} (jais.py:44-51). */
    public static double muPScale(HfConfig cfg, HfLoadContext ctx, String which) {
        Object flat = cfg.flat().get("mup_embeddings_scale");
        if (flat == null) {
            flat = cfg.flat().get("embeddings_scale");
        }
        if (flat instanceof Number) {
            return ((Number) flat).doubleValue();
        }
        throw new GgufFormatException(ctx.config().source()
                + ": either 'mup_embeddings_scale' or 'embeddings_scale' is required");
    }

    /** The MuP width scale of {@code JaisModel.__init__} (jais.py:53-60). */
    public static double muPWidthScale(HfConfig cfg, HfLoadContext ctx) {
        Object alpha = cfg.flat().get("mup_output_alpha");
        Object width = cfg.flat().get("mup_width_scale");
        if (alpha instanceof Number && width instanceof Number) {
            return ((Number) alpha).doubleValue() * ((Number) width).doubleValue();
        }
        Object flat = cfg.flat().get("width_scale");
        if (flat instanceof Number) {
            return ((Number) flat).doubleValue();
        }
        throw new GgufFormatException(ctx.config().source()
                + ": either 'mup_output_alpha'/'mup_width_scale' or 'width_scale' is required");
    }

    // ------------------------------------------------------------------ StableLM

    /** The per-load per-layer Q norm buffers (stablelm.py:58-93), indexed by layer id. */
    private static final String Q_NORM_STATE = HfDenseClassicHooksD.class.getName() + ".stableQNorm";
    /** The per-load per-layer K norm buffers. */
    private static final String K_NORM_STATE = HfDenseClassicHooksD.class.getName() + ".stableKNorm";

    /**
     * The vocabulary of {@code StableLMModel.set_vocab} (stablelm.py:24-28): the GPT-2 tokenizer when found, else
     * the Qwen-2-styled vocabulary.
     */
    public static Map<String, Object> stablelmVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        boolean gpt2 = ctx.tokenizerFiles().tokenizerJson() != null;
        return gpt2 ? HfDenseClassicHooksA.gpt2Vocab(ctx, pre)
                : HfVocabWriters.qwen(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code StableLMModel.set_gguf_parameters} (stablelm.py:30-47): hand-set with the RoPE width of the partial
     * rotary factor, per-head key/value counts and the epsilon alias {@code layer_norm_eps|norm_eps}.
     */
    public static void stablelmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        double partial = cfg.rope().global().partialRotaryFactor().orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'partial_rotary_factor' is missing"));
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_position_embeddings"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                (int) (partial * (cfg.requireInt("hidden_size") / heads)));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.requireInt("num_key_value_heads"));
        kv.set(GgufKeys.USE_PARALLEL_RESIDUAL, cfg.findBool(true, "use_parallel_residual"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDoubleAlias(cfg, "layer_norm_eps", "norm_eps"));
    }

    /** Two-alias required string; the first present wins. */
    private static String requireString(HfConfig cfg, String a, String b) {
        return cfg.findString(a, b).orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": '" + a + "' or '" + b + "' is required"));
    }

    /**
     * {@code StableLMModel.modify_tensors} (stablelm.py:53-73): the per-head {@code q_layernorm.norms.<x>.weight}
     * and {@code k_layernorm.norms.<x>.weight} tensors stack into one per layer and head count; every other tensor
     * passes on.
     */
    public static void stablelmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        String qNormMarker = "q_layernorm.norms";
        String kNormMarker = "k_layernorm.norms";
        if (hfName.contains(qNormMarker) || hfName.contains(kNormMarker)) {
            int bid = HfLoadContext.bidOf(hfName);
            boolean q = hfName.contains(qNormMarker);
            String kind = q ? "q_layernorm" : "k_layernorm";
            int expected = q ? cfg.requireInt("num_attention_heads") : cfg.requireInt("num_key_value_heads");
            Map<Integer, List<String>> names = ctx.state(q ? Q_NORM_STATE : K_NORM_STATE, Map.class, LinkedHashMap::new);
            List<String> list = names.computeIfAbsent(bid, b -> new ArrayList<>());
            if (!list.contains(hfName)) {
                list.add(hfName);
            }
            if (list.size() < expected) {
                return;
            }
            List<RowSource> sources = new ArrayList<>(expected);
            for (int x = 0; x < expected; x++) {
                String member = "model.layers." + bid + ".self_attn." + kind + ".norms." + x + ".weight";
                sources.add(ctx.index().get(member));
            }
            String merged = "model.layers." + bid + ".self_attn." + kind + ".weight";
            out.add(ctx.names().mapOrThrow(merged), ExpertTransforms.stack(sources));
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /**
     * The {@code prepare_tensors} check of {@code StableLMModel} (stablelm.py:95-97): no per-head norm may be left
     * unstored after the checkpoint was planned.
     */
    public static void stablelmExtraChecks(HfLoadContext ctx, TensorSink out) {
        for (String key : new String[]{Q_NORM_STATE, K_NORM_STATE}) {
            Map<Integer, List<String>> state = ctx.state(key, Map.class, LinkedHashMap::new);
            for (Map.Entry<Integer, List<String>> e : state.entrySet()) {
                if (!e.getValue().isEmpty()) {
                    throw new GgufFormatException("unprocessed q/k norms of layer " + e.getKey() + ": "
                            + e.getValue());
                }
            }
        }
    }

    // ------------------------------------------------------------------ MPT

    /**
     * {@code MPTModel.set_vocab} (mpt.py:16-22): the GPT-2 tokenizer, falling back to the SentencePiece one with
     * the SEA-LION token ids (no BOS, pad 3, eos 1, unk 0), when the usual files are absent.
     */
    public static Map<String, Object> mptVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.tokenizerFiles().tokenizerJson() != null) {
            return HfDenseClassicHooksA.gpt2Vocab(ctx, pre);
        }
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(
                HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_ADD_BOS, false);
        md.put(GgufKeys.TOKENIZER_PAD_ID, 3);
        md.put(GgufKeys.TOKENIZER_EOS_ID, 1);
        md.put(GgufKeys.TOKENIZER_UNK_ID, 0);
        return md;
    }

    /**
     * {@code MPTModel.set_gguf_parameters} (mpt.py:24-38), hand-set: the {@code max_seq_len} context, the
     * {@code d_model} width and 4x FFN, the head count with the optional per-layer KV heads of {@code attn_config},
     * a fixed 1e-5 epsilon, and the optional clip over the ALiBi maximum.
     */
    public static void mptMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        com.eonmux.neuralfree.utils.json.JsonView attn = ctx.config().raw().obj("attn_config");
        double clip = cfg.findFloat(0.0, "clip_qkv");
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_seq_len"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("d_model"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * cfg.requireInt("d_model"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("n_heads"));
        if (attn.has("kv_n_heads")) {
            kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, attn.i32("kv_n_heads"));
        }
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, 1e-5);
        if (clip > 0) {
            kv.set(GgufKeys.ATTENTION_CLAMP_KQV, clip);
        }
        boolean alibi = Boolean.TRUE.equals(attn.has("alibi") ? Boolean.TRUE : Boolean.TRUE);
        if (bigBooleans(cfg, "attn_config.alibi") != null && Boolean.TRUE.equals(bigBooleans(cfg, "attn_config.alibi"))) {
            kv.set(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, cfg.findFloat(0.0, "alibi_bias_max"));
        } else {
            kv.set(GgufKeys.ATTENTION_MAX_ALIBI_BIAS, 0.0);
        }
    }

    static Boolean bigBooleans(HfConfig cfg, String path) {
        Object v = cfg.flat().get(path);
        return v instanceof Boolean ? (Boolean) v : null;
    }

    /**
     * {@code MPTModel.modify_tensors} (mpt.py:40-48): a scale tensor maps to the act-scale GGUF name through the
     * {@code .scales} suffix chain; every other tensor maps through the usual ones.
     */
    public static void mptTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if (hfName.contains("scales")) {
            String mapped = ctx.names().mapOrThrow(hfName.replace("scales", "act.scales"));
            out.add(mapped, src);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Maincoder

    /**
     * {@code MaincoderModel.set_gguf_parameters} (maincoder.py:10-13): the inherited keys, then the RoPE width
     * {@code head_dim} when the configuration carries one.
     */
    public static void maincoderMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig rope = ctx.config();
        int ropeHeads = rope.requireInt("num_attention_heads");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT,
                rope.findInt(rope.requireInt("hidden_size") / ropeHeads, "head_dim"));
    }

    // ------------------------------------------------------------------ MiniCPM

    /**
     * {@code MiniCPMModel.set_gguf_parameters} (minicpm.py:20-31): the inherited keys, then the embedding scale
     * {@code scale_emb}, the residual scale {@code scale_depth / sqrt(num_hidden_layers)} and the logit scale
     * {@code hidden_size / dim_model_base}.
     */
    public static void miniCPMMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        double embed = requireDouble(cfg, "scale_emb");
        kv.set(GgufKeys.EMBEDDING_SCALE, embed);
        kv.set(GgufKeys.RESIDUAL_SCALE, requireDouble(cfg, "scale_depth")
                / Math.pow(cfg.requireInt("num_hidden_layers"), 0.5));
        kv.set(GgufKeys.LOGIT_SCALE, cfg.requireInt("hidden_size") / (double) cfg.requireInt("dim_model_base"));
    }

    /** {@code MiniCPMModel.set_vocab} (minicpm.py:54): the SentencePiece vocabulary. */
    public static Map<String, Object> miniCPMVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.tokenizerFiles().hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(ctx.tokenizerFiles());
        }
        return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code MiniCPMModel.modify_tensors} (minicpm.py:56-63): q/k weights reverse the HF permutation like
     * {@link HfLlamaHooks#permuteQk}; every other tensor passes on.
     */
    public static void miniCPMTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        RowSource data = src;
        if (hfName.endsWith("q_proj.weight") || hfName.endsWith("k_proj.weight")) {
            int heads = cfg.requireInt("num_attention_heads");
            int kvHeads = cfg.findInt(heads, "num_key_value_heads");
            data = RowTransforms.permuteQk(data, heads, hfName.endsWith("k_proj.weight") ? kvHeads : heads);
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /**
     * {@code MiniCPMModel.generate_extra_tensors} (minicpm.py:33-52): the long and short LongRoPE factors
     * (minicpm.py:38-52), emitted when the configuration carries them.
     */
    public static void miniCPMExtra(HfLoadContext ctx, TensorSink out) throws IOException {
        com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.longropeFactors(ctx.config(),
                com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.Factors.LONG)
                .ifPresent(f -> out.add(ctx.names().ggufName("ROPE_FACTORS_LONG", -1, ".weight"), f));
        com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.longropeFactors(ctx.config(),
                com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.Factors.SHORT)
                .ifPresent(f -> out.add(ctx.names().ggufName("ROPE_FACTORS_SHORT", -1, ".weight"), f));
    }

    // ------------------------------------------------------------------ Nemotron

    /** The SentencePiece vocabulary with the Nemotron sentinel ids (nemotron.py:368-373). */
    public static Map<String, Object> nemotronVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(miniCPMVocab(ctx, pre));
        md.put(GgufKeys.TOKENIZER_PAD_ID, 0);
        md.put(GgufKeys.TOKENIZER_UNK_ID, 1);
        return md;
    }

    /**
     * {@code NemotronModel.set_gguf_parameters} (nemotron.py:375-390): the inherited keys, the exact
     * {@code vocab_size}, the epsilon alias chain, the partial-rotary RoPE width, and the linear scaling when a
     * {@code factor} is present (or the explicit none).
     */
    public static void nemotronMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDoubleAlias(cfg, "layer_norm_eps", "layer_norm_epsilon",
                "norm_epsilon", "norm_eps"));
        double partial = cfg.rope().global().partialRotaryFactor().orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'partial_rotary_factor' is missing"));
        int width = cfg.findInt(cfg.requireInt("hidden_size"), "hidden_size", "n_embd");
        int heads = cfg.findInt(cfg.requireInt("num_attention_heads"), "num_attention_heads", "n_head");
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) partial * width / heads);
        OptionalDouble factor = cfg.rope().global().factor();
        if (factor.isPresent()) {
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "linear");
        } else {
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");
        }
    }

    static double requireDoubleAlias(HfConfig cfg, String... aliases) {
        OptionalDouble v = cfg.findFloat(aliases);
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key " + Arrays.asList(aliases) + " is missing");
        }
        return v.getAsDouble();
    }

    /** The double convenience of {@link Double#compare}. */
    static double cmpDouble(double a, double b) {
        return Math.signum(a - b);
    }

    /**
     * {@code NemotronModel.modify_tensors}' layernorm1p step (nemotron.py:396-402): every named
     * {@code *norm.weight} tensor is stored plus one (the model-side epsilon shift).
     */
    public static void nemotronTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data = HfHookSupport.isNorm(hfName) ? HfHookSupport.normPlusOne(src) : src;
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    // ------------------------------------------------------------------ MuseGlimmer

    /**
     * {@code MuseGlimmerModel.set_vocab}'s eot id (muse_glimmer.py:39-45): the GPT-2 vocabulary, then the
     * {@code <|eot|>} token id resolved over the tokenizer.json vocabulary.
     */
    public static void museGlimmerEot(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson tj = ctx.tokenizerFiles().readTokenizerJson();
        for (com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson.AddedToken t : tj.addedTokens()) {
            if ("<|eot|>".equals(t.content())) {
                kv.set(GgufKeys.TOKENIZER_EOT_ID, t.id());
                return;
            }
        }
    }

    /**
     * {@code MuseGlimmerModel.set_gguf_parameters} (muse_glimmer.py:47-53): the inherited keys, then the final
     * soft-capping, the output multiplier as logit scale, the sliding window and its pattern.
     */
    public static Map<String, Object> museGlimmerVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfDenseClassicHooksA.gpt2Vocab(ctx, pre));
        return md;
    }

    public static void museGlimmerMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.FINAL_LOGIT_SOFTCAPPING, requireDouble(cfg, "final_logit_softcapping"));
        kv.set(GgufKeys.LOGIT_SCALE, requireDouble(cfg, "output_multiplier"));
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        List<Boolean> pattern = new ArrayList<>();
        Object layerTypes = cfg.flat().get("layer_types");
        for (Object t : (List<?>) layerTypes) {
            pattern.add("sliding_attention".equals(String.valueOf(t)));
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, Collections.unmodifiableList(pattern));
    }

    /**
     * {@code MuseGlimmerModel.modify_tensors} (muse_glimmer.py:55-93): the +1 of the per-layer norms
     * (muse_glimmer.py:70-72), q/k un-permuted into the GGML interleaved RoPE layout (muse_glimmer.py:14-23,
     * 77-82), and the synthesized q/k norm tensors of the q_scale factor (T9, muse_glimmer.py:84-93); every other
     * tensor passes on.
     */
    public static void museGlimmerTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        RowSource data = src;
        if (hfName.endsWith("layernorm.weight")) {
            data = ElementTransforms.addScalar(data, 1.0f);
        }
        if (hfName.contains(".self_attn.q_proj.")) {
            data = RowTransforms.rotateHalfUnpermute(data, cfg.requireInt("num_attention_heads"));
        } else if (hfName.contains(".self_attn.k_proj.")) {
            data = RowTransforms.rotateHalfUnpermute(data, cfg.requireInt("num_key_value_heads"));
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && hfName.endsWith(".self_attn.q_proj.weight")) {
            int headDim = cfg.requireInt("head_dim");
            double qScale = requireDouble(cfg, "qk_scale_factor");
            out.add(ctx.names().ggufName("ATTN_Q_NORM", bid, ".weight"),
                    RopeTensors.constant(new long[]{headDim}, (float) qScale));
            out.add(ctx.names().ggufName("ATTN_K_NORM", bid, ".weight"),
                    RopeTensors.constant(new long[]{headDim}, 1.0f));
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }
}
