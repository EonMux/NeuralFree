package com.eonmux.neuralfree.gguf.hf.hooks;

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
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Hooks of the llama.cpp classic-dense converter classes of Appendix A table HK-D1, part E: {@code InternLM2Model} (LC/conversion/internlm.py:18-118), {@code OrionModel} (orion.py:8-36), {@code OpenELMModel} (openelm.py:13-83), {@code PlamoModel} (plamo.py:17-71) and {@code Plamo3Model} (plamo.py:154-221), {@code PanguEmbeddedModel} (pangu.py:15-63) and {@code Phi2Model} (phi.py:18-39); the tail of part E (from {@code Spark2_5Model} to {@code GraniteSwitchModel}) lives in {@link HfDenseClassicHooksF}. <p>Every method is a public static hook reference that the table composes and subclasses reuse where the converter calls {@code super()}.</p> */
public final class HfDenseClassicHooksE {

    private HfDenseClassicHooksE() {
    }

    /** {@code self.hparams[key]} as a double, required, with a clear message. */
    static double requireDouble(HfConfig cfg, String key) {
        return cfg.findFloat(key).orElseThrow(() -> new GgufFormatException(cfg.source() + ": required key '" + key
                + "' is missing"));
    }

    /** The {@code _set_vocab_sentencepiece} vocabulary (base.py:2296-2396): model {@code llama}, pre {@code default}, the SentencePiece token list sized by {@code vocab_size}. @throws GgufFormatException when the checkpoint has no {@code tokenizer.model} */
    public static Map<String, Object> sentencePieceVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
    }

    /** {@code InternLM2Model.set_gguf_parameters}: the class overrides no parameter, so the inherited keys. */
    public static void internlm2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
    }

    /** {@code InternLM2Model.set_vocab} (internlm.py:30-116): the SentencePiece vocabulary with its fixups — the {@code \x00} piece is replaced by the dragon character (internlm.py:49-52), and the chat model's {@code <|im_end|>} token replaces the special-vocabulary eos id (internlm.py:104-111). @throws GgufFormatException when the checkpoint has no {@code tokenizer.model} */
    public static Map<String, Object> internlm2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx),
                pre));
        Object tokensObj = md.get(GgufKeys.TOKENIZER_LIST);
        if (tokensObj instanceof List<?>) {
            List<String> tokens = new ArrayList<>();
            int imEnd = -1;
            for (Object t : (List<?>) tokensObj) {
                String token = String.valueOf(t);
                if ("\u0000".equals(token)) {
                    token = "\ud83d\udc09";
                }
                if ("<|im_end|>".equals(token) && imEnd < 0) {
                    imEnd = tokens.size();
                }
                tokens.add(token);
            }
            md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(tokens));
            if (imEnd >= 0) {
                md.put(GgufKeys.TOKENIZER_EOS_ID, imEnd);
            }
        }
        return Collections.unmodifiableMap(md);
    }

    /** {@code InternLM2Model.modify_tensors} (internlm.py:119-141): a block's fused {@code attention.wqkv} splits into {@code [num_groups, q_per_kv + 2, headDim, ...]} along dim 0 — the first {@code q_per_kv} slices are the query heads, the last two the key and value slices (internlm.py:128-132) — and the query and key weights take the Llama permutation ({@code LlamaModel.permute}, internlm.py:135-136); every other tensor passes on. @throws GgufFormatException when {@code num_attention_heads} is not a multiple of {@code num_key_value_heads} */
    public static void internlm2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && hfName.contains(".attention.wqkv")) {
            HfConfig cfg = ctx.config();
            int heads = cfg.requireInt("num_attention_heads");
            int kvHeads = cfg.requireInt("num_key_value_heads");
            if (heads % kvHeads != 0) {
                throw new GgufFormatException(cfg.source() + ": " + heads + " query heads are not a multiple of "
                        + kvHeads + " key/value heads");
            }
            int qPerKv = heads / kvHeads;
            int headDim = cfg.requireInt("hidden_size") / heads;
            int[] qIdx = new int[heads];
            int[] kIdx = new int[kvHeads];
            int[] vIdx = new int[kvHeads];
            for (int o = 0; o < heads; o++) {
                int g = o / qPerKv;
                qIdx[o] = g * (qPerKv + 2) + o % qPerKv;
            }
            for (int g = 0; g < kvHeads; g++) {
                kIdx[g] = g * (qPerKv + 2) + qPerKv;
                vIdx[g] = g * (qPerKv + 2) + qPerKv + 1;
            }
            out.add(ctx.names().ggufName("ATTN_Q", bid, ".weight"),
                    RowTransforms.permuteQk(RowTransforms.gatherRows(src, headDim, qIdx), heads, heads));
            out.add(ctx.names().ggufName("ATTN_K", bid, ".weight"),
                    RowTransforms.permuteQk(RowTransforms.gatherRows(src, headDim, kIdx), heads, kvHeads));
            out.add(ctx.names().ggufName("ATTN_V", bid, ".weight"), RowTransforms.gatherRows(src, headDim, vIdx));
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code OrionModel.set_gguf_parameters} (orion.py:14-36), hand-set: the context length from {@code max_sequence_length | max_position_embeddings | model_max_length}, the width, block count, 1:1 FFN, the per-head key/value counts and the {@code rms_norm_eps} stored as a layer norm epsilon, with the {@code Meta AI original pth} tensor data layout. @throws GgufFormatException when none of the context keys is present */
    public static void orionMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int ctxLen = cfg.findInt("max_sequence_length", "max_position_embeddings", "model_max_length")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": can not find ctx length parameter"));
        kv.set(GgufKeys.CONTEXT_LENGTH, ctxLen);
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.findInt(heads, "num_key_value_heads"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, requireDouble(cfg, "rms_norm_eps"));
        kv.set(GgufKeys.TENSOR_DATA_LAYOUT, "Meta AI original pth");
    }

    /** The FFN widths of {@code OpenELMModel.__init__} (openelm.py:24-36): every entry of {@code ffn_multipliers} times the model width, made divisible like {@code _make_divisible} (openelm.py:17-21). */
    public static List<Integer> openelmFfnDims(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int modelDim = cfg.requireInt("model_dim");
        int divisor = cfg.requireInt("ffn_dim_divisor");
        Object flat = cfg.flat().get("ffn_multipliers");
        if (!(flat instanceof List<?>)) {
            throw new GgufFormatException(cfg.source() + ": required key 'ffn_multipliers' is missing");
        }
        List<Integer> dims = new ArrayList<>();
        for (Object m : (List<?>) flat) {
            dims.add((int) makeDivisible(((Number) m).doubleValue() * modelDim, divisor));
        }
        return Collections.unmodifiableList(dims);
    }

    /** {@code OpenELMModel._make_divisible} (openelm.py:17-21): round to a multiple of {@code divisor}, and never round down by more than 10 percent. */
    static long makeDivisible(double v, int divisor) {
        long newV = Math.max(divisor, (long) (v + divisor / 2.0) / divisor * divisor);
        if (newV < 0.9 * v) {
            newV += divisor;
        }
        return newV;
    }

    /** {@code OpenELMModel.set_gguf_parameters} (openelm.py:45-59), hand-set: the block, context, model width, the per-layer FFN, query head and key/value head arrays, the {@code rope_freq_constant} RoPE frequency base, a fixed 1e-6 rms epsilon and the full-width RoPE, key and value lengths of {@code head_dim}. */
    public static void openelmMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int headDim = cfg.requireInt("head_dim");
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.requireInt("max_context_length"));
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("model_dim"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, openelmFfnDims(ctx));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.findIntList("num_query_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, cfg.findIntList("num_kv_heads"));
        kv.set(GgufKeys.ROPE_FREQ_BASE, requireDouble(cfg, "rope_freq_constant"));
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, 1e-6);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, headDim);
        kv.set(GgufKeys.ATTENTION_KEY_LENGTH, headDim);
        kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, headDim);
    }

    /** {@code OpenELMModel.set_vocab} (openelm.py:38-43): the SentencePiece vocabulary, or the builtin {@code llama-spm} one over {@code vocab_size} when the tokenizer files are absent (openelm.py:45). */
    public static Map<String, Object> openelmVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (HfVocabWriters.canSentencePiece(HfHookSupport.vocabInput(ctx))) {
            return HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
        }
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        return HfVocabWriters.builtin(in, "llama-spm", ctx.config().requireInt("vocab_size"));
    }

    /** {@code OpenELMModel.modify_tensors} (openelm.py:63-76): a block's {@code ffn.proj_1} weight splits into the gate part ({@link RowTransforms#sliceRows}, openelm.py:66-69); every other tensor passes on. @throws GgufFormatException when the configuration has no per-layer FFN width for the block */
    public static void openelmTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        int bid = HfLoadContext.bidOf(hfName);
        if (bid >= 0 && hfName.equals("transformer.layers." + bid + ".ffn.proj_1.weight")) {
            int ffnDim = openelmFfnDims(ctx).get(bid);
            out.add(ctx.names().ggufName("FFN_GATE", bid, ".weight"), RowTransforms.sliceRows(src, 0, ffnDim));
            out.add(ctx.names().ggufName("FFN_UP", bid, ".weight"),
                    RowTransforms.sliceRows(src, ffnDim, src.rows() - ffnDim));
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code PlamoModel.set_gguf_parameters} (plamo.py:23-32), hand-set: a fixed 4096 context, the width, the 1:1 FFN, the heads and a fixed 5 key/value heads (the configuration's own value is wrong, plamo.py:30), and the rms epsilon. */
    public static void plamoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.CONTEXT_LENGTH, 4096);
        kv.set(GgufKeys.EMBEDDING_LENGTH, cfg.requireInt("hidden_size"));
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, cfg.requireInt("num_attention_heads"));
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, 5);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, requireDouble(cfg, "rms_norm_eps"));
    }

    /** The GQA broadcast order of {@code shuffle_attn_q_weight} and {@code shuffle_attn_output_weight} (plamo.py:34-50): unit {@code t} of {@link RowTransforms#gatherRows} reads the head unit {@code (t % groups) * kvHeads + t / groups}. */
    private static int[] plamoShuffleOrder(int groups, int kvHeads) {
        int[] idx = new int[groups * kvHeads];
        for (int t = 0; t < idx.length; t++) {
            idx[t] = (t % groups) * kvHeads + t / groups;
        }
        return idx;
    }

    /** {@code PlamoModel.modify_tensors} (plamo.py:53-69): the attention query weights regroup the heads toward the GQA head broadcast (plamo.py:35-40) and the attention output weights regroup the columns (plamo.py:42-47); every other tensor passes on. @throws GgufFormatException when the head layout does not divide the weight */
    public static void plamoTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        int heads = cfg.requireInt("num_attention_heads");
        int headDim = cfg.requireInt("hidden_size") / heads;
        int[] order = plamoShuffleOrder(heads / 5, 5);
        if (hfName.endsWith("attn_q.weight")) {
            HfTensorHook.DEFAULT.apply(ctx, hfName, RowTransforms.gatherRows(src, headDim, order), out);
            return;
        }
        if (hfName.endsWith("attn_output.weight")) {
            int cols = heads * headDim;
            int[] colOrder = new int[cols];
            for (int c = 0; c < cols; c++) {
                colOrder[c] = order[c / headDim] * headDim + c % headDim;
            }
            HfTensorHook.DEFAULT.apply(ctx, hfName, ElementTransforms.gatherInner(src, colOrder), out);
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code Plamo3Model.set_gguf_parameters} (plamo.py:185-191): the inherited keys, then the exact {@code vocab_size}, the optional sliding window (of {@code window_size | sliding_window}) and its pattern, and the YaRN rope of the flat {@code rope_scaling_factor} (plamo.py:19-33). */
    public static void plamo3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int window = cfg.findInt("window_size", "sliding_window").orElse(0);
        if (window > 0) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, window);
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, booleanList(ctx, "sliding_window_pattern"));
        }
        double factor = cfg.findFloat(1.0, "rope_scaling_factor");
        if (factor != 1.0 && !cfg.rope().raw().containsKey("rope_type")) {
            kv.set(GgufKeys.ROPE_SCALING_TYPE, "yarn");
            kv.set(GgufKeys.ROPE_SCALING_FACTOR, factor);
            kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, cfg.requireInt("initial_context_length"));
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_FAST, 32.0);
            kv.set(GgufKeys.ROPE_SCALING_YARN_BETA_SLOW, 1.0);
            kv.set(GgufKeys.ROPE_SCALING_YARN_TRUNCATE, Boolean.FALSE);
        }
    }

    /** The boolean list of a configuration key, read from its flat values. */
    private static List<Boolean> booleanList(HfLoadContext ctx, String key) {
        Object flat = ctx.config().flat().get(key);
        if (!(flat instanceof List<?>)) {
            throw new GgufFormatException(ctx.config().source() + ": required key '" + key + "' is missing");
        }
        List<Boolean> out = new ArrayList<>();
        for (Object t : (List<?>) flat) {
            out.add(t instanceof Boolean ? (Boolean) t : Boolean.parseBoolean(String.valueOf(t)));
        }
        return Collections.unmodifiableList(out);
    }

    /** {@code Plamo3Model.set_vocab} (plamo.py:168-183): the PLaMo vocabulary with the {@code <|plamo:tag|>} end of text token and the {@code <|plamo:begin_} / {@code <|plamo:end_} / {@code :plamo|>} ordinary-token parts, then the chat template of {@code tokenizer_config.json} or {@code chat_template.jinja}. */
    public static Map<String, Object> plamo3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.plamo(HfHookSupport.vocabInput(ctx),
                "<|plamo:tag|>", Arrays.asList("<|plamo:begin_", "<|plamo:end_", ":plamo|>")));
        md.putAll(HfChatTemplates.resolve(ctx.tokenizerFiles(), null));
        return Collections.unmodifiableMap(md);
    }

    /** {@code Plamo3Model.modify_tensors} (plamo.py:193-214): the mixer norm offsets — {@code pre_mixer_norm} +1, {@code post_mixer_norm} +1/5, {@code pre_mlp_norm} +1, {@code post_mlp_norm} +1/5<sup>1.5</sup>, the mixer q/k norms +1, and any remaining {@code .norm.weight} +1 (plamo.py:196-212) — then the inherited mapping. */
    public static void plamo3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        RowSource data;
        if (hfName.endsWith(".pre_mixer_norm.weight") || hfName.endsWith(".pre_mlp_norm.weight")) {
            data = ElementTransforms.addScalar(src, 1.0f);
        } else if (hfName.endsWith(".post_mixer_norm.weight")) {
            data = ElementTransforms.addScalar(src, 1.0f / 5);
        } else if (hfName.endsWith(".post_mlp_norm.weight")) {
            data = ElementTransforms.addScalar(src, (float) (1.0 / Math.pow(5, 1.5)));
        } else if (hfName.endsWith(".mixer.q_norm.weight") || hfName.endsWith(".mixer.k_norm.weight")
                || hfName.endsWith(".norm.weight")) {
            data = ElementTransforms.addScalar(src, 1.0f);
        } else {
            data = src;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, data, out);
    }

    /** {@code PanguEmbeddedModel.set_gguf_parameters} (pangu.py:29-47): the inherited keys, then the exact {@code vocab_size}, the RoPE width {@code head_dim | hidden_size / num_attention_heads}, and, when the configuration has no {@code head_dim}, that width as key and value length. */
    public static void panguMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfGenericMetadata.apply(ctx, kv);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        int ropeDim = cfg.findInt("head_dim")
                .orElse(cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads"));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDim);
        if (!cfg.findInt("head_dim").isPresent()) {
            kv.set(GgufKeys.ATTENTION_KEY_LENGTH, ropeDim);
            kv.set(GgufKeys.ATTENTION_VALUE_LENGTH, ropeDim);
        }
    }

    /** {@code PanguEmbeddedModel.set_vocab}'s {@code add_prefix_space} (pangu.py:20-39): the SentencePiece vocabulary, then the tokenizer config's flag as {@code add_space_prefix}. */
    public static Map<String, Object> panguVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(sentencePieceVocab(ctx, pre));
        Object addPrefix = ctx.tokenizerFiles().readTokenizerConfig().raw("add_prefix_space");
        if (addPrefix instanceof Boolean) {
            md.put(GgufKeys.TOKENIZER_ADD_PREFIX, addPrefix);
        }
        return Collections.unmodifiableMap(md);
    }

    /** {@code PanguEmbeddedModel.modify_tensors} (pangu.py:51-57): a tied {@code lm_head.weight} is skipped (pangu.py:52-54); every other tensor passes on. */
    public static void panguTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        if ("lm_head.weight".equals(hfName) && ctx.config().findBool(false, "tie_word_embeddings")) {
            out.drop(hfName, "tied to token_embd");
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    /** {@code Phi2Model.set_gguf_parameters} (phi.py:22-38), hand-set: the {@code n_positions | max_position_embeddings} context, the {@code hidden_size | n_embd} width and 4x FFN, the heads over the same count as key/value heads, the {@code layer_norm_epsilon | layer_norm_eps} epsilon, the {@code partial_rotary_factor}-scaled RoPE width, and no BOS token. */
    public static void phi2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int hidden = cfg.findInt("hidden_size", "n_embd").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'hidden_size' or 'n_embd' is missing"));
        int heads = cfg.findInt("num_attention_heads", "n_head").orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'num_attention_heads' or 'n_head' is missing"));
        double partial = cfg.rope().global().partialRotaryFactor().orElseThrow(() ->
                new GgufFormatException(cfg.source() + ": required key 'partial_rotary_factor' is missing"));
        kv.set(GgufKeys.CONTEXT_LENGTH, cfg.findInt("n_positions", "max_position_embeddings")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'n_positions' or 'max_position_embeddings' is missing")));
        kv.set(GgufKeys.EMBEDDING_LENGTH, hidden);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, 4 * hidden);
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, heads);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, heads);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, cfg.findFloat("layer_norm_epsilon", "layer_norm_eps")
                .orElseThrow(() -> new GgufFormatException(cfg.source()
                        + ": required key 'layer_norm_epsilon' or 'layer_norm_eps' is missing")));
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, (int) (partial * hidden) / heads);
        kv.set(GgufKeys.TOKENIZER_ADD_BOS, false);
    }

    /** The weight-layer count of the tensor names, per granite.py:242-259; forwarded to {@link HfDenseClassicHooksF}. */
    static int graniteSwitchWeightLayers(List<String> names, HfLoadContext ctx) {
        return HfDenseClassicHooksF.graniteSwitchWeightLayers(names, ctx);
    }
}