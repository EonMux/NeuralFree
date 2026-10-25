package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.ExpertTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/**
 * Hooks of the BERT sub-families (LC/conversion/bert.py): {@code DistilBertModel} (:243-264),
 * {@code NomicBertModel} (:317-403, whose {@code NOMIC_BERT}/{@code NOMIC_BERT_MOE} switch is bert.py:327-328),
 * {@code NeoBert} (:406-435), {@code EuroBertModel} (:438-464), {@code JinaBertV2Model} (:570-595),
 * {@code ModernBertModel} (:598-652) and its decision checkpoint {@code ModernBertDecisionModel} (:655-747,
 * whose {@code _load_decision_hparams} loader :661-675 merges the encoder and decision configurations).
 * Every method composes {@link HfBertHooks} where the converter calls {@code super()}; the sub-families differ
 * from it in the metadata, name renames, vocabularies and — for NomicBert — the MoE expert views.
 */
public final class HfBertVariantHooks {

    /** The DistilBert layer-norm epsilon (bert.py:249). */
    public static final double DISTIL_EPS = 1e-12;
    /** The tokenizer prefix marks NomicBertModel reads (bert.py:395-403). */
    public static final String NOMIC_TOKENIZER_JSON = HfTokenizerFiles.TOKENIZER_JSON;
    /** The pre-tokenizer of a Roberta-class tokenizer.json that becomes a whitespace writer (bert.py:589). */
    public static final String WHITESPACE_PRE_TOKENIZER = "Whitespace";
    /** The {@code decision.type} of a LAYA-style decision head (bert.py:718). */
    public static final String DECISION_TYPE_LAYA = "laya";
    /** The {@code tokenizer} sub-directory of a decision checkpoint (bert.py:685-686). */
    public static final String TOKENIZER_DIR = "tokenizer";
    /** the root-side files a decision checkpoint stores its head configuration in (bert.py:655-658). */
    static final String ENCODER_CONFIG = "encoder/config.json";
    static final String RL_AGENT_CONFIG = "rl_agent_config.json";
    static final String JULIA_CONFIG = "julia_config.json";
    /** The decision temperature names, in config order (bert.py:721). */
    private static final String[] TEMPERATURE_NAMES = {"choice", "score", "noul"};
    /** The default {@code head_max_len} of the decision head (bert.py:720). */
    private static final int DECISION_MAX_HEAD_DEFAULT = 256;

    private HfBertVariantHooks() {
    }

    // ------------------------------------------------------------------ DistilBert

    /** {@code DistilBertModel.filter_tensors} (bert.py:253-264): the prefix strip, the MLM drop, then BERT's. */
    public static String distilFilter(String name, UnaryOperator<String> base) {
        String n = name.startsWith("distilbert.") ? name.substring(11) : name;
        if (n.startsWith("vocab_")) {
            return null;
        }
        return HfBertHooks.filter(n, base);
    }

    /** {@code DistilBertModel.set_gguf_parameters} (bert.py:248-251): the 1e-12 epsilon, then BERT's. */
    public static void distilMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_EPS, DISTIL_EPS);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, DISTIL_EPS);
    }

    /** DistilBert's vocabulary: the WordPiece vocabulary like {@code BertModel}. */
    public static Map<String, Object> distilVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfBertHooks.vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ NomicBert

    /**
     * {@code NomicBertModel.__init__}'s arch switch (bert.py:327-328): {@link Arch#NOMIC_BERT_MOE} with a set
     * {@code moe_every_n_layers}, else {@link Arch#NOMIC_BERT}.
     */
    public static Arch nomicArch(HfConfig cfg, HfTensorIndex index) {
        return moeEveryN(cfg) > 0 ? Arch.NOMIC_BERT_MOE : Arch.NOMIC_BERT;
    }

    /** {@code int(hparams.get("moe_every_n_layers"))}, 0 when unset (bert.py:327). */
    public static int moeEveryN(HfConfig cfg) {
        Object v = cfg.find("moe_every_n_layers").orElse(null);
        if (v == null || Boolean.FALSE.equals(v)) {
            return 0;
        }
        if (v instanceof Number) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(cfg.source() + ": moe_every_n_layers must be a number, got " + v);
    }

    /**
     * {@code NomicBertModel.__init__}'s fixup (bert.py:332-358): with the XLM-R tokenizer files the shared
     * position offset (bert.py:332-334), the {@code n_positions} retarget of the known model sizes (:336-342)
     * and the class's configuration assertions (:344-358).
     */
    public static HfConfig nomicFixup(HfConfig cfg) {
        try {
            HfConfig fixed = nomicFixupChecked(cfg);
            assertNomicConfig(fixed);
            return fixed;
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": the NomicBert tokenizer files cannot be read: "
                    + e.getMessage(), e);
        }
    }

    private static HfConfig nomicFixupChecked(HfConfig cfg) throws IOException {
        HfTokenizerFiles files = HfTokenizerFiles.locate(cfg.dir());
        HfTokenizerJson tj = files.readTokenizerJson();
        if (tj == null) {
            throw new GgufFormatException(cfg.source() + ": NomicBertModel reads tokenizer.json");
        }
        switch (tj.model().type()) {
            case UNIGRAM:
                cfg = HfBertHooks.fixup(cfg);                                              // bert.py:333-334
                break;
            case WORD_PIECE:
                break;
            default:
                throw new GgufFormatException(tj.source() + ": unknown tokenizer: "
                        + tj.model().type().jsonName());                                   // bert.py:403
        }
        cfg = cfg.withFlat(Collections.singletonMap("n_positions",
                (Object) (long) matchModelPositions(cfg)));                                 // bert.py:336-342
        assertNomicConfig(cfg);                                                             // bert.py:344-358
        return cfg;
    }

    private static int matchModelPositions(HfConfig cfg) {
        int npos = cfg.requireInt("n_positions");
        int mtp = cfg.findInt(2048, "max_trained_positions");
        if (npos == 8192 && mtp == 2048) {
            return 2048;
        }
        if (npos == 2048 && mtp == 2048) {
            return 512;
        }
        throw new GgufFormatException(cfg.source() + ": n_positions=" + npos
                + ", max_trained_positions=" + mtp + " does not match the known nomic-bert model sizes");
    }

    /** The configuration assertions of bert.py:344-358, stated as the converter's {@code assert} messages. */
    private static void assertNomicConfig(HfConfig cfg) {
        requireString(cfg, moeEveryN(cfg) > 0 ? "gelu" : "swiglu", "activation_function");  // bert.py:344
        requireBool(cfg, false, "causal");                                                 // bert.py:347
        requireBool(cfg, moeEveryN(cfg) > 0, "qkv_proj_bias");                             // bert.py:349
        requireBool(cfg, moeEveryN(cfg) > 0, "mlp_fc1_bias");                              // bert.py:350
        requireBool(cfg, moeEveryN(cfg) > 0, "mlp_fc2_bias");                              // bert.py:351
        requireBool(cfg, false, "prenorm");                                                // bert.py:354
        if (cfg.findFloat(1.0, "rotary_emb_fraction") != 1.0) {
            throw new GgufFormatException(cfg.source() + ": rotary_emb_fraction must be 1.0 (bert.py:356)");
        }
        if (cfg.find("rotary_emb_interleaved").isPresent()
                && !Boolean.FALSE.equals(cfg.find("rotary_emb_interleaved").orElse(Boolean.FALSE))) {
            throw new GgufFormatException(cfg.source() + ": rotary_emb_interleaved must be false (bert.py:357)");
        }
    }

    private static void requireString(HfConfig cfg, String wanted, String key) {
        String got = cfg.findString(key).orElse(null);
        if (!wanted.equals(got)) {
            throw new GgufFormatException(cfg.source() + ": assert " + wanted + " for " + key + ", got " + got);
        }
    }

    private static void requireBool(HfConfig cfg, boolean expected, String key) {
        Boolean got = cfg.findBool(key).orElse(null);
        if (got == null || got.booleanValue() != expected) {
            throw new GgufFormatException(cfg.source() + ": " + key + " must be " + expected + " (bert.py)"
                    + ", got " + got);
        }
    }

    /** {@code NomicBertModel.filter_tensors} (bert.py:365-373): the expert bias drop, then BERT's. */
    public static String nomicFilter(String name, UnaryOperator<String> base) {
        if (name.contains("mlp.experts.bias")) {
            return null;
        }
        return HfBertHooks.filter(name, base);
    }

    /**
     * {@code NomicBertModel.modify_tensors} (bert.py:375-387): the {@code w1}/{@code w2} expert views into the
     * three-dimensional shape, the {@code w2} transpose, then the Bert step.
     */
    public static void nomicTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = hfName;
        if (name.contains("mlp.experts.mlp.w1") || name.contains("mlp.experts.mlp.w2")) {
            int nExperts = ctx.config().requireInt("num_local_experts", "num_experts");
            int nInner = ctx.config().requireInt("n_inner");
            int nEmbd = ctx.config().requireInt("n_embd");
            RowSource viewed = ElementTransforms.reshape(src, nExperts, nInner, nEmbd);      // bert.py:378
            if (name.endsWith("w2")) {
                viewed = ExpertTransforms.transposeLast2(viewed);                            // bert.py:384
            }
            HfBertHooks.tensors(ctx, name + ".weight", viewed, out);                         // bert.py:379, 385
            return;
        }
        HfBertHooks.tensors(ctx, name, src, out);
    }

    /** {@code NomicBertModel.set_gguf_parameters} (bert.py:389-393): BERT's, then the MoE keys. */
    public static void nomicMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        if (moeEveryN(cfg) > 0) {
            kv.set(GgufKeys.MOE_EVERY_N_LAYERS, moeEveryN(cfg));                            // bert.py:392
            kv.set(GgufKeys.EXPERT_USED_COUNT, cfg.requireInt("moe_top_k"));                // bert.py:393
        }
    }

    /** {@code NomicBertModel.set_vocab} (bert.py:360-363): the XLM-R Unigram tokenizer, else the WordPiece. */
    public static Map<String, Object> nomicVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        boolean xlm = HfBertHooks.isTokenizerXlmRoberta(ctx);
        return xlm ? HfBertHooks.xlmRobertaVocab(ctx, pre, moeEveryN(ctx.config()) > 0)
                : HfBertHooks.vocab(ctx, pre);
    }

    // ------------------------------------------------------------------ NeoBert

    /** {@code NeoBert.filter_tensors} (bert.py:425-435): the decoder drop and the {@code model.} strip. */
    public static String neoFilter(String name, UnaryOperator<String> base) {
        String n = name;
        if (n.startsWith("decoder.")) {
            return null;
        }
        if (n.startsWith("model.")) {
            n = n.substring(6);
        }
        return HfBertHooks.filter(n, base);
    }

    /** {@code NeoBert.set_gguf_parameters} (bert.py:411-423): BERT's, then the model keys. */
    public static void neoMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        long ffn = (long) (2.0 * cfg.requireInt("intermediate_size") / 3);                   // bert.py:415
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, ffn);
        kv.set(GgufKeys.ROPE_FREQ_BASE, 10000.0);                                            // bert.py:416
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");                                          // bert.py:417
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, cfg.findFloat(1e-6, "norm_eps"));       // bert.py:419-421
        kv.set(GgufKeys.POOLING_TYPE, PoolingKind.CLS.ggufCode());                            // bert.py:423
    }

    // ------------------------------------------------------------------ EuroBert

    /** {@code EuroBertModel.filter_tensors} (bert.py:457-464): the {@code model.} strip. */
    public static String euroFilter(String name, UnaryOperator<String> base) {
        String n = name.startsWith("model.") ? name.substring(6) : name;
        return HfBertHooks.filter(n, base);
    }

    /** {@code EuroBertModel.set_gguf_parameters} (bert.py:447-455): BERT's plus the bidirectional keys. */
    public static void euroMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);                                            // bert.py:451
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");                                          // bert.py:453
    }

    /** {@code EuroBertModel.set_vocab} (bert.py:443-445): the GPT-2 vocabulary with the bos flag off. */
    public static Map<String, Object> euroVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_ADD_BOS, false);
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ JinaBertV2

    /**
     * {@code JinaBertV2Model.set_vocab} (bert.py:575-595): the tokenizer_config {@code tokenizer_class}
     * dispatch — the WordPiece BERT branch, or the Roberta branch reading tokenizer.json's pre-tokenizer with
     * a token-type count of 2; anything else raises the converter's {@code NotImplementedError}.
     */
    public static Map<String, Object> jinaV2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        JsonView tc = ctx.tokenizerFiles().readTokenizerConfig();
        String tokenizerClass = "BertTokenizer";                                             // bert.py:576-578
        Object declared = tc.raw("tokenizer_class");
        if (declared instanceof String) {
            tokenizerClass = (String) declared;
        }
        if (HfBertHooks.BERT_TOKENIZER_CLASS.equals(tokenizerClass)) {
            return HfBertHooks.vocab(ctx, pre);
        }
        if (HfBertHooks.ROBERTA_TOKENIZER_CLASS.equals(tokenizerClass)) {
            if (isWhitespacePreTokenizer(ctx)) {
                return HfVocabWriters.whitespace(HfHookSupport.vocabInput(ctx), pre);        // bert.py:590
            }
            Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.gpt2(
                    HfHookSupport.vocabInput(ctx), pre));                                   // bert.py:592
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, 2);                                  // bert.py:593
            return Collections.unmodifiableMap(md);
        }
        throw new GgufFormatException("Tokenizer " + tokenizerClass + " is not supported for JinaBertModel");
    }

    private static boolean isWhitespacePreTokenizer(HfLoadContext ctx) throws IOException {
        HfTokenizerJson tj = ctx.tokenizerFiles().readTokenizerJson();
        JsonView preTokenizer = tj == null ? null : tj.preTokenizer();
        return preTokenizer != null && preTokenizer.has("type")
                && WHITESPACE_PRE_TOKENIZER.equals(preTokenizer.str("type"));
    }

    // ------------------------------------------------------------------ ModernBert

    /** {@code ModernBertModel.filter_tensors} (bert.py:634-641): the {@code model.} strip. */
    public static String modernFilter(String name, UnaryOperator<String> base) {
        String n = name.startsWith("model.") ? name.substring(6) : name;
        return HfBertHooks.filter(n, base);
    }

    /** {@code ModernBertModel.set_vocab} (bert.py:603-607): the GPT-2 vocabulary with bos, eos and sep. */
    public static Map<String, Object> modernVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>(
                HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre));
        md.put(GgufKeys.TOKENIZER_ADD_BOS, true);
        md.put(GgufKeys.TOKENIZER_ADD_EOS, true);
        md.put(GgufKeys.TOKENIZER_ADD_SEP, true);
        return md;
    }

    /** {@code ModernBertModel.set_gguf_parameters} (bert.py:620-632): BERT's, then the ModernBert keys. */
    public static void modernMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertHooks.meta(ctx, kv, plan);
        HfConfig cfg = ctx.config();
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("local_attention"));        // bert.py:622
        Object pattern = cfg.find("global_attn_every_n_layers").orElse(null);               // bert.py:623-624
        if (pattern instanceof Number) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, ((Number) pattern).intValue());
        } else if (pattern != null) {
            throw new GgufFormatException(cfg.source() + ": global_attn_every_n_layers must be a number, got "
                    + pattern);
        }
        kv.set(GgufKeys.ROPE_SCALING_TYPE, "none");                                          // bert.py:625
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));                           // bert.py:626
        Object hiddenAct = cfg.find("hidden_activation").orElse(null);                       // bert.py:631-632
        if (hiddenAct instanceof String) {
            kv.set(GgufKeys.HIDDEN_ACT, hiddenAct);
        }
    }

    /** {@code ModernBertModel.modify_tensors} (bert.py:643-652): the classifier renames inside BERT's step. */
    public static void modernTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        HfBertHooks.tensors(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ ModernBertDecisionModel

    /**
     * {@code _load_decision_hparams} (bert.py:661-675): the encoder configuration replaces the loader's
     * members, the decision file's values become {@code decision}, the head blocks are appended to the
     * encoder blocks with a 4x FFN and the architectures list names the class. NeuralFree resolves the
     * checkpoint by the root {@code config.json} naming {@code ModernBertDecisionModel}; this fixup rejects
     * such a checkpoint without the decision side files of bert.py:655-658.
     *
     * @throws GgufFormatException when a decision side file cannot be read
     */
    public static HfConfig decisionFixup(HfConfig cfg) {
        if (!"ModernBertDecisionModel".equals(cfg.architecture())) {
            return cfg;
        }
        try {
            return decisionFixupChecked(cfg);
        } catch (IOException e) {
            throw new GgufFormatException(cfg.source() + ": the decision checkpoint side files cannot be read: "
                    + e.getMessage(), e);
        }
    }

    private static HfConfig decisionFixupChecked(HfConfig cfg) throws IOException {
        if (!"ModernBertDecisionModel".equals(cfg.architecture())) {
            return cfg;
        }
        Path rlConfig = cfg.sideFile(RL_AGENT_CONFIG);
        Path juliaConfig = cfg.sideFile(JULIA_CONFIG);
        if (!Files.isRegularFile(rlConfig) && !Files.isRegularFile(juliaConfig)) {
            throw new GgufFormatException(cfg.source() + ": the decision checkpoint has no "
                    + RL_AGENT_CONFIG + " or " + JULIA_CONFIG + " (bert.py:655-658)");
        }
        Map<String, Object> decision = readObject(Files.isRegularFile(rlConfig) ? rlConfig : juliaConfig);
        Map<String, Object> encoder = readObject(cfg.sideFile(ENCODER_CONFIG));
        int headLayers = decisionInt(decision, "head_layers");
        int nLayer = asInt(encoder.get("num_hidden_layers"), ENCODER_CONFIG + ": num_hidden_layers");
        int hidden = asInt(encoder.get("hidden_size"), ENCODER_CONFIG + ": hidden_size");
        Object inner = encoder.get("intermediate_size");
        if (!(inner instanceof List) || ((List<?>) inner).size() != nLayer) {
            throw new GgufFormatException(ENCODER_CONFIG + ": intermediate_size must be a "
                    + nLayer + "-entry list (bert.py:674)");
        }
        List<Long> ff = new ArrayList<>(nLayer + headLayers);
        for (int i = 0; i < nLayer; i++) {
            ff.add(asLong(((List<?>) inner).get(i), "intermediate_size[" + i + "]"));
        }
        for (int i = 0; i < headLayers; i++) {
            ff.add(4L * hidden);                                                            // bert.py:674
        }
        Map<String, Object> merged = new LinkedHashMap<>(cfg.flat());
        merged.putAll(encoder);
        merged.put("decision", decision);                                                   // bert.py:671
        merged.put("num_hidden_layers", (long) (nLayer + headLayers));                      // bert.py:673
        merged.put("intermediate_size", ff);                                                // bert.py:674
        return cfg.withFlat(merged);
    }

    /** {@code ModernBertDecisionModel.filter_tensors} (bert.py:727-738): the head drops, then ModernBert's. */
    public static String decisionFilter(String name, UnaryOperator<String> base) {
        String n = name;
        if (n.startsWith("act_head.") || n.equals(TEMPERATURE_TENSOR)) {
            return null;                                                                     // bert.py:731-732
        }
        if (n.startsWith("encoder.")) {
            n = n.substring(8);                                                              // bert.py:735-736
        }
        return modernFilter(n, base);
    }

    /**
     * {@code ModernBertDecisionModel.set_gguf_parameters} (bert.py:715-725): ModernBert's, then the decision
     * head keys and the report temperatures, including the per-option ones whose name the converter maps
     * ({@code choice:3-5} → {@code choice.3_5}).
     */
    public static void decisionMeta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfBertVariantHooks.modernMeta(ctx, kv, plan);
        Map<?, ?> decision = decisionOf(ctx.config());
        kv.set(GgufKeys.DECISION_TYPE, DECISION_TYPE_LAYA);                                  // bert.py:718
        kv.set(GgufKeys.DECISION_BLOCK_COUNT, decisionInt(decision, "head_layers"));         // bert.py:719
        kv.set(GgufKeys.DECISION_MAX_HEAD_TOKENS, decisionInt(decision, DECISION_MAX_HEAD_DEFAULT,
                "head_max_len"));                                                            // bert.py:720
        Object temperature = decision.get("temperature");
        if (temperature instanceof List) {                                                    // bert.py:721-722
            List<?> values = (List<?>) temperature;
            for (int i = 0; i < values.size() && i < TEMPERATURE_NAMES.length; i++) {
                kv.setAt(GgufKeys.DECISION_TEMPERATURE, asDouble(values.get(i)), TEMPERATURE_NAMES[i]);
            }
        }
        Object byOptions = decision.get("temperature_by_options");
        if (byOptions instanceof Map) {                                                       // bert.py:724-725
            for (Map.Entry<?, ?> e : ((Map<?, ?>) byOptions).entrySet()) {
                kv.setAt(GgufKeys.DECISION_TEMPERATURE, asDouble(e.getValue()),
                        temperatureName(e.getKey()));
            }
        }
    }

    /** The temperature key name of a {@code temperature_by_options} entry (bert.py:725). */
    static String temperatureName(Object key) {
        return String.valueOf(key).replace(":", ".").replace("-", "_").replace("+", "");
    }

    /**
     * {@code ModernBertDecisionModel.modify_tensors} (bert.py:740-747): the head blocks move after the encoder
     * blocks ({@code bid += block_count - head_layers}), and {@code in_proj_*} becomes the fused
     * {@code in_proj.*} name, then the ModernBert step.
     */
    public static void decisionTensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = hfName;
        if (name.startsWith("head.layers.")) {                                                // bert.py:741-746
            String[] parts = name.split("\\.", 4);
            if (parts.length == 4) {
                int moved = headLayerId(name, parts[2]) + ctx.blockCount()
                        - decisionInt(decisionOf(ctx.config()), "head_layers");
                name = "head.layers." + moved + "." + parts[3].replace("in_proj_", "in_proj.");
            }
        }
        HfBertHooks.tensors(ctx, name, src, out);
    }

    /**
     * {@code ModernBertDecisionModel.set_vocab} (bert.py:683-692): the tokenizer sub-directory's vocabulary
     * with a token-type count of 3 and the systemone chat template.
     */
    public static Map<String, Object> decisionVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles().withTokenizerDir(
                ctx.config().sideFile(TOKENIZER_DIR));                                        // bert.py:685-686
        Map<String, Object> md = new LinkedHashMap<>(HfBertHooks.vocab(ctx, pre));             // bert.py:688
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, 3);                                        // bert.py:691
        md.putAll(HfChatTemplates.toGguf(Collections.singletonList(systemOne(ctx, files))));   // bert.py:692
        return Collections.unmodifiableMap(md);
    }

    /** The systemone chat template of bert.py:694-713. */
    private static Map<String, String> systemOne(HfLoadContext ctx, HfTokenizerFiles files) throws IOException {
        JsonView tc = files.readTokenizerConfig();
        String clsToken = tc.str("cls_token");
        String sepToken = tc.str("sep_token");
        String maskToken = tc.str("mask_token");
        Object architecture = decisionOf(ctx.config()).get("architecture");
        String option = TEMPLATE_JULIA_OPTION;
        if (!"JuliaDecisionModel".equals(architecture)) {                                      // bert.py:699
            option = TEMPLATE_LAYA_OPTION;
        }
        Map<String, String> template = new LinkedHashMap<>();
        template.put("name", "systemone");
        template.put("template", clsToken + "{{ type }} question: {{ instructions | tojson }}" + sepToken
                + "{% for o in options %}" + maskToken + " " + option + "{% endfor %}" + sepToken
                + "{{ state | tojson }}" + sepToken);
        return template;
    }

    /** The Julia option body of the systemone template (bert.py:700). */
    private static final String TEMPLATE_JULIA_OPTION =
            "{% if o.description %}{{ o.description | tojson }}{% else %}{{ o.key }}{% endif %}";
    /** The LAYA option body of the systemone template (bert.py:702-708). */
    private static final String TEMPLATE_LAYA_OPTION =
            "{% if type == 'choice' %}{{ o.key }}{% if o.description %}: {{ o.description | tojson }}"
            + "{% endif %}{% elif type == 'score' %}level {{ o.key }}: {{ o.description | tojson }}"
            + "{% else %}{{ o.key }}: {% if o.description %}{{ o.description | tojson }}"
            + "{% elif o.key == 'true' %}yes, the statement holds"
            + "{% else %}no, the statement does not hold{% endif %}{% endif %}";
    /** The name of the act-head temperature tensor, dropped with the temperatures (bert.py:731-732). */
    static final String TEMPERATURE_TENSOR = "temperature";

    // ------------------------------------------------------------------ shared decision helpers

    /**
     * The decision object of a decision checkpoint's flat configuration, required (the {@code decision} of
     * bert.py:671).
     *
     * @throws GgufFormatException when the fixup has not run or the object is missing
     */
    public static Map<?, ?> decisionOf(HfConfig cfg) {
        Object d = cfg.find("decision").orElse(null);
        if (d instanceof Map) {
            return (Map<?, ?>) d;
        }
        throw new GgufFormatException(cfg.source() + ": decision checkpoints need a decision configuration"
                + " object");
    }

    /** The int of a decision file member, required. */
    static int decisionInt(Map<?, ?> decision, String key) {
        Object v = decision.get(key);
        if (!(v instanceof Number)) {
            throw new GgufFormatException("the decision configuration needs the int member '" + key + "'");
        }
        return ((Number) v).intValue();
    }

    /** The int of a decision file member, or the default. */
    static int decisionInt(Map<?, ?> decision, int dflt, String key) {
        Object v = decision.get(key);
        return v instanceof Number ? ((Number) v).intValue() : dflt;
    }

    private static int headLayerId(String name, String part) {
        try {
            return Integer.parseInt(part);
        } catch (NumberFormatException e) {
            throw new GgufFormatException("head layer '" + name + "' has no layer number");
        }
    }

    private static double asDouble(Object v) {
        if (v instanceof Number) {
            return ((Number) v).doubleValue();
        }
        throw new GgufFormatException("a decision temperature must be a number, got " + v);
    }

    private static int asInt(Object v, String where) {
        if (v instanceof Number) {
            return ((Number) v).intValue();
        }
        throw new GgufFormatException(where + " must be an int, got " + v);
    }

    private static long asLong(Object v, String where) {
        if (v instanceof Number) {
            return ((Number) v).longValue();
        }
        throw new GgufFormatException(where + " must be an int, got " + v);
    }

    // ------------------------------------------------------------------ JSON side files

    /** The JSON object of {@code file}, or an empty map when it does not exist. */
    static JsonView readJsonView(Path file) throws IOException {
        Object parsed = Files.isRegularFile(file) ? parse(file) : null;
        return parsed instanceof Map ? JsonView.of(parsed, file.toString())
                : JsonView.empty(file.toString(), file.toString());
    }

    /** The JSON object of {@code file}, required. */
    static Map<String, Object> readObject(Path file) throws IOException {
        Object parsed = parse(file);
        if (!(parsed instanceof Map)) {
            throw new GgufFormatException(file + ": expected a JSON object");
        }
        return (Map<String, Object>) parsed;
    }

    private static Object parse(Path file) throws IOException {
        try {
            return Json.parse(file, HfConfigLoader.MAX_CONFIG_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw new GgufFormatException(e.getMessage(), e);
        }
    }
}
