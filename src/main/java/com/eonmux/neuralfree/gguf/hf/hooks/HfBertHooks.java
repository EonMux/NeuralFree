package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfSpecialVocab;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.SentencePieceModel;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.function.UnaryOperator;

/**
 * Hooks of the BERT classes and the RoBERTa / XLM-R family (LC/conversion/bert.py): {@code BertModel} (:19-240),
 * {@code RobertaModel} (:267-314) and {@code XLMRobertaModel} (:467-567), whose arch is JINA_BERT_V3 when the
 * configuration lists the Jina-v3 {@code lora_adaptations} (:474-483). Ports the class {@code filter_tensors}
 * renames and drops (:70-92), the classifier output labels (:26-38), the position-embedding chop of the offset
 * classes (:105-112, :308-314, :520-546 — the {@link RowTransforms#sliceRows} of the converter's
 * {@code data_torch[offset:]}) and the Unigram → t5-tokenizer vocabulary with the HF realignment (:114-240),
 * which also serves the next file's NomicBert variant. The phantom-space WordPiece vocabulary and its
 * {@code token_type_count} live in {@link HfVocabWriters#bertWpm} (bert.py:47).
 *
 * <p>Every method is a public static hook reference that entries compose where the converter class calls
 * {@code super()}.</p>
 */
public final class HfBertHooks {

    /** The tokenizer_config class of the plain BERT branch {@code JinaBertV2Model} reads (bert.py:576-580). */
    public static final String BERT_TOKENIZER_CLASS = "BertTokenizer";
    /** The RobertaTokenizer class of its second branch (bert.py:582). */
    public static final String ROBERTA_TOKENIZER_CLASS = "RobertaTokenizer";
    /** The suffix of a jina-v3 task-adapter factor tensor (bert.py:526). */
    static final String LORA_A = ".0.lora_A";
    /** The suffix of a jina-v3 task-adapter factor tensor (bert.py:526). */
    static final String LORA_B = ".0.lora_B";
    /** The mask piece NOMIC_BERT_MOE needs (bert.py:222-226). */
    static final String MASK_TOKEN = "<mask>";
    /** The id at which {@link #MASK_TOKEN} is injected (bert.py:224). */
    static final int MASK_TOKEN_ID = 250001;

    /** {@code gguf.SentencePieceTokenTypes.NORMAL} (LC/gguf-py/gguf/constants.py). */
    static final int SP_NORMAL = 1;
    /** {@code SentencePieceTokenTypes.UNKNOWN}. */
    static final int SP_UNKNOWN = 2;
    /** {@code SentencePieceTokenTypes.CONTROL}. */
    static final int SP_CONTROL = 3;
    /** {@code SentencePieceTokenTypes.USER_DEFINED}. */
    static final int SP_USER_DEFINED = 4;
    /** {@code SentencePieceTokenTypes.UNUSED}. */
    static final int SP_UNUSED = 5;
    /** The four slots of the SentencePiece realignment (bert.py:212-220). */
    static final String[] XLMR_PREFIX = {"<s>", "<pad>", "</s>", "<unk>"};
    /** Their types: control, control, control, unknown (bert.py:215-220). */
    static final int[] XLMR_PREFIX_TYPES = {3, 3, 3, 2};

    private static final String ID2LABEL = "id2label";
    private static final String POSITION_EMBEDDINGS = "embeddings.position_embeddings.weight";
    private static final List<String> DROPS = Collections.unmodifiableList(Arrays.asList(
            "embeddings.position_ids", "pooler.dense.weight", "pooler.dense.bias"));    // bert.py:83-84
    private static final List<String> HEAD_PREFIXES = Collections.unmodifiableList(Arrays.asList(
            "cls.predictions", "cls.seq_relationship"));                                 // bert.py:86-91

    private HfBertHooks() {
    }

    // ------------------------------------------------------------------ configuration

    /**
     * {@code RobertaModel.__init__}'s fixup (bert.py:105-112, 272-281): with a {@code pad_token_id},
     * {@code max_position_embeddings} reduced by {@code pad_token_id + 1}, so
     * {@link #robertaTensor}/{@link #xlmRobertaTensor} slice the position embeddings from that row.
     */
    public static HfConfig fixup(HfConfig cfg) {
        Integer pad = cfg.findInt("pad_token_id").isPresent() ? cfg.findInt("pad_token_id").getAsInt() : null;
        if (pad == null) {
            return cfg;
        }
        if (cfg.findInt("max_position_embeddings").isPresent()) {
            cfg = cfg.withFlat(Collections.singletonMap("max_position_embeddings",
                    (long) cfg.requireInt("max_position_embeddings") - (1 + pad)));
        }
        return cfg;
    }

    /** The per-load {@code 1 + pad_token_id} the fixup applied, or null without a {@code pad_token_id}. */
    public static Integer positionOffset(HfLoadContext ctx) {
        return ctx.config().findInt("pad_token_id").isPresent()
                ? ctx.config().findInt("pad_token_id").getAsInt() + 1 : null;
    }

    // ------------------------------------------------------------------ names

    /**
     * {@code BertModel.filter_tensors} (bert.py:70-92) and {@code Roberta}'s leading {@code roberta.} strip
     * (:303-306): the prefix renames, then the drops, then the inherited filters.
     */
    public static String filter(String name, UnaryOperator<String> base) {
        String n = name.startsWith("roberta.") ? name.substring(8) : name;              // bert.py:303-306
        n = bertPrefix(n);
        if (DROPS.contains(n) || headDrop(n)) {
            return null;
        }
        return base.apply(n);
    }

    /** {@code BertModel.filter_tensors} as an {@link HfNameFilter}, for the BERT and DistilBert entries. */
    public static final HfNameFilter FILTER = HfBertHooks::filter;

    /**
     * {@code RobertaModel.filter_tensors} (bert.py:296-306): it is {@link #filter} with the RobertaModel
     * {@code roberta.} strip already inside, so the entry reuses the same hook.
     */
    public static String robertaFilter(String name, UnaryOperator<String> base) {
        return filter(name, base);
    }

    /** {@code XLMRobertaModel.filter_tensors} (bert.py:503-518) with the jina-v3 parametrization renames. */
    public static final HfNameFilter FILTER_XLMR = (name, base) -> {
        String n = name.startsWith("roberta.") ? name.substring(8) : name;
        if (n.contains(".parametrizations.")) {
            n = n.replace(".parametrizations.", ".");
            if (n.endsWith(".original")) {
                n = n.substring(0, n.length() - 9);
            }
        }
        return filter(n, base);
    };

    /** The {@code bert.} strip, the {@code .gamma} weight and the {@code .beta} bias renames (bert.py:73-80). */
    static String bertPrefix(String name) {
        String n = name;
        if (n.startsWith("bert.")) {
            n = n.substring(5);
        }
        if (n.endsWith(".gamma")) {
            n = n.substring(0, n.length() - 6) + ".weight";
        }
        if (n.endsWith(".beta")) {
            n = n.substring(0, n.length() - 5) + ".bias";
        }
        return n;
    }

    private static boolean headDrop(String name) {
        for (String prefix : HEAD_PREFIXES) {
            if (name.startsWith(prefix)) {
                return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ metadata

    /**
     * {@code BertModel.set_gguf_parameters} (bert.py:32-38): {@code super().set_gguf_parameters()}, then
     * non-causal attention (:34), the sentence-transformers pooling of {@link HfQwenHooks#pooling}
     * ({@code _try_set_pooling_type}, base.py:2510-2562) and the classifier output labels.
     */
    public static void meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.ATTENTION_CAUSAL, false);                                      // bert.py:34
        HfQwenHooks.pooling(ctx, kv);                                                    // bert.py:35
        Map<Integer, String> labels = id2label(ctx.config());
        if (!labels.isEmpty()) {
            kv.set(GgufKeys.CLASSIFIER_OUTPUT_LABELS, new ArrayList<>(labels.values())); // bert.py:37-38
        }
    }

    /**
     * {@code self.cls_out_labels} (bert.py:26-30): the {@code id2label} map sorted by id, without the dummy
     * {@code LABEL_0}/{@code LABEL_1} pair {@code AutoConfig} adds; empty when none remains.
     */
    static Map<Integer, String> id2label(HfConfig cfg) {
        Object raw = cfg.find(ID2LABEL).orElse(null);
        if (!(raw instanceof Map)) {
            return Collections.emptyMap();
        }
        Map<Integer, String> out = new TreeMap<>();
        for (Map.Entry<?, ?> e : ((Map<?, ?>) raw).entrySet()) {
            int id;
            try {
                id = Integer.parseInt(String.valueOf(e.getKey()));
            } catch (NumberFormatException ex) {
                throw new GgufFormatException(cfg.source() + ": id2label key '" + e.getKey()
                        + "' is not a label id");
            }
            out.put(id, String.valueOf(e.getValue()));
        }
        if (out.size() == 2 && "LABEL_0".equals(out.get(0)) && "LABEL_1".equals(out.get(1))) {
            return Collections.emptyMap();
        }
        return Collections.unmodifiableMap(out);
    }

    // ------------------------------------------------------------------ tensors

    /** {@code BertModel.modify_tensors} (bert.py:94-103): the classifier renames, then the mapped name. */
    public static void tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfTensorHook.DEFAULT.apply(ctx, renameClassifier(ctx, bertPrefix(hfName)), src, out);
    }

    /** {@code RobertaModel.modify_tensors} (bert.py:308-314): the position chop, then the Bert step. */
    public static void robertaTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = bertPrefix(hfName);
        Integer offset = positionOffset(ctx);
        if (POSITION_EMBEDDINGS.equals(name) && offset != null) {
            out.add(ctx.names().mapOrThrow(name), chop(ctx, hfName, src, offset));
            return;
        }
        tensors(ctx, hfName, src, out);
    }

    /**
     * {@code XLMRobertaModel.modify_tensors} (bert.py:520-546): the position chop, then the dropped pooler and
     * task-adapter LoRA factors of a JINA_BERT_V3 checkpoint (bert.py:526-544, planned by llama.cpp as
     * separate adapter GGUFs), then the Bert step.
     */
    public static void xlmRobertaTensor(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        String name = bertPrefix(hfName);
        Integer offset = positionOffset(ctx);
        if (POSITION_EMBEDDINGS.equals(name) && offset != null) {
            out.add(ctx.names().mapOrThrow(name), chop(ctx, hfName, src, offset));
            return;
        }
        if (name.endsWith(LORA_A) || name.endsWith(LORA_B)) {
            if (name.startsWith("pooler.dense")) {
                out.drop(hfName, "the pooler LoRA adapters of the dropped pooler are not planned");
                return;
            }
            ctx.log("dropping the jina-v3 LoRA task adapter '" + hfName + "': llama.cpp writes it as a"
                    + " separate lora-<task>- adapter GGUF");
            out.drop(hfName, name.endsWith(LORA_B) ? "lora_B of a jina-v3 task adapter"
                    : "lora_A of a jina-v3 task adapter");
            return;
        }
        tensors(ctx, hfName, src, out);
    }

    private static RowSource chop(HfLoadContext ctx, String hfName, RowSource src, int offset) {
        long[] shape = src.shape();
        if (shape.length == 0) {
            throw new GgufFormatException("tensor '" + hfName + "' is a scalar");
        }
        if (offset >= shape[0]) {
            throw new GgufFormatException("tensor '" + hfName + "' has " + shape[0] + " rows, fewer than the "
                    + offset + " skipped by pad_token_id + 1");
        }
        return RowTransforms.sliceRows(src, offset, shape[0] - offset);
    }

    /**
     * The classifier renames of {@code BertForSequenceClassification} (bert.py:95-101, 643-651): when the
     * configuration keeps real {@code id2label} labels, {@code classifier.weight}/{@code classifier.bias}
     * become {@code classifier.out_proj.weight}/{@code classifier.out_proj.bias}.
     */
    static String renameClassifier(HfLoadContext ctx, String name) {
        if (!id2label(ctx.config()).isEmpty() && name.startsWith("classifier.")
                && (name.endsWith(".weight") || name.endsWith(".bias"))) {
            return name.replace("classifier.", "classifier.out_proj.");
        }
        return name;
    }

    // ------------------------------------------------------------------ BERT vocabularies

    /** {@code BertModel.set_vocab} (bert.py:40-67): the phantom-space WordPiece vocabulary. */
    public static Map<String, Object> vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.bertWpm(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code RobertaModel.set_vocab} (bert.py:283-295): the GPT-2 BPE vocabulary with a token-type count when
     * the checkpoint has a {@code tokenizer.json}, else the WordPiece vocabulary like {@code BertModel}.
     */
    public static Map<String, Object> robertaVocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (ctx.tokenizerFiles().readTokenizerJson() != null) {
            Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.gpt2(
                    HfHookSupport.vocabInput(ctx), pre));                          // bert.py:287
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, ctx.config().findInt(1, "type_vocab_size")); // :292
            return Collections.unmodifiableMap(md);
        }
        return vocab(ctx, pre);
    }

    /**
     * {@code XLMRobertaModel.set_vocab} (bert.py:500-502) around {@code _xlmroberta_set_vocab}
     * (bert.py:114-240), also for the NomicBert checkpoints whose tokenizer is a XLM-R Unigram
     * (bert.py:395-403): the SentencePiece Unigram model becomes a t5-tokenizer with the HF realignment
     * (:211-226). {@code nomicMoe} adds the {@code <mask>} piece at id 250001 (:222-226).
     *
     * @throws GgufFormatException when the checkpoint has no Unigram tokenizer source
     */
    public static Map<String, Object> xlmRobertaVocab(HfLoadContext ctx, PreNameResolver pre, boolean nomicMoe)
            throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        Path spPath;
        if (files.sentencePieceBpeModel() != null) {
            spPath = files.sentencePieceBpeModel();
        } else if (files.sentencePieceModel() != null) {
            spPath = files.sentencePieceModel();
        } else {
            HfTokenizerJson tj = files.readTokenizerJson();
            if (tj == null || tj.model().type() != HfTokenizerJson.ModelType.UNIGRAM) {
                throw new GgufFormatException(ctx.config().source() + ": the XLM-RoBERTa vocabulary needs"
                        + " sentencepiece.bpe.model, or a tokenizer.json Unigram model (bert.py:121-130)");
            }
            return tokenizerJsonUnigram(ctx, tj);
        }
        SentencePieceModel sp = SentencePieceModel.read(spPath);
        if (sp.trainerSpec().modelType() != SentencePieceModel.ModelType.UNIGRAM) {
            throw new GgufFormatException(spPath + ": the " + sp.trainerSpec().modelType()
                    + " SentencePiece model is not a Unigram model (bert.py:151)");
        }
        int size = vocabSize(ctx, Math.max(ctx.config().findInt(0, "vocab_size"), sp.pieceCount()));
        String[] texts = padList(size);
        float[] scores = new float[size];
        Arrays.fill(scores, -10000.0f);
        int[] types = new int[size];
        Arrays.fill(types, SP_UNUSED);
        int unkId = sp.trainerSpec().unkId();
        for (int id = 0; id < sp.pieceCount() && id < size; id++) {
            texts[id] = sp.piece(id);
            scores[id] = sp.score(id);
            types[id] = id == unkId ? SP_UNKNOWN : pieceType(sp.type(id));
        }
        for (Map.Entry<String, Integer> e : files.readAddedTokens().entrySet()) {
            int id = e.getValue();
            if (id >= 0 && id < size) {
                texts[id] = e.getKey();
                scores[id] = -1000.0f;
                types[id] = SP_USER_DEFINED;
            }
        }
        // tokens = [<s>, <pad>, </s>, <unk>] + tokens[3:-1] (bert.py:211-226)
        String[] realigned = new String[size];
        float[] realignedScores = new float[size];
        int[] realignedTypes = new int[size];
        for (int k = 0; k < size && k < 4; k++) {
            realigned[k] = XLMR_PREFIX[k];
            realignedScores[k] = 0.0f;
            realignedTypes[k] = XLMR_PREFIX_TYPES[k];
        }
        for (int k = 3; k < size - 1; k++) {
            realigned[k + 1] = texts[k];
            realignedScores[k + 1] = scores[k];
            realignedTypes[k + 1] = types[k];
        }
        if (nomicMoe && size > MASK_TOKEN_ID) {
            realigned[MASK_TOKEN_ID] = MASK_TOKEN;
            realignedScores[MASK_TOKEN_ID] = 0.0f;
            realignedTypes[MASK_TOKEN_ID] = SP_CONTROL;
        }
        return unigramKv(ctx, sp.normalizerSpec(), realigned, realignedScores, realignedTypes);
    }

    /** The {@code AutoTokenizer} Unigram fallback (bert.py:184-209): pieces and scores of tokenizer.json. */
    private static Map<String, Object> tokenizerJsonUnigram(HfLoadContext ctx, HfTokenizerJson tj)
            throws IOException {
        HfConfig cfg = ctx.config();
        int size = vocabSize(ctx, Math.max(cfg.findInt(0, "vocab_size"), tj.model().vocab().length));
        String[] texts = tj.model().vocab();
        float[] scores = new float[size];
        int[] types = new int[size];
        double[] pieceScores = tj.model().scores();                                        // bert.py:194
        int unkId = tj.model().unkId();
        Map<String, Integer> added = ctx.tokenizerFiles().readAddedTokens();
        for (int id = 0; id < size; id++) {
            if (id < pieceScores.length) {
                scores[id] = (float) pieceScores[id];
            }
            if (id >= texts.length) {
                continue;                                                                  // bert.py:190-192
            }
            if (id == unkId) {
                types[id] = SP_UNKNOWN;
            } else if (added.containsKey(texts[id])) {
                types[id] = SP_USER_DEFINED;
            } else {
                types[id] = isSpecial(ctx, texts[id]) ? SP_CONTROL : SP_NORMAL;
            }
        }
        JsonView normalizerView = tj.normalizer();
        Object normalizer = normalizerView == null ? null : normalizerView.raw("precompiled_charsmap");
        byte[] chars = normalizer instanceof String && !((String) normalizer).isEmpty()
                ? Base64.getDecoder().decode((String) normalizer) : new byte[0];         // bert.py:145
        return unigramKv(ctx, chars, texts, scores, types);
    }

    private static boolean isSpecial(HfLoadContext ctx, String token) throws IOException {
        JsonView specials = ctx.tokenizerFiles().readSpecialTokensMap();
        return specials != null && specials.has(token);
    }

    private static int vocabSize(HfLoadContext ctx, int want) {
        if (want < 1) {
            throw new GgufFormatException(ctx.config().source() + ": the XLM-RoBERTa vocabulary has no tokens");
        }
        return want;
    }

    private static String[] padList(int size) {
        String[] out = new String[size];
        for (int id = 0; id < size; id++) {
            out[id] = "[PAD" + id + "]";
        }
        return out;
    }

    private static int pieceType(SentencePieceModel.PieceType type) {
        switch (type) {
            case UNKNOWN: return SP_UNKNOWN;
            case CONTROL: return SP_CONTROL;
            case UNUSED: return SP_UNUSED;
            case BYTE: return 6;
            case USER_DEFINED: return SP_USER_DEFINED;
            default: return SP_NORMAL;
        }
    }

    private static Map<String, Object> unigramKv(HfLoadContext ctx, SentencePieceModel.NormalizerSpec norm,
                                                 String[] texts, float[] scores, int[] types) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "t5");                                         // bert.py:228
        md.put(GgufKeys.TOKENIZER_PRE, "default");
        md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(Arrays.asList(texts)));
        md.put(GgufKeys.TOKENIZER_SCORES, floats(scores));
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, ints(types));
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, norm.addDummyPrefix());                   // bert.py:233
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, ctx.config().findInt(1, "type_vocab_size")); // :234
        md.put(GgufKeys.TOKENIZER_REMOVE_EXTRA_WS, norm.removeExtraWhitespaces());
        md.put(GgufKeys.TOKENIZER_PRECOMPILED_CHARSMAP, ints(norm.precompiledCharsmap())); // :236-237
        specialVocab(ctx, texts.length).writeTo(md);                                     // :239-240
        return md;
    }

    private static Map<String, Object> unigramKv(HfLoadContext ctx, byte[] charsmap, String[] texts,
                                                 float[] scores, int[] types) throws IOException {
        Map<String, Object> md = new LinkedHashMap<>();
        md.put(GgufKeys.TOKENIZER_MODEL, "t5");
        md.put(GgufKeys.TOKENIZER_PRE, "default");
        md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(Arrays.asList(texts)));
        md.put(GgufKeys.TOKENIZER_SCORES, floats(scores));
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, ints(types));
        md.put(GgufKeys.TOKENIZER_TOKEN_TYPE_COUNT, ctx.config().findInt(1, "type_vocab_size"));
        if (charsmap.length > 0) {
            md.put(GgufKeys.TOKENIZER_PRECOMPILED_CHARSMAP, ints(charsmap));
        }
        specialVocab(ctx, texts.length).writeTo(md);
        return md;
    }

    private static HfSpecialVocab specialVocab(HfLoadContext ctx, int size) throws IOException {
        return HfSpecialVocab.load(ctx.tokenizerFiles(), ctx.tokenizerFiles().readTokenizerJson(),
                ctx.config().raw(), false, size);
    }

    private static List<Integer> ints(int[] values) {
        List<Integer> out = new ArrayList<>(values.length);
        for (int v : values) {
            out.add(v);
        }
        return out;
    }

    private static List<Integer> ints(byte[] bytes) {
        List<Integer> out = new ArrayList<>(bytes.length);
        for (byte b : bytes) {
            out.add(b & 0xFF);
        }
        return out;
    }

    private static List<Float> floats(float[] scores) {
        List<Float> out = new ArrayList<>(scores.length);
        for (float v : scores) {
            out.add(v);
        }
        return out;
    }

    /** The XLM-R entry's step-1 arch switch (bert.py:478-481): JINA_BERT_V3 with task adapters, else BERT. */
    public static Arch xlmRobertaArch(HfConfig cfg, HfTensorIndex index) {
        Optional<Object> loras = cfg.find("lora_adaptations");
        return loras.isPresent() && loras.get() instanceof List ? Arch.JINA_BERT_V3 : Arch.BERT;
    }

    /** The XLM-R entry's vocabulary reference: {@link #xlmRobertaVocab} without the NomicBert mask piece. */
    public static Map<String, Object> xlmRobertaVocabEntry(HfLoadContext ctx, PreNameResolver pre)
            throws IOException {
        return xlmRobertaVocab(ctx, pre, false);
    }

    // ------------------------------------------------------------------ the NomicBert tokenizer probe

    /** {@code _is_tokenizer_xlmroberta} (bert.py:395-403): a Unigram tokenizer.json means the XLM-R tokenizer. */
    public static boolean isTokenizerXlmRoberta(HfLoadContext ctx) throws IOException {
        HfTokenizerJson tj = ctx.tokenizerFiles().readTokenizerJson();
        if (tj == null) {
            throw new GgufFormatException("NomicBertModel reads tokenizer.json, which the checkpoint lacks");
        }
        switch (tj.model().type()) {
            case UNIGRAM:
                return true;
            case WORD_PIECE:
                return false;
            default:
                throw new GgufFormatException(tj.source() + ": unknown tokenizer: "
                        + tj.model().type().jsonName());
        }
    }
}
