package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.names.HfTensorFilters;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Hooks of the dense Qwen converter classes other families build on (LC/conversion/qwen.py): {@code Qwen2Model}
 * (:49-78), {@code Qwen3Model} (:164-265) with its reranker and voyage-4-nano heads, and the {@code _QwenMtpMixin}
 * (:287-377) multi-token-prediction wiring of Qwen3-Next and Qwen3.5. Also the vocabulary input every Qwen hook builds
 * and {@code _try_set_pooling_type} (LC/conversion/base.py:2510-2562).
 *
 * <p>Every method is a public static hook reference that subclasses compose where the converter calls
 * {@code super()}. The MTP name filter and block count are per-load factories ({@link #mtpFilter},
 * {@link #mtpBlockCount}) because the converter's filter reads instance state.</p>
 */
public final class HfQwenHooks {

    /** The headless Qwen2 class, whose tensors lack the {@code model.} prefix (qwen.py:76-77). */
    public static final String QWEN2_HEADLESS = "Qwen2Model";
    /** The inner text class of Qwen3.5, which never owns the {@code mtp.*} tensors (qwen.py:679-686). */
    public static final String QWEN35_TEXT_MODEL = "Qwen3_5TextModel";
    /** The InternS1 checkpoints whose vocabulary keeps the U+2581 added tokens (qwen.py:208-213, :278-283). */
    public static final String INTERN_S1 = "InternS1ForConditionalGeneration";
    /** The {@code tokenizer_class} of the HybridDNA tokenizer (base.py:2176-2181). */
    public static final String HYBRID_DNA_TOKENIZER = "HybridDNATokenizer";

    /** The Qwen3 reranker chat template (qwen.py:232-237). */
    static final String RERANK_TEMPLATE = "<|im_start|>system\nJudge whether the Document meets the requirements"
            + " based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            + "<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that"
            + " answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    /** The name of the reranker chat template. */
    static final String RERANK_TEMPLATE_NAME = "rerank";
    /** The reranker's classifier labels, in {@code cls.output} row order (qwen.py:231). */
    static final List<String> RERANK_LABELS = Collections.unmodifiableList(Arrays.asList("yes", "no"));

    /** The voyage-4-nano output projection (qwen.py:261). */
    static final String VOYAGE_LINEAR = "linear.weight";
    /** The converter name the voyage-4-nano projection is mapped under (qwen.py:263). */
    static final String VOYAGE_DENSE = "dense_2_out.weight";

    private static final String README = "README.md";
    private static final String MODULES_JSON = "modules.json";
    private static final String EMBEDDING_CONFIG_JSON = "embedding_config.json";
    private static final String MTP_PREFIX = "mtp.";
    private static final String MODEL_MTP_PREFIX = "model.mtp.";
    private static final String MODEL_LAYERS = "model.layers.";
    private static final String RERANK_STATE = HfQwenHooks.class.getName() + ".reranker";

    /** {@code _QwenMtpMixin.filter_tensors}' renames of the MTP head's tensors in layer {@code n} (qwen.py:337-342). */
    private static final Map<String, String> MTP_HEAD_RENAMES;

    static {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("fc", "eh_proj");
        m.put("pre_fc_norm_embedding", "enorm");
        m.put("pre_fc_norm_hidden", "hnorm");
        m.put("norm", "shared_head.norm");
        MTP_HEAD_RENAMES = Collections.unmodifiableMap(m);
    }

    private HfQwenHooks() {
    }

    // ------------------------------------------------------------------ shared inputs

    /**
     * The root {@code architectures[0]} of the checkpoint, which the Qwen classes keep as {@code origin_hf_arch}
     * (qwen.py:178-180, :273-276); the empty string when absent.
     */
    public static String originArchitecture(HfConfig cfg) {
        Object archs = cfg.raw().raw("architectures");
        if (archs instanceof List && !((List<?>) archs).isEmpty() && ((List<?>) archs).get(0) instanceof String) {
            return (String) ((List<?>) archs).get(0);
        }
        return "";
    }

    // ------------------------------------------------------------------ Qwen2Model

    /**
     * {@code Qwen2Model.set_vocab} (qwen.py:61-69): the HybridDNA vocabulary when {@code tokenizer_config.json} names
     * {@value #HYBRID_DNA_TOKENIZER}, else SentencePiece when {@code tokenizer.model} exists, else GPT-2 BPE.
     */
    public static Map<String, Object> qwen2Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        if (isHybridDna(ctx.tokenizerFiles())) {
            return HfVocabWriters.hybridDna(in, pre);
        }
        return HfVocabWriters.canSentencePiece(in) ? HfVocabWriters.sentencePiece(in, pre)
                : HfVocabWriters.gpt2(in, pre);
    }

    /** {@code _is_hybriddna} (base.py:2176-2181). */
    static boolean isHybridDna(HfTokenizerFiles files) throws IOException {
        return HYBRID_DNA_TOKENIZER.equals(files.readTokenizerConfig().raw("tokenizer_class"));
    }

    /** {@code Qwen2Model.set_gguf_parameters} (qwen.py:71-73): the generic metadata, then the pooling type. */
    public static void qwen2Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        HfGenericMetadata.apply(ctx, kv);
        pooling(ctx, kv);
    }

    /**
     * {@code Qwen2Model.modify_tensors} (qwen.py:75-78): the headless {@value #QWEN2_HEADLESS} gets the {@code model.}
     * prefix of {@code Qwen2ForCausalLM}; then the default mapping.
     */
    public static void qwen2Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        String name = QWEN2_HEADLESS.equals(ctx.config().architecture()) ? "model." + hfName : hfName;
        HfTensorHook.DEFAULT.apply(ctx, name, src, out);
    }

    /**
     * {@code _try_set_pooling_type} (base.py:2510-2562): the pooling of the sentence-transformers {@code *Pooling}
     * module named by {@code modules.json}, else {@code embedding_config.json}'s {@code last_token} pooling; then
     * the classifier pooling and activation of the configuration.
     *
     * @throws GgufFormatException when a side file is malformed or names an unsupported pooling or activation
     * @throws IOException         when a side file cannot be read or its name escapes the model directory
     */
    public static void pooling(HfLoadContext ctx, HfMetadataBuilder kv) throws IOException {
        HfConfig cfg = ctx.config();
        String poolingPath = poolingModulePath(cfg);
        if (poolingPath != null) {
            Path file = cfg.sideFile(poolingPath.isEmpty() ? HfConfig.CONFIG_JSON
                    : poolingPath + "/" + HfConfig.CONFIG_JSON);
            Map<?, ?> pooling = readObject(file);
            kv.set(GgufKeys.POOLING_TYPE, poolingMode(pooling).ggufCode());
        } else {
            Path file = cfg.sideFile(EMBEDDING_CONFIG_JSON);
            if (Files.isRegularFile(file)) {
                Object mode = readObject(file).get("pooling");
                if ("last_token".equals(mode)) {
                    kv.set(GgufKeys.POOLING_TYPE, PoolingKind.LAST.ggufCode());
                } else if (mode != null) {
                    throw new GgufFormatException(file + ": unsupported embedding_config.json pooling "
                            + (mode instanceof String ? "'" + mode + "'" : mode));
                }
            }
        }
        Optional<String> classifierPooling = cfg.findString("classifier_pooling");
        if (classifierPooling.isPresent()) {
            String p = classifierPooling.get();
            if (!p.equals("cls") && !p.equals("mean")) {
                throw new GgufFormatException(cfg.source() + ": unsupported classifier_pooling: " + p);
            }
            kv.set(GgufKeys.CLASSIFIER_POOLING_TYPE,
                    (p.equals("cls") ? PoolingKind.CLS : PoolingKind.MEAN).ggufCode());
        }
        Optional<String> activation = cfg.findString("classifier_activation");
        if (activation.isPresent()) {
            String a = activation.get();
            if (!a.equals("gelu") && !a.equals("silu") && !a.equals("tanh")) {
                throw new GgufFormatException(cfg.source() + ": unsupported classifier_activation: " + a);
            }
            kv.set(GgufKeys.CLASSIFIER_ACTIVATION, a);
        }
    }

    /** The {@code path} of the first {@code modules.json} module whose type ends with {@code Pooling}, or null. */
    private static String poolingModulePath(HfConfig cfg) throws IOException {
        Path file = cfg.sideFile(MODULES_JSON);
        if (!Files.isRegularFile(file)) {
            return null;
        }
        Object parsed = readJson(file);
        if (!(parsed instanceof List)) {
            throw new GgufFormatException(file + ": expected an array of modules");
        }
        List<?> modules = (List<?>) parsed;
        for (int i = 0; i < modules.size(); i++) {
            Object m = modules.get(i);
            Object type = m instanceof Map ? ((Map<?, ?>) m).get("type") : null;
            if (!(type instanceof String)) {
                throw new GgufFormatException(file + ": module " + i + " has no string 'type'");
            }
            if (((String) type).endsWith("Pooling")) {
                Object path = ((Map<?, ?>) m).get("path");
                if (!(path instanceof String)) {
                    throw new GgufFormatException(file + ": pooling module " + i + " has no string 'path'");
                }
                return (String) path;
            }
        }
        return null;
    }

    /** The pooling a sentence-transformers {@code Pooling} configuration selects (base.py:2532-2541). */
    private static PoolingKind poolingMode(Map<?, ?> pooling) {
        if (truthy(pooling.get("pooling_mode_mean_tokens"))) {
            return PoolingKind.MEAN;
        }
        if (truthy(pooling.get("pooling_mode_cls_token"))) {
            return PoolingKind.CLS;
        }
        if (truthy(pooling.get("pooling_mode_lasttoken"))) {
            return PoolingKind.LAST;
        }
        Object mode = pooling.get("pooling_mode");
        if ("mean".equals(mode)) {
            return PoolingKind.MEAN;
        }
        if ("cls".equals(mode)) {
            return PoolingKind.CLS;
        }
        if ("lasttoken".equals(mode)) {
            return PoolingKind.LAST;
        }
        throw new GgufFormatException("Only MEAN, CLS, and LAST pooling types supported (pooling configuration "
                + pooling + ")");
    }

    /** Python truthiness of a JSON value. */
    private static boolean truthy(Object v) {
        if (v == null) {
            return false;
        }
        if (v instanceof Boolean) {
            return (Boolean) v;
        }
        if (v instanceof Number) {
            return ((Number) v).doubleValue() != 0.0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        return !(v instanceof Map) || !((Map<?, ?>) v).isEmpty();
    }

    private static Map<?, ?> readObject(Path file) throws IOException {
        Object parsed = readJson(file);
        if (!(parsed instanceof Map)) {
            throw new GgufFormatException(file + ": expected a JSON object");
        }
        return (Map<?, ?>) parsed;
    }

    private static Object readJson(Path file) throws IOException {
        try {
            return Json.parse(file, HfConfigLoader.MAX_CONFIG_BYTES, Json.Limits.LENIENT);
        } catch (JsonException e) {
            throw new GgufFormatException(e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------ Qwen3Model

    /**
     * {@code Qwen3Model.set_vocab} (qwen.py:208-214): the InternS1 vocabulary for {@value #INTERN_S1} checkpoints,
     * else {@link #qwen2Vocab}.
     */
    public static Map<String, Object> qwen3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        if (INTERN_S1.equals(originArchitecture(ctx.config()))) {
            return HfVocabWriters.internS1(HfHookSupport.vocabInput(ctx), pre);
        }
        return qwen2Vocab(ctx, pre);
    }

    /**
     * {@code Qwen3Model.set_gguf_parameters} (qwen.py:228-238): {@link #qwen2Meta}, then for a reranker the RANK
     * pooling, the {@code [yes, no]} classifier labels and the {@value #RERANK_TEMPLATE_NAME} chat template.
     */
    public static void qwen3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) throws IOException {
        qwen2Meta(ctx, kv, plan);
        if (reranker(ctx).active) {
            kv.set(GgufKeys.POOLING_TYPE, PoolingKind.RANK.ggufCode());
            kv.set(GgufKeys.CLASSIFIER_OUTPUT_LABELS, RERANK_LABELS);
            Map<String, String> template = new LinkedHashMap<>();
            template.put("name", RERANK_TEMPLATE_NAME);
            template.put("template", RERANK_TEMPLATE);
            kv.putAll(HfChatTemplates.toGguf(Collections.singletonList(template)));
        }
    }

    /**
     * {@code Qwen3Model.modify_tensors} (qwen.py:246-265). For a reranker, the output head (the token embedding when
     * {@code tie_word_embeddings}, else {@code lm_head}) yields {@code cls.output.weight}: its {@code yes} and
     * {@code no} rows, in that order; a tied embedding is planned as well, an untied {@code lm_head} is dropped. The
     * voyage-4-nano {@value #VOYAGE_LINEAR} records {@code embedding_length_out} and becomes {@code dense_2}. Then
     * {@link #qwen2Tensors}.
     */
    public static void qwen3Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        Reranker r = reranker(ctx);
        if (r.active) {
            boolean tiedHead = r.tied && hfName.contains("embed_tokens");
            boolean realHead = !r.tied && hfName.contains("lm_head");
            if (tiedHead || realHead) {
                out.add(TensorNames.weight(TensorNames.CLS_OUT), clsOut(r, hfName, src));
                if (tiedHead) {
                    qwen2Tensors(ctx, hfName, src, out);
                } else {
                    out.drop(hfName, "the Qwen3 reranker keeps only the yes/no rows of the output head");
                }
                return;
            }
        }
        String name = hfName;
        if (VOYAGE_LINEAR.equals(name)) {
            long[] shape = src.shape();
            if (shape.length == 0) {
                throw new GgufFormatException("tensor '" + hfName + "' is a scalar, not an output projection");
            }
            out.addKv(String.format(Locale.ROOT, GgufKeys.EMBEDDING_LENGTH_OUT, ctx.arch().ggufName()),
                    GGUFValueType.UINT32, shape[0]);
            name = VOYAGE_DENSE;
        }
        qwen2Tensors(ctx, name, src, out);
    }

    /** {@code _get_cls_out_tensor} (qwen.py:240-244): {@code stack([head[yes], head[no]])}. */
    private static RowSource clsOut(Reranker r, String hfName, RowSource head) {
        long[] shape = head.shape();
        if (shape.length != 2) {
            throw new GgufFormatException("reranker head '" + hfName + "' has shape " + Arrays.toString(shape)
                    + ", not [vocab, hidden]");
        }
        for (int id : new int[]{r.yes, r.no}) {
            if (id < 0 || id >= shape[0]) {
                throw new GgufFormatException("reranker head '" + hfName + "' has " + shape[0]
                        + " rows; token id " + id + " is outside it");
            }
        }
        return RowTransforms.gatherRows(head, 1, new int[]{r.yes, r.no});
    }

    /**
     * Whether the checkpoint is a Qwen3 reranker ({@code _is_qwen3_reranker}, qwen.py:185-206): its README has a
     * {@code # qwen3-reranker} or {@code # qwen3-vl-reranker} heading, the directory name, {@code _name_or_path},
     * {@code model_type} or root architecture names one, or the root architecture is a sequence classifier.
     * Decided once per load.
     *
     * @throws IOException         when the README or a tokenizer file cannot be read
     * @throws GgufFormatException when a reranker's tokenizer lacks {@code yes} or {@code no}
     */
    public static boolean isQwen3Reranker(HfLoadContext ctx) throws IOException {
        return reranker(ctx).active;
    }

    /** The reranker facts of {@code __init__} and {@code _find_rerank_config} (qwen.py:175-183, :216-226). */
    private static final class Reranker {
        static final Reranker NONE = new Reranker(false, false, -1, -1);

        final boolean active;
        final boolean tied;
        final int yes;
        final int no;

        Reranker(boolean active, boolean tied, int yes, int no) {
            this.active = active;
            this.tied = tied;
            this.yes = yes;
            this.no = no;
        }
    }

    /** The per-load cache of the reranker decision. */
    private static final class RerankerState {
        Reranker value;
    }

    private static Reranker reranker(HfLoadContext ctx) throws IOException {
        RerankerState st = ctx.state(RERANK_STATE, RerankerState.class, RerankerState::new);
        synchronized (st) {
            if (st.value == null) {
                st.value = detectReranker(ctx);
            }
            return st.value;
        }
    }

    private static Reranker detectReranker(HfLoadContext ctx) throws IOException {
        HfConfig cfg = ctx.config();
        if (!rerankerHints(cfg)) {
            return Reranker.NONE;
        }
        HfTokenizerFiles files = ctx.tokenizerFiles();
        boolean tied = cfg.findBool(false, "tie_word_embeddings");
        Reranker r = new Reranker(true, tied, tokenId(files, "yes"), tokenId(files, "no"));
        ctx.log("Qwen3 reranker: cls.output from the " + (tied ? "tied token embedding" : "lm_head")
                + " rows of 'yes' (" + r.yes + ") and 'no' (" + r.no + ")");
        return r;
    }

    /** {@code _is_qwen3_reranker} (qwen.py:185-206). */
    static boolean rerankerHints(HfConfig cfg) throws IOException {
        String readme = readReadme(cfg.dir());
        if (readme.contains("# qwen3-reranker") || readme.contains("# qwen3-vl-reranker")) {
            return true;
        }
        String origin = originArchitecture(cfg);
        Path dirName = cfg.dir().getFileName();
        Object nameOrPath = cfg.flat().get("_name_or_path");
        Object modelType = cfg.flat().get("model_type");
        for (Object hint : new Object[]{dirName, nameOrPath, modelType, origin}) {
            String h = hint == null ? "" : String.valueOf(hint).toLowerCase(Locale.ROOT);
            if (h.contains("qwen3-reranker") || h.contains("qwen3-vl-reranker")) {
                return true;
            }
        }
        return origin.toLowerCase(Locale.ROOT).contains("sequenceclassification");
    }

    /**
     * The lower-cased {@code README.md} of the model directory, read as Python text mode reads it with
     * {@code encoding="utf-8"} ({@link HfTokenizerFiles#readText}), or the empty string when there is none.
     *
     * @throws IOException when the file exceeds {@link HfTokenizerFiles#MAX_FILE_BYTES} or is not valid UTF-8
     */
    private static String readReadme(Path dir) throws IOException {
        Path file = dir.resolve(README);
        return Files.isRegularFile(file) ? HfTokenizerFiles.readText(file).toLowerCase(Locale.ROOT) : "";
    }

    /**
     * {@code tokenizer.convert_tokens_to_ids(token)}: the id of {@code token} in the GGUF tokenizer source, else
     * among the added tokens, else in the model vocabulary.
     *
     * @throws GgufFormatException when the tokenizer has no such token
     */
    static int tokenId(HfTokenizerFiles files, String token) throws IOException {
        if (files.hasGgufTokenizer()) {
            Object tokens = files.ggufTokenizerKeys().get(GgufKeys.TOKENIZER_LIST);
            int id = tokens instanceof List ? ((List<?>) tokens).indexOf(token) : -1;
            if (id >= 0) {
                return id;
            }
            throw missingToken(files, token);
        }
        HfTokenizerJson tj = files.readTokenizerJson();
        if (tj != null) {
            HfTokenizerJson.AddedToken added = tj.addedToken(token);
            if (added != null) {
                return added.id();
            }
        }
        Integer added = files.readAddedTokens().get(token);
        if (added != null) {
            return added;
        }
        if (tj != null) {
            HfTokenizerJson.Model model = tj.model();
            for (int i = 0; i < model.vocabSize(); i++) {
                if (token.equals(model.token(i))) {
                    return i;
                }
            }
        }
        Integer id = files.readVocabJson().get(token);
        if (id != null) {
            return id;
        }
        throw missingToken(files, token);
    }

    private static GgufFormatException missingToken(HfTokenizerFiles files, String token) {
        return new GgufFormatException("Qwen3 reranker: the tokenizer in '" + files.dir() + "' has no token '"
                + token + "'");
    }

    // ------------------------------------------------------------------ _QwenMtpMixin

    /**
     * Whether the MTP layers are kept: the load option ({@code --no-mtp} turns it off) and not the inner
     * {@value #QWEN35_TEXT_MODEL}, which never owns them (qwen.py:679-686).
     */
    public static boolean includeMtp(HfLoadContext ctx) {
        return ctx.options().includeMtp() && !QWEN35_TEXT_MODEL.equals(ctx.config().architecture());
    }

    /**
     * {@code _QwenMtpMixin.__init__}'s block count (qwen.py:306-316): the configuration's layer count
     * ({@link HfConfig#blockCount()}, the first of {@code n_layers, num_hidden_layers, n_layer, num_layers}) plus
     * {@code mtp_num_hidden_layers}; when that is absent or 0, plus the largest {@code mtp.layers.<i>} index + 1 among
     * the checkpoint index names ({@code ctx.index()} before the name filters, as {@code index_tensors} sees them);
     * the layer count alone when the MTP layers are not kept ({@link #includeMtp}).
     *
     * @throws GgufFormatException when the configuration has no layer count, a negative MTP count, or the total
     *                             exceeds {@link HfConfig#MAX_LAYERS}
     */
    public static int mtpBlockCount(HfLoadContext ctx) {
        HfConfig cfg = ctx.config();
        int n = cfg.blockCount();
        if (!includeMtp(ctx)) {
            return n;
        }
        int nMtp = cfg.findInt(0, "mtp_num_hidden_layers");
        if (nMtp < 0) {
            throw new GgufFormatException(cfg.source() + ": mtp_num_hidden_layers " + nMtp + " is negative");
        }
        if (nMtp == 0) {
            nMtp = indexedMtpLayers(ctx.index(), ctx.entry().kind());
        }
        long total = (long) n + nMtp;
        if (total > HfConfig.MAX_LAYERS) {
            throw new GgufFormatException(cfg.source() + ": " + n + " layers and " + nMtp + " MTP layers exceed "
                    + HfConfig.MAX_LAYERS);
        }
        return (int) total;
    }

    /**
     * {@code opt_num_mtp_layers}: the largest {@code mtp.layers.<i>} index + 1 of the names after the inherited filters
     * of {@code kind}, or 0.
     */
    private static int indexedMtpLayers(HfTensorIndex index, HfArchEntry.Kind kind) {
        int count = 0;
        for (String raw : index.names()) {
            String name = HfTensorFilters.filterName(raw, kind);
            if (name == null) {
                continue;
            }
            int idx = mtpLayerIndex(withoutModelPrefix(name));
            if (idx >= 0) {
                count = Math.max(count, idx + 1);
            }
        }
        return count;
    }

    /**
     * {@code _QwenMtpMixin.filter_tensors} (qwen.py:325-357) bound to {@code ctx}: after the inherited filters,
     * {@code model.mtp.} becomes {@code mtp.}; every {@code mtp.*} name is dropped when the MTP layers are not kept
     * ({@link #includeMtp}); {@code mtp.layers.<i>.*} becomes {@code model.layers.<n+i>.*} and {@code mtp.fc},
     * {@code mtp.pre_fc_norm_embedding}, {@code mtp.pre_fc_norm_hidden} and {@code mtp.norm} become
     * {@code eh_proj}, {@code enorm}, {@code hnorm} and {@code shared_head.norm} of layer {@code n}, where {@code n}
     * is the configuration's layer count ({@link HfConfig#blockCount()}).
     *
     * @throws GgufFormatException when the configuration has no layer count, or a renumbered layer exceeds
     *                             {@link HfConfig#MAX_LAYERS}
     */
    public static HfNameFilter mtpFilter(HfLoadContext ctx) {
        int n = ctx.config().blockCount();
        boolean keep = includeMtp(ctx);
        return (name, base) -> {
            String t = base.apply(name);
            return t == null ? null : mtpName(withoutModelPrefix(t), n, keep);
        };
    }

    /** {@code name} with a leading {@code model.mtp.} turned into {@code mtp.}. */
    private static String withoutModelPrefix(String name) {
        return name.startsWith(MODEL_MTP_PREFIX) ? name.substring("model.".length()) : name;
    }

    /** The converter name of a filtered {@code name} (qwen.py:333-347). */
    private static String mtpName(String name, int n, boolean keep) {
        if (!name.startsWith(MTP_PREFIX)) {
            return name;
        }
        if (!keep) {
            return null;
        }
        String[] parts = name.split("\\.", 4);
        int idx = mtpLayerIndex(name);
        if (idx >= 0) {
            return MODEL_LAYERS + layer(n, idx) + "." + parts[3];
        }
        if (parts.length == 3 && MTP_HEAD_RENAMES.containsKey(parts[1])) {
            return MODEL_LAYERS + n + "." + MTP_HEAD_RENAMES.get(parts[1]) + "." + parts[2];
        }
        return name;
    }

    /** The {@code <i>} of {@code mtp.layers.<i>.<rest>}, or -1 for any other name. */
    private static int mtpLayerIndex(String name) {
        if (!name.startsWith(MTP_PREFIX)) {
            return -1;
        }
        String[] parts = name.split("\\.", 4);
        if (parts.length != 4 || !parts[1].equals("layers") || !isDecimal(parts[2])) {
            return -1;
        }
        if (parts[2].length() > 9) {
            throw new GgufFormatException("tensor '" + name + "': MTP layer number " + parts[2] + " is too large");
        }
        return Integer.parseInt(parts[2]);
    }

    private static int layer(int n, int idx) {
        long b = (long) n + idx;
        if (b >= HfConfig.MAX_LAYERS) {
            throw new GgufFormatException("MTP layer " + idx + " after " + n + " layers exceeds the "
                    + HfConfig.MAX_LAYERS + "-layer limit");
        }
        return (int) b;
    }

    private static boolean isDecimal(String s) {
        if (s.isEmpty()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c < '0' || c > '9') {
                return false;
            }
        }
        return true;
    }

    /**
     * {@code _QwenMtpMixin.set_gguf_parameters}' own part (qwen.py:359-364), run after its {@code super()}: when the
     * MTP layers are kept, {@code nextn_predict_layers} = block count - {@code num_hidden_layers} if positive.
     *
     * @throws GgufFormatException when {@code num_hidden_layers} is missing
     */
    public static void mtpMeta(HfLoadContext ctx, HfMetadataBuilder kv) {
        if (!includeMtp(ctx)) {
            return;
        }
        int n = ctx.blockCount() - ctx.config().requireInt("num_hidden_layers");
        if (n > 0) {
            kv.set(GgufKeys.NEXTN_PREDICT_LAYERS, n);
        }
    }
}
