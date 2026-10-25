package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfRope;
import com.eonmux.neuralfree.gguf.hf.meta.HfGenericMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.RowTransforms;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerJson;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfVocabWriters;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.SentencePieceModel;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.function.UnaryOperator;

/**
 * The hooks of three dense converter classes without a shared parent besides {@code TextModel}:
 * {@code Exaone4Model} (LC/conversion/exaone.py:65-129), {@code Ernie4_5Model} (LC/conversion/ernie.py:19-75) and
 * {@code Phi3MiniModel} (LC/conversion/phi.py:41-211), plus the vocabulary input and sliding-window helpers the
 * Gemma hooks of this package share.
 */
public final class HfBaseDenseHooks {

    private static final Logger LOG = LogManager.getLogger(HfBaseDenseHooks.class);

    private static final String TOKEN_EMBD = "TOKEN_EMBD";
    private static final String ROPE_FREQS = "ROPE_FREQS";
    private static final String ROPE_FACTORS_LONG = "ROPE_FACTORS_LONG";
    private static final String ROPE_FACTORS_SHORT = "ROPE_FACTORS_SHORT";
    private static final String WEIGHT = ".weight";
    private static final String SLIDING_ATTENTION = "sliding_attention";
    private static final String LAYER_TYPES = "layer_types";

    /** {@code SentencePieceTokenTypes} (LC/conversion/base.py), equal to {@code gguf.TokenType}. */
    private static final int NORMAL = 1;
    private static final int UNKNOWN = 2;
    private static final int CONTROL = 3;
    private static final int USER_DEFINED = 4;
    private static final int UNUSED = 5;
    private static final int BYTE = 6;
    private static final float PAD_SCORE = -10000f;
    private static final float ADDED_SCORE = -1000f;

    private HfBaseDenseHooks() {
    }

    // ------------------------------------------------------------------ shared helpers

    /**
     * {@code [t == "sliding_attention" for t in hparams["layer_types"]]}, required.
     *
     * @throws GgufFormatException when {@code layer_types} is missing or not an array
     */
    static List<Boolean> slidingFlags(HfConfig cfg) {
        Object v = cfg.flat().get(LAYER_TYPES);
        if (!(v instanceof List)) {
            throw new GgufFormatException(cfg.source() + ": '" + LAYER_TYPES + "' must be an array of layer types, got "
                    + (v == null ? "nothing" : v.getClass().getSimpleName()));
        }
        List<Boolean> out = new ArrayList<>();
        for (Object t : (List<?>) v) {
            out.add(SLIDING_ATTENTION.equals(t));
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * A required floating-point hyperparameter ({@code self.find_hparam(aliases)}).
     *
     * @throws GgufFormatException when no alias has a value
     */
    static double requireFloat(HfConfig cfg, String... aliases) {
        OptionalDouble v = cfg.findFloat(aliases);
        if (!v.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key " + Arrays.toString(aliases)
                    + " is missing");
        }
        return v.getAsDouble();
    }

    /** The integral value {@code v} of {@code where}, as an {@code int}. */
    static int toInt(Object v, String where) {
        if (!(v instanceof Number)) {
            throw new GgufFormatException(where + " must be an integer, got "
                    + (v == null ? "null" : v.getClass().getSimpleName()));
        }
        double d = ((Number) v).doubleValue();
        if (d != Math.rint(d) || d < Integer.MIN_VALUE || d > Integer.MAX_VALUE) {
            throw new GgufFormatException(where + " must be an integer, got " + v);
        }
        return (int) d;
    }

    /** The number {@code v}, or {@code dflt} when {@code v} is null. */
    static double numberOr(Object v, double dflt, String where) {
        if (v == null) {
            return dflt;
        }
        if (!(v instanceof Number)) {
            throw new GgufFormatException(where + " must be a number, got " + v.getClass().getSimpleName());
        }
        return ((Number) v).doubleValue();
    }

    /** {@code map_tensor_name(name)} then the tensor as is ({@code TextModel.modify_tensors}). */
    private static void plan(HfLoadContext ctx, String hfName, RowSource src, TensorSink out) throws IOException {
        HfTensorHook.DEFAULT.apply(ctx, hfName, src, out);
    }

    // ------------------------------------------------------------------ Exaone4Model

    /** {@code Exaone4Model.set_vocab} (LC/conversion/exaone.py:68-76): the {@code _set_vocab_gpt2} vocabulary. */
    public static Map<String, Object> exaone4Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        return HfVocabWriters.gpt2(HfHookSupport.vocabInput(ctx), pre);
    }

    /**
     * {@code Exaone4Model.set_gguf_parameters} (LC/conversion/exaone.py:78-96): the generic metadata, the vocabulary
     * size and, with a sliding window, the window and its per-layer pattern from {@code layer_types}, else from
     * {@code sliding_window_pattern} ({@code "LLLG"}-style string or period), written only when it covers every
     * layer.
     */
    public static void exaone4Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        HfGenericMetadata.apply(ctx, kv);
        kv.set(GgufKeys.VOCAB_SIZE, cfg.requireInt("vocab_size"));
        if (!cfg.find("sliding_window").isPresent()) {
            return;
        }
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.requireInt("sliding_window"));
        Map<String, Object> flat = cfg.flat();
        if (flat.containsKey(LAYER_TYPES)) {
            kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, slidingFlags(cfg));
        } else if (flat.containsKey("sliding_window_pattern")) {
            int nLayers = cfg.requireInt("num_hidden_layers");
            List<Boolean> pattern = exaone4Pattern(cfg, flat.get("sliding_window_pattern"), nLayers);
            if (pattern.size() == nLayers) {
                kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
            }
        }
    }

    /** The per-layer flags of a {@code sliding_window_pattern} string or period; empty for any other value. */
    private static List<Boolean> exaone4Pattern(HfConfig cfg, Object p, int nLayers) {
        List<Boolean> out = new ArrayList<>(nLayers);
        if (p instanceof String) {
            String s = (String) p;
            if (s.isEmpty()) {
                throw new GgufFormatException(cfg.source() + ": 'sliding_window_pattern' is an empty string");
            }
            for (int i = 0; i < nLayers; i++) {
                out.add(s.charAt(i % s.length()) == 'L');
            }
        } else if (p instanceof Integer || p instanceof Long) {
            long period = ((Number) p).longValue();
            if (period == 0) {
                throw new GgufFormatException(cfg.source() + ": 'sliding_window_pattern' is 0");
            }
            for (int i = 0; i < nLayers; i++) {
                out.add(Math.floorMod(i + 1L, period) != 0);
            }
        }
        return out;
    }

    /**
     * {@code Exaone4Model.generate_extra_tensors} (LC/conversion/exaone.py:98-129): llama3 {@code rope_freqs} when the
     * global RoPE type is {@code llama3}, with the converter's defaults (theta 10000, factor 16, low 1, high 4,
     * original context 8192).
     */
    public static void exaone4Extra(HfLoadContext ctx, TensorSink out) {
        HfConfig cfg = ctx.config();
        HfRope.Params p = cfg.rope().global();
        if (p.raw().isEmpty() || !"llama3".equals(p.type().orElse("").toLowerCase(Locale.ROOT))) {
            return;
        }
        OptionalInt headDim = cfg.findInt("head_dim");
        int dim = headDim.isPresent() ? headDim.getAsInt()
                : cfg.requireInt("hidden_size") / cfg.requireInt("num_attention_heads");
        out.add(ctx.names().ggufName(ROPE_FREQS, -1, WEIGHT), RopeTensors.llama3Freqs(p.theta().orElse(10000.0),
                dim, 0, p.factor().orElse(16.0), p.lowFreqFactor().orElse(1.0), p.highFreqFactor().orElse(4.0),
                p.originalMaxPositionEmbeddings().orElse(8192), false));
    }

    // ------------------------------------------------------------------ Ernie4_5Model

    /** {@code Ernie4_5Model.filter_tensors} (LC/conversion/ernie.py:36-42): {@code ernie.} becomes {@code model.}. */
    public static String ernie45Filter(String name, UnaryOperator<String> base) {
        return base.apply(name.contains("ernie.") ? name.replace("ernie.", "model.") : name);
    }

    /**
     * {@code Ernie4_5Model.modify_tensors} (LC/conversion/ernie.py:44-75): the fused {@code qkv_proj} is split into
     * {@code [nh*hd, nkv*hd, nkv*hd]} rows, the fused {@code up_gate_proj} into halves (gate first, then up).
     */
    public static void ernie45Tensors(HfLoadContext ctx, String hfName, RowSource src, TensorSink out)
            throws IOException {
        if (out.isConsumed(hfName)) {
            return;
        }
        HfConfig cfg = ctx.config();
        if (hfName.contains("qkv_proj")) {
            int nHead = cfg.requireInt("num_attention_heads");
            int nKv = cfg.requireInt("num_key_value_heads");
            OptionalInt hd = cfg.findInt("head_dim");
            long headDim = hd.isPresent() ? hd.getAsInt() : cfg.requireInt("hidden_size") / nHead;
            RowSource[] qkv = RowTransforms.splitRows(src, nHead * headDim, nKv * headDim, nKv * headDim);
            plan(ctx, hfName.replace("qkv_proj", "q_proj"), qkv[0], out);
            plan(ctx, hfName.replace("qkv_proj", "k_proj"), qkv[1], out);
            plan(ctx, hfName.replace("qkv_proj", "v_proj"), qkv[2], out);
        } else if (hfName.contains("up_gate_proj")) {
            long[] shape = src.shape();
            if (shape.length == 0 || shape[0] % 2 != 0) {
                throw new GgufFormatException("tensor '" + hfName + "' " + Arrays.toString(shape)
                        + " cannot be split into equal gate and up halves");
            }
            long half = shape[0] / 2;
            RowSource[] gateUp = RowTransforms.splitRows(src, half, half);
            plan(ctx, hfName.replace("up_gate_proj", "gate_proj"), gateUp[0], out);
            plan(ctx, hfName.replace("up_gate_proj", "up_proj"), gateUp[1], out);
        } else {
            plan(ctx, hfName, src, out);
        }
    }

    /**
     * {@code Ernie4_5Model.set_vocab} (LC/conversion/ernie.py:22-30): the SentencePiece vocabulary, with
     * {@code add_space_prefix} from tokenizer_config.json {@code add_prefix_space} when it is set.
     */
    public static Map<String, Object> ernie45Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        Map<String, Object> base = HfVocabWriters.sentencePiece(HfHookSupport.vocabInput(ctx), pre);
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return base;
        }
        JsonView tc = files.readTokenizerConfig();
        if (!tc.has("add_prefix_space")) {
            return base;
        }
        Map<String, Object> md = new LinkedHashMap<>(base);
        md.put(GgufKeys.TOKENIZER_ADD_PREFIX, tc.bool("add_prefix_space"));
        return Collections.unmodifiableMap(md);
    }

    // ------------------------------------------------------------------ Phi3MiniModel

    /**
     * {@code Phi3MiniModel.set_gguf_parameters} (LC/conversion/phi.py:148-173), which does not call the generic
     * metadata: context, original context, width, FFN width, blocks, heads, KV heads, RMS epsilon, RoPE dimensions
     * {@code int(rot_pct * n_embd) // n_head}, the global RoPE base, and the sliding window (0 when absent, which
     * tells Phi-4 apart).
     */
    public static void phi3Meta(HfLoadContext ctx, HfMetadataBuilder kv, HfPlanView plan) {
        HfConfig cfg = ctx.config();
        int nEmbd = cfg.requireInt("hidden_size", "n_embd");
        int nHead = cfg.requireInt("num_attention_heads", "n_head");
        int nHeadKv = cfg.requireInt("num_key_value_heads", "n_head_kv");
        double rmsEps = requireFloat(cfg, "rms_norm_eps");
        int maxPos = cfg.requireInt("n_positions", "max_position_embeddings");
        int origMaxPos = phi3OriginalContext(cfg);
        double rotPct = numberOr(cfg.rope().raw().get("partial_rotary_factor"), 1.0,
                cfg.source() + ": rope_parameters.partial_rotary_factor");
        int ropeDims = (int) (rotPct * nEmbd) / nHead;
        OptionalDouble theta = cfg.rope().global().theta();
        if (!theta.isPresent()) {
            throw new GgufFormatException(cfg.source() + ": required key rope_parameters.rope_theta is missing");
        }
        kv.set(GgufKeys.CONTEXT_LENGTH, maxPos);
        kv.set(GgufKeys.ROPE_SCALING_ORIG_CTX_LEN, origMaxPos);
        kv.set(GgufKeys.EMBEDDING_LENGTH, nEmbd);
        kv.set(GgufKeys.FEED_FORWARD_LENGTH, cfg.requireInt("intermediate_size"));
        kv.set(GgufKeys.BLOCK_COUNT, ctx.blockCount());
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT, nHead);
        kv.set(GgufKeys.ATTENTION_HEAD_COUNT_KV, nHeadKv);
        kv.set(GgufKeys.ATTENTION_LAYERNORM_RMS_EPS, rmsEps);
        kv.set(GgufKeys.ROPE_DIMENSION_COUNT, ropeDims);
        kv.set(GgufKeys.ROPE_FREQ_BASE, theta.getAsDouble());
        kv.set(GgufKeys.ATTENTION_SLIDING_WINDOW, cfg.findInt(0, "sliding_window"));
    }

    /** {@code self.rope_parameters["original_max_position_embeddings"]}, required and positive. */
    private static int phi3OriginalContext(HfConfig cfg) {
        String where = cfg.source() + ": rope_parameters.original_max_position_embeddings";
        Object v = cfg.rope().raw().get("original_max_position_embeddings");
        if (v == null) {
            throw new GgufFormatException(where + " is required");
        }
        int orig = toInt(v, where);
        if (orig <= 0) {
            throw new GgufFormatException(where + " must be positive, got " + orig);
        }
        return orig;
    }

    /**
     * {@code Phi3MiniModel.generate_extra_tensors} (LC/conversion/phi.py:175-211): with LongRoPE {@code long_factor},
     * the {@code rope.scaling.attn_factor} key ({@code s = max_pos / orig}: {@code sqrt(1 + ln s / ln orig)} for
     * {@code su}/{@code longrope}, {@code 0.1 ln s + 1} for {@code yarn}, 1.0 when {@code s <= 1}) and the
     * {@code rope_factors_long}/{@code rope_factors_short} tensors.
     *
     * @throws GgufFormatException when the RoPE type is missing or unsupported, or the factor lists do not fit
     */
    public static void phi3Extra(HfLoadContext ctx, TensorSink out) {
        HfConfig cfg = ctx.config();
        Map<String, Object> rope = cfg.rope().raw();
        Object longFactors = rope.get("long_factor");
        if (!(longFactors instanceof List) || ((List<?>) longFactors).isEmpty()) {
            return;
        }
        int maxPos = cfg.requireInt("n_positions", "max_position_embeddings");
        int orig = phi3OriginalContext(cfg);
        double scale = (double) maxPos / orig;
        Object typeValue = rope.get("rope_type");
        String type = typeValue == null ? "" : String.valueOf(typeValue).toLowerCase(Locale.ROOT);
        if (type.isEmpty()) {
            throw new GgufFormatException(cfg.source() + ": missing the required key rope_scaling.type");
        }
        double attnFactor;
        if (type.equals("su") || type.equals("longrope")) {
            if (scale > 1.0 && orig == 1) {
                throw new GgufFormatException(cfg.source() + ": the LongRoPE attention factor needs an original"
                        + " context above 1");
            }
            attnFactor = scale > 1.0 ? Math.sqrt(1 + Math.log(scale) / Math.log(orig)) : 1.0;
        } else if (type.equals("yarn")) {
            attnFactor = scale > 1.0 ? 0.1 * Math.log(scale) + 1.0 : 1.0;
        } else {
            throw new GgufFormatException(cfg.source() + ": the rope scaling type " + type + " is not supported");
        }
        out.addKv(String.format(Locale.ROOT, GgufKeys.ROPE_SCALING_ATTN_FACTOR, ctx.arch().ggufName()),
                GGUFValueType.FLOAT32, (float) attnFactor);
        RowSource longF = RopeTensors.longropeFactors(cfg, RopeTensors.Factors.LONG).orElseThrow();
        RowSource shortF = RopeTensors.longropeFactors(cfg, RopeTensors.Factors.SHORT).orElseThrow();
        out.add(ctx.names().ggufName(ROPE_FACTORS_LONG, -1, WEIGHT), longF);
        out.add(ctx.names().ggufName(ROPE_FACTORS_SHORT, -1, WEIGHT), shortF);
    }

    /**
     * {@code Phi3MiniModel.set_vocab} (LC/conversion/phi.py:44-146): the GPT-2 vocabulary when tokenizer_config.json
     * names {@code GPT2Tokenizer} (Phi-4), else the class's own SentencePiece list: {@code vocab_size} slots of
     * {@code [PAD<i>]}, every piece, then {@code added_tokens.json}, tokenizer_config {@code added_tokens_decoder}
     * and tokenizer.json {@code added_tokens} in turn (score -1000, CONTROL when special, else USER_DEFINED, the text
     * kept as written).
     *
     * @throws GgufFormatException when the checkpoint has no {@code tokenizer.model}, or an added token id is outside
     *                             the vocabulary
     */
    public static Map<String, Object> phi3Vocab(HfLoadContext ctx, PreNameResolver pre) throws IOException {
        HfTokenizerFiles files = ctx.tokenizerFiles();
        if (files.hasGgufTokenizer()) {
            return HfVocabWriters.fromGguf(files);
        }
        HfVocabWriters.Input in = HfHookSupport.vocabInput(ctx);
        Object tokenizerClass = files.readTokenizerConfig().raw("tokenizer_class");
        if ("GPT2Tokenizer".equals(tokenizerClass)) {
            return HfVocabWriters.gpt2(in, pre);
        }
        Path model = files.file(HfTokenizerFiles.TOKENIZER_MODEL);
        if (model == null) {
            throw new GgufFormatException("Error: Missing " + HfTokenizerFiles.TOKENIZER_MODEL);
        }
        Map<String, Object> md = new LinkedHashMap<>(HfVocabWriters.sentencePiece(in, pre));
        int size = ((List<?>) md.get(GgufKeys.TOKENIZER_LIST)).size();
        Phi3Tokens tokens = new Phi3Tokens(size);
        tokens.addPieces(SentencePieceModel.read(model), model.toString());
        for (Map.Entry<String, Integer> e : files.readAddedTokens().entrySet()) {
            tokens.addAddedJson(e.getKey(), e.getValue());
        }
        JsonView tc = files.readTokenizerConfig();
        Object decoder = tc.raw("added_tokens_decoder");
        if (decoder instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) decoder).entrySet()) {
                String where = tc.where("added_tokens_decoder") + "." + e.getKey();
                if (!(e.getValue() instanceof Map) || !(((Map<?, ?>) e.getValue()).get("content") instanceof String)) {
                    throw new GgufFormatException(where + ": expected an object with a string content");
                }
                Map<?, ?> data = (Map<?, ?>) e.getValue();
                tokens.addSpecial(parseId(String.valueOf(e.getKey()), where), (String) data.get("content"),
                        truthy(data.get("special")), where);
            }
        }
        HfTokenizerJson tj = files.readTokenizerJson();
        if (tj != null) {
            for (HfTokenizerJson.AddedToken t : tj.addedTokens()) {
                tokens.addSpecial(t.id(), t.content(), t.special(), tj.source() + ": added token " + t.id());
            }
        }
        tokens.writeTo(md);
        return Collections.unmodifiableMap(md);
    }

    private static int parseId(String key, String where) {
        try {
            return Integer.parseInt(key);
        } catch (NumberFormatException e) {
            throw new GgufFormatException(where + ": '" + key + "' is not a token id", e);
        }
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
            return ((Number) v).doubleValue() != 0;
        }
        if (v instanceof String) {
            return !((String) v).isEmpty();
        }
        if (v instanceof List) {
            return !((List<?>) v).isEmpty();
        }
        return !(v instanceof Map) || !((Map<?, ?>) v).isEmpty();
    }

    /** The token list {@code Phi3MiniModel.set_vocab} builds. Mutable while it is built; one per call. */
    private static final class Phi3Tokens {
        private final String[] texts;
        private final float[] scores;
        private final int[] types;

        Phi3Tokens(int size) {
            texts = new String[size];
            scores = new float[size];
            types = new int[size];
            for (int i = 0; i < size; i++) {
                texts[i] = "[PAD" + i + "]";
            }
            Arrays.fill(scores, PAD_SCORE);
            Arrays.fill(types, UNUSED);
        }

        void addPieces(SentencePieceModel sp, String where) {
            if (sp.pieceCount() > texts.length) {
                throw new GgufFormatException(where + ": " + sp.pieceCount() + " pieces do not fit the vocabulary"
                        + " size " + texts.length);
            }
            for (int id = 0; id < sp.pieceCount(); id++) {
                texts[id] = sp.piece(id);
                scores[id] = sp.score(id);
                types[id] = pieceType(sp.type(id));
            }
        }

        void addAddedJson(String token, int id) {
            if (id < 0 || id >= texts.length) {
                LOG.debug("ignore token {}: id is out of range, max={}", id, texts.length - 1);
                return;
            }
            texts[id] = token;
            scores[id] = ADDED_SCORE;
            types[id] = USER_DEFINED;
        }

        void addSpecial(int id, String token, boolean special, String where) {
            if (id < 0 || id >= texts.length) {
                throw new GgufFormatException(where + ": token id " + id + " is outside the vocabulary of "
                        + texts.length + " tokens");
            }
            if (types[id] != UNUSED && !texts[id].equals(token)) {
                LOG.warn("replacing token {}: '{}' -> '{}'", id, texts[id], token);
            }
            texts[id] = token;
            scores[id] = ADDED_SCORE;
            types[id] = special ? CONTROL : USER_DEFINED;
        }

        void writeTo(Map<String, Object> md) {
            List<String> t = new ArrayList<>(texts.length);
            List<Float> s = new ArrayList<>(texts.length);
            List<Integer> ty = new ArrayList<>(texts.length);
            for (int i = 0; i < texts.length; i++) {
                t.add(texts[i]);
                s.add(scores[i]);
                ty.add(types[i]);
            }
            md.put(GgufKeys.TOKENIZER_LIST, Collections.unmodifiableList(t));
            md.put(GgufKeys.TOKENIZER_SCORES, Collections.unmodifiableList(s));
            md.put(GgufKeys.TOKENIZER_TOKEN_TYPE, Collections.unmodifiableList(ty));
        }

        /** The type the class gives a piece: UNKNOWN, CONTROL, UNUSED and BYTE kept, every other piece NORMAL. */
        private static int pieceType(SentencePieceModel.PieceType type) {
            switch (type) {
                case UNKNOWN: return UNKNOWN;
                case CONTROL: return CONTROL;
                case UNUSED: return UNUSED;
                case BYTE: return BYTE;
                default: return NORMAL;
            }
        }
    }
}
