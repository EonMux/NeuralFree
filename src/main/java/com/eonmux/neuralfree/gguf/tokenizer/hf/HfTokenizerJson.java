package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable typed view of a Hugging Face {@code tokenizer.json}: the {@link Model}, the added tokens, and the
 * {@code normalizer}, {@code pre_tokenizer}, {@code post_processor} and {@code decoder} objects as {@link JsonView}s.
 *
 * <p>The model type comes from {@code model.type}; a model without one (an older serialization, as Florence-2 ships)
 * is a Unigram model when its vocabulary is a list and a BPE model otherwise, the order in which the tokenizers
 * library tries the untagged forms. Merges are normalized to {@code "a b"} strings as llama.cpp's
 * {@code SpecialVocab} does (vocab.py:191-222): the pair form {@code ["a", "b"]} is joined with a space after
 * re-encoding each space inside a part as U+0120. WordLevel models are read like the others; rejecting them is the
 * vocabulary writer's decision.</p>
 *
 * <p>The component views are read-only; {@code root().raw("normalizer")} (and likewise for the other components) is
 * the parsed value that {@code Normalizer.fromJson}, {@code PreTokenizerSpec.fromJson} and
 * {@code TokenDecoder.fromJson} take.</p>
 */
public final class HfTokenizerJson {

    /** {@code model.type}. */
    public enum ModelType {
        BPE("BPE"), WORD_PIECE("WordPiece"), UNIGRAM("Unigram"), WORD_LEVEL("WordLevel");

        private final String jsonName;

        ModelType(String jsonName) {
            this.jsonName = jsonName;
        }

        /** The {@code type} string of tokenizer.json. */
        public String jsonName() { return jsonName; }

        static ModelType of(String name, JsonView model) {
            for (ModelType t : values()) {
                if (t.jsonName.equals(name)) {
                    return t;
                }
            }
            throw new JsonException(model.where("type") + ": unknown model type '" + name
                    + "', expected BPE, WordPiece, Unigram or WordLevel");
        }
    }

    /** One {@code added_tokens} entry. */
    public static final class AddedToken {
        private final int id;
        private final String content;
        private final boolean special;
        private final boolean lstrip;
        private final boolean rstrip;
        private final boolean singleWord;
        private final boolean normalized;

        public AddedToken(int id, String content, boolean special, boolean lstrip, boolean rstrip,
                          boolean singleWord, boolean normalized) {
            this.id = id;
            this.content = Objects.requireNonNull(content, "content");
            this.special = special;
            this.lstrip = lstrip;
            this.rstrip = rstrip;
            this.singleWord = singleWord;
            this.normalized = normalized;
        }

        public int id() { return id; }

        public String content() { return content; }

        public boolean special() { return special; }

        public boolean lstrip() { return lstrip; }

        public boolean rstrip() { return rstrip; }

        public boolean singleWord() { return singleWord; }

        public boolean normalized() { return normalized; }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof AddedToken)) return false;
            AddedToken t = (AddedToken) o;
            return id == t.id && special == t.special && lstrip == t.lstrip && rstrip == t.rstrip
                    && singleWord == t.singleWord && normalized == t.normalized && content.equals(t.content);
        }

        @Override
        public int hashCode() {
            return Objects.hash(id, content, special, lstrip, rstrip, singleWord, normalized);
        }

        @Override
        public String toString() {
            return "AddedToken{" + id + ", '" + content + "'" + (special ? ", special" : "")
                    + (lstrip ? ", lstrip" : "") + (rstrip ? ", rstrip" : "") + (singleWord ? ", single_word" : "")
                    + (normalized ? ", normalized" : "") + '}';
        }
    }

    /**
     * The {@code model} object. Absent fields take the defaults of the tokenizers library's model builders
     * (WordPiece: {@code unk_token "[UNK]"}, {@code continuing_subword_prefix "##"}; WordLevel:
     * {@code unk_token "<unk>"}); fields a model type does not define keep theirs: no merges, null strings, false
     * flags, {@code unkId() == -1}, no scores.
     */
    public static final class Model {
        private final ModelType type;
        private final String[] vocab;
        private final String[] merges;
        private final double[] scores;
        private final String unkToken;
        private final int unkId;
        private final boolean fuseUnk;
        private final boolean byteFallback;
        private final String continuingSubwordPrefix;
        private final String endOfWordSuffix;
        private final Double dropout;
        private final boolean ignoreMerges;
        private final int maxInputCharsPerWord;

        Model(ModelType type, String[] vocab, String[] merges, double[] scores, String unkToken, int unkId,
              boolean fuseUnk, boolean byteFallback, String continuingSubwordPrefix, String endOfWordSuffix,
              Double dropout, boolean ignoreMerges, int maxInputCharsPerWord) {
            this.type = type;
            this.vocab = vocab;
            this.merges = merges;
            this.scores = scores;
            this.unkToken = unkToken;
            this.unkId = unkId;
            this.fuseUnk = fuseUnk;
            this.byteFallback = byteFallback;
            this.continuingSubwordPrefix = continuingSubwordPrefix;
            this.endOfWordSuffix = endOfWordSuffix;
            this.dropout = dropout;
            this.ignoreMerges = ignoreMerges;
            this.maxInputCharsPerWord = maxInputCharsPerWord;
        }

        public ModelType type() { return type; }

        /** The tokens by id (a copy); ids that no token has are null. */
        public String[] vocab() { return vocab.clone(); }

        /** Length of {@link #vocab()}: the largest id plus one. */
        public int vocabSize() { return vocab.length; }

        /** The token with {@code id}, or null for an id no token has. */
        public String token(int id) { return vocab[id]; }

        /** The merges as {@code "a b"} strings in rank order (a copy); empty for non-BPE models. */
        public String[] merges() { return merges.clone(); }

        public int mergeCount() { return merges.length; }

        public String merge(int rank) { return merges[rank]; }

        /** Unigram piece scores by id (a copy); empty for other models. */
        public double[] scores() { return scores.clone(); }

        /** {@code unk_token} (BPE, WordPiece, WordLevel); null for BPE without one and for Unigram. */
        public String unkToken() { return unkToken; }

        /** Unigram {@code unk_id}, or -1. */
        public int unkId() { return unkId; }

        /** BPE {@code fuse_unk}. */
        public boolean fuseUnk() { return fuseUnk; }

        /** BPE and Unigram {@code byte_fallback}. */
        public boolean byteFallback() { return byteFallback; }

        /** BPE and WordPiece {@code continuing_subword_prefix}; null for BPE without one and other models. */
        public String continuingSubwordPrefix() { return continuingSubwordPrefix; }

        /** BPE {@code end_of_word_suffix}, or null. */
        public String endOfWordSuffix() { return endOfWordSuffix; }

        /** BPE {@code dropout}, or null (inference ignores it). */
        public Double dropout() { return dropout; }

        /** BPE {@code ignore_merges}. */
        public boolean ignoreMerges() { return ignoreMerges; }

        /** WordPiece {@code max_input_chars_per_word} (default 100). */
        public int maxInputCharsPerWord() { return maxInputCharsPerWord; }

        @Override
        public String toString() {
            return "Model{" + type.jsonName + ", vocab=" + vocab.length + ", merges=" + merges.length + '}';
        }
    }

    private static final int DEFAULT_MAX_INPUT_CHARS_PER_WORD = 100;
    private static final String WORD_PIECE_UNK = "[UNK]";
    private static final String WORD_PIECE_PREFIX = "##";
    private static final String WORD_LEVEL_UNK = "<unk>";
    private static final char SPACE_IN_MERGE = 'Ġ';

    private final String source;
    private final JsonView root;
    private final Model model;
    private final List<AddedToken> addedTokens;
    private final JsonView normalizer;
    private final JsonView preTokenizer;
    private final JsonView postProcessor;
    private final JsonView decoder;

    private HfTokenizerJson(String source, JsonView root, Model model, List<AddedToken> addedTokens,
                            JsonView normalizer, JsonView preTokenizer, JsonView postProcessor, JsonView decoder) {
        this.source = source;
        this.root = root;
        this.model = model;
        this.addedTokens = addedTokens;
        this.normalizer = normalizer;
        this.preTokenizer = preTokenizer;
        this.postProcessor = postProcessor;
        this.decoder = decoder;
    }

    /**
     * Reads {@code file} ({@link Json.Limits#LENIENT}, at most {@link HfTokenizerFiles#MAX_FILE_BYTES}).
     *
     * @throws IOException {@code "Invalid Hugging Face checkpoint '<dir>': <file>:<path>: ..."} when the file is not
     *                     valid JSON or not a valid tokenizer.json
     */
    public static HfTokenizerJson parse(Path file) throws IOException {
        Objects.requireNonNull(file, "file");
        try {
            return of(Json.parse(file, HfTokenizerFiles.MAX_FILE_BYTES, Json.Limits.LENIENT), file.toString());
        } catch (JsonException e) {
            throw HfTokenizerFiles.invalid(file, e);
        }
    }

    /**
     * Reads a parsed tokenizer.json document; {@code source} names it in messages.
     *
     * @throws JsonException when the document is not a valid tokenizer.json
     */
    public static HfTokenizerJson of(Object parsed, String source) {
        Objects.requireNonNull(source, "source");
        JsonView root = JsonView.of(parsed, source);
        Model model = readModel(root.obj("model"));
        List<AddedToken> added = readAddedTokens(root);
        return new HfTokenizerJson(source, root, model, added, component(root, "normalizer"),
                component(root, "pre_tokenizer"), component(root, "post_processor"), component(root, "decoder"));
    }

    /** The document name used in messages. */
    public String source() { return source; }

    /** The whole document (read-only). */
    public JsonView root() { return root; }

    public Model model() { return model; }

    /** The {@code added_tokens} entries in file order (unmodifiable). */
    public List<AddedToken> addedTokens() { return addedTokens; }

    /** The added token whose content is {@code content} (the first in file order), or null. */
    public AddedToken addedToken(String content) {
        for (AddedToken t : addedTokens) {
            if (t.content.equals(content)) {
                return t;
            }
        }
        return null;
    }

    /** The {@code normalizer} object, or null when it is absent or JSON null. */
    public JsonView normalizer() { return normalizer; }

    /** The {@code pre_tokenizer} object, or null when it is absent or JSON null. */
    public JsonView preTokenizer() { return preTokenizer; }

    /** The {@code post_processor} object, or null when it is absent or JSON null. */
    public JsonView postProcessor() { return postProcessor; }

    /** The {@code decoder} object, or null when it is absent or JSON null. */
    public JsonView decoder() { return decoder; }

    @Override
    public String toString() {
        return "HfTokenizerJson{" + source + ", " + model + ", added=" + addedTokens.size() + '}';
    }

    // ------------------------------------------------------------------ reading

    private static JsonView component(JsonView root, String key) {
        Object raw = root.raw(key);
        if (raw == null) {
            return null;
        }
        if (!(raw instanceof Map)) {
            throw new JsonException(root.where(key) + ": expected object or null, got " + typeName(raw));
        }
        return root.obj(key);
    }

    private static Model readModel(JsonView m) {
        ModelType type = modelType(m);
        String[] vocab = null;
        double[] scores = new double[0];
        int unkId = -1;
        if (type == ModelType.UNIGRAM) {
            List<Object> pieces = m.list("vocab");
            checkTokenCount(m, pieces.size());
            vocab = new String[pieces.size()];
            scores = new double[pieces.size()];
            for (int i = 0; i < pieces.size(); i++) {
                readUnigramPiece(m, i, pieces.get(i), vocab, scores);
            }
            if (m.raw("unk_id") != null) {
                unkId = m.i32("unk_id");
                if (unkId < 0 || unkId >= vocab.length) {
                    throw new JsonException(m.where("unk_id") + ": " + unkId + " is not an id of the "
                            + vocab.length + "-piece vocabulary");
                }
            }
        } else {
            vocab = readVocabMap(m.obj("vocab"));
        }
        String[] merges = type == ModelType.BPE ? readMerges(m) : new String[0];
        Double dropout = m.raw("dropout") == null ? null : m.f64("dropout");
        int maxChars = type == ModelType.WORD_PIECE
                ? m.optI32("max_input_chars_per_word", DEFAULT_MAX_INPUT_CHARS_PER_WORD)
                : DEFAULT_MAX_INPUT_CHARS_PER_WORD;
        String defaultUnk = type == ModelType.WORD_PIECE ? WORD_PIECE_UNK
                : type == ModelType.WORD_LEVEL ? WORD_LEVEL_UNK : null;
        String defaultPrefix = type == ModelType.WORD_PIECE ? WORD_PIECE_PREFIX : null;
        return new Model(type, vocab, merges, scores, m.optStr("unk_token", defaultUnk), unkId,
                m.optBool("fuse_unk", false), m.optBool("byte_fallback", false),
                m.optStr("continuing_subword_prefix", defaultPrefix), m.optStr("end_of_word_suffix", null), dropout,
                m.optBool("ignore_merges", false), maxChars);
    }

    private static ModelType modelType(JsonView m) {
        String type = m.optStr("type", null);
        if (type != null) {
            return ModelType.of(type, m);
        }
        return m.raw("vocab") instanceof List ? ModelType.UNIGRAM : ModelType.BPE;
    }

    /** A token → id object as tokens by id; ids are unique and in [0, MAX_TOKENS). */
    private static String[] readVocabMap(JsonView v) {
        checkTokenCount(v, v.keys().size());
        int[] ids = new int[v.keys().size()];
        String[] tokens = new String[ids.length];
        int n = 0;
        int maxId = -1;
        for (String token : v.keys()) {
            int id = v.i32(token);
            if (id < 0 || id >= HfTokenizerFiles.MAX_TOKENS) {
                throw new JsonException(v.where(token) + ": token id " + id + " is outside [0, "
                        + HfTokenizerFiles.MAX_TOKENS + ")");
            }
            ids[n] = id;
            tokens[n++] = token;
            maxId = Math.max(maxId, id);
        }
        String[] byId = new String[maxId + 1];
        for (int i = 0; i < n; i++) {
            String previous = byId[ids[i]];
            if (previous != null) {
                throw new JsonException(v.source() + ":" + v.path() + ": tokens '" + previous + "' and '" + tokens[i]
                        + "' share id " + ids[i]);
            }
            byId[ids[i]] = tokens[i];
        }
        return byId;
    }

    private static void readUnigramPiece(JsonView m, int i, Object item, String[] vocab, double[] scores) {
        String at = m.where("vocab") + "[" + i + "]";
        if (!(item instanceof List) || ((List<?>) item).size() != 2) {
            throw new JsonException(at + ": expected [piece, score], got " + describe(item));
        }
        List<?> pair = (List<?>) item;
        if (!(pair.get(0) instanceof String)) {
            throw new JsonException(at + "[0]: expected string, got " + typeName(pair.get(0)));
        }
        Object score = pair.get(1);
        if (!(score instanceof Number)) {
            throw new JsonException(at + "[1]: expected number, got " + typeName(score));
        }
        vocab[i] = (String) pair.get(0);
        scores[i] = ((Number) score).doubleValue();
    }

    /**
     * {@code model.merges} as {@code "a b"} strings. Every entry has the form of the first: a string holding exactly
     * one space, or a pair of strings whose spaces are re-encoded as U+0120.
     */
    private static String[] readMerges(JsonView m) {
        if (m.raw("merges") == null) {
            return new String[0];
        }
        List<Object> raw = m.list("merges");
        String[] merges = new String[raw.size()];
        boolean pairs = !raw.isEmpty() && raw.get(0) instanceof List;
        for (int i = 0; i < merges.length; i++) {
            Object item = raw.get(i);
            String at = m.where("merges") + "[" + i + "]";
            if (pairs) {
                if (!(item instanceof List) || ((List<?>) item).size() != 2
                        || !(((List<?>) item).get(0) instanceof String)
                        || !(((List<?>) item).get(1) instanceof String)) {
                    throw new JsonException(at + ": expected a pair of strings like merges[0], got " + describe(item));
                }
                List<?> pair = (List<?>) item;
                merges[i] = encodeSpaces((String) pair.get(0)) + " " + encodeSpaces((String) pair.get(1));
            } else {
                if (!(item instanceof String)) {
                    throw new JsonException(at + ": expected a string like merges[0], got " + describe(item));
                }
                String s = (String) item;
                int space = s.indexOf(' ');
                if (space < 0 || s.indexOf(' ', space + 1) >= 0) {
                    throw new JsonException(at + ": expected two parts separated by one space, got '" + s + "'");
                }
                merges[i] = s;
            }
        }
        return merges;
    }

    private static String encodeSpaces(String part) {
        return part.replace(' ', SPACE_IN_MERGE);
    }

    private static List<AddedToken> readAddedTokens(JsonView root) {
        if (root.raw("added_tokens") == null) {
            return Collections.emptyList();
        }
        List<Object> raw = root.list("added_tokens");
        checkTokenCount(root, raw.size());
        List<AddedToken> out = new ArrayList<>(raw.size());
        for (int i = 0; i < raw.size(); i++) {
            JsonView t = JsonView.of(raw.get(i), root.source(), "added_tokens[" + i + "]");
            int id = t.i32("id");
            if (id < 0 || id >= HfTokenizerFiles.MAX_TOKENS) {
                throw new JsonException(t.where("id") + ": token id " + id + " is outside [0, "
                        + HfTokenizerFiles.MAX_TOKENS + ")");
            }
            boolean special = t.optBool("special", false);
            out.add(new AddedToken(id, t.str("content"), special, t.optBool("lstrip", false),
                    t.optBool("rstrip", false), t.optBool("single_word", false), t.optBool("normalized", !special)));
        }
        return Collections.unmodifiableList(out);
    }

    private static void checkTokenCount(JsonView v, int count) {
        if (count > HfTokenizerFiles.MAX_TOKENS) {
            throw new JsonException(v.source() + ":" + (v.path().isEmpty() ? "<root>" : v.path()) + ": " + count
                    + " tokens exceed the limit of " + HfTokenizerFiles.MAX_TOKENS);
        }
    }

    private static String describe(Object v) {
        if (v instanceof List) {
            return "array of " + ((List<?>) v).size();
        }
        return typeName(v);
    }

    private static String typeName(Object v) {
        if (v == null) return "null";
        if (v instanceof Boolean) return "boolean";
        if (v instanceof Number) return "number";
        if (v instanceof String) return "string";
        if (v instanceof List) return "array";
        if (v instanceof Map) return "object";
        return v.getClass().getSimpleName();
    }
}
