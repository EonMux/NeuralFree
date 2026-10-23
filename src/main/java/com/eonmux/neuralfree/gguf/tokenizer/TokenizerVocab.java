package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable tokenizer vocabulary loaded from GGUF {@code tokenizer.*} metadata.
 *
 * <p>Mirrors {@code llama_vocab::impl::load} (llama.cpp src/llama-vocab.cpp:2196-3424): vocabulary
 * type from {@code tokenizer.ggml.model}, token texts, scores and attributes, BPE merge ranks,
 * and the resolved encoding flags (type defaults, then pre-tokenizer flags, then GGUF overrides).
 * Special-token ids and end-of-generation sets live in {@link SpecialTokens}.</p>
 */
public final class TokenizerVocab {

    private static final Logger LOGGER = LogManager.getLogger(TokenizerVocab.class);

    /**
     * llama.cpp {@code llama_vocab_type} (include/llama.h:72-82); {@link #code()} is the llama.h
     * value. The test-only {@code LLAMA_VOCAB_TYPE_TEST} (7) has no GGUF tokenizer model.
     */
    public enum Type {
        NONE(0), SPM(1), BPE(2), WPM(3), UGM(4), RWKV(5), PLAMO2(6), PLAMO3(8);

        private final int code;

        Type(int code) { this.code = code; }

        /** The {@code llama_vocab_type} enum value. */
        public int code() { return code; }
    }

    /** llama.cpp {@code llama_token_attr} bit flags (include/llama.h:104-114). */
    public static final int ATTR_UNDEFINED = 0;
    public static final int ATTR_UNKNOWN = 1;
    public static final int ATTR_UNUSED = 1 << 1;
    public static final int ATTR_NORMAL = 1 << 2;
    public static final int ATTR_CONTROL = 1 << 3;
    public static final int ATTR_USER_DEFINED = 1 << 4;
    public static final int ATTR_BYTE = 1 << 5;
    public static final int ATTR_NORMALIZED = 1 << 6;
    public static final int ATTR_LSTRIP = 1 << 7;
    public static final int ATTR_RSTRIP = 1 << 8;
    public static final int ATTR_SINGLE_WORD = 1 << 9;

    public static final String KEY_MODEL = "tokenizer.ggml.model";
    public static final String KEY_PRE = "tokenizer.ggml.pre";
    public static final String KEY_TOKENS = "tokenizer.ggml.tokens";
    public static final String KEY_TOKEN_TYPE = "tokenizer.ggml.token_type";
    public static final String KEY_SCORES = "tokenizer.ggml.scores";
    public static final String KEY_MERGES = "tokenizer.ggml.merges";
    public static final String KEY_PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap";
    public static final String KEY_ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix";
    public static final String KEY_REMOVE_EXTRA_WS = "tokenizer.ggml.remove_extra_whitespaces";
    public static final String KEY_ADD_BOS = "tokenizer.ggml.add_bos_token";
    public static final String KEY_ADD_EOS = "tokenizer.ggml.add_eos_token";
    public static final String KEY_ADD_SEP = "tokenizer.ggml.add_sep_token";
    public static final String KEY_DNA_AUTO_TAGS = "tokenizer.ggml.dna_auto_tags";
    public static final String KEY_NORMALIZER_NFC = "tokenizer.ggml.normalizer.nfc";
    public static final String KEY_NORMALIZER_LOWERCASE = "tokenizer.ggml.normalizer.lowercase";
    public static final String KEY_NORMALIZER_STRIP_ACCENTS = "tokenizer.ggml.normalizer.strip_accents";
    public static final String KEY_ARCHITECTURE = "general.architecture";
    public static final String KEY_NAME = "general.name";

    /** Marker suffix that keeps hybriddna k-mer tokens distinct (llama-vocab.cpp:1784). */
    static final String DNA_KMER_MARKER = "\uE000";

    private final String model;
    private final String pre;
    private final Type type;
    private final String architecture;
    private final String[] texts;
    private final float[] scores;
    private final int[] attrs;
    private final Map<String, Integer> tokenToId;
    private final Map<String, Map<String, Integer>> mergeRanks;
    private final int mergeCount;
    private final byte[] precompiledCharsmap;
    private final PreTokenizerRegistry.PreTokenizer preTokenizer;
    private final Map<String, Object> metadata;

    private final boolean addSpacePrefix;
    private final boolean removeExtraWhitespaces;
    private final boolean escapeWhitespaces;
    private final boolean cleanSpaces;
    private final boolean ignoreMerges;
    private final boolean addBos;
    private final boolean addEos;
    private final boolean addSep;
    private final boolean dnaAutoTags;
    private final boolean normalizerNfc;
    private final boolean normalizerLowercase;
    private final boolean normalizerStripAccents;

    private TokenizerVocab(Loader l) {
        this.model = l.model;
        this.pre = l.pre;
        this.type = l.type;
        this.architecture = l.architecture;
        this.texts = l.texts;
        this.scores = l.scores;
        this.attrs = l.attrs;
        this.tokenToId = Collections.unmodifiableMap(l.tokenToId);
        this.mergeRanks = Collections.unmodifiableMap(l.mergeRanks);
        this.mergeCount = l.mergeCount;
        this.precompiledCharsmap = l.precompiledCharsmap;
        this.preTokenizer = l.preTokenizer;
        this.metadata = Collections.unmodifiableMap(l.metadata);
        this.addSpacePrefix = l.addSpacePrefix;
        this.removeExtraWhitespaces = l.removeExtraWhitespaces;
        this.escapeWhitespaces = l.escapeWhitespaces;
        this.cleanSpaces = l.cleanSpaces;
        this.ignoreMerges = l.ignoreMerges;
        this.addBos = l.addBos;
        this.addEos = l.addEos;
        this.addSep = l.addSep;
        this.dnaAutoTags = l.dnaAutoTags;
        this.normalizerNfc = l.normalizerNfc;
        this.normalizerLowercase = l.normalizerLowercase;
        this.normalizerStripAccents = l.normalizerStripAccents;
    }

    /** Loads the vocabulary from an open GGUF file. */
    public static TokenizerVocab fromGguf(GGUFFileParser parser) {
        if (parser == null) throw new IllegalArgumentException("GGUF parser is null");
        return fromMetadata(parser.getMetadata());
    }

    /**
     * Loads the vocabulary from GGUF-shaped metadata: strings, {@code List<Object>} arrays,
     * {@code Number} scalars and {@code Boolean} flags, keyed by GGUF key names.
     *
     * @throws IllegalArgumentException when required keys are missing or malformed
     */
    public static TokenizerVocab fromMetadata(Map<String, Object> md) {
        if (md == null) throw new IllegalArgumentException("GGUF metadata is null");
        return new TokenizerVocab(new Loader(md).load());
    }

    // ---------------------------------------------------------------------------------------------
    // Accessors
    // ---------------------------------------------------------------------------------------------

    public String model() { return model; }
    /** The pre-tokenizer name; "gemma4" for the gemma4 model, as llama.cpp forces it. */
    public String pre() { return pre; }
    public Type type() { return type; }
    public String architecture() { return architecture; }
    public int size() { return texts.length; }

    /** Token text as used for detokenization (hybriddna k-mers carry no marker suffix). */
    public String text(int id) { return id >= 0 && id < texts.length ? texts[id] : ""; }
    public float score(int id) { return id >= 0 && id < scores.length ? scores[id] : 0.0f; }
    public int attr(int id) { return id >= 0 && id < attrs.length ? attrs[id] : ATTR_UNDEFINED; }
    public boolean hasAttr(int id, int mask) { return (attr(id) & mask) != 0; }
    public boolean isNormal(int id) { return hasAttr(id, ATTR_NORMAL); }
    public boolean isUnknown(int id) { return hasAttr(id, ATTR_UNKNOWN); }
    public boolean isControl(int id) { return hasAttr(id, ATTR_CONTROL); }
    public boolean isByte(int id) { return hasAttr(id, ATTR_BYTE); }
    public boolean isUserDefined(int id) { return hasAttr(id, ATTR_USER_DEFINED); }
    public boolean isUnused(int id) { return hasAttr(id, ATTR_UNUSED); }

    /** {@code text_to_token}: the id of an exact token text, or -1. */
    public int tokenId(String text) {
        Integer id = tokenToId.get(text);
        return id != null ? id : -1;
    }

    /** {@code find_bpe_rank}: the merge rank of the pair, or -1 when no such merge exists. */
    public int mergeRank(String left, String right) {
        Map<String, Integer> rights = mergeRanks.get(left);
        if (rights == null) return -1;
        Integer rank = rights.get(right);
        return rank != null ? rank : -1;
    }

    public int mergeCount() { return mergeCount; }

    /**
     * {@code token_to_byte}: the byte value of a byte token whose text is {@code <0xXX>},
     * or -1 when the token is not of that form.
     */
    public int byteValue(int id) {
        String t = text(id);
        if (t.length() != 6 || !t.startsWith("<0x") || t.charAt(5) != '>') return -1;
        int hi = Character.digit(t.charAt(3), 16);
        int lo = Character.digit(t.charAt(4), 16);
        return hi < 0 || lo < 0 ? -1 : (hi << 4) | lo;
    }

    /** A copy of {@code tokenizer.ggml.precompiled_charsmap}, or null when absent. */
    public byte[] precompiledCharsmap() {
        return precompiledCharsmap != null ? precompiledCharsmap.clone() : null;
    }

    /** The BPE pre-tokenizer; null for vocabulary types other than BPE. */
    public PreTokenizerRegistry.PreTokenizer preTokenizer() { return preTokenizer; }

    /**
     * Scalar tokenizer metadata: every {@code tokenizer.*} key except the bulk token arrays,
     * plus {@code general.architecture}, {@code general.name} and {@code <arch>.vocab_size}.
     */
    public Map<String, Object> metadata() { return metadata; }

    public Object metadata(String key) { return metadata.get(key); }

    public boolean addSpacePrefix() { return addSpacePrefix; }
    public boolean removeExtraWhitespaces() { return removeExtraWhitespaces; }
    public boolean escapeWhitespaces() { return escapeWhitespaces; }
    public boolean cleanSpaces() { return cleanSpaces; }
    public boolean ignoreMerges() { return ignoreMerges; }
    public boolean addBos() { return addBos; }
    public boolean addEos() { return addEos; }
    public boolean addSep() { return addSep; }
    public boolean dnaAutoTags() { return dnaAutoTags; }
    public boolean normalizerNfc() { return normalizerNfc; }
    public boolean normalizerLowercase() { return normalizerLowercase; }
    public boolean normalizerStripAccents() { return normalizerStripAccents; }

    /** llama.cpp {@code unicode_cpt_flags::is_whitespace} (src/unicode-data.cpp:2286). */
    public static boolean isWhitespace(int cp) {
        switch (cp) {
            case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
            case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
            case 0x202F: case 0x205F: case 0x3000:
                return true;
            default:
                return cp >= 0x2000 && cp <= 0x200A;
        }
    }

    /** {@code llama_escape_whitespace}: every ASCII space becomes U+2581. */
    public static String escapeWhitespace(String text) {
        return text.replace(' ', '\u2581');
    }

    /** {@code llama_unescape_whitespace}: every U+2581 becomes an ASCII space. */
    public static String unescapeWhitespace(String text) {
        return text.replace('\u2581', ' ');
    }

    // ---------------------------------------------------------------------------------------------
    // Loading
    // ---------------------------------------------------------------------------------------------

    /** Mutable load state, discarded once the immutable vocabulary is built. */
    private static final class Loader {
        final Map<String, Object> md;
        String model;
        String pre;
        Type type;
        String architecture;
        String[] texts;
        float[] scores;
        int[] attrs;
        Map<String, Integer> tokenToId = new HashMap<>();
        Map<String, Map<String, Integer>> mergeRanks = new HashMap<>();
        int mergeCount;
        byte[] precompiledCharsmap;
        PreTokenizerRegistry.PreTokenizer preTokenizer;
        Map<String, Object> metadata = new LinkedHashMap<>();
        boolean addSpacePrefix;
        boolean removeExtraWhitespaces;
        boolean escapeWhitespaces = true;
        boolean cleanSpaces;
        boolean ignoreMerges;
        boolean addBos;
        boolean addEos;
        boolean addSep;
        boolean dnaAutoTags;
        boolean normalizerNfc;
        boolean normalizerLowercase = true;
        boolean normalizerStripAccents;

        Loader(Map<String, Object> md) { this.md = md; }

        Loader load() {
            model = str(KEY_MODEL);
            if (model == null) {
                throw new IllegalArgumentException("GGUF metadata has no " + KEY_MODEL
                        + "; the file carries no tokenizer");
            }
            pre = str(KEY_PRE);
            architecture = str(KEY_ARCHITECTURE) != null ? str(KEY_ARCHITECTURE) : "";
            type = typeOf(model);
            if ("gemma4".equals(model)) pre = "gemma4";
            copyMetadata();
            if (type == Type.NONE) {
                loadNoVocab();
                return this;
            }
            if (type == Type.BPE) loadMerges();
            if (type == Type.UGM) loadCharsmap();
            applyTypeDefaults();
            Boolean asp = bool(KEY_ADD_SPACE_PREFIX);
            if (asp != null) addSpacePrefix = asp;
            Boolean rew = bool(KEY_REMOVE_EXTRA_WS);
            if (rew != null) removeExtraWhitespaces = rew;
            loadTokens();
            applySpecialFlags();
            return this;
        }

        private static Type typeOf(String model) {
            switch (model) {
                case "no_vocab": case "none": return Type.NONE;
                case "llama": return Type.SPM;
                case "bert": return Type.WPM;
                case "gpt2": case "hybriddna": case "whitespace": case "gemma4": return Type.BPE;
                case "t5": return Type.UGM;
                case "rwkv": return Type.RWKV;
                case "plamo2": return Type.PLAMO2;
                case "plamo3": return Type.PLAMO3;
                default:
                    throw new IllegalArgumentException("Unsupported tokenizer model '" + model
                            + "' (" + KEY_MODEL + "); supported: llama, gpt2, hybriddna, whitespace,"
                            + " gemma4, bert, t5, rwkv, plamo2, plamo3, none, no_vocab");
            }
        }

        private void copyMetadata() {
            for (Map.Entry<String, Object> e : md.entrySet()) {
                String k = e.getKey();
                if (!k.startsWith("tokenizer.") || k.equals(KEY_TOKENS) || k.equals(KEY_TOKEN_TYPE)
                        || k.equals(KEY_SCORES) || k.equals(KEY_MERGES) || k.equals(KEY_PRECOMPILED_CHARSMAP)) {
                    continue;
                }
                metadata.put(k, frozen(e.getValue()));
            }
            String vocabSizeKey = architecture + ".vocab_size";
            for (String k : new String[]{KEY_ARCHITECTURE, KEY_NAME, vocabSizeKey}) {
                if (md.containsKey(k)) metadata.put(k, frozen(md.get(k)));
            }
        }

        private static Object frozen(Object v) {
            if (v instanceof List) return Collections.unmodifiableList(new java.util.ArrayList<>((List<?>) v));
            return v;
        }

        /** llama-vocab.cpp:2206-2228: no tokens, only a dummy vocabulary of {@code <arch>.vocab_size}. */
        private void loadNoVocab() {
            Object n = md.get(architecture + ".vocab_size");
            int count = n instanceof Number ? ((Number) n).intValue() : 0;
            if (count < 0) throw new IllegalArgumentException("Negative " + architecture + ".vocab_size: " + count);
            texts = new String[count];
            java.util.Arrays.fill(texts, "");
            scores = new float[count];
            attrs = new int[count];
        }

        /** llama-vocab.cpp:2254-2290 and 2372-2396; the first occurrence of a pair keeps its rank. */
        private void loadMerges() {
            List<?> merges = list(KEY_MERGES);
            if (merges == null) {
                if ("kimi-k2".equals(pre)) return;
                throw new IllegalArgumentException("Cannot find " + KEY_MERGES + " for BPE tokenizer model '"
                        + model + "'");
            }
            for (int i = 0; i < merges.size(); i++) {
                Object o = merges.get(i);
                if (!(o instanceof String)) {
                    throw new IllegalArgumentException(KEY_MERGES + "[" + i + "] is not a string");
                }
                String word = (String) o;
                int sep = word.isEmpty() ? -1 : word.indexOf(' ', 1);
                if (sep < 0) continue;
                mergeRanks.computeIfAbsent(word.substring(0, sep), k -> new HashMap<>())
                        .putIfAbsent(word.substring(sep + 1), i);
            }
            mergeCount = merges.size();
        }

        /** llama-vocab.cpp:2307-2336: the XCDA blob size must leave room for the replacements. */
        private void loadCharsmap() {
            List<?> raw = list(KEY_PRECOMPILED_CHARSMAP);
            if (raw == null) return;
            byte[] blob = new byte[raw.size()];
            for (int i = 0; i < blob.length; i++) {
                Object o = raw.get(i);
                if (!(o instanceof Number)) {
                    throw new IllegalArgumentException(KEY_PRECOMPILED_CHARSMAP + "[" + i + "] is not a byte");
                }
                blob[i] = (byte) ((Number) o).intValue();
            }
            if (blob.length < 4) {
                throw new IllegalArgumentException(KEY_PRECOMPILED_CHARSMAP + " too small for the xcda size header");
            }
            long xcdaSize = (blob[0] & 0xFFL) | (blob[1] & 0xFFL) << 8 | (blob[2] & 0xFFL) << 16 | (blob[3] & 0xFFL) << 24;
            if (xcdaSize + 4 >= blob.length) {
                throw new IllegalArgumentException("Index out of array bounds in " + KEY_PRECOMPILED_CHARSMAP
                        + " (xcda size " + xcdaSize + ", blob " + blob.length + " bytes)");
            }
            precompiledCharsmap = blob;
        }

        /** llama-vocab.cpp:2413-2726: per-type defaults, BPE pre-tokenizer flags. */
        private void applyTypeDefaults() {
            switch (type) {
                case BPE:
                    addSpacePrefix = false;
                    preTokenizer = PreTokenizerRegistry.forName(pre);
                    cleanSpaces = preTokenizer.cleanSpaces();
                    ignoreMerges = preTokenizer.ignoreMerges();
                    escapeWhitespaces = preTokenizer.escapeWhitespaces();
                    if (preTokenizer.addBos() != null) addBos = preTokenizer.addBos();
                    if (preTokenizer.addSep()) addSep = true;
                    if (preTokenizer.normalizerLowercase() != null) {
                        normalizerLowercase = preTokenizer.normalizerLowercase();
                    }
                    break;
                case SPM:
                    addSpacePrefix = true;
                    cleanSpaces = false;
                    addBos = true;
                    addEos = false;
                    break;
                case WPM:
                    addSpacePrefix = false;
                    cleanSpaces = true;
                    addBos = true;
                    addEos = false;
                    addSep = true;
                    break;
                case UGM:
                    addBos = false;
                    addEos = true;
                    break;
                case RWKV:
                    addSpacePrefix = false;
                    cleanSpaces = false;
                    addBos = false;
                    addEos = false;
                    break;
                default:
                    break;
            }
        }

        /**
         * llama-vocab.cpp:2731-2815: texts, scores and attributes; empty texts get a placeholder and
         * every text must be unique (llama.cpp asserts {@code id_to_token.size() == token_to_id.size()}).
         */
        private void loadTokens() {
            List<?> tokens = list(KEY_TOKENS);
            if (tokens == null) {
                throw new IllegalArgumentException("Cannot find " + KEY_TOKENS + " for tokenizer model '" + model + "'");
            }
            int n = tokens.size();
            List<?> sc = list(KEY_SCORES);
            if (sc != null && sc.size() < n) {
                throw new IllegalArgumentException("Index out of array bounds for scores (" + sc.size() + " < " + n + ")");
            }
            List<?> tt = list(KEY_TOKEN_TYPE);
            if (tt != null && tt.size() < n) {
                throw new IllegalArgumentException("Index out of array bounds for token types (" + tt.size() + " < " + n + ")");
            }
            texts = new String[n];
            scores = new float[n];
            attrs = new int[n];
            for (int i = 0; i < n; i++) {
                Object o = tokens.get(i);
                if (!(o instanceof String)) throw new IllegalArgumentException(KEY_TOKENS + "[" + i + "] is not a string");
                String word = (String) o;
                if (word.isEmpty()) word = "[EMPTY_" + i + "]";
                Integer previous = tokenToId.put(word, i);
                if (previous != null) {
                    throw new IllegalArgumentException(KEY_TOKENS + " has duplicate token text '" + word
                            + "' at ids " + previous + " and " + i);
                }
                texts[i] = word;
                scores[i] = sc != null ? number(sc, i, KEY_SCORES).floatValue() : 0.0f;
                attrs[i] = tt != null ? attrOfType(number(tt, i, KEY_TOKEN_TYPE).intValue()) : ATTR_NORMAL;
            }
            if ("hybriddna".equals(model)) stripDnaMarkers();
        }

        /** {@code LLAMA_TOKEN_TYPE_*} to {@code LLAMA_TOKEN_ATTR_*} (llama-vocab.cpp:2802-2813). */
        private static int attrOfType(int tokenType) {
            switch (tokenType) {
                case 1: return ATTR_NORMAL;
                case 2: return ATTR_UNKNOWN;
                case 3: return ATTR_CONTROL;
                case 4: return ATTR_USER_DEFINED;
                case 5: return ATTR_UNUSED;
                case 6: return ATTR_BYTE;
                default: return ATTR_UNDEFINED;
            }
        }

        /** llama-vocab.cpp:2817-2837: k-mers after {@code <oov>} detokenize to their bare bases. */
        private void stripDnaMarkers() {
            Integer oov = tokenToId.get("<oov>");
            if (oov == null) return;
            for (int i = oov + 1; i < texts.length; i++) {
                String t = texts[i];
                if (t.length() > DNA_KMER_MARKER.length() && t.endsWith(DNA_KMER_MARKER)) {
                    t = t.substring(0, t.length() - DNA_KMER_MARKER.length());
                    int last = t.length() - 1;
                    while (last >= 0 && t.charAt(last) == 'N') last--;
                    if (last >= 0) t = t.substring(0, last + 1);
                    texts[i] = t;
                }
            }
        }

        /** llama-vocab.cpp:2914-2944: add_bos/add_eos/add_sep, Gemma4 override, normalizer options. */
        private void applySpecialFlags() {
            Boolean b = bool(KEY_ADD_BOS);
            if (b != null) addBos = b;
            b = bool(KEY_ADD_EOS);
            if (b != null) addEos = b;
            b = bool(KEY_ADD_SEP);
            if (b != null) addSep = b;
            b = bool(KEY_DNA_AUTO_TAGS);
            if (b != null) dnaAutoTags = b;
            if (preTokenizer != null && preTokenizer.forceAddBos() && !addBos) {
                addBos = true;
                LOGGER.warn("override '{}' to 'true' for pre-tokenizer '{}'", KEY_ADD_BOS, preTokenizer.name());
            }
            b = bool(KEY_NORMALIZER_NFC);
            if (b != null) normalizerNfc = b;
            b = bool(KEY_NORMALIZER_LOWERCASE);
            if (b != null) normalizerLowercase = b;
            normalizerStripAccents = normalizerLowercase;
            b = bool(KEY_NORMALIZER_STRIP_ACCENTS);
            if (b != null) normalizerStripAccents = b;
        }

        private String str(String key) {
            Object v = md.get(key);
            if (v == null) return null;
            if (!(v instanceof String)) throw new IllegalArgumentException(key + " is not a string: " + v);
            return (String) v;
        }

        private Boolean bool(String key) {
            Object v = md.get(key);
            if (v == null) return null;
            if (!(v instanceof Boolean)) throw new IllegalArgumentException(key + " is not a boolean: " + v);
            return (Boolean) v;
        }

        private List<?> list(String key) {
            Object v = md.get(key);
            if (v == null) return null;
            if (!(v instanceof List)) throw new IllegalArgumentException(key + " is not an array");
            return (List<?>) v;
        }

        private static Number number(List<?> l, int i, String key) {
            Object o = l.get(i);
            if (!(o instanceof Number)) throw new IllegalArgumentException(key + "[" + i + "] is not a number");
            return (Number) o;
        }
    }
}
