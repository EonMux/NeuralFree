package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Immutable tokenizer vocabulary loaded from GGUF {@code tokenizer.*} metadata.
 *
 * <p>Mirrors {@code llama_vocab::impl::load} (llama.cpp src/llama-vocab.cpp:2196-3424): vocabulary
 * type from {@code tokenizer.ggml.model}, token texts, scores and attributes, BPE merge ranks,
 * and the resolved encoding flags (type defaults, then pre-tokenizer flags, then GGUF overrides).
 * Special-token ids and end-of-generation sets live in {@link SpecialTokens}.</p>
 *
 * <p>Hugging Face behaviour that GGUF cannot express arrives through the optional
 * {@code neuralfree.tokenizer.*} overlay keys ({@link GgufKeys#NF_TOKENIZER_TOKEN_ATTRS} and
 * following), applied last. A vocabulary without them is exactly the llama.cpp one, and a NONE
 * vocabulary reads none of them.</p>
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
    /** Every defined attribute bit. */
    static final int ATTR_ALL = (1 << 10) - 1;

    public static final String KEY_MODEL = "tokenizer.ggml.model";
    public static final String KEY_PRE = "tokenizer.ggml.pre";
    public static final String KEY_TOKENS = "tokenizer.ggml.tokens";
    public static final String KEY_TOKEN_TYPE = "tokenizer.ggml.token_type";
    public static final String KEY_SCORES = "tokenizer.ggml.scores";
    public static final String KEY_MERGES = "tokenizer.ggml.merges";
    public static final String KEY_PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap";
    public static final String KEY_ADD_SPACE_PREFIX = "tokenizer.ggml.add_space_prefix";
    public static final String KEY_REMOVE_EXTRA_WS = "tokenizer.ggml.remove_extra_whitespaces";
    public static final String KEY_CLEAN_SPACES = "tokenizer.ggml.clean_spaces";
    public static final String KEY_METASPACE_PREPEND_SCHEME = "tokenizer.ggml.metaspace.prepend_scheme";
    public static final String KEY_ADD_BOS = "tokenizer.ggml.add_bos_token";
    public static final String KEY_ADD_EOS = "tokenizer.ggml.add_eos_token";
    public static final String KEY_ADD_SEP = "tokenizer.ggml.add_sep_token";
    public static final String KEY_DNA_AUTO_TAGS = "tokenizer.ggml.dna_auto_tags";
    public static final String KEY_NORMALIZER_NFC = "tokenizer.ggml.normalizer.nfc";
    public static final String KEY_NORMALIZER_LOWERCASE = "tokenizer.ggml.normalizer.lowercase";
    public static final String KEY_NORMALIZER_STRIP_ACCENTS = "tokenizer.ggml.normalizer.strip_accents";
    public static final String KEY_ARCHITECTURE = "general.architecture";
    public static final String KEY_NAME = "general.name";

    /** The overlay keys only BPE encoding reads. */
    private static final String[] BPE_ONLY_KEYS = {
            GgufKeys.NF_TOKENIZER_NORMALIZER, GgufKeys.NF_TOKENIZER_PRE_TOKENIZER,
            GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE,
            GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK,
            GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX, GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX,
    };

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
    private final PreTokenizerRegistry.PreTokenizer namedPreTokenizer;
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

    private final Normalizer normalizer;
    private final TokenDecoder decoder;
    private final PreTokenizerSpec.PrependScheme prependScheme;
    private final PreTokenizerSpec.PrependScheme metaspacePrependScheme;
    private final int[] prefixIds;
    private final int[] suffixIds;
    private final int[] extraEogIds;
    private final Boolean cleanUpTokenizationSpaces;
    private final boolean bpeAddPrefixSpace;
    private final boolean bpeModelOverlay;
    private final boolean bpeUnkFuse;
    private final Boolean bpeByteFallback;
    private final String bpeContinuingSubwordPrefix;
    private final String bpeEndOfWordSuffix;

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
        this.namedPreTokenizer = l.namedPreTokenizer;
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
        this.normalizer = l.normalizer;
        this.decoder = l.decoder;
        this.prependScheme = l.prependScheme;
        this.metaspacePrependScheme = l.metaspacePrependScheme;
        this.prefixIds = l.prefixIds;
        this.suffixIds = l.suffixIds;
        this.extraEogIds = l.extraEogIds;
        this.cleanUpTokenizationSpaces = l.cleanUpTokenizationSpaces;
        this.bpeAddPrefixSpace = l.bpeAddPrefixSpace;
        this.bpeModelOverlay = l.bpeModelOverlay;
        this.bpeUnkFuse = l.bpeUnkFuse;
        this.bpeByteFallback = l.bpeByteFallback;
        this.bpeContinuingSubwordPrefix = l.bpeContinuingSubwordPrefix;
        this.bpeEndOfWordSuffix = l.bpeEndOfWordSuffix;
    }

    /** Loads the vocabulary from the {@code tokenizer.*} metadata of an open model. */
    public static TokenizerVocab fromSource(ModelSource source) {
        if (source == null) throw new IllegalArgumentException("model source is null");
        return fromMetadata(source.getMetadata());
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

    /**
     * The BPE pre-tokenizer that splits text: the one built from
     * {@link GgufKeys#NF_TOKENIZER_PRE_TOKENIZER} when present, else the registry one of
     * {@code tokenizer.ggml.pre}; null for vocabulary types other than BPE.
     */
    public PreTokenizerRegistry.PreTokenizer preTokenizer() { return preTokenizer; }

    /**
     * The registry pre-tokenizer of {@code tokenizer.ggml.pre}, whose llama.cpp flags the
     * vocabulary and the special-token rules resolve from even when a spec overlay splits the
     * text; null for vocabulary types other than BPE.
     */
    PreTokenizerRegistry.PreTokenizer namedPreTokenizer() { return namedPreTokenizer; }

    /**
     * Scalar tokenizer metadata: every {@code tokenizer.*} key except the bulk token arrays,
     * plus {@code general.architecture}, {@code general.name} and {@code <arch>.vocab_size}.
     */
    public Map<String, Object> metadata() { return metadata; }

    public Object metadata(String key) { return metadata.get(key); }

    public boolean addSpacePrefix() { return addSpacePrefix; }
    public boolean removeExtraWhitespaces() { return removeExtraWhitespaces; }
    public boolean escapeWhitespaces() { return escapeWhitespaces; }
    /**
     * llama.cpp {@code clean_spaces}, applied by detokenization: the default of the pre-tokenizer
     * name or vocabulary type, overridden by {@link #KEY_CLEAN_SPACES}; false whenever
     * {@link #cleanUpTokenizationSpaces()} is present, since the Hugging Face clean-up then
     * replaces it.
     */
    public boolean cleanSpaces() { return cleanSpaces; }
    /**
     * {@code ignore_merges}: a BPE word that is itself a token skips merging. The pre-tokenizer
     * name's flag, overridden by {@link GgufKeys#NF_TOKENIZER_BPE_IGNORE_MERGES}.
     */
    public boolean ignoreMerges() { return ignoreMerges; }
    public boolean addBos() { return addBos; }
    public boolean addEos() { return addEos; }
    public boolean addSep() { return addSep; }
    public boolean dnaAutoTags() { return dnaAutoTags; }
    public boolean normalizerNfc() { return normalizerNfc; }
    public boolean normalizerLowercase() { return normalizerLowercase; }
    public boolean normalizerStripAccents() { return normalizerStripAccents; }

    /**
     * The Hugging Face normalizer chain BPE encoding applies to each text fragment
     * ({@link GgufKeys#NF_TOKENIZER_NORMALIZER}); the identity when absent. Never null.
     */
    public Normalizer normalizer() { return normalizer; }

    /**
     * The Hugging Face decoder chain that replaces piece concatenation in detokenization
     * ({@link GgufKeys#NF_TOKENIZER_DECODER}); null when absent.
     */
    public TokenDecoder decoder() { return decoder; }

    /**
     * When SPM encoding prefixes a fragment with a space ({@link GgufKeys#NF_TOKENIZER_PREPEND_SCHEME}):
     * {@code ALWAYS} (the default, llama.cpp) at the start and after every special token,
     * {@code FIRST} only at the start of the input when the text does not begin with a space,
     * {@code NEVER} not at all.
     */
    public PreTokenizerSpec.PrependScheme prependScheme() { return prependScheme; }

    /**
     * The {@link #KEY_METASPACE_PREPEND_SCHEME} a BPE vocabulary applies: {@code FIRST} puts U+2581
     * before the first fragment of the input unless it starts with one, and detokenization drops
     * the leading space of the first rendered piece; {@code NEVER} otherwise, which is also the
     * value for the other vocabulary types and for a BPE vocabulary whose
     * {@link GgufKeys#NF_TOKENIZER_PRE_TOKENIZER} spec carries its own Metaspace step.
     */
    public PreTokenizerSpec.PrependScheme metaspacePrependScheme() { return metaspacePrependScheme; }

    /**
     * The post-processor ids put before the text in place of BOS when special tokens are added
     * ({@link GgufKeys#NF_TOKENIZER_PREFIX_IDS}); a copy, or null when absent.
     */
    public int[] prefixIds() { return prefixIds != null ? prefixIds.clone() : null; }

    /**
     * The post-processor ids put after the text in place of EOS when special tokens are added
     * ({@link GgufKeys#NF_TOKENIZER_SUFFIX_IDS}); a copy, or null when absent.
     */
    public int[] suffixIds() { return suffixIds != null ? suffixIds.clone() : null; }

    /**
     * End-of-generation ids beyond EOS/EOT/EOM ({@link GgufKeys#NF_TOKENIZER_EXTRA_EOG_IDS}); a
     * copy, empty when absent.
     */
    public int[] extraEogIds() { return extraEogIds.clone(); }

    /**
     * Hugging Face {@code clean_up_tokenization_spaces} ({@link GgufKeys#NF_TOKENIZER_CLEAN_SPACES}),
     * applied after detokenization; null when absent, in which case {@link #cleanSpaces()} applies.
     */
    public Boolean cleanUpTokenizationSpaces() { return cleanUpTokenizationSpaces; }

    /** BPE {@code add_prefix_space}: a space before each text fragment that does not start with one. */
    public boolean bpeAddPrefixSpace() { return bpeAddPrefixSpace; }

    /**
     * True when any of {@link GgufKeys#NF_TOKENIZER_BPE_UNK_FUSE},
     * {@link GgufKeys#NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX} or
     * {@link GgufKeys#NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX} is present, or
     * {@link GgufKeys#NF_TOKENIZER_BPE_BYTE_FALLBACK} is true: BPE words then merge as the Hugging
     * Face BPE model does, with byte fallback or the unknown token for characters that are not
     * tokens.
     */
    public boolean bpeModelOverlay() { return bpeModelOverlay; }

    /** BPE {@code fuse_unk}: consecutive unknown characters become one unknown token. */
    public boolean bpeUnkFuse() { return bpeUnkFuse; }

    /**
     * BPE {@code byte_fallback} ({@link GgufKeys#NF_TOKENIZER_BPE_BYTE_FALLBACK}): whether a
     * character that is not a token becomes its {@code <0xXX>} byte tokens; null when absent, in
     * which case encoding falls back to byte tokens whenever the pre-tokenizer does not
     * byte-encode and the byte tokens exist.
     */
    public Boolean bpeByteFallback() { return bpeByteFallback; }

    /** BPE {@code continuing_subword_prefix} of every symbol but a word's first; empty when absent. */
    public String bpeContinuingSubwordPrefix() { return bpeContinuingSubwordPrefix; }

    /** BPE {@code end_of_word_suffix} of a word's last symbol; empty when absent. */
    public String bpeEndOfWordSuffix() { return bpeEndOfWordSuffix; }

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
        PreTokenizerRegistry.PreTokenizer namedPreTokenizer;
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
        Normalizer normalizer = Normalizer.identity();
        TokenDecoder decoder;
        PreTokenizerSpec.PrependScheme prependScheme = PreTokenizerSpec.PrependScheme.ALWAYS;
        PreTokenizerSpec.PrependScheme metaspacePrependScheme = PreTokenizerSpec.PrependScheme.NEVER;
        boolean preTokenizerSpecOverlay;
        int[] prefixIds;
        int[] suffixIds;
        int[] extraEogIds = new int[0];
        Boolean cleanUpTokenizationSpaces;
        boolean bpeAddPrefixSpace;
        boolean bpeModelOverlay;
        boolean bpeUnkFuse;
        Boolean bpeByteFallback;
        String bpeContinuingSubwordPrefix = "";
        String bpeEndOfWordSuffix = "";

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
            Boolean clean = bool(KEY_CLEAN_SPACES);
            if (clean != null) cleanSpaces = clean;
            loadTokens();
            applySpecialFlags();
            String scheme = metaspaceScheme();
            applyOverlays();
            applyMetaspaceScheme(scheme);
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
            for (String k : new String[]{KEY_ARCHITECTURE, KEY_NAME, vocabSizeKey, GgufKeys.NF_RERANK_SUFFIX_IDS}) {
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

        /**
         * llama-vocab.cpp:2307-2336: the XCDA blob size must leave room for the replacements. The
         * blob is a {@code byte[]} or a list of byte numbers.
         */
        private void loadCharsmap() {
            Object value = md.get(KEY_PRECOMPILED_CHARSMAP);
            if (value == null) return;
            byte[] blob;
            if (value instanceof byte[]) {
                blob = ((byte[]) value).clone();
            } else {
                List<?> raw = list(KEY_PRECOMPILED_CHARSMAP);
                blob = new byte[raw.size()];
                for (int i = 0; i < blob.length; i++) {
                    Object o = raw.get(i);
                    if (!(o instanceof Number)) {
                        throw new IllegalArgumentException(KEY_PRECOMPILED_CHARSMAP + "[" + i + "] is not a byte");
                    }
                    blob[i] = (byte) ((Number) o).intValue();
                }
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
                    namedPreTokenizer = preTokenizer;
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

        // -----------------------------------------------------------------------------------------
        // Hugging Face overlays (neuralfree.tokenizer.*); each absent key leaves its default
        // -----------------------------------------------------------------------------------------

        /**
         * The {@code neuralfree.tokenizer.*} keys, after every GGUF key: token attributes, extra
         * end-of-generation ids, the decoder and clean-up for every type; the normalizer,
         * pre-tokenizer spec, post-processor ids and BPE model options for BPE; the prepend scheme
         * and post-processor ids for SPM. A key for a type that does not read it is logged and ignored.
         */
        private void applyOverlays() {
            int n = texts.length;
            applyTokenAttrs(n);
            extraEogIds = ids(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, n, extraEogIds);
            String json = str(GgufKeys.NF_TOKENIZER_DECODER);
            if (json != null) {
                decoder = parsed(GgufKeys.NF_TOKENIZER_DECODER, () -> TokenDecoder.fromCanonicalJson(json));
            }
            Boolean clean = bool(GgufKeys.NF_TOKENIZER_CLEAN_SPACES);
            if (clean != null) {
                cleanUpTokenizationSpaces = clean;
                cleanSpaces = false;
            }
            if (type == Type.BPE || type == Type.SPM) {
                prefixIds = ids(GgufKeys.NF_TOKENIZER_PREFIX_IDS, n, null);
                suffixIds = ids(GgufKeys.NF_TOKENIZER_SUFFIX_IDS, n, null);
            } else {
                ignored(GgufKeys.NF_TOKENIZER_PREFIX_IDS);
                ignored(GgufKeys.NF_TOKENIZER_SUFFIX_IDS);
            }
            if (type == Type.SPM) {
                String scheme = str(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME);
                if (scheme != null) prependScheme = prependScheme(scheme);
            } else {
                ignored(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME);
            }
            if (type == Type.BPE) {
                applyBpeOverlays();
            } else {
                for (String key : BPE_ONLY_KEYS) ignored(key);
            }
        }

        /** {@link GgufKeys#NF_TOKENIZER_TOKEN_ATTRS}: one attribute mask per token, OR-ed in. */
        private void applyTokenAttrs(int n) {
            List<?> raw = list(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS);
            if (raw == null) return;
            String key = GgufKeys.NF_TOKENIZER_TOKEN_ATTRS;
            if (raw.size() != n) {
                throw new GgufFormatException(key + " has " + raw.size() + " entries for " + n + " tokens");
            }
            for (int i = 0; i < n; i++) {
                long bits = integer(raw.get(i), key, i);
                if (bits < 0 || (bits & ~ATTR_ALL) != 0) {
                    throw new GgufFormatException(key + "[" + i + "] = " + bits + " has bits outside the token"
                            + " attribute mask 0x" + Integer.toHexString(ATTR_ALL));
                }
                attrs[i] |= (int) bits;
            }
        }

        /**
         * Normalizer, pre-tokenizer spec and BPE model options of a BPE vocabulary. The gemma4
         * model keeps its forced gemma4 pre-tokenizer, as llama.cpp does.
         */
        private void applyBpeOverlays() {
            String json = str(GgufKeys.NF_TOKENIZER_NORMALIZER);
            if (json != null) {
                normalizer = parsed(GgufKeys.NF_TOKENIZER_NORMALIZER, () -> Normalizer.fromCanonicalJson(json));
            }
            if ("gemma4".equals(model)) {
                if (md.containsKey(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER)) {
                    LOGGER.warn("tokenizer metadata '{}' is ignored: model 'gemma4' always uses the gemma4"
                            + " pre-tokenizer", GgufKeys.NF_TOKENIZER_PRE_TOKENIZER);
                }
            } else {
                String spec = str(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER);
                if (spec != null) {
                    preTokenizerSpecOverlay = true;
                    preTokenizer = parsed(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER,
                            () -> PreTokenizerRegistry.fromSpec(PreTokenizerSpec.fromCanonicalJson(spec)));
                }
            }
            Boolean b = bool(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES);
            if (b != null) ignoreMerges = b;
            b = bool(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE);
            if (b != null) bpeAddPrefixSpace = b;
            Boolean fuse = bool(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE);
            String prefix = str(GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX);
            String suffix = str(GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX);
            bpeByteFallback = bool(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK);
            bpeModelOverlay = fuse != null || prefix != null || suffix != null || Boolean.TRUE.equals(bpeByteFallback);
            if (fuse != null) bpeUnkFuse = fuse;
            if (prefix != null) bpeContinuingSubwordPrefix = prefix;
            if (suffix != null) bpeEndOfWordSuffix = suffix;
        }

        /**
         * An optional list of token ids, each in {@code [0, n)}; {@code absent} when the key is
         * missing.
         */
        private int[] ids(String key, int n, int[] absent) {
            List<?> raw = list(key);
            if (raw == null) return absent;
            int[] out = new int[raw.size()];
            for (int i = 0; i < out.length; i++) {
                long id = integer(raw.get(i), key, i);
                if (id < 0 || id >= n) {
                    throw new GgufFormatException(key + "[" + i + "] = " + id + " is outside the vocabulary [0, "
                            + n + ")");
                }
                out[i] = (int) id;
            }
            return out;
        }

        private static long integer(Object o, String key, int i) {
            if (!(o instanceof Byte || o instanceof Short || o instanceof Integer || o instanceof Long)) {
                throw new GgufFormatException(key + "[" + i + "] is not an integer: " + o);
            }
            return ((Number) o).longValue();
        }

        private static PreTokenizerSpec.PrependScheme prependScheme(String value) {
            switch (value) {
                case "always": return PreTokenizerSpec.PrependScheme.ALWAYS;
                case "first": return PreTokenizerSpec.PrependScheme.FIRST;
                case "never": return PreTokenizerSpec.PrependScheme.NEVER;
                default:
                    throw new GgufFormatException(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME + " is '" + value
                            + "'; expected always, first or never");
            }
        }

        /**
         * {@link #KEY_METASPACE_PREPEND_SCHEME} (llama-vocab.cpp:3022-3038): absent, never, first or
         * always, any other value failing the load; {@code always} on a BPE vocabulary fails it too
         * unless a {@link GgufKeys#NF_TOKENIZER_PRE_TOKENIZER} spec is present, which {@link
         * #applyMetaspaceScheme} learns after the overlays. Null when absent.
         */
        private String metaspaceScheme() {
            String scheme = str(KEY_METASPACE_PREPEND_SCHEME);
            if (scheme == null || scheme.equals("never") || scheme.equals("first") || scheme.equals("always")) {
                return scheme;
            }
            throw new IllegalArgumentException("unknown metaspace prepend_scheme: '" + scheme + "' ("
                    + KEY_METASPACE_PREPEND_SCHEME + ")");
        }

        /**
         * The scheme a BPE vocabulary applies; the other types ignore the key, as llama.cpp does. A
         * {@link GgufKeys#NF_TOKENIZER_PRE_TOKENIZER} spec splits with its own Metaspace step, so the
         * key does not apply on top of it.
         */
        private void applyMetaspaceScheme(String scheme) {
            if (type != Type.BPE || scheme == null || preTokenizerSpecOverlay) return;
            if (scheme.equals("always")) {
                throw new IllegalArgumentException("metaspace prepend_scheme 'always' is not supported for BPE vocab ("
                        + KEY_METASPACE_PREPEND_SCHEME + ")");
            }
            if (scheme.equals("first")) metaspacePrependScheme = PreTokenizerSpec.PrependScheme.FIRST;
        }

        /** Builds an overlay value; a malformed one fails the load naming its key. */
        private static <T> T parsed(String key, Supplier<T> build) {
            try {
                return build.get();
            } catch (IllegalArgumentException e) {
                throw new GgufFormatException(key + ": " + e.getMessage(), e);
            }
        }

        /** Logs an overlay key that is present but not read by this vocabulary type. */
        private void ignored(String key) {
            if (md.containsKey(key)) {
                LOGGER.warn("tokenizer metadata '{}' does not apply to {} vocabularies (model '{}'); it is ignored",
                        key, type, model);
            }
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
