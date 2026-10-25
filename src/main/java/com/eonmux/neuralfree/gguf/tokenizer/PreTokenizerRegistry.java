package com.eonmux.neuralfree.gguf.tokenizer;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Maps a GGUF {@code tokenizer.ggml.pre} name to its BPE pre-tokenizer: the
 * {@link PreTokenizerType} and the vocabulary flags llama.cpp sets for that name in
 * {@code llama_vocab::impl::load} (src/llama-vocab.cpp:2418-2699). An unknown name is an error,
 * as in llama.cpp; a missing name selects {@code default} with a quality warning.
 *
 * <p>Every name is a pre-built {@link PreTokenizerSpec} of its type's llama.cpp regex steps
 * ({@link #specFor}); {@link #fromSpec} builds a pre-tokenizer from any other spec, such as a
 * Hugging Face {@code pre_tokenizer} configuration.</p>
 */
public final class PreTokenizerRegistry {

    /** The name used when {@code tokenizer.ggml.pre} is absent or empty. */
    public static final String DEFAULT_NAME = "default";

    /** The {@link PreTokenizer#name()} of a pre-tokenizer built by {@link #fromSpec}. */
    public static final String CUSTOM_NAME = "custom";

    private static final Logger LOGGER = LogManager.getLogger(PreTokenizerRegistry.class);
    private static final Map<String, Entry> TABLE = buildTable();
    private static final ConcurrentMap<String, PreTokenizer> CACHE = new ConcurrentHashMap<>();

    private PreTokenizerRegistry() {}

    /**
     * Returns the pre-tokenizer for a {@code tokenizer.ggml.pre} value.
     *
     * @throws IllegalArgumentException for a name llama.cpp does not know
     */
    public static PreTokenizer forName(String pre) {
        String name = pre;
        if (name == null || name.isEmpty()) {
            LOGGER.warn("missing pre-tokenizer type, using '{}'; GENERATION QUALITY WILL BE DEGRADED,"
                    + " consider regenerating the model", DEFAULT_NAME);
            name = DEFAULT_NAME;
        }
        Entry e = entry(name);
        return CACHE.computeIfAbsent(name, n -> new PreTokenizer(n, e, specOf(e.type)));
    }

    /**
     * The spec a registry name is built from: its type's llama.cpp regex steps, each
     * {@link PreTokenizerSpec.Behaviour#ISOLATED}, and its type's byte-encoding flag.
     *
     * @throws IllegalArgumentException for a name llama.cpp does not know
     */
    public static PreTokenizerSpec specFor(String pre) {
        if (pre == null) throw new IllegalArgumentException("pre-tokenizer name is null");
        return specOf(entry(pre).type);
    }

    /**
     * A pre-tokenizer that splits as {@code spec} describes, with llama.cpp's BPE default
     * vocabulary flags (those of {@value #DEFAULT_NAME}); its {@link PreTokenizer#type()} is null
     * and its name is {@value #CUSTOM_NAME}.
     *
     * @throws IllegalArgumentException when a regex of the spec cannot be compiled
     */
    public static PreTokenizer fromSpec(PreTokenizerSpec spec) {
        if (spec == null) throw new IllegalArgumentException("pre-tokenizer spec is null");
        return new PreTokenizer(CUSTOM_NAME, new Entry(null, true, false, false, null, false, false, null), spec);
    }

    /** True when llama.cpp knows the {@code tokenizer.ggml.pre} name. */
    public static boolean isKnown(String pre) {
        return pre != null && TABLE.containsKey(pre);
    }

    /** Every supported {@code tokenizer.ggml.pre} name, sorted. */
    public static Set<String> names() {
        return Collections.unmodifiableSet(new TreeSet<>(TABLE.keySet()));
    }

    private static Entry entry(String name) {
        Entry e = TABLE.get(name);
        if (e == null) {
            throw new IllegalArgumentException("unknown pre-tokenizer type: '" + name + "' (tokenizer.ggml.pre);"
                    + " " + TABLE.size() + " names are supported, see PreTokenizerRegistry.names()");
        }
        return e;
    }

    private static PreTokenizerSpec specOf(PreTokenizerType type) {
        return PreTokenizerSpec.ofRegexes(type.regexes(), type.byteEncode());
    }

    // ---------------------------------------------------------------------------------------------
    // Name table (src/llama-vocab.cpp:2418-2699, in source order)
    // ---------------------------------------------------------------------------------------------

    private static Map<String, Entry> buildTable() {
        Map<String, Entry> m = new HashMap<>();
        put(m, e(PreTokenizerType.FLINT).noCleanSpaces(), "flint");
        put(m, e(PreTokenizerType.DEFAULT), "default");
        put(m, e(PreTokenizerType.MINICPM5).ignoreMerges(), "minicpm5");
        put(m, e(PreTokenizerType.LLAMA3).ignoreMerges().addBos(), "llama3", "llama-v3", "llama-bpe", "falcon3",
                "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano");
        put(m, e(PreTokenizerType.DEEPSEEK_LLM).noCleanSpaces(), "deepseek-llm");
        put(m, e(PreTokenizerType.DEEPSEEK_CODER).noCleanSpaces(), "deepseek-coder");
        put(m, e(PreTokenizerType.DEEPSEEK3_LLM).noCleanSpaces(), "deepseek-v3");
        put(m, e(PreTokenizerType.SPARK2_5).noCleanSpaces(), "spark2_5");
        put(m, e(PreTokenizerType.YOUTU).noCleanSpaces().ignoreMerges(), "youtu");
        put(m, e(PreTokenizerType.FALCON), "falcon");
        put(m, e(PreTokenizerType.MPT), "mpt");
        put(m, e(PreTokenizerType.STARCODER), "starcoder");
        put(m, e(PreTokenizerType.GPT2), "gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es",
                "jina-v2-de", "a.x-4.0", "mellum", "modern-bert");
        put(m, e(PreTokenizerType.JAIS2), "jais-2");
        put(m, e(PreTokenizerType.GEMMA4).escapeWhitespaces(), "gemma4", "granite-embed-multi-311m");
        put(m, e(PreTokenizerType.MMBERT).escapeWhitespaces(), "mmbert");
        put(m, e(PreTokenizerType.SARVAM_MOE).escapeWhitespaces().noCleanSpaces(), "sarvam-moe");
        put(m, e(PreTokenizerType.GPT2).addSep(), "jina-v1-en", "jina-v2-code", "roberta-bpe");
        put(m, e(PreTokenizerType.WHITESPACE).normalizerLowercase(false), "whitespace");
        put(m, e(PreTokenizerType.REFACT), "refact");
        put(m, e(PreTokenizerType.COMMAND_R).noCleanSpaces(), "command-r");
        put(m, e(PreTokenizerType.QWEN2).noCleanSpaces(), "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2");
        put(m, e(PreTokenizerType.QWEN35).noCleanSpaces(), "qwen35");
        put(m, e(PreTokenizerType.STABLELM2), "stablelm2");
        put(m, e(PreTokenizerType.OLMO), "olmo");
        put(m, e(PreTokenizerType.DBRX), "dbrx");
        put(m, e(PreTokenizerType.SMAUG), "smaug-bpe");
        put(m, e(PreTokenizerType.PORO).noCleanSpaces(), "poro-chat");
        put(m, e(PreTokenizerType.CHATGLM4).suppressBos(), "chatglm-bpe");
        put(m, e(PreTokenizerType.CHATGLM4).suppressBos().ignoreMerges(), "glm4", "glm5");
        put(m, e(PreTokenizerType.VIKING).noCleanSpaces(), "viking");
        put(m, e(PreTokenizerType.JAIS), "jais");
        put(m, e(PreTokenizerType.TEKKEN).noCleanSpaces().ignoreMerges().addBos(), "tekken");
        put(m, e(PreTokenizerType.SMOLLM).noCleanSpaces(), "smollm");
        put(m, e(PreTokenizerType.CODESHELL), "codeshell");
        put(m, e(PreTokenizerType.BLOOM), "bloom");
        put(m, e(PreTokenizerType.GPT3_FINNISH), "gpt3-finnish");
        put(m, e(PreTokenizerType.EXAONE), "exaone");
        put(m, e(PreTokenizerType.GPT2), "exaone4");
        put(m, e(PreTokenizerType.EXAONE_MOE), "exaone-moe");
        put(m, e(PreTokenizerType.CHAMELEON).addBos().noCleanSpaces(), "chameleon");
        put(m, e(PreTokenizerType.MINERVA), "minerva-7b");
        put(m, e(PreTokenizerType.QWEN2), "megrez");
        put(m, e(PreTokenizerType.GPT4O).noCleanSpaces(), "gpt-4o", "llama4", "kanana2", "talkie");
        put(m, e(PreTokenizerType.GRANITE_EMB_MULTI).noCleanSpaces().ignoreMerges(), "granite-embed-multi-97m");
        put(m, e(PreTokenizerType.TINY_AYA).noCleanSpaces(), "tiny_aya", "cohere2moe");
        put(m, e(PreTokenizerType.SUPERBPE).noCleanSpaces(), "superbpe");
        put(m, e(PreTokenizerType.TRILLION).noCleanSpaces(), "trillion");
        put(m, e(PreTokenizerType.GRANITE_DOCLING).noCleanSpaces(), "granite-docling");
        put(m, e(PreTokenizerType.BAILINGMOE).noCleanSpaces(), "bailingmoe", "bailingmoe2", "llada-moe");
        put(m, e(PreTokenizerType.SEED_CODER).noCleanSpaces(), "seed-coder");
        put(m, e(PreTokenizerType.HUNYUAN).noCleanSpaces(), "hunyuan");
        put(m, e(PreTokenizerType.HUNYUAN_DENSE).noCleanSpaces(), "hunyuan-dense");
        put(m, e(PreTokenizerType.HY_V4).noCleanSpaces(), "hy_v4");
        put(m, e(PreTokenizerType.JOYAI_LLM).noCleanSpaces(), "joyai-llm");
        put(m, e(PreTokenizerType.KIMI_K2).noCleanSpaces(), "kimi-k2");
        put(m, e(PreTokenizerType.UFAKZEKA).noCleanSpaces(), "ufakzeka");
        put(m, e(PreTokenizerType.GROK_2).noCleanSpaces(), "grok-2");
        put(m, e(PreTokenizerType.AFMOE).noCleanSpaces(), "afmoe");
        put(m, e(PreTokenizerType.LAGUNA).noCleanSpaces(), "laguna");
        put(m, e(PreTokenizerType.MINIMAX_M2).noCleanSpaces(), "minimax-m2");
        put(m, e(PreTokenizerType.SOLAR_OPEN).noCleanSpaces(), "solar-open");
        put(m, e(PreTokenizerType.MELLUM2), "mellum2");
        put(m, e(PreTokenizerType.K2_HORIZON).noCleanSpaces(), "k2-horizon");
        return Collections.unmodifiableMap(m);
    }

    private static Entry e(PreTokenizerType type) {
        return new Entry(type, true, false, false, null, false, false, null);
    }

    private static void put(Map<String, Entry> m, Entry entry, String... names) {
        for (String n : names) {
            if (m.put(n, entry) != null) throw new IllegalStateException("Duplicate pre-tokenizer name: " + n);
        }
    }

    /**
     * The flags llama.cpp sets for one name, starting from its BPE defaults
     * ({@code clean_spaces = true}, {@code escape_whitespaces = false}). The type is null for a
     * spec-built pre-tokenizer. Immutable.
     */
    private static final class Entry {
        final PreTokenizerType type;
        final boolean cleanSpaces;
        final boolean ignoreMerges;
        final boolean escapeWhitespaces;
        final Boolean addBos;
        final boolean addSep;
        final boolean suppressBos;
        final Boolean normalizerLowercase;

        Entry(PreTokenizerType type, boolean cleanSpaces, boolean ignoreMerges, boolean escapeWhitespaces,
              Boolean addBos, boolean addSep, boolean suppressBos, Boolean normalizerLowercase) {
            this.type = type;
            this.cleanSpaces = cleanSpaces;
            this.ignoreMerges = ignoreMerges;
            this.escapeWhitespaces = escapeWhitespaces;
            this.addBos = addBos;
            this.addSep = addSep;
            this.suppressBos = suppressBos;
            this.normalizerLowercase = normalizerLowercase;
        }

        Entry noCleanSpaces() {
            return new Entry(type, false, ignoreMerges, escapeWhitespaces,
                    addBos, addSep, suppressBos, normalizerLowercase);
        }

        Entry ignoreMerges() {
            return new Entry(type, cleanSpaces, true, escapeWhitespaces,
                    addBos, addSep, suppressBos, normalizerLowercase);
        }

        Entry escapeWhitespaces() {
            return new Entry(type, cleanSpaces, ignoreMerges, true, addBos, addSep, suppressBos, normalizerLowercase);
        }

        Entry addBos() {
            return new Entry(type, cleanSpaces, ignoreMerges, escapeWhitespaces, Boolean.TRUE, addSep, suppressBos,
                    normalizerLowercase);
        }

        Entry addSep() {
            return new Entry(type, cleanSpaces, ignoreMerges, escapeWhitespaces,
                    addBos, true, suppressBos, normalizerLowercase);
        }

        Entry suppressBos() {
            return new Entry(type, cleanSpaces, ignoreMerges, escapeWhitespaces,
                    addBos, addSep, true, normalizerLowercase);
        }

        Entry normalizerLowercase(boolean v) {
            return new Entry(type, cleanSpaces, ignoreMerges, escapeWhitespaces, addBos, addSep, suppressBos, v);
        }
    }

    /**
     * An immutable BPE pre-tokenizer: the compiled steps of its {@link PreTokenizerSpec} plus the
     * vocabulary flags llama.cpp ties to its {@code tokenizer.ggml.pre} name.
     */
    public static final class PreTokenizer {
        private final String name;
        private final Entry entry;
        private final PreTokenizerSpec spec;
        private final List<UnicodeRegexSplit.Step> steps;
        private final boolean encodesInSplit;

        private PreTokenizer(String name, Entry entry, PreTokenizerSpec spec) {
            this.name = name;
            this.entry = entry;
            this.spec = spec;
            this.steps = UnicodeRegexSplit.compile(spec);
            this.encodesInSplit = spec.encodesBeforeLastStep();
        }

        /**
         * The {@code tokenizer.ggml.pre} name ({@link #DEFAULT_NAME} when the key is absent,
         * {@link #CUSTOM_NAME} for a spec-built pre-tokenizer).
         */
        public String name() { return name; }
        /** The llama.cpp pre-tokenizer type; null for a spec-built pre-tokenizer. */
        public PreTokenizerType type() { return entry.type; }
        /** The spec this pre-tokenizer splits by. */
        public PreTokenizerSpec spec() { return spec; }
        /**
         * The regex sequence, in order: the llama.cpp {@code regex_exprs} of a named
         * pre-tokenizer, the patterns of the {@link PreTokenizerSpec.Kind#REGEX_SPLIT} steps of a
         * spec-built one.
         */
        public List<String> regexes() {
            if (entry.type != null) return entry.type.regexes();
            List<String> out = new ArrayList<>();
            for (PreTokenizerSpec.Step s : spec.steps()) {
                if (s.kind() == PreTokenizerSpec.Kind.REGEX_SPLIT) out.add(s.pattern());
            }
            return Collections.unmodifiableList(out);
        }
        /** GPT-2 byte-to-unicode encoding of each word before merging. */
        public boolean byteEncode() { return spec.byteEncode(); }
        /** {@code clean_spaces} (clean_up_tokenization_spaces) applied by detokenization. */
        public boolean cleanSpaces() { return entry.cleanSpaces; }
        /** {@code ignore_merges}: a word that is itself a vocabulary token skips the merge loop. */
        public boolean ignoreMerges() { return entry.ignoreMerges; }
        /** Spaces are escaped to U+2581 before splitting (SPM-style BPE). */
        public boolean escapeWhitespaces() { return entry.escapeWhitespaces; }
        /** The name's {@code add_bos} default, or null to keep the vocabulary-type default. */
        public Boolean addBos() { return entry.addBos; }
        /** {@code add_bos} is forced on after the GGUF override (Gemma 4 workaround). */
        public boolean forceAddBos() { return entry.type != null && entry.type.forceAddBos(); }
        /** The name sets {@code add_sep}. */
        public boolean addSep() { return entry.addSep; }
        /** The name clears the default BOS id (ChatGLM, GLM-4/5). */
        public boolean suppressBos() { return entry.suppressBos; }
        /** The name's lowercase-normalizer default, or null to keep the default. */
        public Boolean normalizerLowercase() { return entry.normalizerLowercase; }
        /** A word made only of newlines that is a vocabulary token is emitted whole (Gemma 4, mmBERT). */
        public boolean wholeNewlineWords() { return entry.type != null && entry.type.wholeNewlineWords(); }
        /** Words are re-split at each U+2581 (mmBERT metaspace pre-tokenizer). */
        public boolean metaspaceWords() { return entry.type != null && entry.type.metaspaceWords(); }

        /**
         * {@code unicode_regex_split} without the byte encoding, for a text that begins the input.
         * Returns a new mutable list of the non-empty words in text order.
         *
         * @throws IllegalStateException when steps follow the spec's BYTE_LEVEL step, so the
         *         words exist only in byte-encoded form ({@link #words})
         */
        public List<String> split(String text) {
            return split(text, true);
        }

        /**
         * As {@link #split(String)}; {@code inputStart} tells whether the text begins the whole
         * input (a fragment after a special token does not), which the Metaspace
         * {@link PreTokenizerSpec.PrependScheme#FIRST} scheme reads.
         */
        public List<String> split(String text, boolean inputStart) {
            if (text == null) throw new IllegalArgumentException("text is null");
            if (encodesInSplit) {
                throw new IllegalStateException("pre-tokenizer '" + name + "' splits byte-level text after its"
                        + " BYTE_LEVEL step, so its words exist only byte-encoded; use words()");
            }
            return UnicodeRegexSplit.split(text, steps, inputStart);
        }

        /**
         * The words handed to merging, for a text that begins the input: the split words,
         * GPT-2 byte-encoded when {@link #byteEncode()} is set. Returns a new mutable list.
         */
        public List<String> words(String text) {
            return words(text, true);
        }

        /** As {@link #words(String)}, with {@code inputStart} as in {@link #split(String, boolean)}. */
        public List<String> words(String text, boolean inputStart) {
            if (text == null) throw new IllegalArgumentException("text is null");
            List<String> words = UnicodeRegexSplit.split(text, steps, inputStart);
            if (spec.byteEncode() && !encodesInSplit) {
                for (int i = 0; i < words.size(); i++) words.set(i, BpeTokenizer.byteEncode(words.get(i)));
            }
            return words;
        }

        @Override
        public String toString() {
            return "PreTokenizer{" + name + " -> " + (entry.type != null ? entry.type : spec) + "}";
        }
    }
}
