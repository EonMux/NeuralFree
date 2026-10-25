package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.Normalizer;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerPatterns;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import com.eonmux.neuralfree.gguf.tokenizer.TokenDecoder;
import com.eonmux.neuralfree.utils.json.JsonException;
import com.eonmux.neuralfree.utils.json.JsonView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Turns a Hugging Face {@code pre_tokenizer} object into a {@link PreTokenizerSpec} and derives the llama.cpp
 * vocabulary flags a tokenizer.json states, for naming the pre-tokenizer as llama.cpp does.
 *
 * <p>{@link #toSpec} is exact: every supported element ({@code Sequence}, {@code Split}, {@code ByteLevel},
 * {@code Metaspace}, {@code Digits}, {@code Punctuation}, {@code Whitespace}, {@code WhitespaceSplit},
 * {@code BertPreTokenizer}, {@code UnicodeScripts}, {@code CharDelimiterSplit}) becomes the step with the same
 * behaviour, members absent from the file take the Hugging Face defaults, nested sequences are flattened, and
 * {@code Split} regexes keep their tokenizer.json text in the {@link PreTokenizerSpec.Dialect#ONIGURUMA} dialect.
 * Unknown element types and members are rejected.</p>
 *
 * <p>{@link #canonicalize} adds the structural form llama.cpp names are compared in
 * ({@link CanonicalPreTokenizer#regexSequence()}) and the llama.cpp vocabulary flags tokenizer.json states:
 * {@code ignore_merges} from the model, whitespace escaping from the pre-tokenizer or normalizer, and
 * {@code add_sep} from the post-processor.</p>
 */
public final class PreTokenizerCanonicalizer {

    private static final String DEFAULT_SOURCE = "tokenizer pre_tokenizer";

    /** Case-insensitive contractions as Hugging Face writes them. */
    private static final String CONTRACTIONS_CASE_INSENSITIVE = "(?i:'s|'t|'re|'ve|'m|'ll|'d)";
    /** The same contractions spelled out case by case, as llama.cpp rewrites them for {@code std::regex}. */
    private static final String CONTRACTIONS_SPELLED_OUT =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";

    /** Hugging Face {@code Digits} as the llama.cpp regex step with the same splits. */
    private static final String DIGITS_INDIVIDUAL = "\\p{N}";
    private static final String DIGITS_RUNS = "\\p{N}+";

    /** The normalizer that escapes spaces to U+2581 (SentencePiece-style BPE). */
    private static final Normalizer SPACE_TO_METASPACE = Normalizer.replace(" ", "▁");

    private PreTokenizerCanonicalizer() {}

    /**
     * A tokenizer.json pre-tokenizer in canonical form: its exact {@link #spec()}, the llama.cpp regex sequence with
     * the same splits when one exists, and the vocabulary flags the file states. Immutable.
     */
    public static final class CanonicalPreTokenizer {
        private final PreTokenizerSpec spec;
        private final List<String> regexSequence;
        private final boolean ignoreMerges;
        private final boolean escapeWhitespaces;
        private final Boolean addSep;

        CanonicalPreTokenizer(PreTokenizerSpec spec, List<String> regexSequence, boolean ignoreMerges,
                              boolean escapeWhitespaces, Boolean addSep) {
            this.spec = spec;
            this.regexSequence = regexSequence;
            this.ignoreMerges = ignoreMerges;
            this.escapeWhitespaces = escapeWhitespaces;
            this.addSep = addSep;
        }

        /** The exact pre-tokenizer ({@link #toSpec}). */
        public PreTokenizerSpec spec() { return spec; }

        /**
         * The splits as llama.cpp-style isolated regex steps in canonical text ({@link #regexSequence(PreTokenizerSpec)}),
         * or null when a step has no such form.
         */
        public List<String> regexSequence() { return regexSequence; }

        /** The words are GPT-2 byte-encoded ({@code ByteLevel} present). */
        public boolean byteEncode() { return spec.byteEncode(); }

        /** BPE {@code model.ignore_merges}; false for other models. */
        public boolean ignoreMerges() { return ignoreMerges; }

        /** Spaces become U+2581 before splitting: a {@code Metaspace} step or a {@code Replace(" ", "▁")} normalizer. */
        public boolean escapeWhitespaces() { return escapeWhitespaces; }

        /**
         * True when the post-processor is (or contains) {@code RobertaProcessing}, which makes llama.cpp's
         * {@code SpecialVocab} set {@code add_sep} (vocab.py:248-251); null when tokenizer.json alone does not
         * decide it.
         */
        public Boolean addSep() { return addSep; }

        @Override
        public String toString() {
            return "CanonicalPreTokenizer{" + regexSequence + ", byteEncode=" + byteEncode() + ", ignoreMerges="
                    + ignoreMerges + ", escapeWhitespaces=" + escapeWhitespaces + ", addSep=" + addSep + "}";
        }
    }

    /**
     * The canonical form of {@code tj}'s pre-tokenizer.
     *
     * @throws GgufFormatException when the pre-tokenizer, normalizer or post-processor is malformed or uses an
     *                             element NeuralFree does not support
     */
    public static CanonicalPreTokenizer canonicalize(HfTokenizerJson tj) {
        Objects.requireNonNull(tj, "tj");
        PreTokenizerSpec spec = toSpec(tj.preTokenizer());
        boolean ignoreMerges = tj.model().type() == HfTokenizerJson.ModelType.BPE && tj.model().ignoreMerges();
        boolean escape = hasStep(spec, PreTokenizerSpec.Kind.METASPACE) || escapesSpaces(tj);
        return new CanonicalPreTokenizer(spec, regexSequence(spec), ignoreMerges, escape, robertaSep(tj));
    }

    /**
     * The exact {@link PreTokenizerSpec} of a tokenizer.json {@code pre_tokenizer} view; a null view (no
     * pre-tokenizer) is the empty spec, which keeps the whole text as one word.
     *
     * @throws GgufFormatException naming the member when the object is malformed or uses an unsupported element
     */
    public static PreTokenizerSpec toSpec(JsonView preTokenizer) {
        if (preTokenizer == null) return new PreTokenizerSpec(Collections.emptyList(), false, false, false, false);
        try {
            Builder b = new Builder();
            read(preTokenizer, b);
            return new PreTokenizerSpec(b.steps, b.byteLevel, b.addPrefixSpace, b.trimOffsets, b.useRegex);
        } catch (JsonException | IllegalArgumentException e) {
            if (e instanceof GgufFormatException) throw (GgufFormatException) e;
            throw new GgufFormatException("Invalid " + DEFAULT_SOURCE + ": " + e.getMessage(), e);
        }
    }

    /**
     * The exact {@link PreTokenizerSpec} of a parsed {@code pre_tokenizer} value; {@code source} names it in
     * messages. JSON {@code null} is the empty spec.
     *
     * @throws GgufFormatException when the value is malformed or uses an unsupported element
     */
    public static PreTokenizerSpec toSpec(Object parsed, String source) {
        String name = source != null ? source : DEFAULT_SOURCE;
        if (parsed == null) return toSpec(null);
        JsonView v;
        try {
            v = JsonView.of(parsed, name);
        } catch (JsonException e) {
            throw new GgufFormatException("Invalid " + DEFAULT_SOURCE + ": " + e.getMessage(), e);
        }
        return toSpec(v);
    }

    /**
     * The splits of {@code spec} as an ordered list of llama.cpp-style {@link Behaviour#ISOLATED} regex steps
     * (the form of {@code PreTokenizerType} sequences), each in {@link #canonicalRegex canonical text}, or null when
     * a step has no such form:
     * <ul>
     *   <li>an isolated regex split is its pattern (inversion has no effect on isolated splits);</li>
     *   <li>{@code Digits} is {@code \p{N}} (individual digits) or {@code \p{N}+} (runs), the same splits;</li>
     *   <li>{@code ByteLevel} must be the last step (byte encoding after every split, as llama.cpp encodes) and
     *       contributes the GPT-2 regex when it splits ({@code use_regex}); its {@code add_prefix_space}, which
     *       llama.cpp's BPE never applies and NeuralFree carries as a vocabulary option, and its
     *       {@code trim_offsets}, which changes only offsets, are not part of the form;</li>
     *   <li>every other step kind and every other split behaviour has no regex form.</li>
     * </ul>
     */
    public static List<String> regexSequence(PreTokenizerSpec spec) {
        Objects.requireNonNull(spec, "spec");
        List<Step> steps = spec.steps();
        List<String> out = new ArrayList<>(steps.size() + 1);
        for (int i = 0; i < steps.size(); i++) {
            Step s = steps.get(i);
            switch (s.kind()) {
                case REGEX_SPLIT:
                    if (s.behaviour() != Behaviour.ISOLATED) return null;
                    out.add(canonicalRegex(s.pattern()));
                    break;
                case DIGITS:
                    out.add(s.individualDigits() ? DIGITS_INDIVIDUAL : DIGITS_RUNS);
                    break;
                case BYTE_LEVEL:
                    if (i != steps.size() - 1) return null;
                    if (spec.useRegex()) out.add(PreTokenizerPatterns.GPT2.get(0));
                    break;
                default:
                    return null;
            }
        }
        return Collections.unmodifiableList(out);
    }

    /**
     * {@code regex} with Hugging Face's case-insensitive contractions group {@code (?i:'s|'t|'re|'ve|'m|'ll|'d)}
     * spelled out as llama.cpp writes it ({@code (?:'[sS]|'[tT]|...)}), the generic dialect rewrite behind most of
     * llama.cpp's "original regex from tokenizer.json" comments; other text is unchanged.
     */
    public static String canonicalRegex(String regex) {
        Objects.requireNonNull(regex, "regex");
        return regex.replace(CONTRACTIONS_CASE_INSENSITIVE, CONTRACTIONS_SPELLED_OUT);
    }

    // ---------------------------------------------------------------------------------------------
    // pre_tokenizer elements (tokenizers pre_tokenizers/*.rs serde forms)
    // ---------------------------------------------------------------------------------------------

    /** Steps and ByteLevel flags gathered while reading. */
    private static final class Builder {
        final List<Step> steps = new ArrayList<>();
        boolean byteLevel;
        boolean addPrefixSpace;
        boolean trimOffsets;
        boolean useRegex;

        void add(Step s, JsonView v) {
            if (steps.size() >= PreTokenizerSpec.MAX_STEPS) {
                throw new JsonException(v.source() + ":" + v.path() + ": more than " + PreTokenizerSpec.MAX_STEPS
                        + " pre-tokenizer steps");
            }
            steps.add(s);
        }
    }

    private static void read(JsonView v, Builder b) {
        String type = v.str("type");
        switch (type) {
            case "Sequence": {
                allowKeys(v, "pretokenizers");
                List<Object> raw = v.list("pretokenizers");
                String base = v.path().isEmpty() ? "pretokenizers" : v.path() + ".pretokenizers";
                for (int i = 0; i < raw.size(); i++) read(JsonView.of(raw.get(i), v.source(), base + "[" + i + "]"), b);
                break;
            }
            case "Split": {
                allowKeys(v, "pattern", "behavior", "invert");
                JsonView pattern = v.obj("pattern");
                boolean regex = readPatternKind(pattern);
                Behaviour behaviour = behaviour(v, v.str("behavior"));
                boolean invert = v.optBool("invert", false);
                b.add(regex
                        ? Step.regex(pattern.str("Regex"), PreTokenizerSpec.Dialect.ONIGURUMA, behaviour, invert)
                        : Step.literal(pattern.str("String"), behaviour, invert), v);
                break;
            }
            case "ByteLevel":
                allowKeys(v, "add_prefix_space", "trim_offsets", "use_regex");
                if (b.byteLevel) {
                    throw new JsonException(v.source() + ":" + v.path() + ": a second ByteLevel pre-tokenizer would"
                            + " byte-encode twice");
                }
                b.add(Step.byteLevel(), v);
                b.byteLevel = true;
                b.addPrefixSpace = v.optBool("add_prefix_space", true);
                b.trimOffsets = v.optBool("trim_offsets", true);
                b.useRegex = v.optBool("use_regex", true);
                break;
            case "Metaspace":
                allowKeys(v, "replacement", "prepend_scheme", "split", "add_prefix_space", "str_rep");
                b.add(Step.metaspace(singleCodePoint(v, "replacement"), TokenDecoder.readPrependScheme(v),
                        v.optBool("split", true)), v);
                break;
            case "Digits":
                allowKeys(v, "individual_digits");
                b.add(Step.digits(v.optBool("individual_digits", false)), v);
                break;
            case "Punctuation":
                allowKeys(v, "behavior");
                b.add(Step.punctuation(v.raw("behavior") == null ? Behaviour.ISOLATED
                        : behaviour(v, v.str("behavior"))), v);
                break;
            case "Whitespace":
                allowKeys(v);
                b.add(Step.whitespace(), v);
                break;
            case "WhitespaceSplit":
                allowKeys(v);
                b.add(Step.whitespaceSplit(), v);
                break;
            case "BertPreTokenizer":
                allowKeys(v);
                b.add(Step.bert(), v);
                break;
            case "UnicodeScripts":
                allowKeys(v);
                b.add(Step.unicodeScripts(), v);
                break;
            case "CharDelimiterSplit":
                allowKeys(v, "delimiter");
                b.add(Step.charDelimiter(singleCodePoint(v, "delimiter")), v);
                break;
            default:
                throw new GgufFormatException(v.where("type") + ": tokenizer pre_tokenizer type '" + type
                        + "' is not supported");
        }
    }

    /** tokenizers {@code SplitDelimiterBehavior} (serialized in PascalCase). */
    private static Behaviour behaviour(JsonView v, String s) {
        switch (s) {
            case "Removed": return Behaviour.REMOVED;
            case "Isolated": return Behaviour.ISOLATED;
            case "MergedWithPrevious": return Behaviour.MERGED_WITH_PREVIOUS;
            case "MergedWithNext": return Behaviour.MERGED_WITH_NEXT;
            case "Contiguous": return Behaviour.CONTIGUOUS;
            default:
                throw new JsonException(v.where("behavior") + ": unknown value '" + s + "', expected Removed, Isolated,"
                        + " MergedWithPrevious, MergedWithNext or Contiguous");
        }
    }

    /** A {@code {"String": s}} or {@code {"Regex": r}} pattern; true for the regex form. */
    private static boolean readPatternKind(JsonView pattern) {
        if (pattern.keys().size() != 1 || !(pattern.has("Regex") || pattern.has("String"))) {
            throw new JsonException(pattern.source() + ":" + pattern.path()
                    + ": expected exactly one of 'String' or 'Regex', got " + pattern.keys());
        }
        return pattern.has("Regex");
    }

    private static int singleCodePoint(JsonView v, String key) {
        String s = v.str(key);
        if (s.isEmpty() || s.codePointCount(0, s.length()) != 1) {
            throw new JsonException(v.where(key) + ": expected exactly one character, got '" + s + "'");
        }
        return s.codePointAt(0);
    }

    /** Rejects members other than {@code type} and {@code allowed}. */
    private static void allowKeys(JsonView v, String... allowed) {
        Set<String> known = new HashSet<>(Arrays.asList(allowed));
        known.add("type");
        for (String k : v.keys()) {
            if (!known.contains(k)) throw new JsonException(v.where(k) + ": unknown member");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Flags stated by the other tokenizer.json components
    // ---------------------------------------------------------------------------------------------

    private static boolean hasStep(PreTokenizerSpec spec, PreTokenizerSpec.Kind kind) {
        for (Step s : spec.steps()) {
            if (s.kind() == kind) return true;
        }
        return false;
    }

    /** The normalizer chain replaces each space with U+2581. */
    private static boolean escapesSpaces(HfTokenizerJson tj) {
        Object raw = tj.root().raw("normalizer");
        if (raw == null) return false;
        return containsSpaceEscape(Normalizer.fromJson(raw, tj.source() + ":normalizer"));
    }

    private static boolean containsSpaceEscape(Normalizer n) {
        if (n.kind() == Normalizer.Kind.SEQUENCE) {
            for (Normalizer c : n.children()) {
                if (containsSpaceEscape(c)) return true;
            }
            return false;
        }
        return n.equals(SPACE_TO_METASPACE);
    }

    /** TRUE when the post-processor is or contains {@code RobertaProcessing}, else null. */
    private static Boolean robertaSep(HfTokenizerJson tj) {
        JsonView pp = tj.postProcessor();
        if (pp == null) return null;
        try {
            if ("RobertaProcessing".equals(pp.str("type"))) return Boolean.TRUE;
            if ("Sequence".equals(pp.str("type")) && pp.raw("processors") != null) {
                List<Object> raw = pp.list("processors");
                String base = pp.path().isEmpty() ? "processors" : pp.path() + ".processors";
                for (int i = 0; i < raw.size(); i++) {
                    JsonView p = JsonView.of(raw.get(i), pp.source(), base + "[" + i + "]");
                    if ("RobertaProcessing".equals(p.str("type"))) return Boolean.TRUE;
                }
            }
            return null;
        } catch (JsonException e) {
            throw new GgufFormatException("Invalid tokenizer post_processor: " + e.getMessage(), e);
        }
    }
}
