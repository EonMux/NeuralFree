package com.eonmux.neuralfree.gguf.tokenizer.hf;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerPatterns;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerRegistry;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Behaviour;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Dialect;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.PrependScheme;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerSpec.Step;
import com.eonmux.neuralfree.gguf.tokenizer.PreTokenizerType;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver.Basis;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver.Resolution;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreTokenizerCanonicalizer.CanonicalPreTokenizer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link PreTokenizerCanonicalizer} and {@link HfPreTokenizerResolver}: hand-written tokenizer.json pre-tokenizers
 * of the llama-bpe, qwen2, gpt-4o, deepseek-v3, GPT-2, DeepSeek-Coder and BERT kinds resolve to their llama.cpp
 * names structurally and by fingerprint, and ambiguous or conflicting evidence is decided as specified.
 */
public class HfPreTokenizerResolverTest {

    /** Meta-Llama-3 tokenizer.json Split regex. */
    private static final String LLAMA3_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}|"
            + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    /** Qwen2 tokenizer.json Split regex. */
    private static final String QWEN2_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
            + " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    /** o200k (GPT-4o) tokenizer.json Split regex. */
    private static final String GPT4O_REGEX = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
            + "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?"
            + "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            + "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    /** DeepSeek-V3 tokenizer.json Split regexes (its line breaks are literal CR and LF characters). */
    private static final String[] DEEPSEEK_V3_REGEXES = {
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+|"
                    + " ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
    };
    /** DeepSeek-Coder tokenizer.json Split regexes, followed by individual Digits. */
    private static final String[] DEEPSEEK_CODER_REGEXES = {
            "[\r\n]", "\\s?\\p{L}+", "\\s?\\p{P}+", "[一-龥ࠀ-一가-퟿]+"
    };

    /** An encoder whose fingerprint is not in llama.cpp's table. */
    private static final Function<String, int[]> UNKNOWN = text -> new int[]{1, 2, 3};

    private final HfPreTokenizerResolver resolver = new HfPreTokenizerResolver();

    // ---------------------------------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------------------------------

    private static Map<String, Object> obj(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    private static List<Object> arr(Object... items) {
        return new ArrayList<>(Arrays.asList(items));
    }

    private static Map<String, Object> split(String regex) {
        return obj("type", "Split", "pattern", obj("Regex", regex), "behavior", "Isolated", "invert", false);
    }

    private static Map<String, Object> byteLevel(boolean useRegex) {
        return obj("type", "ByteLevel", "add_prefix_space", false, "trim_offsets", true, "use_regex", useRegex);
    }

    private static Map<String, Object> sequence(Object... pretokenizers) {
        return obj("type", "Sequence", "pretokenizers", arr(pretokenizers));
    }

    private static Map<String, Object> splitsThenByteLevel(String... regexes) {
        List<Object> list = new ArrayList<>();
        for (String r : regexes) list.add(split(r));
        list.add(byteLevel(false));
        return obj("type", "Sequence", "pretokenizers", list);
    }

    private static HfTokenizerJson bpe(Map<String, Object> preTokenizer, boolean ignoreMerges) {
        return bpe(preTokenizer, ignoreMerges, null, null);
    }

    private static HfTokenizerJson bpe(Map<String, Object> preTokenizer, boolean ignoreMerges,
                                       Map<String, Object> normalizer, Map<String, Object> postProcessor) {
        Map<String, Object> model = obj("type", "BPE", "vocab", obj(), "merges", arr(), "ignore_merges", ignoreMerges);
        return HfTokenizerJson.of(obj("normalizer", normalizer, "pre_tokenizer", preTokenizer,
                "post_processor", postProcessor, "model", model), "tokenizer.json");
    }

    private static Function<String, int[]> encodes(int[] ids) {
        return text -> {
            assertEquals(ChkhshGoldenTest.CHKTXT, text);
            return ids.clone();
        };
    }

    private static PreTokenizerType typeOf(String name) {
        return PreTokenizerRegistry.forName(name).type();
    }

    // ---------------------------------------------------------------------------------------------
    // Canonicalization and structural names
    // ---------------------------------------------------------------------------------------------

    @Test
    public void llamaBpeStyleCanonicalizesToTheLlama3Sequence() {
        CanonicalPreTokenizer c = PreTokenizerCanonicalizer.canonicalize(bpe(splitsThenByteLevel(LLAMA3_REGEX), true));
        PreTokenizerSpec spec = c.spec();
        assertEquals(Arrays.asList(Step.regex(LLAMA3_REGEX, Dialect.ONIGURUMA, Behaviour.ISOLATED, false),
                Step.byteLevel()), spec.steps());
        assertTrue(spec.byteEncode());
        assertFalse(spec.addPrefixSpace());
        assertTrue(spec.trimOffsets());
        assertFalse(spec.useRegex());
        assertEquals(PreTokenizerPatterns.LLAMA3, c.regexSequence());
        assertTrue(c.ignoreMerges());
        assertFalse(c.escapeWhitespaces());
        assertNull(c.addSep());
    }

    @Test
    public void llamaBpeStyleResolvesToLlamaBpeByFingerprint() {
        Resolution r = resolver.explain(bpe(splitsThenByteLevel(LLAMA3_REGEX), true),
                encodes(ChkhshGoldenTest.LLAMA_BPE_IDS));
        assertEquals("llama-bpe", r.name());
        assertEquals(Basis.HASH_CONFIRMED, r.basis());
        assertTrue(r.candidates().containsAll(Arrays.asList("llama-bpe", "llama3", "dbrx", "smaug-bpe", "glm4")));
        assertEquals("llama-bpe", resolver.resolve(bpe(splitsThenByteLevel(LLAMA3_REGEX), true),
                encodes(ChkhshGoldenTest.LLAMA_BPE_IDS)));
    }

    @Test
    public void llamaBpeStyleWithoutFingerprintKeepsTheLlama3Type() {
        Resolution r = resolver.explain(bpe(splitsThenByteLevel(LLAMA3_REGEX), true), UNKNOWN);
        assertEquals(Basis.STRUCTURAL_AMBIGUOUS, r.basis());
        assertEquals(PreTokenizerType.LLAMA3, typeOf(r.name()));
        assertTrue(r.flagMatches().contains("llama-bpe"));
        assertFalse(r.flagMatches().contains("dbrx"));
        assertFalse(r.flagMatches().contains("smaug-bpe"));
        assertFalse(r.flagMatches().contains("chatglm-bpe"));
    }

    @Test
    public void ignoreMergesSeparatesTypesThatSplitAlike() {
        Resolution r = resolver.explain(bpe(splitsThenByteLevel(LLAMA3_REGEX), false), UNKNOWN);
        assertEquals("dbrx", r.name());
        assertEquals(Arrays.asList("dbrx", "smaug-bpe", "chatglm-bpe"), r.flagMatches());
    }

    @Test
    public void qwen2StyleResolvesToQwen2() {
        HfTokenizerJson tj = bpe(splitsThenByteLevel(QWEN2_REGEX), false);
        Resolution r = resolver.explain(tj, encodes(ChkhshGoldenTest.QWEN2_IDS));
        assertEquals("qwen2", r.name());
        assertEquals(Basis.HASH_CONFIRMED, r.basis());
        assertEquals("stablelm2", r.candidates().get(0));

        Resolution structural = resolver.explain(tj, UNKNOWN);
        assertEquals(Basis.STRUCTURAL_AMBIGUOUS, structural.basis());
        assertEquals("stablelm2", structural.name());
        assertTrue(structural.candidates().contains("qwen2"));
    }

    @Test
    public void gpt4oStyleResolvesToGpt4o() {
        HfTokenizerJson tj = bpe(splitsThenByteLevel(GPT4O_REGEX), false);
        Resolution r = resolver.explain(tj, UNKNOWN);
        assertEquals("gpt-4o", r.name());
        assertEquals(Arrays.asList("gpt-4o", "kanana2", "llama4", "talkie", "minimax-m2"), r.candidates());
    }

    @Test
    public void deepseekV3StyleResolvesToDeepseekV3() {
        Resolution r = resolver.explain(bpe(splitsThenByteLevel(DEEPSEEK_V3_REGEXES), false), UNKNOWN);
        assertEquals("deepseek-v3", r.name());
        assertEquals(Basis.STRUCTURAL_AMBIGUOUS, r.basis());
        assertEquals(Arrays.asList("deepseek-v3", "hunyuan-dense", "joyai-llm", "hy_v4"), r.candidates());
    }

    @Test
    public void byteLevelWithItsOwnRegexMatchesTheGpt2Sequence() {
        HfTokenizerJson tj = bpe(byteLevel(true), false);
        assertEquals(PreTokenizerPatterns.GPT2, PreTokenizerCanonicalizer.canonicalize(tj).regexSequence());
        Resolution r = resolver.explain(tj, encodes(ChkhshGoldenTest.GPT2_IDS));
        assertEquals("gpt-2", r.name());
        assertEquals(Basis.HASH_CONFIRMED, r.basis());
        assertEquals("mpt", r.candidates().get(0));
    }

    @Test
    public void digitsMapToTheirRegexStep() {
        List<Object> list = new ArrayList<>();
        for (String regex : DEEPSEEK_CODER_REGEXES) list.add(split(regex));
        list.add(obj("type", "Digits", "individual_digits", true));
        list.add(byteLevel(false));
        HfTokenizerJson tj = bpe(obj("type", "Sequence", "pretokenizers", list), false);
        assertEquals(PreTokenizerPatterns.DEEPSEEK_CODER, PreTokenizerCanonicalizer.canonicalize(tj).regexSequence());
        Resolution r = resolver.explain(tj, UNKNOWN);
        assertEquals("deepseek-coder", r.name());
        assertEquals(Basis.STRUCTURAL, r.basis());
    }

    // ---------------------------------------------------------------------------------------------
    // Ambiguity and conflicting evidence
    // ---------------------------------------------------------------------------------------------

    @Test
    public void fingerprintOutsideTheCandidatesYieldsToTheSingleStructuralMatch() {
        List<Object> list = new ArrayList<>();
        for (String regex : DEEPSEEK_CODER_REGEXES) list.add(split(regex));
        list.add(obj("type", "Digits", "individual_digits", true));
        list.add(byteLevel(false));
        Resolution r = resolver.explain(bpe(obj("type", "Sequence", "pretokenizers", list), false),
                encodes(ChkhshGoldenTest.GPT2_IDS));
        assertEquals("deepseek-coder", r.name());
        assertEquals(Basis.STRUCTURAL, r.basis());
        assertEquals(Collections.singletonList("gpt-2"), r.hashNames());
    }

    @Test
    public void sharedFingerprintPrefersTheConvertersChoiceAmongCandidates() {
        Resolution r = resolver.explain(bpe(byteLevel(true), false), encodes(ChkhshGoldenTest.MPT_IDS));
        assertEquals(Arrays.asList("mpt", "olmo"), r.hashNames());
        assertEquals("olmo", r.name());
        assertEquals(Basis.HASH_CONFIRMED, r.basis());
    }

    @Test
    public void robertaPostProcessorSelectsTheNamesThatAddSep() {
        HfTokenizerJson tj = bpe(byteLevel(true), false, null,
                obj("type", "RobertaProcessing", "sep", arr("</s>", 2L), "cls", arr("<s>", 0L),
                        "trim_offsets", true, "add_prefix_space", false));
        Resolution r = resolver.explain(tj, UNKNOWN);
        assertEquals(Arrays.asList("jina-v1-en", "jina-v2-code", "roberta-bpe"), r.flagMatches());
        assertEquals("jina-v1-en", r.name());
        assertTrue(PreTokenizerRegistry.forName(r.name()).addSep());
    }

    @Test
    public void withoutStructuralCandidatesTheFingerprintDecides() {
        Map<String, Object> model = obj("type", "WordPiece", "vocab", obj(), "unk_token", "[UNK]");
        HfTokenizerJson bert = HfTokenizerJson.of(obj("pre_tokenizer", obj("type", "BertPreTokenizer"),
                "model", model), "tokenizer.json");
        Resolution r = resolver.explain(bert, encodes(ChkhshGoldenTest.BERT_BGE_IDS));
        assertEquals("jina-v2-en", r.name());
        assertEquals(Basis.HASH_ONLY, r.basis());
        assertTrue(r.candidates().isEmpty());

        assertNull(resolver.resolve(bert, UNKNOWN));
        assertEquals(Basis.NONE, resolver.explain(bert, UNKNOWN).basis());
    }

    @Test
    public void tokenizerWithoutTokenizerJsonUsesOnlyTheFingerprint() {
        assertEquals("qwen2", resolver.resolve(null, encodes(ChkhshGoldenTest.QWEN2_IDS)));
        assertNull(resolver.resolve(null, UNKNOWN));
    }

    @Test
    public void knownFingerprintDecidesWhenThePreTokenizerCannotBeCanonicalized() {
        HfTokenizerJson tj = bpe(obj("type", "FixedLength", "length", 5L), false);
        Resolution r = resolver.explain(tj, encodes(ChkhshGoldenTest.GPT2_IDS));
        assertEquals("gpt-2", r.name());
        assertEquals(Basis.HASH_ONLY, r.basis());
        assertTrue(r.candidates().isEmpty());
        assertTrue(r.canonicalizationError(), r.canonicalizationError().contains("'FixedLength' is not supported"));

        GgufFormatException e = assertThrows(GgufFormatException.class, () -> resolver.resolve(tj, UNKNOWN));
        assertTrue(e.getMessage(), e.getMessage().contains("'FixedLength' is not supported"));
    }

    @Test
    public void encoderReturningNullIsAnError() {
        assertThrows(IllegalStateException.class, () -> resolver.resolve(null, text -> null));
    }

    // ---------------------------------------------------------------------------------------------
    // Element forms
    // ---------------------------------------------------------------------------------------------

    @Test
    public void nestedSequencesFlattenAndDefaultsApply() {
        PreTokenizerSpec nested = PreTokenizerCanonicalizer.toSpec(
                sequence(sequence(obj("type", "Digits")), obj("type", "ByteLevel")), "pre");
        assertEquals(Arrays.asList(Step.digits(false), Step.byteLevel()), nested.steps());
        assertTrue(nested.addPrefixSpace());
        assertTrue(nested.trimOffsets());
        assertTrue(nested.useRegex());
        assertEquals(Arrays.asList("\\p{N}+", PreTokenizerPatterns.GPT2.get(0)),
                PreTokenizerCanonicalizer.regexSequence(nested));
    }

    @Test
    public void metaspaceFollowsTheTokenizersDeserializer() {
        PreTokenizerSpec legacy = PreTokenizerCanonicalizer.toSpec(obj("type", "Metaspace", "replacement", "▁",
                "add_prefix_space", false, "prepend_scheme", "always", "str_rep", "▁"), "pre");
        assertEquals(Step.metaspace(0x2581, PrependScheme.NEVER, true), legacy.steps().get(0));
        PreTokenizerSpec first = PreTokenizerCanonicalizer.toSpec(obj("type", "Metaspace", "replacement", "▁",
                "prepend_scheme", "first", "split", false), "pre");
        assertEquals(Step.metaspace(0x2581, PrependScheme.FIRST, false), first.steps().get(0));
        assertNull(PreTokenizerCanonicalizer.regexSequence(first));

        CanonicalPreTokenizer c = PreTokenizerCanonicalizer.canonicalize(bpe(obj("type", "Metaspace",
                "replacement", "▁", "prepend_scheme", "first", "split", false), false));
        assertTrue(c.escapeWhitespaces());
        assertFalse(c.byteEncode());
    }

    @Test
    public void spaceEscapingNormalizerSetsTheEscapeFlag() {
        Map<String, Object> normalizer = obj("type", "Sequence", "normalizers", arr(
                obj("type", "Prepend", "prepend", "▁"),
                obj("type", "Replace", "pattern", obj("String", " "), "content", "▁")));
        CanonicalPreTokenizer c = PreTokenizerCanonicalizer.canonicalize(
                bpe(split("[^\\n]+|[\\n]+"), false, normalizer, null));
        assertTrue(c.escapeWhitespaces());
        assertEquals(PreTokenizerPatterns.NEWLINE_RUNS, c.regexSequence());
    }

    @Test
    public void otherElementsKeepTheirBehaviour() {
        PreTokenizerSpec spec = PreTokenizerCanonicalizer.toSpec(sequence(
                obj("type", "Split", "pattern", obj("String", "a.b"), "behavior", "Removed", "invert", true),
                obj("type", "Punctuation"),
                obj("type", "Punctuation", "behavior", "MergedWithNext"),
                obj("type", "Whitespace"), obj("type", "WhitespaceSplit"), obj("type", "BertPreTokenizer"),
                obj("type", "UnicodeScripts"), obj("type", "CharDelimiterSplit", "delimiter", "|")), "pre");
        assertEquals(Arrays.asList(Step.literal("a.b", Behaviour.REMOVED, true), Step.punctuation(Behaviour.ISOLATED),
                Step.punctuation(Behaviour.MERGED_WITH_NEXT), Step.whitespace(), Step.whitespaceSplit(), Step.bert(),
                Step.unicodeScripts(), Step.charDelimiter('|')), spec.steps());
        assertFalse(spec.byteEncode());
        assertNull(PreTokenizerCanonicalizer.regexSequence(spec));
    }

    @Test
    public void byteLevelBeforeOtherStepsHasNoRegexForm() {
        PreTokenizerSpec spec = PreTokenizerCanonicalizer.toSpec(
                sequence(byteLevel(false), split("\\p{N}")), "pre");
        assertTrue(spec.byteEncode());
        assertNull(PreTokenizerCanonicalizer.regexSequence(spec));
    }

    @Test
    public void missingPreTokenizerIsTheEmptySpec() {
        CanonicalPreTokenizer c = PreTokenizerCanonicalizer.canonicalize(bpe(null, false));
        assertTrue(c.spec().steps().isEmpty());
        assertEquals(Collections.emptyList(), c.regexSequence());
        assertTrue(HfPreTokenizerResolver.structuralCandidates(c).isEmpty());
    }

    @Test
    public void malformedOrUnsupportedElementsAreRejected() {
        assertRejected("'FixedLength' is not supported", obj("type", "FixedLength", "length", 5L));
        assertRejected("unknown member", obj("type", "Digits", "individual_digits", true, "extra", 1L));
        assertRejected("byte-encode twice", sequence(byteLevel(false), byteLevel(false)));
        assertRejected("unknown value 'Split'", obj("type", "Split", "pattern", obj("Regex", "a"),
                "behavior", "Split"));
        assertRejected("exactly one of 'String' or 'Regex'", obj("type", "Split",
                "pattern", obj("Regex", "a", "String", "a"), "behavior", "Isolated"));
        assertRejected("exactly one character", obj("type", "Metaspace", "replacement", "ab"));
        assertRejected("pretokenizers[0]", sequence("not an object"));
    }

    private static void assertRejected(String expected, Map<String, Object> preTokenizer) {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PreTokenizerCanonicalizer.canonicalize(bpe(preTokenizer, false)));
        assertTrue(e.getMessage(), e.getMessage().contains(expected));
    }
}
