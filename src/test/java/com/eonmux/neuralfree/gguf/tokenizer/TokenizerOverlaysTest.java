package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code neuralfree.tokenizer.*} vocabulary overlays on synthetic metadata: each overlay
 * changes exactly its behaviour, and a vocabulary without overlay keys keeps the llama.cpp one.
 * Token types follow {@code LLAMA_TOKEN_TYPE_*}: 1 normal, 2 unknown, 3 control, 4 user-defined,
 * 6 byte.
 */
public class TokenizerOverlaysTest {

    private static final String G = "Ġ";
    private static final String SP = "▁";
    private static final int NORMAL = 1;
    private static final int UNKNOWN = 2;
    private static final int CONTROL = 3;
    private static final int USER_DEFINED = 4;
    private static final int BYTE = 6;

    /** A gpt2 vocabulary of {@code tokens} with the given types (NORMAL when shorter) and merges. */
    private static Map<String, Object> bpe(String pre, List<String> tokens, List<Integer> types, List<String> merges) {
        List<Object> typeList = new ArrayList<>();
        for (int i = 0; i < tokens.size(); i++) typeList.add(i < types.size() ? types.get(i) : NORMAL);
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "gpt2");
        md.put(TokenizerVocab.KEY_PRE, pre);
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, typeList);
        md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>(merges));
        return md;
    }

    private static Map<String, Object> bpe(List<String> tokens, List<String> merges) {
        return bpe("gpt-2", tokens, Collections.<Integer>emptyList(), merges);
    }

    /** An SPM vocabulary: unk, BOS, EOS, a control token, then normal pieces. */
    private static Map<String, Object> spm() {
        List<Object> tokens = new ArrayList<Object>(Arrays.asList("<unk>", "<s>", "</s>", "<c>", SP, "a", SP + "a"));
        List<Object> types = new ArrayList<Object>(
                Arrays.asList(UNKNOWN, CONTROL, CONTROL, CONTROL, NORMAL, NORMAL, NORMAL));
        List<Object> scores = new ArrayList<Object>(Arrays.asList(0f, 0f, 0f, 0f, -1f, -1f, -2f));
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "llama");
        md.put(TokenizerVocab.KEY_TOKENS, tokens);
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, types);
        md.put(TokenizerVocab.KEY_SCORES, scores);
        return md;
    }

    /** One attribute mask per token: {@code mask} at {@code index}, 0 elsewhere. */
    private static List<Object> attrs(int count, int index, int mask) {
        List<Object> out = new ArrayList<>();
        for (int i = 0; i < count; i++) out.add(i == index ? mask : 0);
        return out;
    }

    private static TokenizerVocab vocab(Map<String, Object> md) {
        return TokenizerVocab.fromMetadata(md);
    }

    private static int[] encode(Map<String, Object> md, String text, boolean addSpecial) {
        TokenizerVocab v = vocab(md);
        return new BpeTokenizer(v, SpecialTokens.of(v)).encode(text, addSpecial);
    }

    private static int[] ids(TokenizerVocab v, String... texts) {
        int[] out = new int[texts.length];
        for (int i = 0; i < texts.length; i++) out[i] = v.tokenId(texts[i]);
        return out;
    }

    // ---------------------------------------------------------------------------------------------
    // No overlay keys
    // ---------------------------------------------------------------------------------------------

    @Test
    public void vocabularyWithoutOverlayKeysKeepsLlamaCppBehaviour() {
        TokenizerVocab v = vocab(bpe(Arrays.asList("a", "b"), Collections.<String>emptyList()));
        assertTrue(v.normalizer().isIdentity());
        assertNull(v.decoder());
        assertEquals(PreTokenizerSpec.PrependScheme.ALWAYS, v.prependScheme());
        assertNull(v.prefixIds());
        assertNull(v.suffixIds());
        assertEquals(0, v.extraEogIds().length);
        assertNull(v.cleanUpTokenizationSpaces());
        assertTrue(v.cleanSpaces());
        assertFalse(v.ignoreMerges());
        assertFalse(v.bpeAddPrefixSpace());
        assertFalse(v.bpeModelOverlay());
        assertEquals("", v.bpeContinuingSubwordPrefix());
        assertEquals("", v.bpeEndOfWordSuffix());
        assertSame(PreTokenizerRegistry.forName("gpt-2"), v.preTokenizer());
        assertSame(v.preTokenizer(), v.namedPreTokenizer());
    }

    @Test
    public void leadingAndTrailingIdsDefaultToBosAndEos() {
        Map<String, Object> md = bpe(Arrays.asList("<s>", "</s>", "a"), Collections.<String>emptyList());
        md.put("tokenizer.ggml.bos_token_id", 0);
        md.put("tokenizer.ggml.eos_token_id", 1);
        md.put(TokenizerVocab.KEY_ADD_BOS, true);
        md.put(TokenizerVocab.KEY_ADD_EOS, true);
        SpecialTokens s = SpecialTokens.of(vocab(md));
        assertArrayEquals(new int[]{0}, s.leadingIds());
        assertArrayEquals(new int[]{1}, s.trailingIds());
        assertArrayEquals(new int[]{0, 2, 1}, encode(md, "a", true));
    }

    // ---------------------------------------------------------------------------------------------
    // Token attributes and added-token matching
    // ---------------------------------------------------------------------------------------------

    @Test
    public void tokenAttrsAreOredIntoAttributes() {
        List<String> tokens = Arrays.asList("<x>", "a", "b");
        Map<String, Object> md = bpe("gpt-2", tokens, Arrays.asList(USER_DEFINED), Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(3, 0, TokenizerVocab.ATTR_LSTRIP));
        TokenizerVocab v = vocab(md);
        assertEquals(TokenizerVocab.ATTR_USER_DEFINED | TokenizerVocab.ATTR_LSTRIP, v.attr(0));
        List<SpecialTokens.Fragment> f = SpecialTokens.of(v).partition("a <x>b", true);
        assertEquals(3, f.size());
        assertEquals("a", f.get(0).text());
        assertEquals(0, f.get(1).token());
        assertEquals("b", f.get(2).text());
    }

    @Test
    public void tokenAttrsAreValidated() {
        Map<String, Object> md = bpe(Arrays.asList("a", "b"), Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(1, 0, TokenizerVocab.ATTR_LSTRIP));
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(2, 1, 1 << 12));
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, new ArrayList<Object>(Arrays.asList(0, 1.5f)));
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
    }

    @Test
    public void singleWordTokensMatchOnlyBetweenNonWordCharacters() {
        List<String> tokens = Arrays.asList("ab", "x");
        Map<String, Object> md = bpe("gpt-2", tokens, Arrays.asList(USER_DEFINED), Collections.<String>emptyList());
        List<SpecialTokens.Fragment> plain = SpecialTokens.of(vocab(md)).partition("xab ab", true);
        assertEquals(4, plain.size());
        assertEquals("x", plain.get(0).text());
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(2, 0, TokenizerVocab.ATTR_SINGLE_WORD));
        List<SpecialTokens.Fragment> f = SpecialTokens.of(vocab(md)).partition("xab ab", true);
        assertEquals(2, f.size());
        assertEquals("xab ", f.get(0).text());
        assertEquals(0, f.get(1).token());
    }

    @Test
    public void singleWordSeesTheNeighboursInTheWholeSentence() {
        List<String> tokens = Arrays.asList("xyz", "ab");
        Map<String, Object> md = bpe("gpt-2", tokens, Arrays.asList(USER_DEFINED, USER_DEFINED),
                Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(2, 1, TokenizerVocab.ATTR_SINGLE_WORD));
        SpecialTokens s = SpecialTokens.of(vocab(md));
        List<SpecialTokens.Fragment> glued = s.partition("xyzab", true);
        assertEquals(2, glued.size());
        assertEquals(0, glued.get(0).token());
        assertEquals("ab", glued.get(1).text());
        List<SpecialTokens.Fragment> spaced = s.partition("xyz ab", true);
        assertEquals(3, spaced.size());
        assertEquals(0, spaced.get(0).token());
        assertEquals(" ", spaced.get(1).text());
        assertEquals(1, spaced.get(2).token());
    }

    @Test
    public void normalizedTokensMatchTheNormalizedText() {
        List<String> tokens = Arrays.asList("hey", "a");
        Map<String, Object> md = bpe("gpt-2", tokens, Arrays.asList(USER_DEFINED), Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, Normalizer.lowercase().toCanonicalJson());
        List<SpecialTokens.Fragment> raw = SpecialTokens.of(vocab(md)).partition("HEY", true, true);
        assertEquals(1, raw.size());
        assertEquals("hey", raw.get(0).text());
        md.put(GgufKeys.NF_TOKENIZER_TOKEN_ATTRS, attrs(2, 0, TokenizerVocab.ATTR_NORMALIZED));
        SpecialTokens s = SpecialTokens.of(vocab(md));
        List<SpecialTokens.Fragment> normalized = s.partition("HEY", true, true);
        assertEquals(1, normalized.size());
        assertEquals(0, normalized.get(0).token());
        List<SpecialTokens.Fragment> unnormalized = s.partition("HEY", true, false);
        assertEquals(1, unnormalized.size());
        assertEquals("HEY", unnormalized.get(0).text());
    }

    // ---------------------------------------------------------------------------------------------
    // Normalizer and pre-tokenizer
    // ---------------------------------------------------------------------------------------------

    private static Map<String, Object> helloVocab() {
        return bpe(Arrays.asList("h", "e", "l", "o", "he", "ll", "hell", "hello"),
                Arrays.asList("h e", "l l", "he ll", "hell o"));
    }

    @Test
    public void normalizerAppliesToEachBpeFragment() {
        Map<String, Object> md = helloVocab();
        assertArrayEquals(new int[0], encode(md, "HELLO", false));
        md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, Normalizer.lowercase().toCanonicalJson());
        TokenizerVocab v = vocab(md);
        assertArrayEquals(ids(v, "hello"), encode(md, "HELLO", false));
    }

    @Test
    public void normalizerIsIgnoredOutsideBpe() {
        Map<String, Object> md = spm();
        md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, Normalizer.lowercase().toCanonicalJson());
        md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, true);
        TokenizerVocab v = vocab(md);
        assertTrue(v.normalizer().isIdentity());
        assertFalse(v.ignoreMerges());
    }

    @Test
    public void malformedOverlayJsonFailsNamingTheKey() {
        Map<String, Object> md = helloVocab();
        md.put(GgufKeys.NF_TOKENIZER_NORMALIZER, "{\"type\":\"NoSuchNormalizer\"}");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> vocab(md));
        assertTrue(e.getMessage(), e.getMessage().contains(GgufKeys.NF_TOKENIZER_NORMALIZER));
        Map<String, Object> ctc = helloVocab();
        ctc.put(GgufKeys.NF_TOKENIZER_DECODER, "{\"type\":\"CTC\"}");
        IllegalArgumentException c = assertThrows(IllegalArgumentException.class, () -> vocab(ctc));
        assertTrue(c.getMessage(), c.getMessage().contains("CTC"));
    }

    @Test
    public void preTokenizerSpecReplacesTheNamedSplitButKeepsItsFlags() {
        Map<String, Object> md = bpe("llama-bpe", Arrays.asList("ab", "cd"), Collections.<Integer>emptyList(),
                Collections.<String>emptyList());
        PreTokenizerSpec spec = new PreTokenizerSpec(Arrays.asList(PreTokenizerSpec.Step.whitespaceSplit()),
                false, false, false, false);
        md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, spec.toCanonicalJson());
        TokenizerVocab v = vocab(md);
        assertEquals(PreTokenizerRegistry.CUSTOM_NAME, v.preTokenizer().name());
        assertEquals("llama-bpe", v.namedPreTokenizer().name());
        assertTrue(v.ignoreMerges());
        BpeTokenizer t = new BpeTokenizer(v, SpecialTokens.of(v));
        assertEquals(Arrays.asList("ab", "cd"), t.words("ab cd"));
        assertArrayEquals(ids(v, "ab", "cd"), t.encode("ab cd", false));
    }

    @Test
    public void wordsComeFromASpecThatSplitsAfterByteLevel() {
        Map<String, Object> md = bpe(Arrays.asList("g", "o", G + "s"), Arrays.asList(G + " s"));
        List<PreTokenizerSpec.Step> steps = Arrays.asList(PreTokenizerSpec.Step.byteLevel(),
                PreTokenizerSpec.Step.literal("o", PreTokenizerSpec.Behaviour.ISOLATED, false));
        PreTokenizerSpec spec = new PreTokenizerSpec(steps, true, false, false, false);
        md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, spec.toCanonicalJson());
        TokenizerVocab v = vocab(md);
        BpeTokenizer t = new BpeTokenizer(v, SpecialTokens.of(v));
        assertEquals(Arrays.asList("g", "o", G + "s", "o"), t.words("go so"));
        assertArrayEquals(ids(v, "g", "o", G + "s", "o"), t.encode("go so", false));
    }

    // ---------------------------------------------------------------------------------------------
    // BPE model options
    // ---------------------------------------------------------------------------------------------

    @Test
    public void ignoreMergesOverlayEmitsWholeWordTokens() {
        Map<String, Object> md = bpe(Arrays.asList("a", "b", "c", "bc", "abc"), Arrays.asList("b c"));
        TokenizerVocab plain = vocab(md);
        assertArrayEquals(ids(plain, "a", "bc"), encode(md, "abc", false));
        md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, true);
        TokenizerVocab v = vocab(md);
        assertTrue(v.ignoreMerges());
        assertArrayEquals(ids(v, "abc"), encode(md, "abc", false));
    }

    @Test
    public void ignoreMergesOverlayOverridesTheNameFlag() {
        Map<String, Object> md = bpe("llama-bpe", Arrays.asList("a", "b"), Collections.<Integer>emptyList(),
                Collections.<String>emptyList());
        assertTrue(vocab(md).ignoreMerges());
        md.put(GgufKeys.NF_TOKENIZER_BPE_IGNORE_MERGES, false);
        assertFalse(vocab(md).ignoreMerges());
    }

    @Test
    public void addPrefixSpacePrefixesEachFragment() {
        Map<String, Object> md = bpe(Arrays.asList(G, "h", "i", G + "h", G + "hi", "hi"),
                Arrays.asList(G + " h", G + "h i", "h i"));
        TokenizerVocab plain = vocab(md);
        assertArrayEquals(ids(plain, "hi"), encode(md, "hi", false));
        md.put(GgufKeys.NF_TOKENIZER_BPE_ADD_PREFIX_SPACE, true);
        TokenizerVocab v = vocab(md);
        assertArrayEquals(ids(v, G + "hi"), encode(md, "hi", false));
        assertArrayEquals(ids(v, G + "hi"), encode(md, " hi", false));
    }

    @Test
    public void gemma4KeepsItsForcedPreTokenizer() {
        Map<String, Object> md = bpe(Arrays.asList("a", "b"), Collections.<String>emptyList());
        md.put(TokenizerVocab.KEY_MODEL, "gemma4");
        PreTokenizerSpec spec = new PreTokenizerSpec(Arrays.asList(PreTokenizerSpec.Step.whitespaceSplit()),
                false, false, false, false);
        md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, spec.toCanonicalJson());
        TokenizerVocab v = vocab(md);
        assertEquals("gemma4", v.preTokenizer().name());
        assertSame(v.namedPreTokenizer(), v.preTokenizer());
    }

    /** A vocabulary split on whitespace without byte encoding, for the Hugging Face BPE model options. */
    private static Map<String, Object> modelVocab(List<String> tokens, List<Integer> types, List<String> merges) {
        Map<String, Object> md = bpe("gpt-2", tokens, types, merges);
        PreTokenizerSpec spec = new PreTokenizerSpec(Arrays.asList(PreTokenizerSpec.Step.whitespaceSplit()),
                false, false, false, false);
        md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, spec.toCanonicalJson());
        return md;
    }

    @Test
    public void unknownCharactersBecomeTheUnknownToken() {
        Map<String, Object> md = modelVocab(Arrays.asList("<unk>", "a", "b"), Arrays.asList(UNKNOWN),
                Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, false);
        TokenizerVocab v = vocab(md);
        assertTrue(v.bpeModelOverlay());
        assertArrayEquals(ids(v, "a", "b"), encode(md, "axyb", false));
        md.put("tokenizer.ggml.unknown_token_id", 0);
        assertArrayEquals(new int[]{1, 0, 0, 2}, encode(md, "axyb", false));
        md.put(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, true);
        assertArrayEquals(new int[]{1, 0, 2}, encode(md, "axyb", false));
        assertArrayEquals(new int[]{0, 1}, encode(md, "xya", false));
    }

    @Test
    public void byteFallbackKeyDecidesWhetherByteTokensAreUsed() {
        Map<String, Object> md = modelVocab(Arrays.asList("<unk>", "a", "b", "<0x78>"),
                Arrays.asList(UNKNOWN, NORMAL, NORMAL, BYTE), Collections.<String>emptyList());
        TokenizerVocab legacy = vocab(md);
        assertNull(legacy.bpeByteFallback());
        assertFalse(legacy.bpeModelOverlay());
        assertArrayEquals(new int[]{1, 3, 2}, encode(md, "axyb", false));

        md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, false);
        TokenizerVocab off = vocab(md);
        assertEquals(Boolean.FALSE, off.bpeByteFallback());
        assertFalse(off.bpeModelOverlay());
        assertArrayEquals(new int[]{1, 2}, encode(md, "axyb", false));
        md.put("tokenizer.ggml.unknown_token_id", 0);
        md.put(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE, false);
        assertArrayEquals(new int[]{1, 0, 0, 2}, encode(md, "axyb", false));

        md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, true);
        assertArrayEquals(new int[]{1, 3, 0, 2}, encode(md, "axyb", false));
        md.remove(GgufKeys.NF_TOKENIZER_BPE_UNK_FUSE);
        md.remove("tokenizer.ggml.unknown_token_id");
        TokenizerVocab on = vocab(md);
        assertEquals(Boolean.TRUE, on.bpeByteFallback());
        assertTrue(on.bpeModelOverlay());
        assertArrayEquals(new int[]{1, 3, 2}, encode(md, "axyb", false));
    }

    @Test
    public void byteFallbackKeyIsIgnoredOutsideBpe() {
        Map<String, Object> md = spm();
        md.put(GgufKeys.NF_TOKENIZER_BPE_BYTE_FALLBACK, true);
        assertNull(vocab(md).bpeByteFallback());
    }

    @Test
    public void endOfWordSuffixMarksTheLastSymbol() {
        Map<String, Object> md = modelVocab(Arrays.asList("l", "o", "w</w>", "lo", "low</w>", "o</w>"),
                Collections.<Integer>emptyList(), Arrays.asList("l o", "lo w</w>"));
        md.put(GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX, "</w>");
        TokenizerVocab v = vocab(md);
        assertEquals("</w>", v.bpeEndOfWordSuffix());
        assertArrayEquals(ids(v, "low</w>", "o", "w</w>", "o</w>"), encode(md, "low ow o", false));
    }

    @Test
    public void continuingSubwordPrefixIsDroppedWhenMerging() {
        Map<String, Object> md = modelVocab(Arrays.asList("a", "##b", "ab", "##c", "abc"),
                Collections.<Integer>emptyList(), Arrays.asList("a ##b", "ab ##c"));
        md.put(GgufKeys.NF_TOKENIZER_BPE_CONTINUING_SUBWORD_PREFIX, "##");
        TokenizerVocab v = vocab(md);
        assertArrayEquals(ids(v, "abc"), encode(md, "abc", false));
        assertArrayEquals(ids(v, "a", "##c"), encode(md, "ac", false));
    }

    // ---------------------------------------------------------------------------------------------
    // Post-processor, prepend scheme and end of generation
    // ---------------------------------------------------------------------------------------------

    @Test
    public void prefixAndSuffixIdsReplaceBosAndEos() {
        Map<String, Object> md = bpe("gpt-2", Arrays.asList("<s>", "</s>", "<lang>", "a"),
                Arrays.asList(CONTROL, CONTROL, CONTROL), Collections.<String>emptyList());
        md.put("tokenizer.ggml.bos_token_id", 0);
        md.put(TokenizerVocab.KEY_ADD_BOS, true);
        md.put(GgufKeys.NF_TOKENIZER_PREFIX_IDS, new ArrayList<Object>(Arrays.asList(0, 2)));
        md.put(GgufKeys.NF_TOKENIZER_SUFFIX_IDS, new ArrayList<Object>(Arrays.asList(1)));
        assertArrayEquals(new int[]{0, 2, 3, 1}, encode(md, "a", true));
        assertArrayEquals(new int[]{3}, encode(md, "a", false));
        md.put(GgufKeys.NF_TOKENIZER_SUFFIX_IDS, new ArrayList<Object>(Arrays.asList(4)));
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
    }

    @Test
    public void prependSchemeControlsTheSpmSpacePrefix() {
        Map<String, Object> md = spm();
        TokenizerVocab always = vocab(md);
        assertArrayEquals(ids(always, SP + "a", "<c>", SP + "a"), encodeSpm(always, "a<c>a"));
        md.put(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, "first");
        TokenizerVocab first = vocab(md);
        assertEquals(PreTokenizerSpec.PrependScheme.FIRST, first.prependScheme());
        assertArrayEquals(ids(first, SP + "a", "<c>", "a"), encodeSpm(first, "a<c>a"));
        assertArrayEquals(ids(first, "<c>", "a"), encodeSpm(first, "<c>a"));
        assertArrayEquals(ids(first, SP + "a"), encodeSpm(first, " a"));
        assertArrayEquals(ids(always, SP, SP + "a"), encodeSpm(always, " a"));
        md.put(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, "never");
        TokenizerVocab never = vocab(md);
        assertArrayEquals(ids(never, "a", "<c>", "a"), encodeSpm(never, "a<c>a"));
        assertEquals("a", new SpmTokenizer(always, SpecialTokens.of(always)).decode(ids(always, SP + "a")));
        assertEquals(" a", new SpmTokenizer(never, SpecialTokens.of(never)).decode(ids(never, SP + "a")));
        md.put(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, "First");
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
    }

    private static int[] encodeSpm(TokenizerVocab v, String text) {
        return new SpmTokenizer(v, SpecialTokens.of(v)).encode(text, false);
    }

    @Test
    public void prependSchemeIsIgnoredOutsideSpm() {
        Map<String, Object> md = bpe(Arrays.asList("a", "b"), Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_PREPEND_SCHEME, "never");
        assertEquals(PreTokenizerSpec.PrependScheme.ALWAYS, vocab(md).prependScheme());
    }

    @Test
    public void extraEogIdsEndGeneration() {
        Map<String, Object> md = bpe(Arrays.asList("a", "b", "<stop>"), Collections.<String>emptyList());
        assertFalse(SpecialTokens.of(vocab(md)).isEog(2));
        md.put(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, new ArrayList<Object>(Arrays.asList(2)));
        TokenizerVocab v = vocab(md);
        assertArrayEquals(new int[]{2}, v.extraEogIds());
        SpecialTokens s = SpecialTokens.of(v);
        assertTrue(s.isEog(2));
        assertTrue(s.isHidden(2));
        md.put(GgufKeys.NF_TOKENIZER_EXTRA_EOG_IDS, new ArrayList<Object>(Arrays.asList(-1)));
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
    }

    // ---------------------------------------------------------------------------------------------
    // Detokenization: decoder chain and clean-up
    // ---------------------------------------------------------------------------------------------

    @Test
    public void decoderOverlayReplacesPieceConcatenation() {
        Map<String, Object> md = modelVocab(Arrays.asList("hi</w>", "yo</w>", "h", "i</w>", "y", "o</w>"),
                Collections.<Integer>emptyList(), Arrays.asList("h i</w>", "y o</w>"));
        md.put(GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX, "</w>");
        TokenizerVocab plain = vocab(md);
        assertEquals("hi</w>yo</w>", new GGUFTokenizer(plain).decode(new int[]{0, 1}));
        md.put(GgufKeys.NF_TOKENIZER_DECODER, TokenDecoder.bpe("</w>").toCanonicalJson());
        TokenizerVocab v = vocab(md);
        assertEquals(TokenDecoder.bpe("</w>"), v.decoder());
        GGUFTokenizer tokenizer = new GGUFTokenizer(v);
        assertEquals("hi yo", tokenizer.decode(new int[]{0, 1}));
        assertEquals("hi yo", tokenizer.decode(tokenizer.encode("hi yo", false)));
    }

    @Test
    public void streamingWithADecoderOverlayMatchesSequenceDecoding() {
        Map<String, Object> md = modelVocab(Arrays.asList("hi</w>", "yo</w>", "h", "i</w>", "y", "o</w>"),
                Collections.<Integer>emptyList(), Arrays.asList("h i</w>", "y o</w>"));
        md.put(GgufKeys.NF_TOKENIZER_BPE_END_OF_WORD_SUFFIX, "</w>");
        md.put(GgufKeys.NF_TOKENIZER_DECODER, TokenDecoder.bpe("</w>").toCanonicalJson());
        GGUFTokenizer tokenizer = new GGUFTokenizer(vocab(md));
        assertEquals("hi", tokenizer.decode(0));
        GGUFTokenizer.StreamingDecoder sd = tokenizer.createStreamingDecoder();
        int[] ids = {0, 1, 0};
        StringBuilder out = new StringBuilder();
        for (int id : ids) out.append(sd.decodeToken(id));
        out.append(sd.flush());
        assertEquals("hi yo hi", tokenizer.decode(ids));
        assertEquals(tokenizer.decode(ids), out.toString());
        sd.reset();
        assertEquals("yo", sd.decodeToken(1));
    }

    @Test
    public void streamingHoldsAnIncompleteCharacterUntilItCompletes() {
        Map<String, Object> md = bpe(Arrays.asList("a", "\u00C3", "\u00A9"), Collections.<String>emptyList());
        md.put(GgufKeys.NF_TOKENIZER_DECODER, TokenDecoder.byteLevel(false, false, false).toCanonicalJson());
        GGUFTokenizer.StreamingDecoder sd = new GGUFTokenizer(vocab(md)).createStreamingDecoder();
        assertEquals("a", sd.decodeToken(0));
        assertEquals("", sd.decodeToken(1));
        assertEquals("\u00E9", sd.decodeToken(2));
        assertEquals("", sd.decodeToken(1));
        assertEquals("\uFFFD", sd.flush());
        assertEquals("", sd.flush());
    }

    /** Pieces that render {@code "I do n't ."} with GPT-2 byte decoding. */
    private static Map<String, Object> contractionVocab() {
        return bpe(Arrays.asList("I", G + "do", G + "n't", G + "."), Collections.<String>emptyList());
    }

    @Test
    public void cleanUpTokenizationSpacesFollowsTransformers() {
        int[] ids = {0, 1, 2, 3};
        assertEquals("I do n't.", new GGUFTokenizer(vocab(contractionVocab())).decode(ids));
        Map<String, Object> md = contractionVocab();
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, true);
        TokenizerVocab v = vocab(md);
        assertTrue(v.cleanUpTokenizationSpaces());
        assertFalse(v.cleanSpaces());
        assertEquals("I don't.", new GGUFTokenizer(v).decode(ids));
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, false);
        assertEquals("I do n't .", new GGUFTokenizer(vocab(md)).decode(ids));
    }

    @Test
    public void streamingAppliesTheCleanUpAcrossTokens() {
        Map<String, Object> md = contractionVocab();
        md.put(GgufKeys.NF_TOKENIZER_CLEAN_SPACES, true);
        GGUFTokenizer.StreamingDecoder sd = new GGUFTokenizer(vocab(md)).createStreamingDecoder();
        assertEquals("I", sd.decodeToken(0));
        assertEquals(" do", sd.decodeToken(1));
        assertEquals("n't", sd.decodeToken(2));
        assertEquals(".", sd.decodeToken(3));
        assertEquals("", sd.flush());
    }

    @Test
    public void cleanUpReplacementsAreTransformersOnes() {
        String spaced = "a . b ? c ! d , it 's I 'm we 've they 're ca n't x ' y";
        assertEquals("a. b? c! d, it's I'm we've they're can't x'y",
                TokenizerAlgorithm.cleanUpTokenizationSpaces(spaced));
        assertEquals(" do not", TokenizerAlgorithm.cleanUpTokenizationSpaces(" do not"));
    }

    // ---------------------------------------------------------------------------------------------
    // Charsmap forms
    // ---------------------------------------------------------------------------------------------

    @Test
    public void precompiledCharsmapAcceptsBytesAndNumberLists() {
        byte[] blob = {4, 0, 0, 0, 1, 2, 3, 4, 0};
        List<Object> numbers = new ArrayList<>();
        for (byte b : blob) numbers.add((int) b);
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, "t5");
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(Arrays.asList("<pad>", "</s>", "<unk>", SP + "a")));
        md.put(TokenizerVocab.KEY_PRECOMPILED_CHARSMAP, numbers);
        byte[] fromList = vocab(md).precompiledCharsmap();
        md.put(TokenizerVocab.KEY_PRECOMPILED_CHARSMAP, blob.clone());
        byte[] fromBytes = vocab(md).precompiledCharsmap();
        assertArrayEquals(blob, fromList);
        assertArrayEquals(blob, fromBytes);
        md.put(TokenizerVocab.KEY_PRECOMPILED_CHARSMAP, new byte[]{9, 0, 0, 0, 1});
        assertThrows(IllegalArgumentException.class, () -> vocab(md));
    }
}
