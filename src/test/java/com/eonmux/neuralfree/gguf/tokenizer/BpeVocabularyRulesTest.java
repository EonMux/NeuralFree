package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Parity tests for the vocabulary rules of llama.cpp's {@code llama_vocab::impl} that depend on the pre-tokenizer
 * type or on vocabulary keys: the {@code soprano} normalization, unknown-token fallback, raw pieces and EOS
 * end-of-generation (src/llama-vocab.cpp:662-680, 800-801, 3325-3327, 4142-4144), the {@code nemotron-3.5-embed}
 * ZWNJ removal (656-661), the {@code tokenizer.ggml.clean_spaces} override (2804) and the
 * {@code tokenizer.ggml.metaspace.prepend_scheme} key (3022-3038, 3902-3907, 4243-4252).
 */
public class BpeVocabularyRulesTest {

    private static final String SP = "▁";

    private static Map<String, Object> metadata(String model, String pre, List<String> tokens, List<Integer> types,
                                                List<String> merges) {
        Map<String, Object> md = new HashMap<>();
        md.put(TokenizerVocab.KEY_MODEL, model);
        if (pre != null) md.put(TokenizerVocab.KEY_PRE, pre);
        md.put(TokenizerVocab.KEY_TOKENS, new ArrayList<Object>(tokens));
        md.put(TokenizerVocab.KEY_TOKEN_TYPE, new ArrayList<Object>(types));
        if (merges != null) md.put(TokenizerVocab.KEY_MERGES, new ArrayList<Object>(merges));
        return md;
    }

    private static List<Integer> normalTypes(List<String> tokens) {
        List<Integer> types = new ArrayList<>();
        for (String t : tokens) types.add(t.startsWith("[") || t.startsWith("<") ? 3 : 1);
        return types;
    }

    private static BpeTokenizer bpe(TokenizerVocab v) {
        return new BpeTokenizer(v, SpecialTokens.of(v));
    }

    private static int[] ids(TokenizerVocab v, String... texts) {
        int[] out = new int[texts.length];
        for (int i = 0; i < texts.length; i++) {
            out[i] = v.tokenId(texts[i]);
            assertTrue(texts[i], out[i] >= 0);
        }
        return out;
    }

    // ------------------------------------------------------------------ soprano

    private static Map<String, Object> sopranoMetadata(boolean withUnk) {
        List<String> tokens = Arrays.asList("[UNK]", "[STOP]", "h", "e", "l", "o", "he", "hel", "hell", "hello", " ",
                "1", "2", "i", "̇", "Ġ");
        List<Integer> types = new ArrayList<>(normalTypes(tokens));
        types.set(0, 2);
        Map<String, Object> md = metadata("gpt2", "soprano", tokens, types,
                Arrays.asList("h e", "he l", "hel l", "hell o"));
        if (withUnk) md.put(GgufKeys.TOKENIZER_UNK_ID, 0);
        md.put(GgufKeys.TOKENIZER_EOS_ID, 1);
        return md;
    }

    @Test
    public void sopranoLowercasesCollapsesWhitespaceAndMapsUnknownSymbolsToUnk() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(sopranoMetadata(true));
        int[] expected = ids(v, "hello", " ", "1", "2");
        int[] got = bpe(v).encode("HeLLo \t\n 12x", false);
        assertArrayEquals(new int[]{expected[0], expected[1], expected[2], expected[3], 0}, got);
    }

    @Test
    public void sopranoLowercasesCapitalIWithDotAboveToTwoCodePoints() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(sopranoMetadata(true));
        assertArrayEquals(ids(v, "i", "̇"), bpe(v).encode("İ", false));
        assertEquals("i̇ a", PreTokenizerType.SOPRANO.normalizeFragment("İ　 A"));
    }

    @Test
    public void sopranoRendersNormalTokensAsTheirRawText() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(sopranoMetadata(true));
        assertArrayEquals("Ġ".getBytes(StandardCharsets.UTF_8), bpe(v).piece(v.tokenId("Ġ")));
        TokenizerVocab gpt2 = TokenizerVocab.fromMetadata(metadata("gpt2", "gpt-2", Arrays.asList("Ġ"),
                Arrays.asList(1), Arrays.<String>asList()));
        assertArrayEquals(new byte[]{' '}, bpe(gpt2).piece(0));
    }

    @Test
    public void sopranoEosEndsGeneration() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(sopranoMetadata(true));
        assertTrue(SpecialTokens.of(v).isEog(1));
    }

    @Test
    public void sopranoWithoutAnUnknownTokenRefusesAnUnknownSymbol() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(sopranoMetadata(false));
        BpeTokenizer t = bpe(v);
        assertArrayEquals(ids(v, "hello"), t.encode("hello", false));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> t.encode("x", false));
        assertTrue(e.getMessage(), e.getMessage().contains("'soprano'"));
        assertTrue(e.getMessage(), e.getMessage().contains(GgufKeys.TOKENIZER_UNK_ID));
    }

    // ------------------------------------------------------------------ nemotron-3.5-embed

    @Test
    public void nemotronEmbedDropsTheZeroWidthNonJoinerAfterBracketAndAngle() {
        assertEquals("[x <y a‌b", PreTokenizerType.NEMOTRON_3_5_EMBD.normalizeFragment("[‌x <‌y a‌b"));
        assertEquals("[‌x", PreTokenizerType.TEKKEN.normalizeFragment("[‌x"));

        String zwnj = BpeTokenizer.byteEncode("‌");
        String c01 = zwnj.substring(0, 2);
        List<String> tokens = Arrays.asList("[", "x", c01, zwnj);
        List<String> merges = Arrays.asList(zwnj.charAt(0) + " " + zwnj.charAt(1), c01 + " " + zwnj.charAt(2));
        List<Integer> types = Arrays.asList(1, 1, 1, 1);
        TokenizerVocab tekken = TokenizerVocab.fromMetadata(metadata("gpt2", "tekken", tokens, types, merges));
        TokenizerVocab nemotron = TokenizerVocab.fromMetadata(
                metadata("gpt2", "nemotron-3.5-embed", tokens, types, merges));
        assertArrayEquals(ids(tekken, "[", zwnj, "x"), bpe(tekken).encode("[‌x", false));
        assertArrayEquals(ids(nemotron, "[", "x"), bpe(nemotron).encode("[‌x", false));
    }

    // ------------------------------------------------------------------ tokenizer.ggml.clean_spaces

    @Test
    public void theCleanSpacesKeyOverridesTheDefaultOfEveryType() {
        List<String> tokens = Arrays.asList("a");
        Map<String, Object> gpt2 = metadata("gpt2", "gpt-2", tokens, Arrays.asList(1), Arrays.<String>asList());
        assertTrue(TokenizerVocab.fromMetadata(gpt2).cleanSpaces());
        gpt2.put(TokenizerVocab.KEY_CLEAN_SPACES, false);
        assertFalse(TokenizerVocab.fromMetadata(gpt2).cleanSpaces());

        Map<String, Object> spm = metadata("llama", null, Arrays.asList("<unk>", SP + "a"), Arrays.asList(2, 1), null);
        assertFalse(TokenizerVocab.fromMetadata(spm).cleanSpaces());
        spm.put(TokenizerVocab.KEY_CLEAN_SPACES, true);
        assertTrue(TokenizerVocab.fromMetadata(spm).cleanSpaces());
    }

    // ------------------------------------------------------------------ tokenizer.ggml.metaspace.prepend_scheme

    private static Map<String, Object> domynMetadata(String scheme) {
        List<String> tokens = Arrays.asList("<s>", "<|x|>", SP, "h", "e", "l", "o", "w", "r", "d", SP + "h", SP + "he",
                SP + "hel", SP + "hell", SP + "hello", SP + "w", SP + "wo", SP + "wor", SP + "worl", SP + "world");
        List<String> merges = Arrays.asList(SP + " h", SP + "h e", SP + "he l", SP + "hel l", SP + "hell o",
                SP + " w", SP + "w o", SP + "wo r", SP + "wor l", SP + "worl d");
        Map<String, Object> md = metadata("gpt2", "domyn-small", tokens, normalTypes(tokens), merges);
        md.put(GgufKeys.TOKENIZER_BOS_ID, 0);
        md.put(TokenizerVocab.KEY_ADD_BOS, false);
        if (scheme != null) md.put(TokenizerVocab.KEY_METASPACE_PREPEND_SCHEME, scheme);
        return md;
    }

    @Test
    public void firstPrependsTheMetaspaceToTheFirstFragmentOnly() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(domynMetadata("first"));
        assertSame(PreTokenizerSpec.PrependScheme.FIRST, v.metaspacePrependScheme());
        BpeTokenizer t = bpe(v);
        assertArrayEquals(ids(v, SP + "hello", SP + "world"), t.encode("hello world", false));
        assertArrayEquals(ids(v, SP + "hello", SP + "world"), t.encode(" hello world", false));
        int[] afterSpecial = t.encode("<|x|>hello", false);
        assertEquals(v.tokenId("<|x|>"), afterSpecial[0]);
        assertFalse(afterSpecial[1] == v.tokenId(SP + "hello"));

        TokenizerVocab never = TokenizerVocab.fromMetadata(domynMetadata(null));
        assertSame(PreTokenizerSpec.PrependScheme.NEVER, never.metaspacePrependScheme());
        assertFalse(Arrays.equals(ids(never, SP + "hello", SP + "world"), bpe(never).encode("hello world", false)));
    }

    @Test
    public void firstRemovesTheLeadingSpaceOfTheFirstRenderedPiece() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(domynMetadata("first"));
        BpeTokenizer t = bpe(v);
        assertEquals("hello world", t.decode(ids(v, SP + "hello", SP + "world")));
        assertEquals("hello world", t.decode(ids(v, "<s>", SP + "hello", SP + "world")));
        assertEquals("e hello", t.decode(ids(v, "e", SP + "hello")));

        TokenizerVocab never = TokenizerVocab.fromMetadata(domynMetadata("never"));
        assertEquals(" hello world", bpe(never).decode(ids(never, SP + "hello", SP + "world")));
    }

    @Test
    public void anUnknownSchemeIsRefusedForEveryType() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TokenizerVocab.fromMetadata(domynMetadata("sometimes")));
        assertTrue(e.getMessage(), e.getMessage().contains("unknown metaspace prepend_scheme: 'sometimes'"));

        Map<String, Object> spm = metadata("llama", null, Arrays.asList("<unk>", SP + "a"), Arrays.asList(2, 1), null);
        spm.put(TokenizerVocab.KEY_METASPACE_PREPEND_SCHEME, "sometimes");
        assertThrows(IllegalArgumentException.class, () -> TokenizerVocab.fromMetadata(spm));
        spm.put(TokenizerVocab.KEY_METASPACE_PREPEND_SCHEME, "always");
        assertSame(PreTokenizerSpec.PrependScheme.NEVER, TokenizerVocab.fromMetadata(spm).metaspacePrependScheme());
    }

    @Test
    public void alwaysIsRefusedForANamedBpePreTokenizer() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TokenizerVocab.fromMetadata(domynMetadata("always")));
        assertTrue(e.getMessage(), e.getMessage().contains(
                "metaspace prepend_scheme 'always' is not supported for BPE vocab"));
    }

    @Test
    public void aPreTokenizerSpecOverlayCarriesTheSchemeItself() {
        for (String scheme : Arrays.asList("always", "first")) {
            Map<String, Object> md = domynMetadata(scheme);
            md.put(GgufKeys.NF_TOKENIZER_PRE_TOKENIZER, PreTokenizerRegistry.specFor("domyn-small").toCanonicalJson());
            TokenizerVocab v = TokenizerVocab.fromMetadata(md);
            assertSame(scheme, PreTokenizerSpec.PrependScheme.NEVER, v.metaspacePrependScheme());
        }
    }
}
