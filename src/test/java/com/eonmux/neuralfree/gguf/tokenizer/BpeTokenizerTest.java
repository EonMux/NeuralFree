package com.eonmux.neuralfree.gguf.tokenizer;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Parity tests for {@link BpeTokenizer} and {@link Gemma4Tokenizer} against llama.cpp
 * {@code llm_tokenizer_bpe_session} on synthetic vocabularies: regex pre-splitting, GPT-2 byte
 * encoding, rank-ordered merges, {@code ignore_merges}, the Gemma 4 newline rule and
 * {@code <0xXX>} byte fallback.
 */
public class BpeTokenizerTest {

    private static final String SP = "\u2581";
    private static final String G_SPACE = "\u0120";

    private static Map<String, Object> metadata(String model, String pre, List<String> tokens, List<String> merges) {
        List<Object> tok = new ArrayList<>(tokens);
        List<Object> types = new ArrayList<>();
        for (String t : tokens) types.add(t.startsWith("<|") || t.startsWith("<bos") || t.startsWith("<eos") ? 3 : t.startsWith("<0x") ? 6 : 1);
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", model);
        if (pre != null) md.put("tokenizer.ggml.pre", pre);
        md.put("tokenizer.ggml.tokens", tok);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.merges", new ArrayList<Object>(merges));
        return md;
    }

    private static Map<String, Object> gpt2Metadata() {
        List<String> tokens = Arrays.asList("<|endoftext|>", "H", "e", "l", "o", G_SPACE, "w", "r", "d",
                "ll", "He", "Hell", "Hello", G_SPACE + "w", "or", G_SPACE + "wor", G_SPACE + "worl", G_SPACE + "world");
        List<String> merges = Arrays.asList("l l", "H e", "He ll", "Hell o", G_SPACE + " w", "o r",
                G_SPACE + "w or", G_SPACE + "wor l", G_SPACE + "worl d");
        Map<String, Object> md = metadata("gpt2", "gpt-2", tokens, merges);
        md.put("tokenizer.ggml.bos_token_id", 0);
        md.put("tokenizer.ggml.eos_token_id", 0);
        return md;
    }

    private static BpeTokenizer bpe(TokenizerVocab v) {
        return new BpeTokenizer(v, SpecialTokens.of(v));
    }

    @Test
    public void gpt2HelloWorldMergesByRank() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gpt2Metadata());
        int[] ids = bpe(v).encode("Hello world", true);
        assertArrayEquals(new int[]{v.tokenId("Hello"), v.tokenId(G_SPACE + "world")}, ids);
    }

    @Test
    public void preTokenizerSplitsAndByteEncodes() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gpt2Metadata());
        assertEquals(Arrays.asList("Hello", G_SPACE + "world"), bpe(v).words("Hello world"));
    }

    @Test
    public void specialTokensAreSplitOut() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gpt2Metadata());
        int[] ids = bpe(v).encode("Hello<|endoftext|>", false);
        assertArrayEquals(new int[]{v.tokenId("Hello"), 0}, ids);
    }

    @Test
    public void nonTokenSymbolsFallBackToSingleByteTokens() {
        List<String> tokens = Arrays.asList("d", "r");
        Map<String, Object> md = metadata("gpt2", "gpt-2", tokens, Arrays.asList("d r"));
        TokenizerVocab v = TokenizerVocab.fromMetadata(md);
        assertArrayEquals(new int[]{v.tokenId("d"), v.tokenId("r")}, bpe(v).encode("dr", false));
        assertArrayEquals(new int[]{v.tokenId("d")}, bpe(v).encode("dx", false));
    }

    @Test
    public void decodeReversesByteEncoding() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gpt2Metadata());
        BpeTokenizer t = bpe(v);
        assertEquals("Hello world", t.decode(new int[]{v.tokenId("Hello"), v.tokenId(G_SPACE + "world")}));
        assertEquals("", t.decode(new int[]{0}));
    }

    @Test
    public void ignoreMergesEmitsWholeVocabularyWords() {
        List<String> tokens = Arrays.asList("<|begin_of_text|>", "H", "e", "l", "o", "Hello");
        Map<String, Object> md = metadata("gpt2", "llama-bpe", tokens, Arrays.asList("l l"));
        md.put("tokenizer.ggml.bos_token_id", 0);
        TokenizerVocab v = TokenizerVocab.fromMetadata(md);
        assertTrue(v.ignoreMerges());
        assertTrue(v.addBos());
        assertArrayEquals(new int[]{0, v.tokenId("Hello")}, bpe(v).encode("Hello", true));
    }

    private static Map<String, Object> gemma4Metadata() {
        List<String> tokens = new ArrayList<>(Arrays.asList("<pad>", "<eos>", "<bos>", SP, "H", "i", "t", "h",
                "e", "r", "\n", "\n\n", "Hi", SP + "t", "he", SP + "the", "re", SP + "there"));
        tokens.add("<0xC3>");
        tokens.add("<0xA9>");
        List<String> merges = Arrays.asList("H i", SP + " t", "h e", SP + "t he", "r e", SP + "the re");
        Map<String, Object> md = metadata("gemma4", null, tokens, merges);
        md.put("tokenizer.ggml.bos_token_id", 2);
        md.put("tokenizer.ggml.eos_token_id", 1);
        md.put("tokenizer.ggml.padding_token_id", 0);
        md.put("tokenizer.ggml.add_bos_token", false);
        return md;
    }

    @Test
    public void gemma4EscapesSpacesSplitsNewlinesAndForcesBos() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gemma4Metadata());
        assertEquals("gemma4", v.pre());
        assertTrue(v.addBos());
        Gemma4Tokenizer t = new Gemma4Tokenizer(v, SpecialTokens.of(v));
        int[] ids = t.encode("Hi there\n\n", true);
        assertArrayEquals(new int[]{2, v.tokenId("Hi"), v.tokenId(SP + "there"), v.tokenId("\n\n")}, ids);
        assertEquals("Hi there\n\n", t.decode(ids));
    }

    @Test
    public void gemma4FallsBackToHexByteTokens() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(gemma4Metadata());
        Gemma4Tokenizer t = new Gemma4Tokenizer(v, SpecialTokens.of(v));
        int[] ids = t.encode("\u00E9", false);
        assertArrayEquals(new int[]{v.tokenId("<0xC3>"), v.tokenId("<0xA9>")}, ids);
        assertEquals("\u00E9", t.decode(ids));
    }

    @Test
    public void mergePairsKeepFirstRank() {
        List<String> tokens = Arrays.asList("a", "b", "ab");
        Map<String, Object> md = metadata("gpt2", "gpt-2", tokens, Arrays.asList("a b", "a b"));
        TokenizerVocab v = TokenizerVocab.fromMetadata(md);
        assertEquals(0, v.mergeRank("a", "b"));
    }
}
