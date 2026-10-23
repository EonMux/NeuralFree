package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Parity tests for {@link SpmTokenizer} against llama.cpp {@code llm_tokenizer_spm_session} on a
 * synthetic SentencePiece vocabulary: whitespace escaping, {@code add_space_prefix}, code-point
 * seeding, score-ordered merging with leftmost tie-breaking, byte fallback and detokenization.
 */
public class SpmTokenizerTest {

    private static final String SP = "\u2581";

    /** Synthetic SPM vocabulary: specials, all 256 byte tokens, then scored pieces. */
    private static Map<String, Object> spmMetadata() {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 0f, 2);
        add(tokens, scores, types, "<s>", 0f, 3);
        add(tokens, scores, types, "</s>", 0f, 3);
        for (int b = 0; b < 256; b++) add(tokens, scores, types, String.format("<0x%02X>", b), 0f, 6);
        Object[][] pieces = {
                {SP, -1.0f}, {"H", -2.0f}, {"e", -2.0f}, {"l", -2.0f}, {"o", -2.0f}, {"w", -2.0f},
                {"r", -2.0f}, {"d", -2.0f}, {"a", -2.0f},
                {"ll", -3.0f}, {SP + "w", -3.1f}, {"or", -3.2f}, {"ld", -3.3f}, {"He", -3.5f},
                {"llo", -4.0f}, {SP + "wor", -4.5f}, {"Hello", -5.0f}, {SP + "world", -5.5f},
                {SP + "Hello", -6.0f}, {"aa", -1.5f}, {"ab", -0.0f}, {"bc", 0.0f},
        };
        for (Object[] p : pieces) add(tokens, scores, types, (String) p[0], (Float) p[1], 1);
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", 1);
        md.put("tokenizer.ggml.eos_token_id", 2);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        return md;
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, float s, int type) {
        tokens.add(t);
        scores.add(s);
        types.add(type);
    }

    private static TokenizerVocab vocab(Map<String, Object> md) {
        return TokenizerVocab.fromMetadata(md);
    }

    private static SpmTokenizer tokenizer(TokenizerVocab v) {
        return new SpmTokenizer(v, SpecialTokens.of(v));
    }

    @Test
    public void helloWorldMatchesLlamaCppTrace() {
        TokenizerVocab v = vocab(spmMetadata());
        int[] ids = tokenizer(v).encode("Hello world", true);
        assertArrayEquals(new int[]{1, v.tokenId(SP + "Hello"), v.tokenId(SP + "world")}, ids);
    }

    @Test
    public void mergeOrderFollowsScoresThenLeftmost() {
        SpmTokenizer t = tokenizer(vocab(spmMetadata()));
        assertEquals(Arrays.asList(SP + "Hello", SP + "world"), t.mergedSymbols(SP + "Hello" + SP + "world"));
        assertEquals(Arrays.asList("aa", "a"), t.mergedSymbols("aaa"));
    }

    @Test
    public void signedZeroScoresTieAndMergeLeftmost() {
        SpmTokenizer t = tokenizer(vocab(spmMetadata()));
        assertEquals(Arrays.asList("ab", "c"), t.mergedSymbols("abc"));
    }

    @Test
    public void addSpacePrefixCanBeDisabled() {
        Map<String, Object> md = spmMetadata();
        md.put("tokenizer.ggml.add_space_prefix", false);
        TokenizerVocab v = vocab(md);
        assertArrayEquals(new int[]{v.tokenId("Hello")}, tokenizer(v).encode("Hello", false));
    }

    @Test
    public void unknownPiecesFallBackToBytes() {
        TokenizerVocab v = vocab(spmMetadata());
        int[] ids = tokenizer(v).encode("H\u00E9", false);
        assertArrayEquals(new int[]{v.tokenId(SP), v.tokenId("H"), v.tokenId("<0xC3>"), v.tokenId("<0xA9>")}, ids);
    }

    @Test
    public void specialTokenStartsNewFragmentWithSpacePrefix() {
        TokenizerVocab v = vocab(spmMetadata());
        int[] ids = tokenizer(v).encode("</s>Hello", false);
        assertArrayEquals(new int[]{2, v.tokenId(SP + "Hello")}, ids);
    }

    @Test
    public void addEosWhenRequested() {
        Map<String, Object> md = spmMetadata();
        md.put("tokenizer.ggml.add_eos_token", true);
        TokenizerVocab v = vocab(md);
        int[] ids = tokenizer(v).encode("Hello", true);
        assertArrayEquals(new int[]{1, v.tokenId(SP + "Hello"), 2}, ids);
    }

    @Test
    public void detokenizeRemovesPrefixSpaceOnlyWithoutLeadingBos() {
        TokenizerVocab v = vocab(spmMetadata());
        SpmTokenizer t = tokenizer(v);
        int hello = v.tokenId(SP + "Hello");
        int world = v.tokenId(SP + "world");
        assertEquals("Hello world", t.decode(new int[]{hello, world}));
        assertEquals(" Hello world", t.decode(new int[]{1, hello, world}));
        assertEquals("\u00E9", t.decode(new int[]{v.tokenId("<0xC3>"), v.tokenId("<0xA9>")}));
    }

    @Test
    public void facadeRoutesLlamaModelToSpm() {
        TokenizerVocab v = vocab(spmMetadata());
        GGUFTokenizer tok = new GGUFTokenizer(v);
        assertArrayEquals(new int[]{1, v.tokenId(SP + "Hello"), v.tokenId(SP + "world")}, tok.encode("Hello world"));
        assertEquals(" Hello", tok.decode(v.tokenId(SP + "Hello")));
        assertEquals("", tok.decode(1));
        GGUFTokenizer.StreamingDecoder sd = tok.createStreamingDecoder();
        assertEquals("", sd.decodeToken(v.tokenId("<0xC3>")));
        assertEquals("\u00E9", sd.decodeToken(v.tokenId("<0xA9>")));
        assertEquals("", sd.flush());
    }
}
