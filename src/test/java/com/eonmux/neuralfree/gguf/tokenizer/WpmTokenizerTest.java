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
 * Parity tests for {@link WpmTokenizer} against llama.cpp {@code llm_tokenizer_wpm_session} on a
 * synthetic BERT vocabulary laid out like bert-base ([PAD] 0, [UNK] 100, [CLS] 101, [SEP] 102,
 * [MASK] 103): pre-processing, greedy longest-match pieces, whole-word unknown fallback,
 * BOS/SEP insertion and detokenization with clean_spaces.
 */
public class WpmTokenizerTest {

    private static final String SP = "▁";
    private static final int CLS = 101;
    private static final int SEP = 102;
    private static final int UNK = 100;

    /** Synthetic WPM vocabulary; word-start pieces carry the U+2581 phantom space. */
    private static Map<String, Object> bertMetadata(String... extraPieces) {
        List<Object> tokens = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, types, "[PAD]", 3);
        for (int i = 1; i < 100; i++) add(tokens, types, "[unused" + i + "]", 5);
        add(tokens, types, "[UNK]", 2);
        add(tokens, types, "[CLS]", 3);
        add(tokens, types, "[SEP]", 3);
        add(tokens, types, "[MASK]", 3);
        String[] pieces = {
                SP + "hello", SP + "world", SP + "un", "aff", "able", "a", SP + "!", SP + ",",
                SP + "cafe", SP + "中", SP + "文", SP + "Hello", SP + "$", SP + "a", SP + "b", "b",
        };
        for (String p : pieces) add(tokens, types, p, 1);
        for (String p : extraPieces) add(tokens, types, p, 1);
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "bert");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.token_type", types);
        return md;
    }

    private static void add(List<Object> tokens, List<Object> types, String t, int type) {
        tokens.add(t);
        types.add(type);
    }

    private static WpmTokenizer tokenizer(TokenizerVocab v) {
        return new WpmTokenizer(v, SpecialTokens.of(v));
    }

    @Test
    public void preprocessSplitsPunctuationSymbolsAndCjkAndStripsAccents() {
        assertEquals(Arrays.asList("hello", ",", "world", "!", "a", "$", "b", "中", "文", "xy"),
                WpmTokenizer.preprocess("Héllo, Wörld!  a$b 中文 x\u0007y", true, true));
    }

    @Test
    public void preprocessKeepsCaseAndAccentsWhenDisabled() {
        assertEquals(Arrays.asList("Héllo", "é"),
                WpmTokenizer.preprocess("Héllo é", false, false));
        assertEquals(Arrays.asList("e"), WpmTokenizer.preprocess("é", true, true));
    }

    @Test
    public void nfdKeepsTheFirstCodePointOfTheDecomposition() {
        assertEquals('e', WpmTokenizer.nfdBase(0x00E9));
        assertEquals('A', WpmTokenizer.nfdBase(0x212B));
        assertEquals('a', WpmTokenizer.nfdBase('a'));
    }

    @Test
    public void helloWorldTraceAddsClsAndSep() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        int[] ids = tokenizer(v).encode("Hello, World!", true);
        assertArrayEquals(new int[]{CLS, v.tokenId(SP + "hello"), v.tokenId(SP + ","), v.tokenId(SP + "world"),
                v.tokenId(SP + "!"), SEP}, ids);
    }

    @Test
    public void greedyLongestMatchSplitsSubwords() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        int[] ids = tokenizer(v).encode("unaffable", false);
        assertArrayEquals(new int[]{v.tokenId(SP + "un"), v.tokenId("aff"), v.tokenId("able")}, ids);
    }

    @Test
    public void greedyMatchDoesNotBacktrack() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata(SP + "una"));
        assertArrayEquals(new int[]{UNK}, tokenizer(v).encode("unaffable", false));
    }

    @Test
    public void unmatchedPositionTurnsTheWholeWordIntoUnknown() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        WpmTokenizer t = tokenizer(v);
        assertArrayEquals(new int[]{UNK}, t.encode("xyz", false));
        assertArrayEquals(new int[]{UNK, v.tokenId(SP + "world")}, t.encode("unaffablez world", false));
    }

    @Test
    public void accentsAndCjkUseTheirOwnPieces() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        int[] ids = tokenizer(v).encode("Café 中文", false);
        assertArrayEquals(new int[]{v.tokenId(SP + "cafe"), v.tokenId(SP + "中"), v.tokenId(SP + "文")}, ids);
    }

    @Test
    public void lowercaseCanBeDisabled() {
        Map<String, Object> md = bertMetadata();
        md.put("tokenizer.ggml.normalizer.lowercase", false);
        TokenizerVocab v = TokenizerVocab.fromMetadata(md);
        assertArrayEquals(new int[]{v.tokenId(SP + "Hello")}, tokenizer(v).encode("Hello", false));
    }

    @Test
    public void specialTokensPassThrough() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        int[] ids = tokenizer(v).encode("hello[SEP]world", false);
        assertArrayEquals(new int[]{v.tokenId(SP + "hello"), SEP, v.tokenId(SP + "world")}, ids);
    }

    @Test
    public void detokenizeUnescapesAndCleansSpaces() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        WpmTokenizer t = tokenizer(v);
        int[] ids = t.encode("Hello, World!", true);
        assertEquals(" hello, world!", t.decode(ids));
        assertEquals(" unaffable", t.decode(t.encode("unaffable", false)));
    }

    @Test
    public void facadeRoutesBertModelToWpm() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(bertMetadata());
        GGUFTokenizer tok = new GGUFTokenizer(v);
        assertArrayEquals(new int[]{CLS, v.tokenId(SP + "hello"), v.tokenId(SP + "world"), SEP}, tok.encode("hello world"));
        assertEquals(" hello", tok.decode(v.tokenId(SP + "hello")));
        assertEquals("", tok.decode(SEP));
    }
}
