package com.eonmux.neuralfree.gguf.tokenizer;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Parity tests for {@link RwkvTokenizer} against llama.cpp {@code llm_tokenizer_rwkv_session} on a
 * synthetic RWKV world vocabulary: escaped token texts, longest trie match without backtracking,
 * no BOS/EOS, and byte-exact detokenization.
 */
public class RwkvTokenizerTest {

    private static final String CHINESE_ZHONG = "\\xe4\\xb8\\xad";

    /**
     * Every single byte (printable ASCII as itself, backslash as {@code \\}, the rest as
     * {@code \xNN}), then multi-byte tokens.
     */
    private static Map<String, Object> rwkvMetadata() {
        List<Object> tokens = new ArrayList<>();
        for (int b = 0; b < 256; b++) {
            if (b == '\\') tokens.add("\\\\");
            else if (b >= 0x20 && b < 0x7F) tokens.add(String.valueOf((char) b));
            else tokens.add(String.format("\\x%02x", b));
        }
        String[] multi = {"hello", " world", "wor", "\\n\\n", CHINESE_ZHONG, "ab", "abcd"};
        for (String m : multi) tokens.add(m);
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "rwkv");
        md.put("tokenizer.ggml.tokens", tokens);
        return md;
    }

    private static RwkvTokenizer tokenizer(TokenizerVocab v) {
        return new RwkvTokenizer(v, SpecialTokens.of(v));
    }

    private static byte[] bytes(String s) {
        return s.getBytes(StandardCharsets.ISO_8859_1);
    }

    @Test
    public void unescapeMatchesLlamaCpp() {
        assertArrayEquals(bytes("a\tb\n\rA\\q"), RwkvTokenizer.unescape("a\\tb\\n\\r\\x41\\\\\\q"));
        assertArrayEquals(new byte[]{(byte) 0xE4, (byte) 0xB8, (byte) 0xAD}, RwkvTokenizer.unescape(CHINESE_ZHONG));
        assertArrayEquals(bytes("ÿ"), RwkvTokenizer.unescape("\\xff"));
    }

    @Test
    public void helloWorldTraceTakesLongestTrieMatches() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(rwkvMetadata());
        int[] ids = tokenizer(v).encode("hello world\n\n中", true);
        assertArrayEquals(new int[]{v.tokenId("hello"), v.tokenId(" world"), v.tokenId("\\n\\n"), v.tokenId(CHINESE_ZHONG)}, ids);
    }

    @Test
    public void longestMatchDoesNotBacktrack() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(rwkvMetadata());
        int[] ids = tokenizer(v).encode("abcx", false);
        assertArrayEquals(new int[]{v.tokenId("ab"), v.tokenId("c"), v.tokenId("x")}, ids);
    }

    @Test
    public void detokenizeRendersUnescapedBytes() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(rwkvMetadata());
        RwkvTokenizer t = tokenizer(v);
        String text = "hello world\n\n中 \\ abcd";
        assertEquals(text, t.decode(t.encode(text, false)));
        assertArrayEquals(new byte[]{'\n', '\n'}, t.piece(v.tokenId("\\n\\n")));
        assertArrayEquals(new byte[0], t.piece(-1));
    }

    @Test
    public void facadeRoutesRwkvModel() {
        TokenizerVocab v = TokenizerVocab.fromMetadata(rwkvMetadata());
        GGUFTokenizer tok = new GGUFTokenizer(v);
        assertArrayEquals(new int[]{v.tokenId("hello"), v.tokenId(" world")}, tok.encode("hello world"));
        assertEquals(" world", tok.decode(v.tokenId(" world")));
    }
}
