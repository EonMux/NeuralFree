package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.GGUFTokenizer;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * The rerank input of the llama.cpp server (tools/server/server-common.cpp:1948-1991): a
 * {@code tokenizer.chat_template.rerank} prompt is filled ({@code {query}} first, then {@code {document}}, as
 * string_replace_all runs) and tokenized with special tokens parsed and none added; without one the query and document
 * are tokenized without special-token parsing and packed with the vocabulary's special tokens, and only then do
 * tokenized inputs apply.
 */
public class RerankInputTest {

    private static final String SP = "▁";
    private static final int BOS = 1;
    private static final int EOS = 2;
    private static final String TEMPLATE = "<s>q:{query}|d:{document}</s>";

    /** A SentencePiece vocabulary with control BOS / EOS, byte fallback and the given rerank prompt (or none). */
    private static GGUFTokenizer tokenizer(String rerankTemplate, boolean addEos) {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        for (String p : new String[]{SP, "a", "b", "c", "d", "q", ":", "|"}) {
            add(tokens, scores, types, p, 1);
        }
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", BOS);
        md.put("tokenizer.ggml.eos_token_id", EOS);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        md.put(TokenizerVocab.KEY_ADD_BOS, true);
        md.put(TokenizerVocab.KEY_ADD_EOS, addEos);
        if (rerankTemplate != null) {
            md.put(GGUFTokenizer.KEY_RERANK_TEMPLATE, rerankTemplate);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, int type) {
        tokens.add(t);
        scores.add(-1.0f);
        types.add(type);
    }

    @Test
    public void rerankPromptIsFilledAndTokenizedWithoutAddedSpecials() {
        GGUFTokenizer tok = tokenizer(TEMPLATE, true);
        assertEquals(TEMPLATE, tok.rerankTemplate());
        int[] input = ClassifierHead.rerankInput(tok, "ab", "cd");
        assertArrayEquals(tok.encode("<s>q:ab|d:cd</s>", false), input);
        assertEquals(BOS, input[0]);
        assertEquals(EOS, input[input.length - 1]);
    }

    @Test
    public void queryPlaceholdersAreReplacedBeforeDocumentPlaceholders() {
        GGUFTokenizer tok = tokenizer(TEMPLATE, false);
        int[] input = ClassifierHead.rerankInput(tok, "a{document}", "cd");
        assertArrayEquals(tok.encode("<s>q:acd|d:cd</s>", false), input);
    }

    @Test
    public void withoutAPromptTheTokensArePacked() {
        GGUFTokenizer tok = tokenizer(null, true);
        assertNull(tok.rerankTemplate());
        int[] q = tok.encode("ab", false);
        int[] d = tok.encode("cd", false);
        int[] expected = ClassifierHead.packRerank(q, d, tok.vocab(), tok.specialTokens());
        assertArrayEquals(expected, ClassifierHead.rerankInput(tok, "ab", "cd"));
        assertArrayEquals(expected, ClassifierHead.rerankInput(tok, q, d));
        assertEquals(BOS, expected[0]);
        assertEquals(EOS, expected[q.length + 1]);
        assertEquals(EOS, expected[expected.length - 1]);
    }

    @Test
    public void withoutAPromptControlTokenTextIsPlainText() {
        GGUFTokenizer tok = tokenizer(null, false);
        int[] q = tok.encode("a</s>", false, false);
        int[] d = tok.encode("<s>c", false, false);
        for (int id : q) {
            assertNotEquals(EOS, id);
        }
        for (int id : d) {
            assertNotEquals(BOS, id);
        }
        int[] parsed = tok.encode("a</s>", false);
        assertEquals(EOS, parsed[parsed.length - 1]);
        int[] expected = ClassifierHead.packRerank(q, d, tok.vocab(), tok.specialTokens());
        assertArrayEquals(expected, ClassifierHead.rerankInput(tok, "a</s>", "<s>c"));
    }

    @Test
    public void tokenizedInputIsRefusedWhenThePromptNeedsText() {
        GGUFTokenizer tok = tokenizer(TEMPLATE, false);
        assertThrows(UnsupportedOperationException.class,
                () -> ClassifierHead.rerankInput(tok, new int[]{3}, new int[]{4}));
    }
}
