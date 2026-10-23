package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.chat.ChatFormatter;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.tokenizer.TokenizerVocab;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Chat prompts keep exactly one BOS: the BOS text a template writes is removed when the tokenizer adds BOS itself,
 * and the EOS text at the end when it adds EOS, as llama.cpp does after rendering (common/chat.cpp:937-949). A model
 * without a chat template renders llama.cpp's ChatML default (common/chat.cpp:752-758).
 */
public class GGUFTokenizerChatBosTest {

    private static final String SP = "▁";
    private static final String TEMPLATE =
            "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}]{{ m['content'] }}{% endfor %}";
    private static final int BOS = 1;

    /** A SentencePiece vocabulary with control BOS/EOS, byte fallback and a chat template. */
    private static GGUFTokenizer tokenizer(Boolean addBos) {
        return tokenizer(addBos, TEMPLATE);
    }

    /** As {@link #tokenizer(Boolean)}, with {@code template} as the chat template (null: no template). */
    private static GGUFTokenizer tokenizer(Boolean addBos, String template) {
        List<Object> tokens = new ArrayList<>();
        List<Object> scores = new ArrayList<>();
        List<Object> types = new ArrayList<>();
        add(tokens, scores, types, "<unk>", 2);
        add(tokens, scores, types, "<s>", 3);
        add(tokens, scores, types, "</s>", 3);
        for (int b = 0; b < 256; b++) {
            add(tokens, scores, types, String.format("<0x%02X>", b), 6);
        }
        for (String p : new String[]{SP, "a", "b", "u", "s", "e", "r", "[", "]"}) {
            add(tokens, scores, types, p, 1);
        }
        Map<String, Object> md = new HashMap<>();
        md.put("tokenizer.ggml.model", "llama");
        md.put("tokenizer.ggml.tokens", tokens);
        md.put("tokenizer.ggml.scores", scores);
        md.put("tokenizer.ggml.token_type", types);
        md.put("tokenizer.ggml.bos_token_id", BOS);
        md.put("tokenizer.ggml.eos_token_id", 2);
        md.put("tokenizer.ggml.unknown_token_id", 0);
        if (template != null) {
            md.put("tokenizer.chat_template", template);
        }
        if (addBos != null) {
            md.put(TokenizerVocab.KEY_ADD_BOS, addBos);
        }
        return new GGUFTokenizer(TokenizerVocab.fromMetadata(md));
    }

    private static void add(List<Object> tokens, List<Object> scores, List<Object> types, String t, int type) {
        tokens.add(t);
        scores.add(-1.0f);
        types.add(type);
    }

    private static int count(int[] ids, int id) {
        int n = 0;
        for (int v : ids) {
            if (v == id) {
                n++;
            }
        }
        return n;
    }

    @Test
    public void templateBosIsRemovedWhenTheTokenizerAddsBos() {
        GGUFTokenizer tok = tokenizer(Boolean.TRUE);
        String prompt = tok.formatChat(null, "ab", false);
        assertFalse(prompt, prompt.startsWith("<s>"));
        int[] ids = tok.encode(prompt);
        assertEquals(BOS, ids[0]);
        assertEquals(1, count(ids, BOS));
    }

    @Test
    public void templateBosIsKeptWhenTheTokenizerDoesNotAddBos() {
        GGUFTokenizer tok = tokenizer(Boolean.FALSE);
        String prompt = tok.formatChat(null, "ab", false);
        assertTrue(prompt, prompt.startsWith("<s>"));
        int[] ids = tok.encode(prompt);
        assertEquals(BOS, ids[0]);
        assertEquals(1, count(ids, BOS));
    }

    @Test
    public void multiTurnPromptsStartWithOneBos() {
        GGUFTokenizer tok = tokenizer(null);
        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> user = new HashMap<>();
        user.put("role", "user");
        user.put("content", "ab");
        messages.add(user);
        int[] ids = tok.encode(tok.formatChat(messages, false));
        assertEquals(BOS, ids[0]);
        assertEquals(1, count(ids, BOS));
    }

    @Test
    public void aModelWithoutTemplateUsesChatMlForBothEntryPoints() {
        GGUFTokenizer tok = tokenizer(Boolean.TRUE, null);
        assertEquals(ChatTemplateSource.CHATML, tok.chatTemplate());
        String expected = "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nab<|im_end|>\n<|im_start|>assistant\n";
        assertEquals(expected, tok.formatChat("sys", "ab", false));
        List<Map<String, String>> messages = new ArrayList<>();
        messages.add(message("system", "sys"));
        messages.add(message("user", "ab"));
        assertEquals(expected, tok.formatChat(messages, false));
    }

    private static Map<String, String> message(String role, String content) {
        Map<String, String> m = new HashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    @Test
    public void stripRemovesOnlyTheConfiguredAffixes() {
        assertEquals("x</s>", ChatFormatter.stripAddedSpecials("<s>x</s>", "<s>", "</s>", true, false));
        assertEquals("<s>x", ChatFormatter.stripAddedSpecials("<s>x</s>", "<s>", "</s>", false, true));
        assertEquals("x", ChatFormatter.stripAddedSpecials("<s>x</s>", "<s>", "</s>", true, true));
        assertEquals("<s>x</s>", ChatFormatter.stripAddedSpecials("<s>x</s>", "<s>", "</s>", false, false));
        assertEquals("x<s>", ChatFormatter.stripAddedSpecials("x<s>", "<s>", "", true, true));
        assertEquals("<s>x", ChatFormatter.stripAddedSpecials("<s>x", "", "", true, true));
    }
}
