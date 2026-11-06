package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * The templates whose llama.cpp formatters upper-case roles do it byte-wise, as C's {@code toupper}
 * does (src/llama-chat.cpp:428, :732, :759): ASCII letters change, every other character is left
 * alone — the template-local {@code ascii_upper} macro keeps that, where Jinja's {@code upper} would
 * case-map the whole Unicode alphabet.
 */
public class ChatTemplateAsciiUpperTest {

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    private static String render(String name, Map<String, String> message) {
        List<Map<String, String>> messages = Collections.singletonList(message);
        ChatFormatter formatter = new ChatFormatter(ChatTemplateSource.resolve(name), "", "", false, false, "llama");
        return formatter.format(ChatRequest.builder().messages(messages).addGenerationPrompt(false).build());
    }

    @Test
    public void openchatUpperCasesTheFirstAsciiLetterOnly() {
        assertEquals("GPT4 Correct User: hi<|end_of_turn|>",
                render("openchat", msg("user", "hi")));
        assertEquals("GPT4 Correct Assistant: hi<|end_of_turn|>",
                render("openchat", msg("assistant", "hi")));
        assertEquals("a non-ASCII first letter stays, as toupper leaves it",
                "GPT4 Correct über: hi<|end_of_turn|>", render("openchat", msg("über", "hi")));
        assertEquals("GPT4 Correct ιdeas: hi<|end_of_turn|>", render("openchat", msg("ιdeas", "hi")));
    }

    @Test
    public void bailingUpperCasesTheAsciiLettersOfEveryNonUserRole() {
        assertEquals("<role>ASSISTANT</role>hi", render("bailing", msg("assistant", "hi")));
        assertEquals("<role>HUMAN</role>hi", render("bailing", msg("user", "hi")));
        assertEquals("mixed case upper-cases letter by letter", "<role>ASSISTANT</role>hi",
                render("bailing", msg("AsSiStAnT", "hi")));
        assertEquals("a non-ASCII letter is left alone, where Unicode upper would map it",
                "<role>ïDEA</role>hi", render("bailing", msg("ïdea", "hi")));
        assertEquals("<role>GRüßE</role>hi", render("bailing", msg("grüße", "hi")));
    }

    @Test
    public void bailingThinkSharesBailingsRoleForm() {
        assertEquals("<role>SYSTEM</role>hi", render("bailing-think", msg("system", "hi")));
        assertEquals("<role>ïDEA</role>hi", render("bailing-think", msg("ïdea", "hi")));
    }

    @Test
    public void bailing2UpperCasesLikeBailing() {
        List<Map<String, String>> messages = Arrays.asList(msg("system", "think"), msg("ïdea", "hi"));
        ChatFormatter formatter = new ChatFormatter(ChatTemplateSource.resolve("bailing2"), "", "", false, false,
                "llama");
        String rendered = formatter.format(ChatRequest.builder().messages(messages)
                .addGenerationPrompt(false).build());
        assertEquals("<role>SYSTEM</role>think<|role_end|><role>ïDEA</role>hi<|role_end|>", rendered);
    }
}
