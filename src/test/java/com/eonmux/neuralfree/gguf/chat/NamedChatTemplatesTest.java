package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The built-in named chat templates of {@link ChatTemplateSource}: each Mistral template renders a conversation to
 * the string llama.cpp's {@code llm_chat_apply_template} builds for it (src/llama-chat.cpp:258-298), written out
 * here by hand.
 */
public class NamedChatTemplatesTest {

    private static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /** A system prompt and two user turns around an assistant reply with surrounding whitespace. */
    private static final List<Map<String, String>> CONVERSATION = Arrays.asList(msg("system", "S"),
            msg("user", "U1"), msg("assistant", " A1\n"), msg("user", "U2"));

    private static String render(String name, List<Map<String, String>> messages) {
        ChatFormatter f = new ChatFormatter(ChatTemplateSource.resolve(name, null), "<s>", "</s>", true, false,
                "llama");
        assertTrue(name + " renders the system role", f.supportsSystemRole());
        return f.format(messages, false);
    }

    @Test
    public void mistralV7TekkenRendersTheTwoTurnConversation() {
        assertEquals("[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]U1[/INST]A1</s>[INST]U2[/INST]",
                render("mistral-v7-tekken", Arrays.asList(msg("system", "S"), msg("user", "U1"),
                        msg("assistant", "A1"), msg("user", "U2"))));
    }

    @Test
    public void mistralV7SeparatesWithSpaces() {
        assertEquals("[SYSTEM_PROMPT] S[/SYSTEM_PROMPT][INST] U1[/INST]  A1\n</s>[INST] U2[/INST]",
                render("mistral-v7", CONVERSATION));
        assertEquals("[SYSTEM_PROMPT]S[/SYSTEM_PROMPT][INST]U1[/INST] A1\n</s>[INST]U2[/INST]",
                render("mistral-v7-tekken", CONVERSATION));
    }

    @Test
    public void mistralV1PutsTheSystemPromptInsideTheFirstTurn() {
        assertEquals(" [INST] S\n\nU1 [/INST]  A1\n</s> [INST] U2 [/INST]", render("mistral-v1", CONVERSATION));
    }

    @Test
    public void mistralV3TrimsAssistantMessages() {
        assertEquals("[INST] S\n\nU1[/INST] A1</s>[INST] U2[/INST]", render("mistral-v3", CONVERSATION));
    }

    @Test
    public void mistralV3TekkenHasNoSpaces() {
        assertEquals("[INST]S\n\nU1[/INST] A1\n</s>[INST]U2[/INST]", render("mistral-v3-tekken", CONVERSATION));
    }

    @Test
    public void namesListChatmlFirstAndTheMistralTemplatesInLlamaCppOrder() {
        List<String> names = Arrays.asList(ChatTemplateSource.names().toArray(new String[0]));
        assertEquals("chatml", names.get(0));
        int v1 = names.indexOf("mistral-v1");
        assertEquals(Arrays.asList("mistral-v1", "mistral-v3", "mistral-v3-tekken", "mistral-v7",
                "mistral-v7-tekken"), names.subList(v1, v1 + 5));
        assertThrows(UnsupportedOperationException.class, () -> ChatTemplateSource.names().add("x"));
        assertTrue(ChatTemplateSource.isName("mistral-v3"));
        assertTrue(ChatTemplateSource.isName(ChatTemplateSource.CHATML_NAME));
        assertFalse(ChatTemplateSource.isName("mistral-v8"));
        assertFalse(ChatTemplateSource.isName(null));
    }

    @Test
    public void namedResolvesBuiltInTemplatesOnce() {
        assertEquals(ChatTemplateSource.CHATML, ChatTemplateSource.named("chatml"));
        String v7 = ChatTemplateSource.named("mistral-v7");
        assertTrue(v7.contains("[SYSTEM_PROMPT]"));
        assertSame("loaded once", v7, ChatTemplateSource.named("mistral-v7"));
        assertEquals(v7, ChatTemplateSource.resolve("mistral-v7"));
        assertNull(ChatTemplateSource.named("{{ messages }}"));
        assertEquals("a template source is not a name", "{{ x }}", ChatTemplateSource.resolve("{{ x }}"));
    }

    @Test
    public void namedTemplateKeysFollowTheGgufWriter() {
        assertEquals(ChatTemplateSource.KEY_TOOL_USE, ChatTemplateSource.namedKey("tool_use"));
        assertEquals("tokenizer.chat_template.rag", ChatTemplateSource.namedKey("rag"));
        assertEquals("tokenizer.chat_templates", ChatTemplateSource.KEY_NAMES);
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.namedKey(""));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplateSource.namedKey(null));
    }
}
