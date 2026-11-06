package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * vLLM's message reasoning keys (protocol.py {@code _normalize_messages_before} and
 * {@code parse_chat_message_content}): {@code reasoning} is current, {@code reasoning_content} its deprecated
 * alias, and an assistant message carries both so a template written for either name renders the same text.
 */
public class ChatMessagesReasoningTest {

    private static Map<String, Object> assistant(String role, Object reasoning) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", role);
        message.put("content", "");
        if (reasoning != null) {
            message.put("reasoning", reasoning);
        }
        return message;
    }

    private static Map<String, Object> copyOfOne(Map<String, Object> message) {
        List<Map<String, Object>> out = ChatMessages.copyOf(Collections.singletonList(message));
        assertEquals(1, out.size());
        return out.get(0);
    }

    @Test
    public void vllmReasoningIsCarriedUnderBothKeys() {
        Map<String, Object> copy = copyOfOne(assistant("assistant", "why"));
        assertEquals("why", copy.get(ChatMessages.REASONING));
        assertEquals("why", copy.get(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void reasoningWinsOverTheDeprecatedAlias() {
        Map<String, Object> message = assistant("assistant", "current");
        message.put(ChatMessages.REASONING_CONTENT, "deprecated");
        Map<String, Object> copy = copyOfOne(message);
        assertEquals("current", copy.get(ChatMessages.REASONING));
        assertEquals("current", copy.get(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void theDeprecatedAliasAloneFillsBothKeys() {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", "");
        message.put(ChatMessages.REASONING_CONTENT, "why");
        Map<String, Object> copy = copyOfOne(message);
        assertEquals("why", copy.get(ChatMessages.REASONING));
        assertEquals("why", copy.get(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void aNonNullReasoningStaysAsSent() {
        Map<String, Object> message = assistant("assistant", null);
        message.put(ChatMessages.REASONING, null);
        Map<String, Object> copy = copyOfOne(message);
        assertTrue(copy.containsKey(ChatMessages.REASONING));
        assertEquals(null, copy.get(ChatMessages.REASONING));
        assertFalse("no alias is invented for a null reasoning", copy.containsKey(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void aUserMessageKeepsItsReasoningAsSent() {
        Map<String, Object> copy = copyOfOne(assistant("user", "why"));
        assertEquals("why", copy.get(ChatMessages.REASONING));
        assertFalse("only an assistant turn carries the alias", copy.containsKey(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void nonStringReasoningIsRefusedByName() {
        Map<String, Object> message = assistant("assistant", Collections.singletonMap("type", "prompt_embeds"));
        try {
            ChatMessages.copyOf(Collections.singletonList(message));
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("'reasoning'"));
            return;
        }
        throw new AssertionError("a non-string reasoning was accepted");
    }

    @Test
    public void theCallersMessageIsNotModified() {
        Map<String, Object> message = assistant("assistant", "why");
        copyOfOne(message);
        assertEquals("why", message.get(ChatMessages.REASONING));
        assertFalse("the alias is added to the copy only", message.containsKey(ChatMessages.REASONING_CONTENT));
    }

    @Test
    public void stepFunTrimmingCoversBothReasoningKeys() {
        Map<String, Object> message = assistant("assistant", "  why\n");
        message.put(ChatMessages.REASONING_CONTENT, "  why\n");
        Map<String, Object> trimmed = ChatMessages.trimAllContent(Collections.singletonList(message)).get(0);
        assertEquals("why", trimmed.get(ChatMessages.REASONING));
        assertEquals("why", trimmed.get(ChatMessages.REASONING_CONTENT));
    }
}
