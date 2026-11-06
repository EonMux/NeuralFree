package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Embedding content parts ({@code image_embeds}/{@code audio_embeds}, vLLM's precomputed embeddings): the
 * flattened message content carries one {@link ChatMessages#PROMPT_EMBEDS_MARKER} per part, text parts around
 * them stay in order, and the marker is distinct from the media marker the text chat path refuses.
 */
public class ChatMessagesEmbedsTest {

    private static Map<String, Object> text(String s) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", "text");
        part.put("text", s);
        return part;
    }

    private static Map<String, Object> embeds(String kind) {
        Map<String, Object> part = new LinkedHashMap<>();
        part.put("type", kind);
        return part;
    }

    private static List<Map<String, Object>> messages(Object... content) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", Arrays.asList(content));
        List<Map<String, Object>> out = new ArrayList<>();
        out.add(message);
        return out;
    }

    @Test
    public void anEmbedsPartLeavesItsMarkerBetweenTheTexts() {
        List<Map<String, Object>> flattened = ChatMessages.forTemplate(
                messages(text("look "), embeds("image_embeds"), text(" and "), embeds("audio_embeds")));
        assertEquals("look <##PROMPT_EMBEDS##> and <##PROMPT_EMBEDS##>",
                String.valueOf(flattened.get(0).get(ChatMessages.CONTENT)));
    }

    @Test
    public void theEmbedsMarkerIsNotTheMediaMarker() {
        List<Map<String, Object>> flattened = ChatMessages.forTemplate(messages(embeds("image_embeds")));
        String content = String.valueOf(flattened.get(0).get(ChatMessages.CONTENT));
        assertTrue(content.contains(ChatMessages.PROMPT_EMBEDS_MARKER));
        assertTrue(!content.contains(ChatMessages.MEDIA_MARKER));
    }

    @Test
    public void anUnknownPartTypeIsStillRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> ChatMessages.forTemplate(messages(text("a"), embeds("video_embeds"))));
    }

    @Test
    public void aStringContentCarriesNoMarker() {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "user");
        message.put("content", "plain text");
        List<Map<String, Object>> in = new ArrayList<>();
        in.add(message);
        List<Map<String, Object>> flattened = ChatMessages.forTemplate(in);
        assertEquals("plain text", String.valueOf(flattened.get(0).get(ChatMessages.CONTENT)));
    }
}
