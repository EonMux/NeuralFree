package com.eonmux.neuralfree.gguf.chat;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/** The chat request's builder validation: the template override and the documents. */
public class ChatRequestTest {

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    @Test
    public void aNewRequestCarriesNoOverrideAndNoDocuments() {
        ChatRequest request = ChatRequest.builder()
                .messages(Collections.singletonList(message("user", "U")))
                .build();
        assertNull(request.chatTemplate());
        assertTrue(request.documents().isEmpty());
    }

    @Test(expected = IllegalArgumentException.class)
    public void anEmptyTemplateOverrideIsRefused() {
        ChatRequest.builder().chatTemplate("");
    }

    @Test(expected = IllegalArgumentException.class)
    public void aNullDocumentIsRefused() {
        List<Map<String, ?>> documents = new ArrayList<>();
        documents.add(null);
        ChatRequest.builder().documents(documents);
    }

    @Test
    public void nullDocumentsClearTheDocuments() {
        Map<String, String> document = new LinkedHashMap<>();
        document.put("t", "x");
        List<Map<String, ?>> given = new ArrayList<>();
        given.add(document);
        ChatRequest request = ChatRequest.builder().documents(given).documents(null).build();
        assertTrue(request.documents().isEmpty());
    }

    @Test
    public void documentsAreCopiedDeep() {
        Map<String, Object> inner = new LinkedHashMap<>();
        inner.put("text", "X");
        Map<String, Object> document = new LinkedHashMap<>();
        document.put("title", "T");
        document.put("meta", inner);
        List<Map<String, ?>> given = new ArrayList<>();
        given.add(document);
        ChatRequest request = ChatRequest.builder().documents(given).build();
        document.put("title", "changed");
        inner.put("text", "changed");
        assertEquals("T", request.documents().get(0).get("title"));
        assertEquals("X", ((Map<?, ?>) request.documents().get(0).get("meta")).get("text"));
    }

    @Test
    public void aToBuilderCarriesTheOverrideAndTheDocuments() {
        Map<String, String> document = new LinkedHashMap<>();
        document.put("title", "T");
        List<Map<String, ?>> given = new ArrayList<>();
        given.add(document);
        ChatRequest request = ChatRequest.builder()
                .messages(Collections.singletonList(message("user", "U")))
                .chatTemplate("T")
                .documents(given)
                .build()
                .toBuilder()
                .build();
        assertEquals("T", request.chatTemplate());
        assertEquals("T", request.documents().get(0).get("title"));
    }
}
