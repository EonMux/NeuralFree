package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The Responses request as a chat-completions body: messages, merged assistant turns, tools and formats. */
class ResponsesRequestsTest {

    private static JsonView body(String json) {
        return JsonView.of(Json.parse(json, Json.Limits.UNTRUSTED), "body");
    }

    private static Map<String, Object> chat(String json) {
        return ResponsesRequests.chatBody(body(json), List.of());
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> messages(Map<String, Object> chat) {
        return (List<Map<String, Object>>) chat.get("messages");
    }

    @Test
    void instructionsLeadAndAStringInputIsOneUserMessage() {
        List<Map<String, Object>> m = messages(chat("{\"instructions\": \"be brief\", \"input\": \"hi\"}"));
        assertEquals(2, m.size());
        assertEquals(Map.of("role", "system", "content", "be brief"), m.get(0));
        assertEquals("user", m.get(1).get("role"));
        assertEquals(List.of(Map.of("type", "text", "text", "hi")), m.get(1).get("content"));
    }

    @Test
    void consecutiveAssistantItemsMergeIntoOneTurnAndOutputsBecomeToolMessages() {
        List<Map<String, Object>> m = messages(chat("""
                {"input": [
                  {"role": "user", "content": "weather?"},
                  {"type": "reasoning", "summary": [{"type": "summary_text", "text": "need a tool"}]},
                  {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
                  {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{}"},
                  {"type": "function_call_output", "call_id": "c1", "output": "sunny"}]}"""));
        assertEquals(3, m.size());
        Map<String, Object> turn = m.get(1);
        assertEquals("assistant", turn.get("role"));
        assertEquals("need a tool", turn.get("reasoning_content"));
        assertEquals(List.of(Map.of("type", "text", "text", "checking")), turn.get("content"));
        assertEquals(1, ((List<?>) turn.get("tool_calls")).size());
        assertEquals(Map.of("role", "tool", "tool_call_id", "c1", "content", "sunny"), m.get(2));
    }

    @Test
    void historyContinuesWithoutItsSystemMessageAndStaysUnchanged() {
        Map<String, Object> said = new java.util.LinkedHashMap<>(Map.of("role", "assistant", "content", "earlier"));
        List<Map<String, Object>> history = List.of(Map.of("role", "system", "content", "old"),
                Map.of("role", "user", "content", "first"), said);
        Map<String, Object> chat = ResponsesRequests.chatBody(body("""
                {"input": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]}"""), history);
        List<Map<String, Object>> m = messages(chat);
        assertEquals(2, m.size(), "the old system message stays out and the call joins the assistant turn");
        assertTrue(m.get(1).containsKey("tool_calls"));
        assertFalse(said.containsKey("tool_calls"), "the stored history message is not modified");
    }

    @Test
    void toolsToolChoiceBudgetEffortAndFormatMapToTheirChatFields() {
        Map<String, Object> chat = chat("""
                {"input": "x", "max_output_tokens": 7, "reasoning": {"effort": "low"}, "top_logprobs": 2,
                 "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}},
                           {"type": "web_search_preview"}],
                 "tool_choice": {"type": "function", "name": "f"},
                 "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}},
                 "extra_params": {"min_p": 0.1}, "temperature": 0.5, "store": false}""");
        assertEquals(7L, ((Number) chat.get("max_tokens")).longValue());
        assertEquals("low", chat.get("reasoning_effort"));
        assertEquals(1, ((List<?>) chat.get("tools")).size(), "the built-in tool has no chat equivalent");
        assertEquals(Map.of("type", "function", "function", Map.of("name", "f")), chat.get("tool_choice"));
        Map<?, ?> format = (Map<?, ?>) chat.get("response_format");
        assertEquals("json_schema", format.get("type"));
        assertEquals("out", ((Map<?, ?>) format.get("json_schema")).get("name"));
        assertEquals(true, chat.get("logprobs"));
        assertEquals(0.1, ((Number) chat.get("min_p")).doubleValue(), 1e-9);
        assertEquals(0.5, ((Number) chat.get("temperature")).doubleValue(), 1e-9);
        assertFalse(chat.containsKey("store") || chat.containsKey("input") || chat.containsKey("text"));
    }

    @Test
    void aStructuralTagFormatPassesThroughAsTheChatResponseFormat() {
        Map<String, Object> chat = chat("""
                {"input": "x", "text": {"format": {"type": "structural_tag",
                  "format": {"type": "const_string", "value": "ok"}}}}""");
        Map<?, ?> format = (Map<?, ?>) chat.get("response_format");
        assertEquals("structural_tag", format.get("type"));
        assertEquals(Map.of("type", "const_string", "value", "ok"), format.get("format"));

        Map<String, Object> legacy = chat("""
                {"input": "x", "text": {"format": {"type": "structural_tag", "triggers": ["<f"],
                  "structures": [{"begin": "<f>", "schema": {}, "end": "</f>"}]}}}""");
        Map<?, ?> legacyFormat = (Map<?, ?>) legacy.get("response_format");
        assertEquals(List.of("<f"), legacyFormat.get("triggers"));
        assertEquals(1, ((List<?>) legacyFormat.get("structures")).size());
    }

    @Test
    void aStructuralTagFormatWithoutAPayloadIsRefusedByName() {
        WebApplicationException e = assertThrows(WebApplicationException.class,
                () -> chat("{\"input\": \"x\", \"text\": {\"format\": {\"type\": \"structural_tag\"}}}"));
        assertEquals(400, e.getResponse().getStatus());
        assertTrue(String.valueOf(e.getResponse().getEntity()).contains("structural_tag requires 'format'"),
                String.valueOf(e.getResponse().getEntity()));
    }

    @Test
    void malformedInputsRefuse() {
        for (String json : new String[]{
                "{}",
                "{\"input\": 3}",
                "{\"input\": [3]}",
                "{\"input\": [null]}",
                "{\"input\": [{\"content\": \"x\"}]}",
                "{\"input\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_file\", \"file_id\": \"f\"}]}]}",
                "{\"input\": [{\"type\": \"item_reference\", \"id\": \"msg_1\"}]}",
                "{\"input\": [{\"type\": \"function_call\", \"name\": \"f\"}]}",
                "{\"input\": \"x\", \"tool_choice\": {\"type\": \"web_search_preview\"}}",
                "{\"input\": \"x\", \"text\": {\"format\": {\"type\": \"yaml\"}}}",
                "{\"input\": \"x\", \"text\": {\"format\": {\"type\": \"json_schema\"}}}",
                "{\"input\": \"x\", \"instructions\": 4}",
                "{\"input\": \"x\", \"extra_params\": [1]}"}) {
            WebApplicationException e = assertThrows(WebApplicationException.class, () -> chat(json), json);
            assertEquals(400, e.getResponse().getStatus(), json);
        }
    }
}
