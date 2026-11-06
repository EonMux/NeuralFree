package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Cohere v2 chat body's translation into the OpenAI chat body {@link ChatPlan} reads, against vLLM's
 * {@code _convert_v2_to_chat_completion}: roles and content blocks, tools and their choice, documents, response
 * formats, the thinking switch and the sampling names, and the refusals that name the unsupported field.
 */
class CohereChatRequestTest {

    private static Map<String, Object> convert(String json) {
        return CohereChatRequest.of(ServerJson.parse(json)).openAiBody();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> firstMessage(Map<String, Object> body) {
        return (Map<String, Object>) ((List<?>) body.get("messages")).get(0);
    }

    @Test
    void systemAndDeveloperBecomeSystemMessages() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": ["
                + "{\"role\": \"system\", \"content\": [{\"type\": \"text\", \"text\": \"be \"},"
                + " {\"type\": \"text\", \"text\": \"helpful\"}]},"
                + "{\"role\": \"developer\", \"content\": \"terse\"},"
                + "{\"role\": \"user\", \"content\": \"hi\"}]}");
        List<?> messages = (List<?>) body.get("messages");
        assertEquals(Map.of("role", "system", "content", "be helpful"), messages.get(0));
        assertEquals(Map.of("role", "system", "content", "terse"), messages.get(1),
                "Cohere's v2 union has no developer role; vLLM aliases it onto system");
        assertEquals(Map.of("role", "user", "content", "hi"), messages.get(2));
    }

    @Test
    void userContentKeepsImagePartsAndFlattensASingleTextPart() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": ["
                + "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"look\"},"
                + " {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://x/i.png\", \"detail\": \"high\"}}]}]}");
        List<?> content = (List<?>) firstMessage(body).get("content");
        assertEquals(Map.of("type", "text", "text", "look"), content.get(0));
        @SuppressWarnings("unchecked")
        Map<String, Object> image = (Map<String, Object>) content.get(1);
        assertEquals("image_url", image.get("type"));
        assertEquals(Map.of("url", "https://x/i.png", "detail", "high"), image.get("image_url"));
        assertEquals("only text", firstMessage(convert("{\"model\": \"m\", \"messages\": ["
                + "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"only text\"}]}]}")).get("content"));
    }

    @Test
    void assistantThinkingAndToolPlanBecomeReasoningAndCallsKeepTheirIds() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": ["
                + "{\"role\": \"assistant\", \"content\": [{\"type\": \"thinking\", \"thinking\": \"hmm\"},"
                + " {\"type\": \"text\", \"text\": \"answer\"}], \"tool_plan\": \"plan\","
                + " \"tool_calls\": [{\"id\": \"c1\", \"type\": \"function\","
                + " \"function\": {\"name\": \"f\", \"arguments\": \"{\\\"a\\\": 1}\"}}]}]}");
        Map<String, Object> message = firstMessage(body);
        assertEquals("assistant", message.get("role"));
        assertEquals("answer", message.get("content"));
        assertEquals("hmmplan", message.get("reasoning"));
        Map<?, ?> call = (Map<?, ?>) ((List<?>) message.get("tool_calls")).get(0);
        assertEquals("c1", call.get("id"));
        assertEquals("function", call.get("type"));
        assertEquals(Map.of("name", "f", "arguments", "{\"a\": 1}"), call.get("function"));
    }

    @Test
    void toolContentDocumentsFlattenToJsonTextParts() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": [{\"role\": \"tool\","
                + " \"tool_call_id\": \"c1\", \"content\": [{\"type\": \"document\","
                + " \"document\": {\"id\": \"d1\", \"data\": {\"text\": \"Paris\"}}},"
                + " {\"type\": \"text\", \"text\": \"the capital\"}]}]}");
        Map<String, Object> message = firstMessage(body);
        assertEquals("tool", message.get("role"));
        assertEquals("c1", message.get("tool_call_id"));
        List<?> content = (List<?>) message.get("content");
        assertEquals(2, content.size());
        Map<?, ?> document = (Map<?, ?>) content.get(0);
        assertEquals("text", document.get("type"));
        assertTrue(String.valueOf(document.get("text")).contains("\"id\":\"d1\""), "the document is JSON text");
        assertEquals(Map.of("type", "text", "text", "the capital"), content.get(1));
    }

    @Test
    void toolsCarryStrictAndTheChoiceMapsOntoOpenAi() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"strict_tools\": true, \"tool_choice\": \"REQUIRED\","
                + " \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"f\","
                + " \"parameters\": {\"type\": \"object\"}}}]}");
        assertEquals("required", body.get("tool_choice"));
        Map<?, ?> tool = (Map<?, ?>) ((List<?>) body.get("tools")).get(0);
        assertEquals("function", tool.get("type"));
        Map<?, ?> fn = (Map<?, ?>) tool.get("function");
        assertEquals("f", fn.get("name"));
        assertEquals(true, fn.get("strict"));
        assertEquals("auto", convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"f\"}}]}").get("tool_choice"));
        assertEquals("none", convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"tool_choice\": \"NONE\"}").get("tool_choice"));
    }

    @Test
    void documentsAndResponseFormatsMapToOpenAiForms() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"documents\": [\"a fact\", {\"id\": \"d2\", \"data\":"
                + " {\"title\": \"t\"}}], \"response_format\": {\"type\": \"json_schema\","
                + " \"json_schema\": {\"type\": \"object\"}}}");
        List<?> documents = (List<?>) body.get("documents");
        assertEquals(Map.of("id", "doc_0", "data", Map.of("text", "a fact")), documents.get(0));
        assertEquals(Map.of("id", "d2", "data", Map.of("title", "t")), documents.get(1));
        Map<?, ?> format = (Map<?, ?>) body.get("response_format");
        assertEquals("json_schema", format.get("type"));
        Map<?, ?> spec = (Map<?, ?>) format.get("json_schema");
        assertEquals("cohere_v2_json_schema", spec.get("name"));
        assertEquals(Map.of("type", "object"), spec.get("schema"));
        assertEquals(Map.of("type", "json_object"), convert("{\"model\": \"m\", \"messages\":"
                + " [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"response_format\": {\"type\": \"json_object\"}}").get("response_format"));
        assertNull(convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"response_format\": {\"type\": \"text\"}}").get("response_format"));
    }

    @Test
    void thinkingAndSamplingFieldsMapByMeaning() {
        Map<String, Object> body = convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"thinking\": {\"type\": \"enabled\", \"token_budget\": 64},"
                + " \"k\": 20, \"p\": 0.9, \"temperature\": 0.5, \"seed\": 7, \"max_tokens\": 12,"
                + " \"stop_sequences\": [\"END\"], \"frequency_penalty\": 0.1, \"presence_penalty\": 0.2,"
                + " \"priority\": 3, \"logprobs\": true}");
        assertEquals(true, body.get("enable_thinking"));
        assertEquals(64L, body.get("thinking_token_budget"));
        assertEquals(20L, body.get("top_k"));
        assertEquals(0.9, body.get("top_p"));
        assertEquals(0.5, body.get("temperature"));
        assertEquals(7L, body.get("seed"));
        assertEquals(12L, body.get("max_tokens"));
        assertEquals(List.of("END"), body.get("stop"));
        assertEquals(3L, body.get("priority"));
        assertEquals(true, body.get("logprobs"));
        assertEquals(false, convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"thinking\": {\"type\": \"disabled\"}}").get("enable_thinking"));
    }

    @Test
    void unsupportedFieldsRefuseByName() {
        assertTrue(errorOf(() -> convert("{\"model\": \"m\"}")).contains("messages"));
        assertTrue(errorOf(() -> convert("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}"))
                .contains("model"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": []}")).contains("messages"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"safety_mode\": \"STRICT\"}")).contains("safety_mode"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"citation_options\": {}}")).contains("citation_options"));
        assertEquals(Map.of("a", 1L), convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"kv_transfer_params\": {\"a\": 1}}").get("kv_transfer_params"),
                "the transfer parameters pass through unchanged");
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"kv_transfer_params\": \"rank\"}")).contains("kv_transfer_params"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"tool_choice\": \"REQUIRED\"}")).contains("REQUIRED"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"tool_choice\": \"AUTO\"}")).contains("tool_choice"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"critic\","
                + " \"content\": \"hi\"}]}")).contains("role"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": [{\"type\": \"audio\"}]}]}")).contains("text or image_url"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"unknown\": 1}")).contains("unknown field 'unknown'"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"assistant\","
                + " \"content\": \"hi\", \"citations\": [{\"start\": 0}]}]}")).contains("citations"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"response_format\": {\"type\": \"xml\"}}"))
                .contains("response_format.type"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"thinking\": {\"type\": \"maybe\"}}")).contains("thinking.type"));
        assertEquals(true, convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"watermarking\": true, \"kv_transfer_params\": {}}").get("watermarking"),
                "the watermarking switch reaches the sampling parameters");
        assertEquals(false, convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"watermarking\": false}").get("watermarking"));
        assertTrue(errorOf(() -> convert("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}], \"watermarking\": \"yes\"}")).contains("watermarking"));
    }

    @Test
    void streamAndLogprobsAreReadFromTheRequest() {
        CohereChatRequest request = CohereChatRequest.of(ServerJson.parse("{\"model\": \"m\","
                + " \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"stream\": true,"
                + " \"logprobs\": true}"));
        assertTrue(request.streamed());
        assertTrue(request.logprobs());
        JsonView body = ServerJson.parse("{\"model\": \"m\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hi\"}]}");
        CohereChatRequest plain = CohereChatRequest.of(body);
        assertFalse(plain.streamed());
        assertFalse(plain.logprobs());
    }
}
