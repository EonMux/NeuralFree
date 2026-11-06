package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatMessages;
import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.grammar.ConstraintSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The autoparser's wiring into a chat request: a structured-output request is constrained in the autoparsed
 * template's own markers, the template's literal stops are appended, and a continued final message is written
 * around the derived reasoning markers.
 */
class AutoparserChatPlanTest {

    private static final String TEMPLATE =
            "{%- for message in messages -%}"
            + "<|{{ message['role'] }}|>{{ message['content'] }}"
            + "{%- if message.get('reasoning_content') -%}<|think|>{{ message['reasoning_content'] }}"
            + "<|/think|>{%- endif -%}<|end|>"
            + "{%- endfor -%}"
            + "{%- if add_generation_prompt -%}<|assistant|><|think|>{%- endif -%}";

    private static String savedTemplate;
    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        savedTemplate = System.getProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, "{# laguna_glm_thinking #}" + TEMPLATE);
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
    }

    @AfterAll
    static void close() {
        service.close();
        if (savedTemplate == null) {
            System.clearProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        } else {
            System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, savedTemplate);
        }
    }

    private static ChatPlan plan(String body) {
        return ChatPlan.of(service, ServerTestSupport.view(body));
    }

    @Test
    void aJsonSchemaResponseFormatUsesTheTemplateGrammar() {
        ChatPlan plan = plan("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"x\","
                + " \"schema\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}},"
                + " \"required\": [\"city\"], \"additionalProperties\": false}}}}");
        ConstraintSpec constraint = plan.constraint();
        assertEquals(ConstraintSpec.Kind.GBNF, constraint.kind());
        assertTrue(constraint.text().contains("```json"), constraint.text());
        assertTrue(constraint.text().contains("city"), "the schema's field is in the grammar");
    }

    @Test
    void aJsonObjectResponseFormatUsesTheTemplateGrammar() {
        ChatPlan plan = plan("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"response_format\": {\"type\": \"json_object\"}}");
        assertEquals(ConstraintSpec.Kind.GBNF, plan.constraint().kind());
    }

    @Test
    void theTemplatesLiteralStopIsAppended() {
        ChatPlan plan = plan("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"temperature\": 0}");
        assertTrue(plan.base().stopStrings().contains("</assistant>"));
    }

    @Test
    void aReasoningOnlyContinuationKeepsTheBlockOpen() {
        Map<String, Object> user = message("user", "hi");
        Map<String, Object> assistant = message("assistant", "");
        assistant.put(ChatMessages.REASONING_CONTENT, "abc");
        ChatRequest request = ChatRequest.builder().addMessage(user).addMessage(assistant)
                .continueFinalMessage(true).build();
        String prompt = service.formatChat(request);
        assertTrue(prompt.endsWith("<|assistant|><|think|>abc"), prompt);
        assertFalse(prompt.contains("abc<|/think|>"), prompt);
    }

    private static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
