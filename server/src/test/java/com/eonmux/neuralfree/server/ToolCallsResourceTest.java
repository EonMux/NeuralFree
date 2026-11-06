package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tool calls end to end over the synthetic llama carrying a hermes-style template (the {@code hermes} parser is
 * detected from it). Its random weights never call a tool on their own, so the forcing tool choices carry the
 * tests: {@code required} and a named function constrain the output with their grammars, and the parsed calls
 * come back as OpenAI {@code tool_calls} (streamed as deltas) and as Anthropic {@code tool_use} blocks.
 */
public class ToolCallsResourceTest {

    private static final String TEMPLATE = "{%- if tools %}<|im_start|>system\nTools:"
            + "{% for t in tools %} {{ t.function.name }}{% endfor %}. Call one inside <tool_call></tool_call>."
            + "<|im_end|>\n{% endif %}"
            + "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
            + "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static final String TOOLS = """
            [{"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
                             "required": ["city"], "additionalProperties": false}}},
             {"type": "function", "function": {"name": "get_time",
              "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}}]""";

    /**
     * Enough tokens for any output of the forcing grammars: llama.cpp's JSON {@code space} rule admits up to 22
     * whitespace characters per slot, which a random-weight model may spend.
     */
    private static final int FORCED_BUDGET = 600;

    private static EngineService service;
    private static OpenAiResource openAi;
    private static AnthropicResource anthropic;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2));
        openAi = new OpenAiResource(service);
        anthropic = new AnthropicResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static String chat(String extra) {
        return chat(FORCED_BUDGET, extra);
    }

    private static String chat(int maxTokens, String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather in Paris?\"}], \"max_tokens\": "
                + maxTokens + ", \"temperature\": 0, \"tools\": " + TOOLS + ", " + extra + "}";
    }

    @Test
    void aNamedFunctionIsCalledWithSchemaValidArguments() {
        JsonView choice = at(view(openAi.chatCompletions(chat("""
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}""")).getEntity()),
                "choices", 0);
        assertEquals("stop", choice.str("finish_reason"), "a named choice finishes with stop, as OpenAI's does");
        JsonView message = choice.obj("message");
        assertNull(message.raw("content"), "the whole output was the call");
        assertEquals(1, size(message, "tool_calls"));
        JsonView call = at(message, "tool_calls", 0);
        assertEquals("function", call.str("type"));
        assertTrue(call.str("id").startsWith("call_"));
        JsonView fn = call.obj("function");
        assertEquals("get_weather", fn.str("name"));
        JsonView args = JsonView.of(Json.parse(fn.str("arguments"), Json.Limits.UNTRUSTED), "arguments");
        assertTrue(List.of("Paris", "Rome").contains(args.str("city")), "the grammar kept the enum");
    }

    @Test
    void requiredCallsADeclaredToolAndFinishesWithToolCalls() {
        JsonView choice = at(view(openAi.chatCompletions(chat("""
                "tool_choice": "required", "parallel_tool_calls": false""")).getEntity()), "choices", 0);
        assertEquals("tool_calls", choice.str("finish_reason"));
        JsonView message = choice.obj("message");
        assertEquals(1, size(message, "tool_calls"), "parallel_tool_calls false allows one call");
        String name = at(message, "tool_calls", 0).obj("function").str("name");
        assertTrue(name.equals("get_weather") || name.equals("get_time"), "a declared tool: " + name);
    }

    @Test
    void theStreamedCallDeltasAssembleTheCall() throws IOException {
        List<JsonView> frames = dataFrames(streamText(openAi.chatCompletions(chat("""
                "stream": true, "tool_choice": {"type": "function", "function": {"name": "get_weather"}}"""))));
        StringBuilder arguments = new StringBuilder();
        String name = null;
        String id = null;
        String finish = null;
        for (JsonView frame : frames) {
            JsonView choice = at(frame, "choices", 0);
            if (choice.raw("finish_reason") != null) {
                finish = choice.str("finish_reason");
            }
            JsonView delta = choice.obj("delta");
            if (!delta.has("tool_calls")) {
                continue;
            }
            JsonView call = at(delta, "tool_calls", 0);
            assertEquals(0L, call.i64("index"));
            if (call.has("id")) {
                assertNull(id, "the call opens once");
                id = call.str("id");
                name = call.obj("function").str("name");
            }
            arguments.append(call.obj("function").str("arguments"));
        }
        assertEquals("get_weather", name);
        assertEquals("stop", finish);
        Object args = Json.parse(arguments.toString(), Json.Limits.UNTRUSTED);
        assertTrue(args instanceof Map<?, ?> m && m.containsKey("city"), "the deltas join into the arguments");
    }

    @Test
    void toolChoiceNoneRendersTheToolsButParsesNothing() {
        JsonView message = at(view(openAi.chatCompletions(chat(4, "\"tool_choice\": \"none\""))
                .getEntity()), "choices", 0).obj("message");
        assertNull(message.raw("tool_calls"));
    }

    @Test
    void toolChoiceRefusalsNameTheProblem() {
        assertTrue(errorOf(() -> openAi.chatCompletions(chat("""
                "tool_choice": {"type": "function", "function": {"name": "get_stock"}}"""))).contains("get_stock"));
        assertEquals(400, statusOf(() -> openAi.chatCompletions(chat("""
                "tool_choice": "required", "guided_choice": ["a"]"""))), "a forcing choice and a constraint");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(chat("\"tool_choice\": \"sometimes\""))));
        assertEquals(400, statusOf(() -> openAi.chatCompletions("""
                {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "retrieval"}]}""")));
    }

    @Test
    void historyCallsAndResultsRender() {
        String body = """
                {"messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Rome\\"}"}}]},
                  {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}],
                 "max_tokens": 2, "temperature": 0, "tools": %s}""".formatted(TOOLS);
        try {
            assertEquals(200, openAi.chatCompletions(body).getStatus());
        } catch (WebApplicationException e) {
            throw new AssertionError("history refused: " + e.getResponse().getEntity(), e);
        }
    }

    // ------------------------------------------------------------------ Anthropic

    private static String message(String extra) {
        return """
                {"model": "test", "max_tokens": %d, "temperature": 0,
                 "messages": [{"role": "user", "content": "Weather in Paris?"}],
                 "tools": [{"name": "get_weather", "description": "Weather of a city",
                            "input_schema": {"type": "object",
                                             "properties": {"city": {"type": "string", "enum": ["Paris", "Rome"]}},
                                             "required": ["city"], "additionalProperties": false}}],
                 %s}""".formatted(FORCED_BUDGET, extra);
    }

    @Test
    void anthropicAnswersAForcedToolWithAToolUseBlock() {
        JsonView body = view(anthropic.messages(message("""
                "tool_choice": {"type": "tool", "name": "get_weather"}""")).getEntity());
        assertEquals("tool_use", body.str("stop_reason"));
        JsonView use = null;
        for (int i = 0; i < size(body, "content"); i++) {
            JsonView block = at(body, "content", i);
            if (block.str("type").equals("tool_use")) {
                use = block;
            }
        }
        assertTrue(use != null, "a tool_use block");
        assertTrue(use.str("id").startsWith("toolu_"));
        assertEquals("get_weather", use.str("name"));
        assertTrue(List.of("Paris", "Rome").contains(use.obj("input").str("city")));
    }

    @Test
    void anthropicStreamsTheToolUseAsInputJsonDeltas() throws IOException {
        String events = streamText(anthropic.messages(message("""
                "stream": true, "tool_choice": {"type": "any"}""")));
        assertTrue(events.contains("\"type\":\"tool_use\""), "a tool_use block opens: " + events);
        assertTrue(events.contains("\"type\":\"input_json_delta\""), "its input streams as JSON deltas");
        assertTrue(events.contains("\"stop_reason\":\"tool_use\""));
        assertTrue(events.endsWith("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
    }

    @Test
    void anthropicHistoryMapsToolUseAndToolResult() {
        String body = """
                {"model": "test", "max_tokens": 2,
                 "messages": [{"role": "user", "content": "Weather?"},
                   {"role": "assistant", "content": [{"type": "thinking", "thinking": "check"},
                     {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Rome"}}]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [{"type": "text", "text": "sunny"}]}, {"type": "text", "text": "And?"}]}],
                 "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}""";
        try {
            assertEquals(200, anthropic.messages(body).getStatus());
        } catch (WebApplicationException e) {
            throw new AssertionError("history refused: " + e.getResponse().getEntity(), e);
        }
        assertTrue(errorOf(() -> anthropic.messages("""
                {"model": "test", "max_tokens": 2, "messages": [{"role": "user", "content": "x"}],
                 "tools": [{"type": "web_search_20250305", "name": "web_search"}]}"""))
                .contains("web_search_20250305"), "server tools are refused by name");
    }
}
