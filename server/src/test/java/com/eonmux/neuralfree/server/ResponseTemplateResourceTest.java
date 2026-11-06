package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTool;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
import com.eonmux.neuralfree.gguf.chat.response.TemplateParsing;
import com.eonmux.neuralfree.gguf.chat.tools.ToolParse;
import com.eonmux.neuralfree.gguf.chat.tools.ToolRouting;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's {@code hf} parsers over the real engine: a synthetic llama whose GGUF metadata carries a
 * {@code tokenizer.response_template}, served with {@code hf} as the reasoning and tool parser. The reasoning filter
 * and the tool parser come from the template, a request whose tool guarantee the template format cannot enforce is a
 * 400 naming the parameter on the OpenAI and Anthropic routes, and a model a named parser cannot serve fails when it
 * loads. The synthetic weights never write the anchors on their own, so no generation asserts on calls; the parsing
 * itself is covered by the core tests.
 */
class ResponseTemplateResourceTest {

    private static final String RESPONSE_TEMPLATE = "{\"start_anchor\": \"<|turn>model\\n\", \"fields\": {"
            + "\"content\": {\"close\": \"<turn|>\"},"
            + "\"thinking\": {\"open\": \"<think>\", \"close\": \"</think>\"},"
            + "\"tool_calls\": {"
            + "\"open_pattern\": \"<tool>call:(?P<name>\\\\w+)\","
            + "\"close\": \"</tool>\","
            + "\"content\": \"json\","
            + "\"repeats\": true,"
            + "\"transform\": {\"type\": \"function\", \"function\": {\"name\": \"{name}\","
            + " \"arguments\": \"{content}\"}}"
            + "}}}";
    private static final String TOOLS = """
            [{"type": "function", "function": {"name": "get_weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]""";

    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        service = new EngineService(model(RESPONSE_TEMPLATE).toString(), engineConfig("hf", "hf"));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** A synthetic llama carrying {@code responseTemplate} as its metadata, or none when null. */
    private static Path model(String responseTemplate) throws IOException {
        SyntheticModels.Config model = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        GgufTestWriter builder = SyntheticModels.llama(model).str("tokenizer.chat_template", "{{ messages }}");
        if (responseTemplate != null) {
            builder.str(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE, responseTemplate);
        }
        return builder.writeTempFile();
    }

    private static EngineConfig engineConfig(String reasoningParser, String toolParser) {
        return config()
                .memory(MemoryConfig.builder()
                        .nCtx(1024).ubatch(8).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .reasoningParser(reasoningParser)
                .toolParser(toolParser)
                .build();
    }

    private static String chat(String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}], \"max_tokens\": 4,"
                + " \"temperature\": 0, \"tools\": " + extra + "}";
    }

    @Test
    void theReasoningFilterComesFromTheTemplateMetadata() {
        ReasoningFilter filter = service.reasoning().forRequest(Map.of());
        TemplateParsing parsing = filter.responseTemplate();
        assertNotNull(parsing);
        assertEquals("hf", parsing.name());
        assertEquals(List.of("<think>"), filter.startTags());
        assertEquals(List.of("</think>"), filter.endTags());
    }

    @Test
    @SuppressWarnings("unchecked")
    void theToolParserReadsTheTemplateSyntax() {
        ChatTool weather = ChatTool.builder("get_weather")
                .parameters((Map<String, Object>) Json.parse(
                        "{\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}}"))
                .build();
        ChatRequest request = ChatRequest.builder()
                .addMessage(Map.of("role", "user", "content", "Weather?"))
                .tools(List.of(weather))
                .build();
        ToolRouting routing = service.toolRouting(request, null, ParallelToolCalls.PARALLEL);
        assertTrue(routing.active());
        // the transform lifts the name capture and the JSON body into the OpenAI shape
        ToolParse parse = routing.parser().parse("Hi<tool>call:get_weather{\"city\": \"Paris\"}</tool>", false);
        assertEquals("Hi", parse.content());
        assertEquals(1, parse.calls().size());
        assertEquals("get_weather", parse.calls().get(0).name());
        assertEquals("{\"city\": \"Paris\"}", parse.calls().get(0).arguments());
    }

    @Test
    void anAutoToolRequestCompletes() {
        JsonView choice = at(view(new OpenAiResource(service).chatCompletions(chat(TOOLS)).getEntity()), "choices", 0);
        assertTrue(choice.obj("message").has("content") || choice.obj("message").has("tool_calls"),
                "the parsed message answers");
    }

    @Test
    void theUnenforceableGuaranteesAreBadRequestsNamingTheirParameter() {
        OpenAiResource openAi = new OpenAiResource(service);
        String parallel = chat(TOOLS + ", \"parallel_tool_calls\": false");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(parallel)));
        assertTrue(errorOf(() -> openAi.chatCompletions(parallel)).contains("'parallel_tool_calls'"));
        String strictTools = TOOLS.replace("\"name\": \"get_weather\",", "\"name\": \"get_weather\", \"strict\": true,");
        assertTrue(errorOf(() -> openAi.chatCompletions(chat(strictTools))).contains("strict tools"));
        String required = chat(TOOLS + ", \"tool_choice\": \"required\"");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(required)));
        assertTrue(errorOf(() -> openAi.chatCompletions(required)).contains("'tool_choice'"));
        // tool_choice none asks for no guarantee
        JsonView none = view(openAi.chatCompletions(chat(strictTools + ", \"tool_choice\": \"none\"")).getEntity());
        assertEquals(1, ((List<?>) none.raw("choices")).size());

        String messages = "{\"model\": \"m\", \"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hi\"}], \"tools\": [{\"name\": \"f\", \"input_schema\": {\"type\": \"object\"}}],"
                + " \"tool_choice\": {\"type\": \"any\"}}";
        AnthropicResource anthropic = new AnthropicResource(service);
        assertEquals(400, statusOf(() -> anthropic.messages(messages)));
        assertTrue(errorOf(() -> anthropic.messages(messages)).contains("required tool choice"));
    }

    @Test
    void anExplicitParallelToolCallsWinsOverTheOneCallTemplateDefault() {
        // llama.cpp serves the request's parallel_tool_calls over the template's
        // supports_parallel_tool_calls (server-common.cpp:1447): explicit true asks for nothing
        OpenAiResource openAi = new OpenAiResource(service);
        JsonView choice = at(view(openAi.chatCompletions(chat(TOOLS + ", \"parallel_tool_calls\": true"))
                .getEntity()), "choices", 0);
        assertTrue(choice.obj("message").has("content") || choice.obj("message").has("tool_calls"));
    }

    @Test
    void theAnthropicDisableParallelToolUseIsTheExplicitFalse() {
        // vLLM's Anthropic mapping serves not disable_parallel_tool_use: the switch is the explicit
        // parallel_tool_calls false, so hf refuses it; the switch off or absent asks for nothing
        AnthropicResource anthropic = new AnthropicResource(service);
        String disabled = "{\"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"tools\": [{\"name\": \"f\", \"input_schema\": {\"type\": \"object\"}}],"
                + " \"tool_choice\": {\"type\": \"auto\", \"disable_parallel_tool_use\": true}}";
        assertEquals(400, statusOf(() -> anthropic.messages(disabled)));
        assertTrue(errorOf(() -> anthropic.messages(disabled)).contains("'parallel_tool_calls'"));
        for (String choice : new String[]{"{\"type\": \"auto\", \"disable_parallel_tool_use\": false}",
                "{\"type\": \"auto\"}"}) {
            String body = "{\"max_tokens\": 4, \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                    + " \"tools\": [{\"name\": \"f\", \"input_schema\": {\"type\": \"object\"}}],"
                    + " \"tool_choice\": " + choice + "}";
            assertEquals(200, statusOf(() -> anthropic.messages(body)));
        }
    }

    @Test
    void aModelANamedParserCannotServeFailsWhenItLoads() throws IOException {
        // vLLM test_parser_manager_validates_checkpoint_metadata_at_startup
        Path bare = model(null);
        IllegalArgumentException tool = assertThrows(IllegalArgumentException.class,
                () -> new EngineService(bare.toString(), engineConfig("auto", "hf")).close());
        assertTrue(tool.getMessage().contains("--tool-call-parser hf requires `response_template`"), tool.getMessage());
        IllegalArgumentException reasoning = assertThrows(IllegalArgumentException.class,
                () -> new EngineService(bare.toString(), engineConfig("hf", "auto")).close());
        assertTrue(reasoning.getMessage().contains(GgufKeys.TOKENIZER_RESPONSE_TEMPLATE), reasoning.getMessage());
        // a template without tool_calls serves the reasoning parser, not the tool parser
        Path toolless = model("{\"start_anchor\": \"<|turn>model\\n\", \"fields\": {"
                + "\"content\": {\"close\": \"<turn|>\"},"
                + "\"thinking\": {\"open\": \"<think>\", \"close\": \"</think>\"}}}");
        IllegalArgumentException noCalls = assertThrows(IllegalArgumentException.class,
                () -> new EngineService(toolless.toString(), engineConfig("hf", "hf")).close());
        assertTrue(noCalls.getMessage().contains("with a tool_calls field"), noCalls.getMessage());
        try (EngineService reasoningOnly = new EngineService(toolless.toString(), engineConfig("hf", "none"))) {
            assertEquals(List.of("<think>"), reasoningOnly.reasoning().forRequest(Map.of()).startTags());
        }
    }
}
