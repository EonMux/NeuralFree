package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ParallelToolCalls;
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

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The chat template's capabilities over the serving endpoints, as llama-server exposes and uses them: {@code /props}
 * reports them as {@code chat_template_caps}, {@code /apply-template} renders the messages in the forms the
 * template reads (string content as a text part for a template that iterates parts, tool-call arguments as JSON
 * text for one that prints them) and reports the rendering template's caps, a template that renders only one call
 * per message makes {@code parallel_tool_calls} default to false, and a request's {@code chat_template} override
 * answers to its own caps.
 */
public class ChatTemplateCapsResourceTest {

    /** Iterates content parts (no string check) and prints only the first tool call's arguments as text. */
    private static final String TEMPLATE = "{%- for m in messages %}<|im_start|>{{ m.role }}\n"
            + "{% for p in m.content %}{{ p.text }}{% endfor %}"
            + "{%- if m.tool_calls %}[{{ m.tool_calls[0].function.name }} {{ m.tool_calls[0].function.arguments }}]"
            + "{%- endif %}<|im_end|>\n{% endfor %}"
            + "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    private static EngineService service;
    private static LlamaCppResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama()).str("tokenizer.chat_template", TEMPLATE)
                .writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 1),
                ServerOptions.defaults().withTrustRequestChatTemplate(true));
        llama = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void propsReportTheTemplateCapabilities() {
        JsonView caps = JsonView.of(view(llama.props(null, null).getEntity()).raw("chat_template_caps"), "chat_template_caps");
        assertEquals(Boolean.FALSE, caps.raw("supports_string_content"));
        assertEquals(Boolean.TRUE, caps.raw("supports_typed_content"));
        assertEquals(Boolean.TRUE, caps.raw("supports_tool_calls"));
        assertEquals(Boolean.FALSE, caps.raw("supports_parallel_tool_calls"));
        assertEquals(Boolean.FALSE, caps.raw("supports_object_arguments"));
        assertEquals(Boolean.FALSE, caps.raw("supports_tools"));
        assertEquals(Boolean.TRUE, caps.raw("supports_system_role"));
    }

    @Test
    void applyTemplateRendersTheFormsTheTemplateReads() {
        String body = """
                {"messages": [{"role": "user", "content": "Weather?"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}],
                 "add_generation_prompt": false}""";
        String prompt = view(llama.applyTemplate(body).getEntity()).str("prompt");
        assertEquals("<|im_start|>user\nWeather?<|im_end|>\n"
                + "<|im_start|>assistant\n[get_weather {\"city\":\"Paris\"}]<|im_end|>\n", prompt);
    }

    @Test
    void malformedToolCallArgumentsAreRefusedWith400() {
        String body = """
                {"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{oops"}}]}]}""";
        assertEquals(400, statusOf(() -> llama.applyTemplate(body)));
        assertTrue(errorOf(() -> llama.applyTemplate(body)).contains("not valid JSON"));
    }

    @Test
    void aSingleCallTemplateDefaultsToOneToolCall() {
        assertFalse(service.defaultParallelToolCalls());
        assertNull(OpenAiRequests.parallelToolCalls(JsonView.of(Json.parse("{}"), "body")),
                "an absent parallel_tool_calls takes the template default");
        assertEquals(Boolean.TRUE, OpenAiRequests.parallelToolCalls(JsonView.of(
                Json.parse("{\"parallel_tool_calls\": true}"), "body")), "an explicit parallel_tool_calls wins");
        assertEquals(Boolean.TRUE, AnthropicRequests.parallelToolCalls(JsonView.of(Json.parse(
                "{\"tool_choice\": {\"type\": \"auto\", \"disable_parallel_tool_use\": false}}"), "body")));
        assertEquals(Boolean.FALSE, AnthropicRequests.parallelToolCalls(JsonView.of(Json.parse(
                "{\"tool_choice\": {\"type\": \"auto\", \"disable_parallel_tool_use\": true}}"), "body")));
        assertNull(AnthropicRequests.parallelToolCalls(JsonView.of(Json.parse("{}"), "body")));
        // the template default serves one call, and it asks for no guarantee: only an explicit false refuses
        ParallelToolCalls served = ParallelToolCalls.of(
                OpenAiRequests.parallelToolCalls(JsonView.of(Json.parse("{}"), "body")),
                service.defaultParallelToolCalls());
        assertFalse(served.allowed());
        assertFalse(served.singleAsked());
    }

    /** Renders every call of a message, so its parallel probe succeeds. */
    private static final String PARALLEL_TEMPLATE =
            "{% for m in messages %}{% for tc in m.tool_calls %}[{{ tc.function.name }}]{% endfor %}{% endfor %}";

    @Test
    void anOverrideCapsShapeTheRequestParallelDefaultAndItsReport() {
        ChatRequest override = ChatRequest.builder()
                .messages(List.of(Map.of("role", "user", "content", "x")))
                .chatTemplate(PARALLEL_TEMPLATE).build();
        assertTrue(service.defaultParallelToolCalls(override), "the override's probe allows several calls");
        assertTrue(service.chatTemplateCaps(override).supportsParallelToolCalls());
        assertTrue(service.chatTemplateCaps(override).supportsStringContent(),
                "the override reads string content, which the model template does not");
        assertFalse(service.defaultParallelToolCalls(), "the model template keeps its one-call default");
        assertFalse(service.chatTemplateCaps(ChatRequest.builder()
                .messages(List.of(Map.of("role", "user", "content", "x"))).build())
                .supportsParallelToolCalls(), "the model template renders no tool calls");
    }

    @Test
    void applyTemplateReportsTheRenderingTemplatesCaps() {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}],"
                + " \"add_generation_prompt\": false}";
        JsonView out = view(llama.applyTemplate(body).getEntity());
        assertEquals("<|im_start|>user\nWeather?<|im_end|>\n", out.str("prompt"));
        JsonView caps = JsonView.of(out.raw("chat_template_caps"), "chat_template_caps");
        assertEquals(Boolean.FALSE, caps.raw("supports_string_content"), "the model template's own caps");
        assertEquals(Boolean.FALSE, caps.raw("supports_parallel_tool_calls"));
        String overridden = "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}],"
                + " \"chat_template\": " + ServerJson.write(PARALLEL_TEMPLATE) + "}";
        JsonView own = JsonView.of(view(llama.applyTemplate(overridden).getEntity()).raw("chat_template_caps"),
                "chat_template_caps");
        assertEquals(Boolean.TRUE, own.raw("supports_parallel_tool_calls"), "a request's override answers for itself");
        assertEquals(Boolean.TRUE, own.raw("supports_tool_calls"));
        assertEquals(Boolean.FALSE, own.raw("supports_typed_content"));
    }
}
