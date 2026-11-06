package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateKwargs;
import com.eonmux.neuralfree.gguf.chat.ChatTemplateSource;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Server-default chat template variables (llama.cpp's {@code --chat-template-kwargs}, vLLM's
 * {@code --default-chat-template-kwargs}): every chat a request renders binds them unless the request sets the
 * variable itself, through {@code chat_template_kwargs} or the top-level {@code enable_thinking}; a request kwarg
 * that is null counts as unset. The server here renders with a probe template (the {@code --chat-template}
 * property) that prints the variables, over the real engine and a synthetic model.
 */
class DefaultChatTemplateKwargsTest {

    private static final String PROBE = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
            + "[{{ style | default('none') }}|{% if enable_thinking is defined %}"
            + "{{ 'on' if enable_thinking else 'off' }}{% else %}unset{% endif %}]";
    private static final String MESSAGES = "\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]";

    private static String savedTemplate;
    private static EngineService service;
    private static OpenAiResource openAi;
    private static LlamaCppResource llama;

    @BeforeAll
    static void open() throws IOException {
        savedTemplate = System.getProperty(ChatTemplateSource.OVERRIDE_PROPERTY);
        System.setProperty(ChatTemplateSource.OVERRIDE_PROPERTY, PROBE);
        Map<String, Object> defaults = ChatTemplateKwargs.parse(
                "{\"style\": \"terse\", \"enable_thinking\": false}", "test");
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 1),
                ServerOptions.defaults().withChatTemplateKwargs(defaults));
        openAi = new OpenAiResource(service);
        llama = new LlamaCppResource(service);
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

    private static String prompt(String extra) {
        return view(llama.applyTemplate("{" + MESSAGES + extra + "}").getEntity()).str("prompt");
    }

    private static long inputTokens(String extra) {
        Object count = view(openAi.chatInputTokens("{" + MESSAGES + extra + "}").getEntity()).raw("input_tokens");
        return ((Number) count).longValue();
    }

    @Test
    void theDefaultsRenderWhenTheRequestLeavesThemUnset() {
        assertEquals("<user>hi[terse|off]", prompt(""));
    }

    @Test
    void theRequestsOwnVariablesWin() {
        assertEquals("<user>hi[long|off]", prompt(", \"chat_template_kwargs\": {\"style\": \"long\"}"));
        assertEquals("<user>hi[terse|on]", prompt(", \"chat_template_kwargs\": {\"enable_thinking\": true}"));
        assertEquals("<user>hi[terse|on]", prompt(", \"enable_thinking\": true"));
        assertEquals("<user>hi[terse|off]", prompt(", \"chat_template_kwargs\": {\"style\": null}"),
                "a null kwarg is unset, as vLLM's merge treats it");
    }

    @Test
    void theChatRoutesRenderWithTheDefaults() {
        long withDefaults = inputTokens("");
        assertEquals(withDefaults,
                inputTokens(", \"chat_template_kwargs\": {\"style\": \"terse\", \"enable_thinking\": false}"),
                "the same variables given explicitly render the same prompt");
        assertNotEquals(withDefaults,
                inputTokens(", \"chat_template_kwargs\": {\"style\": \"a considerably longer style value\"}"));
        String tokenized = "{" + MESSAGES + "}";
        assertEquals(withDefaults, ServerTestSupport.size(view(llama.tokenize(tokenized).getEntity()), "tokens"),
                "/tokenize with messages renders the same prompt");
    }

    @Test
    void anthropicRequestsTakeTheDefaultsUnderTheirThinkingSwitch() {
        ChatRequest plain = AnthropicRequests.chat(
                ServerTestSupport.view("{" + MESSAGES + ", \"max_tokens\": 1}"), service.options());
        assertEquals("terse", plain.templateKwargs().get("style"));
        assertEquals(Boolean.FALSE, plain.templateVariables().get("enable_thinking"));
        ChatRequest thinking = AnthropicRequests.chat(ServerTestSupport.view("{" + MESSAGES
                + ", \"max_tokens\": 64, \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": 16}}"),
                service.options());
        assertEquals(Boolean.TRUE, thinking.templateVariables().get("enable_thinking"),
                "the request's thinking switch wins over the default");
    }

    @Test
    void theOptionsCarryTheDefaultsThroughEveryCopy() {
        assertTrue(ServerOptions.defaults().chatTemplateKwargs().isEmpty());
        Map<String, Object> kwargs = new LinkedHashMap<>();
        kwargs.put("style", null);
        ServerOptions options = ServerOptions.defaults().withChatTemplateKwargs(kwargs);
        kwargs.put("later", 1);
        assertEquals(1, options.chatTemplateKwargs().size(), "the options copy the map");
        assertTrue(options.chatTemplateKwargs().containsKey("style"), "null values are kept");
        assertThrows(UnsupportedOperationException.class, () -> options.chatTemplateKwargs().put("x", 1));
        ServerOptions copied = options.withResponseStore(4).withTrustRequestChatTemplate(true)
                .withDataParallel(DpLayout.internal(1));
        assertEquals(options.chatTemplateKwargs(), copied.chatTemplateKwargs());
    }

    @Test
    void thePropertyConfiguresTheServerAndAMalformedOneIsRefused() {
        String saved = System.getProperty(ChatTemplateKwargs.PROPERTY);
        try {
            System.setProperty(ChatTemplateKwargs.PROPERTY, "{\"enable_thinking\": false}");
            assertEquals(Map.of("enable_thinking", false), ServerOptions.fromSystemProperties().chatTemplateKwargs());
            System.setProperty(ChatTemplateKwargs.PROPERTY, "[false]");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    ServerOptions::fromSystemProperties);
            assertTrue(e.getMessage().startsWith(ChatTemplateKwargs.PROPERTY), e.getMessage());
        } finally {
            if (saved == null) {
                System.clearProperty(ChatTemplateKwargs.PROPERTY);
            } else {
                System.setProperty(ChatTemplateKwargs.PROPERTY, saved);
            }
        }
    }

    @Test
    void bothLaunchFlagsWriteThePropertyAndRepeatedFlagsMerge() {
        Map<String, String> properties = ServerLaunchArgs.parse("--chat-template-kwargs",
                "{\"enable_thinking\": false, \"style\": \"a\"}", "--default-chat-template-kwargs",
                "{\"style\": \"b\", \"n\": [1, 2]}").properties();
        String json = properties.get(ChatTemplateKwargs.PROPERTY);
        Map<String, Object> merged = ChatTemplateKwargs.parse(json, "property");
        assertEquals(Map.of("enable_thinking", false, "style", "b", "n", List.of(1L, 2L)), merged);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.parse("--default-chat-template-kwargs", "true"));
        assertTrue(e.getMessage().startsWith("--default-chat-template-kwargs must be a JSON object"), e.getMessage());
        e = assertThrows(IllegalArgumentException.class, () -> ServerLaunchArgs.parse("--chat-template-kwargs"));
        assertTrue(e.getMessage().contains("--chat-template-kwargs"), e.getMessage());
    }
}
