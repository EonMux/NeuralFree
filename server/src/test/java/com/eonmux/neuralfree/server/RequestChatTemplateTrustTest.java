package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's {@code --trust-request-chat-template}: a server launched without it (the default) refuses a request's
 * non-empty {@code chat_template}, and a {@code chat_template} inside {@code chat_template_kwargs}, with 400 naming
 * the field and the flag, on every route that renders a chat (chat completions streamed or not, the Responses API,
 * the input-token counts, {@code /apply-template} and {@code /tokenize}); an empty template stays unset as in vLLM.
 * The flag and its {@code neuralfree.server.trust.request.chat.template} property turn the trust on.
 */
class RequestChatTemplateTrustTest {

    private static final String OVERRIDE = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}";
    private static final String MESSAGES = "\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]";
    private static final String WITH_TEMPLATE = MESSAGES + ", \"chat_template\": \"" + OVERRIDE + "\"";

    private static EngineService service;
    private static OpenAiResource openAi;
    private static ResponsesResource responses;
    private static LlamaCppResource llama;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 2));
        openAi = new OpenAiResource(service);
        responses = new ResponsesResource(service);
        llama = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static void assertRefused(String field, Runnable call) {
        assertEquals(400, statusOf(call));
        String message = errorOf(call);
        assertTrue(message.startsWith(field + " is passed with the request"), message);
        assertTrue(message.contains(ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_FLAG), message);
        assertTrue(message.contains(ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY), message);
    }

    @Test
    void theTrustIsOffByDefault() {
        assertFalse(ServerOptions.defaults().trustRequestChatTemplate());
        assertFalse(service.options().trustRequestChatTemplate());
        assertTrue(ServerOptions.defaults().withTrustRequestChatTemplate(true).trustRequestChatTemplate());
        assertEquals(8, ServerOptions.defaults().withTrustRequestChatTemplate(true).withResponseStore(8)
                .responseStore());
        assertTrue(ServerOptions.defaults().withTrustRequestChatTemplate(true).withResponseStore(8)
                .trustRequestChatTemplate(), "copies keep the trust");
    }

    @Test
    void everyChatRouteRefusesARequestTemplate() {
        String chat = "{" + WITH_TEMPLATE + ", \"max_tokens\": 1}";
        assertRefused("chat_template", () -> openAi.chatCompletions(chat));
        assertRefused("chat_template", () -> openAi.chatCompletions(
                "{" + WITH_TEMPLATE + ", \"max_tokens\": 1, \"stream\": true}"));
        assertRefused("chat_template", () -> openAi.chatInputTokens("{" + WITH_TEMPLATE + "}"));
        assertRefused("chat_template", () -> llama.applyTemplate("{" + WITH_TEMPLATE + "}"));
        assertRefused("chat_template", () -> llama.tokenize("{" + WITH_TEMPLATE + "}"));
        String response = "{\"input\": \"hi\", \"max_output_tokens\": 1, \"chat_template\": \"" + OVERRIDE + "\"}";
        assertRefused("chat_template", () -> responses.create(response));
        assertRefused("chat_template", () -> responses.inputTokens(response));
    }

    @Test
    void aTemplateInsideTheKwargsIsRefusedToo() {
        assertRefused("chat_template_kwargs.chat_template", () -> llama.applyTemplate(
                "{" + MESSAGES + ", \"chat_template_kwargs\": {\"chat_template\": \"" + OVERRIDE + "\"}}"));
        assertEquals(200, statusOf(() -> llama.applyTemplate(
                "{" + MESSAGES + ", \"chat_template_kwargs\": {\"enable_thinking\": false}}")),
                "other kwargs pass");
    }

    @Test
    void anEmptyTemplateStaysUnsetAndRenders() {
        String plain = view(llama.applyTemplate("{" + MESSAGES + "}").getEntity()).str("prompt");
        String empty = view(llama.applyTemplate("{" + MESSAGES + ", \"chat_template\": \"\"}").getEntity())
                .str("prompt");
        assertEquals(plain, empty, "an empty chat_template is unset, so it renders the model template");
        assertEquals(200, statusOf(() -> openAi.chatCompletions(
                "{" + MESSAGES + ", \"chat_template\": \"\", \"max_tokens\": 1}")));
    }

    @Test
    void theFlagAndThePropertyTurnTheTrustOn() {
        assertFalse(ServerLaunchArgs.parse("model.gguf").properties()
                .containsKey(ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY));
        withProperty(null, options -> assertFalse(options.trustRequestChatTemplate()));
        withProperty(" true ", options -> assertTrue(options.trustRequestChatTemplate()));
        withProperty("false", options -> assertFalse(options.trustRequestChatTemplate()));
    }

    /** Runs {@code check} over the system-property options with the trust property at {@code value}. */
    private static void withProperty(String value, Consumer<ServerOptions> check) {
        String key = ServerOptions.TRUST_REQUEST_CHAT_TEMPLATE_PROPERTY;
        String saved = System.getProperty(key);
        try {
            if (value == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, value);
            }
            check.accept(ServerOptions.fromSystemProperties());
        } finally {
            if (saved == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, saved);
            }
        }
    }
}
