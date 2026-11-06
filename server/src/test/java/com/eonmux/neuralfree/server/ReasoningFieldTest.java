package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.chat.ChatRequest;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's {@code reasoning} field: an assistant message accepts it as the current name of the deprecated
 * {@code reasoning_content} (both reach the template, {@code reasoning} winning where both are sent), and an
 * answer carries the reasoning under both names in its message and stream deltas, matching llama.cpp's
 * {@code reasoning_content} and vLLM's {@code reasoning}.
 */
class ReasoningFieldTest {

    /** Renders each message's role and both reasoning names, a missing value as {@code -}. */
    private static final String TEMPLATE = "{%- for m in messages -%}{{ m.role }}["
            + "{%- if m.reasoning is defined and m.reasoning is not none %}{{ m.reasoning }}{% else %}-{% endif -%}"
            + "|"
            + "{%- if m.reasoning_content is defined and m.reasoning_content is not none %}{{ m.reasoning_content }}"
            + "{% else %}-{% endif -%}"
            + "]{%- endfor -%}";

    private static EngineService service;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        EngineConfig engine = ServerTestSupport.pagedConfig(1024, 2).toBuilder()
                .reasoningParser("nemotron_v3").build();
        service = new EngineService(model.toString(), engine);
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** The template's rendering of one assistant message. */
    private static String render(String message) {
        ChatRequest chat = OpenAiRequests.chat(view("{\"messages\": [" + message + "]}"), false);
        return service.formatChat(chat, false);
    }

    @Test
    void vllmReasoningReachesTheTemplateUnderBothKeys() {
        assertEquals("assistant[R|R]",
                render("{\"role\": \"assistant\", \"content\": \"\", \"reasoning\": \"R\"}"));
    }

    @Test
    void reasoningWinsOverTheDeprecatedAlias() {
        assertEquals("assistant[R|R]",
                render("{\"role\": \"assistant\", \"content\": \"\", \"reasoning\": \"R\","
                        + " \"reasoning_content\": \"C\"}"));
    }

    @Test
    void theDeprecatedAliasAloneFillsBothKeys() {
        assertEquals("assistant[C|C]",
                render("{\"role\": \"assistant\", \"content\": \"\", \"reasoning_content\": \"C\"}"));
    }

    @Test
    void nonStringReasoningIsRefusedByName() {
        String error = errorOf(() -> openAi.chatCompletions("{\"messages\": [{\"role\": \"assistant\","
                + " \"content\": \"\", \"reasoning\": {\"type\": \"prompt_embeds\"}}]}"));
        assertTrue(error.contains("reasoning"), error);
    }

    /** A non-streamed answer: both reasoning names carry the same text (the id-less greedy answer is all reasoning). */
    @Test
    void aNonStreamedAnswerCarriesBothReasoningFields() {
        JsonView message = at(view(openAi.chatCompletions(answer(false)).getEntity()), "choices", 0)
                .obj("message");
        String reasoning = message.str("reasoning_content");
        assertFalse(reasoning.isEmpty(), "the nemotron parser keeps the whole answer as reasoning");
        assertEquals(reasoning, message.str("reasoning"));
    }

    @Test
    void includeReasoningFalseDropsBothReasoningFields() {
        JsonView message = at(view(openAi.chatCompletions(answer(true)).getEntity()), "choices", 0)
                .obj("message");
        assertNull(message.raw("reasoning_content"));
        assertNull(message.raw("reasoning"));
    }

    @Test
    void aStreamedAnswerCarriesBothReasoningFields() throws IOException {
        StringBuilder alias = new StringBuilder();
        StringBuilder reasoning = new StringBuilder();
        String stream = streamText(openAi.chatCompletions(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 16,"
                        + " \"temperature\": 0, \"stream\": true}"));
        for (JsonView frame : dataFrames(stream)) {
            List<?> choices = (List<?>) frame.raw("choices");
            if (choices == null || choices.isEmpty()) {
                continue;
            }
            JsonView delta = at(frame, "choices", 0).optObj("delta");
            if (delta.raw("reasoning_content") instanceof String text) {
                alias.append(text);
            }
            if (delta.raw("reasoning") instanceof String text) {
                reasoning.append(text);
            }
        }
        assertFalse(alias.toString().isEmpty(), "reasoning streams under reasoning_content");
        assertEquals(alias.toString(), reasoning.toString(), "and under reasoning");
    }

    /** A chat body whose answer is all reasoning, optionally with {@code include_reasoning} false. */
    private static String answer(boolean includeReasoning) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 16,"
                + " \"temperature\": 0" + (includeReasoning ? ", \"include_reasoning\": false" : "") + "}";
    }
}
