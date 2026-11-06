package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.Json;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * llama.cpp's {@code reasoning_format: none} on a harmony (gpt-oss) model, over a synthetic model whose chat template
 * selects the harmony filter: the output is forced with a one-entry {@code guided_choice} ({@code include_reasoning}
 * false so the constraint holds from the first token), and the answer keeps the analysis message inline in the
 * content, headers and terminator included, while tool calls are still parsed (llama.cpp common/parsers/gpt-oss.cpp
 * with {@code extract_reasoning} false, tests/test-chat.cpp). The default format splits the analysis off.
 */
class HarmonyReasoningFormatTest {

    private static final String TEMPLATE = "{%- for m in messages %}{%- if m.role == 'assistant' %}"
            + "<|start|>assistant<|channel|>final<|message|>{{ m.content }}<|end|>"
            + "{%- else %}<|start|>{{ m.role }}<|message|>{{ m.content }}<|end|>{%- endif %}{% endfor %}"
            + "{%- if add_generation_prompt %}<|start|>assistant{% endif %}";

    private static final String ANSWER = "<|channel|>analysis<|message|>I'm thinking<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Hello, world!";

    private static final String CALL = "<|channel|>analysis<|message|>Need the weather.<|end|>"
            + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
            + "<|message|>{\\\"city\\\": \\\"Paris\\\"}";

    private static final String TOOLS = """
            "tools": [{"type": "function", "function": {"name": "get_weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]""";

    private static EngineService service;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).str("tokenizer.chat_template", TEMPLATE).writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2));
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** A chat body forcing {@code output}, with {@code extra} fields appended. */
    private static String chat(String output, String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather in Paris?\"}], \"max_tokens\": 400,"
                + " \"temperature\": 0, \"include_reasoning\": false, \"guided_choice\": [\"" + output + "\"]"
                + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    private static JsonView choice(String body) {
        return at(view(openAi.chatCompletions(body).getEntity()), "choices", 0);
    }

    @Test
    void theModelRunsTheHarmonyFilter() {
        assertEquals(com.eonmux.neuralfree.gguf.engine.ReasoningFilter.Mode.HARMONY_CHANNELS,
                service.reasoning().mode());
    }

    @Test
    void noneKeepsTheAnalysisInlineAndDropsTheFinalHeader() {
        JsonView raw = choice(chat(ANSWER, "\"reasoning_format\": \"none\""));
        assertEquals("<|channel|>analysis<|message|>I'm thinking<|end|>Hello, world!",
                raw.obj("message").str("content"));
        assertEquals("stop", raw.str("finish_reason"));
        JsonView split = choice(chat(ANSWER, ""));
        assertEquals("Hello, world!", split.obj("message").str("content"), "the default splits the analysis off");
    }

    @Test
    void noneStillParsesToolCalls() {
        JsonView raw = choice(chat(CALL, TOOLS + ", \"reasoning_format\": \"none\""));
        assertEquals("tool_calls", raw.str("finish_reason"));
        JsonView message = raw.obj("message");
        assertEquals("<|channel|>analysis<|message|>Need the weather.<|end|>", message.str("content"));
        JsonView function = at(message, "tool_calls", 0).obj("function");
        assertEquals("get_weather", function.str("name"));
        assertEquals("Paris", JsonView.of(Json.parse(function.str("arguments"), Json.Limits.UNTRUSTED), "args")
                .str("city"));
        JsonView split = choice(chat(CALL, TOOLS)).obj("message");
        assertEquals("get_weather", at(split, "tool_calls", 0).obj("function").str("name"));
        Object content = split.raw("content");
        assertTrue(content == null || "".equals(content), "the default leaves no content: " + content);
    }

    @Test
    void aNoneStreamCarriesTheSameContentAndTheToolCall() throws IOException {
        String stream = streamText(openAi.chatCompletions(
                chat(CALL, TOOLS + ", \"reasoning_format\": \"none\", \"stream\": true")));
        StringBuilder content = new StringBuilder();
        StringBuilder names = new StringBuilder();
        boolean reasoning = false;
        for (JsonView frame : dataFrames(stream)) {
            List<?> choices = (List<?>) frame.raw("choices");
            if (choices == null || choices.isEmpty()) {
                continue;
            }
            JsonView delta = at(frame, "choices", 0).optObj("delta");
            if (delta.raw("content") instanceof String text) {
                content.append(text);
            }
            reasoning |= delta.has("reasoning_content");
            if (delta.raw("tool_calls") instanceof List<?> calls) {
                for (int i = 0; i < calls.size(); i++) {
                    JsonView function = at(delta, "tool_calls", i).optObj("function");
                    if (function.raw("name") instanceof String name) {
                        names.append(name);
                    }
                }
            }
        }
        assertEquals("<|channel|>analysis<|message|>Need the weather.<|end|>", content.toString());
        assertEquals("get_weather", names.toString());
        assertFalse(reasoning, "no reasoning_content delta under none");
    }
}
