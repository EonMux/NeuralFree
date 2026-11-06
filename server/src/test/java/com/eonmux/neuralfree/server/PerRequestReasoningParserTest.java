package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A reasoning parser that reads the request over the real engine: the synthetic llama served with
 * {@code neuralfree.reasoning.parser=hy_v3} answers a forced output as plain content by default (vLLM's
 * {@code no_think}), splits it at {@code </think>} once {@code reasoning_effort} (top-level, or in
 * {@code chat_template_kwargs}) asks for thinking, and refuses a non-string effort with a 400 naming the field.
 */
class PerRequestReasoningParserTest {

    private static final String OUTPUT = "plan</think>answer";

    private static EngineService service;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).writeTempFile();
        EngineConfig engine = ServerTestSupport.pagedConfig(1024, 2).toBuilder().reasoningParser("hy_v3").build();
        service = new EngineService(model.toString(), engine);
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    /** A chat body forcing {@link #OUTPUT} (the reasoning hidden, so the constraint holds from the first token). */
    private static String chat(String extra) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 200,"
                + " \"temperature\": 0, \"include_reasoning\": false, \"guided_choice\": [\"" + OUTPUT + "\"]"
                + (extra.isEmpty() ? "" : ", " + extra) + "}";
    }

    private static String content(String body) {
        JsonView choice = at(view(openAi.chatCompletions(body).getEntity()), "choices", 0);
        return choice.obj("message").str("content");
    }

    @Test
    void theDefaultEffortIsNoThinkSoTheOutputIsContent() {
        assertTrue(service.reasoning().selectsPerRequest());
        assertEquals(OUTPUT, content(chat("")));
        assertEquals(OUTPUT, content(chat("\"reasoning_effort\": \"no_think\"")));
    }

    @Test
    void aThinkingEffortSplitsTheReasoningOff() {
        assertEquals("answer", content(chat("\"reasoning_effort\": \"high\"")));
        assertEquals("answer", content(chat("\"chat_template_kwargs\": {\"reasoning_effort\": \"low\"}")));
        assertEquals(OUTPUT, content(chat(
                "\"reasoning_effort\": \"high\", \"chat_template_kwargs\": {\"reasoning_effort\": \"no_think\"}")),
                "the kwargs win over the top-level field");
    }

    @Test
    void aNonStringEffortIsRefusedByName() {
        String body = chat("\"chat_template_kwargs\": {\"reasoning_effort\": 3}");
        assertEquals(400, statusOf(() -> openAi.chatCompletions(body)));
        assertTrue(errorOf(() -> openAi.chatCompletions(body)).contains("reasoning_effort"));
    }
}
