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
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's non-streamed reasoning-to-content swap over the real engine: the synthetic llama served with
 * {@code neuralfree.reasoning.parser=nemotron_v3} reasons from its first token and never closes the block, so its
 * whole greedy answer is reasoning; with {@code force_nonempty_content} the assembled message carries that reasoning
 * as its content only — no {@code reasoning_content} — and without it the message keeps the reasoning and no content.
 */
class ContentFallbackSwapTest {

    private static EngineService service;
    private static OpenAiResource openAi;

    @BeforeAll
    static void open() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.llama().toBuilder().nCtxTrain(2048).build();
        Path model = SyntheticModels.llama(config).writeTempFile();
        EngineConfig engine = ServerTestSupport.pagedConfig(1024, 2).toBuilder().reasoningParser("nemotron_v3").build();
        service = new EngineService(model.toString(), engine);
        openAi = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static JsonView message(boolean forceContent) {
        String body = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 8,"
                + " \"temperature\": 0, \"chat_template_kwargs\": {\"force_nonempty_content\": " + forceContent + "}}";
        return at(view(openAi.chatCompletions(body).getEntity()), "choices", 0).obj("message");
    }

    @Test
    void anAnswerWithoutContentCarriesItsReasoningAsTheContentOnly() {
        JsonView plain = message(false);
        String reasoning = plain.str("reasoning_content");
        assertFalse(reasoning.isEmpty(), "the greedy answer is all reasoning");
        assertTrue(plain.raw("content") == null || plain.str("content").isEmpty(), "and has no content");

        JsonView swapped = message(true);
        assertEquals(reasoning, swapped.str("content"), "the reasoning becomes the content");
        assertNull(swapped.raw("reasoning_content"), "and is not repeated as reasoning");
    }
}
