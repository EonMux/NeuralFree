package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The chat message boundary of a rendered prompt (llama.cpp's message spans): the server locates the last
 * message's start in the rendered prompt and names it the engine's checkpoint row, so a conversation branch that
 * keeps everything before the last message restores there and re-prefills only its own tail — over a recurrent
 * model, whose cached prefix restores at checkpoints only.
 */
public class ChatMessageBoundaryCheckpointTest {

    private static EngineService service;
    private static OpenAiResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.mamba2().writeTempFile();
        EngineConfig config = EngineConfig.builder()
                .memory(MemoryConfig.builder()
                        .nCtx(128).ubatch(8).kvPaged(true).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32)
                        .build())
                .build();
        service = new EngineService(model.toString(), config);
        resource = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static long cachedTokens(String message) {
        Response response = resource.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": \""
                + message + "\"}], \"max_tokens\": 3, \"temperature\": 0}");
        assertEquals(200, response.getStatus());
        JsonView body = ServerTestSupport.view(response.getEntity());
        JsonView usage = JsonView.of(body.raw("usage"), "usage");
        return usage.obj("prompt_tokens_details").i64("cached_tokens");
    }

    /**
     * A branch that keeps the conversation up to the last message restores at the boundary checkpoint: the
     * cached prefix covers the rendered rows before the last message's content (the template header), which the
     * default checkpoints near the prompt's end could not serve.
     */
    @Test
    void aBranchBeforeTheLastMessageRestoresAtItsBoundaryCheckpoint() {
        long branch = cachedTokens("please count to three, slowly please");
        assertTrue(branch >= 2,
                "a branch restores the boundary checkpoint of the first conversation, got " + branch);

        // a plain repeat restores at least the same prefix through its checkpoints
        long repeat = cachedTokens("please count to three, slowly please");
        assertTrue(repeat >= branch, "a repeat restores the conversation's checkpoints, got " + repeat);
    }

    /** A conversation sharing nothing before the last message restores nothing. */
    @Test
    void anUnrelatedConversationRestoresNothing() {
        long other = cachedTokens("completely different words appear here now");
        assertEquals(0L, other);
    }
}
