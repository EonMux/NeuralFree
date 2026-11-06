package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The llama.cpp tokenizer endpoints with their vLLM spellings: {@code /tokenize} over content, prompt and chat
 * messages, {@code with_pieces}, {@code /detokenize} round trips, {@code /apply-template} and the health alias.
 */
public class LlamaCppResourceTest {

    private static EngineService service;
    private static LlamaCppResource resource;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 2));
        resource = new LlamaCppResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    @Test
    void tokenizeFollowsBothSpellingsOfTheSpecialTokenDefault() {
        JsonView llama = view(resource.tokenize("{\"content\": \"hello world\"}").getEntity());
        JsonView vllm = view(resource.tokenize("{\"prompt\": \"hello world\"}").getEntity());
        int plain = size(llama, "tokens");
        assertEquals(plain, llama.i64("count"));
        assertEquals(plain + 1, size(vllm, "tokens"), "vLLM adds the BOS by default, llama.cpp does not");
        assertEquals(1L, ((Number) ((List<?>) vllm.raw("tokens")).get(0)).longValue(), "the BOS id leads");
        assertEquals(service.maxModelLength(), vllm.i64("max_model_len"));
        JsonView withBos = view(resource.tokenize("{\"content\": \"hello world\", \"add_special\": true}")
                .getEntity());
        assertEquals(plain + 1, size(withBos, "tokens"));
    }

    @Test
    void withPiecesPairsEachIdWithItsText() {
        JsonView body = view(resource.tokenize("{\"content\": \"hello\", \"with_pieces\": true}").getEntity());
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < size(body, "tokens"); i++) {
            JsonView entry = at(body, "tokens", i);
            assertTrue(entry.i64("id") >= 0);
            text.append(entry.str("piece"));
        }
        assertTrue(text.toString().contains("hello"), "the pieces spell the content: " + text);
    }

    @Test
    void detokenizeRoundTripsTheTokens() {
        List<?> tokens = (List<?>) view(resource.tokenize("{\"content\": \"hello world\"}").getEntity())
                .raw("tokens");
        StringBuilder ids = new StringBuilder();
        for (Object id : tokens) {
            ids.append(ids.length() == 0 ? "" : ", ").append(id);
        }
        JsonView body = view(resource.detokenize("{\"tokens\": [" + ids + "]}").getEntity());
        assertEquals("hello world", body.str("content").strip());
        assertEquals(body.str("content"), body.str("prompt"));
        assertEquals("", view(resource.detokenize("{\"tokens\": []}").getEntity()).str("content"));
        assertEquals(400, statusOf(() -> resource.detokenize("{\"tokens\": [100000]}")));
        assertEquals(400, statusOf(() -> resource.detokenize("{}")));
    }

    @Test
    void chatMessagesTokenizeAndRenderThroughTheTemplate() {
        String chat = "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}";
        String prompt = view(resource.applyTemplate(chat).getEntity()).str("prompt");
        assertTrue(prompt.contains("hi"), "the template rendered the message: " + prompt);
        JsonView tokens = view(resource.tokenize(chat).getEntity());
        assertEquals(service.encode(prompt, false, true).length, tokens.i64("count"));
        assertEquals(400, statusOf(() -> resource.tokenize("{}")));
        assertEquals(400, statusOf(() -> resource.applyTemplate("{\"messages\": []}")));
    }

    @Test
    void healthAnswersOnBothPaths() {
        assertEquals("ok", view(resource.healthAlias().getEntity()).str("status"));
        assertEquals("ok", view(resource.health().getEntity()).str("status"));
    }
}
