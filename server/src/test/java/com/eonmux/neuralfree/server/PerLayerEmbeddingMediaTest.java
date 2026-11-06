package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticGemmaModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.function.Supplier;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Embedding rows over the real service on a gemma3n-shaped model, whose blocks read per-layer token embeddings: a
 * chat image through a Gemma 3 projector and a {@code prompt_embeds} completion both generate, the embedding rows
 * taking the padding token's per-layer row as llama.cpp does, and the image's rows count in the usage.
 */
class PerLayerEmbeddingMediaTest {

    private static EngineService service;
    private static OpenAiResource openai;
    private static String png64;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticGemmaModels.gemma3n().writeTempFile();
        Path mmproj = MmprojTestFiles.gemma3(SyntheticGemmaModels.gemma3nConfig().nEmbd());
        png64 = Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(12, 8));
        EngineConfig config = ServerTestSupport.config()
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        openai = new OpenAiResource(service);
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static String chat(String parts) {
        return "{\"messages\": [{\"role\": \"user\", \"content\": [" + parts + "]}], \"max_tokens\": 3,"
                + " \"temperature\": 0}";
    }

    private static String text(String t) {
        return "{\"type\": \"text\", \"text\": \"" + t + "\"}";
    }

    private static String image() {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64," + png64 + "\"}}";
    }

    /** The response of a call that must succeed; a refusal fails the test with its message. */
    private static Response ok(Supplier<Response> call) {
        try {
            Response response = call.get();
            assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
            return response;
        } catch (WebApplicationException e) {
            throw new AssertionError("refused: " + e.getResponse().getEntity(), e);
        }
    }

    private static long promptTokens(Response response) {
        return JsonView.of(view(response.getEntity()).raw("usage"), "usage").i64("prompt_tokens");
    }

    @Test
    void aChatImageGeneratesAndItsRowsCountInTheUsage() {
        String withImage = chat(text("describe ") + "," + image());
        Response response = ok(() -> openai.chatCompletions(withImage));
        JsonView choice = JsonView.of(((List<?>) view(response.getEntity()).raw("choices")).get(0), "choices[0]");
        assertFalse(choice.str("finish_reason").isEmpty());
        long prompt = promptTokens(response);
        long textOnly = promptTokens(ok(() -> openai.chatCompletions(chat(text("describe ")))));
        assertTrue(prompt > textOnly, prompt + " rows with the image, " + textOnly + " without");
        assertEquals(prompt, view(ok(() -> openai.chatInputTokens(withImage)).getEntity()).i64("input_tokens"),
                "the token count encodes the image like the generation");
    }

    @Test
    void aPromptEmbedsCompletionGenerates() {
        int width = service.embeddingLength();
        StringBuilder rows = new StringBuilder("[");
        for (int r = 0; r < 3; r++) {
            rows.append(r > 0 ? ",[" : "[");
            for (int d = 0; d < width; d++) {
                rows.append(d > 0 ? "," : "").append(0.25f * (float) Math.sin(0.3 * (r * width + d)));
            }
            rows.append(']');
        }
        Response response = ok(() -> openai.completions("{\"prompt_embeds\": " + rows.append(']')
                + ", \"max_tokens\": 2, \"temperature\": 0}"));
        assertEquals(3L, promptTokens(response));
    }
}
