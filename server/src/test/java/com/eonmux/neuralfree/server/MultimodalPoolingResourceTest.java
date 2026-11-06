package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.mm.MmprojTestFiles;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A generative model with a projector served as a pooling model (vLLM's Qwen2-VL / Phi-3-V embedding runners):
 * a conversation with image parts embeds as one media prompt on {@code /v1/embeddings}, {@code /pooling} and
 * Cohere's {@code /v2/embed}, its usage counting the image rows; {@code /score} and {@code /rerank} answer the
 * cosine of the pooled media vectors; the text tasks and truncation refuse media by name, as does the same model
 * without a projector.
 */
class MultimodalPoolingResourceTest {

    /** The llava fixture's rows per image. */
    private static final int IMAGE_ROWS = 4;
    private static final double TOLERANCE = 1e-5;

    private static EngineService service;
    private static EngineService textOnly;
    private static PoolingResource pooling;
    private static PoolingResource textPooling;
    private static String image;
    private static String otherImage;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Path mmproj = MmprojTestFiles.llava(SyntheticModels.Config.llama().nEmbd());
        EngineConfig config = ServerTestSupport.config().pooling(PoolingKind.LAST)
                .memory(MemoryConfig.builder()
                        .nCtx(256).ubatch(16).parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        service = new EngineService(model.toString(), config, ServerOptions.defaults(),
                MediaOptions.of(mmproj.toString()));
        textOnly = new EngineService(model.toString(), config);
        pooling = new PoolingResource(service);
        textPooling = new PoolingResource(textOnly);
        image = dataUrl(MultimodalChatResourceTest.png(12, 8));
        otherImage = dataUrl(MultimodalChatResourceTest.png(5, 17));
    }

    @AfterAll
    static void close() {
        service.close();
        textOnly.close();
    }

    private static String dataUrl(byte[] png) {
        return "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
    }

    private static String imagePart(String url) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}";
    }

    private static String conversation(String... parts) {
        return "[{\"role\": \"user\", \"content\": [" + String.join(", ", parts) + "]}]";
    }

    private static String text(String t) {
        return "{\"type\": \"text\", \"text\": \"" + t + "\"}";
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    private static double[] numbers(Object list) {
        List<?> values = (List<?>) list;
        double[] out = new double[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).doubleValue();
        }
        return out;
    }

    private static double[] embedding(String messages) {
        return numbers(at(ok(pooling.embeddings("{\"messages\": " + messages + "}")), "data", 0).raw("embedding"));
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }

    @Test
    void aMediaConversationEmbedsAsOneMediaPromptAndCountsItsImageRows() {
        String withImage = conversation(imagePart(image), text("a cat"));
        JsonView body = ok(pooling.embeddings("{\"messages\": " + withImage + "}"));
        double[] vector = numbers(at(body, "data", 0).raw("embedding"));
        assertEquals(SyntheticModels.Config.llama().nEmbd(), vector.length);
        long mediaTokens = body.obj("usage").i64("prompt_tokens");
        long textTokens = ok(pooling.embeddings("{\"messages\": " + conversation(text("a cat")) + "}"))
                .obj("usage").i64("prompt_tokens");
        assertTrue(mediaTokens >= textTokens + IMAGE_ROWS, mediaTokens + " rows vs " + textTokens + " text tokens");
        assertArrayEquals(vector, embedding(withImage), TOLERANCE, "the same prompt pools the same");
        assertFalse(Arrays.equals(vector, embedding(conversation(imagePart(otherImage), text("a cat")))),
                "another image pools differently");
        assertFalse(Arrays.equals(vector, embedding(conversation(text("a cat")))), "the image rows count");
    }

    @Test
    void thePoolingEmbedTaskAndCohereEmbedServeTheSameVector() {
        String messages = conversation(imagePart(image), text("a cat"));
        double[] expected = embedding(messages);
        JsonView pooled = ok(pooling.pooling("{\"task\": \"embed\", \"messages\": " + messages + "}"));
        assertArrayEquals(expected, numbers(at(pooled, "data", 0).raw("data")), TOLERANCE);
        JsonView cohere = ok(new CohereEmbedResource(service).embed("{\"inputs\": [{\"content\": ["
                + imagePart(image) + ", " + text("a cat") + "]}]}"));
        assertArrayEquals(expected, numbers(((List<?>) cohere.obj("embeddings").raw("float")).get(0)), TOLERANCE);
        assertEquals(0, cohere.obj("meta").obj("billed_units").i64("input_tokens"));
    }

    @Test
    void scoreAndRerankAnswerTheCosineOfThePooledMediaVectors() {
        String document = "{\"content\": [" + imagePart(image) + ", " + text("a cat") + "]}";
        JsonView scored = ok(pooling.score("{\"queries\": \"a cat\", \"documents\": [" + document + ", \"a dog\"]}"));
        assertEquals(2, size(scored, "data"));
        double[] query = numbers(at(ok(pooling.embeddings("{\"input\": \"a cat\"}")), "data", 0).raw("embedding"));
        double[] media = embedding(conversation(imagePart(image), text("a cat")));
        assertEquals(cosine(query, media), at(scored, "data", 0).f64("score"), 1e-4);
        JsonView ranked = ok(pooling.rerank("{\"query\": \"a cat\", \"documents\": [" + document + ", \"a dog\"],"
                + " \"return_documents\": true}"));
        assertEquals(2, size(ranked, "results"));
    }

    @Test
    void textTasksTruncationAndAModelWithoutAProjectorRefuseMediaByName() {
        String messages = conversation(imagePart(image), text("a cat"));
        assertTrue(errorOf(() -> pooling.pooling("{\"task\": \"classify\", \"messages\": " + messages + "}"))
                .contains("task 'classify' needs a RANK classifier"));
        assertTrue(errorOf(() -> pooling.embeddings("{\"messages\": " + messages + ","
                + " \"truncate_prompt_tokens\": 4}")).contains("truncate_prompt_tokens cannot be combined with media"));
        String refusal = errorOf(() -> textPooling.embeddings("{\"messages\": " + messages + "}"));
        assertTrue(refusal.contains(MediaOptions.MMPROJ_PROPERTY), refusal);
        String score = errorOf(() -> textPooling.score("{\"queries\": \"a cat\", \"documents\": [{\"content\": ["
                + imagePart(image) + "]}]}"));
        assertTrue(score.contains("image"), score);
    }
}
