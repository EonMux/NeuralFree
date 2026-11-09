package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.mm.ClipDualEncoder;
import com.eonmux.neuralfree.gguf.mm.ClipDualFixtures;
import com.eonmux.neuralfree.gguf.mm.ClipFamily;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A Hugging Face {@code SiglipModel} checkpoint served as the model (vLLM's {@code SiglipEmbeddingModel}): its text
 * and image embeddings are the dual encoder's own, {@code /score} and {@code /rerank} answer SigLIP's sigmoid
 * probability of the pair's cosine similarity, and the generative routes refuse it by name.
 */
class SiglipDualEncoderResourceTest {

    private static final String TEXT = "a b";
    private static final double TOLERANCE = 1e-5;

    private static EngineService siglip;
    private static PoolingResource pooling;
    private static ClipDualEncoder reference;
    private static String image;
    private static String otherImage;
    private static float[] textVector;
    private static float[] imageVector;
    private static float[] otherImageVector;

    @BeforeAll
    static void open() throws IOException {
        Path dir = ClipDualFixtures.siglipCheckpoint(Files.createTempDirectory("siglip"));
        EngineConfig config = ServerTestSupport.config().build();
        siglip = new EngineService(dir.toString(), config);
        pooling = new PoolingResource(siglip);
        reference = ServerTestSupport.dualReference(dir.toString(), config);
        byte[] png = MultimodalChatResourceTest.png(12, 8);
        byte[] otherPng = MultimodalChatResourceTest.png(5, 17);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
        otherImage = "data:image/png;base64," + Base64.getEncoder().encodeToString(otherPng);
        textVector = reference.embedText(reference.tokenizer().encode(TEXT, true, true));
        imageVector = reference.embedImage(png);
        otherImageVector = reference.embedImage(otherPng);
    }

    @AfterAll
    static void close() throws IOException {
        siglip.close();
        reference.close();
    }

    @Test
    void theCheckpointServesItsTowersAsTheDualEncoder() {
        assertTrue(siglip.pooling().dualEncoder());
        assertEquals(ClipFamily.SIGLIP, reference.family());
        JsonView text = ok(pooling.embeddings("{\"input\": \"" + TEXT + "\", \"embd_normalize\": -1}"));
        assertVector(textVector, at(text, "data", 0).raw("embedding"));
        JsonView images = ok(pooling.embeddings("{\"messages\": [[{\"role\": \"user\", \"content\": ["
                + imagePart(image) + "]}]], \"embd_normalize\": -1}"));
        assertVector(imageVector, at(images, "data", 0).raw("embedding"));
    }

    @Test
    void scoreAndRerankAnswerTheSigmoidProbabilityOfThePair() {
        double textImage = reference.score(cosine(textVector, imageVector));
        double textOther = reference.score(cosine(textVector, otherImageVector));
        assertTrue(textImage > 0 && textImage < 1, "a probability: " + textImage);
        JsonView scored = ok(pooling.score("{\"queries\": \"" + TEXT + "\", \"documents\": [" + imageEntry(image)
                + ", " + imageEntry(otherImage) + "]}"));
        assertEquals(textImage, at(scored, "data", 0).f64("score"), TOLERANCE);
        assertEquals(textOther, at(scored, "data", 1).f64("score"), TOLERANCE);
        JsonView ranked = ok(pooling.rerank("{\"query\": \"" + TEXT + "\", \"documents\": [" + imageEntry(image)
                + ", " + imageEntry(otherImage) + "]}"));
        for (int r = 0; r < 2; r++) {
            JsonView result = at(ranked, "results", r);
            double want = result.i64("index") == 0 ? textImage : textOther;
            assertEquals(want, result.f64("relevance_score"), TOLERANCE);
        }
    }

    @Test
    void generativeRoutesRefuseTheSiglipDualEncoderByName() {
        OpenAiResource openai = new OpenAiResource(siglip);
        String refusal = errorOf(() -> openai.completions("{\"prompt\": \"a\", \"max_tokens\": 1}"));
        assertTrue(refusal.contains("SigLIP dual encoder") && refusal.contains("SiglipEmbeddingModel"), refusal);
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    private static String imagePart(String url) {
        return "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + url + "\"}}";
    }

    private static String imageEntry(String url) {
        return "{\"content\": [" + imagePart(url) + "]}";
    }

    private static void assertVector(float[] expected, Object actual) {
        List<?> values = (List<?>) actual;
        assertEquals(expected.length, values.size(), "the vector width");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], ((Number) values.get(i)).doubleValue(), TOLERANCE, "value " + i);
        }
    }

    private static double cosine(float[] a, float[] b) {
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
}
