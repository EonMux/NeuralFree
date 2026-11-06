package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.mm.ClipDualFixtures;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cohere's {@code /v2/embed} over the real service, against vLLM's protocol: {@code texts} equal to the OpenAI
 * embeddings of the same inputs, the {@code binary} / {@code ubinary} bit packing and the {@code base64} floats of
 * {@code embedding_types}, {@code output_dimension}, the {@code END} / {@code START} / {@code NONE} truncation of
 * {@code max_tokens}, {@code images} and mixed {@code inputs} through a CLIP dual encoder with the billed image
 * units, and the named refusals.
 */
class CohereEmbedResourceTest {

    private static final double TOLERANCE = 1e-6;
    private static final String LONG = "the quick brown fox jumps over the lazy dog again and again";

    private static EngineService bert;
    private static EngineService clip;
    private static CohereEmbedResource cohere;
    private static CohereEmbedResource clipCohere;
    private static PoolingResource openai;
    private static PoolingResource clipOpenai;
    private static String image;

    @BeforeAll
    static void open() throws IOException {
        bert = new EngineService(SyntheticModels.bert().writeTempFile().toString(),
                ServerTestSupport.config().build());
        ModelSourceOptions source = ModelSourceOptions.builder().tokenizerDir(ClipDualFixtures.tokenizerDir()).build();
        clip = new EngineService(ClipDualFixtures.twoTower().toString(),
                ServerTestSupport.config().sourceOptions(source).build());
        cohere = new CohereEmbedResource(bert);
        clipCohere = new CohereEmbedResource(clip);
        openai = new PoolingResource(bert);
        clipOpenai = new PoolingResource(clip);
        image = "data:image/png;base64," + Base64.getEncoder().encodeToString(MultimodalChatResourceTest.png(12, 8));
    }

    @AfterAll
    static void close() {
        bert.close();
        clip.close();
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

    private static double[] floats(JsonView body, int i) {
        return numbers(((List<?>) body.obj("embeddings").raw("float")).get(i));
    }

    private static double[] openAi(PoolingResource resource, String body, int i) {
        return numbers(at(ok(resource.embeddings(body)), "data", i).raw("embedding"));
    }

    @Test
    void textsEqualTheOpenAiEmbeddingsAndEchoTheTexts() {
        JsonView body = ok(cohere.embed("{\"texts\": [\"hello world\", \"a recipe\"]}"));
        assertEquals("embeddings_by_type", body.str("response_type"));
        assertTrue(body.str("id").startsWith("embd-"));
        assertEquals(List.of("hello world", "a recipe"), body.raw("texts"));
        assertEquals(List.of("float"), List.copyOf(body.obj("embeddings").keys()), "float is the default type");
        for (int i = 0; i < 2; i++) {
            assertArrayEquals(openAi(openai, "{\"input\": [\"hello world\", \"a recipe\"]}", i), floats(body, i),
                    TOLERANCE);
        }
        JsonView meta = body.obj("meta");
        assertEquals("2", meta.obj("api_version").str("version"));
        long tokens = ok(openai.embeddings("{\"input\": [\"hello world\", \"a recipe\"]}")).obj("usage")
                .i64("prompt_tokens");
        assertEquals(tokens, meta.obj("billed_units").i64("input_tokens"));
        assertEquals(0, meta.obj("billed_units").i64("image_tokens"));
    }

    @Test
    void embeddingTypesPackTheSignBitsAndEncodeTheFloats() {
        JsonView body = ok(cohere.embed("{\"texts\": [\"hello world\"],"
                + " \"embedding_types\": [\"float\", \"binary\", \"ubinary\", \"base64\"]}"));
        JsonView types = body.obj("embeddings");
        assertEquals(List.of("float", "binary", "ubinary", "base64"), List.copyOf(types.keys()));
        double[] v = floats(body, 0);
        double[] ubinary = numbers(((List<?>) types.raw("ubinary")).get(0));
        double[] binary = numbers(((List<?>) types.raw("binary")).get(0));
        assertEquals(v.length / 8, ubinary.length);
        for (int b = 0; b < ubinary.length; b++) {
            int expected = 0;
            for (int i = 0; i < 8; i++) {
                expected = expected << 1 | (v[8 * b + i] >= 0 ? 1 : 0);
            }
            assertEquals(expected, (int) ubinary[b], "numpy packbits, most significant bit first");
            assertEquals(expected - 128, (int) binary[b], "binary is the signed form");
        }
        byte[] raw = Base64.getDecoder().decode((String) ((List<?>) types.raw("base64")).get(0));
        ByteBuffer floats = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(v.length * Float.BYTES, raw.length);
        for (double x : v) {
            assertEquals((float) x, floats.getFloat(), 0f);
        }
    }

    @Test
    void outputDimensionKeepsTheLeadingValuesNormalized() {
        JsonView body = ok(cohere.embed("{\"texts\": [\"hello world\"], \"output_dimension\": 16}"));
        double[] v = floats(body, 0);
        assertEquals(16, v.length);
        assertArrayEquals(openAi(openai, "{\"input\": \"hello world\", \"dimensions\": 16}", 0), v, TOLERANCE);
    }

    @Test
    void truncateEndKeepsTheFirstTokensAndStartTheLast() {
        double[] end = floats(ok(cohere.embed("{\"texts\": [\"" + LONG + "\"], \"max_tokens\": 4}")), 0);
        double[] start = floats(ok(cohere.embed("{\"texts\": [\"" + LONG + "\"], \"max_tokens\": 4,"
                + " \"truncate\": \"START\"}")), 0);
        assertArrayEquals(openAi(openai, "{\"input\": \"" + LONG + "\", \"truncate_prompt_tokens\": 4,"
                + " \"truncation_side\": \"right\"}", 0), end, TOLERANCE);
        assertArrayEquals(openAi(openai, "{\"input\": \"" + LONG + "\", \"truncate_prompt_tokens\": 4}", 0), start,
                TOLERANCE);
        assertFalse(java.util.Arrays.equals(end, start), "the two sides keep different tokens");
        JsonView usage = ok(cohere.embed("{\"texts\": [\"" + LONG + "\"], \"max_tokens\": 4}")).obj("meta");
        assertEquals(4, usage.obj("billed_units").i64("input_tokens"));
    }

    @Test
    void truncateNoneRefusesAnInputOverMaxTokens() {
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"" + LONG + "\"], \"max_tokens\": 4,"
                + " \"truncate\": \"NONE\"}")).contains("exceeds max_tokens=4 with truncate=NONE"));
        ok(cohere.embed("{\"texts\": [\"hi\"], \"max_tokens\": 64, \"truncate\": \"NONE\"}"));
    }

    @Test
    void imagesEmbedThroughTheDualEncoderAndBillImageUnits() {
        JsonView body = ok(clipCohere.embed("{\"images\": [\"" + image + "\"]}"));
        double[] expected = openAi(clipOpenai, "{\"messages\": [{\"role\": \"user\", \"content\": ["
                + "{\"type\": \"image_url\", \"image_url\": {\"url\": \"" + image + "\"}}]}]}", 0);
        assertArrayEquals(expected, floats(body, 0), TOLERANCE);
        assertNull(body.raw("texts"), "only texts are echoed");
        JsonView inputs = ok(clipCohere.embed("{\"inputs\": [{\"content\": [{\"type\": \"image_url\","
                + " \"image_url\": {\"url\": \"" + image + "\"}}]}]}"));
        assertArrayEquals(expected, floats(inputs, 0), TOLERANCE);
        assertEquals(0, inputs.obj("meta").obj("billed_units").i64("input_tokens"),
                "vLLM bills every token of a request holding an image as image tokens");
        assertTrue(errorOf(() -> clipCohere.embed("{\"inputs\": [{\"content\": [{\"type\": \"text\", \"text\":"
                + " \"a cat\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"" + image + "\"}}]}]}"))
                .contains("CLIP accepts text-only or image-only inputs, not both"));
    }

    @Test
    void refusalsNameTheField() {
        assertEquals(400, statusOf(() -> cohere.embed("{}")));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"a\"], \"images\": [\"b\"]}"))
                .contains("Exactly one of texts, images, or inputs"));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": []}")).contains("texts must be a non-empty array"));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"a\"], \"input_type\": \"search_query\"}"))
                .contains("Unsupported input_type 'search_query'"));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"a\"], \"embedding_types\": [\"int8\"]}"))
                .contains("embedding_types[0]"));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"a\"], \"truncate\": \"MIDDLE\"}"))
                .contains("truncate must be NONE, START or END"));
        assertTrue(errorOf(() -> cohere.embed("{\"texts\": [\"a\"], \"dimensions\": 4}"))
                .contains("unknown field 'dimensions'"));
        assertTrue(errorOf(() -> cohere.embed("{\"inputs\": [{\"content\": [{\"type\": \"image_url\"}]}]}"))
                .contains("inputs[0].content[0]: CohereEmbedContent with type='image_url' requires image_url.url"));
        assertTrue(errorOf(() -> clipCohere.embed("{\"images\": [\"" + image + "\"],"
                + " \"embedding_types\": [\"binary\"]}")).contains("multiple of 8"),
                "the 6-wide CLIP embedding does not pack into bytes");
        assertTrue(errorOf(() -> clipCohere.embed("{\"images\": [\"" + image + "\"], \"max_tokens\": 8}"))
                .contains("an image input keeps its rows"));
    }
}
