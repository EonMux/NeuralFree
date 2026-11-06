package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The pooling endpoints over the real service: embeddings from the synthetic bert (text and token-id inputs, the
 * llama.cpp path alias), the generative llama's named pooling refusal on embeddings and rerank, and rerank's body
 * validation.
 */
public class PoolingResourceTest {

    private static EngineService bert;
    private static EngineService llama;
    private static PoolingResource embeddings;
    private static PoolingResource generative;

    @BeforeAll
    static void open() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        bert = new EngineService(bertModel.toString(), ServerTestSupport.config().build());
        Path llamaModel = SyntheticModels.llama().writeTempFile();
        llama = new EngineService(llamaModel.toString(), ServerTestSupport.pagedConfig(64, 2));
        embeddings = new PoolingResource(bert);
        generative = new PoolingResource(llama);
    }

    @AfterAll
    static void close() {
        bert.close();
        llama.close();
    }

    @Test
    void embeddingsServeTextInputsFromAnEmbeddingModel() {
        Response response;
        try {
            response = embeddings.embeddings("""
                    {"input": ["hello world", "second"]}""");
        } catch (WebApplicationException e) {
            throw new AssertionError("embeddings refused: " + e.getResponse().getEntity(), e);
        }
        assertEquals(200, response.getStatus());
        JsonView body = view(response.getEntity());
        assertEquals("list", body.str("object"));
        assertEquals(2, size(body, "data"));
        JsonView first = at(body, "data", 0);
        assertEquals("embedding", first.str("object"));
        assertEquals(0L, first.i64("index"));
        assertTrue(((List<?>) first.raw("embedding")).size() > 8, "the pooled vector is non-trivial");
    }

    @Test
    void embeddingsTakeTokenIdInputsAndTheAliasPath() {
        JsonView ids = view(embeddings.embeddings("{\"input\": [5, 6, 7]}").getEntity());
        assertEquals(1, size(ids, "data"));
        assertEquals(3L, JsonView.of(ids.raw("usage"), "usage").i64("prompt_tokens"), "the ids count as given");
        JsonView nested = view(embeddings.embeddings("{\"input\": [[5, 6], [7]]}").getEntity());
        assertEquals(2, size(nested, "data"));
        assertEquals(400, statusOf(() -> embeddings.embeddings("{\"input\": [5, -1]}")));
        assertEquals(400, statusOf(() -> embeddings.embeddings("{\"input\": [[]]}")));
        assertEquals(400, statusOf(() -> embeddings.embeddings("{\"input\": 3}")));
    }

    private static float[] vector(Object embedding) {
        List<?> values = (List<?>) embedding;
        float[] out = new float[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).floatValue();
        }
        return out;
    }

    private static float[] first(String body) {
        return vector(at(view(embeddings.embeddings(body).getEntity()), "data", 0).raw("embedding"));
    }

    private static double norm(float[] v, double p) {
        double sum = 0;
        for (float x : v) {
            sum += Math.pow(Math.abs(x), p);
        }
        return Math.pow(sum, 1 / p);
    }

    @Test
    void textGetsTheSpecialTokensUnlessAskedNotTo() {
        long with = view(embeddings.embeddings("{\"input\": \"hello world\"}").getEntity()).obj("usage")
                .i64("prompt_tokens");
        assertEquals(bert.encode("hello world", true, true).length, with, "CLS and SEP frame the text");
        long without = view(embeddings.embeddings("{\"input\": \"hello world\", \"add_special_tokens\": false}")
                .getEntity()).obj("usage").i64("prompt_tokens");
        assertEquals(with - 2, without);
        long cut = view(embeddings.embeddings("{\"input\": \"hello world\", \"truncate_prompt_tokens\": 2}")
                .getEntity()).obj("usage").i64("prompt_tokens");
        assertEquals(2, cut);
    }

    @Test
    void vectorsAreEuclideanNormalizedByDefaultAndShapedOnRequest() {
        String text = "\"input\": \"hello world\"";
        assertEquals(1.0, norm(first("{" + text + "}"), 2), 1e-5);
        float[] raw = first("{" + text + ", \"use_activation\": false}");
        assertTrue(Math.abs(norm(raw, 2) - 1.0) > 1e-3, "use_activation false leaves the pooled vector raw");
        assertArrayEquals(raw, first("{" + text + ", \"embd_normalize\": -1}"), 1e-6f);
        assertEquals(1.0, norm(first("{" + text + ", \"embd_normalize\": 1}"), 1), 1e-5);
        float[] cut = first("{" + text + ", \"dimensions\": 4}");
        assertEquals(4, cut.length);
        assertEquals(1.0, norm(cut, 2), 1e-5, "the cut vector is normalized after the cut");
        assertEquals(400, statusOf(() -> embeddings.embeddings("{" + text + ", \"dimensions\": 100000}")));
    }

    @Test
    void base64CarriesTheFloat32Bytes() {
        float[] floats = first("{\"input\": \"hello\"}");
        String encoded = at(view(embeddings.embeddings("{\"input\": \"hello\", \"encoding_format\": \"base64\"}")
                .getEntity()), "data", 0).str("embedding");
        java.nio.ByteBuffer bytes = java.nio.ByteBuffer.wrap(java.util.Base64.getDecoder().decode(encoded))
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] decoded = new float[bytes.remaining() / Float.BYTES];
        for (int i = 0; i < decoded.length; i++) {
            decoded[i] = bytes.getFloat();
        }
        assertArrayEquals(floats, decoded, 1e-6f);
        for (String bad : new String[]{"\"encoding_format\": \"bytes\"", "\"embed_dtype\": \"float16\"",
                "\"endianness\": \"middle\"", "\"embd_normalize\": -2", "\"truncate_prompt_tokens\": 0"}) {
            assertEquals(400, statusOf(() -> embeddings.embeddings("{\"input\": \"x\", " + bad + "}")), bad);
        }
    }

    @Test
    void theNativeRoutesAnswerLlamaCppsArray() {
        Object parsed = ServerTestSupport.parsed(embeddings.nativeEmbeddings("{\"content\": \"hello\"}").getEntity());
        assertTrue(parsed instanceof List<?> list && list.size() == 1, "an array even for one prompt");
        JsonView entry = JsonView.of(((List<?>) parsed).get(0), "entry");
        assertEquals(0L, entry.i64("index"));
        List<?> vectors = (List<?>) entry.raw("embedding");
        assertEquals(1, vectors.size(), "a pooled model answers one vector per prompt");
        assertArrayEquals(first("{\"input\": \"hello\"}"), vector(vectors.get(0)), 1e-6f);
        Object two = ServerTestSupport.parsed(embeddings.nativeEmbedding("{\"input\": [\"a\", \"b\"]}").getEntity());
        assertEquals(2, ((List<?>) two).size());
        assertEquals(400, statusOf(() -> embeddings.nativeEmbeddings("{\"n\": 1}")));
    }

    @Test
    void concurrentEmbeddingsMatchTheSequentialOnes() throws Exception {
        String a = "{\"input\": \"the first input\"}";
        String b = "{\"input\": \"another, longer second input\"}";
        Object expectedA = view(embeddings.embeddings(a).getEntity()).raw("data");
        Object expectedB = view(embeddings.embeddings(b).getEntity()).raw("data");
        java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(4);
        try {
            List<java.util.concurrent.Future<Object>> results = new java.util.ArrayList<>();
            for (int i = 0; i < 16; i++) {
                String body = i % 2 == 0 ? a : b;
                results.add(pool.submit(() -> view(embeddings.embeddings(body).getEntity()).raw("data")));
            }
            for (int i = 0; i < results.size(); i++) {
                assertEquals(i % 2 == 0 ? expectedA : expectedB, results.get(i).get(),
                        "the model forwards run one at a time on the pump thread");
            }
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void aGenerativeModelRefusesPoolingByName() {
        assertTrue(errorOf(() -> generative.embeddings("{\"input\": \"hello\"}")).contains("pool"),
                "the embedding refusal names the pooling");
        assertTrue(errorOf(() -> generative.rerank("{\"query\": \"q\", \"documents\": [\"a\", \"b\"]}"))
                .contains("pool"), "the rerank refusal names the pooling");
        assertEquals(400, statusOf(() -> generative.rerankingAlias(
                "{\"query\": \"q\", \"documents\": [\"a\"]}")));
    }

    @Test
    void rerankValidatesItsBody() {
        assertEquals(400, statusOf(() -> generative.rerank("{\"query\": \"q\"}")));
        assertEquals(400, statusOf(() -> generative.rerank("{\"query\": \"q\", \"documents\": [42]}")));
        assertEquals(400, statusOf(() -> generative.rerank("{\"documents\": [\"a\"]}")));
    }
}
