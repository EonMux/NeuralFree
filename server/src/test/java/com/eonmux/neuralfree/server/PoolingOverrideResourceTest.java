package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The pooling type override at the endpoints: a service launched with the override ({@code neuralfree.pooling},
 * llama.cpp {@code --pooling}, vLLM's pooler config) pools every consumer by it — {@code /v1/embeddings},
 * {@code /pooling} and {@code /score} — and a {@code rank} override on a model without a classifier head refuses
 * the launch by name.
 */
public class PoolingOverrideResourceTest {

    private static EngineService bert;
    private static EngineService clsBert;
    private static PoolingResource embeddings;
    private static PoolingResource clsEmbeddings;

    @BeforeAll
    static void open() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        bert = new EngineService(bertModel.toString(), ServerTestSupport.config().build());
        clsBert = new EngineService(bertModel.toString(),
                ServerTestSupport.config().pooling(PoolingKind.CLS).build());
        embeddings = new PoolingResource(bert);
        clsEmbeddings = new PoolingResource(clsBert);
    }

    @AfterAll
    static void close() {
        bert.close();
        clsBert.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static float[] vector(Object embedding) {
        List<?> values = (List<?>) embedding;
        float[] out = new float[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) values.get(i)).floatValue();
        }
        return out;
    }

    private static float[] first(PoolingResource resource, String body) {
        return vector(at(view(resource.embeddings(body).getEntity()), "data", 0).raw("embedding"));
    }

    @Test
    void theOverrideCarriesIntoTheServiceKind() {
        assertEquals(PoolingKind.MEAN, bert.pooling().poolingKind(), "the GGUF's MEAN without the override");
        assertEquals(PoolingKind.CLS, clsBert.pooling().poolingKind());
    }

    @Test
    void embeddingsPoolByTheOverride() {
        float[] mean = first(embeddings, "{\"input\": \"hello world\"}");
        float[] cls = first(clsEmbeddings, "{\"input\": \"hello world\"}");
        assertEquals(mean.length, cls.length, "the width does not change");
        assertFalse(Arrays.equals(mean, cls), "CLS pools the first row, MEAN the average");
    }

    @Test
    void thePoolingTaskFollowsTheOverride() {
        // The default task of the overridden model is the pooled embed, exactly the base model's shape.
        JsonView body = ok(clsEmbeddings.pooling("{\"input\": [\"hello world\"]}"));
        assertEquals(1, size(body, "data"), "the overridden model answers its pooling task");
        assertEquals("pooling", at(body, "data", 0).str("object"));
        assertEquals(400, statusOf(() -> clsEmbeddings.pooling(
                "{\"input\": [\"hello world\"], \"task\": \"token_embed\"}")),
                "token_embed needs per-token pooling");
        assertTrue(errorOf(() -> clsEmbeddings.pooling("{\"input\": [\"hello world\"], \"task\": \"token_embed\"}"))
                .contains("CLS"));
    }

    @Test
    void scoreServesTheOverriddenEmbeddings() {
        JsonView body = ok(clsEmbeddings.score("{\"queries\": \"q\", \"documents\": [\"d one\", \"d two\"]}"));
        assertEquals(2, size(body, "data"));
    }

    /**
     * A MEAN-pooling GGUF that stores a classifier head reranks through that head (cross-encoder) under the
     * {@code rank} override, where the same model without it reranks by the cosine of its MEAN embeddings
     * (bi-encoder).
     */
    @Test
    void rankOverrideServesRerankOnAClassifierModel() throws IOException {
        GgufTestWriter w = SyntheticModels.rank("yes");
        String key = w.archKey("pooling_type");
        Path model = w.remove(key).u32(key, SyntheticModels.POOLING_MEAN).writeTempFile();
        String request = "{\"query\": \"q\", \"documents\": [\"a one\", \"b two\"]}";
        Map<Long, Double> biEncoder;
        EngineService mean = new EngineService(model.toString(), ServerTestSupport.config().build());
        try {
            assertEquals(PoolingKind.MEAN, mean.pooling().poolingKind());
            biEncoder = relevanceScores(ok(new PoolingResource(mean).rerank(request)));
            assertEquals(2, biEncoder.size());
        } finally {
            mean.close();
        }
        EngineService rank = new EngineService(model.toString(),
                ServerTestSupport.config().pooling(PoolingKind.RANK).build());
        try {
            assertEquals(PoolingKind.RANK, rank.pooling().poolingKind());
            JsonView body = ok(new PoolingResource(rank).rerank(request));
            assertEquals(2, size(body, "results"));
            assertNotEquals(biEncoder, relevanceScores(body), "the classifier head scores, not the cosine");
        } finally {
            rank.close();
        }
    }

    /** The relevance score of each document of a rerank response, by document index. */
    private static Map<Long, Double> relevanceScores(JsonView body) {
        Map<Long, Double> scores = new java.util.TreeMap<>();
        for (int i = 0; i < size(body, "results"); i++) {
            JsonView result = at(body, "results", i);
            scores.put(result.i64("index"), ((Number) result.raw("relevance_score")).doubleValue());
        }
        return scores;
    }

    /** The {@code neuralfree.pooling} property, which the launch flags write, reaches the service's model. */
    @Test
    void thePropertyRouteReachesTheService() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        java.util.Properties p = new java.util.Properties();
        p.setProperty(EngineConfig.PROP_POOLING, "last");
        EngineConfig fromProperty = EngineConfig.fromProperties(p);
        EngineService last = new EngineService(bertModel.toString(),
                ServerTestSupport.config().pooling(fromProperty.pooling()).build());
        try {
            assertEquals(PoolingKind.LAST, last.pooling().poolingKind());
            float[] lastRow = first(new PoolingResource(last), "{\"input\": \"hello world\"}");
            float[] meanRow = first(embeddings, "{\"input\": \"hello world\"}");
            assertFalse(Arrays.equals(meanRow, lastRow), "LAST pools the last row, MEAN the average");
        } finally {
            last.close();
        }
    }

    @Test
    void rankOverrideOnAClassifierLessModelRefusesTheLaunchByName() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        EngineConfig rank = ServerTestSupport.config().pooling(PoolingKind.RANK).build();
        RuntimeException e = assertThrows(RuntimeException.class,
                () -> new EngineService(bertModel.toString(), rank));
        assertTrue(e.getMessage().contains("rank"), e.getMessage());
        assertTrue(e.getMessage().contains("classifier"), e.getMessage());
    }
}
