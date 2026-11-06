package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The ColBERT late-interaction and SPLADE sparse heads over the real service: a per-token embedding model answers
 * {@code token_embed} with L2-normalized projection rows and scores /score and /rerank by the MaxSim of the two
 * token-row sets; a SPLADE model answers the {@code embed} task with the sparse {@code log1p(relu(x))} reduction
 * and refuses the per-token tasks by name.
 */
public class PoolingSparseColBertTest {

    private static EngineService colbertService;
    private static EngineService spladeMaxService;
    private static EngineService spladeSumService;
    private static PoolingResource colbert;
    private static PoolingResource spladeMax;
    private static PoolingResource spladeSum;

    @BeforeAll
    static void open() throws IOException {
        // the CPU backend keeps every forward of these fixtures bitwise identical, which the exact
        // reduction and MaxSim comparisons below compare across the two SPLADE services
        Path colbertModel = SyntheticModels.colbert().writeTempFile();
        colbertService = new EngineService(colbertModel.toString(),
                config().backend(EngineConfig.BackendMode.CPU).build());
        Path spladeMaxModel = SyntheticModels.splade("log1p_relu_max").writeTempFile();
        spladeMaxService = new EngineService(spladeMaxModel.toString(),
                config().backend(EngineConfig.BackendMode.CPU).build());
        Path spladeSumModel = SyntheticModels.splade("log1p_relu_sum").writeTempFile();
        spladeSumService = new EngineService(spladeSumModel.toString(),
                config().backend(EngineConfig.BackendMode.CPU).build());
        colbert = new PoolingResource(colbertService);
        spladeMax = new PoolingResource(spladeMaxService);
        spladeSum = new PoolingResource(spladeSumService);
    }

    @AfterAll
    static void close() {
        colbertService.close();
        spladeMaxService.close();
        spladeSumService.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static float[] floats(Object raw) {
        List<?> list = (List<?>) raw;
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).floatValue();
        }
        return out;
    }

    @Test
    void colbertProjectsAndNormalizesEveryTokenRow() {
        assertTrue(colbertService.perTokenEmbeddings());
        JsonView body = ok(colbert.pooling("{\"input\": \"hello there\", \"task\": \"token_embed\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertTrue(rows.size() >= 1, "one row per token");
        for (Object row : rows) {
            float[] values = floats(row);
            assertEquals(SyntheticModels.Config.bert().nEmbd(), values.length, "one value per projection dim");
            double sum = 0;
            for (float v : values) {
                sum += (double) v * v;
            }
            assertEquals(1.0, Math.sqrt(sum), 1e-3, "every row is L2-normalized");
        }
    }

    @Test
    void colbertScoresByLateInteraction() {
        JsonView body = ok(colbert.score("{\"queries\": \"a red car\", \"documents\": [\"a red car\", \"an ocean\"]}"));
        assertEquals(2, size(body, "data"));
        double same = at(body, "data", 0).f64("score");
        double other = at(body, "data", 1).f64("score");
        assertTrue(same > other, "the identical text scores higher than a different one, got " + same + " / " + other);
        JsonView again = ok(colbert.score("{\"queries\": \"a red car\", \"documents\": [\"an ocean\"]}"));
        assertEquals(other, at(again, "data", 0).f64("score"), 1e-6, "the score is deterministic");
    }

    @Test
    void colbertReranksByTheSameMaxSim() {
        JsonView reranked = ok(colbert.rerank(
                "{\"query\": \"a red car\", \"documents\": [\"an ocean\", \"a red car\"], \"top_n\": 2}"));
        assertEquals(1, at(reranked, "results", 0).i64("index"), "the identical text ranks first");
        double score = at(reranked, "results", 0).f64("relevance_score");
        double scored = at(ok(colbert.score("{\"queries\": \"a red car\", \"documents\": [\"a red car\"]}")),
                "data", 0).f64("score");
        assertEquals(scored, score, 1e-6, "rerank and score answer the same MaxSim");
        assertTrue(errorOf(() -> colbert.rerank(
                        "{\"query\": \"q\", \"documents\": [\"d\"], \"instruction\": \"find\"}"))
                .contains("late-interaction"));
    }

    @Test
    void colbertRefusesTheOtherEmbeddingOutputs() {
        assertEquals(400, statusOf(() -> colbert.embeddings("{\"input\": \"hello\"}")));
        assertTrue(errorOf(() -> colbert.pooling("{\"input\": \"hello\", \"task\": \"embed\"}"))
                .contains("pooled embedding"));
    }

    private static Map<String, Double> weights(Object raw) {
        Map<?, ?> map = (Map<?, ?>) raw;
        Map<String, Double> out = new java.util.LinkedHashMap<>();
        for (Map.Entry<?, ?> e : map.entrySet()) {
            out.put((String) e.getKey(), ((Number) e.getValue()).doubleValue());
        }
        return out;
    }

    @Test
    void spladeAnswersTheEmbedTaskByDefault() {
        assertTrue(spladeMaxService.sparsePooling() != null);
        JsonView body = ok(spladeMax.pooling("{\"input\": \"hello world\"}"));
        Map<String, Double> vector = weights(at(body, "data", 0).raw("data"));
        assertFalse(vector.isEmpty(), "the vocabulary's active entries");
        for (double v : vector.values()) {
            assertTrue(v > 0.0, "log1p(relu(x)) is non-negative and zeros are omitted");
        }
        assertEquals(400, statusOf(() -> spladeMax.pooling("{\"input\": \"hello\", \"task\": \"token_classify\"}")));
        assertEquals(200, statusOf(() -> spladeMax.pooling("{\"input\": \"hello\", \"task\": \"token_embed\"}")),
                "vLLM's SPLADE pooler also answers the plain token_embed rows");
        assertEquals(200, statusOf(() -> spladeMax.embeddings("{\"input\": \"hello\"}")));
    }

    @Test
    void spladeTokenEmbedAnswersTheNormalizedEmbeddingRows() {
        JsonView body = ok(spladeMax.pooling("{\"input\": \"hello world\", \"task\": \"token_embed\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        int width = SyntheticModels.Config.bert().nEmbd();
        for (Object row : rows) {
            float[] values = floats(row);
            assertEquals(width, values.length,
                    "the token-embed rows hold the embedding chain's width, not the vocabulary");
            double sum = 0;
            for (float v : values) {
                sum += (double) v * v;
            }
            assertEquals(1.0, Math.sqrt(sum), 1e-3, "vLLM's token_embed pooler L2-normalizes every row");
        }
    }

    @Test
    void spladeWeightsStayRawAndRefuseTheShapingFields() {
        double maxNormSquared = 0;
        for (double v : weights(at(ok(spladeMax.pooling("{\"input\": \"hello world\"}")), "data", 0)
                .raw("data")).values()) {
            maxNormSquared += v * v;
        }
        assertTrue(Math.sqrt(maxNormSquared) > 1.0, "the max weights keep their raw scale, not the unit sphere"
                + " (norm " + Math.sqrt(maxNormSquared) + ")");
        double largestSum = 0;
        for (double v : weights(at(ok(spladeSum.pooling("{\"input\": \"hello world\"}")), "data", 0)
                .raw("data")).values()) {
            largestSum = Math.max(largestSum, v);
        }
        assertTrue(largestSum > 1.0, "the sum weights keep their raw scale, not the unit sphere (largest "
                + largestSum + ")");
        assertTrue(errorOf(() -> spladeMax.pooling("{\"input\": \"hello\", \"embd_normalize\": 2}"))
                .contains("embd_normalize"));
        assertTrue(errorOf(() -> spladeMax.embeddings("{\"input\": \"hello\", \"dimensions\": 8}"))
                .contains("dimensions"));
    }

    @Test
    void spladeSumReductionDominatesTheMax() {
        Map<String, Double> max = weights(at(ok(spladeMax.pooling("{\"input\": \"hello world\"}")), "data", 0)
                .raw("data"));
        Map<String, Double> sum = weights(at(ok(spladeSum.pooling("{\"input\": \"hello world\"}")), "data", 0)
                .raw("data"));
        for (Map.Entry<String, Double> e : max.entrySet()) {
            assertTrue(sum.getOrDefault(e.getKey(), 0.0) >= e.getValue() - 1e-6, "the sum over non-negative"
                    + " weights is at least their max at token " + e.getKey());
        }
    }
}
