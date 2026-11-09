package com.eonmux.neuralfree.server;

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
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The BGE-M3 multi-output model over the real service: {@code embed} answers the CLS-pooled dense vector, {@code
 * token_embed} the L2-normalized {@code cls.colbert} rows, {@code token_classify} vLLM's sparse {@code
 * {token_id: weight}} shape without the BOS/EOS rows, and {@code embed&token_classify} the concatenated dense
 * embedding and flattened weights. The process-reward default {@code classifier.step_tag_id} applies to a bare
 * {@code token_classify} request.
 */
public class PoolingMultiOutputTest {

    private static final int DIM = SyntheticModels.Config.bert().nEmbd();
    private static final int BOS = 2;
    private static final int EOS = 3;

    private static EngineService multiService;
    private static EngineService stepTagService;
    private static PoolingResource multi;
    private static PoolingResource stepTag;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.bgeM3().writeTempFile();
        multiService = new EngineService(model.toString(),
                config().backend(EngineConfig.BackendMode.CPU).build());
        multi = new PoolingResource(multiService);

        GgufTestWriter w = SyntheticModels.processReward();
        w.u32(w.archKey("classifier.step_tag_id"), 6);
        Path step = w.writeTempFile();
        stepTagService = new EngineService(step.toString(),
                config().backend(EngineConfig.BackendMode.CPU).build());
        stepTag = new PoolingResource(stepTagService);
    }

    @AfterAll
    static void close() {
        multiService.close();
        stepTagService.close();
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
    void theDefaultTaskIsTheDenseEmbedding() {
        assertTrue(multiService.pooling().tokenEmbeddingProjection());
        assertTrue(multiService.pooling().tokenScoreProjection());
        JsonView body = ok(multi.pooling("{\"input\": \"hello world\"}"));
        float[] vector = floats(at(body, "data", 0).raw("data"));
        assertEquals(DIM, vector.length, "the dense CLS embedding, not the per-token projection");
        double norm = 0;
        for (float v : vector) {
            norm += (double) v * v;
        }
        assertEquals(1.0, Math.sqrt(norm), 1e-3, "the pooled embedding is L2-normalized");
    }

    @Test
    void tokenEmbedAnswersTheNormalizedColbertRows() {
        JsonView body = ok(multi.pooling("{\"input\": [2, 5, 6, 3], \"task\": \"token_embed\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertEquals(3, rows.size(), "the leading BOS row drops; the token-embed filter leaves the EOS row");
        for (Object row : rows) {
            float[] values = floats(row);
            assertEquals(DIM, values.length, "the colbert projection's width");
            double norm = 0;
            for (float v : values) {
                norm += (double) v * v;
            }
            assertEquals(1.0, Math.sqrt(norm), 1e-3, "every token row is L2-normalized");
        }
    }

    @Test
    void tokenClassifyAnswersTheSparseTokenIdWeights() {
        JsonView body = ok(multi.pooling("{\"input\": [2, 5, 5, 7, 3], \"task\": \"token_classify\"}"));
        Object raw = at(body, "data", 0).raw("data");
        assertTrue(raw instanceof Map<?, ?>, "vLLM's sparse shape is a {token_id: weight} object, got " + raw);
        Map<?, ?> weights = (Map<?, ?>) raw;
        assertEquals(2, weights.size(), "the two distinct non-boundary ids, the repeated id merged");
        assertNotNull(weights.get("5"));
        assertNotNull(weights.get("7"));
        assertFalse(weights.containsKey(Integer.toString(BOS)), "the BOS row is not scored");
        assertFalse(weights.containsKey(Integer.toString(EOS)), "the EOS row is not scored");
        for (Object v : weights.values()) {
            assertTrue(((Number) v).floatValue() >= 0.0f, "the token pooler's ReLU keeps non-negative weights");
        }
    }

    @Test
    void theCombinedTaskConcatenatesTheEmbeddingAndTheTokenWeights() {
        JsonView pooled = ok(multi.pooling("{\"input\": [2, 5, 6, 3]}"));
        float[] embedding = floats(at(pooled, "data", 0).raw("data"));
        JsonView combined = ok(multi.pooling(
                "{\"input\": [2, 5, 6, 3], \"task\": \"embed&token_classify\"}"));
        float[] values = floats(at(combined, "data", 0).raw("data"));
        assertEquals(DIM + 2, values.length, "the dense vector then one weight per kept token");
        for (int i = 0; i < DIM; i++) {
            assertEquals(embedding[i], values[i], 1e-6, "the dense prefix is the embed task's vector");
        }
    }

    @Test
    void theSparseShapeStaysOnThePoolingRoute() {
        // the OpenAI and llama.cpp routes keep their dense-array protocols
        JsonView openAi = ok(multi.embeddings("{\"input\": \"hello world\"}"));
        float[] vector = floats(at(openAi, "data", 0).raw("embedding"));
        assertEquals(DIM, vector.length);
    }

    @Test
    void theCombinedTaskRefusesTheOtherModels() {
        assertTrue(errorOf(() -> stepTag.pooling("{\"input\": [5, 6], \"task\": \"embed&token_classify\"}"))
                .contains("embed&token_classify"));
    }

    @Test
    void theDefaultStepTagFiltersABareTokenClassify() {
        assertEquals(6, stepTagService.pooling().defaultStepTagId());
        JsonView body = ok(stepTag.pooling("{\"input\": [5, 6, 7], \"task\": \"token_classify\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertEquals(1, rows.size(), "only the token carrying the config's step id keeps its row");
        assertEquals(2, ((List<?>) rows.get(0)).size(), "both step scores stay");
    }

    @Test
    void anExplicitStepTagOverridesTheDefault() {
        JsonView body = ok(stepTag.pooling(
                "{\"input\": [5, 6, 7], \"task\": \"token_classify\", \"step_tag_id\": 7}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertEquals(1, rows.size());
    }

    @Test
    void theDefaultAppliesOnlyToTheTokenClassifyTask() {
        assertEquals(400, statusOf(() -> stepTag.pooling("{\"input\": [5, 6], \"task\": \"classify\"}")));
    }
}
