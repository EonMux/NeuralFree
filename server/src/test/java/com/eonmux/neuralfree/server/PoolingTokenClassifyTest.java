package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
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
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * vLLM's classifier heads over the real service: a token classifier answers {@code token_classify} with one score
 * row per token, the reward and process-reward models do the same with their one-value and step-score rows, a
 * one-output sequence classifier answers {@code classify} with its raw score (a softmax of one would flatten it),
 * and a pooled sequence classifier answers {@code classify} and scores pairs as a cross-encoder. Each task refuses
 * the other models' tasks by name.
 */
public class PoolingTokenClassifyTest {

    private static EngineService tokenClassifier;
    private static EngineService reward;
    private static EngineService process;
    private static EngineService oneOutput;
    private static EngineService sequence;
    private static PoolingResource token;
    private static PoolingResource scored;
    private static PoolingResource steps;
    private static PoolingResource single;
    private static PoolingResource pooled;

    @BeforeAll
    static void open() throws IOException {
        Path tokenModel = SyntheticModels.tokenClassifier("O", "B-X", "I-X").writeTempFile();
        tokenClassifier = new EngineService(tokenModel.toString(), ServerTestSupport.config().build());
        Path rewardModel = SyntheticModels.reward().writeTempFile();
        reward = new EngineService(rewardModel.toString(), ServerTestSupport.config().build());
        Path processModel = SyntheticModels.processReward().writeTempFile();
        process = new EngineService(processModel.toString(), ServerTestSupport.config().build());
        Path oneOutputModel = SyntheticModels.rank("reward").writeTempFile();
        oneOutput = new EngineService(oneOutputModel.toString(), ServerTestSupport.config().build());
        Path sequenceModel = SyntheticModels.sequenceClassify("neg", "pos").writeTempFile();
        sequence = new EngineService(sequenceModel.toString(), ServerTestSupport.config().build());
        token = new PoolingResource(tokenClassifier);
        scored = new PoolingResource(reward);
        steps = new PoolingResource(process);
        single = new PoolingResource(oneOutput);
        pooled = new PoolingResource(sequence);
    }

    @AfterAll
    static void close() {
        tokenClassifier.close();
        reward.close();
        process.close();
        oneOutput.close();
        sequence.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus());
        return view(response.getEntity());
    }

    private static int listSize(Object raw) {
        return ((List<?>) raw).size();
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
    void tokenClassifierScoresEveryToken() {
        assertTrue(tokenClassifier.pooling().tokenClassifier());
        JsonView body = ok(token.pooling("{\"input\": \"hello\", \"task\": \"token_classify\"}"));
        assertEquals(1, size(body, "data"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertTrue(rows.size() >= 1, "one row per token");
        for (Object row : rows) {
            assertEquals(3, listSize(row), "one score per classifier label");
            floats(row);
        }
        assertTrue(at(body, "data", 0).raw("data") instanceof List, "the rows are JSON vectors");
    }

    @Test
    void tokenClassifierIsTheDefaultPoolingTask() {
        JsonView body = ok(token.pooling("{\"input\": \"hello\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertEquals(3, listSize(rows.get(0)));
        JsonView base64 = ok(token.pooling(
                "{\"input\": \"hello\", \"task\": \"token_classify\", \"encoding_format\": \"base64\"}"));
        assertTrue(at(base64, "data", 0).raw("data") instanceof String);
    }

    @Test
    void tokenClassifierRefusesTheEmbeddingAndSequenceTasks() {
        assertTrue(errorOf(() -> token.pooling("{\"input\": \"hello\", \"task\": \"embed\"}"))
                .contains("classif"));
        assertTrue(errorOf(() -> token.pooling("{\"input\": \"hello\", \"task\": \"token_embed\"}"))
                .contains("token_embed"));
        assertTrue(errorOf(() -> token.pooling("{\"input\": \"hello\", \"task\": \"classify\"}"))
                .contains("token_classify"));
        assertEquals(400, statusOf(() -> token.embeddings("{\"input\": \"hello\"}")));
        assertEquals(400, statusOf(() -> token.classify("{\"input\": \"hello\"}")));
    }

    @Test
    void rewardModelScoresOneValuePerToken() {
        assertTrue(reward.pooling().tokenClassifier());
        JsonView body = ok(scored.pooling("{\"input\": [\"hello\", \"goodbye\"]}"));
        assertEquals(2, size(body, "data"));
        for (int i = 0; i < 2; i++) {
            List<?> rows = (List<?>) at(body, "data", i).raw("data");
            assertTrue(rows.size() >= 1, "one row per token");
            for (Object row : rows) {
                assertEquals(1, listSize(row), "a reward head has one score per token");
                floats(row);
            }
        }
        assertEquals(400, statusOf(() -> scored.classify("{\"input\": \"hello\"}")));
        assertEquals(400, statusOf(() -> scored.embeddings("{\"input\": \"hello\"}")));
        assertTrue(errorOf(() -> scored.pooling("{\"input\": \"hello\", \"task\": \"classify\"}"))
                .contains("token_classify"));
    }

    @Test
    void processRewardScoresAStepRowPerToken() {
        assertTrue(process.pooling().tokenClassifier());
        JsonView body = ok(steps.pooling("{\"input\": \"hello\", \"task\": \"token_classify\"}"));
        List<?> rows = (List<?>) at(body, "data", 0).raw("data");
        assertTrue(rows.size() >= 1, "one row per token");
        for (Object row : rows) {
            assertEquals(2, listSize(row), "two step scores per token");
            floats(row);
        }
        assertTrue(errorOf(() -> steps.pooling("{\"input\": \"hello\", \"task\": \"classify\"}"))
                .contains("token_classify"));
    }

    @Test
    void oneOutputSequenceClassifierReturnsTheRawScore() {
        assertTrue(oneOutput.pooling().sequenceClassifier());
        JsonView classified = ok(single.classify("{\"input\": \"hello\"}"));
        JsonView entry = at(classified, "data", 0);
        assertEquals(1, entry.i64("num_classes"));
        float[] scores = floats(entry.raw("probs"));
        assertEquals(1, scores.length, "a one-output head has one score");
        assertNotEquals(1.0f, scores[0], 1e-6f, "a softmax of one output would flatten the score to 1");
        JsonView body = ok(single.pooling("{\"input\": \"hello\", \"task\": \"classify\"}"));
        assertEquals(scores[0], floats(at(body, "data", 0).raw("data"))[0], 1e-6f,
                "classify and /pooling agree on the raw score");
    }

    @Test
    void sequenceClassifierAnswersClassifyAndScoresPairs() {
        assertTrue(sequence.pooling().sequenceClassifier());
        JsonView body = ok(pooled.pooling("{\"input\": [2, 4, 5]}"));
        assertEquals(2, listSize(at(body, "data", 0).raw("data")), "one score per label");
        JsonView classified = ok(pooled.classify("{\"input\": \"hello\"}"));
        assertEquals(1, size(classified, "data"));
        assertEquals(2, listSize(at(classified, "data", 0).raw("probs")), "one probability per label");
        JsonView tokens = ok(pooled.pooling("{\"input\": [2, 4, 5], \"task\": \"token_classify\"}"));
        List<?> rows = (List<?>) at(tokens, "data", 0).raw("data");
        assertEquals(3, rows.size(), "the sequence classifier also scores every token");
        for (Object row : rows) {
            assertEquals(2, listSize(row), "one score per label");
        }
        double reranked = at(ok(pooled.rerank(
                "{\"query\": \"q\", \"documents\": [\"d\"], \"return_documents\": true}")),
                "results", 0).f64("relevance_score");
        double score = at(ok(pooled.score("{\"queries\": \"q\", \"documents\": [\"d\"]}")),
                "data", 0).f64("score");
        assertEquals(reranked, score, 1e-9, "the sequence classifier scores both endpoints alike");
    }
}
