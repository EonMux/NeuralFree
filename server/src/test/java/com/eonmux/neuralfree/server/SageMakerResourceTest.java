package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * SageMaker's {@code /invocations} over the real service: a body is answered by the endpoint its fields name — a
 * chat or completion request by the generation routes, an {@code input} by the embedding route (classification on a
 * sequence classifier), a rerank by the rerank route and a score by the score route — and a body naming none refuses
 * with vLLM's list of expected request types.
 */
class SageMakerResourceTest {

    private static EngineService llama;
    private static EngineService bert;
    private static EngineService rank;
    private static EngineService unloaded;
    private static SageMakerResource generation;
    private static SageMakerResource embeddings;
    private static SageMakerResource classifier;
    private static SageMakerResource unavailable;

    @BeforeAll
    static void open() throws IOException {
        llama = new EngineService(SyntheticModels.llama().writeTempFile().toString(),
                ServerTestSupport.pagedConfig(64, 2));
        bert = new EngineService(SyntheticModels.bert().writeTempFile().toString(),
                ServerTestSupport.config().build());
        rank = new EngineService(SyntheticModels.rank("neg", "pos").writeTempFile().toString(),
                ServerTestSupport.config().build());
        unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2));
        generation = new SageMakerResource(llama);
        embeddings = new SageMakerResource(bert);
        classifier = new SageMakerResource(rank);
        unavailable = new SageMakerResource(unloaded);
    }

    @AfterAll
    static void close() {
        llama.close();
        bert.close();
        rank.close();
        unloaded.close();
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    @Test
    @Timeout(120)
    void aMessagesBodyAnswersAsAChatCompletion() {
        JsonView body = ok(generation.invocations("{\"messages\": [{\"role\": \"user\", \"content\":"
                + " \"hello\"}], \"max_tokens\": 4, \"temperature\": 0}"));
        assertEquals("chat.completion", body.str("object"));
        assertEquals("assistant", at(body, "choices", 0).obj("message").str("role"));
        assertEquals(4L, body.obj("usage").i64("completion_tokens"));
    }

    @Test
    @Timeout(120)
    void aPromptBodyAnswersAsACompletion() {
        JsonView body = ok(generation.invocations("{\"prompt\": \"hello\", \"max_tokens\": 4,"
                + " \"temperature\": 0}"));
        assertEquals("text_completion", body.str("object"));
        assertEquals(4L, body.obj("usage").i64("completion_tokens"));
    }

    @Test
    @Timeout(120)
    void anInputBodyAnswersAsAnEmbeddingOrAClassification() {
        JsonView embedding = ok(embeddings.invocations("{\"input\": [\"hello\", \"world\"]}"));
        assertEquals("list", embedding.str("object"));
        assertEquals(2, size(embedding, "data"));
        JsonView classification = ok(classifier.invocations("{\"input\": [\"good\", \"bad\"]}"));
        assertTrue(classification.str("id").startsWith("classify-"), classification.str("id"));
        assertEquals(2, size(classification, "data"));
        assertEquals(2, ((List<?>) at(classification, "data", 0).raw("probs")).size());
    }

    @Test
    @Timeout(120)
    void aQueryWithDocumentsAnswersAsARerankAndAScorePairAsAScore() {
        JsonView rerank = ok(classifier.invocations("{\"query\": \"q\", \"documents\": [\"a\", \"b\"]}"));
        assertTrue(rerank.str("id").startsWith("rerank-"), rerank.str("id"));
        assertEquals(2, size(rerank, "results"));
        assertEquals(rerank.obj("usage").i64("total_tokens"), rerank.obj("usage").i64("prompt_tokens"),
                "vLLM's rerank usage reports the prompt count under both names");
        JsonView score = ok(classifier.invocations("{\"queries\": [\"q\"], \"documents\": [\"a\", \"b\"]}"));
        assertTrue(score.str("id").startsWith("score-"), score.str("id"));
        assertEquals(2, size(score, "data"));
        assertTrue(at(score, "data", 0).f64("score") <= 1.0);
    }

    @Test
    @Timeout(120)
    void aTaskNamedInputAnswersAsTheGenericPoolingRequest() {
        JsonView body = ok(embeddings.invocations("{\"input\": \"hello\", \"task\": \"embed\"}"));
        assertTrue(body.str("id").startsWith("pooling-"), body.str("id"));
    }

    @Test
    @Timeout(120)
    void aBodyNamingNoTypeRefusesWithVllmsMessage() {
        String message = errorOf(() -> generation.invocations("{\"unrelated\": 1}"));
        assertTrue(message.contains("Cannot find suitable handler for request. Expected one of:"));
        assertTrue(message.contains("RerankRequest"));
        message = errorOf(() -> generation.invocations("{\"query\": \"q\"}"));
        assertTrue(message.contains("Cannot find suitable handler"), "a query without documents names no type");
        assertEquals(503, statusOf(() -> unavailable.invocations(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}")));
    }
}
