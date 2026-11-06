package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpServerRequest;
import io.vertx.core.http.HttpServerResponse;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;

import static com.eonmux.neuralfree.server.ServerTestSupport.config;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The response id of every pooling route: the {@code X-Request-Id} header names it first (vLLM's {@code
 * _base_request_id}), the body {@code request_id} when the header is absent, and a fresh id when neither names
 * one. {@code /v2/embed} reads the header alone, as vLLM's Cohere protocol carries no {@code request_id}.
 */
public class PoolingRequestIdHeaderTest {

    private static EngineService bert;
    private static EngineService rank;
    private static PoolingResource pooled;
    private static PoolingResource classifier;
    private static CohereEmbedResource cohere;

    @BeforeAll
    static void open() throws IOException {
        Path bertModel = SyntheticModels.bert().writeTempFile();
        bert = new EngineService(bertModel.toString(), config().build());
        Path rankModel = SyntheticModels.rank("neg", "pos").writeTempFile();
        rank = new EngineService(rankModel.toString(), config().build());
        pooled = new PoolingResource(bert);
        classifier = new PoolingResource(rank);
        cohere = new CohereEmbedResource(bert);
    }

    @AfterAll
    static void close() {
        bert.close();
        rank.close();
    }

    /** An HTTP request stub whose {@code headers()} answers the given {@code X-Request-Id}. */
    private static HttpServerRequest stub(String value) {
        MultiMap headers = MultiMap.caseInsensitiveMultiMap();
        if (value != null) {
            headers.add("X-Request-Id", value);
        }
        Object response = Proxy.newProxyInstance(PoolingRequestIdHeaderTest.class.getClassLoader(),
                new Class<?>[]{HttpServerResponse.class},
                (proxy, method, args) -> null);
        return (HttpServerRequest) Proxy.newProxyInstance(PoolingRequestIdHeaderTest.class.getClassLoader(),
                new Class<?>[]{HttpServerRequest.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("headers")) {
                        return headers;
                    }
                    if (method.getName().equals("response")) {
                        return response;
                    }
                    Class<?> type = method.getReturnType();
                    if (type == boolean.class) {
                        return false;
                    }
                    if (type == int.class) {
                        return 0;
                    }
                    if (type == long.class) {
                        return 0L;
                    }
                    return null;
                });
    }

    private static void header(String value) {
        HttpServerRequest http = stub(value);
        pooled.http = http;
        classifier.http = http;
        cohere.http = http;
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }

    @Test
    void theHeaderNamesEveryPoolingResponse() {
        header("hdr-1");
        try {
            assertEquals("embd-hdr-1", ok(pooled.embeddings("{\"input\": \"a\"}")).str("id"));
            assertEquals("score-hdr-1", ok(classifier.score(
                    "{\"queries\": \"a\", \"documents\": \"b\"}")).str("id"));
            assertEquals("rerank-hdr-1", ok(classifier.rerank(
                    "{\"query\": \"a\", \"documents\": [\"b\"]}")).str("id"));
            assertEquals("classify-hdr-1", ok(classifier.classify("{\"input\": \"a\"}")).str("id"));
            assertEquals("pooling-hdr-1", ok(pooled.pooling("{\"input\": \"a\"}")).str("id"));
            assertEquals("embd-hdr-1", ok(cohere.embed("{\"texts\": [\"a\"]}")).str("id"));
        } finally {
            header(null);
        }
    }

    @Test
    void theHeaderWinsOverTheBodyRequestId() {
        header("hdr-2");
        try {
            assertEquals("score-hdr-2", ok(classifier.score(
                    "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": \"body-2\"}")).str("id"));
        } finally {
            header(null);
        }
        assertEquals("score-body-2", ok(classifier.score(
                "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": \"body-2\"}")).str("id"),
                "without the header the body names the response");
    }

    @Test
    void anEmptyHeaderFallsBackToTheBody() {
        header("");
        try {
            assertEquals("score-body-3", ok(classifier.score(
                    "{\"queries\": \"a\", \"documents\": \"b\", \"request_id\": \"body-3\"}")).str("id"));
        } finally {
            header(null);
        }
    }

    @Test
    void withoutEitherTheIdIsFresh() {
        header(null);
        String named = ok(classifier.score("{\"queries\": \"a\", \"documents\": \"b\"}"))
                .str("id").substring("score-".length());
        assertTrue(named.matches("\\d+"), "the fresh id is the engine's: " + named);
    }
}
