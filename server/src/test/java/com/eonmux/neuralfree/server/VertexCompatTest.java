package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Vertex AI prediction protocol: the {@code AIP_*} variables that enable and name the routes, the
 * {@code @requestFormat} dispatch of each instance to its endpoint (path or camelCase alias), the per-instance error
 * envelopes the predictions carry, and the envelope refusals.
 */
class VertexCompatTest {

    private static EngineService llama;
    private static EngineService bert;
    private static VertexCompat generation;
    private static VertexCompat embeddings;

    @BeforeAll
    static void open() throws IOException {
        llama = new EngineService(SyntheticModels.llama().writeTempFile().toString(),
                ServerTestSupport.pagedConfig(64, 2));
        bert = new EngineService(SyntheticModels.bert().writeTempFile().toString(),
                ServerTestSupport.config().build());
        generation = new VertexCompat(llama, null);
        embeddings = new VertexCompat(bert, null);
    }

    @AfterAll
    static void close() {
        llama.close();
        bert.close();
    }

    @Test
    void theProtocolIsEnabledOnlyInPredictionMode() {
        assertTrue(VertexCompat.enabled(Map.of(VertexCompat.MODE_ENV, "PREDICTION"), new Properties()));
        assertFalse(VertexCompat.enabled(Map.of(VertexCompat.MODE_ENV, "SERVING"), new Properties()));
        assertFalse(VertexCompat.enabled(Map.of(), new Properties()));
        Properties property = new Properties();
        property.setProperty(VertexCompat.MODE_PROPERTY, "PREDICTION");
        assertTrue(VertexCompat.enabled(Map.of(), property), "a -D launch can enable it without the environment");
    }

    @Test
    void thePathsDefaultAndCanonicalize() {
        assertEquals("/predict", VertexCompat.predictPath(Map.of(), new Properties()));
        assertEquals("/invoke", VertexCompat.predictPath(Map.of(VertexCompat.PREDICT_ENV, "invoke"), new Properties()));
        assertNull(VertexCompat.healthPath(Map.of(), new Properties()));
        assertEquals("/healthz", VertexCompat.healthPath(
                Map.of(VertexCompat.HEALTH_ENV, "/healthz"), new Properties()));
        Properties property = new Properties();
        property.setProperty(VertexCompat.PREDICT_PROPERTY, "/v1/predict");
        assertEquals("/v1/predict", VertexCompat.predictPath(Map.of(), property));
        assertEquals("/v1/predict", VertexCompat.canonical(VertexCompat.PREDICT_ENV, "v1/predict"));
    }

    @Test
    void theContainersPortOverridesTheServersInPredictionMode() {
        Map<String, String> properties = new LinkedHashMap<>();
        properties.put("quarkus.http.port", "8080");
        ServerLaunchArgs.vertexPort(properties, Map.of(VertexCompat.MODE_ENV, "PREDICTION",
                "AIP_HTTP_PORT", "9090"));
        assertEquals("9090", properties.get("quarkus.http.port"));
        Map<String, String> serving = new LinkedHashMap<>();
        ServerLaunchArgs.vertexPort(serving, Map.of("AIP_HTTP_PORT", "9090"));
        assertTrue(serving.isEmpty(), "without prediction mode the port is the server's");
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> ServerLaunchArgs.vertexPort(new LinkedHashMap<>(), Map.of(VertexCompat.MODE_ENV,
                        "PREDICTION", "AIP_HTTP_PORT", "70000")));
        assertTrue(refused.getMessage().startsWith("AIP_HTTP_PORT"));
    }

    @Test
    void unusablePathsRefuseNamingTheVariable() {
        for (String value : new String[]{"", "//", "a b", "a?b", "a#b", "a*", "a:b", ".."}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> VertexCompat.canonical(VertexCompat.PREDICT_ENV, value), value);
            assertTrue(e.getMessage().startsWith(VertexCompat.PREDICT_ENV), value);
        }
    }

    @Test
    @Timeout(120)
    void instancesDispatchByFormatAndPathIntoThePredictionsArray() {
        JsonView body = ok(generation.predict("{\"instances\": ["
                + "{\"@requestFormat\": \"chatCompletions\", \"messages\": [{\"role\": \"user\","
                + " \"content\": \"hello\"}], \"max_tokens\": 4, \"temperature\": 0},"
                + "{\"@requestFormat\": \"/v1/completions\", \"prompt\": \"hello\", \"max_tokens\": 4,"
                + " \"temperature\": 0}]}"));
        List<?> predictions = (List<?>) body.raw("predictions");
        assertEquals(2, predictions.size());
        JsonView chat = JsonView.of(predictions.get(0), "predictions[0]");
        assertEquals("chat.completion", chat.str("object"));
        JsonView completion = JsonView.of(predictions.get(1), "predictions[1]");
        assertEquals("text_completion", completion.str("object"));
    }

    @Test
    @Timeout(120)
    void anInstanceErrorIsItsPrediction() {
        JsonView body = ok(generation.predict("{\"instances\": ["
                + "{\"@requestFormat\": \"nope\"},"
                + "{\"messages\": []},"
                + "{\"@requestFormat\": \"chatCompletions\", \"stream\": true,"
                + " \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]},"
                + "{\"@requestFormat\": \"chatCompletions\", \"messages\": []}]}"));
        List<?> predictions = (List<?>) body.raw("predictions");
        assertEquals(4, predictions.size());
        assertTrue(JsonView.of(predictions.get(0), "predictions[0]").obj("error").str("message")
                .contains("no handler registered for @requestFormat: nope"));
        assertTrue(JsonView.of(predictions.get(1), "predictions[1]").obj("error").str("message")
                .contains("@requestFormat"));
        assertTrue(JsonView.of(predictions.get(2), "predictions[2]").obj("error").str("message")
                .contains("streaming is not supported"));
        assertTrue(JsonView.of(predictions.get(3), "predictions[3]").obj("error").str("message")
                .contains("messages is required"), "the endpoint's own refusal becomes the prediction");
    }

    @Test
    @Timeout(120)
    void embeddingsDispatchToo() {
        JsonView body = ok(embeddings.predict("{\"instances\": [{\"@requestFormat\": \"embeddings\","
                + " \"input\": [\"hello\"]}]}"));
        JsonView prediction = JsonView.of(((List<?>) body.raw("predictions")).get(0), "predictions[0]");
        assertEquals("list", prediction.str("object"));
        assertEquals(1, ((List<?>) prediction.raw("data")).size());
    }

    @Test
    void anEmptyEnvelopeAnswersAnEmptyPredictionsArray() {
        JsonView body = ok(generation.predict("{\"instances\": []}"));
        assertEquals(0, ((List<?>) body.raw("predictions")).size());
    }

    @Test
    void aMalformedEnvelopeRefuses() {
        assertTrue(errorOf(() -> generation.predict("{\"instances\": 3}"))
                .contains("array field named instances"));
        assertTrue(errorOf(() -> generation.predict("{}")).contains("instances"));
        StringBuilder tooMany = new StringBuilder("{\"instances\": [");
        for (int i = 0; i <= VertexCompat.MAX_INSTANCES; i++) {
            tooMany.append(i == 0 ? "" : ",").append("{\"@requestFormat\": \"chatCompletions\"}");
        }
        tooMany.append("]}");
        assertTrue(errorOf(() -> generation.predict(tooMany.toString()))
                .contains("exceeds maximum size of " + VertexCompat.MAX_INSTANCES));
    }

    @Test
    void theEnvelopeIsAPredictionResponse() {
        Response response = generation.predict("{\"instances\": []}");
        assertEquals(200, response.getStatus());
        assertEquals("application/json", response.getMediaType().toString());
    }

    private static JsonView ok(Response response) {
        assertEquals(200, response.getStatus(), () -> String.valueOf(response.getEntity()));
        return view(response.getEntity());
    }
}
