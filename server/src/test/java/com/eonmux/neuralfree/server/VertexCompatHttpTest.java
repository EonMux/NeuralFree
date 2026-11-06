package com.eonmux.neuralfree.server;

import io.quarkus.test.common.QuarkusTestResource;
import io.quarkus.test.common.QuarkusTestResourceLifecycleManager;
import io.quarkus.test.common.http.TestHTTPResource;
import io.quarkus.test.junit.QuarkusTest;
import io.quarkus.test.junit.QuarkusTestProfile;
import io.quarkus.test.junit.TestProfile;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The Vertex AI routes on the real HTTP server: in prediction mode the configured prediction and health paths are
 * registered on the Vert.x router (the health path is not the server's own {@code /health}), the prediction route
 * reads the body and answers {@code {"predictions": [...]}}, a malformed envelope is a 400, and a prediction that
 * cannot be served still arrives in the array as its error envelope.
 */
@QuarkusTest
@QuarkusTestResource(value = VertexCompatHttpTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(VertexCompatHttpTest.PredictionProfile.class)
class VertexCompatHttpTest {

    private static final String PREDICT = "/vertex-predict";
    private static final String HEALTH = "/vertex-health";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    /** The values the launch properties had before this class set them, restored before a later class reads them. */
    private static final Map<String, String> PREVIOUS = new LinkedHashMap<>();

    @TestHTTPResource
    URI base;

    private static HttpClient client;

    @BeforeAll
    static void open() {
        client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    @AfterAll
    static void restoreLaunchProperties() {
        PREVIOUS.forEach((name, value) -> {
            if (value == null) {
                System.clearProperty(name);
            } else {
                System.setProperty(name, value);
            }
        });
    }

    @Test
    void theConfiguredPredictionPathAnswersTheEnvelope() throws Exception {
        HttpResponse<String> response = post(PREDICT, "{\"instances\": []}");
        assertEquals(200, response.statusCode(), response.body());
        assertTrue(response.headers().firstValue("Content-Type").orElse("").startsWith("application/json"));
        assertTrue(response.body().contains("\"predictions\":[]"), response.body());
    }

    @Test
    void anInstanceErrorIsStillAPrediction() throws Exception {
        HttpResponse<String> response = post(PREDICT,
                "{\"instances\": [{\"@requestFormat\": \"nope\"},"
                        + " {\"@requestFormat\": \"chatCompletions\", \"messages\": []}]}");
        assertEquals(200, response.statusCode(), response.body());
        assertTrue(response.body().contains("no handler registered for @requestFormat: nope"), response.body());
        assertTrue(response.body().contains("no model loaded"), response.body());
    }

    @Test
    void aMalformedEnvelopeIsA400() throws Exception {
        HttpResponse<String> response = post(PREDICT, "{\"bad\": 1}");
        assertEquals(400, response.statusCode(), response.body());
        assertTrue(response.body().contains("instances"), response.body());
    }

    @Test
    void theConfiguredHealthPathAnswersTheReadinessProbe() throws Exception {
        HttpResponse<String> response = send("GET", HEALTH, null);
        assertEquals(503, response.statusCode(), response.body());
        assertTrue(response.headers().firstValue("Content-Type").orElse("").startsWith("application/json"));
        assertTrue(response.body().contains("no model loaded"), response.body());
    }

    @Test
    void theDefaultPredictPathIsNotRegisteredWhenAnotherIsConfigured() throws Exception {
        int status = post("/predict", "{\"instances\": []}").statusCode();
        assertTrue(status == 404 || status == 405, "the default path answers no prediction: " + status);
        assertFalse(PREDICT.equals("/predict"));
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return send("POST", path, body);
    }

    private HttpResponse<String> send(String method, String path, String body) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + base.getPort() + path);
        HttpRequest.Builder request = HttpRequest.newBuilder(uri).timeout(TIMEOUT)
                .header("Content-Type", "application/json");
        HttpRequest.BodyPublisher publisher = body == null
                ? HttpRequest.BodyPublishers.noBody()
                : HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8);
        return client.send(request.method(method, publisher).build(), HttpResponse.BodyHandlers.ofString());
    }

    /** Enables prediction mode and names both routes before Quarkus startup. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            Map.of(VertexCompat.MODE_PROPERTY, "PREDICTION",
                            VertexCompat.PREDICT_PROPERTY, PREDICT,
                            VertexCompat.HEALTH_PROPERTY, HEALTH)
                    .forEach((name, value) -> PREVIOUS.put(name, System.setProperty(name, value)));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so this class's routes are registered alone. */
    public static class PredictionProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
