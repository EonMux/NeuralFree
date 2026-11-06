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

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The API-key refusals of the routes that answer ahead of Quarkus's CORS filter: a browser on another origin can
 * read the 401 because the guard stamps the configured CORS headers, as llama.cpp's pre-routing handler does.
 */
@QuarkusTest
@QuarkusTestResource(value = CorsRefusalHttpTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(CorsRefusalHttpTest.KeyedProfile.class)
class CorsRefusalHttpTest {

    private static final String KEY = "test-cors-key";
    private static final String ORIGIN = "http://ui.example";
    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    /** The values the launch properties had before this class set them, restored before any later class reads them. */
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
    void aKeyedMetricsRefusalIsReadableCrossOrigin() throws Exception {
        HttpResponse<String> refused = send("GET", "/metrics", Map.of("Origin", ORIGIN));
        assertEquals(401, refused.statusCode(), refused.body());
        assertEquals(ORIGIN, header(refused, "Access-Control-Allow-Origin"));
        assertEquals("true", header(refused, "Access-Control-Allow-Credentials"));
        assertTrue(refused.body().contains("authentication_error"), refused.body());
    }

    @Test
    void aKeyedApiRefusalIsReadableCrossOriginToo() throws Exception {
        HttpResponse<String> refused = send("GET", "/v1/models", Map.of("Origin", ORIGIN));
        assertEquals(401, refused.statusCode(), refused.body());
        assertEquals(ORIGIN, header(refused, "Access-Control-Allow-Origin"));
    }

    @Test
    void aRefusalWithoutOriginCarriesNoCorsHeaders() throws Exception {
        HttpResponse<String> refused = send("GET", "/metrics", Map.of());
        assertEquals(401, refused.statusCode(), refused.body());
        assertNull(header(refused, "Access-Control-Allow-Origin"));
        assertNull(header(refused, "Access-Control-Allow-Credentials"));
    }

    @Test
    void aPresentedKeyReachesMetricsAndPreflightStillAnswers() throws Exception {
        HttpResponse<String> served = send("GET", "/metrics", Map.of("Origin", ORIGIN,
                "Authorization", "Bearer " + KEY));
        assertEquals(200, served.statusCode(), served.body());
        HttpResponse<String> preflight = send("OPTIONS", "/v1/models", Map.of("Origin", ORIGIN,
                "Access-Control-Request-Method", "GET"));
        assertEquals(200, preflight.statusCode(), preflight.body());
        assertEquals(ORIGIN, header(preflight, "Access-Control-Allow-Origin"));
    }

    private static String header(HttpResponse<String> response, String name) {
        return response.headers().firstValue(name).orElse(null);
    }

    private HttpResponse<String> send(String method, String path, Map<String, String> headers)
            throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + base.getPort() + path);
        HttpRequest.Builder request = HttpRequest.newBuilder(uri).timeout(TIMEOUT);
        headers.forEach(request::header);
        return client.send(request.method(method, HttpRequest.BodyPublishers.noBody()).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    /** Installs the API key before Quarkus startup and restores the previous values after the class. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            Map.of(ApiKeys.PROPERTY, KEY, CorsLaunchFlags.ORIGINS, "*",
                            CorsLaunchFlags.CREDENTIALS, "true")
                    .forEach((name, value) -> PREVIOUS.put(name, System.setProperty(name, value)));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so the keyed application serves this class alone and later tests boot unkeyed. */
    public static class KeyedProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
