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
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * The development routes on the wire with the switch on: a launch whose {@code neuralfree.server.dev.mode} is true
 * reaches every vLLM development and RL control route, which then answer their own statuses (the unloaded engine's
 * 503 refusals and empty 200s) instead of the off switch's 404.
 */
@QuarkusTest
@QuarkusTestResource(value = DevModeEnabledHttpTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(DevModeEnabledHttpTest.OnProfile.class)
class DevModeEnabledHttpTest {

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
    void everyGatedRouteReachesItsHandler() throws Exception {
        // the reporting routes answer the unloaded server, the cache resets drop nothing, and abort has none to stop
        assertEquals(200, send("GET", "/server_info", "").statusCode());
        assertEquals(200, send("GET", "/is_sleeping", "").statusCode());
        assertEquals(200, send("GET", "/is_paused", "").statusCode());
        assertEquals(200, send("POST", "/resume", "").statusCode());
        assertEquals(200, send("POST", "/abort_requests", "").statusCode());
        assertEquals(200, send("POST", "/reset_mm_cache", "").statusCode());
        assertEquals(200, send("POST", "/reset_encoder_cache", "").statusCode());
        assertEquals(200, send("POST", "/shutdown_weight_transfer_engine", "").statusCode());
        assertEquals(200, send("POST", "/collective_rpc",
                "{\"method\": \"shutdown_weight_transfer_engine\"}").statusCode());
        assertEquals(503, send("POST", "/collective_rpc",
                "{\"method\": \"get_weight_version\"}").statusCode(), "the read needs an open model");

        // the routes that need an open model refuse with their own 503, not the off switch's 404
        for (String[] route : new String[][]{{"GET", "/weight_info"}, {"POST", "/sleep"}, {"POST", "/wake_up"},
                {"POST", "/release_kv_cache_memory"}, {"POST", "/pause"}, {"POST", "/reset_prefix_cache"},
                {"POST", "/init_weight_transfer_engine"}, {"POST", "/start_weight_update"},
                {"POST", "/update_weights"}, {"POST", "/finish_weight_update"},
                {"POST", "/update_weight_version"}}) {
            HttpResponse<String> answered = send(route[0], route[1], "{}");
            assertEquals(503, answered.statusCode(), route[0] + " " + route[1] + ": " + answered.body());
            assertNotEquals(404, answered.statusCode());
        }
    }

    private HttpResponse<String> send(String method, String path, String body) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + base.getPort() + path);
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(TIMEOUT)
                .header("Content-Type", "application/json")
                .method(method, HttpRequest.BodyPublishers.ofString(body)).build();
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    /** Turns the switches on before Quarkus startup and restores the previous values after the class. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            PREVIOUS.put(DevMode.PROPERTY, System.setProperty(DevMode.PROPERTY, "true"));
            PREVIOUS.put(SleepModeSwitch.PROPERTY, System.setProperty(SleepModeSwitch.PROPERTY, "true"));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so the switch-on application serves this class alone. */
    public static class OnProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
