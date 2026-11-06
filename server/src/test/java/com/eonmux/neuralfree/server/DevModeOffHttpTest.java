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
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The development endpoints on the real HTTP server with the switch off (the launch default): every vLLM
 * development and RL control route answers the 404 of an unregistered route — before any engine check, so
 * {@code /sleep} answers 404 rather than the no-model or sleep-mode refusal — while the routes outside the dev
 * mode serve as before. Sleep mode and the profiler are enabled, so the checkpoint and profile windows show that
 * the dev switch alone does not gate them.
 */
@QuarkusTest
@QuarkusTestResource(value = DevModeOffHttpTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(DevModeOffHttpTest.DefaultProfile.class)
class DevModeOffHttpTest {

    /** The gated routes and the request each takes, all of which must answer 404 while the switch is off. */
    private static final List<Map.Entry<String, String>> GATED = List.of(
            Map.entry("GET", "/server_info"),
            Map.entry("GET", "/is_sleeping"),
            Map.entry("GET", "/is_paused"),
            Map.entry("GET", "/weight_info"),
            Map.entry("POST", "/sleep?level=0"),
            Map.entry("POST", "/wake_up"),
            Map.entry("POST", "/release_kv_cache_memory"),
            Map.entry("POST", "/pause"),
            Map.entry("POST", "/resume"),
            Map.entry("POST", "/abort_requests"),
            Map.entry("POST", "/reset_prefix_cache"),
            Map.entry("POST", "/reset_mm_cache"),
            Map.entry("POST", "/reset_encoder_cache"),
            Map.entry("POST", "/init_weight_transfer_engine"),
            Map.entry("POST", "/shutdown_weight_transfer_engine"),
            Map.entry("POST", "/start_weight_update"),
            Map.entry("POST", "/update_weights"),
            Map.entry("POST", "/finish_weight_update"),
            Map.entry("POST", "/update_weight_version"),
            Map.entry("POST", "/collective_rpc"));

    private static final Duration TIMEOUT = Duration.ofSeconds(30);

    /** The message of every gated route's 404, as the unregistered-route refusal writes it. */
    private static final String REFUSAL = "development endpoints are off";

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
    void everyGatedRouteAnswersTheUnregisteredRoutes404() throws Exception {
        for (Map.Entry<String, String> route : GATED) {
            HttpResponse<String> refused = send(route.getKey(), route.getValue());
            assertEquals(404, refused.statusCode(), route.getKey() + " " + route.getValue() + ": " + refused.body());
            assertTrue(refused.body().contains(REFUSAL), refused.body());
            assertTrue(refused.body().contains("NotFoundError"), refused.body());
        }
    }

    @Test
    void theRoutesOutsideTheDevModeServeAsBefore() throws Exception {
        HttpResponse<String> version = send("GET", "/version");
        assertEquals(200, version.statusCode());
        assertFalse(version.body().contains(REFUSAL));
        assertEquals(503, send("GET", "/health").statusCode(), "unloaded, as every probe reports it");
        assertNotEquals(404, send("GET", "/tokenizer_info").statusCode(),
                "tokenizer_info is a serve route, not a development one");
        assertNotEquals(404, send("POST", "/start_profile").statusCode(),
                "the profile window is a serve route, not a development one");
        assertNotEquals(404, send("GET", "/checkpoint/status").statusCode(),
                "the checkpoint window is vLLM's sleep-mode route, not a development one");
    }

    private HttpResponse<String> send(String method, String path) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + base.getPort() + path);
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(TIMEOUT)
                .method(method, HttpRequest.BodyPublishers.noBody()).build();
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    /** Turns the switch off before Quarkus startup and restores the previous value after the class. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            PREVIOUS.put(DevMode.PROPERTY, System.setProperty(DevMode.PROPERTY, "false"));
            PREVIOUS.put(SleepModeSwitch.PROPERTY, System.setProperty(SleepModeSwitch.PROPERTY, "true"));
            String profileDir = java.nio.file.Path.of(System.getProperty("java.io.tmpdir"),
                    "nf-devoff-profile").toString();
            PREVIOUS.put(ProfilerConfig.PROPERTY, System.setProperty(ProfilerConfig.PROPERTY,
                    "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"" + profileDir + "\"}"));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so the default (switch-off) application serves this class alone. */
    public static class DefaultProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
