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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The sleep-mode and profiler gates on the real HTTP server with their switches off (the launch defaults): the
 * sleep operations answer the sleep-mode refusal naming {@code --enable-sleep-mode}, the checkpoint and profile
 * windows answer the 404 of an unregistered route, and the open routes still serve. The dev mode is on, so the
 * answers come from the sleep-mode and profiler gates, not the development switch.
 */
@QuarkusTest
@QuarkusTestResource(value = SleepProfilerGateHttpTest.Launch.class, restrictToAnnotatedClass = true)
@TestProfile(SleepProfilerGateHttpTest.DefaultProfile.class)
class SleepProfilerGateHttpTest {

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
    void theSleepOperationsRefuseWithoutSleepMode() throws Exception {
        for (String route : new String[]{"/sleep", "/wake_up", "/release_kv_cache_memory"}) {
            HttpResponse<String> refused = send("POST", route);
            assertEquals(400, refused.statusCode(), route + ": " + refused.body());
            assertTrue(refused.body().contains(SleepModeSwitch.FLAG), refused.body());
            assertTrue(refused.body().contains(SleepModeSwitch.PROPERTY), refused.body());
        }
        assertEquals(200, send("GET", "/is_sleeping").statusCode(), "the state route is not a sleep operation");
    }

    @Test
    void theCheckpointWindowAnswers404WithoutSleepMode() throws Exception {
        for (String route : new String[]{"/checkpoint/prepare", "/checkpoint/restore"}) {
            HttpResponse<String> refused = send("POST", route);
            assertEquals(404, refused.statusCode(), route + ": " + refused.body());
            assertTrue(refused.body().contains(SleepModeSwitch.FLAG), refused.body());
            assertTrue(refused.body().contains("NotFoundError"), refused.body());
        }
        HttpResponse<String> status = send("GET", "/checkpoint/status");
        assertEquals(404, status.statusCode(), status.body());
        assertTrue(status.body().contains(SleepModeSwitch.FLAG), status.body());
    }

    @Test
    void theProfileRoutesAnswer404WithoutAProfiler() throws Exception {
        for (String route : new String[]{"/start_profile", "/stop_profile"}) {
            HttpResponse<String> refused = send("POST", route);
            assertEquals(404, refused.statusCode(), route + ": " + refused.body());
            assertTrue(refused.body().contains(ProfilerConfig.FLAG), refused.body());
            assertTrue(refused.body().contains(ProfilerConfig.PROPERTY), refused.body());
            assertTrue(refused.body().contains("NotFoundError"), refused.body());
        }
    }

    private HttpResponse<String> send(String method, String path) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + base.getPort() + path);
        HttpRequest request = HttpRequest.newBuilder(uri).timeout(TIMEOUT)
                .header("Content-Type", "application/json")
                .method(method, HttpRequest.BodyPublishers.ofString("{}")).build();
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    /** Turns the dev mode on and leaves the sleep-mode and profiler switches off before Quarkus startup. */
    public static class Launch implements QuarkusTestResourceLifecycleManager {
        @Override
        public Map<String, String> start() {
            PREVIOUS.put(DevMode.PROPERTY, System.setProperty(DevMode.PROPERTY, "true"));
            PREVIOUS.put(SleepModeSwitch.PROPERTY, System.setProperty(SleepModeSwitch.PROPERTY, "false"));
            PREVIOUS.put(ProfilerConfig.PROPERTY, System.setProperty(ProfilerConfig.PROPERTY, ""));
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }

        @Override
        public void stop() {
            // the class's @AfterAll restores the properties before a later test application starts
        }
    }

    /** A profile of its own, so this gate's application serves this class alone. */
    public static class DefaultProfile implements QuarkusTestProfile {
        @Override
        public Map<String, String> getConfigOverrides() {
            return Map.of("quarkus.http.test-port", "0", "quarkus.grpc.server.test-port", "0");
        }
    }
}
