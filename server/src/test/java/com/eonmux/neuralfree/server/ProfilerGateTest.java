package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The profiler gate over the real service (vLLM's {@code --profiler-config}): without a configured profiler
 * {@code /start_profile} and {@code /stop_profile} answer the 404 of an unregistered route, and with one they serve
 * the profile window. In router mode the gate precedes the model query.
 */
class ProfilerGateTest {

    @TempDir
    static Path dir;

    private static EngineService unloaded;
    private static ModelRouter router;
    private static OpsResource off;
    private static OpsResource on;
    private static OpsResource routerOff;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        Files.copy(model, dir.resolve("a.gguf"));
        Path profileDir = Files.createDirectories(dir.resolve("profile"));

        unloaded = new EngineService(null, pagedConfig(64, 2));
        off = new OpsResource(unloaded, new EngineProfiler(profileDir), DevMode.ON, ProfilerConfig.OFF);
        on = new OpsResource(unloaded, new EngineProfiler(profileDir), DevMode.ON,
                ProfilerConfig.of(ProfilerConfig.TORCH, profileDir));

        ServerOptions options = new ServerOptions(null, Duration.ofMinutes(2));
        EngineConfig config = pagedConfig(64, 2);
        router = new ModelRouter(dir, null, 0, 0, true, config, options, ModelRouter.systemDeviceMemory());
        routerOff = new OpsResource(unloaded, new EngineProfiler(profileDir), router, DevMode.ON,
                ProfilerConfig.OFF);
    }

    @AfterAll
    static void close() {
        router.close();
        unloaded.close();
    }

    @Test
    void theProfileRoutesAnswer404WithoutAProfiler() {
        for (String route : new String[]{"start", "stop"}) {
            assertEquals(404, statusOf(() -> profile(off, route)), route);
            String message = errorOf(() -> profile(off, route));
            assertTrue(message.contains(ProfilerConfig.FLAG), route + ": " + message);
            assertTrue(message.contains(ProfilerConfig.PROPERTY), route + ": " + message);
        }
    }

    @Test
    void theConfiguredProfilerServesTheWindow() {
        assertEquals(200, on.startProfile(null).getStatus());
        assertEquals(400, statusOf(() -> on.startProfile(null)), "a second window is refused");
        JsonView report = view(on.stopProfile().getEntity());
        assertTrue(report.f64("duration_seconds") >= 0);
        assertTrue(report.str("recording").endsWith(".jfr"));
        assertEquals(400, statusOf(() -> on.stopProfile()), "the window closed");
    }

    @Test
    @Timeout(120)
    void routerModeGatesBeforeTheModelQuery() {
        // a missing model would answer 400 and an unknown one 404; the profiler gate answers first
        assertEquals(404, statusOf(() -> routerOff.startProfile(null, null, null)));
        assertTrue(errorOf(() -> routerOff.startProfile(null, null, null)).contains(ProfilerConfig.FLAG));
        assertEquals(404, statusOf(() -> routerOff.startProfile(null, "zzz", null)));
        assertTrue(errorOf(() -> routerOff.startProfile(null, "zzz", null)).contains(ProfilerConfig.FLAG));
        assertEquals(404, statusOf(() -> routerOff.stopProfile("zzz", null)));
        assertTrue(errorOf(() -> routerOff.stopProfile("zzz", null)).contains(ProfilerConfig.FLAG));
    }

    /** One profile route on {@code resource}, named by {@code route}. */
    private static void profile(OpsResource resource, String route) {
        if (route.equals("start")) {
            resource.startProfile(null);
        } else {
            resource.stopProfile();
        }
    }
}
