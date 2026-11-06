package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;

import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** The readiness probe: up while the service holds an open model, down before that, and up in router mode. */
public class EngineHealthTest {

    @TempDir
    static Path dir;

    private static EngineService service;
    private static EngineService unloaded;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), pagedConfig(64, 2));
        unloaded = new EngineService(null, pagedConfig(64, 2));
    }

    @AfterAll
    static void close() {
        service.close();
        unloaded.close();
    }

    @Test
    void readinessFollowsTheLoadState() {
        assertEquals(HealthCheckResponse.Status.UP, new EngineHealth(service).call().getStatus());
        assertEquals(HealthCheckResponse.Status.DOWN, new EngineHealth(unloaded).call().getStatus());
        assertEquals("engine-model", new EngineHealth(service).call().getName());
    }

    @Test
    void theRouterIsReadyWithoutALoadedLifecycleService() throws IOException {
        ModelRouter router = new ModelRouter(dir, null, 0, 0, true, pagedConfig(64, 2),
                new ServerOptions(null, Duration.ofSeconds(2)), ModelRouter.systemDeviceMemory());
        try {
            assertEquals(HealthCheckResponse.Status.UP, new EngineHealth(unloaded, router).call().getStatus(),
                    "the router itself serves, as its /health and /ready answer");
        } finally {
            router.close();
        }
    }
}
