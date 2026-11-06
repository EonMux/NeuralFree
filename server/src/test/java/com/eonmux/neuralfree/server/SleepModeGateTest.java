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
import java.util.List;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.pagedConfig;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The sleep-mode gate over the real engine (vLLM's {@code --enable-sleep-mode}): without the switch the checkpoint
 * window answers the 404 of an unregistered route, {@code /sleep}, {@code /wake_up} and
 * {@code /release_kv_cache_memory} answer a 400 naming the flag, and the other lifecycle routes still serve; with
 * the switch the window and the sleep operations work. In router mode the gate precedes the model query.
 */
class SleepModeGateTest {

    @TempDir
    static Path dir;

    private static EngineService service;
    private static EngineService unloaded;
    private static ModelRouter router;
    private static SleepResource off;
    private static SleepResource on;
    private static SleepResource routerOff;
    private static SleepResource routerOn;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), pagedConfig(256, 2));
        Files.copy(model, dir.resolve("a.gguf"));

        off = new SleepResource(service, DevMode.ON, SleepModeSwitch.OFF);
        on = new SleepResource(service, DevMode.ON, SleepModeSwitch.ON);

        ServerOptions options = new ServerOptions(null, Duration.ofMinutes(2));
        EngineConfig config = pagedConfig(256, 2);
        unloaded = new EngineService(null, config, options);
        router = new ModelRouter(dir, null, 0, 0, true, config, options, ModelRouter.systemDeviceMemory());
        routerOff = new SleepResource(unloaded, router, DevMode.ON, SleepModeSwitch.OFF);
        routerOn = new SleepResource(unloaded, router, DevMode.ON, SleepModeSwitch.ON);
    }

    @AfterAll
    static void close() throws InterruptedException {
        if (service.isSleeping()) {
            service.wakeUp(List.of());
        }
        if (service.isPaused()) {
            service.resume();
        }
        router.close();
        unloaded.close();
        service.close();
    }

    @Test
    void theCheckpointWindowAnswers404WithoutSleepMode() {
        for (String route : new String[]{"prepare", "restore", "status"}) {
            assertEquals(404, statusOf(() -> checkpoint(off, route)), route);
            String message = errorOf(() -> checkpoint(off, route));
            assertTrue(message.contains(SleepModeSwitch.FLAG), route + ": " + message);
            assertTrue(message.contains(SleepModeSwitch.PROPERTY), route + ": " + message);
            assertTrue(message.contains("checkpoint"), route + ": " + message);
        }
    }

    @Test
    void theSleepOperationsRefuseWithoutSleepMode() {
        assertEquals(400, statusOf(() -> off.sleep("0", "abort", null, null)));
        assertTrue(errorOf(() -> off.sleep("0", "abort", null, null)).contains(SleepModeSwitch.FLAG));
        assertEquals(400, statusOf(() -> off.wakeUp(null, null, null)));
        assertTrue(errorOf(() -> off.wakeUp(null, null, null)).contains(SleepModeSwitch.FLAG));
        assertEquals(400, statusOf(() -> off.releaseKvCacheMemory(null, null)));
        assertTrue(errorOf(() -> off.releaseKvCacheMemory(null, null)).contains(SleepModeSwitch.FLAG));

        assertFalse(service.isSleeping(), "the refused sleep did not touch the engine");
        assertTrue(service.loaded());
    }

    @Test
    void theOtherLifecycleRoutesServeWithoutSleepMode() {
        assertEquals(Boolean.FALSE, view(off.isSleeping(null, null).getEntity()).raw("is_sleeping"));
        assertEquals("paused", view(off.pause("keep", null, null, null).getEntity()).str("status"));
        assertEquals(Boolean.TRUE, view(off.isPaused(null, null).getEntity()).raw("is_paused"));
        assertEquals("resumed", view(off.resume(null, null).getEntity()).str("status"));
        assertEquals(Boolean.FALSE, view(off.isPaused(null, null).getEntity()).raw("is_paused"));
    }

    @Test
    @Timeout(120)
    void theSwitchTurnsTheSleepOperationsAndTheCheckpointWindowOn() throws InterruptedException {
        assertEquals("sleeping", view(on.sleep("0", "abort", null, null).getEntity()).str("status"));
        assertTrue(service.isSleeping());
        assertEquals("awake", view(on.wakeUp(null, null, null).getEntity()).str("status"));
        assertFalse(service.isSleeping());

        assertEquals(503, on.checkpointStatus().getStatus(), "the status serves and reports the running state");
        assertEquals("prepared", view(on.checkpointPrepare("abort").getEntity()).str("state"));
        assertEquals("prepared", view(on.checkpointStatus().getEntity()).str("state"));
        assertEquals("running", view(on.checkpointRestore().getEntity()).str("state"));
    }

    @Test
    @Timeout(120)
    void routerModeGatesBeforeTheModelQuery() {
        // an unknown model would answer 404 and a missing one 400; the sleep-mode gate answers first
        int sleepStatus = statusOf(() -> routerOff.sleep("0", "abort", "zzz", null));
        assertEquals(400, sleepStatus);
        assertTrue(errorOf(() -> routerOff.sleep("0", "abort", "zzz", null)).contains(SleepModeSwitch.FLAG));
        assertTrue(errorOf(() -> routerOff.sleep("0", "abort", null, null)).contains(SleepModeSwitch.FLAG));
        assertEquals(404, statusOf(() -> routerOff.checkpointStatus(null, null)));
        assertTrue(errorOf(() -> routerOff.checkpointStatus("zzz", null)).contains(SleepModeSwitch.FLAG));
        assertEquals(404, statusOf(() -> routerOff.checkpointPrepare(null, null, null)));

        JsonView slept = view(routerOn.sleep("0", "abort", "a", null).getEntity());
        assertEquals("sleeping", slept.str("status"));
        assertEquals("awake", view(routerOn.wakeUp(null, "a", null).getEntity()).str("status"));
    }

    /** One checkpoint route on {@code resource}, named by {@code route}. */
    private static void checkpoint(SleepResource resource, String route) {
        switch (route) {
            case "prepare" -> resource.checkpointPrepare("abort");
            case "restore" -> resource.checkpointRestore();
            case "status" -> resource.checkpointStatus();
            default -> throw new IllegalArgumentException(route);
        }
    }
}
