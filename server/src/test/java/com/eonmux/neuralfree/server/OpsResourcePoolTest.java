package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The operational endpoints over a two-replica pool and its sleep: the abort, the load and the prefix-cache reset
 * span every replica, and a sleeping pool stays ready on {@code /health}, {@code /ready}, {@code /ping} and the
 * readiness check without the probes waking it.
 */
class OpsResourcePoolTest {

    private static final TokenCallback IGNORED = (token, text) -> { };

    private static Path model;

    private EngineService pool;
    private OpsResource ops;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @BeforeEach
    void open() throws IOException {
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        ops = new OpsResource(pool, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF);
    }

    @AfterEach
    void close() {
        pool.close();
    }

    @Test
    @Timeout(120)
    void abortLoadAndResetSpanEveryReplica() throws InterruptedException {
        GenerationRequest busy = GenerationRequest.builder().maxNewTokens(480).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        EngineService.Submission first = pool.submit(pool.encode("the first prompt", true, false), busy, IGNORED);
        awaitRunning(1);
        EngineService.Submission second = pool.submit(pool.encode("a second one", true, false), busy, IGNORED);
        awaitRunning(2);
        assertTrue(pool.replicaStats().get(0).running() > 0 && pool.replicaStats().get(1).running() > 0,
                "the scheduler spread the requests over both replicas");

        assertEquals(2L, view(ops.load().getEntity()).i64("server_load"), "the load sums the replicas");
        assertFalse(view(ops.resetPrefixCache(false, false).getEntity()).bool("success"),
                "busy replicas hold pages of their memory, so the reset reports failure");
        assertEquals(2L, view(ops.abortRequests(null).getEntity()).i64("aborted"),
                "the abort reaches the requests of every replica");
        assertTrue(first.await(30, TimeUnit.SECONDS));
        assertTrue(second.await(30, TimeUnit.SECONDS));
        assertEquals(StopReason.CANCELLED, first.sequence().result().stopReason());
        assertEquals(StopReason.CANCELLED, second.sequence().result().stopReason());

        assertEquals(0L, view(ops.load().getEntity()).i64("server_load"));
        awaitIdle();
        assertTrue(view(ops.resetPrefixCache(false, false).getEntity()).bool("success"));
    }

    @Test
    @Timeout(120)
    void aSleepingPoolStaysReadyAndTheProbesDoNotWakeIt() throws InterruptedException {
        String architecture = pool.architecture();
        pool.sleep(2, SleepControl.Mode.ABORT);
        assertFalse(pool.loaded(), "a level-2 sleep releases the models");

        assertEquals(200, ops.ready().getStatus());
        assertEquals(200, ops.readyz().getStatus());
        assertEquals(200, ops.ping().getStatus());
        assertEquals(HealthCheckResponse.Status.UP, new EngineHealth(pool).call().getStatus());
        JsonView health = view(new LlamaCppResource(pool).health().getEntity());
        assertEquals("ok", health.str("status"));
        assertEquals(Boolean.TRUE, health.raw("is_sleeping"));

        JsonView info = view(ops.serverInfo(null).getEntity());
        assertFalse(info.bool("loaded"));
        assertEquals(Boolean.TRUE, info.raw("is_sleeping"));
        assertNotNull(architecture);
        assertEquals(architecture, info.str("model_arch"), "the captured setup answers while sleeping");
        assertTrue(info.i64("context_length") > 0);
        assertEquals(2L, info.i64("data_parallel_size"));
        assertTrue(view(ops.resetPrefixCache(false, false).getEntity()).bool("success"),
                "a sleeping pool holds no cached prefix");

        assertTrue(pool.isSleeping(), "no probe woke the pool");
        assertFalse(pool.loaded());
        assertEquals("awake", pool.wakeUp(List.of()));
        assertTrue(pool.loaded());
    }

    /** Waits until the pool runs at least {@code sequences} sequences. */
    private void awaitRunning(int sequences) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (pool.stats().running() < sequences) {
            assertTrue(System.nanoTime() < deadline, "the pool never ran " + sequences + " sequence(s)");
            Thread.sleep(2);
        }
    }

    /** Waits until every replica published an idle snapshot. */
    private void awaitIdle() throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (pool.stats().running() + pool.stats().waiting() > 0) {
            assertTrue(System.nanoTime() < deadline, "the pool never went idle");
            Thread.sleep(5);
        }
    }
}
