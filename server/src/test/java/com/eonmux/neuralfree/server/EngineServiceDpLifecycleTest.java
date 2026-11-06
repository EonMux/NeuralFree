package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The serving lifecycle over the data-parallel pool: a sleep releases every replica and the wake restores them, a
 * pause gates the whole pool in its mode, the checkpoint window prepares and restores on every replica, and the
 * idle sleep of the pool wakes on the next request. The single-replica behavior itself is {@code SleepResourceTest}'s
 * subject; here the operations must reach every replica.
 */
@org.junit.jupiter.api.TestMethodOrder(org.junit.jupiter.api.MethodOrderer.OrderAnnotation.class)
class EngineServiceDpLifecycleTest {

    private static Path model;
    private static EngineService pool;
    private static EngineService idle;
    private static SleepResource sleepResource;
    private static SleepResource idleResource;
    private static LlamaCompletionResource completions;
    private static LlamaCppResource llama;
    private static ExecutorService executor;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        sleepResource = new SleepResource(pool, DevMode.ON, SleepModeSwitch.ON);
        completions = new LlamaCompletionResource(pool);
        llama = new LlamaCppResource(pool);
        // idle seconds 0: the replicas sleep as soon as their pump finds nothing to do
        idle = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, null, 0, 2, null));
        idleResource = new SleepResource(idle, DevMode.ON, SleepModeSwitch.ON);
        executor = Executors.newCachedThreadPool();
    }

    @AfterAll
    static void close() {
        executor.shutdownNow();
        pool.close();
        idle.close();
    }

    /** A short greedy completion body the pool finishes deterministically. */
    private static String completion(int nPredict) {
        return "{\"prompt\": \"hello\", \"n_predict\": " + nPredict + ", \"ignore_eos\": true, \"temperature\": 0}";
    }

    /** Waits until the pool reports running sequences, bounded for a CI run. */
    private static void awaitBusyPool() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 30_000;
        while (pool.stats().running() == 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(5);
        }
        assertNotEquals(0, pool.stats().running(), "the generation never reached the pool");
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(1)
    void thePoolSleepsWakesAndServesAgain() {
        try {
            assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));

            JsonView slept = view(sleepResource.sleep("2", "abort").getEntity());
            assertEquals("sleeping", slept.str("status"));
            assertTrue(pool.isSleeping());
            assertFalse(pool.loaded(), "a level-2 sleep releases every replica's weights");
            IllegalStateException refused = assertThrows(IllegalStateException.class,
                    () -> pool.submit(new int[]{1}, GenerationRequest.builder().build(), (token, text) -> { }));
            assertTrue(refused.getMessage().contains("/wake_up"), refused.getMessage());
            assertEquals(503, statusOf(() -> completions.completion(completion(2))));
            assertTrue(errorOf(() -> completions.completion(completion(2))).contains("/wake_up"));
        } finally {
            if (pool.isSleeping()) {
                sleepResource.wakeUp(null);
            }
        }
        assertFalse(pool.isSleeping());
        assertTrue(pool.loaded(), "the wake re-opens every replica's model");
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(2)
    void theSleepingPoolServesItsCachedSetup() {
        try {
            assertEquals("sleeping", view(sleepResource.sleep("2", "abort").getEntity()).str("status"));
            JsonView props = view(llama.props(null, null).getEntity());
            assertEquals(Boolean.TRUE, props.raw("is_sleeping"));
            assertEquals(2L, props.i64("data_parallel_size"), "the cached setup keeps the layout");
            assertNotEquals(null, props.str("model_arch"), "the cached meta answers while the models are closed");
            JsonView health = view(llama.health().getEntity());
            assertEquals("ok", health.str("status"), "a sleeping pool answers again the moment a request wakes it");
            assertEquals(Boolean.TRUE, health.raw("is_sleeping"));
            assertEquals(2L, health.i64("data_parallel_size"));
        } finally {
            if (pool.isSleeping()) {
                sleepResource.wakeUp(null);
            }
        }
        assertFalse(pool.isSleeping());
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(3)
    void aCheckpointWindowPreparesAndRestoresOnEveryReplica() {
        try {
            assertEquals("prepared", view(sleepResource.checkpointPrepare("abort").getEntity()).str("state"));
            assertEquals("prepared", pool.checkpointStatus().state());
            assertEquals(503, statusOf(() -> completions.completion(completion(2))),
                    "a prepared pool keeps new work out");
        } finally {
            if (!"running".equals(pool.checkpointStatus().state())) {
                sleepResource.checkpointRestore();
            }
        }
        assertEquals("running", pool.checkpointStatus().state());
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"),
                "the pool serves again after the restore");
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(4)
    void aKeepPauseFreezesTheWholePoolUntilTheResume() throws Exception {
        try {
            assertEquals("paused", view(sleepResource.pause("keep", null).getEntity()).str("status"));
            assertTrue(pool.isPaused());
            Future<Object> frozen = executor.submit(() -> completions.completion(completion(4)).getEntity());
            assertStillWaiting(frozen, 500, "the frozen request must still wait");
            assertEquals("resumed", view(sleepResource.resume().getEntity()).str("status"));
            assertFalse(pool.isPaused());
            Object answered = frozen.get(30, TimeUnit.SECONDS);
            assertEquals(4L, view(answered).i64("tokens_predicted"));
        } finally {
            if (pool.isPaused()) {
                sleepResource.resume();
            }
        }
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(5)
    void anAbortPauseCancelsInFlightRequestsOnEveryReplica() throws Exception {
        try {
            // a budget the abort, not the context, must end
            Future<Object> first = executor.submit(() -> completions.completion(completion(500)).getEntity());
            awaitBusyPool();
            Future<Object> second = executor.submit(() -> completions.completion(completion(500)).getEntity());

            assertEquals("paused", view(sleepResource.pause("abort", "true").getEntity()).str("status"));
            assertTrue(pool.isPaused());
            assertTrue(predicted(first) > 0 && predicted(first) < 500, "the aborted generation ended early");
            // the second request raced the pause: the abort ended it early, or the pause refused it at the door
            long secondPredicted = predicted(second);
            assertTrue(secondPredicted == -1 || secondPredicted < 500, "the other replica's request ended early");
        } finally {
            if (pool.isPaused()) {
                sleepResource.resume();
            }
        }
        assertFalse(pool.isPaused());
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"),
                "the pool serves again after the resume");
    }

    /** The tokens predicted by a finished completion; a 503 the pause refused at the door answers -1. */
    private static long predicted(Future<Object> future) throws Exception {
        try {
            return view(future.get(30, TimeUnit.SECONDS)).i64("tokens_predicted");
        } catch (ExecutionException e) {
            if (e.getCause() instanceof jakarta.ws.rs.WebApplicationException wae
                    && wae.getResponse().getStatus() == 503) {
                return -1;
            }
            throw e;
        }
    }

    @Test
    @Timeout(180)
    @org.junit.jupiter.api.Order(6)
    void theIdleSleepReleasesEveryReplicaAndTheNextRequestWakesThePool() throws Exception {
        long deadline = System.currentTimeMillis() + 30_000;
        while (!idle.isSleeping() && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        assertTrue(idle.isSleeping(), "the replicas sleep after their idle seconds");
        assertFalse(idle.loaded(), "the idle sleep releases every model");
        JsonView props = view(new LlamaCppResource(idle).props(null, null).getEntity());
        assertEquals(Boolean.TRUE, props.raw("is_sleeping"));

        assertTrue(idle.encode("wake up", true, false).length > 0, "the request wakes the pool");
        assertFalse(idle.isSleeping());
        assertTrue(idle.loaded());
        assertEquals(Boolean.FALSE, view(idleResource.isSleeping().getEntity()).raw("is_sleeping"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(7)
    void theKvReleaseRunsOnEveryReplica() {
        completions.completion(completion(2));
        long cachedFirst = pool.stats().promptTokensCached();
        completions.completion(completion(2));
        long cachedBeforeRelease = pool.stats().promptTokensCached();
        assertTrue(cachedBeforeRelease > cachedFirst, "the second identical prompt reused its cached prefix");
        assertEquals("kv_cache_released", view(sleepResource.releaseKvCacheMemory().getEntity()).str("status"));
        completions.completion(completion(2));
        assertEquals(cachedBeforeRelease, pool.stats().promptTokensCached(),
                "the released cache recomputes the prompt on every replica");
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(8)
    void aPauseLandingAfterTheServingCheckRefusesTheSubmissionWith503() throws Exception {
        int[] prompt = pool.encode("hello", true, false);
        GenerationBatch.Job job = new GenerationBatch.Job(prompt,
                GenerationRequest.builder().maxNewTokens(4).ignoreEos(true).build(), null, null, null);
        try {
            assertEquals("paused", view(sleepResource.pause("abort", null).getEntity()).str("status"));
            // a request already past ServerJson.requireLoaded reaches the submission while the pool is paused
            assertEquals(503, statusOf(() -> GenerationBatch.start(pool, List.of(job), null)));
            String refusal = errorOf(() -> GenerationBatch.start(pool, List.of(job), null));
            assertTrue(refusal.contains("/resume"), refusal);
        } finally {
            if (pool.isPaused()) {
                sleepResource.resume();
            }
        }
        GenerationBatch served = GenerationBatch.start(pool, List.of(job), null);
        served.awaitAll();
        assertEquals(4, served.result(0).tokens().length, "the pool serves the same job after the resume");
    }

    /** Fails when {@code future} answers within {@code millis}. */
    private static void assertStillWaiting(Future<Object> future, long millis, String message)
            throws InterruptedException {
        try {
            future.get(millis, TimeUnit.MILLISECONDS);
            throw new AssertionError(message);
        } catch (TimeoutException expected) {
            // the frozen request still waits, as the keep mode promises
        } catch (ExecutionException e) {
            throw new AssertionError("the frozen request was refused instead of queueing", e);
        }
    }
}
