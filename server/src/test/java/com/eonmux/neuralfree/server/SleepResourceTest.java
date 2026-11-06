package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The serving lifecycle endpoints (vLLM's sleep mode, pause routes and checkpoint routes): a sleep pauses the
 * engine and releases memory at its level, the wake restores generation with the same results, a pause gates the
 * scheduling in its mode, the KV cache release drops the cached prefixes, the checkpoint window prepares and
 * restores, and the idle sleep releases the engine by itself and wakes on the next request.
 *
 * <p>The tests run in a declared order: each returns the shared service to a serving state, and the declared
 * order keeps an engine left sleeping by one test out of the way of the next.
 */
@org.junit.jupiter.api.TestMethodOrder(org.junit.jupiter.api.MethodOrderer.OrderAnnotation.class)
class SleepResourceTest {

    private static EngineService service;
    private static EngineService idle;
    private static SleepResource sleepResource;
    private static SleepResource idleResource;
    private static LlamaCompletionResource completions;
    private static LlamaCompletionResource idleCompletions;
    private static LlamaCppResource llama;
    private static LlamaCppResource idleLlama;
    private static ExecutorService executor;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2));
        sleepResource = new SleepResource(service, DevMode.ON, SleepModeSwitch.ON);
        completions = new LlamaCompletionResource(service);
        llama = new LlamaCppResource(service);
        idle = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, Duration.ofSeconds(60), false, 0, 1));
        idleResource = new SleepResource(idle, DevMode.ON, SleepModeSwitch.ON);
        idleCompletions = new LlamaCompletionResource(idle);
        idleLlama = new LlamaCppResource(idle);
        executor = Executors.newCachedThreadPool();
    }

    @AfterAll
    static void close() {
        executor.shutdownNow();
        service.close();
        idle.close();
    }

    /** A short greedy completion body the engine finishes deterministically. */
    private static String completion(int nPredict) {
        return "{\"prompt\": \"hello\", \"n_predict\": " + nPredict + ", \"ignore_eos\": true, \"temperature\": 0}";
    }

    /** The prompt tokens a completion cached from earlier prefixes, as the engine's stats report them. */
    private static long cachedPromptTokens() {
        return service.stats().promptTokensCached();
    }

    /** Waits until the engine reports running sequences, bounded for a CI run. */
    private static void awaitBusyEngine() throws InterruptedException {
        long deadline = System.currentTimeMillis() + 30_000;
        while (service.stats().running() == 0 && System.currentTimeMillis() < deadline) {
            Thread.sleep(5);
        }
        assertNotEquals(0, service.stats().running(), "the generation never reached the engine");
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(1)
    void sleepWakesUpAndGenerationProducesTheSameResult() {
        long before = view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted");
        assertEquals(4L, before);

        JsonView slept = view(sleepResource.sleep("1", "abort").getEntity());
        assertEquals("sleeping", slept.str("status"));
        assertEquals(1L, slept.i64("level"));
        assertTrue(service.isSleeping());
        assertEquals(Boolean.TRUE, view(sleepResource.isSleeping().getEntity()).raw("is_sleeping"));
        assertEquals(503, statusOf(() -> completions.completion(completion(4))));
        assertTrue(errorOf(() -> completions.completion(completion(4))).contains("/wake_up"));

        JsonView awake = view(sleepResource.wakeUp(null).getEntity());
        assertEquals("awake", awake.str("status"));
        assertEquals(java.util.List.of(), awake.raw("tags"));
        assertFalse(service.isSleeping());
        assertEquals(before, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(2)
    void aLevelTwoSleepReleasesTheModelAndTheWakeReopensIt() {
        assertEquals("sleeping", view(sleepResource.sleep("2", "abort").getEntity()).str("status"));
        assertFalse(service.loaded(), "a level-2 sleep releases the weights");
        assertEquals("ok", view(llama.health().getEntity()).str("status"));
        assertEquals(Boolean.TRUE, view(llama.health().getEntity()).raw("is_sleeping"));
        assertEquals(503, statusOf(() -> completions.completion(completion(2))));

        assertEquals("awake", view(sleepResource.wakeUp(null).getEntity()).str("status"));
        assertTrue(service.loaded(), "the wake re-opens the model");
        assertEquals(Boolean.FALSE, view(llama.health().getEntity()).raw("is_sleeping"));
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(3)
    void aLevelZeroSleepPausesTheSchedulingWithoutTouchingMemory() {
        assertTrue(service.loaded());
        assertEquals("sleeping", view(sleepResource.sleep("0", "abort").getEntity()).str("status"));
        assertTrue(service.loaded(), "a level-0 sleep keeps the model resident");
        assertEquals(503, statusOf(() -> completions.completion(completion(2))));
        assertEquals("awake", view(sleepResource.wakeUp(null).getEntity()).str("status"));
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(4)
    void theSchedulingOnlyWakeLeavesADeeperSleepStanding() {
        assertEquals("sleeping", view(sleepResource.sleep("1", "abort").getEntity()).str("status"));
        JsonView woken = view(sleepResource.wakeUp(java.util.List.of("scheduling")).getEntity());
        assertEquals("sleeping", woken.str("status"));
        assertEquals(java.util.List.of("scheduling"), woken.raw("tags"));
        assertTrue(service.isSleeping());
        assertEquals(503, statusOf(() -> completions.completion(completion(2))));
        assertEquals("awake", view(sleepResource.wakeUp(null).getEntity()).str("status"));
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(5)
    void theLifecycleRefusesUnsupportedInputsByName() {
        for (String level : new String[]{"7", "-1", "x"}) {
            assertEquals(400, statusOf(() -> sleepResource.sleep(level, "abort")), level);
            assertTrue(errorOf(() -> sleepResource.sleep(level, "abort")).contains("level"), level);
        }
        assertEquals(400, statusOf(() -> sleepResource.sleep("1", "nope")));
        assertTrue(errorOf(() -> sleepResource.sleep("1", "nope")).contains("mode"));
        assertEquals(400, statusOf(() -> sleepResource.pause("nope", null)));
        assertTrue(errorOf(() -> sleepResource.pause("nope", null)).contains("mode"));
        assertEquals(400, statusOf(() -> sleepResource.pause("abort", "sometimes")));
        assertTrue(errorOf(() -> sleepResource.pause("abort", "sometimes")).contains("clear_cache"));
        assertEquals(400, statusOf(() -> sleepResource.wakeUp(java.util.List.of("bogus"))));
        assertTrue(errorOf(() -> sleepResource.wakeUp(java.util.List.of("bogus"))).contains("bogus"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(9)
    void anAbortPauseCancelsInFlightRequestsAndRefusesUntilTheResume() throws Exception {
        // a budget the engine cannot reach before the pause: the abort, not the context, must end it
        Future<Object> generation = executor.submit(() -> completions.completion(completion(500)).getEntity());
        awaitBusyEngine();

        assertEquals("paused", view(sleepResource.pause("abort", "true").getEntity()).str("status"));
        assertTrue(service.isPaused());
        Object cancelled = generation.get(30, TimeUnit.SECONDS);
        assertTrue(view(cancelled).i64("tokens_predicted") < 500, "the aborted generation ended early");

        assertEquals(503, statusOf(() -> completions.completion(completion(2))));
        assertTrue(errorOf(() -> completions.completion(completion(2))).contains("/resume"));

        assertEquals("resumed", view(sleepResource.resume().getEntity()).str("status"));
        assertFalse(service.isPaused());
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(10)
    void aKeepPauseQueuesRequestsAndSchedulesNothingUntilTheResume() throws Exception {
        assertEquals("paused", view(sleepResource.pause("keep", null).getEntity()).str("status"));
        assertTrue(service.isPaused());
        Future<Object> frozen = executor.submit(() -> completions.completion(completion(4)).getEntity());
        assertStillWaiting(frozen, 500, "the frozen request must still wait");
        assertEquals("resumed", view(sleepResource.resume().getEntity()).str("status"));
        Object answered = frozen.get(30, TimeUnit.SECONDS);
        assertEquals(4L, view(answered).i64("tokens_predicted"));
    }

    /** Fails when {@code future} answers within {@code millis}. */
    private static void assertStillWaiting(Future<Object> future, long millis, String message)
            throws InterruptedException {
        try {
            future.get(millis, TimeUnit.MILLISECONDS);
            throw new AssertionError(message);
        } catch (java.util.concurrent.TimeoutException expected) {
            // the frozen request still waits, as the keep mode promises
        } catch (java.util.concurrent.ExecutionException e) {
            throw new AssertionError("the frozen request was refused instead of queueing", e);
        }
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(8)
    void theKvCacheReleaseDropsTheCachedPrefixes() {
        completions.completion(completion(2));
        long cachedFirst = cachedPromptTokens();
        completions.completion(completion(2));
        assertTrue(cachedPromptTokens() > cachedFirst,
                "the second identical prompt reuses its cached prefix");
        long cachedBeforeRelease = cachedPromptTokens();

        assertEquals("kv_cache_released", view(sleepResource.releaseKvCacheMemory().getEntity()).str("status"));
        completions.completion(completion(2));
        assertEquals(cachedBeforeRelease, cachedPromptTokens(),
                "the released cache recomputes the prompt");
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(12)
    void theCheckpointWindowPreparesRestoresAndGatesSleep() throws Exception {
        String started = view(sleepResource.checkpointPrepare("wait").getEntity()).str("state");
        assertTrue(started.equals("preparing") || started.equals("prepared"),
                "the preparation starts as preparing or prepared, got " + started);
        long deadline = System.currentTimeMillis() + 30_000;
        while (!"prepared".equals(service.checkpointStatus().state()) && System.currentTimeMillis() < deadline) {
            Thread.sleep(20);
        }
        assertEquals("prepared", service.checkpointStatus().state());
        assertEquals(200, sleepResource.checkpointStatus().getStatus());
        assertEquals("prepared", view(sleepResource.checkpointPrepare("wait").getEntity()).str("state"));

        assertEquals(503, statusOf(() -> completions.completion(completion(2))));
        assertEquals(503, statusOf(() -> sleepResource.sleep("1", "abort")));
        assertTrue(errorOf(() -> sleepResource.sleep("1", "abort")).contains("checkpoint"));

        assertEquals("running", view(sleepResource.checkpointRestore().getEntity()).str("state"));
        assertEquals(503, sleepResource.checkpointStatus().getStatus());
        assertEquals(4L, view(completions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(11)
    void anAbortCheckpointPreparesAnIdleEngineAtOnce() {
        assertEquals("prepared", view(sleepResource.checkpointPrepare("abort").getEntity()).str("state"));
        assertEquals("prepared", service.checkpointStatus().state());
        assertEquals("running", view(sleepResource.checkpointRestore().getEntity()).str("state"));
    }

    @Test
    @Timeout(180)
    @org.junit.jupiter.api.Order(13)
    void theIdleSleepReleasesTheEngineAndTheNextRequestWakesIt() throws Exception {
        long deadline = System.currentTimeMillis() + 30_000;
        while (!idle.isSleeping() && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        assertTrue(idle.isSleeping(), "the engine sleeps after its idle seconds");
        assertFalse(idle.loaded(), "the idle sleep releases the model");

        JsonView props = view(idleLlama.props(null, null).getEntity());
        assertEquals(Boolean.TRUE, props.raw("is_sleeping"));
        assertEquals("ok", view(idleLlama.health().getEntity()).str("status"));

        assertEquals(4L, view(idleCompletions.completion(completion(4)).getEntity()).i64("tokens_predicted"));
        assertFalse(idle.isSleeping(), "the request woke the engine");
        assertEquals(Boolean.FALSE, view(idleResource.isSleeping().getEntity()).raw("is_sleeping"));
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(7)
    void theIdleSecondsComeFromTheProperty() {
        String property = "neuralfree.server.sleep.idle.seconds";
        String saved = System.getProperty(property);
        try {
            System.clearProperty(property);
            assertEquals(ServerOptions.DEFAULT_SLEEP_IDLE_SECONDS, ServerOptions.fromSystemProperties()
                    .sleepIdleSeconds());
            System.setProperty(property, "5");
            assertEquals(5, ServerOptions.fromSystemProperties().sleepIdleSeconds());
            System.setProperty(property, "junk");
            org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                    ServerOptions::fromSystemProperties);
        } finally {
            if (saved == null) {
                System.clearProperty(property);
            } else {
                System.setProperty(property, saved);
            }
        }
    }

    @Test
    @Timeout(120)
    @org.junit.jupiter.api.Order(6)
    void anUnconfiguredServiceRefusesItsLifecycleRoutes() throws IOException {
        EngineService unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2));
        try {
            SleepResource resource = new SleepResource(unloaded, DevMode.ON, SleepModeSwitch.ON);
            assertEquals(Boolean.FALSE, view(resource.isSleeping().getEntity()).raw("is_sleeping"));
            assertEquals(503, statusOf(() -> resource.sleep("1", "abort")));
            assertTrue(errorOf(() -> resource.sleep("1", "abort")).contains("no model"));
            assertEquals(503, statusOf(() -> resource.wakeUp(null)));
        } finally {
            unloaded.close();
        }
    }
}
