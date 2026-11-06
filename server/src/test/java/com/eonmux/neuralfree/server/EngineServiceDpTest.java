package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * The data-parallel pool: two engine replicas behind the one service. The scheduler routes a request to the least
 * loaded replica, an affinity match never beats a busier replica but breaks the tie between idle ones, the
 * endpoints and meters report the layout, and the aggregate stats sum the replicas.
 */
class EngineServiceDpTest {

    private static Path model;

    private EngineService pool;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @BeforeEach
    void open() throws IOException {
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(512, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
    }

    @AfterEach
    void close() {
        pool.close();
    }

    /** A greedy request that keeps a replica busy for a while, so the next route sees its load. */
    private static GenerationRequest busyRequest() {
        return GenerationRequest.builder().maxNewTokens(300).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    /** A callback that drops the tokens: the content-only submissions route through it. */
    private static final TokenCallback IGNORED = (token, text) -> { };

    private int[] tokens(String text) {
        return pool.encode(text, true, false);
    }

    /** Waits until replica {@code replica} runs at least one sequence. */
    private void awaitRunning(int replica) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (pool.replicaStats().get(replica).running() == 0) {
            assertTrue(System.nanoTime() < deadline, "replica " + replica + " never started");
            Thread.sleep(2);
        }
    }

    /** Waits until every replica published an idle snapshot. */
    private void awaitIdle() throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (true) {
            int busy = 0;
            for (BatchStats stats : pool.replicaStats()) {
                busy += stats.running() + stats.waiting();
            }
            if (busy == 0) {
                return;
            }
            assertTrue(System.nanoTime() < deadline, "the pool never went idle");
            Thread.sleep(5);
        }
    }

    @Test
    void twoReplicasServeRequestsBehindOneService() throws InterruptedException {
        assertEquals(2, pool.replicaCount());
        assertTrue(pool.loaded());
        int[] first = tokens("the first prompt runs long");
        int[] second = tokens("another prompt entirely");
        long before0 = pool.replicaStats().get(0).promptTokens();
        long before1 = pool.replicaStats().get(1).promptTokens();
        EngineService.Submission busy = pool.submit(first, busyRequest(), IGNORED);
        awaitRunning(0);
        EngineService.Submission other = pool.submit(second, busyRequest(), IGNORED);
        busy.await(30, TimeUnit.SECONDS);
        other.await(30, TimeUnit.SECONDS);
        awaitIdle();
        List<BatchStats> replicas = pool.replicaStats();
        assertEquals(before0 + first.length, replicas.get(0).promptTokens(),
                "the long request prefilled on replica 0");
        assertEquals(before1 + second.length, replicas.get(1).promptTokens(),
                "the busy replica pushed the request over");
        assertEquals(replicas.get(0).promptTokens() + replicas.get(1).promptTokens(),
                pool.stats().promptTokens(), "the service aggregates the replicas");
    }

    @Test
    void anAffinityMatchNeverBeatsABusierReplica() throws InterruptedException {
        int[] base = tokens("the shared prefix tokens");
        int[] longer = pool.encode("the shared prefix tokens and more", true, false);
        long before1 = pool.replicaStats().get(1).promptTokens();
        EngineService.Submission busy = pool.submit(base, busyRequest(), IGNORED);
        awaitRunning(0);
        EngineService.Submission other = pool.submit(longer, busyRequest(), IGNORED);
        busy.await(30, TimeUnit.SECONDS);
        other.await(30, TimeUnit.SECONDS);
        awaitIdle();
        assertEquals(before1 + longer.length, pool.replicaStats().get(1).promptTokens(),
                "the extension of the busy replica's prompt still routed to the idle one");
    }

    @Test
    void anIdlePoolRepeatsAPromptOnTheReplicaThatServedIt() throws InterruptedException {
        int[] prompt = tokens("the repeated prompt");
        EngineService.Submission first = pool.submit(prompt, busyRequest(), IGNORED);
        first.await(30, TimeUnit.SECONDS);
        awaitIdle();
        long before = pool.replicaStats().get(0).promptTokens() + pool.replicaStats().get(0).promptTokensCached();
        assertEquals(prompt.length, before, "the first run prefilled on the round robin replica");
        EngineService.Submission second = pool.submit(prompt, busyRequest(), IGNORED);
        second.await(30, TimeUnit.SECONDS);
        awaitIdle();
        long after = pool.replicaStats().get(0).promptTokens() + pool.replicaStats().get(0).promptTokensCached();
        assertEquals(before + prompt.length, after, "the repeat prefilled or reused on the same replica");
        assertEquals(0, pool.replicaStats().get(1).promptTokens() + pool.replicaStats().get(1).promptTokensCached(),
                "the idle tie went to the replica with the prefix, not round robin");
    }

    @Test
    void theEndpointsAndMetersReportThePool() throws InterruptedException, IOException {
        EngineConfig cpu = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(256).ubatch(8).parallel(2)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
        EngineService devices = new EngineService(model.toString(), cpu,
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2,
                        List.of("NVIDIA GeForce RTX 5080", "Intel(R) Arc")));
        try {
            assertEquals("NVIDIA GeForce RTX 5080", devices.replicaDevice(0));
            assertEquals("Intel(R) Arc", devices.replicaDevice(1));
            devices.submit(devices.encode("hello", true, false),
                    GenerationRequest.builder().maxNewTokens(2).ignoreEos(true)
                            .sampler(SamplerConfig.greedyDecoding()).build(),
                    IGNORED).await(30, TimeUnit.SECONDS);
            OpenAiResource openAi = new OpenAiResource(devices);
            Response completion = openAi.completions("""
                    {"prompt": "hello", "max_tokens": 2, "temperature": 0}""");
            assertEquals(200, completion.getStatus());
            assertNotNull(at(view(completion.getEntity()), "choices", 0).str("text"));
            LlamaCppResource resource = new LlamaCppResource(devices);
            assertEquals("ok", view(resource.health().getEntity()).str("status"));
            assertEquals(2, view(resource.health().getEntity()).i64("data_parallel_size"));
            JsonView props = view(resource.props(null, null).getEntity());
            assertEquals(2, props.i64("data_parallel_size"));
            assertEquals(2, props.i64("total_slots"), "the pool's slots sum over the replicas");
            assertEquals(2, size(props, "replicas"));
            JsonView replica0 = at(props, "replicas", 0);
            assertEquals("NVIDIA GeForce RTX 5080", replica0.str("device"));
            assertTrue(replica0.i64("n_ctx") > 0);
            assertEquals(2, replica0.i64("n_slots"), "the serving replica's slots");
            assertEquals(0, at(props, "replicas", 1).i64("n_slots"), "the idle replica never built an engine");
            EngineLifecycle lifecycle = new EngineLifecycle() {
                @Override
                EngineService current() {
                    return devices;
                }
            };
            SimpleMeterRegistry registry = new SimpleMeterRegistry();
            new EngineMetrics(lifecycle).bindTo(registry);
            assertEquals(2.0, registry.get("neuralfree.dp.replicas").gauge().value());
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (registry.get("neuralfree.prompt.tokens").functionCounter().count()
                    < devices.stats().promptTokens() && System.nanoTime() < deadline) {
                Thread.sleep(5);
            }
            assertEquals(devices.stats().promptTokens(),
                    (long) registry.get("neuralfree.prompt.tokens").functionCounter().count(),
                    "the meter reads the aggregate over the replicas");
        } finally {
            devices.close();
        }
    }

    @Test
    void anUnloadedServiceHasNoReplicasAndRefusesByName() throws IOException {
        try (EngineService unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2))) {
            assertEquals(0, unloaded.replicaCount());
            assertEquals(0L, unloaded.stats().promptTokens());
            IllegalStateException e = assertThrows(IllegalStateException.class,
                    () -> unloaded.submit(new int[]{1}, GenerationRequest.builder().build(), IGNORED));
            assertEquals("no model loaded: set -Dneuralfree.server.model=<gguf path>", e.getMessage());
            assertEquals(503, statusOf(() -> new LlamaCppResource(unloaded).props(null, null)));
            assertEquals(ServerJson.NO_MODEL, errorOf(() -> new LlamaCppResource(unloaded).props(null, null)));
        }
    }
}
