package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.errorOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The elastic scale over the in-process data-parallel pool: a grow opens the new replicas and a shrink drains the
 * removed ones, the layout and the routes report the new size, and the status phases follow vLLM's external elastic
 * EP lifecycle. A server that holds only part of the deployment refuses the scale by name.
 */
class EngineServiceElasticEpTest {

    private static Path model;

    private EngineService pool;
    private ElasticEpResource resource;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @BeforeEach
    void open() throws IOException {
        pool = new EngineService(model.toString(), ServerTestSupport.pagedConfig(2048, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null));
        resource = new ElasticEpResource(pool, new ClusterSwitches(true, true));
    }

    @AfterEach
    void close() {
        pool.close();
    }

    /** A callback that drops the tokens. */
    private static final TokenCallback IGNORED = (token, text) -> { };

    @Test
    void aGrowOpensReplicasAndAShrinkDrainsAndClosesThem() throws Exception {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return pool;
            }
        };
        io.micrometer.core.instrument.simple.SimpleMeterRegistry registry =
                new io.micrometer.core.instrument.simple.SimpleMeterRegistry();
        new EngineMetrics(lifecycle).bindTo(registry);
        assertEquals(2.0, registry.get("neuralfree.dp.replicas").gauge().value());

        Response grown = resource.scale("{\"new_data_parallel_size\": 3}");
        assertEquals(200, grown.getStatus());
        assertTrue(view(grown.getEntity()).str("message").contains("Scaled to 3"));
        assertEquals(3, pool.replicaCount());
        assertEquals(3, pool.dataParallelSize());
        assertEquals(2, pool.replicaRank(2), "the new replica serves the third rank");
        assertEquals(3.0, registry.get("neuralfree.dp.replicas").gauge().value(),
                "the replicas gauge follows the scale");

        JsonView props = view(new LlamaCppResource(pool).props(null, null).getEntity());
        assertEquals(3, props.i64("data_parallel_size"));
        assertEquals(3, props.i64("data_parallel_size_local"));
        assertEquals(2, at(props, "replicas", 2).i64("rank"));

        Response served = new OpenAiResource(pool).completions(
                "{\"prompt\": \"hello\", \"max_tokens\": 2, \"temperature\": 0}");
        assertEquals(200, served.getStatus(), "the grown pool serves");

        Response shrunk = resource.scale("{\"new_data_parallel_size\": 1, \"drain_timeout\": 10}");
        assertEquals(200, shrunk.getStatus());
        assertEquals(1, pool.replicaCount());
        assertEquals(1, pool.dataParallelSize());
        assertEquals(0, pool.replicaRank(0));

        Response again = new OpenAiResource(pool).completions(
                "{\"prompt\": \"hello\", \"max_tokens\": 2, \"temperature\": 0}");
        assertEquals(200, again.getStatus(), "the shrunk pool serves");
    }

    @Test
    void theStatusFollowsTheScaleLifecycle() {
        JsonView idle = view(resource.status().getEntity());
        assertFalse(idle.bool("is_scaling_elastic_ep"));
        assertEquals("idle", idle.str("phase"));
        assertNull(idle.raw("epoch"));

        resource.scale("{\"new_data_parallel_size\": 2}");
        JsonView completed = view(resource.status().getEntity());
        assertFalse(completed.bool("is_scaling_elastic_ep"));
        assertEquals("completed", completed.str("phase"));
        assertEquals("1", completed.str("epoch"));
        assertEquals(2, completed.i64("requested_data_parallel_size"));
    }

    @Test
    void unusableBodiesRefuseByField() {
        assertEquals("new_data_parallel_size is required",
                errorOf(() -> resource.scale("{}")));
        assertTrue(errorOf(() -> resource.scale("{\"new_data_parallel_size\": 0}"))
                .contains("new_data_parallel_size"));
        assertTrue(errorOf(() -> resource.scale("{\"new_data_parallel_size\": 2.5}"))
                .contains("new_data_parallel_size"));
        assertTrue(errorOf(() -> resource.scale("{\"new_data_parallel_size\": 2, \"drain_timeout\": 0}"))
                .contains("drain_timeout"));
        assertTrue(errorOf(() -> resource.scale("[1]")).contains("expected object"));
    }

    @Test
    void theRouteIsOffWithoutTheSwitch() {
        ElasticEpResource off = new ElasticEpResource(pool, new ClusterSwitches(false, true));
        assertEquals(404, statusOf(() -> off.scale("{\"new_data_parallel_size\": 2}")));
        assertEquals(404, statusOf(off::status));
        assertTrue(errorOf(() -> off.scale("{\"new_data_parallel_size\": 2}"))
                .contains(ClusterSwitches.ELASTIC_FLAG));
    }

    @Test
    void aServerHoldingSomeRanksLeavesTheTopologyToItsBalancer() throws IOException {
        try (EngineService external = new EngineService(model.toString(), ServerTestSupport.pagedConfig(64, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 2, null,
                        false, new DpLayout(4, 0, LoadBalance.SHORTEST_QUEUE)))) {
            ElasticEpResource partial = new ElasticEpResource(external, new ClusterSwitches(true, false));
            assertTrue(errorOf(() -> partial.scale("{\"new_data_parallel_size\": 3}"))
                    .contains("holds 2 of 4"));
        }
    }

    @Test
    @Timeout(120)
    void aBusyRemovedReplicaTimesOutTheDrain() throws Exception {
        // the keep-mode pause queues requests without scheduling them, so the second replica stays busy for good
        pool.pause(SleepControl.Mode.KEEP, false);
        int[] first = pool.encode("alpha alpha alpha holds the first replica", false, false);
        int[] second = pool.encode("zulu zulu zulu holds the second replica", false, false);
        GenerationRequest queued = GenerationRequest.builder().maxNewTokens(8).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        pool.submit(first, queued, IGNORED);
        pool.submit(second, queued, IGNORED);
        assertTrue(pool.replicas().get(1).load() > 0, "the second replica holds a queued request");
        try {
            assertEquals(408, statusOf(() -> resource.scale(
                    "{\"new_data_parallel_size\": 1, \"drain_timeout\": 1}")));
            assertEquals(2, pool.replicaCount(), "the timed-out scale leaves the pool as it was");
            JsonView status = view(resource.status().getEntity());
            assertEquals("failed", status.str("phase"));
            assertTrue(status.str("error").contains("drain"));
        } finally {
            pool.abortAll();
            pool.resume();
        }
    }

    @Test
    void theDeviceListBoundsTheGrowth() throws IOException {
        EngineConfig config = ServerTestSupport.pagedConfig(64, 2);
        try (EngineService devices = new EngineService(model.toString(), config,
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, 1,
                        java.util.List.of("Intel(R) Arc")))) {
            ElasticEpResource bounded = new ElasticEpResource(devices, new ClusterSwitches(true, false));
            assertTrue(errorOf(() -> bounded.scale("{\"new_data_parallel_size\": 2}"))
                    .contains("has no device"));
        }
    }
}
