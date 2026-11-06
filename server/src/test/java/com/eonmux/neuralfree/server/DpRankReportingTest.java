package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The data-parallel rank layout as the endpoints and the meters report it: a server serving one rank behind an
 * external balancer (vLLM's external-LB mode) and one serving a slice of ranks (hybrid mode) name their ranks on
 * {@code /health}, {@code /props}, {@code /server_info} and the replica-tagged series, and those series register
 * when the service opens after the registry bound the binder.
 */
class DpRankReportingTest {

    private static final TokenCallback IGNORED = (token, text) -> { };

    private static Path model;

    @BeforeAll
    static void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    private static EngineService open(int replicas, DpLayout layout) throws IOException {
        return new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2),
                new ServerOptions(null, ServerOptions.DEFAULT_TIMEOUT, false, 0, replicas, null)
                        .withDataParallel(layout));
    }

    private static EngineLifecycle over(EngineService service) {
        return new EngineLifecycle() {
            @Override
            EngineService current() {
                return service;
            }
        };
    }

    @Test
    void anExternalLbServerNamesItsRankOnHealthPropsAndServerInfo() throws IOException {
        try (EngineService rank = open(1, new DpLayout(4, 2, LoadBalance.SHORTEST_QUEUE))) {
            LlamaCppResource resource = new LlamaCppResource(rank);
            for (JsonView body : new JsonView[]{view(resource.health().getEntity()),
                    view(resource.props(null, null).getEntity()),
                    view(new OpsResource(rank, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF).serverInfo(null).getEntity())}) {
                assertEquals(4L, body.i64("data_parallel_size"), "the deployment's global size");
                assertEquals(1L, body.i64("data_parallel_size_local"));
                assertEquals(2L, body.i64("data_parallel_rank"));
                assertEquals(Boolean.TRUE, body.raw("data_parallel_external_lb"));
                assertEquals("external", body.str("data_parallel_mode"));
                assertEquals("shortest_queue", body.str("load_balance_method"));
            }
            assertFalse(view(resource.props(null, null).getEntity()).has("replicas"), "one replica lists no replicas");
        }
    }

    @Test
    void anInternalPoolReportsEveryRankHere() throws IOException {
        try (EngineService internal = open(2, null)) {
            JsonView health = view(new LlamaCppResource(internal).health().getEntity());
            assertEquals(2L, health.i64("data_parallel_size"));
            assertEquals(2L, health.i64("data_parallel_size_local"));
            assertEquals(0L, health.i64("data_parallel_rank"));
            assertEquals(Boolean.FALSE, health.raw("data_parallel_external_lb"));
            assertEquals("internal", health.str("data_parallel_mode"));
        }
    }

    @Test
    void anUnloadedServerServesNoRank() throws IOException {
        try (EngineService unloaded = new EngineService(null, ServerTestSupport.pagedConfig(64, 2))) {
            JsonView info = view(new OpsResource(unloaded, new EngineProfiler(), DevMode.ON, ProfilerConfig.OFF)
                    .serverInfo(null).getEntity());
            assertEquals(0L, info.i64("data_parallel_size"));
            assertEquals(0L, info.i64("data_parallel_size_local"));
            assertFalse(info.has("data_parallel_rank"));
        }
    }

    @Test
    @Timeout(120)
    void aHybridSliceTagsItsReplicasWithTheirGlobalRanks() throws IOException, InterruptedException {
        try (EngineService hybrid = open(2, new DpLayout(4, 2, LoadBalance.ROUND_ROBIN))) {
            int[] prompt = hybrid.encode("hello there", true, false);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(2).ignoreEos(true)
                    .sampler(SamplerConfig.greedyDecoding()).build();
            assertTrue(hybrid.submit(prompt, request, IGNORED).await(30, TimeUnit.SECONDS));
            JsonView props = view(new LlamaCppResource(hybrid).props(null, null).getEntity());
            assertEquals("hybrid", props.str("data_parallel_mode"));
            assertEquals("round_robin", props.str("load_balance_method"));
            assertEquals(2L, at(props, "replicas", 0).i64("rank"));
            assertEquals(3L, at(props, "replicas", 1).i64("rank"));
            assertEquals(0L, at(props, "replicas", 0).i64("prompt_tokens_in_flight"), "the request finished");
            double usage = ((Number) at(props, "replicas", 0).raw("kv_cache_usage")).doubleValue();
            assertTrue(usage >= 0 && usage <= 1);

            SimpleMeterRegistry registry = new SimpleMeterRegistry();
            new EngineMetrics(over(hybrid)).bindTo(registry);
            assertEquals(4.0, registry.get("neuralfree.dp.size").gauge().value());
            assertEquals(2.0, registry.get("neuralfree.dp.replicas").gauge().value());
            for (String name : new String[]{"neuralfree.replica.requests.processing",
                    "neuralfree.replica.requests.deferred", "neuralfree.replica.kv.cache.usage",
                    "neuralfree.replica.prompt.tokens.in.flight"}) {
                assertEquals(2, registry.get(name).gauges().size(), name + " has one series per replica");
                registry.get(name).tag("replica", "0").tag("dp_rank", "2").gauge();
                registry.get(name).tag("replica", "1").tag("dp_rank", "3").gauge();
            }
        }
    }

    @Test
    @Timeout(120)
    void theReplicaSeriesRegisterWhenTheServiceOpensAfterTheBind() throws InterruptedException {
        Map<String, String> launch = Map.of(
                "neuralfree.server.model", model.toString(),
                ServerOptions.REPLICAS_PROPERTY, "2",
                MemoryConfig.PROP_CTX, "256",
                MemoryConfig.PROP_PARALLEL, "2",
                MemoryConfig.PROP_KV_TYPE_K, "f32",
                MemoryConfig.PROP_KV_TYPE_V, "f32",
                EngineConfig.PROP_BACKEND, "cpu");
        Map<String, String> saved = new HashMap<>();
        launch.forEach((key, value) -> {
            saved.put(key, System.getProperty(key));
            System.setProperty(key, value);
        });
        EngineLifecycle lifecycle = new EngineLifecycle();
        try {
            SimpleMeterRegistry registry = new SimpleMeterRegistry();
            new EngineMetrics(lifecycle).bindTo(registry);
            assertNull(registry.find("neuralfree.replica.requests.processing").gauge(),
                    "no replica exists before the service opens");
            assertEquals(0.0, registry.get("neuralfree.dp.replicas").gauge().value());

            EngineService service = lifecycle.engineService();
            assertEquals(2, registry.find("neuralfree.replica.requests.processing").gauges().size(),
                    "opening the service registered one series per replica");
            assertEquals(2.0, registry.get("neuralfree.dp.replicas").gauge().value());

            GenerationRequest busy = GenerationRequest.builder().maxNewTokens(200).ignoreEos(true)
                    .sampler(SamplerConfig.greedyDecoding()).build();
            int[] prompt = service.encode("a prompt to watch", true, false);
            EngineService.Submission submission = service.submit(prompt, busy, IGNORED);
            Gauge tokens = registry.get("neuralfree.replica.prompt.tokens.in.flight").tag("replica", "0").gauge();
            Gauge other = registry.get("neuralfree.replica.prompt.tokens.in.flight").tag("replica", "1").gauge();
            assertEquals(prompt.length, tokens.value() + other.value(), "the request counts on its replica");
            Gauge running0 = registry.get("neuralfree.replica.requests.processing").tag("replica", "0").gauge();
            Gauge running1 = registry.get("neuralfree.replica.requests.processing").tag("replica", "1").gauge();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
            while (running0.value() + running1.value() < 1) {
                assertTrue(System.nanoTime() < deadline, "the replica series never showed the running request");
                Thread.sleep(2);
            }
            service.abortAll();
            assertTrue(submission.await(30, TimeUnit.SECONDS));
            assertEquals(0.0, tokens.value() + other.value());
        } finally {
            lifecycle.onShutdown(null);
            saved.forEach((key, value) -> {
                if (value == null) {
                    System.clearProperty(key);
                } else {
                    System.setProperty(key, value);
                }
            });
        }
    }
}
