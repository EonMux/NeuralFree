package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The engine meters read the pump thread's published snapshot, and read zero before a service opens. */
class EngineMetricsTest {

    private static EngineService service;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
    }

    @AfterAll
    static void close() {
        service.close();
    }

    private static SimpleMeterRegistry bound(EngineService current) {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return current;
            }
        };
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        new EngineMetrics(lifecycle).bindTo(registry);
        return registry;
    }

    @Test
    void withoutAServiceEveryMeterReadsZero() {
        SimpleMeterRegistry registry = bound(null);
        assertEquals(0.0, registry.get("neuralfree.prompt.tokens").functionCounter().count());
        assertEquals(0.0, registry.get("neuralfree.requests.processing").gauge().value());
    }

    @Test
    void finishedRequestsShowInTheCountersAndTheGaugesSettle() throws InterruptedException {
        SimpleMeterRegistry registry = bound(service);
        double before = registry.get("neuralfree.request.success").tag("finished_reason", "length")
                .functionCounter().count();
        OpenAiResource openAi = new OpenAiResource(service);
        openAi.completions("""
                {"prompt": "hello there", "max_tokens": 5, "ignore_eos": true, "n": 2}""");
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (registry.get("neuralfree.request.success").tag("finished_reason", "length").functionCounter()
                .count() < before + 2 && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        assertEquals(before + 2, registry.get("neuralfree.request.success").tag("finished_reason", "length")
                .functionCounter().count(), "the snapshot after the finishing step");
        assertTrue(registry.get("neuralfree.prompt.tokens").functionCounter().count()
                + registry.get("neuralfree.prompt.tokens.cached").functionCounter().count() > 0);
        assertTrue(registry.get("neuralfree.tokens.predicted").functionCounter().count() >= 10);
        assertTrue(registry.get("neuralfree.decode").functionCounter().count() > 0);
        assertEquals(0.0, registry.get("neuralfree.requests.processing").gauge().value());
        double usage = registry.get("neuralfree.kv.cache.usage").gauge().value();
        assertTrue(usage >= 0 && usage <= 1);
        assertEquals(0.0, registry.get("neuralfree.request.failures").functionCounter().count());
    }
}
