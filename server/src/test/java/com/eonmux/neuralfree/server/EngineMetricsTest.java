package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.distribution.CountAtBucket;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The engine meters read the pump thread's published snapshot, and read zero before a service opens. */
class EngineMetricsTest {

    private static EngineService service;
    /** The bound meters; micrometer reads them through a weak reference, so the test keeps them reachable. */
    private static volatile EngineMetrics bound;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(256, 2));
    }

    @AfterAll
    static void close() {
        service.close();
        bound = null;
    }

    private static SimpleMeterRegistry bound(EngineService current) {
        EngineLifecycle lifecycle = new EngineLifecycle() {
            @Override
            EngineService current() {
                return current;
            }
        };
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        bound = new EngineMetrics(lifecycle);
        bound.bindTo(registry);
        return registry;
    }

    @Test
    void withoutAServiceEveryMeterReadsZero() {
        SimpleMeterRegistry registry = bound(null);
        assertEquals(0.0, registry.get("neuralfree.prompt.tokens").functionCounter().count());
        assertEquals(0.0, registry.get("neuralfree.tokens.predicted.device").functionCounter().count());
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

    @Test
    void aRepetitionStopCountsUnderItsOwnFinishedReason() throws InterruptedException {
        SimpleMeterRegistry registry = bound(service);
        double before = registry.get("neuralfree.request.success").tag("finished_reason", "repetition")
                .functionCounter().count();
        new OpenAiResource(service).completions("""
                {"prompt": "hello there", "max_tokens": 8, "temperature": 0, "allowed_token_ids": [20],
                 "repetition_detection": {"max_pattern_size": 1, "min_count": 3}}""");
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (registry.get("neuralfree.request.success").tag("finished_reason", "repetition").functionCounter()
                .count() < before + 1 && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        assertEquals(before + 1, registry.get("neuralfree.request.success").tag("finished_reason", "repetition")
                .functionCounter().count(), "vLLM's finished_reason=\"repetition\" label");
    }

    @Test
    void everyRequestFeedsVllmsLatencyHistograms() throws InterruptedException {
        SimpleMeterRegistry registry = bound(service);
        bound.bindOpened(registry, service);
        bound.bindOpened(registry, service);
        new OpenAiResource(service).completions("""
                {"prompt": "hello there", "max_tokens": 5, "ignore_eos": true, "n": 2}""");
        long deadline = System.nanoTime() + 5_000_000_000L;
        while ((registry.get("neuralfree.e2e.request.latency").timer().count() < 2
                || registry.get("neuralfree.request.queue.time").timer().count() < 2)
                && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        Timer ttft = registry.get("neuralfree.time.to.first.token").timer();
        Timer itl = registry.get("neuralfree.inter.token.latency").timer();
        Timer e2e = registry.get("neuralfree.e2e.request.latency").timer();
        Timer queue = registry.get("neuralfree.request.queue.time").timer();
        assertEquals(2, ttft.count(), "one first token per choice");
        assertEquals(8, itl.count(), "four gaps between each choice's five tokens");
        assertEquals(2, e2e.count());
        assertEquals(2, queue.count());
        assertTrue(e2e.max(TimeUnit.NANOSECONDS) >= ttft.max(TimeUnit.NANOSECONDS),
                "a request ends after its first token");
        assertTrue(ttft.max(TimeUnit.NANOSECONDS) >= queue.max(TimeUnit.NANOSECONDS),
                "the first token follows the admission");
        CountAtBucket[] buckets = ttft.takeSnapshot().histogramCounts();
        assertEquals(22, buckets.length, "vLLM's time-to-first-token buckets");
        assertEquals(0.001, buckets[0].bucket(TimeUnit.SECONDS), 1e-12);
        assertEquals(2560.0, buckets[buckets.length - 1].bucket(TimeUnit.SECONDS), 1e-9);
        assertTrue(registry.get("neuralfree.host.sample.seconds").functionCounter().count() > 0);
    }
}
