package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The serving metrics against vLLM's {@code calculate_metrics}: the completed and failed counts, the
 * TTFT, TPOT (first token excluded), ITL and end-to-end statistics, the goodput, the per-second peak,
 * the report layout and the result JSON (the per-request arrays only under {@code --save-detailed}).
 */
public class ServeBenchMetricsTest {

    private static BenchRequestOutput success(double ttft, double latency, Integer outputTokens,
                                             double... itls) {
        BenchRequestOutput out = new BenchRequestOutput();
        out.success = true;
        out.ttft = ttft;
        out.latency = latency;
        out.outputTokens = outputTokens;
        out.promptLen = 10;
        out.startTime = 0.0;
        out.clientQueueTime = 0.0;
        for (double itl : itls) {
            out.itls.add(itl);
        }
        out.generatedText.append("text");
        return out;
    }

    private static BenchRequestOutput failure(String error) {
        BenchRequestOutput out = new BenchRequestOutput();
        out.success = false;
        out.error = error;
        return out;
    }

    @Test
    public void theStatisticsFollowTheReference() {
        List<BenchRequestOutput> outputs = Arrays.asList(
                success(0.1, 0.30, 3, 0.1, 0.1),
                success(0.2, 0.50, 2, 0.3),
                failure("HTTP 500"));
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(outputs, 2.0,
                Arrays.asList(50.0, 99.0), Collections.emptyMap());
        assertEquals(2, metrics.completed);
        assertEquals(1, metrics.failed);
        assertEquals(20, metrics.totalInput);
        assertEquals(5, metrics.totalOutput);
        assertEquals(1.0, metrics.requestThroughput, 1e-12);
        assertEquals(2.5, metrics.outputThroughput, 1e-12);
        assertEquals(12.5, metrics.totalTokenThroughput, 1e-12);
        assertNotNull(metrics.ttft);
        assertEquals(0.15, metrics.ttft.mean / 1000.0, 1e-12);
        assertEquals(0.15, metrics.ttft.median / 1000.0, 1e-12);
        // the three-token request has two TPOT samples, the two-token one has one
        assertEquals(200.0, metrics.tpot.mean, 1e-9);
        // the two intervals of the first request and the one of the second
        assertEquals((0.1 + 0.1 + 0.3) / 3, metrics.itl.mean / 1000.0, 1e-12);
        assertEquals(0.4, metrics.e2el.mean / 1000.0, 1e-12);
        assertEquals(0.4, metrics.e2el.percentiles.get(50.0) / 1000.0, 0.15);
        // a failed request counts zero output tokens
        assertEquals(Arrays.asList(3, 2, 0), metrics.actualOutputLens);
    }

    @Test
    public void allFailedRequestsLeaveEverySeriesNull() {
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(
                Collections.singletonList(failure("boom")), 1.0, Collections.singletonList(99.0),
                Collections.emptyMap());
        assertEquals(0, metrics.completed);
        assertEquals(1, metrics.failed);
        assertNull(metrics.ttft);
        assertNotNull("the end-to-end series stays reported, as numpy reports an empty sample", metrics.e2el);
        assertEquals(0.0, metrics.e2el.mean, 0.0);
        assertEquals(0.0, metrics.requestThroughput, 0.0);
    }

    @Test
    public void theGoodputCountsTheRequestsMeetingEveryObjective() {
        List<BenchRequestOutput> outputs = Arrays.asList(
                success(0.1, 0.3, 3, 0.1, 0.1),
                success(1.0, 2.0, 2, 1.0));
        Map<String, Double> goodput = new LinkedHashMap<>();
        goodput.put("ttft", 500.0);
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(outputs, 2.0,
                Collections.singletonList(99.0), goodput);
        assertEquals(0.5, metrics.requestGoodput, 0.0);
    }

    @Test
    public void theReportPrintsTheReferenceLayout() throws Exception {
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(
                Collections.singletonList(success(0.1, 0.3, 3, 0.1, 0.1)), 1.0,
                Arrays.asList(50.0, 99.0), Collections.emptyMap());
        StringBuilder out = new StringBuilder();
        metrics.print(out, 1.0, Arrays.asList("ttft", "tpot", "itl", "e2el"), Double.POSITIVE_INFINITY,
                null, false);
        String report = out.toString();
        assertTrue(report.contains(" Serving Benchmark Result "));
        assertTrue(report.contains("Successful requests:"));
        assertTrue(report.contains("Failed requests:"));
        assertTrue(report.contains("Benchmark duration (s):"));
        assertTrue(report.contains("Total input tokens:"));
        assertTrue(report.contains("Total generated tokens:"));
        assertTrue(report.contains("Request throughput (req/s):"));
        assertTrue(report.contains("Output token throughput (tok/s):"));
        assertTrue(report.contains("Peak output token throughput (tok/s):"));
        assertTrue(report.contains("Peak concurrent requests:"));
        assertTrue(report.contains("Total token throughput (tok/s):"));
        assertTrue(report.contains("Time to First Token"));
        assertTrue(report.contains("Time per Output Token (excl. 1st token)"));
        assertTrue(report.contains("Inter-token Latency"));
        assertTrue(report.contains("End-to-end Latency"));
        assertTrue(report.contains("P50 TTFT (ms):"));
        assertTrue(report.contains("P99 TTFT (ms):"));
        assertFalse(report.contains("Request rate configured"));
        assertFalse(report.contains("Request goodput"));
    }

    @Test
    public void theResultJsonKeepsTheReferenceKeys() {
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(
                Collections.singletonList(success(0.1, 0.3, 3, 0.1, 0.1)), 1.0,
                Collections.singletonList(99.0), Collections.emptyMap());
        Map<String, Object> result = metrics.resultJson(1.0, Collections.singletonList(
                success(0.1, 0.3, 3, 0.1, 0.1)), Arrays.asList("ttft", "e2el"), false, false);
        assertEquals(1.0, (Double) result.get("duration"), 0.0);
        assertEquals(1, result.get("completed"));
        assertEquals(3L, ((Number) result.get("total_output_tokens")).longValue());
        assertNotNull(result.get("mean_ttft_ms"));
        assertNotNull(result.get("p99_ttft_ms"));
        assertNotNull(result.get("mean_e2el_ms"));
        assertNull("an unselected metric carries no statistic", result.get("mean_itl_ms"));
        assertFalse("the per-request arrays stay out without --save-detailed", result.containsKey("ttfts"));
        assertTrue(result.containsKey("input_lens"));
        assertTrue(result.containsKey("output_lens"));
        assertTrue(result.containsKey("latencies"));
        assertTrue(result.containsKey("queue_times"));
        assertNull(result.get("request_goodput"));
    }

    @Test
    public void theDetailedResultJsonCarriesThePerRequestArrays() {
        BenchRequestOutput out = success(0.1, 0.3, 3, 0.1, 0.1);
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(Collections.singletonList(out), 1.0,
                Collections.singletonList(99.0), Collections.emptyMap());
        Map<String, Object> result = metrics.resultJson(1.0, Collections.singletonList(out),
                Collections.singletonList("ttft"), false, true);
        assertEquals(Collections.singletonList(0.1), result.get("ttfts"));
        assertEquals(Collections.singletonList(Arrays.asList(0.1, 0.1)), result.get("itls"));
        assertEquals(Collections.singletonList(0.3), result.get("latencies"));
        assertEquals(Collections.singletonList("text"), result.get("generated_texts"));
        assertEquals(Collections.singletonList(null), result.get("errors"));
    }

    @Test
    public void thePeakCountsTheBusiestSecond() {
        BenchRequestOutput first = success(0.0, 1.0, 3, 0.5, 0.5);
        first.startTime = 0.0;
        BenchRequestOutput second = success(0.0, 1.0, 3, 0.5, 0.5);
        second.startTime = 0.0;
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(Arrays.asList(first, second), 1.0,
                Collections.singletonList(99.0), Collections.emptyMap());
        // each request emits its first token at 0 s and two more at 0.5 s and 1 s: the busiest
        // second carries four tokens (two per request)
        assertEquals(4.0, metrics.maxOutputTokensPerS, 0.0);
        assertEquals(2, metrics.maxConcurrentRequests);
    }

    @Test
    public void theQueueSeriesReportUnderAConcurrencyCap() {
        BenchRequestOutput out = success(0.1, 0.3, 3, 0.1, 0.1);
        out.clientQueueTime = 0.2;
        ServeBenchMetrics metrics = ServeBenchMetrics.compute(Collections.singletonList(out), 1.0,
                Collections.singletonList(50.0), Collections.emptyMap(), 1, 2.0);
        assertNotNull(metrics.clientQueueTime);
        assertEquals(200.0, metrics.clientQueueTime.mean, 1e-9);
        assertNotNull(metrics.e2elIncludingClientQueue);
        assertEquals(500.0, metrics.e2elIncludingClientQueue.mean, 1e-9);
        Map<String, Object> result = metrics.resultJson(1.0, Collections.singletonList(out),
                Arrays.asList("client_queue_time", "e2el_including_client_queue"), false, false);
        assertNotNull(result.get("mean_client_queue_time_ms"));
        assertNotNull(result.get("p50_e2el_including_client_queue_ms"));
        ServeBenchMetrics uncapped = ServeBenchMetrics.compute(Collections.singletonList(out), 1.0,
                Collections.singletonList(50.0), Collections.emptyMap());
        assertNull(uncapped.clientQueueTime);
        assertNull(uncapped.e2elIncludingClientQueue);
    }

    @Test
    public void aNonIntegerPercentileKeepsItsDecimalLabel() {
        assertEquals("99", ServeBenchMetrics.percentileName(99.0));
        assertEquals("99.9", ServeBenchMetrics.percentileName(99.9));
    }
}
