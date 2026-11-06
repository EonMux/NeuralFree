package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The option values shared by the commands: the range-ratio forms, the request rate spellings and the
 * metric pin lists with their defaults.
 */
public class ServeBenchOptionsTest {

    @Test
    public void aPlainRangeRatioAppliesToBothLengths() {
        ServeBenchOptions.RangeRatio ratio = ServeBenchOptions.parseRangeRatio("0.5");
        assertEquals(0.5, ratio.input(), 0.0);
        assertEquals(0.5, ratio.output(), 0.0);
    }

    @Test
    public void theJsonRangeRatioCarriesIndependentLengths() {
        ServeBenchOptions.RangeRatio ratio = ServeBenchOptions.parseRangeRatio("{\"input\":0.2,\"output\":0.8}");
        assertEquals(0.2, ratio.input(), 0.0);
        assertEquals(0.8, ratio.output(), 0.0);
    }

    @Test
    public void aRangeRatioOutsideZeroToOneIsRefused() {
        for (String value : new String[]{"1.0", "-0.1", "{\"input\":1,\"output\":0.5}",
                "{\"input\":0.5}"}) {
            try {
                ServeBenchOptions.parseRangeRatio(value);
                fail("the range ratio must be refused: " + value);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("range ratio"));
            }
        }
    }

    @Test
    public void theRequestRateAcceptsInfinityAndRefusesNonPositiveValues() {
        assertEquals(Double.POSITIVE_INFINITY, ServeBenchOptions.parseRequestRate("inf"), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, ServeBenchOptions.parseRequestRate("Infinity"), 0.0);
        assertEquals(12.5, ServeBenchOptions.parseRequestRate("12.5"), 0.0);
        for (String value : new String[]{"0", "-1", "NaN", "fast"}) {
            try {
                ServeBenchOptions.parseRequestRate(value);
                fail("the request rate must be refused: " + value);
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("--request-rate"));
            }
        }
    }

    @Test
    public void theDefaultMetricListsMatchTheGenerativeReference() {
        ServeBenchOptions opts = ServeBenchOptions.builder().build();
        assertEquals(Arrays.asList("ttft", "tpot", "itl"), opts.percentileMetricNames());
        assertEquals(Arrays.asList(99.0), opts.percentiles());
        assertEquals("127.0.0.1", opts.host());
        assertEquals(8000, opts.port());
        assertEquals("/v1/completions", opts.endpoint());
        assertEquals("random", opts.datasetName());
        assertEquals(1000, opts.numPrompts());
        assertTrue(Double.isInfinite(opts.requestRate()));
        assertEquals(1.0, opts.burstiness(), 0.0);
        assertEquals("http://127.0.0.1:8000", opts.resolvedBaseUrl());
    }

    @Test
    public void aBaseUrlReplacesTheHostAndPort() {
        ServeBenchOptions opts = ServeBenchOptions.builder().host("other").port(1)
                .baseUrl("https://example.test:8443").build();
        assertEquals("https://example.test:8443", opts.resolvedBaseUrl());
    }

    @Test
    public void theMetricPinListsParseAndRefuseUnknownNames() {
        ServeBenchOptions opts = ServeBenchOptions.builder()
                .percentileMetrics("ttft,itl,e2el").metricPercentiles("25,50,99.9").build();
        assertEquals(Arrays.asList("ttft", "itl", "e2el"), opts.percentileMetricNames());
        assertEquals(Arrays.asList(25.0, 50.0, 99.9), opts.percentiles());
        try {
            ServeBenchOptions.builder().percentileMetrics("ttft,tps").build().percentileMetricNames();
            fail("an unknown percentile metric must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("'tps'"));
        }
        try {
            ServeBenchOptions.builder().metricPercentiles("101").build().percentiles();
            fail("a percentile above 100 must be refused");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("[0, 100]"));
        }
    }

    @Test
    public void theDepositMapsAreCopies() {
        ServeBenchOptions.Builder builder = ServeBenchOptions.builder()
                .header("X-A", "1").metadata(java.util.Map.of("run", "one")).samplingParam("temperature", 0.5);
        ServeBenchOptions opts = builder.build();
        builder.header("X-B", "2");
        assertFalse(opts.headers().containsKey("X-B"));
        assertEquals("1", opts.headers().get("X-A"));
        assertEquals(0.5, (Double) opts.samplingParams().get("temperature"), 0.0);
        opts.metadata().put("run", "two");
        assertEquals("one", opts.metadata().get("run"));
    }

    @Test
    public void theLoraListIsNullUnlessNamed() {
        assertEquals(null, ServeBenchOptions.builder().build().loraModules());
        List<String> loras = ServeBenchOptions.builder().loraModules(Arrays.asList("a", "b")).build()
                .loraModules();
        assertEquals(Arrays.asList("a", "b"), loras);
    }
}
