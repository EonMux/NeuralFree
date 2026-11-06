package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Optional;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The Prometheus exposition reader and the speculative-decoding diff: vLLM's series, NeuralFree's
 * equivalent counters, the per-position rates and the absent-metric cases.
 */
public class PrometheusMetricsTest {

    private static final String VLLM = "# HELP vllm:spec_decode_num_drafts_total drafts\n"
            + "vllm:spec_decode_num_drafts_total 10\n"
            + "vllm:spec_decode_num_draft_tokens_total 40\n"
            + "vllm:spec_decode_num_accepted_tokens_total 20\n"
            + "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"0\"} 10\n"
            + "vllm:spec_decode_num_accepted_tokens_per_pos_total{position=\"1\"} 5\n"
            + "other_metric 3\n";

    private static final String ZERO = "vllm:spec_decode_num_drafts_total 0\n"
            + "vllm:spec_decode_num_draft_tokens_total 0\n"
            + "vllm:spec_decode_num_accepted_tokens_total 0\n";

    @Test
    public void theSpecDecodeDiffReportsRatesAndLength() {
        PrometheusMetrics before = PrometheusMetrics.parse(ZERO);
        PrometheusMetrics after = PrometheusMetrics.parse(VLLM);
        Optional<PrometheusMetrics.SpecDecode> stats = PrometheusMetrics.specDecode(before, after);
        assertTrue(stats.isPresent());
        assertEquals(10, stats.get().drafts);
        assertEquals(40, stats.get().draftTokens);
        assertEquals(20, stats.get().acceptedTokens);
        assertEquals(50.0, stats.get().acceptanceRate(), 1e-9);
        assertEquals(3.0, stats.get().acceptanceLength(), 1e-9);
        assertEquals(2, stats.get().perPositionRates.size());
        assertEquals(1.0, stats.get().perPositionRates.get(0), 1e-9);
        assertEquals(0.5, stats.get().perPositionRates.get(1), 1e-9);
    }

    @Test
    public void theNeuralFreeCountersStandInForVllm() {
        String before = "neuralfree_spec_rounds_total 2\n"
                + "neuralfree_spec_drafted_tokens_total 8\n"
                + "neuralfree_spec_accepted_tokens_total 4\n";
        String after = "neuralfree_spec_rounds_total 6\n"
                + "neuralfree_spec_drafted_tokens_total 24\n"
                + "neuralfree_spec_accepted_tokens_total 12\n";
        PrometheusMetrics.SpecDecode stats = PrometheusMetrics.specDecode(
                PrometheusMetrics.parse(before), PrometheusMetrics.parse(after)).get();
        assertEquals(4, stats.drafts);
        assertEquals(16, stats.draftTokens);
        assertEquals(8, stats.acceptedTokens);
        assertEquals(50.0, stats.acceptanceRate(), 1e-9);
        assertEquals(3.0, stats.acceptanceLength(), 1e-9);
    }

    @Test
    public void aServerWithoutTheSeriesReportsNoAcceptance() {
        assertFalse(PrometheusMetrics.specDecode(PrometheusMetrics.parse("other 1\n"),
                PrometheusMetrics.parse("other 2\n")).isPresent());
        assertFalse(PrometheusMetrics.specDecode(null, PrometheusMetrics.parse(VLLM)).isPresent());
    }

    @Test
    public void aRunThatDraftedNothingReportsNoAcceptance() {
        assertFalse(PrometheusMetrics.specDecode(PrometheusMetrics.parse(ZERO),
                PrometheusMetrics.parse(ZERO)).isPresent());
    }

    @Test
    public void theResultKeysAreTheReferences() {
        java.util.Map<String, Object> result = new java.util.LinkedHashMap<>();
        PrometheusMetrics.specDecode(PrometheusMetrics.parse(ZERO), PrometheusMetrics.parse(VLLM))
                .get().putInto(result);
        assertEquals(50.0, (Double) result.get("spec_decode_acceptance_rate"), 1e-9);
        assertEquals(3.0, (Double) result.get("spec_decode_acceptance_length"), 1e-9);
        assertEquals(10L, result.get("spec_decode_num_drafts"));
        assertEquals(40L, result.get("spec_decode_draft_tokens"));
        assertEquals(20L, result.get("spec_decode_accepted_tokens"));
        assertTrue(result.containsKey("spec_decode_per_position_acceptance_rates"));
    }
}
