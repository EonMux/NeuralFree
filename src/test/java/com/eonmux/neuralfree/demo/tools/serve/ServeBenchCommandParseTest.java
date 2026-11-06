package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The serving benchmark's command line: the reference's defaults, the underscore spellings of its
 * long flags, the general length mapping onto the dataset flags, and the refusal by name of unknown
 * or unsupported options.
 */
public class ServeBenchCommandParseTest {

    private static ServeBenchOptions parse(String... args) throws ServeBenchCommand.UsageException {
        return ServeBenchCommand.parse(args);
    }

    @Test
    public void theDefaultsAreTheReferenceDefaults() throws Exception {
        ServeBenchOptions opts = parse();
        assertEquals("127.0.0.1", opts.host());
        assertEquals(8000, opts.port());
        assertEquals("/v1/completions", opts.endpoint());
        assertEquals("random", opts.datasetName());
        assertEquals(1000, opts.numPrompts());
        assertEquals(0L, opts.seed());
        assertEquals(Double.POSITIVE_INFINITY, opts.requestRate(), 0.0);
        assertEquals(1.0, opts.burstiness(), 0.0);
        assertNull(opts.maxConcurrency());
        assertEquals("99", opts.metricPercentiles());
        assertEquals(Arrays.asList("ttft", "tpot", "itl"), opts.percentileMetricNames());
        assertEquals(Arrays.asList(99.0), opts.percentiles());
        assertFalse(opts.saveResult());
        assertFalse(opts.insecure());
    }

    @Test
    public void underscoresAreDashesInLongFlags() throws Exception {
        ServeBenchOptions opts = parse("--num_prompts", "5", "--random_input_len", "16",
                "--random_output_len", "8", "--request_rate", "2", "--base_url", "http://example:9");
        assertEquals(5, opts.numPrompts());
        assertEquals(16, opts.randomInputLen());
        assertEquals(8, opts.randomOutputLen());
        assertEquals(2.0, opts.requestRate(), 0.0);
        assertEquals("http://example:9", opts.resolvedBaseUrl());
    }

    @Test
    public void theGeneralLengthsMapOntoTheDatasetFlags() throws Exception {
        ServeBenchOptions opts = parse("--input-len", "64", "--output-len", "32");
        assertEquals(Integer.valueOf(64), opts.inputLen());
        assertEquals(Integer.valueOf(32), opts.outputLen());
        ServeBenchOptions sharegpt = parse("--dataset-name", "sharegpt", "--dataset-path", "/tmp/x.json",
                "--output-len", "7");
        assertEquals(Integer.valueOf(7), sharegpt.sharegptOutputLen());
    }

    @Test
    public void anUnknownOptionIsRefusedByName() {
        try {
            parse("--no-such-flag");
            fail("an unknown option must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--no-such-flag"));
        }
    }

    @Test
    public void repeatableFlagsTakeEveryFollowingValue() throws Exception {
        ServeBenchOptions opts = parse("--header", "a=1", "b=2", "--metadata", "x=1", "y=2",
                "--goodput", "ttft:100", "tpot:50");
        assertEquals("1", opts.headers().get("a"));
        assertEquals("2", opts.headers().get("b"));
        assertEquals("1", opts.metadata().get("x"));
        assertEquals("2", opts.metadata().get("y"));
        assertEquals(Double.valueOf(100.0), opts.goodput().get("ttft"));
        assertEquals(Double.valueOf(50.0), opts.goodput().get("tpot"));
    }

    @Test
    public void aRepeatableFlagNeedsAValue() {
        for (String flag : new String[]{"--header", "--metadata", "--goodput"}) {
            try {
                parse(flag);
                fail(flag + " needs a value");
            } catch (ServeBenchCommand.UsageException e) {
                assertTrue(e.getMessage(), e.getMessage().contains(flag));
            }
        }
    }

    @Test
    public void theChatEndpointRefusesTheTokenIdDatasets() {
        try {
            parse("--endpoint", "/v1/chat/completions", "--dataset-name", "random");
            fail("a chat message cannot carry token ids");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("/v1/chat/completions"));
        }
    }

    @Test
    public void theOpenAiBackendIsAccepted() throws Exception {
        assertEquals("/v1/completions", parse("--backend", "openai").endpoint());
    }

    @Test
    public void anUnsupportedReferenceOptionIsRefusedByName() {
        assertRefused(new String[]{"--backend", "vllm"}, "--backend");
        assertRefused(new String[]{"--trust-remote-code"}, "--trust-remote-code");
    }

    @Test
    public void remoteCodeIsRefusedForItsReason() {
        try {
            parse("--trust-remote-code");
            fail("--trust-remote-code must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--trust-remote-code"));
            assertTrue(e.getMessage(), e.getMessage().contains("remote code"));
        }
    }

    @Test
    public void theProfilePlotAndTemplateFlagsParse() throws Exception {
        ServeBenchOptions opts = parse("--profile", "--plot-timeline", "--plot-dataset-stats",
                "--timeline-itl-thresholds", "10,20", "--skip-chat-template", "--no-stream",
                "--chat-template-kwargs", "{\"enable_thinking\": false}");
        assertTrue(opts.profile());
        assertTrue(opts.plotTimeline());
        assertTrue(opts.plotDatasetStats());
        assertEquals(Arrays.asList(10.0, 20.0), opts.timelineItlThresholds());
        assertTrue(opts.skipChatTemplate());
        assertTrue(opts.noStream());
        assertEquals(Boolean.FALSE, opts.chatTemplateKwargs().get("enable_thinking"));
        assertEquals(Arrays.asList(25.0, 50.0), parse().timelineItlThresholds());
    }

    @Test
    public void malformedTimelineThresholdsAreRefused() {
        assertInvalid(new String[]{"--timeline-itl-thresholds", "25"}, "--timeline-itl-thresholds");
        assertInvalid(new String[]{"--timeline-itl-thresholds", "25,50,75"},
                "--timeline-itl-thresholds");
        assertInvalid(new String[]{"--timeline-itl-thresholds", "50,25"}, "--timeline-itl-thresholds");
        assertInvalid(new String[]{"--timeline-itl-thresholds", "-1,25"}, "--timeline-itl-thresholds");
    }

    @Test
    public void malformedChatTemplateKwargsAreRefused() {
        assertInvalid(new String[]{"--chat-template-kwargs", "[]"}, "--chat-template-kwargs");
        assertInvalid(new String[]{"--chat-template-kwargs", "not json"}, "--chat-template-kwargs");
        assertInvalid(new String[]{"--chat-template-kwargs", "{\"a\": 1}", "--skip-tokenizer-init",
                "--dataset-name", "custom", "--dataset-path", "/tmp/c.jsonl"}, "--chat-template-kwargs");
    }

    /** A command line that must refuse, its message naming the offending flag. */
    private static void assertInvalid(String[] args, String flag) {
        try {
            parse(args);
            fail(flag + " must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(flag));
        }
    }

    /** A command line that must refuse, its message naming the offending flag. */
    private static void assertRefused(String[] args, String flag) {
        try {
            parse(args);
            fail(flag + " must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(flag));
            assertTrue(e.getMessage(), e.getMessage().contains("not supported")
                    || e.getMessage().contains("unknown option"));
        }
    }

    @Test
    public void aPositionalArgumentIsRefused() {
        try {
            parse("model.gguf");
            fail("a positional argument must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("model.gguf"));
        }
    }

    @Test
    public void theRangeRatioIsAFloatOrAJsonObject() throws Exception {
        ServeBenchOptions flat = parse("--random-range-ratio", "0.25");
        assertEquals(0.25, flat.randomRangeRatio().input(), 0.0);
        assertEquals(0.25, flat.randomRangeRatio().output(), 0.0);
        ServeBenchOptions split = parse("--random-range-ratio", "{\"input\": 0.1, \"output\": 0.2}");
        assertEquals(0.1, split.randomRangeRatio().input(), 0.0);
        assertEquals(0.2, split.randomRangeRatio().output(), 0.0);
    }

    @Test
    public void aRangeRatioOutsideTheRangeIsRefused() {
        try {
            ServeBenchOptions.parseRangeRatio("1.5");
            fail("a ratio of 1.5 must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("1.5"));
        }
        try {
            ServeBenchOptions.parseRangeRatio("{\"input\": 0.1}");
            fail("a missing output ratio must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("output"));
        }
    }

    @Test
    public void goodputPairsCarryTheirServiceLevels() throws Exception {
        ServeBenchOptions opts = parse("--goodput", "ttft:100", "tpot:50.5", "e2el:2000");
        Map<String, Double> goodput = opts.goodput();
        assertEquals(3, goodput.size());
        assertEquals(100.0, goodput.get("ttft"), 0.0);
        assertEquals(50.5, goodput.get("tpot"), 0.0);
        assertEquals(2000.0, goodput.get("e2el"), 0.0);
    }

    @Test
    public void anUnknownServiceLevelObjectiveIsRefused() throws Exception {
        try {
            parse("--goodput", "itl:10");
            fail("an unknown SLO metric must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("itl"));
        }
    }

    @Test
    public void headersAndMetadataAreKeyValuePairs() throws Exception {
        ServeBenchOptions opts = parse("--header", "X-Tenant=a", "--metadata", "version=1.2",
                "--metadata", "tp=1");
        assertEquals("a", opts.headers().get("X-Tenant"));
        assertEquals("1.2", opts.metadata().get("version"));
        assertEquals("1", opts.metadata().get("tp"));
    }

    @Test
    public void aRampUpRefusesARequestRateAndNeedsBothBounds() throws Exception {
        try {
            parse("--ramp-up-strategy", "linear", "--ramp-up-start-rps", "1", "--ramp-up-end-rps", "2",
                    "--request-rate", "3");
            fail("a ramp-up with --request-rate must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--request-rate"));
        }
        try {
            parse("--ramp-up-strategy", "linear", "--ramp-up-start-rps", "1");
            fail("a ramp-up without an end RPS must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--ramp-up-end-rps"));
        }
        try {
            parse("--ramp-up-strategy", "exponential", "--ramp-up-start-rps", "0", "--ramp-up-end-rps", "4");
            fail("an exponential ramp-up from zero must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("start RPS"));
        }
    }

    @Test
    public void aSyntheticDatasetRefusesADatasetPath() {
        try {
            parse("--dataset-name", "random", "--dataset-path", "/tmp/x.json");
            fail("the random dataset must refuse --dataset-path");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--dataset-path"));
        }
    }

    @Test
    public void anUnknownDatasetIsRefusedAndTheDatasetsAreNamed() {
        try {
            parse("--dataset-name", "bogus");
            fail("an unknown dataset must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("bogus"));
            assertTrue(e.getMessage(), e.getMessage().contains("spec_bench"));
            assertTrue(e.getMessage(), e.getMessage().contains("hf"));
        }
    }

    @Test
    public void thePercentileListsAreParsedAndValidated() throws Exception {
        ServeBenchOptions opts = parse("--percentile-metrics", "ttft,e2el", "--metric-percentiles", "25,50,99");
        assertEquals(Arrays.asList("ttft", "e2el"), opts.percentileMetricNames());
        assertEquals(Arrays.asList(25.0, 50.0, 99.0), opts.percentiles());
        try {
            parse("--percentile-metrics", "ttft,bogus");
            fail("an unknown percentile metric must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("bogus"));
        }
        try {
            parse("--metric-percentiles", "101");
            fail("a percentile above 100 must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("101"));
        }
    }

    @Test
    public void samplingParametersBecomeTheRequestFields() throws Exception {
        ServeBenchOptions opts = parse("--temperature", "0.5", "--top-p", "0.9", "--top-k", "40",
                "--min-p", "0.1", "--logprobs", "3", "--extra-body", "{\"seed\": 7}");
        Map<String, Object> params = opts.samplingParams();
        assertEquals(0.5, params.get("temperature"));
        assertEquals(0.9, params.get("top_p"));
        assertEquals(40, params.get("top_k"));
        assertEquals(0.1, params.get("min_p"));
        assertEquals(7L, params.get("seed"));
        assertEquals(Integer.valueOf(3), opts.logprobs());
    }

    @Test
    public void loraModulesAreListedAndAssigned() throws Exception {
        ServeBenchOptions opts = parse("--lora-modules", "a", "b", "--lora-assignment", "round-robin");
        assertEquals(Arrays.asList("a", "b"), opts.loraModules());
        assertEquals("round-robin", opts.loraAssignment());
        try {
            parse("--lora-assignment", "sequential");
            fail("an unknown assignment must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("sequential"));
        }
    }

    @Test
    public void theHelpPrintsTheSurfaceAndParsesNothing() throws Exception {
        assertNull(ServeBenchCommand.parse(new String[]{"--help"}));
    }

    @Test
    public void requestRatesAcceptTheInfinitySpellings() {
        assertEquals(Double.POSITIVE_INFINITY, ServeBenchOptions.parseRequestRate("inf"), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, ServeBenchOptions.parseRequestRate("Infinity"), 0.0);
        assertEquals(2.5, ServeBenchOptions.parseRequestRate("2.5"), 0.0);
        try {
            ServeBenchOptions.parseRequestRate("0");
            fail("a zero rate must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("0"));
        }
    }

    @Test
    public void anEmptyMetricListIsRefused() {
        try {
            ServeBenchOptions.builder().percentileMetrics("").build().percentileMetricNames();
            fail("an empty metric list must be refused");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("at least one"));
        }
    }

    @Test
    public void thePercentileNamesOfNonIntegerValuesKeepTheirDecimals() {
        assertEquals("99", ServeBenchMetrics.percentileName(99.0));
        assertEquals("99.9", ServeBenchMetrics.percentileName(99.9));
    }

    @Test
    public void theCustomDatasetOutputLengthAcceptsMinusOne() throws Exception {
        assertEquals(-1, parse("--custom-output-len", "-1").customOutputLen());
    }

    @Test
    public void theInsecureFlagIsCarried() throws Exception {
        assertTrue(parse("--insecure").insecure());
    }

    @Test
    public void aMissingFlagValueIsRefused() {
        try {
            parse("--host");
            fail("--host without a value must be refused");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--host"));
        }
    }

    @Test
    public void theReportedMetricNamesAreDistinct() throws Exception {
        List<String> names = parse("--percentile-metrics", "ttft,ttft,e2el").percentileMetricNames();
        assertEquals(Arrays.asList("ttft", "e2el"), names);
    }

    @Test
    public void theNewDatasetFlagsParse() throws Exception {
        ServeBenchOptions opts = parse("--dataset-name", "random-mm", "--num-prompts", "3",
                "--random-mm-base-items-per-request", "2",
                "--random-mm-num-mm-items-range-ratio", "0.5",
                "--random-mm-limit-mm-per-prompt", "{\"image\": 3}",
                "--random-mm-bucket-config", "{(4, 4, 1): 1.0}",
                "--random-batch-size", "2", "--no-reranker");
        assertEquals(2, opts.randomMmBaseItemsPerRequest());
        assertEquals(0.5, opts.randomMmNumMmItemsRangeRatio(), 0.0);
        assertEquals(Integer.valueOf(3), opts.randomMmLimitMmPerPrompt().get("image"));
        assertEquals(1.0, opts.randomMmBucketConfig().get("4,4,1"), 0.0);
        assertEquals(2, opts.randomBatchSize());
        assertTrue(opts.noReranker());
        ServeBenchOptions trace = parse("--dataset-name", "timed_trace", "--dataset-path", "/tmp/t.jsonl",
                "--self-timed", "--timed-trace-chunk-hash-size", "4", "--timed-trace-sec-multiplier", "0.001",
                "--timed-trace-label-timestamp", "ts", "--timed-trace-label-input-length", "in",
                "--timed-trace-label-output-length", "out", "--timed-trace-label-hash-ids", "hashes");
        assertTrue(trace.selfTimed());
        assertEquals(4, trace.timedTraceChunkHashSize());
        assertEquals(0.001, trace.timedTraceSecMultiplier(), 0.0);
        assertEquals("ts", trace.timedTraceLabelTimestamp());
        ServeBenchOptions edges = parse("--dataset-name", "blazedit", "--dataset-path", "/tmp/e.jsonl",
                "--blazedit-min-distance", "0.1", "--blazedit-max-distance", "0.9",
                "--asr-min-audio-len-sec", "0.5", "--asr-max-audio-len-sec", "10");
        assertEquals(0.1, edges.blazeditMinDistance(), 0.0);
        assertEquals(0.9, edges.blazeditMaxDistance(), 0.0);
        assertEquals(0.5, edges.asrMinAudioLenSec(), 0.0);
        assertEquals(10.0, edges.asrMaxAudioLenSec(), 0.0);
    }

    @Test
    public void theTokenizerAndProbeFlagsParse() throws Exception {
        String dir = System.getProperty("java.io.tmpdir");
        ServeBenchOptions local = parse("--tokenizer", dir, "--tokenizer-mode", "hf");
        assertEquals(dir, local.tokenizerPath());
        assertEquals("hf", local.tokenizerMode());
        ServeBenchOptions free = parse("--dataset-name", "custom", "--dataset-path", "/tmp/c.jsonl",
                "--skip-tokenizer-init");
        assertTrue(free.skipTokenizerInit());
        assertTrue(parse("--use-beam-search").useBeamSearch());
        assertEquals(2.5, parse("--probe-request-rate", "2.5").probeRequestRate(), 0.0);
        try {
            parse("--skip-tokenizer-init");
            fail("the random dataset needs a tokenizer");
        } catch (ServeBenchCommand.UsageException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("--skip-tokenizer-init"));
        }
    }
}
