package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.Test;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code bench-serve}'s command line: every flag of the reference parse into the options, underscores
 * spell like dashes, and the options the harness cannot honor are refused by name rather than
 * ignored.
 */
public class ServeBenchCommandTest {

    private static ServeBenchOptions parse(String... args) throws ServeBenchCommand.UsageException {
        return ServeBenchCommand.parse(args);
    }

    @Test
    public void everySupportedFlagParses() throws Exception {
        ServeBenchOptions opts = parse(
                "--host", "10.0.0.1", "--port", "9000", "--endpoint", "/v1/chat/completions",
                "--model", "m", "--served-model-name", "served", "--label", "run",
                "--header", "X-Key=value",
                "--dataset-name", "sharegpt", "--dataset-path", "data.json",
                "--num-prompts", "12", "--seed", "7", "--no-oversample", "--disable-shuffle",
                "--input-len", "64", "--output-len", "16",
                "--random-input-len", "128", "--random-output-len", "32",
                "--random-range-ratio", "{\"input\":0.1,\"output\":0.2}", "--random-prefix-len", "8",
                "--prefix-repetition-prefix-len", "16", "--prefix-repetition-suffix-len", "17",
                "--prefix-repetition-num-prefixes", "4", "--prefix-repetition-output-len", "9",
                "--sharegpt-output-len", "20", "--custom-output-len", "-1",
                "--request-rate", "3.5", "--burstiness", "2", "--max-concurrency", "6",
                "--percentile-metrics", "ttft,e2el", "--metric-percentiles", "50,90",
                "--goodput", "ttft:100", "e2el:2000",
                "--ignore-eos", "--num-warmups", "3", "--ready-check-timeout-sec", "9",
                "--save-result", "--save-detailed", "--append-result",
                "--result-dir", "out", "--result-filename", "r.json",
                "--metadata", "version=1", "--request-id-prefix", "p-", "--disable-tqdm",
                "--temperature", "0.7", "--top-p", "0.9", "--top-k", "40", "--min-p", "0.05",
                "--frequency-penalty", "0.1", "--presence-penalty", "0.2", "--repetition-penalty", "1.1",
                "--logprobs", "2", "--extra-body", "{\"seed\":42}",
                "--lora-modules", "a", "b", "--lora-assignment", "round-robin", "--insecure");
        assertEquals("10.0.0.1", opts.host());
        assertEquals(9000, opts.port());
        assertEquals("/v1/chat/completions", opts.endpoint());
        assertEquals("m", opts.model());
        assertEquals("served", opts.servedModelName());
        assertEquals("run", opts.label());
        assertEquals("value", opts.headers().get("X-Key"));
        assertEquals("sharegpt", opts.datasetName());
        assertEquals("data.json", opts.datasetPath());
        assertEquals(12, opts.numPrompts());
        assertEquals(7, opts.seed());
        assertTrue(opts.noOversample());
        assertTrue(opts.disableShuffle());
        assertEquals(Integer.valueOf(64), opts.inputLen());
        assertEquals(Integer.valueOf(16), opts.outputLen());
        // --input-len/--output-len override the dataset-specific lengths, as the reference maps them
        assertEquals(64, opts.randomInputLen());
        assertEquals(16, opts.randomOutputLen());
        assertEquals(0.1, opts.randomRangeRatio().input(), 0.0);
        assertEquals(0.2, opts.randomRangeRatio().output(), 0.0);
        assertEquals(8, opts.randomPrefixLen());
        assertEquals(16, opts.prefixRepetitionPrefixLen());
        assertEquals(17, opts.prefixRepetitionSuffixLen());
        assertEquals(4, opts.prefixRepetitionNumPrefixes());
        // --output-len overrides the dataset-specific output flags, as the reference maps them
        assertEquals(16, opts.prefixRepetitionOutputLen());
        assertEquals(Integer.valueOf(16), opts.sharegptOutputLen());
        assertEquals(16, opts.customOutputLen());
        assertEquals(3.5, opts.requestRate(), 0.0);
        assertEquals(2.0, opts.burstiness(), 0.0);
        assertEquals(Integer.valueOf(6), opts.maxConcurrency());
        assertEquals("ttft,e2el", opts.percentileMetrics());
        assertEquals("50,90", opts.metricPercentiles());
        assertEquals(100.0, opts.goodput().get("ttft"), 0.0);
        assertEquals(2000.0, opts.goodput().get("e2el"), 0.0);
        assertTrue(opts.ignoreEos());
        assertEquals(3, opts.numWarmups());
        assertEquals(9, opts.readyCheckTimeoutSec());
        assertTrue(opts.saveResult());
        assertTrue(opts.saveDetailed());
        assertTrue(opts.appendResult());
        assertEquals("out", opts.resultDir());
        assertEquals("r.json", opts.resultFilename());
        assertEquals("1", opts.metadata().get("version"));
        assertEquals("p-", opts.requestIdPrefix());
        assertTrue(opts.disableTqdm());
        Map<String, Object> sampling = opts.samplingParams();
        assertEquals(0.7, (Double) sampling.get("temperature"), 0.0);
        assertEquals(0.9, (Double) sampling.get("top_p"), 0.0);
        assertEquals(40, sampling.get("top_k"));
        assertEquals(0.05, (Double) sampling.get("min_p"), 0.0);
        assertEquals(0.1, (Double) sampling.get("frequency_penalty"), 0.0);
        assertEquals(0.2, (Double) sampling.get("presence_penalty"), 0.0);
        assertEquals(1.1, (Double) sampling.get("repetition_penalty"), 0.0);
        assertEquals(42L, sampling.get("seed"));
        assertEquals(Integer.valueOf(2), opts.logprobs());
        assertEquals(Arrays.asList("a", "b"), opts.loraModules());
        assertEquals("round-robin", opts.loraAssignment());
        assertTrue(opts.insecure());
    }

    @Test
    public void underscoresSpellLikeDashes() throws Exception {
        ServeBenchOptions opts = parse("--random_input_len", "64", "--num_prompts", "5",
                "--dataset_name", "random");
        assertEquals(64, opts.randomInputLen());
        assertEquals(5, opts.numPrompts());
        assertEquals("random", opts.datasetName());
    }

    @Test
    public void theGeneralLengthsMapOntoTheDatasetFlags() throws Exception {
        ServeBenchOptions opts = parse("--input-len", "48", "--output-len", "24");
        assertEquals(48, opts.randomInputLen());
        assertEquals(24, opts.randomOutputLen());
        assertEquals(Integer.valueOf(24), opts.sharegptOutputLen());
        assertEquals(24, opts.customOutputLen());
        assertEquals(24, opts.prefixRepetitionOutputLen());
        assertEquals(Integer.valueOf(48), opts.inputLen());
    }

    @Test
    public void theRampUpTakesThePlaceOfTheRequestRate() throws Exception {
        ServeBenchOptions opts = parse("--ramp-up-strategy", "linear", "--ramp-up-start-rps", "1",
                "--ramp-up-end-rps", "8");
        assertEquals("linear", opts.rampUpStrategy());
        assertEquals(Integer.valueOf(1), opts.rampUpStartRps());
        assertEquals(Integer.valueOf(8), opts.rampUpEndRps());
        try {
            parse("--ramp-up-strategy", "linear", "--ramp-up-start-rps", "1",
                    "--ramp-up-end-rps", "8", "--request-rate", "2");
            fail("a ramp-up and a request rate cannot be combined");
        } catch (ServeBenchCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("--request-rate"));
        }
        try {
            parse("--ramp-up-strategy", "linear", "--ramp-up-start-rps", "1");
            fail("a ramp-up needs both rates");
        } catch (ServeBenchCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("--ramp-up-end-rps"));
        }
        try {
            parse("--ramp-up-strategy", "exponential", "--ramp-up-start-rps", "0",
                    "--ramp-up-end-rps", "8");
            fail("an exponential ramp-up cannot start at 0");
        } catch (ServeBenchCommand.UsageException expected) {
            assertTrue(expected.getMessage().contains("cannot be 0"));
        }
    }

    @Test
    public void unknownAndUnsupportedOptionsAreRefusedByName() {
        assertRefused(new String[]{"--no-such-flag"}, "unknown option --no-such-flag");
        assertRefused(new String[]{"--backend", "vllm"}, "--backend");
        assertRefused(new String[]{"--tokenizer", "does-not-exist"}, "--tokenizer");
        assertRefused(new String[]{"--tokenizer-mode", "mistral"}, "--tokenizer-mode");
        assertRefused(new String[]{"--skip-tokenizer-init"}, "--skip-tokenizer-init");
        assertRefused(new String[]{"--use-beam-search", "--dataset-name", "custom",
                "--endpoint", "/rerank"}, "--use-beam-search");
        assertRefused(new String[]{"--trust-remote-code"}, "--trust-remote-code");
        assertRefused(new String[]{"--timeline-itl-thresholds", "1"}, "--timeline-itl-thresholds");
        assertRefused(new String[]{"--chat-template-kwargs", "[]"}, "--chat-template-kwargs");
        assertRefused(new String[]{"--self-timed"}, "--self-timed");
        assertRefused(new String[]{"--probe-request-rate", "1", "--dataset-name", "custom",
                "--endpoint", "/rerank"}, "--probe-request-rate");
        assertRefused(new String[]{"--dataset-name", "bogus"}, "bogus");
        assertRefused(new String[]{"--goodput", "tps:10"}, "ttft, tpot or e2el");
        assertRefused(new String[]{"--header", "no-equals"}, "--header");
        assertRefused(new String[]{"--metadata", "no-equals"}, "--metadata");
        assertRefused(new String[]{"--lora-assignment", "first"}, "--lora-assignment");
        assertRefused(new String[]{"--extra-body", "[]"}, "--extra-body");
        assertRefused(new String[]{"positional"}, "no positional argument");
        assertRefused(new String[]{"--port", "70000"}, "[0, 65535]");
        assertRefused(new String[]{"--random-input-len", "0"}, "positive integer");
        assertRefused(new String[]{"--metric-percentiles", "abc"}, "--metric-percentiles");
        assertRefused(new String[]{"--percentile-metrics", "tps"}, "'tps'");
        assertRefused(new String[]{"--request-rate", "0"}, "must be positive");
        assertRefused(new String[]{"--burstiness", "0"}, "must be positive");
    }

    @Test
    public void aSyntheticDatasetRefusesADatasetPath() {
        assertRefused(new String[]{"--dataset-name", "random", "--dataset-path", "x.json"}, "--dataset-path");
        assertRefused(new String[]{"--dataset-name", "prefix_repetition", "--dataset-path", "x.json"},
                "--dataset-path");
    }

    @Test
    public void theHelpSurfaceNamesEveryFlag() throws Exception {
        assertNull(parse("--help"));
        assertTrue(ServeBenchCommand.USAGE.contains("--request-rate"));
        assertTrue(ServeBenchCommand.USAGE.contains("--percentile-metrics"));
        assertTrue(ServeBenchCommand.USAGE.contains("--metadata"));
        assertTrue(ServeBenchCommand.USAGE.contains("--lora-modules"));
        assertTrue(ServeBenchCommand.USAGE.contains("--save-result"));
    }

    private static void assertRefused(String[] args, String fragment) {
        try {
            parse(args);
            fail("the command line must be refused: " + Arrays.toString(args));
        } catch (ServeBenchCommand.UsageException expected) {
            assertTrue("expected '" + fragment + "' in: " + expected.getMessage(),
                    expected.getMessage().contains(fragment));
        } catch (IllegalArgumentException expected) {
            assertTrue("expected '" + fragment + "' in: " + expected.getMessage(),
                    expected.getMessage().contains(fragment));
        }
    }
}
