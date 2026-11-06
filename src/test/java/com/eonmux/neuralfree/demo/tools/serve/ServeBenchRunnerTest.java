package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The whole serving benchmark against an OpenAI-compatible server: the discovery, the random
 * workload, the streamed run, the metrics, the result JSON and its file, the failure report and the
 * append mode.
 */
public class ServeBenchRunnerTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private FakeOpenAiServer server;

    @Before
    public void start() throws IOException {
        server = new FakeOpenAiServer();
    }

    @After
    public void stop() {
        server.close();
    }

    private ServeBenchOptions.Builder options() {
        return ServeBenchOptions.builder()
                .baseUrl(server.baseUrl())
                .datasetName("random")
                .numPrompts(3)
                .seed(1)
                .randomInputLen(4)
                .randomOutputLen(2)
                .percentileMetrics("ttft,tpot,itl,e2el")
                .metricPercentiles("50,99");
    }

    @Test
    public void aRunReportsItsMetricsAndSavesTheResult() throws IOException {
        server.tokensPerResponse = 2;
        ServeBenchOptions opts = options().saveResult(true).resultDir(tmp.getRoot().getAbsolutePath()).build();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        Map<String, Object> result = ServeBenchRunner.execute(opts, out, err);

        assertEquals(3, result.get("completed"));
        assertEquals(0, result.get("failed"));
        assertEquals(12L, result.get("total_input_tokens"));
        assertEquals(6L, result.get("total_output_tokens"));
        assertEquals("/models/base.gguf", result.get("model_id"));
        assertEquals("/models/base.gguf", result.get("tokenizer_id"));
        assertEquals("openai", result.get("endpoint_type"));
        assertEquals(3, result.get("num_prompts"));
        assertNotNull(result.get("mean_ttft_ms"));
        assertNotNull(result.get("p50_ttft_ms"));
        assertNotNull(result.get("p99_e2el_ms"));
        assertNotNull(result.get("mean_tpot_ms"));
        assertNotNull(result.get("max_output_tokens_per_s"));
        assertTrue((Double) result.get("request_throughput") > 0.0);
        assertFalse("the per-request arrays need --save-detailed", result.containsKey("ttfts"));

        String report = out.toString();
        assertTrue(report, report.contains("First model name: served-model"));
        assertTrue(report, report.contains("Skipping endpoint ready check."));
        assertTrue(report, report.contains("Starting main benchmark run..."));
        assertTrue(report, report.contains("Traffic request rate: inf"));
        assertTrue(report, report.contains("Burstiness factor: 1.0 (Poisson process)"));
        assertTrue(report, report.contains("Successful requests:"));
        assertTrue(report, report.contains(" Serving Benchmark Result "));

        java.io.File[] files = tmp.getRoot().listFiles((dir, name) -> name.endsWith(".json"));
        assertNotNull(files);
        assertEquals(1, files.length);
        assertTrue(files[0].getName(), files[0].getName().startsWith("openai-infqps-base.gguf-"));
        String saved = Files.readString(files[0].toPath());
        assertTrue(saved, saved.contains("\"completed\""));
    }

    @Test
    public void theDetailedResultCarriesThePerRequestArrays() throws IOException {
        server.tokensPerResponse = 3;
        ServeBenchOptions opts = options().saveResult(true).saveDetailed(true)
                .resultDir(tmp.getRoot().getAbsolutePath()).build();
        Map<String, Object> result = ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertEquals(3, ((List<?>) result.get("ttfts")).size());
        assertEquals(3, ((List<?>) result.get("itls")).size());
        assertEquals(3, ((List<?>) result.get("generated_texts")).size());
        assertEquals(Arrays.asList(4, 4, 4), result.get("input_lens"));
        assertEquals(Arrays.asList(3, 3, 3), result.get("output_lens"));
    }

    @Test
    public void failedRequestsAreCountedAndReported() throws IOException {
        server.generationStatus = 500;
        ServeBenchOptions opts = options().build();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        Map<String, Object> result = ServeBenchRunner.execute(opts, out, err);
        assertEquals(0, result.get("completed"));
        assertEquals(3, result.get("failed"));
        assertTrue(err.toString(), err.toString().contains("Failed requests during the benchmark run"));
        assertTrue(err.toString(), err.toString().contains("HTTP 500"));
    }

    @Test
    public void theRequestIdPrefixNamesEveryRequest() throws IOException {
        ServeBenchOptions opts = options().numPrompts(2).requestIdPrefix("probe-").build();
        ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertTrue(server.requestIds().contains("probe-0"));
        assertTrue(server.requestIds().contains("probe-1"));
    }

    @Test
    public void appendResultAddsASecondRecord() throws IOException {
        Path dir = tmp.newFolder("results").toPath();
        Map<String, Object> first = ServeBenchRunner.execute(options().numPrompts(1).saveResult(true)
                .resultDir(dir.toString()).resultFilename("runs.json").build(),
                new StringBuilder(), new StringBuilder());
        assertNotNull(first);
        ServeBenchRunner.execute(options().numPrompts(1).appendResult(true)
                .resultDir(dir.toString()).resultFilename("runs.json").build(),
                new StringBuilder(), new StringBuilder());
        String text = Files.readString(dir.resolve("runs.json"), StandardCharsets.UTF_8);
        assertEquals("two records appended", 2, text.split("\n").length);
    }

    @Test
    public void anExplicitModelSkipsTheDiscovery() throws IOException {
        ServeBenchOptions opts = options().model("given-model").servedModelName("served-name").build();
        StringBuilder out = new StringBuilder();
        ServeBenchRunner.execute(opts, out, new StringBuilder());
        assertFalse(out.toString().contains("Model not specified"));
        assertEquals("served-name", server.bodies().get(0).get("model"));
    }

    @Test
    public void theSpeculativeDecodingMetricsAreDiffedAcrossTheRun() throws IOException {
        server.metricsDraftsPerScrape = 10;
        server.metricsDraftTokensPerScrape = 40;
        server.metricsAcceptedTokensPerScrape = 20;
        server.metricsPerPositionAcceptedPerScrape = 5;
        Map<String, Object> result = ServeBenchRunner.execute(options().build(),
                new StringBuilder(), new StringBuilder());
        assertEquals(50.0, (Double) result.get("spec_decode_acceptance_rate"), 1e-9);
        assertEquals(3.0, (Double) result.get("spec_decode_acceptance_length"), 1e-9);
        assertEquals(10L, result.get("spec_decode_num_drafts"));
        assertEquals(40L, result.get("spec_decode_draft_tokens"));
        assertEquals(20L, result.get("spec_decode_accepted_tokens"));
        assertEquals(2, ((List<?>) result.get("spec_decode_per_position_acceptance_rates")).size());
    }

    @Test
    public void anAudioRunReportsItsRtfx() throws IOException {
        Path dir = tmp.newFolder("audio").toPath();
        Files.write(dir.resolve("clip.wav"), wav(0.2));
        Path dataset = dir.resolve("audio.jsonl");
        Files.write(dataset, "{\"prompt\":\"hi\",\"audio\":\"clip.wav\"}\n"
                .getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = ServeBenchOptions.builder().baseUrl(server.baseUrl())
                .datasetName("custom_audio").datasetPath(dataset.toString()).numPrompts(1)
                .customOutputLen(2).seed(1).build();
        Map<String, Object> result = ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertEquals("openai-audio", result.get("endpoint_type"));
        assertEquals(1, result.get("completed"));
        assertTrue("the audio duration feeds RTFx", (Double) result.get("rtfx") > 0.0);
    }

    @Test
    public void aRerankRunReportsThePoolingMetrics() throws IOException {
        ServeBenchOptions opts = ServeBenchOptions.builder().baseUrl(server.baseUrl())
                .datasetName("random-rerank").numPrompts(2).seed(1).randomInputLen(10)
                .requestIdPrefix("rr-").build();
        Map<String, Object> result = ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertEquals("vllm-rerank", result.get("endpoint_type"));
        assertEquals(2, result.get("completed"));
        assertEquals(2L, result.get("total_input_sequences"));
        assertNotNull(result.get("input_sequence_throughput"));
        assertNotNull(result.get("mean_e2el_ms"));
        assertFalse("a pooling run reports no generation metrics", result.containsKey("mean_ttft_ms"));
        assertFalse("a pooling run carries no placeholder rtfx", result.containsKey("rtfx"));
    }

    @Test
    public void probeRequestsReportTheirLatency() throws IOException {
        server.tokenDelayMillis = 1;
        ServeBenchOptions opts = options().numPrompts(20).probeRequestRate(100.0).build();
        StringBuilder out = new StringBuilder();
        Map<String, Object> result = ServeBenchRunner.execute(opts, out, new StringBuilder());
        assertNotNull(result.get("probe_completed"));
        assertTrue(out.toString(), out.toString().contains("Probe Requests"));
    }

    @Test
    public void aLocalTokenizerCountsWithoutTheServerTokenizeRoute() throws IOException {
        Path model = com.eonmux.neuralfree.gguf.testutil.SyntheticModels.llama().writeTempFile();
        Path dataset = tmp.newFile("local.jsonl").toPath();
        Files.write(dataset, "{\"prompt\":\"hello world\"}\n".getBytes(StandardCharsets.UTF_8));
        ServeBenchOptions opts = ServeBenchOptions.builder().baseUrl(server.baseUrl())
                .datasetName("custom").datasetPath(dataset.toString()).numPrompts(1)
                .customOutputLen(2).tokenizerPath(model.toString()).seed(1).build();
        Map<String, Object> result = ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertEquals(1, result.get("completed"));
        assertEquals("the local tokenizer counts, the server is never asked", 0, server.tokenizeCalls);
        assertEquals(model.toString(), result.get("tokenizer_id"));
    }

    @Test
    public void profilePlotsAndTheResultWriteTheirFiles() throws IOException {
        Path dir = tmp.newFolder("profile-results").toPath();
        ServeBenchOptions opts = options().numPrompts(2).profile(true).plotTimeline(true)
                .timelineItlThresholds(Arrays.asList(10.0, 20.0)).plotDatasetStats(true)
                .saveResult(true).resultDir(dir.toString()).resultFilename("run.json").build();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        ServeBenchRunner.execute(opts, out, err);
        assertEquals("the run brackets the workload with the profile window", 1, server.profileStarts);
        assertEquals(1, server.profileStops);
        assertTrue(out.toString(), out.toString().contains("Profiler started"));
        assertTrue(out.toString(), out.toString().contains("Profiler stopped"));
        assertTrue(Files.isRegularFile(dir.resolve("run.json")));
        Path timeline = dir.resolve("run.timeline.html");
        Path stats = dir.resolve("run.dataset_stats.svg");
        assertTrue(Files.isRegularFile(timeline));
        assertTrue(Files.isRegularFile(stats));
        assertTrue(Files.readString(timeline).contains("Request Timeline"));
        assertTrue(Files.readString(stats).contains("Prompt Tokens Distribution"));
        assertTrue(out.toString(), out.toString().contains("Timeline plot saved to:"));
        assertTrue(out.toString(), out.toString().contains("Dataset statistics plot saved to:"));
    }

    @Test
    public void aRefusedProfileWindowIsReportedAndTheRunContinues() throws IOException {
        server.profileStatus = 400;
        ServeBenchOptions opts = options().numPrompts(1).profile(true).build();
        StringBuilder out = new StringBuilder();
        StringBuilder err = new StringBuilder();
        Map<String, Object> result = ServeBenchRunner.execute(opts, out, err);
        assertEquals(1, result.get("completed"));
        assertTrue(err.toString(), err.toString().contains("Profiler start failed"));
        assertTrue(err.toString(), err.toString().contains("Profiler stop failed"));
    }

    @Test
    public void thePlotOnlyRunWritesNoResultJson() throws IOException {
        Path dir = tmp.newFolder("plot-only").toPath();
        ServeBenchOptions opts = options().numPrompts(1).plotTimeline(true)
                .resultDir(dir.toString()).resultFilename("plot.json").build();
        ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
        assertFalse(Files.exists(dir.resolve("plot.json")));
        assertTrue(Files.isRegularFile(dir.resolve("plot.timeline.html")));
    }

    /** A PCM WAVE file of {@code seconds} seconds at 8 kHz, mono, 16 bits. */
    private static byte[] wav(double seconds) {
        int sampleRate = 8000;
        int dataSize = (int) Math.round(seconds * sampleRate) * 2;
        java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(44 + dataSize)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        buffer.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(36 + dataSize);
        buffer.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        buffer.put("fmt ".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(16);
        buffer.putShort((short) 1);
        buffer.putShort((short) 1);
        buffer.putInt(sampleRate);
        buffer.putInt(sampleRate * 2);
        buffer.putShort((short) 2);
        buffer.putShort((short) 16);
        buffer.put("data".getBytes(StandardCharsets.US_ASCII));
        buffer.putInt(dataSize);
        buffer.put(new byte[dataSize]);
        return buffer.array();
    }
}
