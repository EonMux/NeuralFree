package com.eonmux.neuralfree.demo.tools.serve;

import com.eonmux.neuralfree.utils.json.Json;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The whole serving benchmark over a real OpenAI-compatible HTTP server: discovery, dataset sampling,
 * streaming, metrics, the report and the saved result file, plus the latency and throughput commands
 * over the same server.
 */
public class ServeBenchEndToEndTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static ServeBenchOptions randomOptions(String baseUrl, Path resultDir) {
        return ServeBenchOptions.builder()
                .baseUrl(baseUrl)
                .datasetName("random")
                .numPrompts(3)
                .seed(1)
                .randomInputLen(8)
                .randomOutputLen(3)
                .requestIdPrefix("e2e-")
                .disableTqdm(true)
                .resultDir(resultDir.toString())
                .resultFilename("result.json")
                .saveResult(true)
                .build();
    }

    @Test
    public void theWholeRunDiscoversStreamsAndSavesItsResult() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            Path dir = tmp.newFolder("out").toPath();
            ServeBenchOptions opts = randomOptions(server.baseUrl(), dir);
            StringBuilder out = new StringBuilder();
            StringBuilder err = new StringBuilder();
            Map<String, Object> result = ServeBenchRunner.execute(opts, out, err);

            assertTrue(out.toString(), out.toString().contains(
                    "First model name: served-model, first model id: /models/base.gguf"));
            assertTrue(out.toString(), out.toString().contains("Starting main benchmark run..."));
            assertEquals("/models/base.gguf", result.get("model_id"));
            assertEquals(3, result.get("completed"));
            assertEquals(0, result.get("failed"));
            assertEquals(24L, ((Number) result.get("total_input_tokens")).longValue());
            assertEquals(6L, ((Number) result.get("total_output_tokens")).longValue());
            assertTrue((Double) result.get("request_throughput") > 0.0);
            assertTrue(result.containsKey("mean_ttft_ms"));
            assertTrue(result.containsKey("mean_tpot_ms"));
            assertTrue(result.containsKey("mean_itl_ms"));
            assertFalse("e2el is not a reported metric unless selected",
                    result.containsKey("mean_e2el_ms"));

            Path file = dir.resolve("result.json");
            assertTrue("the result file is written", Files.isRegularFile(file));
            Object saved = Json.parse(Files.readString(file), Json.Limits.UNTRUSTED);
            assertTrue(saved instanceof Map);
            assertEquals(3, ((Number) ((Map<?, ?>) saved).get("completed")).intValue());

            assertEquals("every request reached the server", 3, server.bodies().size());
            for (Map<String, Object> body : server.bodies()) {
                assertEquals("served-model", body.get("model"));
                assertEquals(true, body.get("stream"));
                assertEquals(true, body.get("ignore_eos"));
                assertEquals(3, ((Number) body.get("max_tokens")).intValue());
                List<?> prompt = (List<?>) body.get("prompt");
                assertEquals("the random prompt is a token-id prompt of the sampled length", 8, prompt.size());
            }
            assertEquals(3, server.requestIds().size());
            assertTrue(server.requestIds().get(0).startsWith("e2e-"));
        }
    }

    @Test
    public void aChatEndpointRunsTextPromptsAsMessages() throws Exception {
        File data = tmp.newFile("prompts.jsonl");
        Files.write(data.toPath(), ("{\"prompt\": \"a b c d\", \"output_tokens\": 4}\n"
                + "{\"prompt\": \"e f g h\", \"output_tokens\": 5}\n").getBytes(StandardCharsets.UTF_8));
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            ServeBenchOptions opts = ServeBenchOptions.builder()
                    .baseUrl(server.baseUrl())
                    .endpoint("/v1/chat/completions")
                    .datasetName("custom")
                    .datasetPath(data.getAbsolutePath())
                    .numPrompts(2)
                    .customOutputLen(-1)
                    .requestRate(Double.POSITIVE_INFINITY)
                    .disableTqdm(true)
                    .build();
            Map<String, Object> result = ServeBenchRunner.execute(opts, new StringBuilder(),
                    new StringBuilder());
            assertEquals(2, result.get("completed"));
            for (Map<String, Object> body : server.bodies()) {
                List<?> messages = (List<?>) body.get("messages");
                assertEquals("user", ((Map<?, ?>) messages.get(0)).get("role"));
                assertTrue(((Map<?, ?>) messages.get(0)).get("content") instanceof String);
            }
        }
    }

    @Test
    public void theReadyCheckWaitsForTheEndpoint() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            ServeBenchOptions opts = ServeBenchOptions.builder()
                    .baseUrl(server.baseUrl())
                    .datasetName("random")
                    .numPrompts(1)
                    .randomInputLen(4)
                    .randomOutputLen(2)
                    .readyCheckTimeoutSec(2)
                    .requestRate(Double.POSITIVE_INFINITY)
                    .disableTqdm(true)
                    .build();
            StringBuilder err = new StringBuilder();
            ServeBenchRunner.execute(opts, new StringBuilder(), err);
            assertTrue(err.toString(), err.toString().contains("Initial test run completed."));
        }
    }

    @Test
    public void theReadyCheckFailureNamesTheArguments() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            server.generationStatus = 500;
            ServeBenchOptions opts = ServeBenchOptions.builder()
                    .baseUrl(server.baseUrl())
                    .datasetName("random")
                    .numPrompts(1)
                    .randomInputLen(4)
                    .randomOutputLen(2)
                    .readyCheckTimeoutSec(1)
                    .disableTqdm(true)
                    .build();
            try {
                ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
                throw new AssertionError("a failed ready check must stop the run");
            } catch (java.io.IOException e) {
                assertTrue(e.getMessage(), e.getMessage().contains("Initial test run failed"));
            }
        }
    }

    @Test
    public void appendingKeepsEveryRunInTheFile() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            Path dir = tmp.newFolder("append").toPath();
            ServeBenchOptions opts = ServeBenchOptions.builder()
                    .baseUrl(server.baseUrl())
                    .datasetName("random")
                    .numPrompts(1)
                    .randomInputLen(4)
                    .randomOutputLen(2)
                    .requestRate(Double.POSITIVE_INFINITY)
                    .disableTqdm(true)
                    .resultDir(dir.toString())
                    .resultFilename("runs.json")
                    .appendResult(true)
                    .build();
            ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
            ServeBenchRunner.execute(opts, new StringBuilder(), new StringBuilder());
            List<String> lines = Files.readAllLines(dir.resolve("runs.json"));
            assertEquals(2, lines.size());
            for (String line : lines) {
                assertTrue(Json.parse(line, Json.Limits.UNTRUSTED) instanceof Map);
            }
        }
    }

    @Test
    public void theLatencyCommandTimesWholeBatches() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            BenchLatencyCommand.Args args = new BenchLatencyCommand.Args(server.baseUrl(),
                    "/v1/completions", null, new LinkedHashMap<>(), false, 4, 2, 2, 1, 0, 3, null);
            StringBuilder out = new StringBuilder();
            Map<String, Object> result = BenchLatencyCommand.bench(args, out);
            assertEquals(3, ((List<?>) result.get("latencies")).size());
            assertTrue((Double) result.get("avg_latency") >= 0.0);
            Map<?, ?> percentiles = (Map<?, ?>) result.get("percentiles");
            assertTrue(percentiles.containsKey("50"));
            assertTrue(out.toString(), out.toString().contains("Avg latency:"));
            assertTrue(out.toString(), out.toString().contains("99% percentile latency:"));
        }
    }

    @Test
    public void theThroughputCommandSendsEverythingAtOnce() throws Exception {
        try (FakeOpenAiServer server = new FakeOpenAiServer()) {
            ServeBenchOptions opts = ServeBenchCommand.parse(new String[]{
                    "--base-url", server.baseUrl(), "--dataset-name", "random", "--num-prompts", "4",
                    "--random-input-len", "4", "--random-output-len", "2", "--disable-tqdm"});
            assertTrue(Double.isInfinite(opts.requestRate()));
            assertEquals(null, opts.maxConcurrency());
            Map<String, Object> result = ServeBenchCommand.execute(opts, new StringBuilder(),
                    new StringBuilder());
            assertEquals(4, result.get("completed"));
        }
    }
}
