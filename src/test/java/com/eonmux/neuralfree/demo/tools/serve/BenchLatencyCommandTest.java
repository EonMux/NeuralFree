package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * {@code bench-latency}: the reference's defaults and fixed sampling, its refused flags, and a run
 * over an OpenAI-compatible server whose report names the average and the percentiles.
 */
public class BenchLatencyCommandTest {

    private FakeOpenAiServer server;

    @After
    public void stop() {
        if (server != null) {
            server.close();
        }
    }

    @Test
    public void theDefaultsMatchTheReference() throws Exception {
        BenchLatencyCommand.Args args = BenchLatencyCommand.parse(new String[]{});
        assertEquals("http://127.0.0.1:8000", args.baseUrl);
        assertEquals("/v1/completions", args.endpoint);
        assertNull(args.model);
        assertEquals(32, args.inputLen);
        assertEquals(128, args.outputLen);
        assertEquals(8, args.batchSize);
        assertEquals(1, args.n);
        assertEquals(10, args.numItersWarmup);
        assertEquals(30, args.numIters);
        assertNull(args.outputJson);
    }

    @Test
    public void theFlagsParse() throws Exception {
        BenchLatencyCommand.Args args = BenchLatencyCommand.parse(new String[]{
                "--host", "h", "--port", "9", "--endpoint", "/v1/chat/completions", "--model", "m",
                "--header", "X=1", "--insecure", "--input-len", "4", "--output-len", "5",
                "--batch-size", "2", "--n", "3", "--num-iters-warmup", "0", "--num-iters", "1",
                "--output-json", "out.json"});
        assertEquals("http://h:9", args.baseUrl);
        assertEquals("/v1/chat/completions", args.endpoint);
        assertEquals("m", args.model);
        assertEquals("1", args.headers.get("X"));
        assertTrue(args.insecure);
        assertEquals(4, args.inputLen);
        assertEquals(5, args.outputLen);
        assertEquals(2, args.batchSize);
        assertEquals(3, args.n);
        assertEquals(0, args.numItersWarmup);
        assertEquals(1, args.numIters);
        assertEquals("out.json", args.outputJson);
    }

    @Test
    public void theUnsupportedFlagsAreRefusedByName() {
        for (String flag : new String[]{"--use-beam-search", "--disable-detokenize", "--nonsense"}) {
            try {
                BenchLatencyCommand.parse(new String[]{flag});
                fail(flag + " must be refused");
            } catch (BenchLatencyCommand.UsageException expected) {
                assertTrue(expected.getMessage().contains(flag));
            }
        }
    }

    @Test
    public void aRunReportsTheAverageAndThePercentiles() throws Exception {
        server = new FakeOpenAiServer();
        server.tokensPerResponse = 2;
        BenchLatencyCommand.Args args = new BenchLatencyCommand.Args(server.baseUrl(), "/v1/completions",
                "served-model", new java.util.LinkedHashMap<>(), false, 2, 2, 2, 1, 0, 2, null);
        StringBuilder out = new StringBuilder();
        Map<String, Object> result = BenchLatencyCommand.bench(args, out);
        assertEquals("served-model", result.get("model_id"));
        assertEquals(2, ((java.util.List<?>) result.get("latencies")).size());
        assertTrue((Double) result.get("avg_latency") >= 0.0);
        assertEquals(6, ((Map<?, ?>) result.get("percentiles")).size());
        assertTrue(out.toString(), out.toString().contains("Avg latency:"));
        assertTrue(out.toString(), out.toString().contains("99% percentile latency:"));
        // the sampling of the reference: temperature 1.0, top-p 1.0, ignore_eos
        Map<String, Object> body = server.bodies().get(0);
        assertEquals(1.0, (Double) body.get("temperature"), 0.0);
        assertEquals(1.0, (Double) body.get("top_p"), 0.0);
        assertEquals(true, body.get("ignore_eos"));
    }

    @Test
    public void theOutputJsonIsWritten() throws Exception {
        server = new FakeOpenAiServer();
        server.tokensPerResponse = 1;
        java.nio.file.Path file = java.nio.file.Files.createTempDirectory("latency").resolve("out.json");
        BenchLatencyCommand.Args args = new BenchLatencyCommand.Args(server.baseUrl(), "/v1/completions",
                "served-model", new java.util.LinkedHashMap<>(), false, 2, 1, 1, 1, 0, 1,
                file.toString());
        BenchLatencyCommand.bench(args, new StringBuilder());
        assertTrue(java.nio.file.Files.exists(file));
        String json = java.nio.file.Files.readString(file);
        assertTrue(json, json.contains("\"avg_latency\""));
        java.nio.file.Files.deleteIfExists(file);
        java.nio.file.Files.deleteIfExists(file.getParent());
    }
}
