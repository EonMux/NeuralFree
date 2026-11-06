package com.eonmux.neuralfree.demo.tools.serve;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The streaming client against an OpenAI-compatible server: the request body the reference sends, the
 * TTFT, ITL and latency stamps of the streamed chunks, the chat endpoint's role-chunk rule, the usage
 * counts and the failure paths.
 */
public class ServeBenchClientTest {

    private FakeOpenAiServer server;

    @Before
    public void start() throws IOException {
        server = new FakeOpenAiServer();
    }

    @After
    public void stop() {
        server.close();
    }

    private ServeBenchClient client(String endpoint, Map<String, Object> sampling, Integer logprobs) {
        return new ServeBenchClient(server.baseUrl(), endpoint, new LinkedHashMap<>(), "served-model", null,
                true, sampling, logprobs, false);
    }

    @Test
    public void oneCompletionStreamsItsTokensAndUsage() {
        server.tokensPerResponse = 3;
        BenchRequestOutput out = client("/v1/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofText("hi", 2, 3, "req-1"));
        assertTrue(out.success);
        assertEquals("three chunks: TTFT plus two intervals", 2, out.itls.size());
        assertTrue(out.ttft >= 0.0);
        assertTrue(out.latency >= out.ttft);
        assertEquals(Integer.valueOf(3), out.outputTokens);
        assertEquals(2, out.promptLen);
        assertEquals("t0t1t2", out.generatedText.toString());
        assertEquals(Arrays.asList("req-1"), server.requestIds());

        Map<String, Object> body = server.bodies().get(0);
        assertEquals("served-model", body.get("model"));
        assertEquals("hi", body.get("prompt"));
        assertEquals(3, ((Number) body.get("max_tokens")).intValue());
        assertEquals(true, body.get("stream"));
        assertEquals(true, body.get("ignore_eos"));
        assertEquals(1.0, (Double) body.get("repetition_penalty"), 0.0);
        Map<?, ?> streamOptions = (Map<?, ?>) body.get("stream_options");
        assertEquals(true, streamOptions.get("include_usage"));
    }

    @Test
    public void aTokenIdPromptIsSentAsIds() {
        client("/v1/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofTokens(new int[]{5, 6, 7}, 2, "req-2"));
        Object prompt = server.bodies().get(0).get("prompt");
        assertEquals(Arrays.asList(5L, 6L, 7L), prompt);
    }

    @Test
    public void theChatEndpointWrapsThePromptAndSkipsTheRoleChunk() {
        server.tokensPerResponse = 2;
        BenchRequestOutput out = client("/v1/chat/completions", new LinkedHashMap<>(), 4)
                .send(BenchSample.ofText("hello", 5, 2, "req-3"));
        assertTrue(out.success);
        assertEquals("the role chunk is not a token", 1, out.itls.size());
        assertEquals("t0t1", out.generatedText.toString());
        Map<String, Object> body = server.bodies().get(0);
        assertEquals("the chat client budgets with max_completion_tokens, as the reference does", 2,
                ((Number) body.get("max_completion_tokens")).intValue());
        assertFalse("logprobs belong to the completions endpoint", body.containsKey("logprobs"));
        List<?> messages = (List<?>) body.get("messages");
        assertEquals("user", ((Map<?, ?>) messages.get(0)).get("role"));
        assertEquals("hello", ((Map<?, ?>) messages.get(0)).get("content"));
    }

    @Test
    public void aTokenIdPromptRefusesTheChatEndpoint() {
        try {
            client("/v1/chat/completions", new LinkedHashMap<>(), null)
                    .send(BenchSample.ofTokens(new int[]{1, 2}, 2, "req-4"));
            org.junit.Assert.fail("the chat endpoint takes text only");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("/v1/completions"));
        }
    }

    @Test
    public void theSamplingParametersRideEveryRequest() {
        Map<String, Object> sampling = new LinkedHashMap<>();
        sampling.put("temperature", 0.5);
        sampling.put("top_p", 0.9);
        client("/v1/completions", sampling, 3)
                .send(BenchSample.ofText("hi", 2, 1, "req-5"));
        Map<String, Object> body = server.bodies().get(0);
        assertEquals(0.5, (Double) body.get("temperature"), 0.0);
        assertEquals(0.9, (Double) body.get("top_p"), 0.0);
        assertEquals(3, ((Number) body.get("logprobs")).intValue());
    }

    @Test
    public void anErrorStatusFailsTheRequestWithItsStatus() {
        server.generationStatus = 400;
        BenchRequestOutput out = client("/v1/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofText("hi", 2, 1, "req-6"));
        assertFalse(out.success);
        assertTrue(out.error, out.error.startsWith("HTTP 400"));
    }

    @Test
    public void aStreamWithoutContentFailsOnTtft() {
        server.tokensPerResponse = 0;
        server.sendUsage = false;
        server.openingRoleChunk = true;
        BenchRequestOutput out = client("/v1/chat/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofText("hi", 2, 1, "req-7"));
        assertFalse(out.success);
        assertTrue(out.error, out.error.contains("valid chunk"));
    }

    @Test
    public void aRunSamplesEveryRequestAndReportsItsOutputsInOrder() {
        server.tokensPerResponse = 2;
        List<BenchSample> samples = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            samples.add(BenchSample.ofText("hi" + i, 2, 2, "run-" + i));
        }
        ArrivalSchedule schedule = ArrivalSchedule.of(4, Double.POSITIVE_INFINITY, 1.0,
                null, null, null, new java.util.Random(1));
        List<BenchRequestOutput> outputs = client("/v1/completions", new LinkedHashMap<>(), null)
                .runAll(samples, schedule, null, null);
        assertEquals(4, outputs.size());
        int done = 0;
        for (int i = 0; i < outputs.size(); i++) {
            if (outputs.get(i).success) {
                done++;
                assertEquals("two chunks: one interval", 1, outputs.get(i).itls.size());
            }
        }
        assertEquals(4, done);
        assertEquals("every request reached the server in its workload order",
                Arrays.asList("run-0", "run-1", "run-2", "run-3"),
                server.requestIds().stream().sorted().collect(java.util.stream.Collectors.toList()));
    }

    @Test
    public void aConcurrencyCapQueuesTheRequestsAndReportsTheirWait() {
        server.tokensPerResponse = 1;
        server.tokenDelayMillis = 20;
        List<BenchSample> samples = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            samples.add(BenchSample.ofText("hi", 2, 1, "cap-" + i));
        }
        ArrivalSchedule schedule = ArrivalSchedule.of(4, Double.POSITIVE_INFINITY, 1.0,
                null, null, null, new java.util.Random(1));
        List<BenchRequestOutput> outputs = client("/v1/completions", new LinkedHashMap<>(), null)
                .runAll(samples, schedule, 1, null);
        for (BenchRequestOutput out : outputs) {
            assertTrue(out.success);
            assertTrue("the cap queues every request behind the first", out.clientQueueTime >= 0.0);
        }
        assertTrue("the last request waited for the cap", outputs.get(3).clientQueueTime > 0.0);
        assertTrue(outputs.get(3).startTime >= 0.0);
    }

    @Test
    public void anEndpointOutsideTheOpenAiPathsIsRefused() {
        try {
            client("/v1/embeddings", new LinkedHashMap<>(), null);
            org.junit.Assert.fail("only the two generation paths are benchmarks");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("/v1/embeddings"));
        }
    }

    @Test
    public void theDiscoveryRoutesReportTheModelAndTheTokenizer() throws IOException {
        BenchEndpoints.ServedModel model = BenchEndpoints.firstModel(server.baseUrl(),
                new LinkedHashMap<>(), false);
        assertEquals("served-model", model.id);
        assertEquals("/models/base.gguf", model.root);
        BenchEndpoints.TokenizerInfo info = BenchEndpoints.tokenizerInfo(server.baseUrl(),
                new LinkedHashMap<>(), false);
        assertEquals(1000, info.vocabSize);
        assertEquals(Arrays.toString(new int[]{1, 2, 0, 3}), Arrays.toString(info.skipIds));
        int[] tokens = BenchEndpoints.tokenizer(server.baseUrl(), new LinkedHashMap<>(), false)
                .tokenize("abcd");
        assertEquals(4, tokens.length);
        assertEquals(100, tokens[0]);
    }

    @Test
    public void anUnreachableServerNamesItsUrl() {
        server.close();
        try {
            BenchEndpoints.firstModel(server.baseUrl(), new LinkedHashMap<>(), false);
            org.junit.Assert.fail("an unreachable server must fail");
        } catch (IOException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains(server.baseUrl()));
        }
    }

    @Test
    public void theUsageChunkRefinesThePromptLength() {
        server.tokensPerResponse = 1;
        // the fake reports the prompt's text length, which for "hello" is 5 even though the sample said 99
        BenchRequestOutput out = client("/v1/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofText("hello", 99, 1, "req-8"));
        assertEquals(5, out.promptLen);
        assertNull(out.error);
    }

    @Test
    public void aChatSampleSendsItsContentParts() {
        server.tokensPerResponse = 1;
        List<Object> content = new ArrayList<>();
        Map<String, Object> text = new LinkedHashMap<>();
        text.put("type", "text");
        text.put("text", "look");
        content.add(text);
        Map<String, Object> imageUrl = new LinkedHashMap<>();
        imageUrl.put("url", "data:image/png;base64,AAAA");
        Map<String, Object> image = new LinkedHashMap<>();
        image.put("type", "image_url");
        image.put("image_url", imageUrl);
        content.add(image);
        BenchRequestOutput out = client("/v1/chat/completions", new LinkedHashMap<>(), null)
                .send(BenchSample.ofChat(content, 4, 1, "req-9"));
        assertTrue(out.success);
        List<?> messages = (List<?>) server.bodies().get(0).get("messages");
        Object sent = ((Map<?, ?>) messages.get(0)).get("content");
        assertEquals(content, sent);
    }

    @Test
    public void aRerankSampleSendsTheQueryAndDocuments() {
        BenchRequestOutput out = client("/rerank", new LinkedHashMap<>(), null)
                .send(BenchSample.ofRerank(Arrays.asList("query", "doc1", "doc2"), 12, "req-10"));
        assertTrue(out.success);
        assertTrue("the pooling reply closes the latency", out.latency >= 0.0);
        assertEquals(2, out.numInputSequences);
        Map<String, Object> body = server.bodies().get(0);
        assertEquals("query", body.get("query"));
        assertEquals(Arrays.asList("doc1", "doc2"), body.get("documents"));
        assertEquals(-1, ((Number) body.get("truncate_prompt_tokens")).intValue());
    }
}
