package com.eonmux.neuralfree.demo.tools.eval;

import org.junit.Test;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The chat client: the request body the reference builds, the bearer token, the retry policy for
 * transport errors, retryable statuses and empty replies, and the reply's usage and timings.
 */
public class EvalClientTest {

    @Test
    public void theRequestBodyCarriesThePromptAndTheSampling() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            EvalServer endpoint = new EvalServer(server.baseUrl(), 2, "one", "sk-test");
            EvalClient client = new EvalClient(endpoint, "my-model", 0, 0.0, null);
            EvalChat answer = client.generate("What is 2 + 3?",
                    new EvalSampling(0.2, 40, 0.9, 0.05, "high", 128), null);

            assertEquals("\\boxed{42}", answer.content());
            assertEquals("stop", answer.finishReason());
            assertEquals(Long.valueOf(7), answer.tokens());
            assertEquals(12.5, answer.tpsGen(), 0.0001);
            assertEquals(560.0, answer.tGenMs(), 0.0001);

            Map<String, Object> body = server.bodies().get(0);
            assertEquals("my-model", body.get("model"));
            assertEquals(128, ((Number) body.get("n_predict")).intValue());
            assertEquals(128, ((Number) body.get("max_completion_tokens")).intValue());
            assertEquals(0.2, ((Number) body.get("temperature")).doubleValue(), 0.0001);
            assertEquals(40, ((Number) body.get("top_k")).intValue());
            assertEquals(0.9, ((Number) body.get("top_p")).doubleValue(), 0.0001);
            assertEquals(0.05, ((Number) body.get("min_p")).doubleValue(), 0.0001);
            assertEquals("high", body.get("reasoning_effort"));
            List<?> messages = (List<?>) body.get("messages");
            assertEquals(1, messages.size());
            assertEquals("user", ((Map<?, ?>) messages.get(0)).get("role"));
            assertEquals("What is 2 + 3?", ((Map<?, ?>) messages.get(0)).get("content"));
            assertEquals(List.of("Bearer sk-test"), server.authorizations());
        }
    }

    @Test
    public void theModelDefaultsToLlamaAndOmittedSamplingIsNotSent() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            new EvalClient(endpoint, null, 0, 0.0, null)
                    .generate("p", new EvalSampling(null, null, null, null, null, -1), null);
            Map<String, Object> body = server.bodies().get(0);
            assertEquals("llama", body.get("model"));
            assertEquals(-1, ((Number) body.get("n_predict")).intValue());
            assertFalse(body.containsKey("max_completion_tokens"));
            assertFalse(body.containsKey("temperature"));
            assertFalse(body.containsKey("top_k"));
            assertTrue(server.authorizations().isEmpty());
        }
    }

    @Test
    public void transientFailuresAreResent() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.failuresBeforeSuccess = 2;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            EvalClient client = new EvalClient(endpoint, "m", 2, 0.0, null);
            EvalChat answer = client.generate("p", new EvalSampling(null, null, null, null, null, -1),
                    null);
            assertEquals("\\boxed{42}", answer.content());
            assertEquals(3, server.chatCalls());
        }
    }

    @Test
    public void anEmptyReplyIsResent() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.emptyBeforeSuccess = 1;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            EvalClient client = new EvalClient(endpoint, "m", 1, 0.0, null);
            assertEquals("\\boxed{42}", client.generate("p",
                    new EvalSampling(null, null, null, null, null, -1), null).content());
            assertEquals(2, server.chatCalls());
        }
    }

    @Test
    public void aClientErrorIsNotResent() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.status = 400;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            EvalClient client = new EvalClient(endpoint, "m", 5, 0.0, null);
            try {
                client.generate("p", new EvalSampling(null, null, null, null, null, -1), null);
                fail("a 400 must fail the request");
            } catch (EvalServerException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("HTTP 400"));
                assertTrue(expected.getMessage(), expected.getMessage().contains("after 1 attempt"));
            }
            assertEquals(1, server.chatCalls());
        }
    }

    @Test
    public void aRateLimitIsRetriedUntilTheBudgetRunsOut() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.status = 429;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            EvalClient client = new EvalClient(endpoint, "m", 2, 0.0, null);
            try {
                client.generate("p", new EvalSampling(null, null, null, null, null, -1), null);
                fail("a 429 must fail after its retries");
            } catch (EvalServerException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("after 3 attempts"));
            }
            assertEquals(3, server.chatCalls());
        }
    }

    @Test
    public void aServerErrorExhaustsTheBudget() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.failuresBeforeSuccess = 10;
            EvalServer endpoint = new EvalServer(server.baseUrl(), 1, "one", null);
            EvalClient client = new EvalClient(endpoint, "m", 1, 0.0, null);
            try {
                client.generate("p", new EvalSampling(null, null, null, null, null, -1), null);
                fail("a 500 must fail after its retries");
            } catch (EvalServerException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("after 2 attempts"));
            }
            assertEquals(2, server.chatCalls());
        }
    }

    @Test
    public void aReplyWithoutUsageReportsZeroTokensAndNoTimings() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.sendUsage = false;
            EvalClient client = new EvalClient(
                    new EvalServer(server.baseUrl(), 1, "one", null), "m", 0, 0.0, null);
            EvalChat answer = client.generate("p",
                    new EvalSampling(null, null, null, null, null, -1), null);
            assertEquals(Long.valueOf(0L), answer.tokens());
            assertNull(answer.tpsGen());
            assertNull(answer.tGenMs());
        }
    }

    @Test
    public void aFinishReasonOtherThanStopIsReported() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            server.finishReason = "length";
            EvalClient client = new EvalClient(
                    new EvalServer(server.baseUrl(), 1, "one", null), "m", 0, 0.0, null);
            assertEquals("length", client.generate("p",
                    new EvalSampling(null, null, null, null, null, 8), null).finishReason());
        }
    }

    @Test
    public void theModelListingIsRead() throws Exception {
        try (FakeEvalServer server = new FakeEvalServer()) {
            EvalClient client = new EvalClient(
                    new EvalServer(server.baseUrl(), 1, "one", "k"), "m", 0, 0.0, null);
            assertEquals(List.of("test-model"), client.models());
        }
    }

    @Test
    public void theOpencodeSessionHeaderOnlyGoesToAnOpencodeHost() {
        EvalServer opencode = new EvalServer("http://opencode.example:8080", 1, "oc", null);
        assertEquals("abc", opencode.headers("abc").get("x-opencode-session"));
        EvalServer other = new EvalServer("http://localhost:8080", 1, "local", null);
        assertNull(other.headers("abc").get("x-opencode-session"));
        assertEquals("Bearer k", new EvalServer("http://localhost:8080", 1, "local", "k")
                .headers(null).get("Authorization"));
    }

    @Test
    public void anUnreachableServerFailsTheListing() {
        EvalClient client = new EvalClient(new EvalServer("http://127.0.0.1:1", 1, "down", null),
                "m", 0, 0.0, null);
        try {
            client.models();
            fail("an unreachable server must fail the listing");
        } catch (EvalServerException expected) {
            assertTrue(expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }
}
