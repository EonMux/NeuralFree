package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.engine.StopReason;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import io.vertx.core.MultiMap;
import io.vertx.core.http.HttpServerRequest;
import jakarta.ws.rs.core.Response;
import jakarta.ws.rs.core.StreamingOutput;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static com.eonmux.neuralfree.server.ServerTestSupport.at;
import static com.eonmux.neuralfree.server.ServerTestSupport.dataFrames;
import static com.eonmux.neuralfree.server.ServerTestSupport.parsed;
import static com.eonmux.neuralfree.server.ServerTestSupport.statusOf;
import static com.eonmux.neuralfree.server.ServerTestSupport.streamText;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The resumable stream over the real engine: a streamed generation whose request carries
 * {@code X-Conversation-Id} keeps running and recording when its client goes away mid-stream — the recorded
 * chunks finish with {@code length}, not {@code abort} — and the session replays through
 * {@code GET /v1/stream}, reports through the lookup, and stops on {@code DELETE /v1/stream}, which cancels the
 * generation. Without the header a failed write cancels as before.
 */
@Timeout(120)
class StreamResumeTest {

    private static EngineService service;
    private static StreamSessions sessions;
    private static StreamResource streams;

    @BeforeAll
    static void open() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        service = new EngineService(model.toString(), ServerTestSupport.pagedConfig(1024, 2));
        sessions = new StreamSessions(1024 * 1024, 300, 60);
        streams = new StreamResource(sessions);
    }

    @AfterAll
    static void close() {
        sessions.close();
        service.close();
    }

    /** An OpenAI resource whose requests name the {@code conversationId} (and optionally the response id). */
    private static OpenAiResource openAi(String conversationId, String requestId) {
        return new OpenAiResource(service, null, client(conversationId, requestId), sessions);
    }

    /** A streamed chat completion of {@code maxTokens} tokens: long enough to outlive a disconnect. */
    private static StreamingOutput producing(String conversationId, String requestId, int maxTokens) {
        Response response = openAi(conversationId, requestId).chatCompletions("{\"messages\":"
                + " [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": " + maxTokens
                + ", \"ignore_eos\": true, \"temperature\": 0, \"stream\": true}");
        return (StreamingOutput) response.getEntity();
    }

    /** Whether the generation batch of {@code responseId} finished (its live tracking left). */
    private static boolean finished(String responseId) {
        return !service.completions().tracks("chatcmpl-" + responseId);
    }

    /** Waits until the session recorded bytes or finished; whether it recorded anything in time. */
    private static boolean awaitedRecording(StreamSession session) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 20000;
        while (System.currentTimeMillis() < deadline && session.totalBytes() == 0 && !session.isDone()) {
            Thread.sleep(5);
        }
        return session.totalBytes() > 0;
    }

    @Test
    void aDisconnectMidStreamKeepsTheGenerationRunningAndRecording() throws Exception {
        StreamingOutput producing = producing("resume-1", null, 40);
        producing.write(new Disconnecting(1));
        StreamSession session = sessions.get("resume-1");
        assertTrue(session.isDone(), "the producer finished after the generation ended");
        assertTrue(session.totalBytes() > 0);

        String replay = streamText(streams.stream("resume-1", null));
        assertTrue(replay.endsWith("data: [DONE]\n\n"), "the recorded stream carries the terminal [DONE]");
        List<JsonView> frames = dataFrames(replay);
        assertFalse(frames.isEmpty());
        JsonView finish = frames.get(frames.size() - 1);
        assertEquals("length", at(finish, "choices", 0).str("finish_reason"),
                "the generation ran to its natural end, not an abort");

        List<?> found = (List<?>) parsed(streams.lookup(
                "{\"conversation_ids\": [\"resume-1\"]}").getEntity());
        JsonView entry = JsonView.of(found.get(0), "entry");
        assertEquals(Boolean.TRUE, entry.raw("is_done"));
        assertEquals(session.totalBytes(), entry.i64("total_bytes"));
        assertEquals(session.completedAt(), entry.i64("completed_at"));
    }

    @Test
    void aClientReattachesToTheRecordedStreamWhileItFlows() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            StreamingOutput producing = producing("resume-2", null, 60);
            StreamSession session = sessions.get("resume-2");
            Future<Object> disconnected = pool.submit(() -> {
                producing.write(new Disconnecting(1));
                return null;
            });
            assertTrue(awaitedRecording(session), "the producer records despite the dead client");
            String replay = streamText(streams.stream("resume-2", null));
            disconnected.get(30, TimeUnit.SECONDS);
            assertTrue(replay.endsWith("data: [DONE]\n\n"), "the reattached reader rides to the end");
            assertTrue(dataFrames(replay).size() > 2, "the replay holds the whole generation");
            assertTrue(session.isDone());
            assertTrue(streamText(streams.stream("resume-2", String.valueOf(session.droppedPrefix())))
                    .endsWith("data: [DONE]\n\n"), "a reattach from a later offset still reaches the end");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void theExplicitStopEndsThePullingAndCancelsTheGeneration() throws Exception {
        ExecutorService pool = Executors.newSingleThreadExecutor();
        try {
            long abortedBefore = service.stats().finished(StopReason.CANCELLED);
            StreamingOutput producing = producing("resume-3", "stop-1", 900);
            StreamSession session = sessions.get("resume-3");
            Future<Object> disconnected = pool.submit(() -> {
                producing.write(new Disconnecting(1));
                return null;
            });
            assertTrue(awaitedRecording(session));

            assertEquals(204, streams.stop("resume-3").getStatus());
            disconnected.get(15, TimeUnit.SECONDS);
            assertNull(sessions.get("resume-3"), "the stop evicted the session");
            assertEquals(404, statusOf(() -> streams.stream("resume-3", null)));
            assertTrue(session.isCancelled(), "the producer saw the stop");
            assertTrue(session.isDone());

            long deadline = System.currentTimeMillis() + 60000;
            while (System.currentTimeMillis() < deadline && (!finished("stop-1")
                    || service.stats().finished(StopReason.CANCELLED) == abortedBefore)) {
                Thread.sleep(20);
            }
            assertTrue(finished("stop-1"), "the cancelled generation left the engine");
            assertTrue(service.stats().finished(StopReason.CANCELLED) > abortedBefore,
                    "the stop cancelled the generation instead of letting it run to its token limit");
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void aStreamWithoutAConversationIdStillCancelsOnAFailedWrite() throws IOException {
        OpenAiResource plain = new OpenAiResource(service);
        Response response = plain.completions("{\"prompt\": \"hi\", \"max_tokens\": 20, \"ignore_eos\": true,"
                + " \"temperature\": 0, \"stream\": true}");
        StreamingOutput stream = (StreamingOutput) response.getEntity();
        IOException failure = assertThrows(IOException.class, () -> stream.write(new Disconnecting(0)),
                "the failed write surfaces, and the batch cancels behind it");
        assertTrue(failure.getMessage().contains("went away"));
    }

    /** An HTTP request whose headers name the conversation and the response id; nothing else is served. */
    private static HttpServerRequest client(String conversationId, String requestId) {
        MultiMap headers = MultiMap.caseInsensitiveMultiMap();
        headers.set("X-Conversation-Id", conversationId);
        if (requestId != null) {
            headers.set("X-Request-Id", requestId);
        }
        return (HttpServerRequest) Proxy.newProxyInstance(StreamResumeTest.class.getClassLoader(),
                new Class<?>[] {HttpServerRequest.class}, (proxy, method, args) -> {
                    if (method.getName().equals("headers")) {
                        return headers;
                    }
                    throw new UnsupportedOperationException(method.getName());
                });
    }

    /**
     * The client side of a stream that dies after {@code bytes} reached it: the rest of the writes fail, the
     * failures stay invisible to the producer through its mirror.
     */
    private static final class Disconnecting extends OutputStream {
        private long remaining;

        Disconnecting(int bytes) {
            remaining = bytes;
        }

        @Override
        public void write(int b) throws IOException {
            fail();
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            if (remaining <= 0) {
                fail();
            }
            remaining -= len;
        }

        @Override
        public void flush() throws IOException {
            if (remaining <= 0) {
                fail();
            }
        }

        private void fail() throws IOException {
            throw new IOException("the client went away");
        }
    }
}
