package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.utils.json.JsonView;
import jakarta.ws.rs.WebApplicationException;
import jakarta.ws.rs.core.Response;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.eonmux.neuralfree.server.ServerTestSupport.backend;
import static com.eonmux.neuralfree.server.ServerTestSupport.size;
import static com.eonmux.neuralfree.server.ServerTestSupport.view;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The per-replica queue limit ({@code neuralfree.server.max.queued.requests}, SGLang's {@code --max-queued-requests})
 * counts requests, not generations, and only those no free sequence slot takes: with a limit of 1 on a two-slot
 * engine, an idle server answers a chat of {@code n = 4}, a completions prompt array, a beam search wider than the
 * slots and a batch embedding; a saturated one (both slots decoding, one request waiting) answers a new request with
 * 429 {@code rate_limit_error} and {@code Retry-After} before submitting any of its choices, while an admitted
 * request's later generations (another choice, a tool loop's next round) are never refused; an offline run-batch
 * wave larger than the slots runs whole; concurrent requests never pass the limit together; and the queue admits
 * again once it drains.
 */
class MaxQueuedRequestsTest {

    private static final int[] PROMPT = {5, 9, 13, 17};

    private static Path model;
    private static EngineService limited;
    private static OpenAiResource openAi;
    private static PoolingResource pooling;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        // two sequence slots, and LAST pooling so the same engine serves embeddings
        MemoryConfig memory = MemoryConfig.builder().nCtx(512).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig config = EngineConfig.builder().backend(backend()).pooling(PoolingKind.LAST)
                .memory(memory).build();
        limited = new EngineService(model.toString(), config, ServerOptions.defaults().withMaxQueuedRequests(1));
        openAi = new OpenAiResource(limited);
        pooling = new PoolingResource(limited);
    }

    @AfterAll
    static void close() {
        limited.close();
    }

    private static GenerationRequest request(AtomicBoolean cancel, int maxTokens) {
        return GenerationRequest.builder().maxNewTokens(maxTokens).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).cancellation(cancel::get).build();
    }

    private static EngineService.Submission submit(AtomicBoolean cancel, int maxTokens, QueueAdmission admission)
            throws InterruptedException {
        return limited.submit(PROMPT, request(cancel, maxTokens), ReasoningFilter.contentTo((token, text) -> { }),
                null, null, null, null, admission);
    }

    /** Waits until {@code submission} holds a sequence slot, bounded for a CI run. */
    private static void awaitRunning(EngineService.Submission submission) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
        while (System.nanoTime() < deadline) {
            BatchSequence sequence = submission.sequence();
            if (sequence != null && sequence.state() == BatchSequence.State.RUNNING) {
                return;
            }
            Thread.sleep(5);
        }
        throw new AssertionError("the generation never ran: " + submission.sequence() + ", refused "
                + submission.refused());
    }

    /** The requests the only replica holds waiting. */
    private static int waiting() {
        return limited.scheduler().queue().waiting(limited.replicas().get(0));
    }

    private static void awaitAll(EngineService.Submission... submissions) throws InterruptedException {
        for (EngineService.Submission submission : submissions) {
            if (submission != null) {
                assertTrue(submission.await(60, TimeUnit.SECONDS), "a generation finished");
            }
        }
    }

    @Test
    void thePropertyParsesAndRefusesANegativeCount() {
        String saved = System.getProperty(ServerOptions.MAX_QUEUED_PROPERTY);
        try {
            System.clearProperty(ServerOptions.MAX_QUEUED_PROPERTY);
            assertEquals(0, ServerOptions.fromSystemProperties().maxQueuedRequests(), "no limit by default");
            System.setProperty(ServerOptions.MAX_QUEUED_PROPERTY, "12");
            assertEquals(12, ServerOptions.fromSystemProperties().maxQueuedRequests());
            System.setProperty(ServerOptions.MAX_QUEUED_PROPERTY, "-1");
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    ServerOptions::fromSystemProperties);
            assertTrue(e.getMessage().contains(ServerOptions.MAX_QUEUED_PROPERTY), e.getMessage());
        } finally {
            if (saved == null) {
                System.clearProperty(ServerOptions.MAX_QUEUED_PROPERTY);
            } else {
                System.setProperty(ServerOptions.MAX_QUEUED_PROPERTY, saved);
            }
        }
    }

    @Test
    void anIdleServerAnswersEveryChoiceOfAMultiChoiceRequest() {
        JsonView chat = view(openAi.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"temperature\": 0, \"max_tokens\": 3, \"n\": 4}").getEntity());
        assertEquals(4, size(chat, "choices"), "four choices on two slots under a limit of one");
        JsonView completions = view(openAi.completions("{\"prompt\": [\"a\", \"b\", \"c\"], \"temperature\": 0,"
                + " \"max_tokens\": 3, \"n\": 2}").getEntity());
        assertEquals(6, size(completions, "choices"), "three prompts of two choices each");
    }

    @Test
    void anIdleServerAnswersABeamSearchWiderThanItsSlots() {
        JsonView body = view(openAi.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                + " \"temperature\": 0, \"max_tokens\": 4, \"use_beam_search\": true, \"n\": 3}").getEntity());
        assertEquals(3, size(body, "choices"), "three beams over four steps");
    }

    @Test
    void anIdleServerAnswersABatchEmbedding() {
        JsonView body = view(pooling.embeddings("{\"input\": [[5, 6, 7], [8, 9], [10, 11, 12, 13]]}").getEntity());
        assertEquals(3, size(body, "data"));
    }

    @Test
    void aSaturatedServerAnswers429WithRetryAfterAndSubmitsNoChoice() throws Exception {
        AtomicBoolean cancel = new AtomicBoolean();
        EngineService.Submission first = null;
        EngineService.Submission second = null;
        EngineService.Submission queued = null;
        try {
            first = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(first);
            second = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(second);
            assertEquals(0, waiting(), "two running generations hold no request waiting");
            queued = submit(cancel, 4, null);
            assertEquals(1, waiting(), "the third generation waits for a slot");

            WebApplicationException refused = assertThrows(WebApplicationException.class,
                    () -> openAi.chatCompletions("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],"
                            + " \"max_tokens\": 3, \"n\": 2}"));
            Response response = refused.getResponse();
            assertEquals(429, response.getStatus());
            assertEquals(String.valueOf(EngineSaturatedException.RETRY_AFTER_SECONDS),
                    response.getHeaderString("Retry-After"));
            JsonView error = view(response.getEntity()).obj("error");
            assertEquals("rate_limit_error", error.str("type"));
            assertTrue(error.str("message").contains(ServerOptions.MAX_QUEUED_PROPERTY), error.str("message"));
            assertEquals(1, waiting(), "the refused request submitted none of its choices");

            EngineSaturatedException direct = assertThrows(EngineSaturatedException.class,
                    () -> submit(cancel, 4, null));
            assertTrue(direct.getMessage().contains("retry after"), direct.getMessage());
        } finally {
            cancel.set(true);
            awaitAll(first, second, queued);
        }
        EngineService.Submission admitted = submit(new AtomicBoolean(), 2, null);
        awaitAll(admitted);
        assertNull(admitted.refused(), "a drained queue admits again");
    }

    @Test
    void anAdmittedRequestsLaterGenerationsAreNeverRefused() throws Exception {
        AtomicBoolean cancel = new AtomicBoolean();
        QueueAdmission admitted = new QueueAdmission();
        EngineService.Submission own = null;
        EngineService.Submission other = null;
        EngineService.Submission queued = null;
        EngineService.Submission later = null;
        try {
            own = submit(cancel, GenerationRequest.UNLIMITED, admitted);
            awaitRunning(own);
            other = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(other);
            queued = submit(cancel, 4, null);
            assertThrows(EngineSaturatedException.class, () -> submit(cancel, 4, null),
                    "a new request finds the queue full");
            later = submit(cancel, 4, admitted);
            assertNotNull(later, "the admitted request's next generation is submitted");
            assertEquals(2, waiting(), "each waiting request counts once");
        } finally {
            cancel.set(true);
            awaitAll(own, other, queued, later);
        }
        assertNull(later.refused(), "the admitted request's later generation ran");
    }

    @Test
    void aLaterRoundOfAnAdmittedRequestIsNeverRefused() throws Exception {
        AtomicBoolean cancel = new AtomicBoolean();
        QueueAdmission rounds = new QueueAdmission();
        GenerationRequest step = request(new AtomicBoolean(), 2);
        GenerationBatch firstRound = GenerationBatch.start(limited,
                List.of(new GenerationBatch.Job(PROMPT, step, null, null, null)), null, null, true, rounds);
        firstRound.awaitAll();
        EngineService.Submission first = null;
        EngineService.Submission second = null;
        EngineService.Submission queued = null;
        GenerationBatch nextRound = null;
        try {
            first = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(first);
            second = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(second);
            queued = submit(cancel, 4, null);
            assertThrows(WebApplicationException.class, () -> GenerationBatch.start(limited,
                    List.of(new GenerationBatch.Job(PROMPT, step, null, null, null)), null),
                    "a new request finds the queue full");
            nextRound = GenerationBatch.start(limited,
                    List.of(new GenerationBatch.Job(PROMPT, step, null, null, null)), null, null, true, rounds);
        } finally {
            cancel.set(true);
            awaitAll(first, second, queued);
        }
        nextRound.awaitAll();
        assertNotNull(nextRound.result(0), "the next round of the admitted request ran");
    }

    @Test
    void aRunBatchWaveLargerThanTheSlotsRunsWhole(@TempDir Path dir) throws IOException {
        Path input = dir.resolve("in.jsonl");
        Path output = dir.resolve("out.jsonl");
        List<String> lines = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            lines.add("{\"custom_id\": \"r" + i + "\", \"method\": \"POST\", \"url\": \"/v1/completions\","
                    + " \"body\": {\"prompt\": \"hello " + i + "\", \"max_tokens\": 2, \"temperature\": 0}}");
        }
        Files.writeString(input, String.join("\n", lines) + "\n");
        BatchRunner.Summary summary = BatchRunner.run(limited, input, output, BatchRunner.DEFAULT_MAX_INFLIGHT);
        assertEquals(5, summary.succeeded(), "every record of the five-job wave answered on two slots");
        assertEquals(0, summary.failed());
    }

    @Test
    void concurrentRequestsNeverPassTheLimitTogether() throws Exception {
        AtomicBoolean cancel = new AtomicBoolean();
        EngineService.Submission first = null;
        EngineService.Submission second = null;
        List<EngineService.Submission> accepted = new ArrayList<>();
        int callers = 16;
        ExecutorService pool = Executors.newFixedThreadPool(callers);
        try {
            first = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(first);
            second = submit(cancel, GenerationRequest.UNLIMITED, null);
            awaitRunning(second);
            CountDownLatch go = new CountDownLatch(1);
            List<Future<EngineService.Submission>> calls = new ArrayList<>();
            for (int i = 0; i < callers; i++) {
                calls.add(pool.submit(() -> {
                    go.await();
                    try {
                        return submit(cancel, 4, null);
                    } catch (EngineSaturatedException e) {
                        return null;
                    }
                }));
            }
            go.countDown();
            for (Future<EngineService.Submission> call : calls) {
                EngineService.Submission submission = call.get(60, TimeUnit.SECONDS);
                if (submission != null) {
                    accepted.add(submission);
                }
            }
            assertEquals(1, accepted.size(), "one of " + callers + " concurrent requests takes the one place");
        } finally {
            cancel.set(true);
            pool.shutdown();
            awaitAll(first, second);
            awaitAll(accepted.toArray(new EngineService.Submission[0]));
        }
    }

    @Test
    void anEngineStateRefusalStays503() {
        WebApplicationException answer = ServerJson.refusal(new EngineUnavailableException("asleep"));
        assertEquals(503, answer.getResponse().getStatus());
        assertNull(answer.getResponse().getHeaderString("Retry-After"));
    }
}
