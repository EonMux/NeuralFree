package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link RequestLatencies.Choice}: the first token against the arrival, the gaps between tokens, the end of a
 * choice that produced output, and the engine queue time once both the end and the engine handle are known, in
 * either order.
 */
class RequestLatenciesTest {

    private static GGUFModel model;
    private static BatchEngine engine;
    private static BatchSequence admitted;
    private static BatchSequence neverAdmitted;

    /** The observations in arrival order, as {@code kind:nanos}. */
    private static final class Recorder implements RequestLatencies.Listener {
        final List<String> seen = new ArrayList<>();

        @Override
        public void firstToken(long nanos) {
            seen.add("first:" + nanos);
        }

        @Override
        public void interToken(long nanos) {
            seen.add("itl:" + nanos);
        }

        @Override
        public void finished(long nanos) {
            seen.add("e2e:" + nanos);
        }

        @Override
        public void queued(long nanos) {
            seen.add("queue:" + nanos);
        }
    }

    @BeforeAll
    static void open() throws IOException {
        Path file = SyntheticModels.llama().writeTempFile();
        EngineConfig paged = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
        model = new GGUFModel(file.toString(), paged);
        engine = BatchEngine.forModel(model, paged);
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(2).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
        admitted = engine.submit(new int[]{5, 6, 7}, request, (TokenCallback) null);
        engine.drain();
        neverAdmitted = engine.submit(new int[]{5, 6, 7}, request, (TokenCallback) null);
    }

    @AfterAll
    static void close() throws IOException {
        engine.close();
        model.close();
    }

    private static List<String> run(boolean handleFirst, BatchSequence handle, int tokens) {
        RequestLatencies latencies = new RequestLatencies();
        Recorder recorder = new Recorder();
        latencies.add(recorder);
        RequestLatencies.Choice choice = latencies.choice(1_000);
        if (handleFirst) {
            choice.submitted(() -> handle);
        }
        for (int i = 0; i < tokens; i++) {
            choice.token(1_500 + 100L * i * i);
        }
        choice.done(5_000);
        if (!handleFirst) {
            choice.submitted(() -> handle);
        }
        return recorder.seen;
    }

    @Test
    void theTokensAndTheEndReportAgainstTheArrival() {
        List<String> seen = run(true, null, 3);
        assertEquals(List.of("first:500", "itl:100", "itl:300", "e2e:4000"), seen);
    }

    @Test
    void theQueueTimeReportsOnceWhicheverOfHandleAndEndComesLast() {
        long queue = admitted.admittedAtNanos() - admitted.submittedAtNanos();
        assertTrue(queue >= 0);
        for (boolean handleFirst : new boolean[]{true, false}) {
            List<String> seen = run(handleFirst, admitted, 1);
            assertEquals(List.of("first:500", "e2e:4000", "queue:" + queue), seen, "handle first: " + handleFirst);
        }
    }

    @Test
    void aChoiceWithoutOutputOrAdmissionReportsNeitherEndNorQueue() {
        assertEquals(BatchSequence.NEVER_ADMITTED, neverAdmitted.admittedAtNanos(), "submitted, never pumped");
        assertEquals(List.of(), run(false, neverAdmitted, 0));
        assertEquals(List.of(), run(true, null, 0));
        assertEquals(List.of(), run(false, null, 0));
    }
}
