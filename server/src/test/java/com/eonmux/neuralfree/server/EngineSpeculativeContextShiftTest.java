package com.eonmux.neuralfree.server;

import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The server's speculative decoding across a context shift: a completion that fills its slot shifts as the plain
 * engine does, the separate draft model brings its own paged cache to the moved text tokens, and the completion
 * keeps the plain engine's greedy tokens while it drafts.
 */
class EngineSpeculativeContextShiftTest {

    private static EngineService plain;
    private static EngineService drafted;
    private static Path model;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        plain = new EngineService(model.toString(), config().build());
        drafted = new EngineService(model.toString(), config().draftModelPath(model.toString())
                .draftMax(3).build());
    }

    @AfterAll
    static void close() {
        if (plain != null) {
            plain.close();
        }
        if (drafted != null) {
            drafted.close();
        }
    }

    /** A paged serving config whose slot context the prompt and the generated tail fill more than once. */
    private static EngineConfig.Builder config() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(24).ubatch(8).parallel(2).kvPageSize(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).contextShift(true).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory);
    }

    private static int[] prompt() {
        return new int[]{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(30).nKeep(4).nDiscard(6).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static BatchSequence run(EngineService service, int[] prompt) throws InterruptedException {
        EngineService.Submission submission = service.submit(prompt, request(),
                (TokenCallback) (token, text) -> { });
        assertTrue(submission.await(30, TimeUnit.SECONDS), "the generation outlived its wait");
        assertNull(submission.refused(), () -> "the engine refused the request: " + submission.refused());
        BatchSequence sequence = submission.sequence();
        assertNotNull(sequence, "the registered request has no sequence");
        return sequence;
    }

    @Test
    void aShiftedGenerationKeepsDraftingAndKeepsTheTargetsTokens() throws InterruptedException {
        BatchSequence expected = run(plain, prompt());
        assertEquals(BatchSequence.State.DONE, expected.state(), String.valueOf(expected.error()));

        long shiftsBefore = drafted.stats().contextShifts();
        long draftedBefore = drafted.stats().specDrafted();
        long acceptedBefore = drafted.stats().specAccepted();
        BatchSequence shifted = run(drafted, prompt());
        assertEquals(BatchSequence.State.DONE, shifted.state(), String.valueOf(shifted.error()));

        assertArrayEquals(expected.result().tokens(), shifted.result().tokens(),
                "a shifted generation with a drafter keeps the plain engine's greedy tokens");
        assertTrue(drafted.stats().contextShifts() > shiftsBefore, "the completion shifted its context");
        assertTrue(drafted.stats().specDrafted() - draftedBefore > 0, "the shifted generation drafted");
        assertTrue(drafted.stats().specAccepted() - acceptedBefore > 0, "the shifted generation accepted drafts");
    }
}
