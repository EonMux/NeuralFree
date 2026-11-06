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
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Two serving generations that share a prompt prefix: the first request primes the engine's prompt cache, the
 * second reuses the cached prefix and drafts with a separate draft model whose own cache is brought to the shared
 * tokens. The second completion keeps the plain engine's greedy tokens and accepts drafts, as a fresh request over
 * the same prompt would.
 */
class EngineSpeculativePrefixReuseTest {

    private static EngineService plain;
    private static EngineService drafted;
    private static Path model;

    @BeforeAll
    static void open() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
        plain = new EngineService(model.toString(), paged().build());
        drafted = new EngineService(model.toString(), paged().draftModelPath(model.toString())
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

    /** A paged serving config (per-slot context nCtx / parallel) that holds the lookup prompt and its prefix. */
    private static EngineConfig.Builder paged() {
        MemoryConfig mc = MemoryConfig.builder().nCtx(768).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build();
        return EngineConfig.builder().memory(mc);
    }

    /**
     * The prompt whose stored {@code (4, 5, v)} triples make the lookup drafter propose from the first decode step
     * on; its greedy continuation is a run of token 86 (pinned once against the synthetic model's fixed weights).
     */
    private static int[] lookupPrompt() {
        int[] prompt = new int[3 * 100 + 2];
        int w = 0;
        for (int v = 6; v <= 105; v++) {
            prompt[w++] = 4;
            prompt[w++] = 5;
            prompt[w++] = v;
        }
        prompt[w++] = 4;
        prompt[w++] = 5;
        return prompt;
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder()
                .maxNewTokens(8)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
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
    void theSecondGenerationReusesThePrefixAndKeepsDrafting() throws InterruptedException {
        int[] full = lookupPrompt();
        int[] prefix = Arrays.copyOf(full, 3 * 80 + 2);

        BatchSequence expected = run(plain, full);
        assertEquals(BatchSequence.State.DONE, expected.state());

        BatchSequence first = run(drafted, prefix);
        assertEquals(BatchSequence.State.DONE, first.state(), String.valueOf(first.error()));
        long cachedBefore = drafted.stats().promptTokensCached();
        long draftedBefore = drafted.stats().specDrafted();
        long acceptedBefore = drafted.stats().specAccepted();

        BatchSequence second = run(drafted, full);
        assertEquals(BatchSequence.State.DONE, second.state(), String.valueOf(second.error()));
        assertArrayEquals(expected.result().tokens(), second.result().tokens(),
                "the reused prefix keeps the plain engine's greedy tokens");
        assertTrue(drafted.stats().promptTokensCached() > cachedBefore,
                "the second request reused its cached prompt prefix");
        assertTrue(drafted.stats().specDrafted() - draftedBefore > 0,
                "the second generation drafted over the reused prefix");
        assertTrue(drafted.stats().specAccepted() - acceptedBefore > 0,
                "the second generation accepted drafts");
    }
}
