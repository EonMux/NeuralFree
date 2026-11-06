package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The slot status ({@code BatchEngine.slotStatuses}, llama.cpp's {@code GET /slots} data): while a sequence runs
 * the status carries its request's counters, sampler parameters and tokens, once it leaves the id reports the
 * task it last served ({@code task_prev}), a free id the bare idle status, and a slot id outside the slots
 * refuses by name. Token arrays copy on the way out.
 */
public class BatchSlotStatusTest {

    private static final int N_NEW = 5;

    private static EngineConfig config() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).parallel(2).kvPageSize(4)
                        .typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(true).build())
                .build();
    }

    /** A temperature that reaches the status's params, with top-k 1 keeping the run deterministic. */
    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().temperature(0.5f).topK(1).build())
                .build();
    }

    /** Fourteen prompt tokens: three whole blocks of four, then a partial block. */
    private static int[] prompt() {
        int[] t = new int[14];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (i * 37) % 250;
        }
        return t;
    }

    @Test
    public void aRunningSequenceReportsItsTask() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            engine.submit(prompt(), request(), (TokenCallback) null);
            engine.pump();
            SlotStatus live = engine.slotStatus(0);
            assertEquals(0, live.id());
            assertEquals(64, live.nCtx());
            assertTrue("the submitted sequence runs", live.processing());
            assertFalse("the engine drafts nothing", live.speculative());
            assertEquals(0, live.idTask());
            assertEquals(14, live.nPromptTokens());
            assertTrue("the prefill processed a prefix of the prompt",
                    live.nPromptTokensProcessed() >= 0 && live.nPromptTokensProcessed() <= 14);
            if (live.nPromptTokensProcessed() < 14) {
                assertFalse("a prefill arms no decode token", live.hasNextToken());
            }
            assertEquals(0, live.nDecoded());
            assertEquals(N_NEW, live.nRemain());
            assertEquals(N_NEW, live.maxNewTokens());
            assertEquals(0.5f, live.params().temperature(), 0f);
            assertEquals(1, live.params().topK());
            assertArrayEquals(prompt(), live.promptTokens());
            assertEquals(0, live.generatedTokens().length);

            List<SlotStatus> all = engine.slotStatuses();
            assertEquals(2, all.size());
            assertEquals(0, all.get(0).id());
            assertTrue(all.get(0).processing());
            assertEquals("the free slot beside the running one reports idle", 1, all.get(1).id());
            assertFalse(all.get(1).processing());
            assertEquals(-1, all.get(1).idTask());
            assertEquals(64, all.get(1).nCtx());
            assertEquals(0, all.get(1).nDecoded());
            assertNull(all.get(1).params());
            assertNull(all.get(1).promptTokens());

            engine.drain();
        }
    }

    @Test
    public void aFinishedRequestStaysAsItsSlotsLastTask() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            int submitId = engine.submit(prompt, request(), (TokenCallback) null).submitId();
            engine.drain();
            SlotStatus last = engine.slotStatus(0);
            assertFalse(last.processing());
            assertFalse("the finished task feeds nothing more", last.hasNextToken());
            assertEquals(submitId, last.idTask());
            assertEquals(14, last.nPromptTokens());
            assertEquals("the whole prompt processed by the end", 14, last.nPromptTokensProcessed());
            assertEquals(0, last.nPromptTokensCache());
            assertEquals("every decode but the last sampled token fed", N_NEW - 1, last.nDecoded());
            assertEquals("the budget holds the unfed token", 1, last.nRemain());
            assertArrayEquals(prompt, last.promptTokens());
            assertEquals(N_NEW - 1, last.generatedTokens().length);
        }
    }

    @Test
    public void aSlotOutsideTheSlotsRefusesByName() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            IllegalArgumentException outside = assertThrows(IllegalArgumentException.class,
                    () -> engine.slotStatus(2));
            assertTrue(outside.getMessage(), outside.getMessage().contains("slot 2"));
            assertThrows(IllegalArgumentException.class, () -> engine.slotStatus(-1));
        }
    }

    @Test
    public void theTokenArraysCopyOnTheWayOut() throws Exception {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config());
             BatchEngine engine = BatchEngine.forModel(m, config())) {
            engine.submit(prompt(), request(), (TokenCallback) null);
            engine.drain();
            SlotStatus status = engine.slotStatus(0);
            int[] prompt = status.promptTokens();
            prompt[0] = -1;
            assertArrayEquals("the status keeps its own prompt tokens", prompt(), status.promptTokens());
            int[] generated = status.generatedTokens();
            java.util.Arrays.fill(generated, -1);
            assertTrue(status.generatedTokens().length > 0);
            assertFalse(java.util.Arrays.equals(generated, status.generatedTokens()));
        }
    }
}
