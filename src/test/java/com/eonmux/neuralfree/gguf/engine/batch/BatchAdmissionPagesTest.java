package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Page-aware admission on a two-slot engine over 64 cells in four pages of 16 (vLLM's {@code allocate_slots}
 * count): a candidate is admitted only while every pool holds the pages its prompt takes beside the admitted
 * prompts still waiting for prefill, whole shared pages free and a shared partial last page costing its
 * copy-on-write copy. A donor and a fork that cannot hold their pages together, or two prompts whose pages
 * outnumber the pool, run one after the other without a preemption or a requeue; pairs that fit still run together.
 * The window pages that context checkpoints pin in a sliding-window pool yield to a candidate, as they do to a
 * running sequence's step.
 */
public class BatchAdmissionPagesTest {

    private static EngineConfig twoSlots() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8).kvPaged(true).parallel(2)
                .typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    private static GenerationRequest request(int n) {
        return GenerationRequest.builder().maxNewTokens(n).ignoreEos(true)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int seed, int length) {
        int[] p = new int[length];
        for (int i = 0; i < length; i++) {
            p[i] = 4 + (seed * 17 + i * 13) % 250;
        }
        return p;
    }

    /**
     * Pumps until every handle finished, asserting none failed and none went back to the queue once running (a
     * prefill that ran out of pages); returns the most handles that ran at once.
     */
    private static int pumpWithoutRequeue(BatchEngine engine, BatchSequence... handles) {
        boolean[] ran = new boolean[handles.length];
        int together = 0;
        for (int step = 0; step < 2_000; step++) {
            boolean all = true;
            int running = 0;
            for (int i = 0; i < handles.length; i++) {
                BatchSequence h = handles[i];
                all &= h.finished();
                if (h.state() == BatchSequence.State.RUNNING) {
                    ran[i] = true;
                    running++;
                }
                assertFalse("handle " + i + " went back to the queue",
                        ran[i] && h.state() == BatchSequence.State.QUEUED);
            }
            together = Math.max(together, running);
            if (all) {
                for (BatchSequence h : handles) {
                    assertEquals("the request completed: " + h.error(), BatchSequence.State.DONE, h.state());
                }
                return together;
            }
            engine.pump();
        }
        throw new AssertionError("the engine stopped making progress");
    }

    @Test
    public void aDonorAndAForkThatCannotHoldTheirPagesTogetherRunOneAfterTheOther() throws IOException {
        // 62 rows fill four pages; the fork would share 61 of them, three whole pages and the partial fourth it
        // copies on its first write: five pages, one more than the pool
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt(1, 62);
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence donor = engine.submit(prompt, request(2), (TokenCallback) null);
            BatchSequence fork = engine.submit(prompt.clone(), request(2), (TokenCallback) null);
            engine.pump();
            assertEquals(BatchSequence.State.RUNNING, donor.state());
            assertEquals("the fork waits for the donor's pages", BatchSequence.State.QUEUED, fork.state());

            assertEquals(1, pumpWithoutRequeue(engine, donor, fork));
            assertEquals("no step was wasted on a preemption", 0, engine.stats().preemptions());
            assertEquals(0, engine.stats().recomputedTokens());
            assertArrayEquals(donor.result().tokens(), fork.result().tokens());
            assertEquals("the second adopts the first's cached rows", prompt.length - 1, fork.reusedPrefix());
            assertTrue(engine.idle());
        }
    }

    @Test
    public void bestOfCandidatesBeyondThePoolAllFinishWithoutAPreemption() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt(2, 62);
        for (int copies = 3; copies <= 4; copies++) {
            try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
                 BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
                BatchSequence[] handles = new BatchSequence[copies];
                for (int c = 0; c < copies; c++) {
                    handles[c] = engine.submit(prompt.clone(), request(2), (TokenCallback) null);
                }
                pumpWithoutRequeue(engine, handles);
                assertEquals(copies + " copies: no preemption", 0, engine.stats().preemptions());
                for (BatchSequence h : handles) {
                    assertArrayEquals(handles[0].result().tokens(), h.result().tokens());
                }
            }
        }
    }

    @Test
    public void promptsWhosePagesOutnumberThePoolAreNotAdmittedTogether() throws IOException {
        // 40 rows take three pages and 20 rows two: 60 cells fit the 64, their five pages do not fit the four
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence first = engine.submit(prompt(3, 40), request(2), (TokenCallback) null);
            BatchSequence second = engine.submit(prompt(4, 20), request(2), (TokenCallback) null);
            engine.pump();
            assertEquals(BatchSequence.State.QUEUED, second.state());
            assertEquals(1, pumpWithoutRequeue(engine, first, second));
            assertEquals(0, engine.stats().preemptions());
        }
    }

    @Test
    public void aDonorAndAForkWhosePagesFitRunTogether() throws IOException {
        // 12 rows take one page; the fork shares 11 of them on that page and copies it: two pages of four
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = prompt(5, 12);
        try (GGUFModel m = new GGUFModel(model.toString(), twoSlots());
             BatchEngine engine = BatchEngine.forModel(m, twoSlots())) {
            BatchSequence donor = engine.submit(prompt, request(4), (TokenCallback) null);
            BatchSequence fork = engine.submit(prompt.clone(), request(4), (TokenCallback) null);
            engine.pump();
            assertEquals(BatchSequence.State.RUNNING, donor.state());
            assertEquals(BatchSequence.State.RUNNING, fork.state());
            assertEquals(2, pumpWithoutRequeue(engine, donor, fork));
            assertEquals(0, engine.stats().preemptions());
            assertEquals("the fork copies the donor's prefix", prompt.length - 1, fork.reusedPrefix());
            assertArrayEquals(donor.result().tokens(), fork.result().tokens());
        }
    }

    /** gemma2 with an 8-row window over a two-slot paged memory of 4-row pages, checkpointing every chunk. */
    private static EngineConfig windowedCheckpoints(boolean reuse) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(128).ubatch(8).kvPaged(true).kvPageSize(4)
                .parallel(2).typeK(KvDType.F32).typeV(KvDType.F32).prefixReuse(reuse).checkpointMinStep(0)
                .build()).build();
    }

    private static Path windowedModel() throws IOException {
        Path path = Files.createTempFile("gemma2-", ".gguf");
        path.toFile().deleteOnExit();
        SyntheticModels.gemma2(SyntheticModels.Config.gemma2().toBuilder().slidingWindow(8).build()).writeTo(path);
        return path;
    }

    private static GenerationResult alone(Path model, int[] prompt, int n) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), windowedCheckpoints(false));
             BatchEngine engine = BatchEngine.forModel(m, windowedCheckpoints(false))) {
            BatchSequence h = engine.submit(prompt, request(n), (TokenCallback) null);
            engine.drain();
            assertEquals("the reference completed: " + h.error(), BatchSequence.State.DONE, h.state());
            return h.result();
        }
    }

    @Test
    public void checkpointWindowPagesYieldToACandidateBesideARunningSequence() throws IOException {
        // the 40-row prefill leaves a checkpoint at every 8-row chunk; their pinned window pages fill the 10-page
        // window pool beside the running sequence's own window, and only the checkpoints can give the candidate
        // the five window pages its prompt takes (the blocks they hang on are the running sequence's)
        Path model = windowedModel();
        int[] first = prompt(6, 40);
        int[] second = prompt(7, 24);
        GenerationResult refFirst = alone(model, first, 24);
        GenerationResult refSecond = alone(model, second, 4);
        try (GGUFModel m = new GGUFModel(model.toString(), windowedCheckpoints(true));
             BatchEngine engine = BatchEngine.forModel(m, windowedCheckpoints(true))) {
            BatchSequence a = engine.submit(first, request(24), (TokenCallback) null);
            for (int step = 0; step < 8; step++) {
                engine.pump();
            }
            assertEquals(BatchSequence.State.RUNNING, a.state());
            BatchSequence b = engine.submit(second, request(4), (TokenCallback) null);
            engine.pump();
            assertEquals("the candidate takes the checkpoints' window pages", BatchSequence.State.RUNNING,
                    b.state());
            assertEquals(2, pumpWithoutRequeue(engine, a, b));
            assertEquals(0, engine.stats().preemptions());
            assertArrayEquals(refFirst.tokens(), a.result().tokens());
            assertArrayEquals(refSecond.tokens(), b.result().tokens());
        }
    }
}
