package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.StopReason;import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Media-aware prompts in the {@link BatchEngine}: a prompt of text, image and text chunks prefills chunk by chunk
 * over the paged memory (the image rows share the chunk's base position, the decode continues at base + nPos), a
 * text-only {@link MediaPrompt} generates exactly the tokens of the same int[] submission, and an oversize media
 * prompt refuses at submit. The synthetic qwen2vl (M-RoPE) runs paged with parallel 2.
 */
public class BatchEngineMediaPromptTest {

    private static final int N_NEW = 6;
    private static final int N_EMBD = 64;
    private static final int TEXT = 4;

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static float[][] imageRows(int n, float v) {
        float[][] r = new float[n][N_EMBD];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < N_EMBD; d++) {
                r[i][d] = v * (1 + 0.02f * ((i + d) % 5));
            }
        }
        return r;
    }

    @Test
    public void aTextOnlyMediaPromptMatchesThePlainTokenSubmission() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = config();
        int[] prompt = tokens(1, 10);
        GenerationResult plain;
        GenerationResult viaMediaPrompt;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence a = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            plain = a.result();
            assertTrue(engine.idle());
            BatchSequence b = engine.submit(MediaPrompt.ofText(prompt), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, b.state());
            viaMediaPrompt = b.result();
        }
        assertArrayEquals(plain.tokens(), viaMediaPrompt.tokens());
        assertEquals(plain.stopReason(), viaMediaPrompt.stopReason());
    }

    @Test
    public void aMediaPromptPrefillsChunkByChunkAndGenerates() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            MediaPrompt prompt = MediaPrompt.builder()
                    .addText(tokens(2, 5))
                    .addMedia(MediaRopePositions.image2d(5, 4, 4), imageRows(16, 0.3f), true)
                    .addText(tokens(3, 3))
                    .build();
            assertEquals(24, prompt.nTokens());
            assertEquals(12, prompt.nPositions());
            BatchSequence b = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            // admits and prefills the head: the prompt's chunks ran (the 16-row image chunk exceeds the default
            // 8-row encoder budget alone, so it takes the untouched step budget), the decode waits
            engine.pump();
            assertEquals(BatchSequence.State.RUNNING, b.state());
            // the stored positions run 0..11 (the chunk took 4 temporal positions for its 16 rows) while the
            // sequence is still running
            HeapModelMemory mem = (HeapModelMemory) m.forwardBackend().memory();
            assertEquals(0, mem.sequences().posMin(b.seq()));
            assertEquals(12 - 1, mem.sequences().posMax(b.seq()));
            engine.drain();
            assertEquals(BatchSequence.State.DONE, b.state());
            assertEquals(StopReason.MAX_TOKENS, b.result().stopReason());
            assertEquals(N_NEW, b.result().tokenCount());
            assertEquals(0, b.reusedPrefix());
            assertTrue("the sequence generated non-empty content", b.result().tokenCount() > 0);
        }
    }

    @Test
    public void anOversizeMediaPromptRefusesAtSubmit() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            MediaPrompt prompt = MediaPrompt.builder()
                    .addMedia(MediaRopePositions.image2d(0, 8, 10), imageRows(80, 0.5f), false)
                    .addText(tokens(1, 4))
                    .build();
            assertEquals(84, prompt.nTokens());
            com.eonmux.neuralfree.gguf.engine.ContextFullException e = assertThrows(
                    com.eonmux.neuralfree.gguf.engine.ContextFullException.class,
                    () -> engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null));
            assertTrue(e.getMessage(), e.getMessage().contains("84"));
        }
    }

    @Test
    public void twoConcurrentMediaPromptsDecodeTogether() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = config();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            MediaPrompt prompt = MediaPrompt.builder()
                    .addText(tokens(6, 4))
                    .addMedia(MediaRopePositions.image2d(4, 4, 4), imageRows(16, 0.4f), true)
                    .addText(tokens(7, 2))
                    .build();
            BatchSequence a = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence b = engine.submit(MediaPrompt.builder()
                    .addText(tokens(8, 4))
                    .addMedia(MediaRopePositions.image2d(4, 4, 4), imageRows(16, 0.6f), true)
                    .addText(tokens(9, 2))
                    .build(), request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(BatchSequence.State.DONE, b.state());
            assertEquals(N_NEW, a.result().tokenCount());
            assertEquals(N_NEW, b.result().tokenCount());
        }
    }

    @Test
    public void aPreemptedMediaSequenceRecomputesItsChunksAndKeepsItsResult() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        // in pages of four, the 24 + 22 prompt rows take 6 + 6 of the 48-cell context's 12 pages, so both are
        // admitted, but their 12 decode cells do not fit beside them: the youngest sequence preempts, recomputes
        // chunk-wise when re-admitted, and must finish with the serial run's tokens
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(48).ubatch(8).kvPageSize(4)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
        MediaPrompt a = MediaPrompt.builder()
                .addText(tokens(2, 5))
                .addMedia(MediaRopePositions.image2d(5, 4, 4), imageRows(16, 0.3f), true)
                .addText(tokens(3, 3))
                .build();
        MediaPrompt b = MediaPrompt.builder()
                .addText(tokens(4, 4))
                .addMedia(MediaRopePositions.image2d(4, 4, 4), imageRows(16, 0.6f), true)
                .addText(tokens(5, 2))
                .build();
        GenerationResult preemptedA;
        GenerationResult preemptedB;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(a, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence hb = engine.submit(b, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertTrue("the context pressure preempted at least once", engine.stats().preemptions() > 0);
            assertEquals(BatchSequence.State.DONE, ha.state());
            assertEquals(BatchSequence.State.DONE, hb.state());
            preemptedA = ha.result();
            preemptedB = hb.result();
        }
        GenerationResult serialA;
        GenerationResult serialB;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(a, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence hb = engine.submit(b, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            serialA = ha.result();
            serialB = hb.result();
        }
        assertArrayEquals(serialA.tokens(), preemptedA.tokens());
        assertArrayEquals(serialB.tokens(), preemptedB.tokens());
    }

    // ------------------------------------------------------------------ encoder cache and budget

    /** The prompt of {@code idTests} with an identified image chunk and plain text around it. */
    private static MediaPrompt identifiedPrompt(int seed, String id, int pos0) {
        return MediaPrompt.builder()
                .addText(tokens(seed, 4))
                .addMedia(id, MediaRopePositions.image2d(pos0, 4, 4), imageRows(16, 0.5f), true)
                .addText(tokens(seed + 1, 2))
                .build();
    }

    @Test
    public void identifiedMediaChunksShareOneCacheEntryAndGenerateTheSameTokens() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).encoderCacheRows(16).build()).build();
        MediaPrompt a = identifiedPrompt(1, "img-a", 4);
        MediaPrompt b = identifiedPrompt(2, "img-a", 4);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(a, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence hb = engine.submit(b, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("one entry holds the image both submissions carry", 16,
                    engine.stats().encoderCacheRows());
            assertEquals("b's intern hit a's entry", 1, engine.stats().encoderCacheHits());
            engine.drain();
            assertEquals(BatchSequence.State.DONE, ha.state());
            assertEquals(BatchSequence.State.DONE, hb.state());
            assertEquals(N_NEW, ha.result().tokenCount());
            assertEquals(N_NEW, hb.result().tokenCount());
            assertEquals(16, engine.stats().encoderCacheRows());
            assertEquals(0, engine.stats().encoderCacheEvictions());
            // the identical run over an unidentified prompt generates the same tokens (the cache changes no rows)
            try (GGUFModel plain = new GGUFModel(model.toString(), cfg);
                 BatchEngine engine2 = BatchEngine.forModel(plain, cfg)) {
                MediaPrompt plainA = MediaPrompt.builder()
                        .addText(tokens(1, 4))
                        .addMedia(MediaRopePositions.image2d(4, 4, 4), imageRows(16, 0.5f), true)
                        .addText(tokens(2, 2))
                        .build();
                BatchSequence h = engine2.submit(plainA, request(),
                        (com.eonmux.neuralfree.gguf.TokenCallback) null);
                engine2.drain();
                assertArrayEquals(h.result().tokens(), ha.result().tokens());
            }
        }
    }

    @Test
    public void anEntryEvictsOnlyAfterItsLastReferenceRetires() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        // a 16-row budget holds one image; b's image can only enter once a's reference retired
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).encoderCacheRows(16).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(identifiedPrompt(1, "img-a", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence hb = engine.submit(identifiedPrompt(2, "img-b", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            assertEquals("b's submit found a still referenced: its rows stayed its own", 16,
                    engine.stats().encoderCacheRows());
            engine.drain();
            assertEquals(BatchSequence.State.DONE, ha.state());
            assertEquals(BatchSequence.State.DONE, hb.state());
            assertEquals("a's entry, never evicted, serves the cache after both retired", 16,
                    engine.stats().encoderCacheRows());
            assertEquals(0, engine.stats().encoderCacheEvictions());
            // a later submit with another image evicts the unreferenced a
            BatchSequence hc = engine.submit(identifiedPrompt(3, "img-c", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, hc.state());
            assertEquals(1, engine.stats().encoderCacheEvictions());
            assertEquals(16, engine.stats().encoderCacheRows());
        }
    }

    @Test
    public void theEncoderBudgetSpreadsAMediaPrefillOverStepsAndKeepsTheResult() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        // three images of 8 rows under a 8-row budget: every pump carries one image chunk at most
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(32)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).maxEncoderTokens(8).build()).build();
        MediaPrompt prompt = MediaPrompt.builder()
                .addText(tokens(1, 2))
                .addMedia("img-1", MediaRopePositions.image2d(2, 8, 1), imageRows(8, 0.2f), false)
                .addMedia("img-2", MediaRopePositions.image2d(10, 8, 1), imageRows(8, 0.4f), false)
                .addMedia("img-3", MediaRopePositions.image2d(18, 8, 1), imageRows(8, 0.6f), false)
                .addText(tokens(2, 2))
                .build();
        assertEquals(28, prompt.nTokens());
        GenerationResult budgeted;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence h = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            assertEquals("only the head and the first image prefilled", 2, prefillPosMax(m, h.seq()));
            engine.pump();
            assertEquals("the second image's step", 10, prefillPosMax(m, h.seq()));
            engine.pump();
            assertEquals("the third image and the free trailing text complete the prefill", 27,
                    prefillPosMax(m, h.seq()));
            engine.drain();
            assertEquals(BatchSequence.State.DONE, h.state());
            assertEquals(N_NEW, h.result().tokenCount());
            budgeted = h.result();
        }
        // the same prompt under a budget that holds every image prefills in one step and generates the same tokens
        EngineConfig wide = EngineConfig.builder().memory(cfg.memory().toBuilder().maxEncoderTokens(64).build())
                .build();
        GenerationResult oneStep;
        try (GGUFModel m = new GGUFModel(model.toString(), wide);
             BatchEngine engine = BatchEngine.forModel(m, wide)) {
            BatchSequence h = engine.submit(prompt, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            assertEquals(27, prefillPosMax(m, h.seq()));
            engine.drain();
            oneStep = h.result();
        }
        assertArrayEquals(oneStep.tokens(), budgeted.tokens());
    }

    @Test
    public void anOversizedChunkSpendsTheStepsEncoderBudgetForEveryOtherPrompt() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        // the default encoder budget is the step's token budget (the 8-row ubatch); each image has 16 rows
        EngineConfig cfg = config();
        MediaPrompt a = identifiedPrompt(1, "img-a", 4);
        MediaPrompt b = identifiedPrompt(2, "img-b", 4);
        GenerationResult solo;
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence h = engine.submit(b, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            solo = h.result();
        }
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(a, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence hb = engine.submit(b, request(), (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.pump();
            assertEquals("a's oversized image took the untouched budget: a prefilled whole", a.nPositions() - 1,
                    prefillPosMax(m, ha.seq()));
            assertEquals("b's image waits for the next step: only its head prefilled", 3,
                    prefillPosMax(m, hb.seq()));
            engine.pump();
            assertEquals("b's image and tail ran on the next step's fresh budget", b.nPositions() - 1,
                    prefillPosMax(m, hb.seq()));
            engine.drain();
            assertEquals(BatchSequence.State.DONE, ha.state());
            assertEquals(BatchSequence.State.DONE, hb.state());
            assertArrayEquals("the deferred prefill generates what b generates alone", solo.tokens(),
                    hb.result().tokens());
        }
    }

    /** The highest stored position of a sequence, read from its paged memory. */
    private static int prefillPosMax(GGUFModel m, int seq) {
        return ((HeapModelMemory) m.forwardBackend().memory()).sequences().posMax(seq);
    }

    @Test
    public void aNegativeEncoderCacheRowsTurnsTheCacheOff() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).encoderCacheRows(-1).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence h = engine.submit(identifiedPrompt(1, "img-a", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, h.state());
            assertEquals(0, engine.stats().encoderCacheRows());
            assertEquals(0, engine.stats().encoderCacheEvictions());
        }
    }

    @Test
    public void twoIdentifiedSubmissionsWithOneImageShareOneEntryAcrossTurns() throws IOException {
        Path model = SyntheticModels.qwen2vl().writeTempFile();
        EngineConfig cfg = EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).encoderCacheRows(16).build()).build();
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence ha = engine.submit(identifiedPrompt(1, "img-a", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            BatchSequence hb = engine.submit(identifiedPrompt(2, "img-a", 4), request(),
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, ha.state());
            assertEquals(BatchSequence.State.DONE, hb.state());
            assertEquals("the second turn hit the entry the first retired left", 16,
                    engine.stats().encoderCacheRows());
            assertEquals(1, engine.stats().encoderCacheHits());
            assertEquals(0, engine.stats().encoderCacheEvictions());
            assertEquals(N_NEW, ha.result().tokenCount());
            assertEquals(N_NEW, hb.result().tokenCount());
        }
    }
}
