package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.ReasoningFilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerPrefilterTest;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.gpu.DeviceModelMemory;
import com.eonmux.neuralfree.gguf.memory.paged.PagedSequences;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Multi-sequence forward batch over the GPU paged memory: {@code
 * GpuForwardBackend.forwardBatch} runs EQUAL-mode multi-sequence ubatches through {@code PagedSequences.prepare},
 * the device page copies and page-table uploads, and the paged flash kernels ({@code LLM_FlashDecodePaged_*} over a
 * whole decode batch, {@code LLM_FlashAttnPaged_*} per multi-token segment), so a batch over several sequences
 * computes the same logits as the same tokens evaluated one sequence at a time on the contiguous CPU path. An
 * unpaged GPU memory refuses the batch by name, and {@code BatchEngine} runs continuous batching over the GPU
 * memory.
 */
public class GpuForwardBatchTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 8;
    private static final int SEQ_A = 0;
    private static final int SEQ_B = 1;
    private static final float TOL = 1e-3f;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        org.junit.Assume.assumeTrue("OpenCL GPU not available", com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
        acceleration = new HWAcceleration();
        CLDevice device = acceleration.getCLDevices().get(0);
        backend = new OclComputeBackend(device);
    }

    @After
    public void tearDown() {
        if (backend != null) {
            backend.close();
        }
        if (acceleration != null) {
            acceleration.shutdown();
        }
    }

    private static EngineConfig.Builder config(int parallel) {
        MemoryConfig.Builder mc = MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32);
        if (parallel > 1) {
            mc = mc.parallel(parallel);
        }
        return EngineConfig.builder().memory(mc.build());
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as the CPU batch tests). */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** A batch whose rows interleave two prompts; the last row of each sequence is its output row. */
    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
            if (i < b.length) {
                batch.add(b[i], i, SEQ_B, i == b.length - 1);
            }
        }
        return batch.build();
    }

    private static ForwardBatch single(int[] a) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, SEQ_A, i == a.length - 1);
        }
        return batch.build();
    }

    private static float[][] reference(Path model, EngineConfig cfg, int[]... prompts) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            float[][] out = new float[prompts.length][];
            for (int i = 0; i < prompts.length; i++) {
                out[i] = one.forwardBackend().forwardRows(prompts[i], 0,
                        new int[]{prompts[i].length - 1})[0];
                one.forwardBackend().reset();
            }
            return out;
        }
    }

    /** Greedy decoding over a bounded budget. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    @Test
    public void aBatchOverTwoSequencesMatchesTheSequentialContiguousRuns() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        float[][] reference = reference(model, config(1).build(), a, b);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            ForwardBackend f = two.forwardBackend();
            DeviceModelMemory mem = (DeviceModelMemory) f.memory();
            PagedSequences paged = mem.pagedSequences();
            assertNotNull(paged);
            assertEquals(2, paged.maxSequences());
            assertEquals(2, f.maxSequences());
            float[][] out = f.forwardBatch(interleave(a, b));
            assertEquals(2, out.length);
            assertArrayEquals("sequence A logits match the contiguous run", reference[0], out[0], TOL);
            assertArrayEquals("sequence B logits match the contiguous run", reference[1], out[1], TOL);
        }
    }

    @Test
    public void aMixedLengthBatchPrefillsThroughThePagedPrefillPath() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(3, N_TOKENS);
        int[] b = tokens(4, 5);
        float[][] reference = reference(model, config(1).build(), a, b);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            ForwardBatch.Builder batch = ForwardBatch.builder(N_TOKENS + 5);
            // rows interleave the two prompts; the last row of each sequence is its output row
            for (int i = 0; i < N_TOKENS; i++) {
                batch.add(a[i], i, SEQ_A, i == N_TOKENS - 1);
                if (i < 5) {
                    batch.add(b[i], i, SEQ_B, i == 4);
                }
            }
            float[][] out = two.forwardBackend().forwardBatch(batch.build());
            assertEquals(2, out.length);
            // the results follow the batch order of the output rows: B's last row (batch row 9) comes first
            assertArrayEquals("short sequence logits match", reference[1], out[0], TOL);
            assertArrayEquals("long sequence logits match", reference[0], out[1], TOL);
        }
    }

    @Test
    public void aDecodeBatchOverRunningSequencesMatchesTheContiguousDecode() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(1, N_TOKENS);
        int[] b = tokens(2, N_TOKENS);
        float[][] ref = new float[2][];
        try (GGUFModel one = new GGUFModel(model.toString(), config(1).build())) {
            ForwardBackend cpu = one.forwardBackend();
            cpu.forwardRows(a, 0, null);
            ref[0] = cpu.decode(a[0], N_TOKENS).clone();
            cpu.reset();
            cpu.forwardRows(b, 0, null);
            ref[1] = cpu.decode(b[0], N_TOKENS).clone();
        }
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            ForwardBackend f = two.forwardBackend();
            f.forwardBatch(interleave(a, b));
            // the next token of each sequence in one batched decode step over the page tables
            ForwardBatch.Builder batch = ForwardBatch.builder(2);
            batch.add(a[0], N_TOKENS, SEQ_A, true);
            batch.add(b[0], N_TOKENS, SEQ_B, true);
            float[][] out = f.forwardBatch(batch.build());
            assertArrayEquals("batched decode of sequence A matches", ref[0], out[0], TOL);
            assertArrayEquals("batched decode of sequence B matches", ref[1], out[1], TOL);
        }
    }

    @Test
    public void thePagedPathAcceptsASingleSequenceBatchAndMatchesTheContiguousOne() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(3, N_TOKENS);
        float[] reference = reference(model, config(1).build(), a)[0];
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            float[][] out = two.forwardBackend().forwardBatch(single(a));
            assertArrayEquals("single-sequence paged run matches the contiguous one", reference, out[0], TOL);
        }
    }

    @Test
    public void anUnpagedGpuMemoryRefusesTheBatchByName() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        try (GGUFModelGPU one = new GGUFModelGPU(model.toString(), backend, config(1).build())) {
            ForwardBackend f = one.forwardBackend();
            assertNotNull(f.batchRefusal());
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> f.forwardBatch(interleave(tokens(1, N_TOKENS), tokens(2, N_TOKENS))));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    @Test
    public void aBatchEngineRunsContinuousBatchingOnTheGpuPagedMemory() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig cpuCfg = config(1).build();
        EngineConfig gpuCfg = config(2).build();
        GenerationRequest request = request(N_NEW);
        // reference: the same prompts through the CPU single-sequence generation loop
        GenerationResult[] refs = new GenerationResult[2];
        try (GGUFModel one = new GGUFModel(model.toString(), cpuCfg)) {
            refs[0] = GenerationLoop.forModel(one, cpuCfg).generate(tokens(1, N_TOKENS), request, null);
            refs[1] = GenerationLoop.forModel(one, cpuCfg).generate(tokens(2, N_TOKENS), request, null);
        }
        List<String> pieces = new ArrayList<>();
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            assertEquals(SupportClass.GEN, two.spec().supportClass());
            assertEquals(2, engine.maxSequences());
            BatchSequence a = engine.submit(tokens(1, N_TOKENS), request, (channel, tokenId, piece) -> {
                if (channel == ReasoningFilter.Channel.CONTENT) {
                    pieces.add(piece);
                }
            }, null);
            BatchSequence b = engine.submit(tokens(2, N_TOKENS), request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, a.state());
            assertEquals(BatchSequence.State.DONE, b.state());
            assertMatches(refs[0], a.result());
            assertMatches(refs[1], b.result());
            StringBuilder text = new StringBuilder();
            for (String p : pieces) {
                text.append(p);
            }
            assertEquals("routed content pieces build the result text", refs[0].text(), text.toString());
        }
    }

    @Test
    public void batchTopKCandidatesEqualTheHostSelection() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int k = 8;
        int[] a = tokens(5, N_TOKENS);
        int[] b = tokens(6, N_TOKENS);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            ForwardBackend f = two.forwardBackend();
            assertTrue(f.batchTopKOnDevice());
            float[][] rows = f.forwardBatch(interleave(a, b));
            assertTopKMatches(RowTopK.of(rows, k), f, a, b, k);
            // a wider cone on the same head rebinds its readout
            assertTopKMatches(RowTopK.of(rows, k + 4), f, a, b, k + 4);
            // a narrower cone reuses the bound readout and reads its first candidates
            assertTopKMatches(RowTopK.of(rows, k), f, a, b, k);
            f.reset();
        }
    }

    /** Resets the backend, reads {@code k} candidates of the two-sequence batch back and compares them. */
    private static void assertTopKMatches(RowTopK host, ForwardBackend f, int[] a, int[] b, int k) {
        f.reset();
        RowTopK device = f.forwardBatchTopK(interleave(a, b), k);
        assertEquals(host.rows(), device.rows());
        assertEquals(host.k(), device.k());
        for (int r = 0; r < host.rows(); r++) {
            assertArrayEquals("row " + r + " ids", host.ids(r), device.ids(r));
            assertArrayEquals("row " + r + " logits", host.logits(r), device.logits(r), TOL);
        }
    }

    /**
     * A batch engine over the GPU paged memory samples the seeds' chains over the device's top candidates and
     * emits the same tokens as the same engine with device sampling off (which reads the rows back whole).
     */
    @Test
    public void aBatchEngineSamplesOverTheDeviceTopKWhenTheChainsFit() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig gpuCfg = config(2).build();
        GenerationRequest request = temperatureRequest(N_NEW);
        int[][] deviceTokens = new int[2][];
        long deviceRows;
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(tokens(1, N_TOKENS), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2, N_TOKENS), request, (TokenCallback) null);
            engine.drain();
            deviceTokens[0] = a.result().tokens();
            deviceTokens[1] = b.result().tokens();
            deviceRows = engine.stats().deviceSampledRows();
        }
        assertTrue("the batch sampled over the device's candidates", deviceRows > 0);

        int[][] hostTokens = new int[2][];
        long hostRows;
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(tokens(1, N_TOKENS), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2, N_TOKENS), request, (TokenCallback) null);
            engine.drain();
            hostTokens[0] = a.result().tokens();
            hostTokens[1] = b.result().tokens();
            hostRows = engine.stats().deviceSampledRows();
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertEquals("device sampling off stays on the host", 0L, hostRows);
        assertArrayEquals(hostTokens[0], deviceTokens[0]);
        assertArrayEquals(hostTokens[1], deviceTokens[1]);
    }

    /**
     * A batch engine samples a greedy chain whose only edits are bans and non-positive biases over the device's
     * candidate prefix ({@code Sampler#coneWidth} without a top-k stage; llama.cpp's logit-bias sampler, Avalon's
     * {@code greedy_argmax}) and emits the same tokens as the same engine with device sampling off.
     */
    @Test
    public void aBatchEngineSamplesAGreedySparseChainOverTheDeviceTopK() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig gpuCfg = config(2).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().greedy(true).topK(0).penaltyLastN(0).build())
                .logitBias(5, -100f)
                .logitBias(9, -2f)
                .build();
        int[][] deviceTokens = new int[2][];
        long deviceRows;
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(tokens(1, N_TOKENS), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2, N_TOKENS), request, (TokenCallback) null);
            engine.drain();
            deviceTokens[0] = a.result().tokens();
            deviceTokens[1] = b.result().tokens();
            deviceRows = engine.stats().deviceSampledRows();
        }
        assertTrue("the batch sampled the ban chain over the device's candidates", deviceRows > 0);

        int[][] hostTokens = new int[2][];
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(tokens(1, N_TOKENS), request, (TokenCallback) null);
            BatchSequence b = engine.submit(tokens(2, N_TOKENS), request, (TokenCallback) null);
            engine.drain();
            hostTokens[0] = a.result().tokens();
            hostTokens[1] = b.result().tokens();
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertArrayEquals(hostTokens[0], deviceTokens[0]);
        assertArrayEquals(hostTokens[1], deviceTokens[1]);
    }

    /**
     * The device prefilter ({@code LLM_LogitsPrefilter}) applies the same logit bias and penalties the host
     * sampler carries, so the candidates it reads back equal the host's top-k of the edited rows.
     */
    @Test
    public void theDevicePrefilterEditsMatchTheHost() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int k = 8;
        int[] a = tokens(5, N_TOKENS);
        int[] b = tokens(6, N_TOKENS);
        SamplerConfig config = SamplerConfig.builder().temperature(0.9f).topK(k).penaltyLastN(8)
                .repeatPenalty(1.3f).frequencyPenalty(0.5f).presencePenalty(0.2f).seed(7).build();
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(3, 4.0f);
        Sampler sa = Sampler.create(config, new int[0], bias);
        Sampler sb = Sampler.create(config, new int[0], Collections.singletonMap(5, -2.0f));
        sa.accept(a, 0, a.length);
        sb.accept(b, 0, b.length);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2).build())) {
            ForwardBackend f = two.forwardBackend();
            assertTrue("the backend applies the prefilter on the device", f.batchPrefilterOnDevice());
            float[][] rows = f.forwardBatch(interleave(a, b));
            LogitPrefilter prefilter = LogitPrefilter.of(List.of(sa.prefilterRow(), sb.prefilterRow()));
            RowTopK host = RowTopK.of(new float[][]{
                    SamplerPrefilterTest.applyPrefilter(rows[0], sa.prefilterRow()),
                    SamplerPrefilterTest.applyPrefilter(rows[1], sb.prefilterRow())}, k);
            f.reset();
            RowTopK device = f.forwardBatchTopK(interleave(a, b), k, prefilter);
            assertEquals(host.rows(), device.rows());
            for (int r = 0; r < host.rows(); r++) {
                assertArrayEquals("row " + r + " ids", host.ids(r), device.ids(r));
                assertArrayEquals("row " + r + " logits", host.logits(r), device.logits(r), TOL);
            }
            f.reset();
        }
    }

    /**
     * A batch engine whose requests carry repetition, presence and frequency penalties and DRY samples over the
     * device's own edited rows and emits the same tokens as the same engine with device sampling off.
     */
    @Test
    public void aBatchEngineSamplesWithPenaltiesOverTheDevicePrefilter() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        EngineConfig gpuCfg = config(2).build();
        // a repeated prompt makes the first draw depend on the penalty window
        int[] promptA = {5, 5, 5, 5, 6, 7, 8, 9, 5, 5, 6, 7};
        int[] promptB = {11, 11, 12, 12, 11, 11, 12, 12, 11, 11, 12, 12};
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.builder().temperature(0.9f).topK(40).topP(0.95f).minP(0.05f)
                        .typicalP(1f).penaltyLastN(12).repeatPenalty(1.3f).frequencyPenalty(0.4f)
                        .presencePenalty(0.2f).dryMultiplier(0.7f).dryBase(2f).dryAllowedLength(2)
                        .dryPenaltyLastN(12).drySequenceBreakers(Collections.emptyList()).seed(99L).build())
                .logitBias(3, 2.5f)
                .build();
        int[][] deviceTokens = new int[2][];
        long deviceRows;
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(promptA, request, (TokenCallback) null);
            BatchSequence b = engine.submit(promptB, request, (TokenCallback) null);
            engine.drain();
            if (a.error() != null) {
                throw new AssertionError("sequence A failed", a.error());
            }
            if (b.error() != null) {
                throw new AssertionError("sequence B failed", b.error());
            }
            deviceTokens[0] = a.result().tokens();
            deviceTokens[1] = b.result().tokens();
            deviceRows = engine.stats().deviceSampledRows();
        }
        assertTrue("the penalty batch sampled over the device's edited rows", deviceRows > 0);

        int[][] hostTokens = new int[2][];
        long hostRows;
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, gpuCfg);
             BatchEngine engine = BatchEngine.forModel(two, gpuCfg)) {
            BatchSequence a = engine.submit(promptA, request, (TokenCallback) null);
            BatchSequence b = engine.submit(promptB, request, (TokenCallback) null);
            engine.drain();
            hostTokens[0] = a.result().tokens();
            hostTokens[1] = b.result().tokens();
            hostRows = engine.stats().deviceSampledRows();
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertEquals("device sampling off stays on the host", 0L, hostRows);
        assertArrayEquals(hostTokens[0], deviceTokens[0]);
        assertArrayEquals(hostTokens[1], deviceTokens[1]);
    }

    /** A seeded chain a device top-k cone covers (temperature, top-k, top-p and min-p). */
    private static GenerationRequest temperatureRequest(int maxNewTokens) {
        return GenerationRequest.builder().maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.builder().temperature(0.9f).topK(40).topP(0.95f).minP(0.05f).typicalP(1f)
                        .penaltyLastN(0).seed(1234L).build())
                .build();
    }

    /** Greedy results agree with the reference run (tokens, or the text when the tokenizer maps them). */
    private static void assertMatches(GenerationResult expected, GenerationResult actual) {
        assertEquals(expected.tokenCount(), actual.tokenCount());
        assertEquals(expected.text(), actual.text());
    }
}
