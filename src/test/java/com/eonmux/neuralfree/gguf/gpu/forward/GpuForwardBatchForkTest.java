package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Fork and prefix-cache adoption over the GPU paged memory: sequences sharing a prompt share the donor's pages
 * (copy-on-write at their divergent tails) and a repeated prompt adopts its cached sequence, so the batch engine
 * decodes every choice of one prompt with the tokens of the single-sequence contiguous run.
 */
public class GpuForwardBatchForkTest {

    private static final int N_TOKENS = 12;
    private static final int N_NEW = 3;

    private HWAcceleration acceleration;
    private ComputeBackend backend;

    @Before
    public void setUp() {
        Assume.assumeTrue("OpenCL GPU not available",
                com.eonmux.neuralfree.acceleration.GpuTestUtil.isOpenCLAvailable());
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

    private static EngineConfig config(int parallel) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .parallel(parallel).typeK(KvDType.F32).typeV(KvDType.F32).build()).build();
    }

    /** The tokens a sequence runs (deterministic ids inside the synthetic vocab, as the CPU batch tests). */
    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Greedy decoding over a bounded budget. */
    private static GenerationRequest request(int maxNewTokens) {
        return GenerationRequest.builder()
                .maxNewTokens(maxNewTokens)
                .sampler(SamplerConfig.greedyDecoding())
                .build();
    }

    private static GenerationResult reference(Path model, EngineConfig cfg, int[] prompt,
                                              GenerationRequest request) throws IOException {
        try (GGUFModel one = new GGUFModel(model.toString(), cfg)) {
            return GenerationLoop.forModel(one, cfg).generate(prompt, request, null);
        }
    }

    @Test
    public void choicesOfOnePromptDecodeWithTheReferenceTokens() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(1, N_TOKENS);
        GenerationRequest request = request(N_NEW);
        GenerationResult expected = reference(model, config(1), prompt, request);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(4));
             BatchEngine engine = BatchEngine.forModel(gpu, config(4))) {
            BatchSequence[] choices = new BatchSequence[3];
            for (int choice = 0; choice < choices.length; choice++) {
                choices[choice] = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            }
            engine.drain();
            for (int choice = 0; choice < choices.length; choice++) {
                assertEquals(BatchSequence.State.DONE, choices[choice].state());
                assertArrayEquals("choice " + choice + " decodes with the reference tokens",
                        expected.tokens(), choices[choice].result().tokens());
            }
        }
    }

    @Test
    public void aRepeatedPromptAdoptsItsCachedPrefix() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] prompt = tokens(2, N_TOKENS);
        GenerationRequest request = request(N_NEW);
        GenerationResult expected = reference(model, config(1), prompt, request);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            BatchEngine engine = BatchEngine.forModel(gpu, config(2));
            BatchSequence first = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, first.state());
            assertArrayEquals("the first run decodes with the reference tokens",
                    expected.tokens(), first.result().tokens());
            BatchSequence second = engine.submit(prompt, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, second.state());
            assertArrayEquals("the repeated prompt decodes with the reference tokens again",
                    expected.tokens(), second.result().tokens());
        }
    }

    @Test
    public void forksWithDivergentTailsMatchTheirOwnReferences() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] shared = tokens(3, N_TOKENS - 4);
        int[] a = new int[N_TOKENS];
        int[] b = new int[N_TOKENS];
        System.arraycopy(shared, 0, a, 0, shared.length);
        System.arraycopy(shared, 0, b, 0, shared.length);
        for (int i = shared.length; i < N_TOKENS; i++) {
            a[i] = 100 + i;
            b[i] = 200 + i;
        }
        GenerationRequest request = request(N_NEW);
        GenerationResult expectedA = reference(model, config(1), a, request);
        GenerationResult expectedB = reference(model, config(1), b, request);
        try (GGUFModelGPU gpu = new GGUFModelGPU(model.toString(), backend, config(2))) {
            BatchEngine engine = BatchEngine.forModel(gpu, config(2));
            List<Integer> aTokens = new ArrayList<>();
            BatchSequence seqA = engine.submit(a, request, (tokenId, piece) -> aTokens.add(tokenId));
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seqA.state());
            assertArrayEquals("the first sequence alone decodes with its own reference tokens",
                    expectedA.tokens(), seqA.result().tokens());

            // the fork joins while a fresh donor decodes: its pages share, the tail prefills beside the decode
            BatchSequence seqA2 = engine.submit(a, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seqA2.state());
            BatchSequence seqB2 = engine.submit(b, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            BatchSequence seqA3 = engine.submit(a, request, (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seqB2.state());
            assertEquals(BatchSequence.State.DONE, seqA3.state());
            assertArrayEquals("the fork decodes with its own reference tokens",
                    expectedB.tokens(), seqB2.result().tokens());
            assertArrayEquals("the donor alongside the fork decodes with its own reference tokens",
                    expectedA.tokens(), seqA3.result().tokens());
        }
    }
}
