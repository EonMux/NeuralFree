package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerPrefilterTest;
import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The pre-top-k logit prefilter ({@code LLM_LogitsPrefilter}) on every available GPU backend: the candidates a
 * batch reads back after the device applied each row's logit bias, penalties and DRY equal the host's top-k of
 * rows whose edits the same tables apply on the host.
 */
@RunWith(Parameterized.class)
public class GpuSamplingPrefilterBackendsTest {

    private static final int TOKENS = 12;
    private static final int K = 8;
    private static final float TOL = 1e-3f;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private boolean previousCuda;

    public GpuSamplingPrefilterBackendsTest(String api) {
        this.api = api;
    }

    @Before
    public void openDevice() {
        previousCuda = HWAcceleration.enableCuda;
        try {
            if (api.equals("cuda")) {
                HWAcceleration.enableCuda = true;
                acceleration = new HWAcceleration();
                Assume.assumeFalse("no CUDA device", acceleration.getCUDevices().isEmpty());
                CUDevice device = acceleration.getCUDevices().get(0);
                backend = new CudaComputeBackend(device);
            } else {
                acceleration = new HWAcceleration();
                Assume.assumeFalse("no OpenCL device", acceleration.getCLDevices().isEmpty());
                backend = new OclComputeBackend(acceleration.getCLDevices().get(0));
            }
            Availability availability = LlmKernels.availability(backend);
            Assume.assumeTrue("LLM kernels unavailable on " + api + ": " + availability.reason(),
                    availability.available());
        } catch (Throwable e) {
            closeDevice();
            Assume.assumeTrue("no usable " + api + " GPU: " + e.getMessage(), false);
        }
    }

    @After
    public void closeDevice() {
        HWAcceleration.enableCuda = previousCuda;
        if (backend != null) {
            backend.close();
            backend = null;
        }
        if (acceleration != null) {
            acceleration.shutdown();
            acceleration = null;
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(2).build()).build();
    }

    private static int[] tokens(int seed) {
        int[] t = new int[TOKENS];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    private static ForwardBatch interleave(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, 0, i == a.length - 1);
            if (i < b.length) {
                batch.add(b[i], i, 1, i == b.length - 1);
            }
        }
        return batch.build();
    }

    @Test
    public void prefilteredCandidatesEqualTheHostTopK() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(5);
        int[] b = tokens(6);
        SamplerConfig sampling = SamplerConfig.builder().temperature(0.9f).topK(K).penaltyLastN(8)
                .repeatPenalty(1.3f).frequencyPenalty(0.5f).presencePenalty(0.2f).seed(7).build();
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(3, 4.0f);
        Sampler sa = Sampler.create(sampling, new int[0], bias);
        Sampler sb = Sampler.create(sampling, new int[0], java.util.Collections.singletonMap(5, -2.0f));
        sa.accept(a, 0, a.length);
        sb.accept(b, 0, b.length);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config())) {
            ForwardBackend f = two.forwardBackend();
            assertTrue("the " + api + " backend applies the prefilter", f.batchPrefilterOnDevice());
            float[][] rows = f.forwardBatch(interleave(a, b));
            LogitPrefilter prefilter = LogitPrefilter.of(List.of(sa.prefilterRow(), sb.prefilterRow()));
            RowTopK host = RowTopK.of(new float[][]{
                    SamplerPrefilterTest.applyPrefilter(rows[0], sa.prefilterRow()),
                    SamplerPrefilterTest.applyPrefilter(rows[1], sb.prefilterRow())}, K);
            f.reset();
            RowTopK device = f.forwardBatchTopK(interleave(a, b), K, prefilter);
            assertEquals(host.rows(), device.rows());
            for (int r = 0; r < host.rows(); r++) {
                assertArrayEquals(api + " row " + r + " ids", host.ids(r), device.ids(r));
                assertArrayEquals(api + " row " + r + " logits", host.logits(r), device.logits(r), TOL);
            }
            f.reset();
        }
    }

    /**
     * DRY's restart sequences run on the device: a row whose window repeats a suffix and whose breakers name a
     * multi-token sequence reads back the same candidates as the host's application of the row's own tables.
     */
    @Test
    public void dryWithRestartSequencesMatchesTheHost() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        // the repeated "a b" suffix inside the window makes DRY penalize, and "ab" tokenizes to head a + tail b
        int[] a = {4, 5, 4, 5, 4, 5, 6, 5, 4, 5, 4, 5};
        int[] b = {7, 8, 7, 8, 7, 8, 9, 8, 7, 8, 7, 8};
        SamplerConfig sampling = SamplerConfig.builder().temperature(0.9f).topK(K).penaltyLastN(0)
                .dryMultiplier(0.8f).dryBase(2f).dryAllowedLength(2).dryPenaltyLastN(12)
                .drySequenceBreakers(Arrays.asList("ab")).seed(11).build();
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config())) {
            SamplerVocab vocab = Sampler.vocab(two.tokenizer());
            Sampler sa = Sampler.create(sampling, new int[0], vocab);
            Sampler sb = Sampler.create(sampling, new int[0], vocab);
            sa.accept(a, 0, a.length);
            sb.accept(b, 0, b.length);
            assertTrue("the row carries restart sequences", sa.prefilterRow().dryBreakers().length > 0);
            ForwardBackend f = two.forwardBackend();
            float[][] rows = f.forwardBatch(interleave(a, b));
            LogitPrefilter prefilter = LogitPrefilter.of(List.of(sa.prefilterRow(), sb.prefilterRow()));
            float[] editedA = SamplerPrefilterTest.applyPrefilter(rows[0], sa.prefilterRow());
            float[] editedB = SamplerPrefilterTest.applyPrefilter(rows[1], sb.prefilterRow());
            assertFalse("DRY edits the first row", Arrays.equals(rows[0], editedA));
            RowTopK host = RowTopK.of(new float[][]{editedA, editedB}, K);
            f.reset();
            RowTopK device = f.forwardBatchTopK(interleave(a, b), K, prefilter);
            assertEquals(host.rows(), device.rows());
            for (int r = 0; r < host.rows(); r++) {
                assertArrayEquals(api + " row " + r + " ids", host.ids(r), device.ids(r));
                assertArrayEquals(api + " row " + r + " logits", host.logits(r), device.logits(r), TOL);
            }
            f.reset();
        }
    }
}
