package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.cuda.CudaComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ocl.OclComputeBackend;
import com.eonmux.neuralfree.acceleration.cuda.CUDevice;
import com.eonmux.neuralfree.acceleration.kernels.Availability;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.TokenCallback;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.ForwardBatch;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.LogitPrefilter;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerPrefilterTest;
import com.eonmux.neuralfree.gguf.engine.SplitReadout;
import com.eonmux.neuralfree.gguf.engine.TokenScore;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The split readout ({@code ForwardBackend.forwardBatchSplit}) on every available GPU backend: one forward reads every
 * output row's device candidates and the whole logits of the flagged rows, which equal the rows a whole-row forward
 * reads (unedited by the prefilter of the other rows), while the other rows' candidates equal the host's top-k of
 * their (prefiltered) rows, over output rows spread across ubatches. A batch engine with one logprobs request among
 * sampled ones splits its steps that way and emits what it emits with every row read whole.
 */
@RunWith(Parameterized.class)
public class GpuSplitReadoutTest {

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

    public GpuSplitReadoutTest(String api) {
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

    private static EngineConfig config(int parallel) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(parallel).build()).build();
    }

    private static int[] tokens(int seed, int n) {
        int[] t = new int[n];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 7 + i * 13) % 250;
        }
        return t;
    }

    /** Two interleaved prompts; every row is an output row, so the output rows span the batch's ubatches. */
    private static ForwardBatch everyRow(int[] a, int[] b) {
        ForwardBatch.Builder batch = ForwardBatch.builder(a.length + b.length);
        for (int i = 0; i < a.length; i++) {
            batch.add(a[i], i, 0, true);
            batch.add(b[i], i, 1, true);
        }
        return batch.build();
    }

    @Test
    public void flaggedRowsComeBackWholeBesideEveryRowsCandidates() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(5, TOKENS);
        int[] b = tokens(6, TOKENS);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2))) {
            ForwardBackend f = two.forwardBackend();
            float[][] rows = f.forwardBatch(everyRow(a, b));
            f.reset();
            int n = rows.length;
            boolean[] full = new boolean[n];
            for (int r = 0; r < n; r += 3) {
                full[r] = true;
            }
            SplitReadout split = f.forwardBatchSplit(everyRow(a, b), full, K, null);
            f.reset();
            RowTopK host = RowTopK.of(rows, K);
            assertEquals(n, split.rows());
            for (int r = 0; r < n; r++) {
                if (full[r]) {
                    assertArrayEquals(api + " whole row " + r, rows[r], split.row(r), TOL);
                } else {
                    assertNull(api + " row " + r + " is not read whole", split.row(r));
                    assertArrayEquals(api + " row " + r + " ids", host.ids(r), split.top().ids(r));
                    assertArrayEquals(api + " row " + r + " logits", host.logits(r), split.top().logits(r), TOL);
                }
            }
        }
    }

    @Test
    public void aWholeRowStaysUneditedBesideAPrefilteredRow() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        int[] a = tokens(5, TOKENS);
        int[] b = tokens(6, TOKENS);
        SamplerConfig penalized = SamplerConfig.builder().temperature(0.9f).topK(K).penaltyLastN(8)
                .repeatPenalty(1.3f).frequencyPenalty(0.5f).presencePenalty(0.2f).seed(7).build();
        Sampler sb = Sampler.create(penalized);
        sb.accept(b, 0, b.length);
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config(2))) {
            ForwardBackend f = two.forwardBackend();
            Assume.assumeTrue(f.batchPrefilterOnDevice());
            float[][] rows = f.forwardBatch(everyRow(a, b));
            f.reset();
            int n = rows.length;
            boolean[] full = new boolean[n];
            List<LogitPrefilter.Row> edits = new ArrayList<>();
            for (int r = 0; r < n; r++) {
                full[r] = r % 2 == 0;
                edits.add(full[r] ? LogitPrefilter.Row.builder().build() : sb.prefilterRow());
            }
            LogitPrefilter prefilter = LogitPrefilter.of(edits);
            SplitReadout split = f.forwardBatchSplit(everyRow(a, b), full, K, prefilter);
            f.reset();
            for (int r = 0; r < n; r++) {
                if (full[r]) {
                    assertArrayEquals(api + " unedited row " + r, rows[r], split.row(r), TOL);
                } else {
                    RowTopK host = RowTopK.of(new float[][]{
                            SamplerPrefilterTest.applyPrefilter(rows[r], prefilter.row(r))}, K);
                    assertArrayEquals(api + " row " + r + " ids", host.ids(0), split.top().ids(r));
                    assertArrayEquals(api + " row " + r + " logits", host.logits(0), split.top().logits(r), TOL);
                }
            }
            List<LogitPrefilter.Row> wrong = new ArrayList<>(edits);
            wrong.set(0, sb.prefilterRow());
            assertThrows(IllegalArgumentException.class,
                    () -> f.forwardBatchSplit(everyRow(a, b), full, K, LogitPrefilter.of(wrong)));
        }
    }

    /** Runs the requests through a batch engine and stores its device-sampled row count in {@code deviceRows[0]}. */
    private List<GenerationResult> run(Path model, List<GenerationRequest> requests, long[] deviceRows)
            throws IOException {
        EngineConfig cfg = config(requests.size());
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), backend, cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            List<BatchSequence> running = new ArrayList<>();
            for (int i = 0; i < requests.size(); i++) {
                running.add(engine.submit(tokens(i + 1, 5 + i), requests.get(i), (TokenCallback) null));
            }
            engine.drain();
            List<GenerationResult> out = new ArrayList<>();
            for (BatchSequence s : running) {
                assertEquals(String.valueOf(s.error()), BatchSequence.State.DONE, s.state());
                out.add(s.result());
            }
            deviceRows[0] = engine.stats().deviceSampledRows();
            return out;
        }
    }

    @Test
    public void aBatchEngineSplitsItsStepsAroundOneLogprobsRequest() throws IOException {
        Path model = SyntheticModels.llama().writeTempFile();
        List<GenerationRequest> requests = new ArrayList<>();
        for (int i = 0; i < 3; i++) {
            requests.add(GenerationRequest.builder().maxNewTokens(8).sampler(SamplerConfig.builder().topK(K)
                    .temperature(0.9f).seed(50L + i).build()).build());
        }
        requests.add(GenerationRequest.builder().maxNewTokens(8).logprobs(3).sampler(SamplerConfig.builder()
                .topK(K).temperature(0.9f).seed(60L).build()).build());
        long[] deviceRows = new long[1];
        List<GenerationResult> split = run(model, requests, deviceRows);
        assertTrue(api + ": the sampled requests read candidates beside the logprobs request", deviceRows[0] > 0);
        long[] hostRows = new long[1];
        List<GenerationResult> whole;
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, "false");
        try {
            whole = run(model, requests, hostRows);
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
        assertEquals(0L, hostRows[0]);
        for (int i = 0; i < requests.size(); i++) {
            assertArrayEquals(api + " request " + i, whole.get(i).tokens(), split.get(i).tokens());
        }
        List<TokenScore[]> expected = whole.get(3).logprobs();
        List<TokenScore[]> actual = split.get(3).logprobs();
        assertEquals(expected.size(), actual.size());
        for (int t = 0; t < expected.size(); t++) {
            for (int j = 0; j < expected.get(t).length; j++) {
                assertEquals(expected.get(t)[j].tokenId(), actual.get(t)[j].tokenId());
                assertEquals(expected.get(t)[j].logprob(), actual.get(t)[j].logprob(), TOL);
            }
        }
    }
}
