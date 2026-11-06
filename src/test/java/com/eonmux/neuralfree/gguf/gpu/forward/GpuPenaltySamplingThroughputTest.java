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
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.engine.batch.BatchStats;
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
import static org.junit.Assert.assertTrue;

/**
 * Token-generation throughput of a penalty-enabled batch on every available GPU backend, with the pre-top-k
 * prefilter on the device (the top-k readback) and with device sampling off (the whole vocabulary row read back
 * per sequence and token). The two modes emit the same tokens; the measured rates are printed to the test
 * output.
 */
@RunWith(Parameterized.class)
public class GpuPenaltySamplingThroughputTest {

    private static final int SEQUENCES = 4;
    private static final int PROMPT = 16;
    private static final int NEW_TOKENS = 64;
    private static final int ROUNDS = 3;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"});
    }

    private final String api;
    private HWAcceleration acceleration;
    private ComputeBackend backend;
    private boolean previousCuda;

    public GpuPenaltySamplingThroughputTest(String api) {
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
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(256).ubatch(8)
                .typeK(KvDType.F32).typeV(KvDType.F32).parallel(SEQUENCES).build()).build();
    }

    private static int[] prompt(int seed) {
        int[] t = new int[PROMPT];
        for (int i = 0; i < t.length; i++) {
            t[i] = 4 + (seed * 37 + i * 13) % 2048;
        }
        return t;
    }

    private static GenerationRequest request(int newTokens) {
        return GenerationRequest.builder().maxNewTokens(newTokens)
                .sampler(SamplerConfig.builder().temperature(0.9f).topK(40).topP(0.95f).minP(0.05f)
                        .typicalP(1f).penaltyLastN(24).repeatPenalty(1.3f).frequencyPenalty(0.4f)
                        .presencePenalty(0.2f).seed(1234L).build())
                .logitBias(3, 2.5f)
                .build();
    }

    /** One mode's measured tokens per second and the tokens it emitted. */
    private static final class Run {
        final double tokensPerSecond;
        final int[][] tokens;

        Run(double tokensPerSecond, int[][] tokens) {
            this.tokensPerSecond = tokensPerSecond;
            this.tokens = tokens;
        }
    }

    private Run run(Path model, boolean deviceSampling) throws IOException {
        System.setProperty(GenerationLoop.PROP_SAMPLING_DEVICE, Boolean.toString(deviceSampling));
        try (GGUFModelGPU two = new GGUFModelGPU(model.toString(), backend, config());
             BatchEngine engine = BatchEngine.forModel(two, config())) {
            engine.submit(prompt(0), request(2), (TokenCallback) null);
            engine.submit(prompt(1), request(2), (TokenCallback) null);
            engine.drain();
            BatchStats before = engine.stats();
            List<BatchSequence> runs = new ArrayList<>();
            for (int i = 0; i < SEQUENCES; i++) {
                runs.add(engine.submit(prompt(i + 2), request(NEW_TOKENS), (TokenCallback) null));
            }
            engine.drain();
            BatchStats after = engine.stats();
            long generated = after.generatedTokens() - before.generatedTokens();
            double seconds = after.decodeSeconds() - before.decodeSeconds();
            assertTrue(api + ": the run generated tokens", generated > 0 && seconds > 0);
            long deviceRows = after.deviceSampledRows() - before.deviceSampledRows();
            if (deviceSampling) {
                assertTrue(api + ": the prefilter path sampled on the device", deviceRows > 0);
            } else {
                assertEquals(api + ": device sampling off stays on the host", 0L, deviceRows);
            }
            int[][] tokens = new int[SEQUENCES][];
            for (int i = 0; i < SEQUENCES; i++) {
                tokens[i] = runs.get(i).result().tokens();
            }
            return new Run(generated / seconds, tokens);
        } finally {
            System.clearProperty(GenerationLoop.PROP_SAMPLING_DEVICE);
        }
    }

    @Test
    public void penaltyThroughputWithAndWithoutTheDevicePrefilter() throws IOException {
        Path model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder()
                .nVocab(32768).nEmbd(256).nLayer(4).nHead(8).nHeadKv(4).headDim(32).nRot(32).nFf(512).build())
                .writeTempFile();
        Run device = null;
        Run host = null;
        double[] deviceRates = new double[ROUNDS];
        double[] hostRates = new double[ROUNDS];
        for (int round = 0; round < ROUNDS; round++) {
            Run d = run(model, true);
            Run h = run(model, false);
            deviceRates[round] = d.tokensPerSecond;
            hostRates[round] = h.tokensPerSecond;
            if (round == 0) {
                device = d;
                host = h;
            }
        }
        for (int i = 0; i < SEQUENCES; i++) {
            assertArrayEquals(api + ": sequence " + i + " emits the same tokens",
                    host.tokens[i], device.tokens[i]);
        }
        double deviceRate = median(deviceRates);
        double hostRate = median(hostRates);
        System.out.printf("tg throughput %s (vocab 32768, top-k 40, %d sequences x %d tokens): device prefilter"
                + " %.1f tok/s, host whole-row %.1f tok/s, %.2fx%n", api, SEQUENCES, NEW_TOKENS, deviceRate,
                hostRate, deviceRate / hostRate);
        assertTrue(api + ": the device prefilter ran", deviceRate > 0);
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }
}
