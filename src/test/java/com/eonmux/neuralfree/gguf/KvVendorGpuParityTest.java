package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The vLLM vendor KV cache formats on the device caches: a GPU load with the caches in device memory
 * ({@code neuralfree.kv.offload=true}, the default) produces the logits of the same load with the caches in host RAM,
 * which decodes through the host path of the same format. The two runs share every projection and FFN kernel, so a
 * difference is the device store, the device flash decode/prefill or the side-plane transfer.
 *
 * <p>Every test repeats over the formats on the machine's OpenCL, CUDA and Vulkan devices.</p>
 */
public class KvVendorGpuParityTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int[] DECODE = {77, 3};
    private static final int GREEDY_TOKENS = 4;
    /** Logit tolerance of a vendor cache against the host path of the same format (its quantization noise). */
    private static final float TOL = 5e-3f;

    private static final List<Format> FORMATS = Arrays.asList(
            new Format(KvDType.NVFP4, KvDType.NVFP4),
            new Format(KvDType.NVFP4_4OVER6, KvDType.NVFP4_4OVER6),
            new Format(KvDType.TQ_K8_KEYS, KvDType.TQ_4BIT_VALUES),
            new Format(KvDType.TQ_4BIT_KEYS, KvDType.TQ_4BIT_VALUES),
            new Format(KvDType.TQ_3BIT_KEYS, KvDType.TQ_3BIT_VALUES),
            new Format(KvDType.TQ_3BIT_KEYS, KvDType.TQ_4BIT_VALUES));

    private Path model;

    @After
    public void closeModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void openClDeviceCachesMatchTheHostPath() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertDeviceParity(EngineConfig.BackendMode.OPENCL, FORMATS);
    }

    @Test
    public void cudaDeviceCachesMatchTheHostPath() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        assertDeviceParity(EngineConfig.BackendMode.CUDA, FORMATS);
    }

    @Test
    public void vulkanDeviceCachesMatchTheHostPath() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        assertDeviceParity(EngineConfig.BackendMode.VULKAN, FORMATS);
    }

    private void assertDeviceParity(EngineConfig.BackendMode backend, List<Format> formats) throws IOException {
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        for (Format f : formats) {
            float[][] host = run(backend, f, false);
            float[][] device = run(backend, f, true);
            for (int i = 0; i < host.length; i++) {
                float[] expected = host[i];
                float[] got = device[i];
                float max = 0f;
                for (int j = 0; j < expected.length; j++) {
                    if (!Float.isFinite(expected[j]) || !Float.isFinite(got[j])) {
                        throw new AssertionError(backend + " " + f + ": step " + i + " logit " + j
                                + " is not finite (host " + expected[j] + ", device " + got[j] + ")");
                    }
                    max = Math.max(max, Math.abs(expected[j] - got[j]));
                }
                if (max > TOL) {
                    throw new AssertionError(backend + " " + f + ": step " + i + " differs from the host path by "
                            + max + " > " + TOL);
                }
            }
        }
    }

    /** The prefill logits of {@link #PROMPT} and the decode logits of {@link #DECODE}. */
    private float[][] run(EngineConfig.BackendMode backend, Format f, boolean offload) throws IOException {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).typeK(f.k).typeV(f.v).build();
        EngineConfig cfg = EngineConfig.builder().backend(backend).gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .kvOffload(offload).memory(memory).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            if (offload) {
                assertFalse("the device run keeps its KV caches on the device", m.kvOnHost());
            } else {
                assertTrue("the host run keeps its KV caches in host RAM", m.kvOnHost());
            }
            ForwardBackend fb = m.forwardBackend();
            float[][] out = new float[1 + DECODE.length][];
            out[0] = fb.prefill(PROMPT, 0).clone();
            for (int i = 0; i < DECODE.length; i++) {
                out[1 + i] = fb.decode(DECODE[i], PROMPT.length + i).clone();
            }
            fb.reset();
            return out;
        }
    }

    /** The greedy continuation of every format, which must agree between the device and the host path. */
    @Test
    public void openClGreedyTokensMatchTheHostPath() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(2).build()).writeTempFile();
        for (Format f : FORMATS) {
            int[] host;
            try {
                host = greedy(EngineConfig.BackendMode.OPENCL, f, false);
            } catch (RuntimeException e) {
                throw new IllegalStateException("host greedy of " + f + ": " + e, e);
            }
            int[] device;
            try {
                device = greedy(EngineConfig.BackendMode.OPENCL, f, true);
            } catch (RuntimeException e) {
                throw new IllegalStateException("device greedy of " + f + ": " + e, e);
            }
            assertArrayEquals("greedy tokens of " + f, host, device);
        }
    }

    private int[] greedy(EngineConfig.BackendMode backend, Format f, boolean offload) throws IOException {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).typeK(f.k).typeV(f.v).build();
        EngineConfig cfg = EngineConfig.builder().backend(backend).gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .kvOffload(offload).memory(memory).build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            int[] tokens = m.generate(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(SamplerConfig.greedyDecoding()).build(), null).tokens();
            m.forwardBackend().reset();
            return tokens;
        }
    }

    /** One K/V cache type pair of the format list. */
    private static final class Format {
        final KvDType k;
        final KvDType v;

        Format(KvDType k, KvDType v) {
            this.k = k;
            this.v = v;
        }

        @Override
        public String toString() {
            return k + "/" + v;
        }
    }
}
