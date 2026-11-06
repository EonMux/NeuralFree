package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
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
import static org.junit.Assert.assertEquals;

/**
 * The DeepSeek MLA latent KV cache formats ({@code fp8_ds_mla}, {@code nvfp4_ds_mla}) end to end on the device caches:
 * a synthetic absorbed-MLA model with the 576-value latent rows of vLLM's kernels ({@code kv_lora_rank} 512 and 64
 * RoPE values) keeps its latent rows and their tile/group scales in device memory and produces the logits and greedy
 * tokens of the CPU engine running the same format. The reference is the CPU path because a host KV cache of an MLA
 * layer is not a placement the engine offers.
 *
 * <p>Every test runs the device store, the device MLA flash attention (whose V side reads the K rows) and the
 * side-plane pages of both latent formats on the machine's OpenCL, CUDA and Vulkan devices.</p>
 */
public class KvVendorMlaGpuTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final int[] DECODE = {77, 3};
    private static final int GREEDY_TOKENS = 3;
    /** Logit tolerance of a latent cache on the device against the CPU engine of the same format. */
    private static final float TOL = 2e-2f;

    /** The two latent row formats, each filling both cache sides. */
    private static final List<KvDType> FORMATS = Arrays.asList(KvDType.FP8_DS_MLA, KvDType.NVFP4_DS_MLA);

    private Path model;

    @After
    public void closeModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void openClLatentCachesMatchTheCpuPath() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertLatentParity(EngineConfig.BackendMode.OPENCL);
    }

    @Test
    public void cudaLatentCachesMatchTheCpuPath() throws IOException {
        GpuTestUtil.assumeCudaAvailable();
        assertLatentParity(EngineConfig.BackendMode.CUDA);
    }

    @Test
    public void vulkanLatentCachesMatchTheCpuPath() throws IOException {
        GpuTestUtil.assumeVulkanAvailable();
        assertLatentParity(EngineConfig.BackendMode.VULKAN);
    }

    private void assertLatentParity(EngineConfig.BackendMode backend) throws IOException {
        model = latentModel();
        for (KvDType type : FORMATS) {
            float[][] cpu = runCpu(type);
            float[][] device = runGpu(backend, type);
            for (int i = 0; i < cpu.length; i++) {
                for (int j = 0; j < cpu[i].length; j++) {
                    if (!Float.isFinite(cpu[i][j]) || !Float.isFinite(device[i][j])) {
                        throw new AssertionError(backend + " " + type + ": step " + i + " logit " + j
                                + " is not finite (cpu " + cpu[i][j] + ", device " + device[i][j] + ")");
                    }
                    float diff = Math.abs(cpu[i][j] - device[i][j]);
                    if (diff > TOL) {
                        throw new AssertionError(backend + " " + type + ": step " + i + " logit " + j
                                + " differs from the CPU path by " + diff + " > " + TOL);
                    }
                }
            }
        }
    }

    @Test
    public void latentPrefillsOfEveryLengthMatchTheCpuPath() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = SyntheticModels.deepseek2(new SyntheticModels.Config.Builder().nLayer(1).nHead(4).nHeadKv(1)
                .headDim(80).nRot(64).kvLora(512).nFf(128).nExpert(4).nExpertUsed(2).nFfExp(48).normEps(1e-6f)
                .build()).writeTempFile();
        for (KvDType type : FORMATS) {
            for (int n : new int[]{1, 3, PROMPT.length}) {
                float[] cpu;
                try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig(type))) {
                    cpu = m.forwardBackend().prefill(Arrays.copyOf(PROMPT, n), 0).clone();
                }
                float[] device;
                try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                        gpuConfig(EngineConfig.BackendMode.OPENCL, type))) {
                    device = m.forwardBackend().prefill(Arrays.copyOf(PROMPT, n), 0).clone();
                }
                assertArrayEquals(type + " prefill of " + n + " token(s)", cpu, device, TOL);
            }
        }
    }

    @Test
    public void openClPagedLatentCachesMatchTheCpuPath() throws Exception {
        GpuTestUtil.assumeOpenCLAvailable();
        model = latentModel();
        for (KvDType type : FORMATS) {
            MemoryConfig paged = memory(type).toBuilder().parallel(2).kvPageSize(8).build();
            assertArrayEquals(type + " paged greedy tokens", pagedGreedy(EngineConfig.BackendMode.CPU, paged),
                    pagedGreedy(EngineConfig.BackendMode.OPENCL, paged));
        }
    }

    /** The greedy continuation of a paged cache, which the batch engine fills page by page. */
    private int[] pagedGreedy(EngineConfig.BackendMode backend, MemoryConfig memory) throws Exception {
        EngineConfig cfg = gpuConfig(backend, memory);
        try (GGUFModel m = new GGUFModel(model.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(m, cfg)) {
            BatchSequence seq = engine.submit(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                    .sampler(SamplerConfig.greedyDecoding()).build(), (TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, seq.state());
            return seq.result().tokens();
        }
    }

    @Test
    public void openClLatentGreedyTokensMatchTheCpuPath() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        model = latentModel();
        for (KvDType type : FORMATS) {
            assertArrayEquals("greedy tokens of " + type, greedyCpu(type),
                    greedyGpu(EngineConfig.BackendMode.OPENCL, type));
        }
    }

    /** A deepseek2 model whose latent cache rows are the 576 values of vLLM's MLA kernels. */
    private static Path latentModel() throws IOException {
        return SyntheticModels.deepseek2(new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(1).headDim(80)
                .nRot(64).kvLora(512).nFf(128).nExpert(4).nExpertUsed(2).nFfExp(48).normEps(1e-6f).build())
                .writeTempFile();
    }

    private static float[][] logits(GGUFModel m) throws IOException {
        try {
            return logits(m.forwardBackend());
        } finally {
            m.close();
        }
    }

    private static float[][] logits(GGUFModelGPU m) throws IOException {
        try {
            return logits(m.forwardBackend());
        } finally {
            m.close();
        }
    }

    private static MemoryConfig memory(KvDType type) {
        return MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(4).typeK(type).typeV(type).build();
    }

    private static EngineConfig cpuConfig(KvDType type) {
        return cpuConfig(memory(type));
    }

    private static EngineConfig cpuConfig(MemoryConfig memory) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory).build();
    }

    private static EngineConfig gpuConfig(EngineConfig.BackendMode backend, KvDType type) {
        return gpuConfig(backend, memory(type));
    }

    private static EngineConfig gpuConfig(EngineConfig.BackendMode backend, MemoryConfig memory) {
        return EngineConfig.builder().backend(backend).gpuLayers(EngineConfig.GPU_LAYERS_ALL).memory(memory).build();
    }

    /** The prefill logits of {@link #PROMPT} and the decode logits of {@link #DECODE} of the CPU engine. */
    private float[][] runCpu(KvDType type) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig(type))) {
            return logits(m.forwardBackend());
        }
    }

    /** The same rows on the device caches of {@code backend}. */
    private float[][] runGpu(EngineConfig.BackendMode backend, KvDType type) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, gpuConfig(backend, type))) {
            return logits(m.forwardBackend());
        }
    }

    private static float[][] logits(ForwardBackend fb) {
        float[][] out = new float[1 + DECODE.length][];
        out[0] = fb.prefill(PROMPT, 0).clone();
        for (int i = 0; i < DECODE.length; i++) {
            out[1 + i] = fb.decode(DECODE[i], PROMPT.length + i).clone();
        }
        fb.reset();
        return out;
    }

    private int[] greedyCpu(KvDType type) throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig(type))) {
            return greedy(m);
        }
    }

    private int[] greedyGpu(EngineConfig.BackendMode backend, KvDType type) throws IOException {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, gpuConfig(backend, type))) {
            return greedy(m);
        }
    }

    private static int[] greedy(GGUFModel m) {
        int[] tokens = m.generate(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.greedyDecoding()).build(), null).tokens();
        m.forwardBackend().reset();
        return tokens;
    }

    private static int[] greedy(GGUFModelGPU m) {
        int[] tokens = m.generate(PROMPT, GenerationRequest.builder().maxNewTokens(GREEDY_TOKENS)
                .sampler(SamplerConfig.greedyDecoding()).build(), null).tokens();
        m.forwardBackend().reset();
        return tokens;
    }
}
