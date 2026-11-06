package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.junit.Assert.assertArrayEquals;

/**
 * End-to-end parity of the fused decode kernels: a synthetic llama decodes the same tokens with
 * {@code -Dneuralfree.gpu.fused=off} and {@code =on} and must produce identical logits on every available GPU
 * backend. The fused run also proves that each structurally eligible layer found its kernel, because the "on" mode
 * refuses a missing one by name instead of falling back.
 */
@RunWith(Parameterized.class)
public class GpuFusedDecodeParityGpuTest {

    private static final int[] TOKENS = {5, 9, 17, 23, 31};
    private static final int N_CTX = 64;
    private static final float TOL = 1e-5f;

    @Parameterized.Parameters(name = "{0}")
    public static List<Object[]> backends() {
        return Arrays.asList(new Object[]{"opencl"}, new Object[]{"cuda"}, new Object[]{"vulkan"});
    }

    private final String api;

    public GpuFusedDecodeParityGpuTest(String api) {
        this.api = api;
    }

    @Test
    public void fusedDecodeMatchesUnfusedLogits() throws IOException {
        EngineConfig.BackendMode backend = backend();
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            float[][] unfused = decodeAll(model, backend, FusedMode.OFF);
            float[][] fused = decodeAll(model, backend, FusedMode.ON);
            for (int step = 0; step < TOKENS.length; step++) {
                assertArrayEquals("logits of decode step " + step + " on " + api, unfused[step], fused[step], TOL);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private EngineConfig.BackendMode backend() {
        if (api.equals("cuda")) {
            GpuTestUtil.assumeCudaAvailable();
            return EngineConfig.BackendMode.CUDA;
        }
        if (api.equals("vulkan")) {
            GpuTestUtil.assumeVulkanAvailable();
            return EngineConfig.BackendMode.VULKAN;
        }
        GpuTestUtil.assumeOpenCLAvailable();
        return EngineConfig.BackendMode.OPENCL;
    }

    private static float[][] decodeAll(Path model, EngineConfig.BackendMode backend, FusedMode mode)
            throws IOException {
        String previous = System.getProperty(FusedMode.PROPERTY);
        System.setProperty(FusedMode.PROPERTY, mode.name().toLowerCase(Locale.ROOT));
        try {
            MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).typeK(KvDType.F32).typeV(KvDType.F32).build();
            EngineConfig cfg = EngineConfig.builder()
                    .backend(backend)
                    .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                    .memory(memory)
                    .build();
            try (EngineModel gpu = LLMModelFactory.open(model.toString(), cfg)) {
                float[][] out = new float[TOKENS.length][];
                for (int i = 0; i < TOKENS.length; i++) {
                    out[i] = gpu.forwardBackend().decode(TOKENS[i], i);
                }
                return out;
            }
        } finally {
            if (previous == null) {
                System.clearProperty(FusedMode.PROPERTY);
            } else {
                System.setProperty(FusedMode.PROPERTY, previous);
            }
        }
    }
}
