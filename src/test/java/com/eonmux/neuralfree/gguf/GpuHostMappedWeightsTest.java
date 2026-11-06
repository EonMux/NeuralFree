package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The UVA zero-copy tier on a CUDA device: a synthetic llama whose dense FFN weights live in host-mapped memory
 * decodes the same logits as the same model with every weight in VRAM, so a kernel reads the mapped weights
 * exactly as it reads resident ones. Skipped when CUDA is unavailable.
 */
public class GpuHostMappedWeightsTest {

    private static final int[] TOKENS = {5, 9, 17, 23};
    private static final int N_CTX = 64;
    private static final float TOL = 1e-5f;

    @Test
    public void hostMappedWeightsDecodeTheSameLogitsAsResidentOnes() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            float[][] resident = decodeAll(model, offload(false));
            float[][] mapped = decodeAll(model, offload(true));
            for (int step = 0; step < TOKENS.length; step++) {
                assertArrayEquals("logits of decode step " + step + " with host-mapped weights", resident[step],
                        mapped[step], TOL);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void theLoadReportsTheHostMappedTierOnlyWhenConfigured() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        Path model = SyntheticModels.llama().writeTempFile();
        try (EngineModel resident = LLMModelFactory.open(model.toString(), offload(false))) {
            assertNull(((GGUFModelGPU) resident).hostMappedWeights());
        }
        try (EngineModel mapped = LLMModelFactory.open(model.toString(), offload(true))) {
            GGUFModelGPU m = (GGUFModelGPU) mapped;
            assertNotNull(m.hostMappedWeights());
            assertEquals(2, m.hostMappedWeights().layerCount());
            assertTrue(m.hostMappedWeights().hostBytes() > 0);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void hostMappedExpertWeightsDecodeTheSameLogitsAsResidentOnes() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
        Path model = SyntheticModels.qwen3moe().writeTempFile();
        try {
            float[][] resident = decodeAll(model, expertOffload(false));
            float[][] mapped = decodeAll(model, expertOffload(true));
            for (int step = 0; step < TOKENS.length; step++) {
                assertArrayEquals("expert logits of decode step " + step + " with host-mapped weights",
                        resident[step], mapped[step], TOL);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static EngineConfig expertOffload(boolean enabled) {
        EngineConfig.Builder b = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(MemoryConfig.builder().nCtx(N_CTX).typeK(KvDType.F32).typeV(KvDType.F32).build());
        if (enabled) {
            b.cpuOffloadMb(1).cpuOffloadParams(Collections.singletonList("ffn_gate_exps"));
        }
        return b.build();
    }

    private static EngineConfig offload(boolean enabled) {
        MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).typeK(KvDType.F32).typeV(KvDType.F32).build();
        EngineConfig.Builder b = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA)
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(memory);
        if (enabled) {
            b.cpuOffloadMb(1).cpuOffloadParams(Collections.singletonList("ffn_gate.weight"));
        }
        return b.build();
    }

    private static float[][] decodeAll(Path model, EngineConfig cfg) throws IOException {
        try (EngineModel gpu = LLMModelFactory.open(model.toString(), cfg)) {
            float[][] out = new float[TOKENS.length][];
            for (int i = 0; i < TOKENS.length; i++) {
                out[i] = gpu.forwardBackend().decode(TOKENS[i], i);
            }
            return out;
        }
    }
}
