package com.eonmux.neuralfree.gguf.gpu.forward;

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

import static org.junit.Assert.assertArrayEquals;

/**
 * The decode replay on an OpenCL device: a synthetic llama decodes twelve tokens with the replay
 * enabled and with it disabled and must produce the same logits. On a device without
 * {@code cl_khr_command_buffer} the queue reports no capture support and both runs execute the
 * trunk directly; on a device with it the replayed run submits the captured command buffer instead,
 * patched per step where {@code cl_khr_command_buffer_mutable_dispatch} allows it.
 */
public class GpuDecodeReplayOpenClTest {

    private static final int[] TOKENS = {5, 9, 17, 23, 31, 2, 7, 13, 19, 29, 3, 11};
    private static final int N_CTX = 64;
    private static final float TOL = 1e-5f;

    @Test
    public void replayMatchesDirectDecodeOnOpenCl() throws IOException {
        Assume.assumeTrue("OpenCL not available", GpuTestUtil.isOpenCLAvailable());
        Path model = SyntheticModels.llama().writeTempFile();
        try {
            float[][] replayed = decodeAll(model, "true");
            float[][] direct = decodeAll(model, "false");
            for (int step = 0; step < TOKENS.length; step++) {
                assertArrayEquals("logits of decode step " + step + " with the decode replay", direct[step],
                        replayed[step], TOL);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    private static float[][] decodeAll(Path model, String replay) throws IOException {
        System.setProperty(GpuDecodeReplay.PROPERTY, replay);
        try {
            MemoryConfig memory = MemoryConfig.builder().nCtx(N_CTX).typeK(KvDType.F32).typeV(KvDType.F32).build();
            EngineConfig cfg = EngineConfig.builder()
                    .backend(EngineConfig.BackendMode.OPENCL)
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
            System.clearProperty(GpuDecodeReplay.PROPERTY);
        }
    }
}
