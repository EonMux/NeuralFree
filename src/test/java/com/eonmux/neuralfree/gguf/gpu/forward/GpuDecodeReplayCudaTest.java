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
 * The decode replay on a CUDA device: a synthetic llama decodes twelve tokens through a captured graph and
 * produces the same logits as the same decode with the replay disabled. The step's per-token scalar
 * arguments (key count, query position) change every step, so a graph that did not pick them up through
 * {@code cuGraphExecKernelNodeSetParams} would diverge. The first steps run the capture's verification and
 * from the sixth on the trunk replays from the patch plan without its body, so the compared steps cover
 * both.
 */
public class GpuDecodeReplayCudaTest {

    private static final int[] TOKENS = {5, 9, 17, 23, 31, 2, 7, 13, 19, 29, 3, 11};
    private static final int N_CTX = 64;
    private static final float TOL = 1e-5f;

    @Test
    public void replayMatchesDirectDecodeOnCuda() throws IOException {
        Assume.assumeTrue("CUDA not available", GpuTestUtil.isCudaAvailable());
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
                    .backend(EngineConfig.BackendMode.CUDA)
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
