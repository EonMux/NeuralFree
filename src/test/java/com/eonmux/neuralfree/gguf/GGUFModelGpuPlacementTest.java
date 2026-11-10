package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * A real GPU load through the placement fit wiring of {@link GGUFModelGPU}: the synthetic qwen3next model (GDN +
 * attention + routed experts) opens on the OpenCL backend with every weight resident, plans its training context and
 * generates. The fitter's streaming placements are covered by {@code PlacementFitterTest}; this covers the load path
 * around them.
 */
public class GGUFModelGpuPlacementTest {

    @Test
    public void moeModelLoadsResidentAndGenerates() throws Exception {
        Path file = SyntheticModels.qwen3next().writeTempFile();
        EngineConfig cfg = EngineConfig.fromSystemProperties().toBuilder()
                .backend(EngineConfig.BackendMode.OPENCL).build();
        GGUFModelGPU model = null;
        try {
            model = new GGUFModelGPU(file.toString(), null, cfg);

            assertEquals("qwen3next", model.architecture());
            assertEquals(4, model.layerCount());
            assertEquals(SyntheticModels.Config.qwen3next().nCtxTrain(), model.memoryPlan().nCtx());
            assertNotNull(model.gpuWeights().find("blk.0.ffn_up_exps.weight"));
            assertNotNull(model.gpuWeights().find("blk.3.attn_q.weight"));

            // greedy and past the end token: the default sampler draws with a fresh seed, and a random synthetic
            // model can draw its end token first
            GenerationResult result = model.generate(model.tokenizer().encode("hello"),
                    GenerationRequest.builder().maxNewTokens(2).sampler(SamplerConfig.greedyDecoding())
                            .ignoreEos(true).build(), null);
            assertEquals(2, result.tokens().length);
            assertTrue(model.kvCacheContext() > 0);
        } finally {
            try {
                if (model != null) {
                    model.close();
                }
            } finally {
                Files.deleteIfExists(file);
            }
        }
    }
}
