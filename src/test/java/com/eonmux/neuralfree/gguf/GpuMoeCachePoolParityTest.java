package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end parity of the pooled expert cache over two OpenCL devices: a hosted MoE model generates the logits and
 * greedy tokens of the one-device load with the same cache, whichever GPU each layer's block runs on, and the two
 * GPUs' caches divide the routed layers between them. Two span layouts are run, so at least one puts a layer's block
 * on the GPU whose cache serves it rather than on the GPU that runs its attention.
 */
public class GpuMoeCachePoolParityTest {

    private static final int LAYERS = 4;
    private static final float TOL = 1e-3f;
    private static final int[] PROMPT = {2, 7, 3, 9, 4, 11};

    private static Path writeModel() throws Exception {
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(LAYERS).build();
        return SyntheticModels.qwen3next(config).writeTempFile();
    }

    private static EngineConfig reference() {
        return SplitFixtures.oneDevice("0").moeCache(MoeCacheConfig.parse("on")).build();
    }

    private static EngineConfig pooled(double[] split) {
        return SplitFixtures.oneDevice("0").splitMode(EngineConfig.SplitMode.LAYER)
                .addGpuDevice("1").tensorSplit(split)
                .moeCache(MoeCacheConfig.parse("on")).moeCachePool(true).build();
    }

    @Test
    public void thePooledCacheMatchesTheOneDeviceLoadAndDividesTheLayers() throws Exception {
        SplitFixtures.assumeTwoDevices();
        Path model = writeModel();
        try {
            float[][] expected = SplitFixtures.logits(model, reference());
            int[] expectedTokens = generate(model, reference());

            boolean anyMoved = false;
            double[][] splits = {{4, 1}, {1, 4}};
            for (double[] split : splits) {
                float[][] got = SplitFixtures.logits(model, pooled(split));
                for (int i = 0; i < expected.length; i++) {
                    assertArrayEquals("pooled cache logits diverge at row " + i + " for tensor split "
                            + split[0] + ":" + split[1], expected[i], got[i], TOL);
                }
                assertArrayEquals("greedy tokens diverge with the pooled cache", expectedTokens,
                        generate(model, pooled(split)));
                anyMoved |= layersMoveOffTheirSpan(model, split);
            }
            assertTrue("the pool never moved a layer's block to the GPU that caches it", anyMoved);
        } finally {
            Files.deleteIfExists(model);
        }
    }

    /**
     * Opens the pooled load and reports whether some routed layer's cache serves it from another GPU than the one
     * running its attention, checking on the way that the two caches partition the routed layers.
     */
    private static boolean layersMoveOffTheirSpan(Path model, double[] split) throws Exception {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, pooled(split))) {
            Set<Integer> served = new LinkedHashSet<>();
            boolean moved = false;
            for (int d = 0; d < 2; d++) {
                MoeExpertCache cache = m.moeExpertCacheOf(d);
                int first = m.firstLayerOf(d);
                int end = first + m.layerCountOf(d);
                if (cache == null) {
                    continue;
                }
                for (MoeExpertCache.LayerPlan plan : cache.layerPlans()) {
                    assertTrue("layer " + plan + " is served twice", served.add(plan.weightLayer()));
                    moved |= plan.weightLayer() < first || plan.weightLayer() >= end;
                }
            }
            assertEquals("the caches did not partition the " + LAYERS + " routed layers", LAYERS, served.size());
            return moved;
        }
    }

    private static int[] generate(Path model, EngineConfig cfg) throws Exception {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            return m.generate(PROMPT, request, null).tokens();
        }
    }
}
