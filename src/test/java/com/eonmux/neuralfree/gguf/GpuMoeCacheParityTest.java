package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.GenerationResult;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuForwardBackend;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertPrefetch;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * End-to-end parity of a GPU load with the MoE expert cache on the real OpenCL backend: the same synthetic
 * qwen3next model generates the same greedy tokens with the cache on and off, the cache serves routed experts
 * (hits in its stats) and keeps its fills visible across steps; a prefill whose experts the prefetch copies into
 * zones lent by the cache, reading the cached ones in place, generates the same tokens as the host-computed misses.
 * The GPU cases are skipped when no OpenCL GPU is available; the resolution of the cache settings
 * ({@code neuralfree.moe.cache.mb} alone keeps the prefetch settings) needs none.
 */
public class GpuMoeCacheParityTest {

    private static Path writeModel() throws Exception {
        return writeModel(1);
    }

    private static Path writeModel(int layers) throws Exception {
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(layers).build();
        return SyntheticModels.qwen3next(config).writeTempFile();
    }

    private static EngineConfig config(MoeCacheConfig moeCache) {
        return EngineConfig.builder()
                .moeCache(moeCache)
                .samplerDefaults(SamplerConfig.defaults().toBuilder().temperature(0f).seed(1L).build())
                .build();
    }

    private static GenerationResult generate(Path model, EngineConfig cfg, int[] prompt) throws Exception {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            return m.generate(prompt, request, null);
        }
    }

    /** The cache stats of a generation with the cache on: the cache must have served routed experts. */
    private static String cacheStats(Path model, int[] prompt) throws Exception {
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, config(MoeCacheConfig.parse("on")))) {
            MoeExpertCache cache = m.moeExpertCache();
            assertNotNull("the cache is not configured for a MoE model", cache);
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                    .sampler(config(MoeCacheConfig.parse("on")).legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            m.generate(prompt, request, null);
            return cache.stats();
        }
    }

    @Test
    public void cacheOnAndOffGenerateTheSameTokens() throws Exception {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = writeModel();
        try {
            int[] prompt = {2, 1, 1, 1, 1, 1};
            int[] off = generate(model, config(MoeCacheConfig.off()), prompt).tokens();
            int[] on = generate(model, config(MoeCacheConfig.parse("on")), prompt).tokens();

            assertTrue("no tokens generated with the cache off", off.length >= 8);
            assertArrayEquals("greedy tokens diverge with the MoE expert cache on", off, on);

            String stats = cacheStats(model, prompt);
            assertTrue("the cache served no expert (stats: " + stats + ")", stats.contains("hits 0, misses 0")
                    ? false : !stats.startsWith("hits 0, misses 0 (0.0%)"));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void aPrefetchedPrefillGeneratesTheSameTokensAsHostMisses() throws Exception {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = writeModel(2);
        try {
            // 12 tokens x 2 routed experts over 4 experts: every prefill step is prefill-sized
            int[] prompt = {2, 7, 3, 9, 4, 11, 5, 13, 6, 8, 10, 12};
            int[] off = generate(model, config(MoeCacheConfig.off()), prompt).tokens();
            int[] hosted = generate(model, config(MoeCacheConfig.parse("on").withPrefetchZones(0)), prompt)
                    .tokens();
            int[] prefetched = generate(model, config(MoeCacheConfig.parse("on").withPrefetchZones(1)), prompt)
                    .tokens();

            assertTrue("no tokens generated with the cache off", off.length >= 8);
            assertArrayEquals("greedy tokens diverge with the misses on the host", off, hosted);
            assertArrayEquals("greedy tokens diverge with the prefill prefetch", off, prefetched);

            EngineConfig cfg = config(MoeCacheConfig.parse("on").withPrefetchZones(1));
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
                MoeExpertPrefetch pool = m.moeExpertPrefetch();
                assertNotNull("the prefetch is on by default with the cache", pool);
                GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                        .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
                int[] first = m.generate(prompt, request, null).tokens();
                assertArrayEquals(off, first);
                m.generate(new int[]{3, 12, 4, 10, 5, 8, 6, 13, 7, 11, 9, 2}, request, null);
                String stats = pool.stats();
                assertTrue("no prefill step was prefetched (" + stats + ")", !stats.startsWith("prefill steps 0,"));
                assertTrue("the second prefill read no cached expert in place (" + stats + ")",
                        pool.residentReads() > 0);
            }
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                    config(MoeCacheConfig.parse("on").withPrefetchZones(0)))) {
                assertTrue("prefetch 0 opens no pool", m.moeExpertPrefetch() == null);
            }
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void aCachedDecodeStepFallsBackFromTheReplayAndKeepsTheTokens() throws Exception {
        GpuTestUtil.assumeOpenCLAvailable();
        Path model = writeModel();
        try {
            int[] prompt = {2, 1, 1, 1, 1, 1};
            System.setProperty("neuralfree.gpu.decodeReplay", "false");
            int[] direct;
            try {
                direct = generate(model, config(MoeCacheConfig.parse("on")), prompt).tokens();
            } finally {
                System.clearProperty("neuralfree.gpu.decodeReplay");
            }

            // capture the very first eligible decode step: the cached-MoE trunk must be refused before capture
            System.setProperty("neuralfree.gpu.graphWarmupSteps", "0");
            int[] replayed;
            String replayStats;
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, config(MoeCacheConfig.parse("on")))) {
                EngineConfig cfg = config(MoeCacheConfig.parse("on"));
                GenerationRequest request = GenerationRequest.builder().maxNewTokens(8)
                        .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
                replayed = m.generate(prompt, request, null).tokens();
                replayStats = ((GpuForwardBackend) m.forwardBackend()).decodeReplayStats();
            } finally {
                System.clearProperty("neuralfree.gpu.graphWarmupSteps");
            }

            assertTrue("no tokens generated with the cache on", direct.length >= 8);
            assertArrayEquals("the replay aborted a cached-MoE step differently", direct, replayed);
            assertTrue("the cached-MoE trunk was captured instead of run directly: " + replayStats,
                    replayStats.contains("warmup steps"));
        } finally {
            Files.deleteIfExists(model);
        }
    }

    @Test
    public void theCacheBudgetPropertyAloneKeepsThePrefetchSettings() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_MOE_CACHE_MB, "512");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_PREFETCH, "3");
        p.setProperty(MoeCacheConfig.PROP_MOE_CACHE_HOSTED_ROWS, "64");
        MoeCacheConfig resolved = GGUFModelGPU.resolveMoeCache(EngineConfig.fromProperties(p));

        assertTrue(resolved.enabled());
        assertEquals(512L, resolved.budgetMiB());
        assertEquals(3, resolved.prefetchZones());
        assertEquals(64, resolved.hostedMaxRows());
    }
}
