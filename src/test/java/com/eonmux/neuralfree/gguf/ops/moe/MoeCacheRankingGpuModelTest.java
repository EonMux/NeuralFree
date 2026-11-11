package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.MoeExpertCache;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The expert-ranking cycle over the GPU model. With resident experts: the router's routed-id readbacks counted into
 * the profile on a real queue (decode and prefill steps, the pending reads drained at close) and the version 2 save
 * with the counted rows added to the saved heats. With the MoE expert cache on: the ranking next to the model, read
 * once by the model's ranking session, fills the cache before the first step (the first step has no miss), the
 * hosted pass counts its routed ids into the same profile, and the save merges them. Skipped without an OpenCL GPU.
 */
public class MoeCacheRankingGpuModelTest {

    private static final int N_PROMPT = 2;
    private static final int N_NEW = 3;

    @Test
    public void preloadRecordAndSaveRoundTripThroughTheGpuModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1).build();
        Path dir = Files.createTempDirectory("moe-ranking-gpu");
        try {
            Path model = writeModel(dir, config);
            Path ranking = dir.resolve(MoeCacheRanking.RANKING_FILE_NAME);
            MoeRankingFile.write(ranking, List.of(new RankedExpert(0, 0, 100)), config.nExpert());
            EngineConfig cfg = engineConfig(MoeCacheConfig.off(), ranking);
            int generated;
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
                generated = generate(m, cfg);
            }
            assertSavedProfile(ranking, config, 100L, generated);
        } finally {
            delete(dir);
        }
    }

    @Test
    public void theExpertCachePreloadsFromTheSessionAndTheHostedPassRecords() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1).build();
        Path dir = Files.createTempDirectory("moe-ranking-gpu-cache");
        try {
            Path model = writeModel(dir, config);
            Path ranking = dir.resolve(MoeCacheRanking.RANKING_FILE_NAME);
            List<RankedExpert> every = new ArrayList<>();
            for (int e = 0; e < config.nExpert(); e++) {
                every.add(new RankedExpert(0, e, 1));
            }
            MoeRankingFile.write(ranking, every, config.nExpert());
            EngineConfig cfg = engineConfig(MoeCacheConfig.parse("on"), ranking);
            int generated;
            String stats;
            try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
                MoeExpertCache cache = m.moeExpertCache();
                assertNotNull("the cache is configured for the MoE model", cache);
                generated = generate(m, cfg);
                stats = cache.stats();
            }
            assertTrue("every routed expert was preloaded, so no step misses: " + stats,
                    stats.matches("hits [1-9][0-9]*, misses 0 .*"));
            assertSavedProfile(ranking, config, config.nExpert(), generated);
        } finally {
            delete(dir);
        }
    }

    /** The saved ranking holds the merged heats plus one row per routed expert of every row the layer runs. */
    private static void assertSavedProfile(Path ranking, SyntheticModels.Config config, long mergedHeat,
                                           int generated) throws IOException {
        List<RankedExpert> saved = MoeRankingFile.read(ranking).entries();
        long routed = 0;
        for (RankedExpert e : saved) {
            assertTrue("no zero heat is written", e.heat() > 0);
            assertEquals("only layer 0 routes", 0, e.layer());
            routed += e.heat();
        }
        assertTrue("the chain ran at most the requested steps", generated <= N_NEW);
        assertEquals("every routed row counted once: the layer is the last one, so the prefill routes only the"
                        + " prompt row that is read, and every generated token but the last goes through it, "
                        + config.nExpertUsed() + " experts each",
                mergedHeat + Math.max(generated, 1) * config.nExpertUsed(), routed);
        assertEquals("the ranking stays hottest first", MoeRankingFile.sorted(saved), saved);
        assertEquals("the header names the model's expert count", config.nExpert(),
                MoeRankingFile.read(ranking).nExperts());
    }

    private static Path writeModel(Path dir, SyntheticModels.Config config) throws IOException {
        Path model = dir.resolve("model.gguf");
        SyntheticModels.qwen3next(config).writeTo(model);
        return model;
    }

    private static int generate(GGUFModelGPU m, EngineConfig cfg) {
        return GenerationLoop.forModel(m, cfg)
                .generate(prompt(), request(), (com.eonmux.neuralfree.gguf.TokenCallback) null)
                .tokenCount();
    }

    /** The engine settings with {@code moeCache} and the profile saved to {@code save}. */
    private static EngineConfig engineConfig(MoeCacheConfig moeCache, Path save) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .moeCache(moeCache.withRanking(
                        new MoeCacheRanking.Options(MoeCacheRanking.PreloadMode.AUTO, null, save)))
                .build();
    }

    private static int[] prompt() {
        return new int[]{4, 9};
    }

    private static GenerationRequest request() {
        return GenerationRequest.builder().maxNewTokens(N_NEW)
                .sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static void delete(Path dir) throws IOException {
        try (java.util.stream.Stream<Path> paths = Files.walk(dir)) {
            paths.sorted(java.util.Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }
}
