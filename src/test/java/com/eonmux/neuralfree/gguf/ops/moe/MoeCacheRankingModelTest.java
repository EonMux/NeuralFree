package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationLoop;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.MoeCacheConfig;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The expert-ranking cycle end to end over a loaded model: the ranking beside the synthetic qwen3next file
 * preloads at open, a real generation's routed rows count through the MoE blocks, and the close of the model
 * writes the version 2 ranking with the counted rows added to the preloaded heats — the
 * {@code GGML_CUDA_MOE_CACHE_PROFILE_SAVE} cycle of the reference, with the preload reading a shipped ranking.
 */
public class MoeCacheRankingModelTest {

    private static final int N_PROMPT = 2;
    private static final int N_NEW = 3;

    @Test
    public void preloadRecordAndSaveRoundTripThroughARealModel() throws IOException {
        SyntheticModels.Config config = SyntheticModels.Config.qwen3next().toBuilder().nLayer(1).build();
        Path dir = Files.createTempDirectory("moe-ranking-model");
        Path model = dir.resolve("model.gguf");
        SyntheticModels.qwen3next(config).writeTo(model);
        Path ranking = dir.resolve(MoeCacheRanking.RANKING_FILE_NAME);
        MoeRankingFile.write(ranking, List.of(new RankedExpert(0, 0, 100)), config.nExpert());
        EngineConfig cfg = engineConfig(ranking);
        try {
            int generated;
            try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
                generated = GenerationLoop.forModel(m, cfg)
                        .generate(prompt(), request(), (com.eonmux.neuralfree.gguf.TokenCallback) null)
                        .tokenCount();
            }
            List<RankedExpert> saved = MoeRankingFile.read(ranking).entries();
            long routed = 0;
            long preloaded = 0;
            for (RankedExpert e : saved) {
                assertTrue("no negative heat", e.heat() > 0);
                if (e.layer() == 0 && e.expert() == 0) {
                    preloaded = e.heat();
                }
                routed += e.heat();
            }
            assertTrue("the preloaded heat is kept and added to", preloaded >= 100);
            assertTrue("the chain ran at most the requested steps", generated <= N_NEW);
            assertEquals("every forwarded token's routed rows counted once: the prefill and every generated token"
                            + " but the last one goes through the layer, " + config.nExpertUsed() + " experts each",
                    100L + (N_PROMPT + Math.max(generated, 1) - 1) * config.nExpertUsed(), routed);
            assertEquals("the ranking stays hottest first", MoeRankingFile.sorted(saved), saved);
        } finally {
            Files.walk(dir)
                    .sorted(java.util.Comparator.reverseOrder())
                    .forEach(p -> p.toFile().delete());
        }
    }

    /** The engine settings with the profile saved to {@code save} ({@code neuralfree.moe.cache.ranking.save}). */
    private static EngineConfig engineConfig(Path save) {
        return EngineConfig.builder().memory(MemoryConfig.builder().nCtx(64).ubatch(8)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .moeCache(MoeCacheConfig.off().withRanking(
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
}
