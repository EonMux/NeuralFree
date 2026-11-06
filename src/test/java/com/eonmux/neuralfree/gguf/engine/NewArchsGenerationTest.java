package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The lfm3 hybrid and the limite dense model generate through the contiguous generation loop over their synthetic
 * GGUFs: lfm3 mixes short convolutions with gated sliding-window attention and a tied output head; limite runs
 * value embeddings, the exclusive self attention, MUDD dense connections, learned residual scales and the sigmoid
 * logit cap. Greedy decoding is deterministic, so two runs over one model agree token for token.
 */
public class NewArchsGenerationTest {

    private static EngineConfig config() {
        return EngineConfig.builder()
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static GenerationRequest request(int nNew) {
        return GenerationRequest.builder().maxNewTokens(nNew).sampler(SamplerConfig.greedyDecoding()).build();
    }

    private static int[] prompt(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = 4 + (i * 13) % 250;
        }
        return t;
    }

    @Test
    public void lfm3HybridGeneratesDeterministically() throws IOException {
        Path model = SyntheticModels.lfm3().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            GenerationLoop loop = GenerationLoop.forModel(m, config());
            GenerationResult first = loop.generate(prompt(6), request(6), null);
            GenerationResult second = loop.generate(prompt(6), request(6), null);
            assertEquals(6, first.tokens().length);
            assertArrayEquals(first.tokens(), second.tokens());
        }
    }

    @Test
    public void limiteDenseModelGeneratesDeterministically() throws IOException {
        Path model = SyntheticModels.limite().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            GenerationLoop loop = GenerationLoop.forModel(m, config());
            GenerationResult first = loop.generate(prompt(6), request(6), null);
            GenerationResult second = loop.generate(prompt(6), request(6), null);
            assertEquals(6, first.tokens().length);
            assertArrayEquals(first.tokens(), second.tokens());
        }
    }

    @Test
    public void mistralLarge4GeneratesDeterministically() throws IOException {
        Path model = SyntheticModels.mistralLarge4().writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            GenerationLoop loop = GenerationLoop.forModel(m, config());
            GenerationResult first = loop.generate(prompt(6), request(6), null);
            GenerationResult second = loop.generate(prompt(6), request(6), null);
            assertEquals(6, first.tokens().length);
            assertArrayEquals(first.tokens(), second.tokens());
        }
    }

    @Test
    public void mistralLarge4WithMixedRopeGeneratesDeterministically() throws IOException {
        Path model = SyntheticModels.mistralLarge4NoPe(SyntheticModels.Config.mistralLarge4()).writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            GenerationLoop loop = GenerationLoop.forModel(m, config());
            GenerationResult first = loop.generate(prompt(6), request(6), null);
            GenerationResult second = loop.generate(prompt(6), request(6), null);
            assertEquals(6, first.tokens().length);
            assertArrayEquals(first.tokens(), second.tokens());
        }
    }

    @Test
    public void lfm3D1RunsTheWholePromptWithoutMemory() throws IOException {
        // the centred short convolution runs an odd tap count (lfm2's decision convention, l_cache 3)
        SyntheticModels.Config c = SyntheticModels.Config.lfm3().toBuilder().ssmConv(3).build();
        Path model = SyntheticModels.lfm3(c, false).writeTempFile();
        try (GGUFModel m = new GGUFModel(model.toString(), config())) {
            GenerationLoop loop = GenerationLoop.forModel(m, config());
            GenerationResult r = loop.generate(prompt(6), request(1), null);
            assertEquals(1, r.tokens().length);
        }
    }
}
