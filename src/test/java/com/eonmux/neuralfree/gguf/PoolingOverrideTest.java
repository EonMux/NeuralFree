package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The pooling type override ({@code neuralfree.pooling}, llama.cpp {@code --pooling}): the resolved spec and the
 * opened model pool by the override instead of the GGUF's {@code bert.pooling_type}, {@code rank} refuses a model
 * that stores no classifier head by name, and every entry point ({@code encode}, {@code encodeTokens},
 * {@code classify}, the spec inspection) honours it.
 */
public class PoolingOverrideTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    /** A bert whose {@code pooling_type} the test replaces. */
    private static com.eonmux.neuralfree.gguf.testutil.GgufTestWriter bertPooling(int pooling) {
        com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w = SyntheticModels.bert();
        String key = w.archKey("pooling_type");
        return w.remove(key).u32(key, pooling);
    }

    private static int[] tokens(int seed) {
        int[] t = new int[5];
        for (int i = 0; i < t.length; i++) {
            t[i] = 8 + (seed * 5 + i * 3) % 180;
        }
        return t;
    }

    @Test
    public void theOverrideReplacesTheGgufPooling() throws IOException {
        Path model = bertPooling(SyntheticModels.POOLING_MEAN).writeTempFile();
        EngineConfig mean = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
        EngineConfig cls = mean.toBuilder().pooling(PoolingKind.CLS).build();
        try (GGUFModel meanModel = new GGUFModel(model.toString(), mean);
             GGUFModel m = new GGUFModel(model.toString(), cls)) {
            assertEquals(LLMModel.PoolingType.MEAN, meanModel.poolingType());
            assertEquals(LLMModel.PoolingType.CLS, m.poolingType());
            assertEquals("the spec carries the override for every consumer",
                    PoolingKind.CLS, m.spec().head().pooling());
            float[] clsRow = m.encode(tokens(3));
            assertEquals("the width does not change", meanModel.encode(tokens(3)).length, clsRow.length);
            assertNotEquals("CLS pools the first row, MEAN the average",
                    meanModel.encode(tokens(3)), clsRow);
        }
    }

    @Test
    public void lastAndNoneOverridesServeTheirEntryPoints() throws IOException {
        Path model = bertPooling(SyntheticModels.POOLING_MEAN).writeTempFile();
        EngineConfig cpu = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
        try (GGUFModel last = new GGUFModel(model.toString(),
                cpu.toBuilder().pooling(PoolingKind.LAST).build())) {
            assertEquals(LLMModel.PoolingType.LAST, last.poolingType());
            assertEquals(64, last.encode(tokens(2)).length);
        }
        try (GGUFModel none = new GGUFModel(model.toString(),
                cpu.toBuilder().pooling(PoolingKind.NONE).build())) {
            assertEquals(LLMModel.PoolingType.NONE, none.poolingType());
            assertEquals("NONE gives one embedding row per token", tokens(2).length,
                    none.encodeTokens(tokens(2)).length);
        }
    }

    @Test
    public void rankRefusesAModelWithoutAClassifierHeadByName() throws IOException {
        Path model = bertPooling(SyntheticModels.POOLING_MEAN).writeTempFile();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .pooling(PoolingKind.RANK).build();
        IllegalArgumentException inspect = assertThrows(IllegalArgumentException.class,
                () -> LLMModelFactory.inspect(model.toString(), cfg));
        assertTrue(inspect.getMessage(), inspect.getMessage().contains("rank"));
        assertTrue(inspect.getMessage(), inspect.getMessage().contains("classifier"));
        IllegalArgumentException open = assertThrows(IllegalArgumentException.class,
                () -> new GGUFModel(model.toString(), cfg));
        assertTrue(open.getMessage(), open.getMessage().contains("rank"));
    }

    @Test
    public void rankOverrideRunsTheClassifierOfARankModel() throws IOException {
        Path rank = SyntheticModels.rank("neg", "pos").writeTempFile();
        EngineConfig cpu = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build();
        try (GGUFModel m = new GGUFModel(rank.toString(), cpu.toBuilder().pooling(PoolingKind.MEAN).build())) {
            assertEquals("the classifier drops: the head pools the rows instead",
                    LLMModel.PoolingType.MEAN, m.poolingType());
            assertEquals(64, m.encode(tokens(1)).length);
        }
        try (GGUFModel r = new GGUFModel(rank.toString(), cpu)) {
            assertEquals(LLMModel.PoolingType.RANK, r.poolingType());
            assertEquals("the GGUF pooling is untouched without an override", 2, r.classify(tokens(1)).length);
        }
    }

    @Test
    public void thePropertyRouteReachesTheModel() throws IOException {
        Path model = bertPooling(SyntheticModels.POOLING_MEAN).writeTempFile();
        Properties p = new Properties();
        p.setProperty("neuralfree.pooling", "cls");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(PoolingKind.CLS, cfg.pooling());
        try (GGUFModel m = new GGUFModel(model.toString(), cfg)) {
            assertEquals(LLMModel.PoolingType.CLS, m.poolingType());
        }
    }

    @Test
    public void inspectAppliesTheOverrideToTheResolvedSpec() throws IOException {
        Path model = bertPooling(SyntheticModels.POOLING_MEAN).writeTempFile();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .pooling(PoolingKind.LAST).build();
        ModelSpec spec = LLMModelFactory.inspect(model.toString(), cfg);
        assertEquals(PoolingKind.LAST, spec.head().pooling());
        ModelSpec plain = LLMModelFactory.inspect(model.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build());
        assertEquals(PoolingKind.MEAN, plain.head().pooling());
    }
}
