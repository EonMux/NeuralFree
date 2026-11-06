package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The training loop: its learning-rate schedule, the epochs it reports, and the fact that the loss of a small
 * overfit run falls — the behaviour the tool exists for.
 */
public class FinetuneRunnerTest {

    private static final int N_CTX = 16;
    private static final int TOKENS = 64;

    private static Path file;
    private static GGUFModel model;

    @BeforeClass
    public static void open() throws IOException {
        file = SyntheticModels.llama().writeTempFile();
        model = new GGUFModel(file.toString(), config());
    }

    @AfterClass
    public static void close() throws IOException {
        if (model != null) {
            model.close();
        }
        if (file != null) {
            Files.deleteIfExists(file);
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(N_CTX).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    /** A repeating pattern of in-vocabulary tokens; a repeated corpus is what a short overfit run can memorise. */
    private static FinetuneDataset data(double valSplit) {
        int[] tokens = new int[TOKENS];
        int[] pattern = {11, 42, 73, 104, 135, 166, 197, 228};
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = pattern[i % pattern.length];
        }
        return FinetuneDataset.of(tokens, N_CTX, valSplit);
    }

    @Test
    public void theScheduleDecaysGeometricallyToTheMinimum() {
        FinetuneRunner.Config config = new FinetuneRunner.Config(4, 1e-2f, 1e-4f, 4f, 0f,
                FinetuneOptimizer.Kind.ADAMW);
        assertEquals(1e-2f, config.learningRate(0), 1e-9);
        assertEquals(1e-3f, config.learningRate(2), 1e-9);
        assertEquals(1e-4f, config.learningRate(4), 1e-9);
        assertEquals(1e-4f, config.learningRate(9), 1e-9);
        // every epoch is below the one before it
        for (int e = 1; e <= 4; e++) {
            assertTrue(config.learningRate(e) < config.learningRate(e - 1));
        }
    }

    @Test
    public void withoutADecayWindowTheRateReturnsToTheFirstEpochAndThenFloors() {
        FinetuneRunner.Config config = new FinetuneRunner.Config(3, 5e-3f, 1e-5f, 0f, 0f,
                FinetuneOptimizer.Kind.SGD);
        assertEquals(5e-3f, config.learningRate(0), 0f);
        assertEquals(1e-5f, config.learningRate(1), 0f);
        assertEquals(1e-5f, config.learningRate(2), 0f);
    }

    @Test
    public void theRunReportsEveryEpochAndTheLossFalls() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            FinetuneDataset data = data(0.0);
            FinetuneRunner.Config config = new FinetuneRunner.Config(4, 5e-2f, 5e-2f, 0f, 0f,
                    FinetuneOptimizer.Kind.ADAMW);
            List<FinetuneRunner.EpochStats> reported = new ArrayList<>();
            List<FinetuneRunner.EpochStats> epochs = FinetuneRunner.run(train, data, config, reported::add);
            assertEquals(4, epochs.size());
            assertEquals(epochs, reported);
            assertEquals(1, epochs.get(0).epoch());
            assertEquals(4, epochs.get(3).epoch());
            assertEquals(data.training().size(), epochs.get(0).windows());
            assertEquals(5e-2f, epochs.get(0).learningRate(), 0f);
            assertTrue("no validation part, so no validation loss",
                    Double.isNaN(epochs.get(0).validationLoss()));
            for (FinetuneRunner.EpochStats stats : epochs) {
                assertTrue("the loss is finite at epoch " + stats.epoch(), Double.isFinite(stats.trainLoss()));
                assertTrue(stats.trainAccuracy() >= 0.0 && stats.trainAccuracy() <= 1.0);
            }
            double first = epochs.get(0).trainLoss();
            double last = epochs.get(epochs.size() - 1).trainLoss();
            assertTrue("the loss fell from " + first + " to " + last, last < first);
        }
    }

    @Test
    public void theValidationWindowsAreScoredButNotTrained() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            FinetuneDataset data = data(0.25);
            assertFalse(data.validation().isEmpty());
            FinetuneRunner.Config config = new FinetuneRunner.Config(1, 1e-2f, 1e-2f, 0f, 0f,
                    FinetuneOptimizer.Kind.SGD);
            FinetuneRunner.EpochStats stats = FinetuneRunner.run(train, data, config, null).get(0);
            assertEquals(data.training().size(), stats.windows());
            assertTrue(Double.isFinite(stats.validationLoss()));
            assertTrue(stats.validationLoss() > 0.0);
            assertTrue(stats.validationAccuracy() >= 0.0 && stats.validationAccuracy() <= 1.0);
        }
    }

    @Test
    public void anInvalidConfigurationIsRefused() {
        try {
            new FinetuneRunner.Config(0, 1e-3f, 1e-4f, 1f, 0f, FinetuneOptimizer.Kind.ADAMW);
            throw new AssertionError("zero epochs were accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("-epochs"));
        }
        try {
            new FinetuneRunner.Config(1, 1e-4f, 1e-3f, 1f, 0f, FinetuneOptimizer.Kind.ADAMW);
            throw new AssertionError("a minimum above the initial rate was accepted");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage().contains("--learning-rate-min"));
        }
        try {
            new FinetuneRunner.Config(1, 1e-3f, 1e-4f, 1f, 0f, null);
            throw new AssertionError("a null optimizer was accepted");
        } catch (NullPointerException expected) {
            assertTrue(expected.getMessage().contains("optimizer"));
        }
    }
}
