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
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The optimizer updates against a double-precision reference of the same formulas: the decoupled decay, the bias
 * correction of AdamW over several steps, and SGD's coupled L2 decay.
 */
public class FinetuneOptimizerTest {

    private static final int N_CTX = 8;
    private static final String TENSOR = "blk.0.ffn_norm.weight";

    private static Path file;
    private static GGUFModel model;

    @BeforeClass
    public static void open() throws IOException {
        file = SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(1).build()).writeTempFile();
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

    /** The model's parameter with every value and gradient set to a fixed value. */
    private static TrainableModel.Parameter seeded(TrainableModel train, float value, float gradient) {
        TrainableModel.Parameter p = train.parameter(TENSOR);
        Arrays.fill(p.value(), value);
        Arrays.fill(p.grad(), gradient);
        return p;
    }

    @Test
    public void adamwCorrectsTheBiasOfItsFirstStep() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            TrainableModel.Parameter p = seeded(train, 0.4f, 2f);
            new FinetuneOptimizer(train, FinetuneOptimizer.Kind.ADAMW, 0f).step(0.1f);
            // mhat = g and vhat = g^2 after the first step's correction, so the update is lr * g / (|g| + eps).
            double expected = 0.4 - 0.1 * (2.0 / (2.0 + FinetuneOptimizer.EPS));
            for (float w : p.value()) {
                assertEquals(expected, w, 1e-7);
            }
        }
    }

    @Test
    public void weightDecayShrinksAWeightWithNoGradient() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            TrainableModel.Parameter p = seeded(train, 0.4f, 0f);
            new FinetuneOptimizer(train, FinetuneOptimizer.Kind.ADAMW, 0.1f).step(0.1f);
            for (float w : p.value()) {
                // w' = w - lr * wd * w; the Adam term is exactly zero because both moments stay zero.
                assertEquals(0.4 * (1.0 - 0.1 * 0.1), w, 1e-7);
            }
        }
    }

    @Test
    public void sgdBringsTheGradientAndTheWeightDecayTogether() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            TrainableModel.Parameter p = seeded(train, 0.4f, 1.5f);
            new FinetuneOptimizer(train, FinetuneOptimizer.Kind.SGD, 0.1f).step(0.05f);
            double expected = 0.4 - 0.05 * (1.5 + 0.1 * 0.4);
            for (float w : p.value()) {
                assertEquals(expected, w, 1e-7);
            }
        }
    }

    @Test
    public void adamwMatchesAReferenceOverSeveralSteps() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            float initial = 0.4f;
            float gradient = 1.5f;
            float lr = 0.02f;
            float wd = 0.05f;
            TrainableModel.Parameter p = seeded(train, initial, gradient);
            FinetuneOptimizer optimizer = new FinetuneOptimizer(train, FinetuneOptimizer.Kind.ADAMW, wd);
            int steps = 4;
            for (int s = 0; s < steps; s++) {
                optimizer.step(lr);
            }
            assertEquals(steps, optimizer.stepCount());
            double expected = referenceAdamW(initial, gradient, steps, lr, wd);
            for (float w : p.value()) {
                assertEquals(expected, w, 1e-6 + 1e-6 * Math.abs(expected));
            }
        }
    }

    /** AdamW in double precision, the update {@code com.eonmux.neuralfree.optim.AdamW} and llama.cpp compute. */
    private static double referenceAdamW(double w, double g, int steps, double lr, double wd) {
        double m = 0;
        double v = 0;
        for (int t = 1; t <= steps; t++) {
            w -= lr * wd * w;
            m = FinetuneOptimizer.BETA1 * m + (1 - FinetuneOptimizer.BETA1) * g;
            v = FinetuneOptimizer.BETA2 * v + (1 - FinetuneOptimizer.BETA2) * g * g;
            double biasCorrection1 = 1 - Math.pow(FinetuneOptimizer.BETA1, t);
            double biasCorrection2 = 1 - Math.pow(FinetuneOptimizer.BETA2, t);
            w -= (lr / biasCorrection1) * m / (Math.sqrt(v / biasCorrection2) + FinetuneOptimizer.EPS);
        }
        return w;
    }

    @Test
    public void anInvalidLearningRateOrDecayIsRefused() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            FinetuneOptimizer optimizer = new FinetuneOptimizer(train, FinetuneOptimizer.Kind.ADAMW, 0f);
            for (float lr : new float[] {0f, -1f, Float.NaN, Float.POSITIVE_INFINITY}) {
                try {
                    optimizer.step(lr);
                    throw new AssertionError("the learning rate " + lr + " was accepted");
                } catch (IllegalArgumentException expected) {
                    assertTrue(expected.getMessage().contains("learning rate"));
                }
            }
            assertEquals(0, optimizer.stepCount());
            try {
                new FinetuneOptimizer(train, FinetuneOptimizer.Kind.SGD, -0.1f);
                throw new AssertionError("a negative weight decay was accepted");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("weight decay"));
            }
        }
    }

    @Test
    public void theHostUpdateSplitOverThePoolMatchesTheSequentialLoopBitForBit() {
        int n = 300_001;
        float[] values = new float[n];
        java.util.Random random = new java.util.Random(5);
        for (int i = 0; i < n; i++) {
            values[i] = random.nextFloat() - 0.5f;
        }
        TrainableModel.Parameter p = new TrainableModel.Parameter("w", com.eonmux.neuralfree.gguf.GGMLType.F32,
                new long[]{n}, 1, n, values.clone());
        float[] w = values.clone();
        float[] m = new float[n];
        float[] v = new float[n];
        float lr = 3e-3f;
        float wd = 0.05f;
        try (FinetuneOptimizer.Update update = FinetuneOptimizer.hostUpdate(java.util.Collections.singletonList(p),
                FinetuneOptimizer.Kind.ADAMW, wd)) {
            for (int step = 1; step <= 3; step++) {
                for (int i = 0; i < n; i++) {
                    p.grad()[i] = (float) Math.sin(i * 0.37 + step);
                }
                float bc1 = 1f - (float) Math.pow(FinetuneOptimizer.BETA1, step);
                float bc2 = 1f - (float) Math.pow(FinetuneOptimizer.BETA2, step);
                update.apply(lr, bc1, bc2);
                float stepSize = lr / bc1;
                for (int i = 0; i < n; i++) {
                    float g = p.grad()[i];
                    w[i] -= lr * wd * w[i];
                    m[i] = FinetuneOptimizer.BETA1 * m[i] + (1f - FinetuneOptimizer.BETA1) * g;
                    v[i] = FinetuneOptimizer.BETA2 * v[i] + (1f - FinetuneOptimizer.BETA2) * g * g;
                    w[i] -= stepSize * m[i] / ((float) Math.sqrt(v[i] / bc2) + FinetuneOptimizer.EPS);
                }
            }
        }
        assertArrayEquals(w, p.value(), 0f);
    }

    @Test
    public void theOptimizerStepsEveryTrainableTensor() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            FinetuneOptimizer optimizer = new FinetuneOptimizer(train, FinetuneOptimizer.Kind.ADAMW, 0f);
            assertEquals(train.parameters().size(), optimizer.parameters().size());
            assertEquals("adamw", FinetuneOptimizer.Kind.ADAMW.flag());
            assertEquals(FinetuneOptimizer.Kind.SGD, FinetuneOptimizer.Kind.parse("SGD"));
            assertEquals(null, FinetuneOptimizer.Kind.parse("lamb"));
        }
    }
}
