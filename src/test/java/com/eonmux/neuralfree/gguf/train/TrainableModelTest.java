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
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The training forward pass against the inference engine on the same weights, and the analytic gradients against
 * central finite differences of the training loss. The fixture is the tiny F32 llama the engine tests run.
 */
public class TrainableModelTest {

    private static final int N_CTX = 16;
    private static final int[] TOKENS = {1, 7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301};
    private static final int[] LABELS = {7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301, 44};

    private static Path file;
    private static GGUFModel model;

    @BeforeClass
    public static void open() throws IOException {
        file = SyntheticModels.llama().writeTempFile();
        model = new GGUFModel(file.toString(), config());
    }

    @AfterClass
    public static void deleteModel() throws IOException {
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
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(N_CTX)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    @Test
    public void theTrainingForwardPassMatchesTheEngineLogits() throws IOException {
        model.forwardBackend().reset();
        float[][] engine = model.forwardBackend().forwardRows(TOKENS, 0, null);
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            float[][] rows = train.logits(TOKENS);
            assertEquals(TOKENS.length, rows.length);
            double worst = 0.0;
            for (int t = 0; t < rows.length; t++) {
                assertEquals(train.nVocab(), rows[t].length);
                for (int v = 0; v < rows[t].length; v++) {
                    worst = Math.max(worst, Math.abs(rows[t][v] - engine[t][v]));
                }
            }
            assertTrue("largest logit difference to the engine: " + worst, worst < 2e-3);
        }
    }

    @Test
    public void thePlanNamesEveryTrainedTensor() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            FinetunePlan plan = train.plan();
            assertEquals("llama", plan.arch());
            assertEquals(2, plan.nLayer());
            for (FinetunePlan.Layer layer : plan.layers()) {
                assertEquals(2, layer.nHeadKv());
                assertEquals(4, layer.nHead());
                assertEquals(16, layer.headDimK());
                assertTrue(train.parameter(layer.attnQ()) != null);
                assertTrue(train.parameter(layer.ffnDown()) != null);
            }
            for (String name : plan.trainableTensorNames()) {
                assertTrue("no parameter bound for " + name, train.parameter(name) != null);
            }
            assertEquals(plan.trainableTensorNames().size(), train.parameters().size());
        }
    }

    @Test
    public void theAnalyticGradientsMatchFiniteDifferences() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            train.zeroGrad();
            train.step(TOKENS, LABELS);
            checkGradient(train, "blk.0.attn_q.weight", 3, 2e-2);
            checkGradient(train, "blk.1.ffn_down.weight", 3, 2e-2);
            checkGradient(train, "blk.0.ffn_gate.weight", 2, 2e-2);
            checkGradient(train, "token_embd.weight", 3, 2e-2);
            checkGradient(train, "output_norm.weight", 2, 2e-2);
            checkGradient(train, "blk.0.attn_v.weight", 2, 2e-2);
        }
    }

    /**
     * Compares the accumulated gradient of {@code samples} pseudo-random elements of {@code name} with a central
     * difference of the training loss. The loss is float32, so the step must be large enough to lift the change above
     * its rounding; the tolerance is relative to the finite difference.
     */
    private static void checkGradient(TrainableModel train, String name, int samples, double tolerance) {
        TrainableModel.Parameter p = train.parameter(name);
        assertTrue("missing parameter " + name, p != null);
        Random random = new Random(1234 + name.hashCode());
        for (int s = 0; s < samples; s++) {
            int index = random.nextInt(p.size());
            float original = p.value()[index];
            double eps = 1e-2;
            p.value()[index] = original + (float) eps;
            double plus = train.forward(TOKENS, LABELS).loss();
            p.value()[index] = original - (float) eps;
            double minus = train.forward(TOKENS, LABELS).loss();
            p.value()[index] = original;
            double numeric = (plus - minus) / (2 * eps);
            double analytic = p.grad()[index];
            double error = Math.abs(numeric - analytic);
            double allowed = 1e-4 + tolerance * Math.abs(numeric);
            assertTrue(name + "[" + index + "]: analytic " + analytic + ", numeric " + numeric
                    + " (" + error + " off, " + allowed + " allowed)", error < allowed);
        }
    }

    @Test
    public void theBackwardPassRecomputesTheForwardStateExactly() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            train.forward(TOKENS, LABELS);
            assertEquals("the rematerialized blocks must reproduce the forward pass", 0.0,
                    train.recomputeDifference(), 0.0);
        }
    }

    @Test
    public void aWindowMustFitTheModel() throws IOException {
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            int[] tooLong = Arrays.copyOf(TOKENS, N_CTX + 1);
            try {
                train.forward(tooLong, Arrays.copyOf(tooLong, tooLong.length));
                throw new AssertionError("a window longer than the fit was accepted");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage().contains("outside"));
            }
        }
    }
}
