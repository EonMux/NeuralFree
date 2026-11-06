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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The wider architectures {@link FinetunePlan} trains beside the dense pre-norm decoder: gemma2's sliding-window and
 * sandwich-norm blocks with attention and final logit soft-caps, and the qwen3 MoE routed FFN. Each family's training
 * forward pass is compared with the inference engine on the same weights, and its parameter gradients with central
 * differences of the training loss.
 */
public class FinetuneArchitecturesTest {

    /** A window longer than the fixture's 32-token sliding window, so the mask is exercised. */
    private static final int N_CTX = 40;
    private static final int[] TOKENS = tokens(40, 11);
    private static final int[] LABELS = tokens(40, 173);

    private static Path gemmaFile;
    private static Path moeFile;
    private static GGUFModel gemma;
    private static GGUFModel moe;

    @BeforeClass
    public static void open() throws IOException {
        gemmaFile = SyntheticModels.gemma2(SyntheticModels.Config.gemma2().toBuilder().nLayer(2).build())
                .writeTempFile();
        moeFile = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build())
                .writeTempFile();
        gemma = new GGUFModel(gemmaFile.toString(), config());
        moe = new GGUFModel(moeFile.toString(), config());
    }

    @AfterClass
    public static void close() throws IOException {
        if (gemma != null) {
            gemma.close();
        }
        if (moe != null) {
            moe.close();
        }
        if (gemmaFile != null) {
            Files.deleteIfExists(gemmaFile);
        }
        if (moeFile != null) {
            Files.deleteIfExists(moeFile);
        }
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(N_CTX).ubatch(N_CTX)
                        .typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static int[] tokens(int n, int seed) {
        int[] out = new int[n];
        int value = seed;
        for (int i = 0; i < n; i++) {
            value = (value * 37 + 11) % 320;
            out[i] = value;
        }
        return out;
    }

    @Test
    public void theGemmaSlidingWindowSandwichBlockIsPlanned() throws IOException {
        try (TrainableModel train = TrainableModel.open(gemma, N_CTX)) {
            FinetunePlan plan = train.plan();
            assertEquals("gemma2", plan.arch());
            assertEquals((float) Math.sqrt(plan.nEmbd()), plan.embdScale(), 1e-5f);
            assertEquals(30f, plan.finalSoftcap(), 0f);
            FinetunePlan.Layer sliding = plan.layer(0);
            assertEquals(32, sliding.swaWindow());
            assertEquals(50f, sliding.attnSoftcap(), 0f);
            assertNotNull(sliding.postAttnNorm());
            assertNotNull(sliding.postFfnNorm());
            FinetunePlan.Layer global = plan.layer(1);
            assertEquals(0, global.swaWindow());
            assertEquals(50f, global.attnSoftcap(), 0f);
            assertNotNull(global.postAttnNorm());
            assertNotNull(global.postFfnNorm());
            for (String name : plan.trainableTensorNames()) {
                assertTrue("no parameter bound for " + name, train.parameter(name) != null);
            }
        }
    }

    @Test
    public void theGemmaForwardPassMatchesTheEngineLogits() throws IOException {
        assertForwardMatchesEngine(gemma, 5e-3, "gemma2");
    }

    @Test
    public void theGemmaGradientsMatchFiniteDifferences() throws IOException {
        try (TrainableModel train = TrainableModel.open(gemma, N_CTX)) {
            train.zeroGrad();
            train.step(TOKENS, LABELS);
            checkGradient(train, "blk.0.attn_q.weight", 2, 3e-2, false);
            checkGradient(train, "blk.0.post_attention_norm.weight", 2, 3e-2, false);
            checkGradient(train, "blk.1.ffn_down.weight", 2, 3e-2, false);
            checkGradient(train, "token_embd.weight", 2, 3e-2, false);
            checkGradient(train, "output_norm.weight", 2, 3e-2, false);
        }
    }

    @Test
    public void theRoutedMoeBlockIsPlanned() throws IOException {
        try (TrainableModel train = TrainableModel.open(moe, N_CTX)) {
            FinetunePlan plan = train.plan();
            assertEquals("qwen3moe", plan.arch());
            for (FinetunePlan.Layer layer : plan.layers()) {
                FinetunePlan.Moe routed = layer.moe();
                assertNotNull(routed);
                assertEquals(4, routed.nExpert());
                assertEquals(2, routed.nUsed());
                assertEquals(32, routed.nFfExp());
                assertTrue(routed.normWeights());
                assertNull(layer.ffnUp());
                assertNotNull(train.parameter(routed.router()));
                assertNotNull(train.parameter(routed.upExps()));
            }
            for (String name : plan.trainableTensorNames()) {
                assertTrue("no parameter bound for " + name, train.parameter(name) != null);
            }
        }
    }

    @Test
    public void theMoeForwardPassMatchesTheEngineLogits() throws IOException {
        assertForwardMatchesEngine(moe, 2e-3, "qwen3moe");
    }

    /**
     * The routed-expert weights and the rest of a MoE block against central differences. The fixture is one layer:
     * with a second layer, a change to any first-layer weight shifts the second router's input and can flip its
     * discrete top-k choice, so the loss stops being differentiable in the first-layer parameters.
     */
    @Test
    public void theMoeGradientsMatchFiniteDifferences() throws IOException {
        Path file = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1).build())
                .writeTempFile();
        try (GGUFModel one = new GGUFModel(file.toString(), config());
             TrainableModel train = TrainableModel.open(one, N_CTX)) {
            train.zeroGrad();
            train.step(TOKENS, LABELS);
            // The expert projections do not feed the router, so their loss is smooth.
            checkGradient(train, "blk.0.ffn_up_exps.weight", 3, 5e-2, false);
            checkGradient(train, "blk.0.ffn_down_exps.weight", 3, 5e-2, false);
            checkGradient(train, "blk.0.ffn_gate_exps.weight", 3, 5e-2, false);
            // The router's top-k selection is a discrete function, and every weight upstream of the router moves its
            // input: a finite difference whose step crosses an expert-ordering boundary compares two different
            // routing decisions and is not a derivative. Those checks skip the boundary-crossing samples.
            checkGradient(train, "blk.0.attn_q.weight", 8, 3e-2, true);
            checkGradient(train, "blk.0.attn_q_norm.weight", 8, 3e-2, true);
            checkGradient(train, "blk.0.attn_k_norm.weight", 8, 3e-2, true);
            checkGradient(train, "blk.0.attn_v.weight", 8, 3e-2, true);
            checkGradient(train, "token_embd.weight", 8, 3e-2, true);
            checkGradient(train, "blk.0.ffn_gate_inp.weight", 8, 5e-2, true);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    /**
     * The router gradient on a fixture whose routing is smooth: with every expert selected there is no top-k
     * boundary, so a central difference is the derivative of the same function the backward pass computes.
     */
    @Test
    public void theRouterGradientIsSmoothWhenEveryExpertIsSelected() throws IOException {
        Path file = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(1)
                .nExpertUsed(4).build()).writeTempFile();
        try (GGUFModel all = new GGUFModel(file.toString(), config());
             TrainableModel train = TrainableModel.open(all, N_CTX)) {
            train.zeroGrad();
            train.step(TOKENS, LABELS);
            checkGradient(train, "blk.0.ffn_gate_inp.weight", 8, 5e-2, false);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    private static void assertForwardMatchesEngine(GGUFModel model, double tolerance, String arch) throws IOException {
        model.forwardBackend().reset();
        float[][] engine = model.forwardBackend().forwardRows(TOKENS, 0, null);
        try (TrainableModel train = TrainableModel.open(model, N_CTX)) {
            float[][] rows = train.logits(TOKENS);
            assertEquals(TOKENS.length, rows.length);
            double worst = 0.0;
            for (int t = 0; t < rows.length; t++) {
                for (int v = 0; v < rows[t].length; v++) {
                    worst = Math.max(worst, Math.abs(rows[t][v] - engine[t][v]));
                }
            }
            assertTrue(arch + ": largest logit difference to the engine: " + worst, worst < tolerance);
        }
    }

    /**
     * Compares the accumulated gradient of {@code samples} pseudo-random elements of {@code name} with a central
     * difference of the training loss. The loss is float32, so the step must be large enough to lift the change above
     * its rounding; the tolerance is relative to the finite difference.
     *
     * <p>With {@code routingSensitive}, elements whose step crosses an expert-ordering boundary are skipped: the loss
     * is discontinuous there, so no central difference is a derivative. The curvature drift
     * {@code |f(+e) + f(-e) - 2 f(0)|} is O(e^2) on a smooth branch and O(jump) across a boundary, so it separates the
     * two.</p>
     */
    private static void checkGradient(TrainableModel train, String name, int samples, double tolerance,
                                      boolean routingSensitive) {
        TrainableModel.Parameter p = train.parameter(name);
        assertNotNull("missing parameter " + name, p);
        double base = train.forward(TOKENS, LABELS).loss();
        Random random = new Random(1234 + name.hashCode());
        int checked = 0;
        for (int s = 0; s < samples; s++) {
            int index = random.nextInt(p.size());
            float original = p.value()[index];
            double eps = 1e-2;
            p.value()[index] = original + (float) eps;
            double plus = train.forward(TOKENS, LABELS).loss();
            p.value()[index] = original - (float) eps;
            double minus = train.forward(TOKENS, LABELS).loss();
            p.value()[index] = original;
            if (routingSensitive && Math.abs(plus + minus - 2.0 * base) > 2e-4) {
                continue;
            }
            checked++;
            double numeric = (plus - minus) / (2 * eps);
            double analytic = p.grad()[index];
            double error = Math.abs(numeric - analytic);
            double allowed = 2e-4 + tolerance * Math.abs(numeric);
            assertTrue(name + "[" + index + "]: analytic " + analytic + ", numeric " + numeric
                    + " (" + error + " off, " + allowed + " allowed)", error < allowed);
        }
        assertTrue("no sample of " + name + " was on a smooth branch", checked > 0);
    }

    /** The window's token sequence is deterministic; keep the fixture interesting by checking its spread. */
    @Test
    public void theFixtureWindowCoversManyTokens() {
        assertEquals(N_CTX, TOKENS.length);
        assertEquals(N_CTX, LABELS.length);
        assertTrue(Arrays.stream(TOKENS).distinct().count() > N_CTX / 2);
    }
}
