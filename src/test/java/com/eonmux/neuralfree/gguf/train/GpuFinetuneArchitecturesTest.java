package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.tensor.gpu.TrainParity;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The device trainer on the wider architectures the plan accepts: the gemma2 sliding-window, sandwich-norm block with
 * attention and final soft-caps, and the qwen3moe routed FFN with per-head Q/K norms. Each family's forward loss and
 * sampled parameter gradients are compared with the host trainer on the same model. Skipped where the {@code train}
 * kernels are unavailable.
 */
public class GpuFinetuneArchitecturesTest {

    private static final int N_CTX = 40;
    private static final int[] TOKENS = tokens(40, 11);
    private static final int[] LABELS = tokens(40, 173);

    private static Path gemmaFile;
    private static Path moeFile;
    private static GGUFModel gemma;
    private static GGUFModel moe;
    private static TrainParity.Session session;
    private static ComputeBackend backend;

    @BeforeClass
    public static void open() throws IOException {
        ComputationGraph.current().clear();
        gemmaFile = SyntheticModels.gemma2(SyntheticModels.Config.gemma2().toBuilder().nLayer(2).build())
                .writeTempFile();
        moeFile = SyntheticModels.qwen3moe(SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build())
                .writeTempFile();
        gemma = new GGUFModel(gemmaFile.toString(), config());
        moe = new GGUFModel(moeFile.toString(), config());
        session = TrainParity.open(TrainParity.Api.OPENCL);
        backend = session.backend();
    }

    @AfterClass
    public static void close() throws IOException {
        if (session != null) {
            session.close();
        }
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

    @After
    public void clearGraph() {
        ComputationGraph.current().clear();
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
    public void theGemmaSandwichAndSoftcapBlocksMatchOnTheDevice() throws IOException {
        checkModel(gemma, "gemma2", new String[]{"blk.0.attn_q.weight", "blk.0.post_attention_norm.weight",
                "blk.0.post_ffw_norm.weight", "blk.1.ffn_down.weight", "token_embd.weight", "output_norm.weight"});
    }

    @Test
    public void theRoutedMoeBlockMatchesOnTheDevice() throws IOException {
        checkModel(moe, "qwen3moe", new String[]{"blk.0.ffn_gate_inp.weight", "blk.0.ffn_up_exps.weight",
                "blk.0.ffn_down_exps.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_q.weight", "token_embd.weight"});
    }

    private static void checkModel(GGUFModel model, String arch, String[] names) throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            TrainableModel.StepResult hostResult = host.forward(TOKENS, LABELS);
            TrainableModel.StepResult gpuResult = gpu.forward(TOKENS, LABELS);
            assertEquals(arch + " loss", hostResult.loss(), gpuResult.loss(), 3e-3);
            assertEquals(arch + " correct", hostResult.correct(), gpuResult.correct());

            host.zeroGrad();
            host.step(TOKENS, LABELS);
            gpu.zeroGrad();
            gpu.step(TOKENS, LABELS);
            Random random = new Random(91);
            for (String name : names) {
                TrainableModel.Parameter hp = host.parameter(name);
                TrainableModel.Parameter gp = gpu.parameter(name);
                assertNotNull(arch + ": " + name, hp);
                assertNotNull(arch + ": " + name, gp);
                for (int s = 0; s < 6; s++) {
                    int i = random.nextInt(hp.size());
                    double expected = hp.grad()[i];
                    double actual = gp.grad()[i];
                    double allowed = 3e-4 + 3e-2 * Math.abs(expected);
                    assertEquals(arch + ": " + name + "[" + i + "]", expected, actual, allowed);
                }
            }
        }
    }

    @Test
    public void theFixtureWindowsExerciseTheMask() {
        assertTrue(N_CTX > 32);
    }
}
