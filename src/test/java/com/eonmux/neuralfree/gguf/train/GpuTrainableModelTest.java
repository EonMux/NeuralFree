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
 * The device trainer against the host trainer on the same model: the forward logits, the training loss and every
 * parameter's gradient agree, and a full optimizer step leaves the same weights. Skipped where the {@code train}
 * kernels are unavailable.
 */
public class GpuTrainableModelTest {

    private static final int N_CTX = 16;
    private static final int[] TOKENS = {1, 7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301};
    private static final int[] LABELS = {7, 19, 42, 77, 100, 128, 201, 3, 55, 240, 17, 88, 129, 260, 301, 44};

    private static Path file;
    private static GGUFModel model;
    private static TrainParity.Session session;
    private static ComputeBackend backend;

    @BeforeClass
    public static void open() throws IOException {
        ComputationGraph.current().clear();
        file = SyntheticModels.llama().writeTempFile();
        model = new GGUFModel(file.toString(), config());
        session = TrainParity.open(TrainParity.Api.OPENCL);
        backend = session.backend();
    }

    @AfterClass
    public static void close() throws IOException {
        if (session != null) {
            session.close();
        }
        if (model != null) {
            model.close();
        }
        if (file != null) {
            Files.deleteIfExists(file);
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

    @Test
    public void theDeviceForwardMatchesTheHostLogits() throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            float[][] expected = host.logits(TOKENS);
            float[][] actual = gpu.logits(TOKENS);
            assertEquals(expected.length, actual.length);
            double worst = 0;
            for (int t = 0; t < expected.length; t++) {
                assertEquals(expected[t].length, actual[t].length);
                for (int v = 0; v < expected[t].length; v++) {
                    worst = Math.max(worst, Math.abs(expected[t][v] - actual[t][v]));
                }
            }
            assertTrue("largest logit difference between the host and device trainers: " + worst, worst < 2e-3);
        }
    }

    @Test
    public void theDeviceLossMatchesTheHostLoss() throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            TrainableModel.StepResult expected = host.forward(TOKENS, LABELS);
            TrainableModel.StepResult actual = gpu.forward(TOKENS, LABELS);
            assertEquals(expected.loss(), actual.loss(), 2e-3);
            assertEquals(expected.correct(), actual.correct());
        }
    }

    @Test
    public void theDeviceGradientsMatchTheHostGradients() throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend)) {
            host.zeroGrad();
            host.step(TOKENS, LABELS);
            gpu.zeroGrad();
            gpu.step(TOKENS, LABELS);
            Random random = new Random(7);
            for (String name : new String[]{"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                    "blk.0.attn_output.weight", "blk.1.ffn_up.weight", "blk.0.ffn_down.weight",
                    "blk.0.ffn_gate.weight", "token_embd.weight", "output_norm.weight", "blk.1.attn_norm.weight"}) {
                TrainableModel.Parameter hp = host.parameter(name);
                TrainableModel.Parameter gp = gpu.parameter(name);
                assertNotNull(name, hp);
                assertNotNull(name, gp);
                assertEquals(name + " size", hp.size(), gp.size());
                for (int s = 0; s < 12; s++) {
                    int i = random.nextInt(hp.size());
                    double expected = hp.grad()[i];
                    double actual = gp.grad()[i];
                    double allowed = 2e-4 + 2e-2 * Math.abs(expected);
                    assertEquals(name + "[" + i + "]", expected, actual, allowed);
                }
            }
        }
    }

    @Test
    public void oneOptimizerStepLeavesTheSameWeights() throws IOException {
        try (TrainableModel host = TrainableModel.open(model, N_CTX);
             GpuTrainableModel gpu = GpuTrainableModel.open(model, N_CTX, backend);
             FinetuneOptimizer hostOptimizer = new FinetuneOptimizer(host, FinetuneOptimizer.Kind.ADAMW, 0.1f);
             FinetuneOptimizer gpuOptimizer = new FinetuneOptimizer(gpu, FinetuneOptimizer.Kind.ADAMW, 0.1f)) {
            host.zeroGrad();
            host.step(TOKENS, LABELS);
            hostOptimizer.step(1e-3f);
            gpu.zeroGrad();
            gpu.step(TOKENS, LABELS);
            gpuOptimizer.step(1e-3f);
            TrainableModel.Parameter hp = host.parameter("blk.0.attn_q.weight");
            TrainableModel.Parameter gp = gpu.parameter("blk.0.attn_q.weight");
            double worst = 0;
            for (int i = 0; i < hp.size(); i++) {
                worst = Math.max(worst, Math.abs(hp.value()[i] - gp.value()[i]));
            }
            assertTrue("largest parameter difference after the step: " + worst, worst < 1e-4);
        }
    }
}
