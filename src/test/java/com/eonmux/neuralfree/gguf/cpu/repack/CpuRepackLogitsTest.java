package com.eonmux.neuralfree.gguf.cpu.repack;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuSimd;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Repacking through a real CPU forward pass: a model whose copies are cut short by the budget produces exactly the
 * logits of the unrepacked model, and the default {@code auto} mode repacks nothing because the plain kernels are
 * faster for every covered type.
 */
public class CpuRepackLogitsTest {

    private static final int[] PROMPT = {1, 2, 3, 4};

    /** A tiny llama with Q8_0 weights whose rows split into whole Q8_0x4 groups. */
    private static GgufTestWriter model() {
        SyntheticModels.Config config = new SyntheticModels.Config.Builder()
                .nVocab(320).nEmbd(128).nLayer(2).nHead(4).nHeadKv(2).headDim(32).nRot(32).nFf(256)
                .nCtxTrain(256).normEps(1e-5f).weightType(GGMLType.Q8_0).build();
        return SyntheticModels.llama(config);
    }

    private static EngineConfig config() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static float[] logits(String path) throws IOException {
        try (GGUFModel m = new GGUFModel(path, config())) {
            return m.forwardBackend().prefill(PROMPT, 0);
        }
    }

    @Test
    public void aBudgetedRunLogitsMatchTheUnrepackedRun() throws IOException {
        Assume.assumeTrue(CpuSimd.available());
        Path path = model().writeTempFile();
        String previousMode = System.getProperty(CpuRepack.PROP_REPACK);
        String previousBudget = System.getProperty(CpuRepack.PROP_BUDGET);
        float[] plain;
        float[] budgeted;
        try {
            System.setProperty(CpuRepack.PROP_REPACK, "off");
            System.clearProperty(CpuRepack.PROP_BUDGET);
            plain = logits(path.toString());

            System.setProperty(CpuRepack.PROP_REPACK, "on");
            System.setProperty(CpuRepack.PROP_BUDGET, "40000");
            try (GGUFModel m = new GGUFModel(path.toString(), config())) {
                budgeted = m.forwardBackend().prefill(PROMPT, 0);
                assertTrue("some weights fit the budget", m.cpuWeights().repack().repackedCount() > 0);
                assertTrue("no weight beyond the budget was copied",
                        m.cpuWeights().repack().repackedBytes() <= 40000L);
            }
        } finally {
            restore(CpuRepack.PROP_REPACK, previousMode);
            restore(CpuRepack.PROP_BUDGET, previousBudget);
            Files.deleteIfExists(path);
        }
        assertArrayEquals("the budgeted logits must equal the unrepacked ones", plain, budgeted, 0f);
    }

    @Test
    public void theAutoModeRepacksNothingBecauseThePlainKernelsAreFaster() throws IOException {
        Assume.assumeTrue(CpuSimd.available());
        Path path = model().writeTempFile();
        String previousMode = System.getProperty(CpuRepack.PROP_REPACK);
        String previousBudget = System.getProperty(CpuRepack.PROP_BUDGET);
        try {
            System.clearProperty(CpuRepack.PROP_REPACK);
            System.clearProperty(CpuRepack.PROP_BUDGET);
            try (GGUFModel m = new GGUFModel(path.toString(), config())) {
                m.forwardBackend().prefill(PROMPT, 0);
                assertEquals(CpuRepack.Mode.AUTO, m.cpuWeights().repack().mode());
                assertEquals("auto selects no type while the repacked kernels are slower", 0,
                        m.cpuWeights().repack().repackedCount());
            }
        } finally {
            restore(CpuRepack.PROP_REPACK, previousMode);
            restore(CpuRepack.PROP_BUDGET, previousBudget);
            Files.deleteIfExists(path);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
