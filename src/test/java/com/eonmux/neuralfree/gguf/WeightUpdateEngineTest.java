package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.ForwardBackend;
import com.eonmux.neuralfree.gguf.engine.WeightUpdateTensor;
import com.eonmux.neuralfree.gguf.engine.WeightUpdater;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static com.eonmux.neuralfree.gguf.SplitFixtures.PROMPT;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineModel#weightUpdater} on both engines: an update round over a running model changes the logits the
 * very next forward pass serves (CPU and, when OpenCL is present, GPU), addresses the output head by its alias, and
 * refuses unknown tensors by name. A layer-split GPU load refuses live updates, its tensors live on several backends.
 */
public class WeightUpdateEngineTest {

    private static final int N_EMBD = 64;
    private static final int N_VOCAB = 320;
    private static final String NORM = "blk.0.attn_norm.weight";

    private Path model;

    @Before
    public void writeModel() throws IOException {
        model = SyntheticModels.llama().writeTempFile();
    }

    @After
    public void deleteModel() throws IOException {
        if (model != null) {
            Files.deleteIfExists(model);
        }
    }

    private static EngineConfig cpuConfig() {
        return EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build())
                .build();
    }

    private static float maxDiff(float[] a, float[] b) {
        assertEquals(a.length, b.length);
        float m = 0f;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }

    @Test
    public void anUpdateRoundChangesTheNextForwardPassOfTheCpuModel() throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig())) {
            ForwardBackend f = m.forwardBackend();
            float[] before = f.prefill(PROMPT, 0).clone();
            f.reset();

            WeightUpdater up = m.weightUpdater();
            assertEquals(WeightUpdater.DEFAULT_VERSION, up.version());
            assertFalse(up.updating());
            up.start();
            assertTrue(up.updating());
            up.apply(WeightUpdateTensor.f32(NORM, new long[]{N_EMBD}, fill(N_EMBD, 3f)));
            up.finish("rl-step-9");
            assertEquals("rl-step-9", up.version());
            assertFalse(up.updating());

            float[] after = m.forwardBackend().prefill(PROMPT, 0);
            m.forwardBackend().reset();
            assertTrue("the norm update reaches the forward pass", maxDiff(before, after) > 1e-4f);
        }
    }

    @Test
    public void theCpuModelUpdatesItsOutputHead() throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig())) {
            ForwardBackend f = m.forwardBackend();
            float[] before = f.prefill(PROMPT, 0).clone();
            f.reset();

            WeightUpdater up = m.weightUpdater();
            up.apply(WeightUpdateTensor.f32("output.weight", new long[]{N_EMBD, N_VOCAB}, fill(N_EMBD * N_VOCAB, 0.5f)));

            float[] after = m.forwardBackend().prefill(PROMPT, 0);
            m.forwardBackend().reset();
            assertTrue("the output-head update reaches the forward pass", maxDiff(before, after) > 1f);
        }
    }

    @Test
    public void theCpuModelRefusesAnUnknownTensorByName() throws IOException {
        try (GGUFModel m = new GGUFModel(model.toString(), cpuConfig())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> m.weightUpdater()
                    .apply(WeightUpdateTensor.f32("blk.9.attn_norm.weight", new long[]{N_EMBD}, fill(N_EMBD, 1f))));
            assertTrue(e.getMessage(), e.getMessage().contains("blk.9.attn_norm.weight"));
        }
    }

    @Test
    public void anUpdateRoundChangesTheNextForwardPassOfTheGpuModel() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, SplitFixtures.oneDevice().build())) {
            ForwardBackend f = m.forwardBackend();
            float[] before = f.prefill(PROMPT, 0).clone();
            f.reset();

            WeightUpdater up = m.weightUpdater();
            up.start();
            up.apply(WeightUpdateTensor.f32(NORM, new long[]{N_EMBD}, fill(N_EMBD, 3f)));
            up.finish(null);

            assertTrue("the norm update reaches the device forward pass",
                    maxDiff(before, f.prefill(PROMPT, 0)) > 1e-4f);
            f.reset();
        }
    }

    @Test
    public void aLayerSplitGpuLoadRefusesLiveUpdates() throws IOException {
        SplitFixtures.assumeTwoDevices();
        EngineConfig cfg = SplitFixtures.split().build();
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    m::weightUpdater);
            assertTrue(e.getMessage(), e.getMessage().contains("live weight update"));
        }
    }

    private static float[] fill(int n, float v) {
        float[] fs = new float[n];
        Arrays.fill(fs, v);
        return fs;
    }
}
