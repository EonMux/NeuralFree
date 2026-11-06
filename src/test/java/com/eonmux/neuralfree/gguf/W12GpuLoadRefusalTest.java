package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import com.eonmux.neuralfree.gguf.engine.W12F16Mode;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.training.LogCapture;
import org.apache.logging.log4j.Level;
import org.junit.After;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code neuralfree.weights.w12} load-time refusals on a real GPU: the flag on a model whose weights give it no
 * W12 target (everything stored as F32) is refused by name, and an F16 model is held to the lossless contract of
 * {@code neuralfree.weights.w12.f16} — its off-grid weights refuse by default, and only the explicit {@code round}
 * brings them onto the bfloat16 grid, reporting the weights that moved. Both are host-side uploader behaviors, so
 * one GPU backend exercises them.
 */
public class W12GpuLoadRefusalTest {

    private final List<Path> files = new ArrayList<>();
    private Path model;

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
        files.clear();
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.defaults().toBuilder().nCtx(128).ubatch(128).build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory);
    }

    private static SpeculativeOptions w12(W12F16Mode f16) {
        return SpeculativeOptions.defaults().toBuilder().w12(true).w12Head(true).w12F16(f16).build();
    }

    @Test
    public void w12OnAnF32ModelIsRefusedByName() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(2).build();
        model = written(SyntheticModels.llama(c));
        UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                () -> new GGUFModelGPU(model.toString(), null, base().speculativeOptions(w12(W12F16Mode.REFUSE)).build()));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12));
        assertTrue(e.getMessage(), e.getMessage().contains("no W12 target"));
    }

    /** An F16 weight off the bfloat16 grid refuses the load by default: the W12 run stays lossless. */
    @Test
    public void anOffGridF16WeightIsRefusedByName() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(2)
                .weightType(GGMLType.F16).build();
        model = written(SyntheticModels.llama(c));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> new GGUFModelGPU(model.toString(), null, base().speculativeOptions(w12(W12F16Mode.REFUSE)).build()));
        assertTrue(e.getMessage(), e.getMessage().contains("is not a bfloat16 value"));
        assertTrue(e.getMessage(), e.getMessage().contains(".weight: row "));
        assertTrue(e.getMessage(), e.getMessage().contains(" element "));
        assertTrue(e.getMessage(), e.getMessage().contains(SpeculativeOptions.PROP_W12_F16));
    }

    /** The explicit {@code round} loads the F16 model on the device and reports the weights the rounding moved. */
    @Test
    public void roundBringsF16OntoTheGridAndReportsTheMovedWeights() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(2)
                .weightType(GGMLType.F16).build();
        model = written(SyntheticModels.llama(c));
        EngineConfig cfg = base().speculativeOptions(w12(W12F16Mode.ROUND)).build();
        try (LogCapture capture = new LogCapture(GpuWeightUploader.class);
             GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, cfg)) {
            GenerationRequest request = GenerationRequest.builder().maxNewTokens(4)
                    .sampler(cfg.legacySampler(0f, 1f, 1f, 0, 0f, 0)).build();
            int[] tokens = m.generate(new int[]{1, 17, 42, 5}, request, null).tokens();
            assertEquals(4, tokens.length);
            m.forwardBackend().reset();
            List<String> reports = capture.messages(Level.INFO,
                    "llama: " + SpeculativeOptions.PROP_W12_F16 + "=round moved ");
            assertEquals(1, reports.size());
            assertTrue(reports.get(0), reports.get(0).contains("weights onto the bfloat16 grid"));
            assertTrue(reports.get(0), reports.get(0).contains("largest relative change"));
        }
    }
}
