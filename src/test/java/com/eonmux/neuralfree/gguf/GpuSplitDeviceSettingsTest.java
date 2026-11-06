package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.demo.tools.CvectorGenerator;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static com.eonmux.neuralfree.gguf.SplitFixtures.TOL;
import static com.eonmux.neuralfree.gguf.SplitFixtures.assertParity;
import static com.eonmux.neuralfree.gguf.SplitFixtures.maxAbsDiff;
import static com.eonmux.neuralfree.gguf.SplitFixtures.oneDevice;
import static com.eonmux.neuralfree.gguf.SplitFixtures.run;
import static com.eonmux.neuralfree.gguf.SplitFixtures.split;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Device settings over the OpenCL devices of the machine: control vectors across a two-device layer split (every
 * device applies the vectors of its own layers, the host the vectors of the host layers), {@code --main-gpu} picking
 * the one GPU of {@code --split-mode none} from the {@code --device} list, and {@code neuralfree.gpu.vram.limit}
 * capping the VRAM of every device so the split moves layers to the host before streaming, as one GPU does; each
 * against the one-device load. A single {@code --device} index ({@link SplitFixtures#oneDevice(String)}) picks its
 * GPU in enumeration order.
 */
public class GpuSplitDeviceSettingsTest {

    private static final int LAYERS = 4;
    private static final float SCALE = 1.5f;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<Path> files = new ArrayList<>();

    @Before
    public void requireTwoDevices() {
        SplitFixtures.assumeTwoDevices();
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private Path llama() throws IOException {
        return written(SyntheticModels.llama(SyntheticModels.Config.llama().toBuilder().nLayer(LAYERS).build()));
    }

    // ------------------------------------------------------------------ control vectors

    @Test
    public void controlVectorsAcrossTheSplitMatchTheOneDeviceLoad() throws IOException {
        Path model = llama();
        String vector = vector(SyntheticModels.Config.llama().nEmbd()).toString();
        SplitFixtures.Run reference = run(model, oneDevice().addControlVector(vector, SCALE).build());
        SplitFixtures.Run controlled = run(model, split().addControlVector(vector, SCALE).build());
        SplitFixtures.Run plain = run(model, split().build());
        assertParity("control vectors over the split", reference, controlled);
        assertTrue("the vectors change the split's logits",
                maxAbsDiff(plain.logits[0], controlled.logits[0]) > 100 * TOL);
    }

    @Test
    public void controlVectorsOfHostLayersApplyOnTheHost() throws IOException {
        Path model = llama();
        String vector = vector(SyntheticModels.Config.llama().nEmbd()).toString();
        SplitFixtures.Run reference = run(model, oneDevice().gpuLayers(3).addControlVector(vector, SCALE).build());
        SplitFixtures.Run controlled = run(model, split().gpuLayers(3).addControlVector(vector, SCALE).build());
        assertTrue(controlled.backend, controlled.backend.contains("host layers [0, 2)"));
        assertParity("control vectors over host layers and the split", reference, controlled);
    }

    /** A control vector of one random direction per layer. */
    private Path vector(int nEmbd) throws IOException {
        List<float[]> rows = new ArrayList<>();
        for (int il = 0; il < LAYERS; il++) {
            Random r = new Random(900 + il);
            float[] row = new float[nEmbd];
            for (int i = 0; i < nEmbd; i++) {
                row[i] = 0.2f * (r.nextFloat() - 0.5f);
            }
            rows.add(row);
        }
        return CvectorGenerator.export(rows, nEmbd, "llama", tmp.getRoot().toPath().resolve("cv.gguf"), true);
    }

    // ------------------------------------------------------------------ main GPU

    @Test
    public void theMainGpuPicksTheOneGpuOfSplitModeNone() throws IOException {
        Path model = llama();
        EngineConfig picked = oneDevice("0").addGpuDevice("1").mainGpu(1).build();
        assertEquals(EngineConfig.SplitMode.NONE, picked.splitMode());
        String expected;
        SplitFixtures.Run reference;
        try (GGUFModelGPU direct = new GGUFModelGPU(model.toString(), null, oneDevice("1").build())) {
            expected = direct.backend().capabilities().getDeviceName();
            reference = new SplitFixtures.Run(SplitFixtures.logits(direct), SplitFixtures.greedy(direct), "");
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, picked)) {
            assertEquals("--main-gpu 1 runs the model on the second --device", expected,
                    m.backend().capabilities().getDeviceName());
            assertEquals(1, m.deviceCount());
            assertParity("-mg 1", reference, new SplitFixtures.Run(SplitFixtures.logits(m), SplitFixtures.greedy(m),
                    ""));
        }
        try (GGUFModelGPU first = new GGUFModelGPU(model.toString(), null,
                oneDevice("1").addGpuDevice("0").build())) {
            assertEquals("without --main-gpu the first --device runs the model", expected,
                    first.backend().capabilities().getDeviceName());
        }
    }

    // ------------------------------------------------------------------ VRAM cap

    @Test
    public void aVramCapMovesLayersToTheHostBeforeStreaming() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama().toBuilder().nLayer(8).nEmbd(256).nHead(4)
                .nHeadKv(2).headDim(64).nRot(64).nFf(1024).build();
        Path model = written(SyntheticModels.llama(c));
        SplitFixtures.Run reference = run(model, oneDevice().build());
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, split().gpuVramLimitMb(10).build())) {
            assertTrue("the capped split moves its first layers to the host: " + m.forwardBackend(),
                    m.hostLayerCount() > 0);
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNull("device " + d + " holds its span resident beside the host layers", m.streamingOf(d));
            }
            assertParity("host-layer split under the cap", reference, new SplitFixtures.Run(SplitFixtures.logits(m),
                    SplitFixtures.greedy(m), m.forwardBackend().toString()));
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                split().gpuVramLimitMb(10).gpuLayers(EngineConfig.GPU_LAYERS_ALL).build())) {
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNotNull("-ngl all streams the span of device " + d, m.streamingOf(d));
            }
            assertParity("streamed split under -ngl all", reference, new SplitFixtures.Run(SplitFixtures.logits(m),
                    SplitFixtures.greedy(m), m.forwardBackend().toString()));
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                oneDevice().gpuVramLimitMb(20).gpuLayers(EngineConfig.GPU_LAYERS_ALL).build())) {
            assertNotNull("the cap streams a one-device load that keeps every layer on the device", m.streamingOf(0));
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null,
                oneDevice().gpuVramLimitMb(20).build())) {
            assertTrue("with -ngl auto the capped one-device load offloads its first layers to the host: "
                    + m.forwardBackend(), m.forwardBackend().toString().contains("Hybrid"));
        }
        try (GGUFModelGPU m = new GGUFModelGPU(model.toString(), null, split().build())) {
            for (int d = 0; d < m.deviceCount(); d++) {
                assertNull("without the cap device " + d + " holds its span resident", m.streamingOf(d));
            }
        }
    }
}
