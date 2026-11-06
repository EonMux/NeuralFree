package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.engine.EngineConfig.DraftType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

/**
 * A separate draft model's own devices ({@code -devd}, {@link EngineConfig#draftDevices()}) reach the model the
 * session opens: it opens on the GPUs the list names beside a CPU target, on the host for the lone {@code none}
 * even beside a GPU target, and beside the target's weights when no list is set. The draft model is visible through
 * {@link ModelDrafter#draftModel()}, so the test fails when the placement is ignored instead of silently loading
 * the draft on the target's backend.
 */
public class DraftDevicePlacementTest {

    private final List<Path> files = new ArrayList<>();
    private Path target;
    private Path draft;

    @Before
    public void writeModels() throws IOException {
        SyntheticModels.Config c = SyntheticModels.Config.llama();
        target = written(SyntheticModels.llama(c.toBuilder().seed(11L).build()));
        draft = written(SyntheticModels.llama(c.toBuilder().seed(12L).build()));
    }

    @After
    public void cleanUp() throws IOException {
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    private Path written(com.eonmux.neuralfree.gguf.testutil.GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private static MemoryConfig memory() {
        return MemoryConfig.defaults().toBuilder().nCtx(64).ubatch(16).build();
    }

    /** A CPU load; the draft list {@code devices} decides where the draft itself opens. */
    private EngineConfig cpu(List<String> devices) {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory())
                .draftModelPath(draft.toString()).draftType(DraftType.MODEL).draftDevices(devices).draftMax(1)
                .build();
    }

    /** The draft model a session over {@code cfg} opened. */
    private EngineModel draftOf(EngineConfig cfg) throws IOException {
        try (SpeculativeSession session = LLMModelFactory.openSpeculative(target.toString(), null, cfg)) {
            assertTrue("the draft file runs as a separate model", session.drafter() instanceof ModelDrafter);
            return ((ModelDrafter) session.drafter()).draftModel();
        }
    }

    @Test
    public void noneKeepsTheDraftOnTheHostBesideACpuTarget() throws IOException {
        assertFalse("the lone none keeps the draft off every GPU",
                draftOf(cpu(List.of(EngineConfig.DRAFT_NO_DEVICE))) instanceof GGUFModelGPU);
    }

    @Test
    public void noListKeepsTheDraftBesideTheCpuTargetsWeights() throws IOException {
        assertFalse("unset devices follow the target's backend",
                draftOf(cpu(List.of())) instanceof GGUFModelGPU);
    }

    @Test
    public void aDraftDeviceListOpensTheDraftOnItsOwnDevice() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        assertTrue("a listed device opens the draft on a GPU beside a CPU target",
                draftOf(cpu(List.of("0"))) instanceof GGUFModelGPU);
    }

    @Test
    public void noneKeepsTheDraftOnTheHostBesideAGpuTarget() throws IOException {
        GpuTestUtil.assumeOpenCLAvailable();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory()).draftModelPath(draft.toString()).draftType(DraftType.MODEL)
                .draftDevices(List.of(EngineConfig.DRAFT_NO_DEVICE)).draftMax(1).build();
        try (SpeculativeSession session = LLMModelFactory.openSpeculative(target.toString(), null, cfg)) {
            assumeTrue("the target opened on the GPU", session.target() instanceof GGUFModelGPU);
            assertFalse("none overrides following the target onto its device",
                    ((ModelDrafter) session.drafter()).draftModel() instanceof GGUFModelGPU);
        }
    }
}
