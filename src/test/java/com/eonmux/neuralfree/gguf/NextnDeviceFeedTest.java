package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.flow.DraftModelLoader;
import com.eonmux.neuralfree.gguf.ops.flow.MtpHead;
import com.eonmux.neuralfree.gguf.ops.flow.NextnDraftModel;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/**
 * The device feed of a NextN drafter over a one-device GPU target: the target's hidden rows are captured on the
 * device ({@code TargetDeviceFeed} over the context the head shares) and the drafts they produce are the same
 * tokens the host feed drafts over the same calls, verifications and rollbacks included. The host path is the
 * device one with the capture switched off ({@code neuralfree.draft.mtp.device.rows}), on the same weights and
 * device, so the comparison is deterministic.
 */
public class NextnDeviceFeedTest {

    private static final int[] PROMPT = {1, 17, 42, 5, 99, 26};
    private static final String DEVICE_ROWS = "neuralfree.draft.mtp.device.rows";

    private final List<Path> files = new ArrayList<>();
    private String deviceRowsProperty;

    @Before
    public void requireOpenCL() {
        GpuTestUtil.assumeOpenCLAvailable();
        deviceRowsProperty = System.clearProperty(DEVICE_ROWS);
    }

    @After
    public void cleanUp() throws IOException {
        if (deviceRowsProperty != null) {
            System.setProperty(DEVICE_ROWS, deviceRowsProperty);
        }
        for (Path p : files) {
            Files.deleteIfExists(p);
        }
    }

    /** Two rounds of speculation over one target: a prompt draft and a fused catch-up after a partial accept. */
    private static int[][] round(NextnDraftModel d, GGUFModelGPU target) {
        int[] verify = new int[3];
        target.forwardBackend().forwardRows(PROMPT, 0, null);
        int[] first = d.draft(PROMPT[PROMPT.length - 1], 3);
        verify[0] = PROMPT[PROMPT.length - 1];
        System.arraycopy(first, 0, verify, 1, Math.min(2, first.length));
        target.forwardBackend().forwardRows(verify, PROMPT.length, null);
        target.forwardBackend().rollback(1);
        d.accept(1);
        int[] second = d.draft(60, 2);
        return new int[][]{first, second};
    }

    @Test
    public void theDeviceFeedDraftsWhatTheHostFeedDrafts() throws IOException {
        Path model = written(SyntheticModels.glm4moe(
                SyntheticModels.Config.qwen3moe().toBuilder().nLayer(2).build(), 1));
        EngineConfig cfg = base().build();
        int[][] device;
        int[][] host;
        try (GGUFModelGPU target = new GGUFModelGPU(model.toString(), null, cfg)) {
            assertSame("the drafter shares the target's context, so the rows stay on the device",
                    target.forwardContext(), target.nextnContext());
            try (NextnDraftModel d = drafter(target)) {
                assertTrue("the device feed is bound: " + d, d.toString().contains("device rows"));
                device = round(d, target);
            }
            target.forwardBackend().reset();
            System.setProperty(DEVICE_ROWS, "false");
            try (NextnDraftModel d = drafter(target)) {
                host = round(d, target);
            } finally {
                System.clearProperty(DEVICE_ROWS);
            }
        }
        assertArrayEquals(device[0], host[0]);
        assertArrayEquals(device[1], host[1]);
    }

    private Path written(GgufTestWriter w) throws IOException {
        Path p = w.writeTempFile();
        files.add(p);
        return p;
    }

    private static NextnDraftModel drafter(GGUFModelGPU target) {
        target.ensureNextnWeights();
        MtpHead head = MtpHead.gpu(target.nextnContext(), target.blockRegistry(), null);
        return (NextnDraftModel) DraftModelLoader.nextn(target, head, DraftModelLoader.Options.DEFAULT);
    }

    private static EngineConfig.Builder base() {
        MemoryConfig memory = MemoryConfig.builder().nCtx(64).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32)
                .build();
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory).mtp(true);
    }
}
