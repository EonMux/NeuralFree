package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.FusedMode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOpsFixture;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Boundary rules of {@link GpuMultiTask} on the host-only fake backend: the kernel's task and buffer limits, the
 * row bodies it runs, and the argument checks of the rotations. Nothing executes on a device.
 */
public class GpuMultiTaskTest {

    private static final int UBATCH = 4;
    private static final int WIDTH = 32;

    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;
    private GpuOps ops;
    private final List<GpuBuffer> buffers = new ArrayList<>();

    @Before
    public void open() {
        backend = FakeComputeBackend.create();
        scratch = GpuScratch.builder(backend, UBATCH).slot(ScratchSlots.X, (long) UBATCH * WIDTH).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
        ops = GpuOpsFixture.withFusedMode(backend.computeQueue(), LlmKernels.open(backend), scratch, placeholders,
                FusedMode.AUTO);
    }

    @After
    public void close() {
        if (ops != null) {
            ops.close();
        }
        for (GpuBuffer b : buffers) {
            b.close();
        }
        if (placeholders != null) {
            placeholders.close();
        }
        if (scratch != null) {
            scratch.close();
        }
        if (backend != null) {
            backend.close();
        }
    }

    private GpuBuffer buffer() {
        GpuBuffer b = backend.allocateBuffer((long) UBATCH * WIDTH * Float.BYTES);
        buffers.add(b);
        return b;
    }

    private static RopeSpec neox(int nDims) {
        return RopeSpec.builder(RopeType.NEOX).nDims(nDims).nOffs(0).freqBase(10000f).build();
    }

    private void rowNorm(GpuMultiTask batch, GpuBuffer x, GpuBuffer w) {
        batch.rowNorm(1, WIDTH, WIDTH, NormKind.RMS, 0, 1, 0, 0, 1e-6f, x, w, placeholders.readOnlyFloats(),
                buffer());
    }

    @Test
    public void usableFollowsTheKernelLibrary() {
        assertTrue(GpuMultiTask.usable(ops));
        FakeComputeBackend bare = FakeComputeBackend.builder()
                .unavailableKernels(FusedKernels.MULTI_TASK).build();
        GpuScratch s = GpuScratch.builder(bare, UBATCH).slot(ScratchSlots.X, (long) UBATCH * WIDTH).build();
        GpuPlaceholders ph = GpuPlaceholders.allocate(bare, bare.computeQueue());
        try {
            GpuOps without = GpuOpsFixture.withFusedMode(bare.computeQueue(), LlmKernels.open(bare), s, ph,
                    FusedMode.AUTO);
            try {
                assertFalse(GpuMultiTask.usable(without));
            } finally {
                without.close();
            }
        } finally {
            ph.close();
            s.close();
            bare.close();
        }
    }

    @Test
    public void localSizeIsAPowerOfTwoUpToTheKernelLimit() {
        assertTrue(GpuMultiTask.localUsable(1));
        assertTrue(GpuMultiTask.localUsable(64));
        assertTrue(GpuMultiTask.localUsable(256));
        assertFalse(GpuMultiTask.localUsable(0));
        assertFalse(GpuMultiTask.localUsable(96));
        assertFalse(GpuMultiTask.localUsable(512));
    }

    @Test
    public void rowNormRunsOnlyTheBodiesTheKernelHas() {
        GpuMultiTask batch = new GpuMultiTask(ops, UBATCH, "test");
        try {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> batch.rowNorm(1, WIDTH, WIDTH, NormKind.LAYER, 0, 1, 0, 0, 1e-6f, buffer(), buffer(),
                            placeholders.readOnlyFloats(), buffer()));
            assertTrue(e.getMessage(), e.getMessage().contains(FusedKernels.MULTI_TASK)
                    && e.getMessage().contains(NormKind.LAYER.name()));
        } finally {
            batch.close();
        }
    }

    @Test
    public void normRopeRefusesARotationWiderThanTheRow() {
        GpuMultiTask batch = new GpuMultiTask(ops, UBATCH, "test");
        try {
            GpuBuffer b = buffer();
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> batch.normRope(neox(16), 1e-6f, 2, 16, 32, 8, 0, 0, 0, 0, 1, false,
                            placeholders.readOnlyInts(), placeholders.readOnlyFloats(), b,
                            placeholders.readOnlyFloats(), b, b));
            assertTrue(e.getMessage(), e.getMessage().contains(FusedKernels.MULTI_TASK));
        } finally {
            batch.close();
        }
    }

    @Test
    public void aLaunchHoldsAtMostFourTasks() {
        GpuMultiTask batch = new GpuMultiTask(ops, UBATCH, "test");
        try {
            for (int i = 0; i < FusedKernels.MULTI_TASK_MAX_TASKS; i++) {
                rowNorm(batch, buffer(), buffer());
            }
            assertEquals(FusedKernels.MULTI_TASK_MAX_TASKS, batch.taskCount());
            assertThrows(IllegalStateException.class, () -> rowNorm(batch, buffer(), buffer()));
        } finally {
            batch.close();
        }
    }

    @Test
    public void aLaunchAddressesAtMostElevenBuffers() {
        GpuMultiTask batch = new GpuMultiTask(ops, UBATCH, "test");
        try {
            // Each task binds four distinct buffers, so three tasks need twelve.
            for (int i = 0; i < 3; i++) {
                batch.rowNorm(1, WIDTH, WIDTH, NormKind.RMS, 0, 1, 0, 0, 1e-6f, buffer(), buffer(), buffer(),
                        buffer());
            }
            assertThrows(IllegalArgumentException.class, batch::build);
        } finally {
            batch.close();
        }
    }

    @Test
    public void runRequiresABuild() {
        GpuMultiTask batch = new GpuMultiTask(ops, UBATCH, "test");
        try {
            rowNorm(batch, buffer(), buffer());
            assertThrows(IllegalStateException.class, () -> batch.run(1));
        } finally {
            batch.close();
        }
    }
}
