package com.eonmux.neuralfree.gguf.ops.fused;

import com.eonmux.neuralfree.acceleration.UnsupportedKernelException;
import com.eonmux.neuralfree.gguf.gpu.forward.FusedMode;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOpsFixture;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuScratch;
import com.eonmux.neuralfree.gguf.gpu.forward.ScratchSlots;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuPlaceholders;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The fused-kernel selection: the {@code neuralfree.gpu.fused} property, its refusal of unknown values, and the
 * three answers of {@link GpuOps#fused(String)} on a backend with and without the kernel.
 */
public class FusedModeTest {

    private String saved;
    private FakeComputeBackend backend;
    private GpuScratch scratch;
    private GpuPlaceholders placeholders;

    @Before
    public void open() {
        saved = System.getProperty(FusedMode.PROPERTY);
        System.clearProperty(FusedMode.PROPERTY);
        backend = FakeComputeBackend.create();
        scratch = GpuScratch.builder(backend, 1).slot(ScratchSlots.X, 16).build();
        placeholders = GpuPlaceholders.allocate(backend, backend.computeQueue());
    }

    @After
    public void close() {
        System.clearProperty(FusedMode.PROPERTY);
        if (saved != null) {
            System.setProperty(FusedMode.PROPERTY, saved);
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

    @Test
    public void configuredReadsTheProperty() {
        assertEquals(FusedMode.AUTO, FusedMode.configured());
        System.setProperty(FusedMode.PROPERTY, "off");
        assertEquals(FusedMode.OFF, FusedMode.configured());
        System.setProperty(FusedMode.PROPERTY, " ON ");
        assertEquals(FusedMode.ON, FusedMode.configured());
        System.setProperty(FusedMode.PROPERTY, "");
        assertEquals(FusedMode.AUTO, FusedMode.configured());
    }

    @Test
    public void configuredRefusesAnUnknownValueByName() {
        System.setProperty(FusedMode.PROPERTY, "sometimes");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, FusedMode::configured);
        assertTrue(e.getMessage(), e.getMessage().contains(FusedMode.PROPERTY)
                && e.getMessage().contains("sometimes"));
    }

    @Test
    public void theModeGatesEveryFusedAnswer() {
        assertTrue(FusedMode.AUTO.permitsFusion());
        assertFalse(FusedMode.AUTO.requiresFusion());
        assertFalse(FusedMode.OFF.permitsFusion());
        assertTrue(FusedMode.ON.requiresFusion());

        GpuOps off = GpuOpsFixture.withFusedMode(backend.computeQueue(), LlmKernels.open(backend), scratch,
                placeholders, FusedMode.OFF);
        GpuOps auto = GpuOpsFixture.withFusedMode(backend.computeQueue(), LlmKernels.open(backend), scratch,
                placeholders, FusedMode.AUTO);
        try {
            assertFalse(off.fused(FusedKernels.RMS_NORM_ROPE));
            assertTrue(auto.fused(FusedKernels.RMS_NORM_ROPE));
        } finally {
            auto.close();
            off.close();
        }
    }

    @Test
    public void aRequiredKernelTheBackendLacksIsRefusedByName() {
        FakeComputeBackend bare = FakeComputeBackend.builder()
                .unavailableKernels(FusedKernels.RMS_NORM_ROPE).build();
        GpuScratch s = GpuScratch.builder(bare, 1).slot(ScratchSlots.X, 16).build();
        GpuPlaceholders ph = GpuPlaceholders.allocate(bare, bare.computeQueue());
        GpuOps ops = GpuOpsFixture.withFusedMode(bare.computeQueue(), LlmKernels.open(bare), s, ph,
                FusedMode.ON);
        try {
            UnsupportedKernelException e = assertThrows(UnsupportedKernelException.class,
                    () -> ops.fused(FusedKernels.RMS_NORM_ROPE));
            assertTrue(e.getMessage(), e.getMessage().contains(FusedKernels.RMS_NORM_ROPE)
                    && e.getMessage().contains(FusedMode.PROPERTY));
            assertTrue(ops.fused(FusedKernels.MULTI_TASK));
        } finally {
            ops.close();
            ph.close();
            s.close();
            bare.close();
        }
    }
}
