package com.eonmux.neuralfree.acceleration.kernels;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import org.junit.Test;

import java.util.EnumSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The engine's kernel policy on an accelerator backend: subgroup reductions follow the device's
 * reported feature (as on Vulkan), the preferred row-reduce work-group size uses the subgroup
 * formula, and the flash-attention shared-memory size stays within the device's shared memory.
 */
public class AcceleratorKernelPolicyTest {

    private static DeviceCapabilities caps(int subgroupSize, long maxWorkGroupSize, long sharedBytes,
                                           boolean subgroupArithmetic) {
        DeviceCapabilities.Builder builder = new DeviceCapabilities.Builder()
                .deviceName("accelerator")
                .computeUnits(64)
                .maxClockFrequencyMhz(1800)
                .maxWorkGroupSize(maxWorkGroupSize)
                .subgroupSize(subgroupSize)
                .maxSharedMemoryBytes(sharedBytes);
        if (subgroupArithmetic) {
            builder.features(EnumSet.of(DeviceFeature.SUBGROUP_ARITHMETIC));
        }
        return builder.build();
    }

    @Test
    public void subgroupsFollowTheDeviceFeature() {
        assertTrue(KernelSet.subgroups(BackendType.SYCL, caps(32, 1024, 64 * 1024, true)));
        assertFalse(KernelSet.subgroups(BackendType.SYCL, caps(32, 1024, 64 * 1024, false)));
        assertTrue("CUDA always uses its warp reductions", KernelSet.subgroups(BackendType.CUDA,
                caps(32, 1024, 64 * 1024, false)));
    }

    @Test
    public void theMatvecWorkGroupUsesTheSubgroupFormula() {
        assertEquals(64, KernelSet.preferredMatvecLocal(BackendType.SYCL, caps(32, 1024, 64 * 1024, false)));
        assertEquals(128, KernelSet.preferredMatvecLocal(BackendType.XPU, caps(64, 1024, 64 * 1024, false)));
        assertEquals("the device's maximum work-group size caps it", 32,
                KernelSet.preferredMatvecLocal(BackendType.MUSA, caps(32, 32, 64 * 1024, false)));
    }

    @Test
    public void theFlashAttentionSharedMemoryStaysWithinTheDevice() {
        assertEquals(FaSmem.MAX_FLOATS, FaSmem.faSmemFloats(BackendType.SYCL, caps(32, 1024, 64 * 1024, false)));
        assertEquals(3072, FaSmem.faSmemFloats(BackendType.CANN, caps(32, 1024, 16 * 1024, false)));
    }
}
