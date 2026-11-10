package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Which devices {@link SubgroupRowForms#served} admits for the pinned 16-lane subgroup forms: an OpenCL program built
 * with {@code NF_SG_ROW_SIZE}, and an Intel Vulkan device whose subgroup range holds 16 lanes and whose kernels use
 * subgroups. NVIDIA (32 lanes fixed), AMD and unidentified devices keep the portable forms whatever their range.
 */
public class SubgroupRowFormsTest {

    private static final int INTEL = 0x8086;
    private static final int NVIDIA = 0x10DE;
    private static final int AMD = 0x1002;

    private static DeviceCapabilities caps(int vendor, int minSubgroup, int maxSubgroup) {
        return new DeviceCapabilities.Builder()
                .deviceName("test")
                .identity(new DeviceIdentity("test", vendor, null, null))
                .supportsSubgroups(true)
                .subgroupSize(maxSubgroup)
                .minSubgroupSize(minSubgroup)
                .maxSubgroupSize(maxSubgroup)
                .feature(DeviceFeature.SUBGROUP_ARITHMETIC)
                .build();
    }

    @Test
    public void anIntelVulkanDeviceWithSixteenLaneSubgroupsServesTheForms() {
        assertTrue(SubgroupRowForms.served(BackendType.VULKAN, caps(INTEL, 8, 32), true));
        assertTrue(SubgroupRowForms.served(BackendType.VULKAN, caps(INTEL, 16, 16), true));
    }

    @Test
    public void vulkanKernelsWithoutSubgroupsDoNotServeThem() {
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(INTEL, 8, 32), false));
    }

    @Test
    public void anIntelRangeWithoutSixteenLanesDoesNotServeThem() {
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(INTEL, 32, 32), true));
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(INTEL, 8, 8), true));
    }

    @Test
    public void otherVendorsKeepThePortableFormsOnVulkan() {
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(NVIDIA, 32, 32), true));
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(AMD, 32, 64), true));
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(AMD, 16, 64), true));
        assertFalse(SubgroupRowForms.served(BackendType.VULKAN, caps(DeviceIdentity.UNKNOWN_VENDOR, 8, 32), true));
    }

    @Test
    public void openClFollowsTheProgramBuildCondition() {
        for (DeviceCapabilities c : new DeviceCapabilities[]{caps(INTEL, 8, 32), caps(NVIDIA, 32, 32),
                caps(INTEL, 32, 32)}) {
            for (boolean subgroups : new boolean[]{true, false}) {
                assertEquals(OpenClKernelLibrary.subgroupRowSize(c, subgroups) > 0,
                        SubgroupRowForms.served(BackendType.OPENCL, c, subgroups));
            }
        }
    }

    @Test
    public void otherBackendsNeverServeThem() {
        for (BackendType b : new BackendType[]{BackendType.CUDA, BackendType.HIP, BackendType.METAL}) {
            assertFalse(b.name(), SubgroupRowForms.served(b, caps(INTEL, 8, 32), true));
        }
    }
}
