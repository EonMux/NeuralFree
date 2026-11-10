package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.acceleration.HWAcceleration;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.kernels.OpenClKernelLibrary;
import com.eonmux.neuralfree.acceleration.ocl.CLDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkDevice;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import org.junit.Assume;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;

import static org.lwjgl.vulkan.VK10.VK_SHADER_STAGE_COMPUTE_BIT;

/**
 * The backends of one API whose LLM kernels can include the pinned 16-lane subgroup forms
 * ({@link SubgroupRowForms}), for the GPU tests of those forms:
 * <ul>
 *   <li>OpenCL: every device that accepts a required 16-lane subgroup
 *       ({@link OpenClKernelLibrary#subgroupRowSize}).</li>
 *   <li>Vulkan: every Intel device whose subgroup size control lets compute pipelines pin 16 lanes, judged from the
 *       physical-device properties alone, so no logical device is created on any other GPU.</li>
 * </ul>
 * {@link #open} skips the test (JUnit assumption) when the API or such a device is absent.
 */
public final class SubgroupFormBackends implements AutoCloseable {

    private static final int PCI_VENDOR_INTEL = 0x8086;

    private final HWAcceleration hw;
    private final List<ComputeBackend> backends;

    private SubgroupFormBackends(HWAcceleration hw, List<ComputeBackend> backends) {
        this.hw = hw;
        this.backends = Collections.unmodifiableList(backends);
    }

    /** The API's backends that can include the forms; skips the calling test when there are none. */
    public static SubgroupFormBackends open(BackendType api) {
        List<ComputeBackend> backends = new ArrayList<>();
        HWAcceleration hw;
        switch (api) {
            case OPENCL:
                GpuTestUtil.assumeOpenCLAvailable();
                hw = new HWAcceleration(EnumSet.of(BackendType.OPENCL));
                for (CLDevice device : hw.getCLDevices()) {
                    ComputeBackend backend = hw.createBackend(device);
                    if (OpenClKernelLibrary.subgroupRowSize(backend.capabilities(), true) > 0) {
                        backends.add(backend);
                    } else {
                        backend.close();
                    }
                }
                break;
            case VULKAN:
                // not GpuTestUtil.assumeVulkanAvailable, which opens the first-ranked device
                hw = vulkanOrSkip();
                for (VkDevice device : hw.getVkDevices()) {
                    if (pinsSixteenLanes(device.getInfo())) {
                        backends.add(hw.createBackend(device));
                    }
                }
                break;
            default:
                throw new IllegalArgumentException("no subgroup forms on " + api);
        }
        if (backends.isEmpty()) {
            hw.shutdown();
            Assume.assumeTrue("no " + api + " device can pin 16-lane subgroups", false);
        }
        return new SubgroupFormBackends(hw, backends);
    }

    private static HWAcceleration vulkanOrSkip() {
        try {
            return new HWAcceleration(EnumSet.of(BackendType.VULKAN));
        } catch (RuntimeException | LinkageError e) {
            Assume.assumeNoException("Vulkan is unavailable", e);
            throw e;
        }
    }

    /** Whether a Vulkan device is an Intel GPU whose compute pipelines may pin 16-lane subgroups. */
    private static boolean pinsSixteenLanes(VkPhysicalDeviceInfo info) {
        return info.vendorId() == PCI_VENDOR_INTEL
                && (info.requiredSubgroupSizeStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0
                && info.minSubgroupSize() <= SubgroupRowForms.SIZE && SubgroupRowForms.SIZE <= info.maxSubgroupSize();
    }

    /** The opened backends, in enumeration order. */
    public List<ComputeBackend> backends() {
        return backends;
    }

    @Override
    public void close() {
        for (ComputeBackend b : backends) {
            b.close();
        }
        hw.shutdown();
    }
}
