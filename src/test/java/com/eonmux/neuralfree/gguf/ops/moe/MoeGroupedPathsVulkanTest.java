package com.eonmux.neuralfree.gguf.ops.moe;

import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.vulkan.VkComputeBackend;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * {@link MoeGroupedPathsParity} on a Vulkan device, over the GLSL twins ({@code moe_misc.comp},
 * {@code moe_gemm.comp}, and {@code moe_gemm_mma.comp} where the device has cooperative matrices). Skips without
 * a Vulkan GPU.
 */
public class MoeGroupedPathsVulkanTest extends MoeGroupedPathsParity {

    private VkComputeBackend backend;

    @Before
    public void setUp() {
        try {
            backend = new VkComputeBackend();
            h = new GroupedGemmHarness(backend, LlmKernels.open(backend));
        } catch (Throwable e) {
            if (backend != null) {
                backend.close();
            }
            backend = null;
        }
        Assume.assumeTrue("no Vulkan GPU", backend != null);
    }

    /**
     * On a device with {@code VK_KHR_cooperative_matrix} the grouped GEMM takes the cooperative-matrix shaders in
     * both forms, except for F32 and BF16 experts, which keep {@code moe_gemm.comp}.
     */
    @Test
    public void cooperativeMatrixDevicesPickTheTensorCoreGroupedGemm() {
        Assume.assumeTrue("the device reports no cooperative matrices",
                backend.capabilities().hasFeature(DeviceFeature.COOPERATIVE_MATRIX));
        assertEquals("F16_GroupedGemmMma", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.F16, false));
        assertEquals("Q8_0_GroupedGemmMma", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.Q8_0, false));
        assertEquals("Q4_K_GroupedGemmMmaInd", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.Q4_K, true));
        assertEquals("BF16_GroupedGemm", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.BF16, false));
        assertEquals("BF16_GroupedGemmInd", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.BF16, true));
        assertEquals("F32_GroupedGemmInd", GpuGroupedRows.groupedKernel(h.ops(), GGMLType.F32, true));
    }

    @After
    public void tearDown() {
        if (h != null) {
            h.close();
        }
        if (backend != null) {
            backend.close();
        }
    }
}
