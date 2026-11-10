package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Selection of the tensor-core prefill kernel ({@code LLM_FlashAttnMma*}) in {@link FlashPlan}:
 * the kernel name follows the cooperative-matrix feature, the F16 cache types and the head dims; a backend
 * without the feature (or without the kernel) keeps the scalar tiled kernel. The one-row step routes through
 * the mma prefill only where the decode fast path cannot serve the step (the plan-level refusals, a paged
 * cache, or at most one workgroup of live keys) and the mma kernel exists. The mixer dispatches the base
 * kernel over a dense cache (its identity paging); the paged kernel serves launches over a
 * paginated cache. The one-row decode kernel {@code LLM_FlashDecodeRow_F16} serves the plain step of a dense F16
 * cache, which then never routes through the mma prefill; on Vulkan it serves NVIDIA devices only.
 */
public class GpuAttentionMixerMmaSelectionTest {

    @Test
    public void f16CacheOnAVectorBackendSelectsTheMmaKernel() {
        assertEquals("LLM_FlashAttnMma_F16",
                FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.F16, 64, 64));
        assertEquals("LLM_FlashAttnMma_F16",
                FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.F16, 128, 128));
        assertEquals("LLM_FlashAttnMma_F16",
                FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.F16, 80, 80));
    }

    @Test
    public void scalarKernelStaysForEveryOtherCase() {
        // no tensor cores on the backend
        assertNull(FlashPlan.mmaKernelName(false, KvDType.F16, KvDType.F16, 64, 64));
        // quantized, mixed-type or F32 caches
        assertNull(FlashPlan.mmaKernelName(true, KvDType.Q8_0, KvDType.Q8_0, 64, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.Q8_0, 64, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.F32, KvDType.F32, 64, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.BF16, KvDType.BF16, 64, 64));
        // head dims outside the mma plan: not multiples of 16, or above the staged tile budget
        assertNull(FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.F16, 72, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.F16, 64, 256));
    }

    @Test
    public void aDepthRoutedStepLeavesTheMmaPrefillPastOneWorkgroupOfKeys() {
        String mma = "LLM_FlashAttnMma_F16";
        assertFalse(FlashPlan.routesDecodeThroughPrefill(mma, true, 129));
        assertFalse(FlashPlan.routesDecodeThroughPrefill(mma, true, Integer.MAX_VALUE));
        assertTrue(FlashPlan.routesDecodeThroughPrefill(mma, true, 128));
        assertTrue(FlashPlan.routesDecodeThroughPrefill(mma, true, 0));
        // A plan whose route does not follow the depth keeps the mma prefill at every step.
        assertTrue(FlashPlan.routesDecodeThroughPrefill(mma, false, 129));
        assertTrue(FlashPlan.routesDecodeThroughPrefill(mma, false, 1000));
        // No mma plan: the decode kernel serves every step.
        assertFalse(FlashPlan.routesDecodeThroughPrefill(null, false, 0));
        assertFalse(FlashPlan.routesDecodeThroughPrefill(null, true, 1000));
    }

    @Test
    public void theRouteFollowsTheDepthOnlyForADenseFastPathPlanWithASmallGqaGroup() {
        String mma = "LLM_FlashAttnMma_F16";
        assertTrue(FlashPlan.depthRouted(mma, true, false, 16, 8));
        assertTrue(FlashPlan.depthRouted(mma, true, false, 8, 8));
        // A group of four query heads per KV head: the mma prefill's shared K/V tile wins at every depth.
        assertFalse(FlashPlan.depthRouted(mma, true, false, 32, 8));
        // No shuffle fast path (Vulkan, OpenCL, refused caches or dims), a paged cache, or no mma plan.
        assertFalse(FlashPlan.depthRouted(mma, false, false, 16, 8));
        assertFalse(FlashPlan.depthRouted(mma, true, true, 16, 8));
        assertFalse(FlashPlan.depthRouted(null, true, false, 16, 8));
    }

    @Test
    public void theRowKernelServesThePlainStepOfADenseF16Cache() {
        String row = AttnKernels.FLASH_DECODE_ROW_F16;
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE, false,
                1024, 1024, false));
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 64, 64, BiasMode.NONE, false,
                512, 512, false));
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 256, 256, BiasMode.NONE, false,
                2048, 2048, false));
    }

    @Test
    public void theGenericDecodeKernelServesEveryOtherStep() {
        // other cache types, or V read through the K rows
        assertNull(FlashPlan.rowKernelName(KvDType.Q8_0, KvDType.Q8_0, false, 128, 128, BiasMode.NONE, false,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F32, false, 128, 128, BiasMode.NONE, false,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, true, 128, 128, BiasMode.NONE, false,
                1024, 1024, false));
        // head dims: unequal, not a multiple of 64, above the kernel's 256
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 64, BiasMode.NONE, false,
                1024, 512, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 96, 96, BiasMode.NONE, false,
                768, 768, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 512, 512, BiasMode.NONE, false,
                4096, 4096, false));
        // position biases, sink logits, odd strides, a paged cache
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.ALIBI, false,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE, true,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE, false,
                1025, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE, false,
                1024, 1024, true));
    }

    @Test
    public void theRowKernelServesVulkanOnNvidiaDevicesOnly() {
        assertTrue(rowPath(BackendType.CUDA, 0x10DE));
        assertTrue(rowPath(BackendType.VULKAN, 0x10DE));
        assertFalse("AMD", rowPath(BackendType.VULKAN, 0x1002));
        assertFalse("Intel", rowPath(BackendType.VULKAN, 0x8086));
        assertFalse("Apple through MoltenVK", rowPath(BackendType.VULKAN, 0x106B));
        assertFalse("ARM Mali", rowPath(BackendType.VULKAN, 0x13B5));
        assertFalse("Qualcomm Adreno", rowPath(BackendType.VULKAN, 0x5143));
        assertFalse("an unknown vendor", rowPath(BackendType.VULKAN, DeviceIdentity.UNKNOWN_VENDOR));
    }

    private static boolean rowPath(BackendType type, int vendorId) {
        DeviceCapabilities caps = new DeviceCapabilities.Builder().deviceName("Device").totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30).computeUnits(32).maxClockFrequencyMhz(1500).maxWorkGroupSize(1024)
                .driverVersion("fake-1.0").supportsSubgroups(true).subgroupSize(32).maxLocalMemoryBytes(48L * 1024)
                .computeCapability(type == BackendType.CUDA ? 120 : 0).feature(DeviceFeature.STORAGE_8BIT)
                .feature(DeviceFeature.STORAGE_16BIT).feature(DeviceFeature.TIMELINE_SEMAPHORE)
                .identity(new DeviceIdentity("Device", vendorId, null, null)).build();
        FakeComputeBackend fake = FakeComputeBackend.builder().backendType(type).capabilities(caps).build();
        try {
            return LlmKernels.open(fake).flashDecodeRowPath();
        } finally {
            fake.close();
        }
    }
}
