package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceFeature;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.kernels.LlmKernels;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec.BiasMode;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Selection of the tensor-core prefill kernel ({@code LLM_FlashAttnMma*}) in {@link FlashPlan}:
 * the kernel name follows the cooperative-matrix feature, the cache types (F16 or Q8_0 for both) and the head
 * dims; a backend without the feature (or without the kernel) keeps the scalar tiled kernel. The one-row step routes through
 * the mma prefill only where the decode fast path cannot serve the step (the plan-level refusals, a paged
 * cache, a Q8_0 cache, or at most one workgroup of live keys) and the mma kernel exists. The mixer dispatches the base
 * kernel over a dense cache (its identity paging); the paged kernel serves launches over a
 * paginated cache. The one-row decode kernels (the per-head {@code LLM_FlashDecodeRow_F16} for a dense F16 cache, the
 * grouped-query {@code LLM_FlashDecodeGqa_<KV>} where it serves) take the plain step, which then never routes through
 * the mma prefill; on Vulkan they serve NVIDIA devices only.
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
    public void q8CacheSelectsItsMmaKernel() {
        assertEquals("LLM_FlashAttnMma_Q8_0",
                FlashPlan.mmaKernelName(true, KvDType.Q8_0, KvDType.Q8_0, 128, 128));
        assertNull(FlashPlan.mmaKernelName(false, KvDType.Q8_0, KvDType.Q8_0, 128, 128));
    }

    @Test
    public void scalarKernelStaysForEveryOtherCase() {
        // no tensor cores on the backend
        assertNull(FlashPlan.mmaKernelName(false, KvDType.F16, KvDType.F16, 64, 64));
        // 4-bit, mixed-type or F32 caches
        assertNull(FlashPlan.mmaKernelName(true, KvDType.Q4_0, KvDType.Q4_0, 64, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.F16, KvDType.Q8_0, 64, 64));
        assertNull(FlashPlan.mmaKernelName(true, KvDType.Q8_0, KvDType.F16, 64, 64));
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
        KvDType f16 = KvDType.F16;
        assertTrue(FlashPlan.depthRouted(mma, true, f16, f16, false, 16, 8));
        assertTrue(FlashPlan.depthRouted(mma, true, f16, f16, false, 8, 8));
        // A group of four query heads per KV head: the mma prefill's shared K/V tile wins at every depth.
        assertFalse(FlashPlan.depthRouted(mma, true, f16, f16, false, 32, 8));
        // No shuffle fast path (Vulkan, OpenCL, refused caches or dims), a paged cache, or no mma plan.
        assertFalse(FlashPlan.depthRouted(mma, false, f16, f16, false, 16, 8));
        assertFalse(FlashPlan.depthRouted(mma, true, f16, f16, true, 16, 8));
        assertFalse(FlashPlan.depthRouted(null, true, f16, f16, false, 16, 8));
        // A Q8_0 cache: the shuffle fast path serves it, the one-row step still keeps the mma prefill.
        KvDType q8 = KvDType.Q8_0;
        assertFalse(FlashPlan.depthRouted("LLM_FlashAttnMma_Q8_0", true, q8, q8, false, 8, 8));
        assertFalse(FlashPlan.depthRouted("LLM_FlashAttnMma_Q8_0", true, q8, q8, false, 16, 8));
    }

    /**
     * The plans of a CUDA tensor-core device for dense-cache layers that neither one-row kernel serves (one query
     * head per KV head of 128 dims, two of 96 dims): over an F16 cache the one-row step follows the depth, over a
     * Q8_0 cache it runs the Q8_0 mma prefill at every depth although the split-K decode's fast path serves Q8_0.
     */
    @Test
    public void aQ8CachesOneRowStepRunsTheMmaPrefillAtEveryDepth() {
        FakeComputeBackend fake = FakeComputeBackend.builder().backendType(BackendType.CUDA)
                .capabilities(tensorCoreCaps()).build();
        try {
            LlmKernels kernels = LlmKernels.open(fake);
            for (int[] shape : new int[][]{{8, 8, 128}, {16, 8, 96}}) {
                String at = shape[0] + "/" + shape[1] + " heads of " + shape[2];
                FlashPlan q8 = densePlan(fake, kernels, KvDType.Q8_0, shape[0], shape[1], shape[2]);
                assertEquals(at, "LLM_FlashAttnMma_Q8_0", q8.mmaKernel());
                assertNull(at, q8.rowKernel());
                assertEquals(at, AttnKernels.flashDecode(KvDType.Q8_0, KvDType.Q8_0), q8.decodeKernel());
                assertTrue(at + ": the fast path's splits", q8.decode().maxSplitLen() <= FlashAttnTiling.WG);
                assertFalse(at, q8.routesDecodeByDepth());
                assertTrue(at, q8.routesDecodeThroughPrefill());
                assertTrue(at, q8.routesDecodeThroughPrefill(FlashAttnTiling.WG + 1));
                assertTrue(at, q8.routesDecodeThroughPrefill(16384));
            }
            FlashPlan f16 = densePlan(fake, kernels, KvDType.F16, 16, 8, 96);
            assertEquals("LLM_FlashAttnMma_F16", f16.mmaKernel());
            assertNull(f16.rowKernel());
            assertTrue(f16.routesDecodeByDepth());
            assertTrue(f16.routesDecodeThroughPrefill(FlashAttnTiling.WG));
            assertFalse(f16.routesDecodeThroughPrefill(FlashAttnTiling.WG + 1));
        } finally {
            fake.close();
        }
    }

    /** The plan of one attention layer of {@code nHead}/{@code nHeadKv} heads over a dense cache of {@code type}. */
    private static FlashPlan densePlan(FakeComputeBackend fake, LlmKernels kernels, KvDType type, int nHead,
                                       int nHeadKv, int headDim) {
        int rows = 4096;
        KvLayout layout = KvLayout.of(rows, false, nHeadKv * headDim, nHeadKv * headDim, type, type,
                KvWindow.FULL_CAUSAL);
        GpuBuffer k = fake.allocateBuffer(layout.rowBytesK() * rows);
        GpuBuffer v = fake.allocateBuffer(layout.rowBytesV() * rows);
        GpuKvLayer kv = new GpuKvLayer() {
            @Override
            public int layer() {
                return 0;
            }

            @Override
            public KvLayout layout() {
                return layout;
            }

            @Override
            public int group() {
                return 0;
            }

            @Override
            public RotationPlan rotation() {
                return RotationPlan.NONE;
            }

            @Override
            public GpuBuffer k() {
                return k;
            }

            @Override
            public GpuBuffer v() {
                return v;
            }
        };
        AttentionSpec a = AttentionSpec.builder().nHead(nHead).nHeadKv(nHeadKv).headDimK(headDim).headDimV(headDim)
                .nRot(headDim).kqScale(0.125f).kvOwnerLayer(0).build();
        return FlashPlan.of(kernels, a, FlashKeys.cached(kv, fake.allocateBuffer(rows * 4L)), null, 512);
    }

    /** A CUDA device with tensor cores and the subgroup shuffle the decode fast path compiles over. */
    private static DeviceCapabilities tensorCoreCaps() {
        return new DeviceCapabilities.Builder().deviceName("FakeTensorCores").totalVramBytes(8L << 30)
                .maxAllocSizeBytes(2L << 30).computeUnits(32).maxClockFrequencyMhz(1500).maxWorkGroupSize(1024)
                .driverVersion("fake-1.0").supportsFloat16(true).supportsSubgroups(true).subgroupSize(32)
                .maxLocalMemoryBytes(48L * 1024).maxWorkGroupCount(new long[]{65535, 65535, 65535})
                .computeCapability(120).feature(DeviceFeature.COOPERATIVE_MATRIX)
                .feature(DeviceFeature.SUBGROUP_SHUFFLE)
                .identity(new DeviceIdentity("FakeTensorCores", FakeComputeBackend.NVIDIA_VENDOR_ID, null, null))
                .build();
    }

    @Test
    public void theRowKernelServesThePlainStepOfADenseF16Cache() {
        String row = AttnKernels.FLASH_DECODE_ROW_F16;
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE,
                1024, 1024, false));
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 64, 64, BiasMode.NONE,
                512, 512, false));
        assertEquals(row, FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 256, 256, BiasMode.NONE,
                2048, 2048, false));
    }

    @Test
    public void theGenericDecodeKernelServesEveryOtherStep() {
        // other cache types, or V read through the K rows
        assertNull(FlashPlan.rowKernelName(KvDType.Q8_0, KvDType.Q8_0, false, 128, 128, BiasMode.NONE,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F32, false, 128, 128, BiasMode.NONE,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, true, 128, 128, BiasMode.NONE,
                1024, 1024, false));
        // head dims: unequal, not a multiple of 64, above the kernel's 256
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 64, BiasMode.NONE,
                1024, 512, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 96, 96, BiasMode.NONE,
                768, 768, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 512, 512, BiasMode.NONE,
                4096, 4096, false));
        // position biases, odd strides, a paged cache
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.ALIBI,
                1024, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE,
                1025, 1024, false));
        assertNull(FlashPlan.rowKernelName(KvDType.F16, KvDType.F16, false, 128, 128, BiasMode.NONE,
                1024, 1024, true));
    }

    @Test
    public void theRowKernelsServeVulkanOnNvidiaDevicesOnly() {
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
