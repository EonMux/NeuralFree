package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
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
 * paginated cache.
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
}
