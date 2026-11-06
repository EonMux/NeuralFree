package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * Selection of the tensor-core prefill kernel ({@code LLM_FlashAttnMma*}) in {@link FlashPlan}:
 * the kernel name follows the cooperative-matrix feature, the F16 cache types and the head dims; a backend
 * without the feature (or without the kernel) keeps the scalar tiled kernel. The mixer dispatches the base
 * kernel over a dense cache (its identity paging); the paged kernel serves launches over a paginated cache.
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
}
