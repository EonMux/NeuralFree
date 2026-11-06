package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * Names of the tensor-core mma kernels ({@code LLM_FlashAttnMma_F16}, {@code LLM_FlashAttnMmaPaged_F16}, the
 * CUDA-only files kernels/llm/flash_mma.cl): only the F16 cache has an mma variant, and the kernels stay out of
 * {@link AttnKernels#all()} so the prewarm of backends without them cannot fail.
 */
public class AttnKernelsMmaTest {

    @Test
    public void f16CarriesTheMmaNames() {
        assertEquals("LLM_FlashAttnMma_F16", AttnKernels.flashAttnMma(KvDType.F16));
        assertEquals("LLM_FlashAttnMmaPaged_F16", AttnKernels.flashAttnMmaPaged(KvDType.F16));
    }

    @Test
    public void otherCacheTypesHaveNoMmaVariant() {
        assertNull(AttnKernels.flashAttnMma(KvDType.F32));
        assertNull(AttnKernels.flashAttnMma(KvDType.BF16));
        assertNull(AttnKernels.flashAttnMma(KvDType.Q8_0));
        assertNull(AttnKernels.flashAttnMma(KvDType.Q4_0));
        assertNull(AttnKernels.flashAttnMmaPaged(KvDType.F32));
        assertNull(AttnKernels.flashAttnMmaPaged(KvDType.Q5_1));
        assertNull(AttnKernels.flashAttnMma(null));
    }

    @Test
    public void mmaKernelsAreNotPrewarmed() {
        assertFalse(AttnKernels.all().contains("LLM_FlashAttnMma_F16"));
        assertFalse(AttnKernels.all().contains("LLM_FlashAttnMmaPaged_F16"));
        assertTrue(AttnKernels.all().contains(AttnKernels.FLASH_ATTN_MIXED));
    }
}
