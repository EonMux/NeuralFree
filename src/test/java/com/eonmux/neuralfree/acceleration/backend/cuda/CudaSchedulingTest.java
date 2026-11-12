package com.eonmux.neuralfree.acceleration.backend.cuda;

import jcuda.driver.CUctx_flags;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class CudaSchedulingTest {

    @Test
    public void namesParseToTheirDriverFlags() {
        assertEquals(CUctx_flags.CU_CTX_SCHED_AUTO, CudaScheduling.parse("auto").flag());
        assertEquals(CUctx_flags.CU_CTX_SCHED_SPIN, CudaScheduling.parse(" Spin ").flag());
        assertEquals(CUctx_flags.CU_CTX_SCHED_YIELD, CudaScheduling.parse("YIELD").flag());
        assertEquals(CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC, CudaScheduling.parse("block").flag());
        assertNull(CudaScheduling.parse(null));
        assertNull(CudaScheduling.parse("  "));
    }

    @Test
    public void anUnknownNameIsRefusedWithThePropertyAndTheChoices() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CudaScheduling.parse("busy"));
        assertTrue(e.getMessage(), e.getMessage().contains(CudaScheduling.PROPERTY)
                && e.getMessage().contains("auto, spin, yield or block"));
    }

    @Test
    public void contextFlagsMapBackToTheirScheduling() {
        assertEquals(CudaScheduling.BLOCK, CudaScheduling.ofFlags(CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC
                | CUctx_flags.CU_CTX_LMEM_RESIZE_TO_MAX));
        assertEquals(CudaScheduling.SPIN, CudaScheduling.ofFlags(CUctx_flags.CU_CTX_SCHED_SPIN));
        assertEquals(CudaScheduling.AUTO, CudaScheduling.ofFlags(0));
    }
}
