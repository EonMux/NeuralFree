package com.eonmux.neuralfree.tensor;

import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The shape rule that runs {@code TR_Scatter_F32} with one work-item per output element
 * ({@link GpuTensorOps#scatterPerTarget}): fewer than {@link GpuTensorOps#SCATTER_TARGETS_BELOW_FIBERS} fibers whose
 * {@code len * limit} is at most {@link GpuTensorOps#SCATTER_TARGETS_MAX_SCAN}. No GPU is needed.
 */
public class ScatterTargetRuleTest {

    /** The rotary-embedding gradients of a 64-token Qwen3 window: 16 query or 8 KV heads of 128 dims per token. */
    @Test
    public void ropeGradientsRunPerTarget() {
        assertTrue(GpuTensorOps.scatterPerTarget(64 * 16, 128, 128));
        assertTrue(GpuTensorOps.scatterPerTarget(64 * 8, 128, 128));
        assertTrue(GpuTensorOps.scatterPerTarget(1, 1, 1));
    }

    @Test
    public void manyFibersKeepTheFiberWalk() {
        assertFalse(GpuTensorOps.scatterPerTarget(GpuTensorOps.SCATTER_TARGETS_BELOW_FIBERS, 2, 2));
        assertTrue(GpuTensorOps.scatterPerTarget(GpuTensorOps.SCATTER_TARGETS_BELOW_FIBERS - 1, 2, 2));
    }

    @Test
    public void longScansKeepTheFiberWalk() {
        assertTrue(GpuTensorOps.scatterPerTarget(4, 256, 256));
        assertFalse(GpuTensorOps.scatterPerTarget(4, 257, 256));
        assertFalse(GpuTensorOps.scatterPerTarget(4, 64, 151936));
    }

    @Test
    public void emptyOutputDimKeepsTheFiberWalk() {
        assertFalse(GpuTensorOps.scatterPerTarget(4, 3, 0));
    }
}
