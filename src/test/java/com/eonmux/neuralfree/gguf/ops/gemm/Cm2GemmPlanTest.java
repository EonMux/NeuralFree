package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Plan geometry of the dense F16 {@code F16_MatMul_CM2} launch: rows of fp16 weights with no block decode, a
 * reduction of 16-byte aligned rows and the split-K twin over the same type.
 */
public class Cm2GemmPlanTest {

    @Test
    public void f16PlansTheDenseKernelOverTwoByteElements() {
        Cm2GemmPlan plan = Cm2GemmPlan.contiguous(GGMLType.F16, 300, 1024, 40);
        assertEquals("F16_MatMul_CM2", plan.kernelName());
        assertEquals(300L * 1024 * Short.BYTES, plan.weightBytes());
        assertEquals(3, plan.rowTiles());
        assertEquals(1, plan.tokenTiles());
        assertEquals((long) Cm2GemmPlan.TILE_TOKENS * 1024, plan.stagedHalves());
        assertTrue(GemmKernels.CM2_TYPES.contains(GGMLType.F16));
        assertFalse("BF16 keeps the exact tiled GEMM on Vulkan", GemmKernels.hasCm2(GGMLType.BF16));
    }

    @Test
    public void f16NeedsSixteenByteAlignedRows() {
        assertTrue(Cm2GemmPlan.denseShapeServes(8));
        assertTrue(Cm2GemmPlan.denseShapeServes(520));
        assertFalse(Cm2GemmPlan.denseShapeServes(252));
        assertFalse(Cm2GemmPlan.denseShapeServes(0));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Cm2GemmPlan.contiguous(GGMLType.F16, 64, 252, 16));
        assertTrue(e.getMessage(), e.getMessage().contains("multiple of " + Cm2GemmPlan.DENSE_K_MULTIPLE));
    }

    @Test
    public void f16SplitKTwinPlansWholeStageSlices() {
        SplitKGemmPlan split = SplitKGemmPlan.of(SplitKGemmPlan.Family.CM2, GGMLType.F16, 256, 1536, 40, 1536, 0, 3,
                SplitKGemmPlan.sliceLength(1536, 3, Cm2GemmPlan.TILE_DEPTH), Cm2GemmPlan.TILE_TOKENS, 256);
        assertEquals("F16_MatMul_CM2_KSplit", split.kernelName());
        assertEquals(0, split.kSlice() % Cm2GemmPlan.TILE_DEPTH);
        assertEquals(3, split.splits());
    }
}
