package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The slice-planes buffer of {@link VulkanGemmPrefillMeasureTest} holds every split lane it measures: the
 * heuristic's split of a small grid takes more slices than the fixed lanes, and its partial launch writes them all.
 */
public class VulkanGemmPrefillMeasureSizingTest {

    private static final int TOKENS = 512;
    private static final int COMPUTE_UNITS = 60;

    @Test
    public void theBufferHoldsTheHeuristicSplitOfASmallGrid() {
        int m = 128;
        int k = 4096;
        SplitKGemmPlan split = SplitKGemmPlan.of(Cm2GemmPlan.contiguous(GGMLType.Q8_0, m, k, TOKENS), COMPUTE_UNITS,
                256);
        assertEquals("two tiles on 60 units split into the most slices", SplitKGemmPlan.MAX_SPLITS, split.splits());
        assertTrue(VulkanGemmPrefillMeasureTest.partialBufferFloats(split, TOKENS, m) >= split.partialElements());
    }

    @Test
    public void theBufferHoldsTheFixedLanesWhereTheHeuristicKeepsTheLaunchWhole() {
        int m = 9728;
        int k = 2560;
        SplitKGemmPlan split = SplitKGemmPlan.of(Cm2GemmPlan.contiguous(GGMLType.Q8_0, m, k, TOKENS), COMPUTE_UNITS,
                256);
        long fixedLanes = 4L * TOKENS * m;
        assertTrue(VulkanGemmPrefillMeasureTest.partialBufferFloats(split, TOKENS, m) >= fixedLanes);
        assertTrue(VulkanGemmPrefillMeasureTest.partialBufferFloats(null, TOKENS, m) >= fixedLanes);
    }
}
