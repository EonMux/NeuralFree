package com.eonmux.neuralfree.gguf.ops.vision;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * The launch geometry and shape checks of {@link VisionKernels}: the work-item counts of the convolution and
 * pool, the one-workgroup-per-query dispatch of the relative-position attention and the named refusals of the
 * shapes the kernels cannot take.
 */
public class VisionKernelsTest {

    @Test
    public void elementGlobalRoundsUpToLocal() {
        assertEquals(VisionKernels.LOCAL, VisionKernels.elementGlobal(1));
        assertEquals(VisionKernels.LOCAL, VisionKernels.elementGlobal(VisionKernels.LOCAL));
        assertEquals(2L * VisionKernels.LOCAL, VisionKernels.elementGlobal(VisionKernels.LOCAL + 1));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.elementGlobal(0));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.elementGlobal(-4));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.elementGlobal(1L << 32));
    }

    @Test
    public void attnDispatchRunsOneWorkgroupPerQuery() {
        // 3x3 windows of one head: 9 workgroups of 32; a 64x64 global grid of 2 heads: 8192 of 32
        assertEquals(9 * 32, VisionKernels.attnWorkItems(1, 1, 3, 32));
        assertEquals(8192 * 32, VisionKernels.attnWorkItems(1, 2, 64, 32));
        assertThrows(IllegalArgumentException.class,
                () -> VisionKernels.attnWorkItems(1, 1, 1 << 16, 32));
    }

    @Test
    public void attnLocalIsAPowerOfTwoOfAtLeast32() {
        assertEquals(64, VisionKernels.attnLocal(64));
        assertEquals(32, VisionKernels.attnLocal(33));
        assertEquals(64, VisionKernels.attnLocal(100));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.attnLocal(16));
    }

    @Test
    public void depthwiseShapesAreChecked() {
        VisionKernels.checkDepthwise(8, 8, 8, 8, 4, 3, 1, 1, 1, 1);
        VisionKernels.checkDepthwise(16, 16, 8, 8, 4, 3, 2, 2, 0, 1);
        assertThrows(IllegalArgumentException.class,
                () -> VisionKernels.checkDepthwise(0, 8, 8, 8, 4, 3, 1, 1, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> VisionKernels.checkDepthwise(8, 8, 0, 8, 4, 3, 1, 1, 1, 1));
    }

    @Test
    public void poolShapesMustTileTheMap() {
        VisionKernels.checkPool(8, 8, 4, 4, 6, 2, 2);
        VisionKernels.checkPool(1, 64, 1, 1, 6, 64, 1);
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.checkPool(8, 8, 3, 4, 6, 2, 2));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.checkPool(8, 8, 4, 4, 6, 0, 2));
    }

    @Test
    public void relPosAttnGroupsMustFitTheStagedProbabilities() {
        VisionKernels.checkRelPosAttn(64, 8, 2, 4);
        VisionKernels.checkRelPosAttn(1, 64, 12, 64);
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.checkRelPosAttn(1, 65, 2, 4));
        assertThrows(IllegalArgumentException.class, () -> VisionKernels.checkRelPosAttn(0, 8, 2, 4));
    }
}
