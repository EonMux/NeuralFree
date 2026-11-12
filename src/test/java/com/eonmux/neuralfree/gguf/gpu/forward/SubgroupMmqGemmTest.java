package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The launch plan of the subgroup MMQ forms ({@link SubgroupMmqGemm}): the work-group of each backend and weight type
 * (the OpenCL tile of one work-item per row, the Vulkan Q8_0, Q4_0, Q4_K and Q5_K tiles split into two token halves,
 * the Vulkan Q6_K tile unsplit), the row grid that follows it, the token grid and the row shapes the kernels accept.
 */
public class SubgroupMmqGemmTest {

    @Test
    public void openClRunsOneWorkItemPerRow() {
        for (GGMLType type : SubgroupMmqGemm.TYPES) {
            assertEquals(type.name(), SubgroupMmqGemm.TILE_ROWS, SubgroupMmqGemm.workGroup(BackendType.OPENCL, type));
        }
    }

    @Test
    public void vulkanSplitsEveryTypesTokensButQ6K() {
        int split = SubgroupMmqGemm.TILE_ROWS * SubgroupMmqGemm.VULKAN_TOKEN_SPLIT;
        for (GGMLType type : new GGMLType[]{GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1,
                GGMLType.Q4_K, GGMLType.Q5_K}) {
            assertEquals(type.name(), split, SubgroupMmqGemm.workGroup(BackendType.VULKAN, type));
        }
        assertEquals(SubgroupMmqGemm.TILE_ROWS, SubgroupMmqGemm.workGroup(BackendType.VULKAN, GGMLType.Q6_K));
    }

    @Test
    public void theRowGridHoldsOneWorkGroupPerTileOfRows() {
        assertEquals(128, SubgroupMmqGemm.globalRows(1, 128));
        assertEquals(256, SubgroupMmqGemm.globalRows(128, 256));
        assertEquals(76 * 256, SubgroupMmqGemm.globalRows(9728, 256));
        assertEquals(77 * 128, SubgroupMmqGemm.globalRows(9729, 128));
        assertThrows(IllegalArgumentException.class, () -> SubgroupMmqGemm.globalRows(0, 128));
    }

    @Test
    public void theTokenGridHoldsOneWorkGroupPerTileOfTokens() {
        assertEquals(1, SubgroupMmqGemm.tokenGroups(1));
        assertEquals(16, SubgroupMmqGemm.tokenGroups(512));
        assertEquals(17, SubgroupMmqGemm.tokenGroups(513));
        assertThrows(IllegalArgumentException.class, () -> SubgroupMmqGemm.tokenGroups(0));
    }

    @Test
    public void rowsMustHoldWholeStagesOfWholeBlocks() {
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q8_0, 64));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q8_0, 96));
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q4_K, 2560));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q4_K, 2560 + 64));
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q4_0, 2560 + 64));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q4_0, 2560 + 32));
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q5_K, 2560));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q5_K, 2560 + 64));
        assertEquals("Q4_0_MatMulSg_MMQ", SubgroupMmqGemm.kernel(GGMLType.Q4_0));
        assertEquals("Q5_K_MatMulSg_MMQ", SubgroupMmqGemm.kernel(GGMLType.Q5_K));
        assertEquals("Q6_K_MatMulSg_MMQ", SubgroupMmqGemm.kernel(GGMLType.Q6_K));
        assertEquals("Q5_0_MatMulSg_MMQ", SubgroupMmqGemm.kernel(GGMLType.Q5_0));
        assertTrue(SubgroupMmqGemm.shapeServes(GGMLType.Q5_0, 2560 + 64));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q5_0, 2560 + 32));
        assertNull(SubgroupMmqGemm.kernel(GGMLType.Q3_K));
        assertFalse(SubgroupMmqGemm.shapeServes(GGMLType.Q3_K, 2560));
    }
}
