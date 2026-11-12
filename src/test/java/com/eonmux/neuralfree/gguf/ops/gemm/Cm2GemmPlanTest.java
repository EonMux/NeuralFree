package com.eonmux.neuralfree.gguf.ops.gemm;

import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.vulkan.VkPhysicalDeviceInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Plan geometry of the CM2 launches: the dense F16 {@code F16_MatMul_CM2} (rows of fp16 weights with no block decode,
 * a reduction of 16-byte aligned rows, the split-K twin over the same type), and the 128-token tile
 * ({@code _N128}) with the rule that picks it, and the device probe that admits both tiles.
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

    @Test
    public void theMediumTilePlansTwiceTheTokenGroupsOverTheSameStaging() {
        Cm2GemmPlan large = Cm2GemmPlan.of(GGMLType.Q8_0, true, 1024, 1024, 300, 1024, 0, 1024, 0);
        Cm2GemmPlan medium = Cm2GemmPlan.of(GGMLType.Q8_0, true, Cm2GemmPlan.MEDIUM_TILE_TOKENS, 1024, 1024, 300,
                1024, 0, 1024, 0);
        assertEquals("Q8_0_MatMul_CM2V_N128", medium.kernelName());
        assertEquals("Q4_K_MatMul_CM2_N128", Cm2GemmPlan.of(GGMLType.Q4_K, false, Cm2GemmPlan.MEDIUM_TILE_TOKENS,
                128, 256, 16, 256, 0, 128, 0).kernelName());
        assertEquals(Cm2GemmPlan.MEDIUM_TILE_TOKENS, medium.tileTokens());
        assertEquals(2, large.tokenTiles());
        assertEquals(3, medium.tokenTiles());
        assertEquals(3L, medium.globalWorkSize()[1]);
        assertEquals("one staging serves both tiles", large.stagedHalves(), medium.stagedHalves());
        IllegalArgumentException tile = assertThrows(IllegalArgumentException.class,
                () -> Cm2GemmPlan.of(GGMLType.Q8_0, false, 64, 128, 256, 16, 256, 0, 128, 0));
        assertTrue(tile.getMessage(), tile.getMessage().contains("64"));
        IllegalArgumentException split = assertThrows(IllegalArgumentException.class,
                () -> SplitKGemmPlan.of(medium, 60, 256));
        assertTrue(split.getMessage(), split.getMessage().contains("no split-K twin"));
    }

    /** On a 60-unit device: the 128-token tile for an unsplit grid of at most 30 tiles of 256 tokens whose 128-token
     *  grid fits one wave; the 256-token tile for fuller grids, one-tile token spans, splitting reductions and a
     *  device without a compute-unit count. */
    @Test
    public void theMediumTileServesSmallUnsplitGrids() {
        int medium = Cm2GemmPlan.MEDIUM_TILE_TOKENS;
        int large = Cm2GemmPlan.TILE_TOKENS;
        assertEquals("16 tiles, K below the split threshold", medium, Cm2GemmPlan.tileTokensFor(60, 1024, 512, 1024));
        assertEquals("8 tiles of 256 tokens", medium, Cm2GemmPlan.tileTokensFor(60, 1024, 256, 1024));
        assertEquals("32 tiles fill more than half", large, Cm2GemmPlan.tileTokensFor(60, 2048, 512, 1024));
        assertEquals("the reduction splits", large, Cm2GemmPlan.tileTokensFor(60, 1024, 512, 2048));
        assertEquals("one token tile either way", large, Cm2GemmPlan.tileTokensFor(60, 1024, 128, 1024));
        assertEquals("unknown compute units", large, Cm2GemmPlan.tileTokensFor(
                DeviceCapabilities.UNKNOWN_COMPUTE_UNITS, 1024, 512, 1024));
        assertThrows(IllegalArgumentException.class, () -> Cm2GemmPlan.tileTokensFor(60, 0, 512, 1024));
        assertThrows(IllegalArgumentException.class, () -> Cm2GemmPlan.tileTokensFor(-2, 1024, 512, 1024));
    }

    /** Wherever the 128-token tile is picked its grid runs in one wave: at most as many workgroups as compute units. */
    @Test
    public void theMediumGridAlwaysFitsOneWave() {
        for (int cu = 1; cu <= 160; cu++) {
            for (int m = 128; m <= 8192; m += 128) {
                for (int n = 1; n <= 2048; n += 37) {
                    if (Cm2GemmPlan.tileTokensFor(cu, m, n, 1024) == Cm2GemmPlan.MEDIUM_TILE_TOKENS) {
                        long tiles = (long) (m / Cm2GemmPlan.TILE_ROWS)
                                * ((n + Cm2GemmPlan.MEDIUM_TILE_TOKENS - 1) / Cm2GemmPlan.MEDIUM_TILE_TOKENS);
                        assertTrue(cu + " units, M=" + m + ", N=" + n + ": " + tiles + " tiles", tiles <= cu);
                    }
                }
            }
        }
    }

    /**
     * The cooperative-matrix2 device probe asks for a token granularity that divides every token matrix of both CM2
     * tiles, the full, half and quarter-width ones (the last token tile of a launch runs at the width its tokens
     * need), so a device that reports the feature can run each of them.
     */
    @Test
    public void theDeviceProbeCoversEveryTokenWidthOfBothTiles() {
        for (int tile : new int[]{Cm2GemmPlan.TILE_TOKENS, Cm2GemmPlan.MEDIUM_TILE_TOKENS}) {
            for (int width : new int[]{tile, tile / 2, tile / 4}) {
                assertEquals(tile + "-token tile, " + width + "-token matrix", 0,
                        width % VkPhysicalDeviceInfo.COOPMAT2_GEMM_MIN_TOKENS);
            }
        }
    }
}
