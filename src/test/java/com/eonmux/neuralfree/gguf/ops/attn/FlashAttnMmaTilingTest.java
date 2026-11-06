package com.eonmux.neuralfree.gguf.ops.attn;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Plans of {@link FlashAttnTiling#mma}, the tensor-core prefill of kernels/llm/flash_mma.cl: the work-group
 * carries {@value FlashAttnTiling#MMA_ROWS} query rows (bq * g_tile), key tiles of {@value
 * FlashAttnTiling#MMA_BK}, head dims that are multiples of 16 of at most {@value
 * FlashAttnTiling#MMA_MAX_HEAD_DIM} whose staged tiles fit the shared memory, and the split, grid and
 * partial rules the scalar tiled plan follows.
 */
public class FlashAttnMmaTilingTest {

    private static final int SMEM = 11264;
    private static final int CU = 48;

    @Test
    public void plansStandardGqaShapes() {
        FlashAttnTiling p = FlashAttnTiling.mma(8, 2, 64, 64, 4096, SMEM, CU);
        assertNotNull(p);
        assertEquals(FlashAttnTiling.Plan.MMA, p.plan());
        assertEquals(4, p.gTile());
        assertEquals(16, p.bq());
        assertEquals(FlashAttnTiling.MMA_BK, p.bk());
        assertEquals(FlashAttnTiling.MMA_ROWS, p.rows());
        assertEquals(FlashAttnTiling.mmaSmemFloats(64, 64), p.smemUsed());
        assertTrue(p.smemUsed() <= SMEM);
    }

    @Test
    public void capsGTileAtEightAndKeepsRowProduct() {
        FlashAttnTiling p = FlashAttnTiling.mma(64, 2, 128, 128, 4096, SMEM, CU);
        assertNotNull(p);
        assertEquals(8, p.gTile());
        assertEquals(8, p.bq());
        FlashAttnTiling odd = FlashAttnTiling.mma(12, 4, 64, 64, 4096, SMEM, CU);
        assertNotNull(odd);
        // G = 3: g_tile 3 does not divide the 64 rows, so the plan steps down to 2
        assertEquals(2, odd.gTile());
        assertEquals(32, odd.bq());
        assertEquals(FlashAttnTiling.MMA_ROWS, odd.rows());
    }

    @Test
    public void rejectsUnsupportedShapesWithNull() {
        assertNull(FlashAttnTiling.mma(8, 2, 72, 64, 4096, SMEM, CU));
        assertNull(FlashAttnTiling.mma(8, 2, 64, 84, 4096, SMEM, CU));
        assertNull(FlashAttnTiling.mma(8, 2, 256, 256, 4096, SMEM, CU));
        assertNull(FlashAttnTiling.mma(8, 2, 64, 64, 4096, 2048, CU));
    }

    @Test
    public void headDimSixteenFitsTheSmallestCache() {
        FlashAttnTiling p = FlashAttnTiling.mma(8, 8, 16, 16, 512, (int) FlashAttnTiling.mmaSmemFloats(16, 16), CU);
        assertNotNull(p);
        assertEquals(FlashAttnTiling.mmaSmemFloats(16, 16), p.smemUsed());
    }

    @Test
    public void smemBudgetHoldsForEverySupportedDim() {
        for (int d = 16; d <= FlashAttnTiling.MMA_MAX_HEAD_DIM; d += 16) {
            assertTrue("d=" + d, FlashAttnTiling.mmaSmemFloats(d, d) <= SMEM);
        }
    }

    @Test
    public void splitsAndGridFollowTheTiledRules() {
        FlashAttnTiling p = FlashAttnTiling.mma(8, 2, 64, 64, 4096, SMEM, CU);
        long capacity = FlashAttnTiling.partialFloats(64 * 8L, FlashAttnTiling.MAX_PREFILL_SPLITS, 64);
        FlashAttnTiling.Split small = p.prefillSplit(8, 64, capacity);
        assertEquals(1, small.count());
        assertTrue(small.direct());

        FlashAttnTiling.Split wide = p.prefillSplit(64, 4096, Long.MAX_VALUE);
        assertTrue(wide.count() >= 1);
        assertTrue(wide.length() % FlashAttnTiling.MMA_BK == 0);
        assertEquals((4096 + wide.length() - 1) / wide.length(), wide.count());

        long[] g = p.prefillGlobal(8, 2);
        assertEquals(128, g[0]);
        assertEquals(2 * p.groupTiles(), g[1]);
        assertEquals(2, g[2]);
    }

    @Test
    public void decodeRulesStillRejectTheMmaPlan() {
        FlashAttnTiling p = FlashAttnTiling.mma(8, 2, 64, 64, 4096, SMEM, CU);
        assertThrows(IllegalStateException.class, () -> p.decodeSplit(16));
        assertThrows(IllegalStateException.class, () -> p.decodePartialFloats());
        assertThrows(IllegalStateException.class, () -> p.decodePagedGlobal(2, 1));
        assertThrows(IllegalStateException.class, () -> p.decodePagedPartialFloats(2));
    }

    @Test
    public void sharedMemoryBudgetMatchesTheKernelArrays() {
        // kernels/llm/flash_mma.cl: q_s 64 x (d_k + 8), k_s 32 x (d_k + 8), v_s d_v x (32 + 8) halves, plus the
        // per-tile key positions and the skip flag, as floats (the formula rounds the counters up).
        long worst = FlashAttnTiling.mmaSmemFloats(FlashAttnTiling.MMA_MAX_HEAD_DIM,
                FlashAttnTiling.MMA_MAX_HEAD_DIM);
        assertEquals((64L * 136 + 32L * 136 + 128L * 40 + 2 * FlashAttnTiling.MMA_BK + FlashAttnTiling.WG) / 2,
                worst);
        assertTrue(worst <= SMEM);
    }
}
