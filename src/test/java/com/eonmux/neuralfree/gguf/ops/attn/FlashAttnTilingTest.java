package com.eonmux.neuralfree.gguf.ops.attn;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Flash-attention tile plans: the traced budgets for head dims 128, 256, 512 and the MLA latent 576/512 at the
 * shared-memory sizes of the smallest device (3072 floats) and of CUDA / 48 KiB Vulkan devices (11264 floats), the
 * per-row decode fallback, the per-step split arithmetic (llama.cpp sizes decode parallelism from the live key count,
 * ggml-cuda/fattn-common.cuh:1141-1214), and the kernel constraints of every plan.
 */
public class FlashAttnTilingTest {

    private static final int SMEM_MIN = 3072;
    private static final int SMEM_48K = 12288 - 1024;
    private static final int CU = 40;
    private static final int N_KV = 4096;

    @Test
    public void decodePlansForCommonHeadDims() {
        // max splits = max(ceil(n_kv / maxSplitLen), min(wanted, ceil(n_kv / 128))), wanted = ceil(2 CU / groups).
        assertDecode(FlashAttnTiling.decode(32, 32, 128, 128, N_KV, SMEM_48K, CU), 1, 11136, 3);
        assertDecode(FlashAttnTiling.decode(32, 8, 128, 128, N_KV, SMEM_48K, CU), 4, 2688, 10);
        assertDecode(FlashAttnTiling.decode(8, 4, 256, 256, N_KV, SMEM_48K, CU), 2, 5376, 20);
        assertDecode(FlashAttnTiling.decode(16, 2, 512, 512, N_KV, SMEM_48K, CU), 8, 896, 32);
        assertDecode(FlashAttnTiling.decode(32, 32, 128, 128, N_KV, SMEM_MIN, CU), 1, 2944, 3);
        assertDecode(FlashAttnTiling.decode(32, 8, 128, 128, N_KV, SMEM_MIN, CU), 4, 640, 10);
        assertDecode(FlashAttnTiling.decode(8, 4, 256, 256, N_KV, SMEM_MIN, CU), 2, 1280, 20);
        // gemma4-style d=512 with 8 heads per KV head: g*512 <= 3072/2 gives g <= 3, balanced as 3+3+2.
        FlashAttnTiling d512 = FlashAttnTiling.decode(16, 2, 512, 512, N_KV, SMEM_MIN, CU);
        assertDecode(d512, 3, 512, 14);
        assertEquals(3, d512.groupTiles());
    }

    @Test
    public void decodePlansForMlaLatentCache() {
        // d_k = kv_lora + rope = 576, d_v = kv_lora = 512, one KV head for 128 query heads.
        FlashAttnTiling big = FlashAttnTiling.decode(128, 1, 576, 512, N_KV, SMEM_48K, CU);
        assertDecode(big, 8, 832, 5);
        assertEquals(SMEM_48K, big.smemUsed());
        assertEquals(16, big.groupTiles());
        FlashAttnTiling small = FlashAttnTiling.decode(128, 1, 576, 512, N_KV, SMEM_MIN, CU);
        assertDecode(small, 2, 960, 5);
        assertEquals(SMEM_MIN, small.smemUsed());
    }

    @Test
    public void prefillPlansForCommonHeadDims() {
        assertTiled(FlashAttnTiling.prefill(32, 32, 128, 128, N_KV, SMEM_48K, CU), 1, 32, 32, 5216);
        assertTiled(FlashAttnTiling.prefill(32, 8, 128, 128, N_KV, SMEM_48K, CU), 4, 8, 32, 5216);
        assertTiled(FlashAttnTiling.prefill(8, 4, 256, 256, N_KV, SMEM_48K, CU), 2, 8, 32, 3632);
        assertTiled(FlashAttnTiling.prefill(16, 2, 512, 512, N_KV, SMEM_48K, CU), 8, 1, 32, 2840);
        assertTiled(FlashAttnTiling.prefill(32, 32, 128, 128, N_KV, SMEM_MIN, CU), 1, 16, 16, 2352);
        assertTiled(FlashAttnTiling.prefill(32, 8, 128, 128, N_KV, SMEM_MIN, CU), 4, 4, 16, 2352);
        assertTiled(FlashAttnTiling.prefill(8, 4, 256, 256, N_KV, SMEM_MIN, CU), 2, 8, 16, 2352);
        assertTiled(FlashAttnTiling.prefill(16, 2, 512, 512, N_KV, SMEM_MIN, CU), 8, 1, 32, 2840);
    }

    @Test
    public void prefillPlansForMlaLatentCache() {
        assertTiled(FlashAttnTiling.prefill(128, 1, 576, 512, N_KV, SMEM_48K, CU), 8, 1, 32, 2840);
        assertTiled(FlashAttnTiling.prefill(128, 1, 576, 512, N_KV, SMEM_MIN, CU), 8, 1, 32, 2840);
    }

    @Test
    public void infeasibleTilingFallsBackToPerRowDecode() {
        // 1100 floats cannot hold one row with a 16-key tile (64 + 1024 + 16 + 3 = 1107), but decode fits.
        FlashAttnTiling t = FlashAttnTiling.prefill(32, 32, 128, 128, N_KV, 1100, CU);
        assertEquals(FlashAttnTiling.Plan.PER_ROW_DECODE, t.plan());
        assertEquals(1, t.gTile());
        assertEquals(960, t.maxSplitLen());
        assertEquals(5, t.maxSplits());
        // wanted = ceil(80 / 32) = 3 splits: split_len = roundUp32(ceil(1000 / 3)) = 352.
        FlashAttnTiling.Split s = t.decodeSplit(1000);
        assertEquals(3, s.count());
        assertEquals(352, s.length());
        assertThrows(IllegalStateException.class, () -> t.prefillSplit(8, 100, 0));
    }

    @Test
    public void everyValidHeadDimUpTo1024HasPlansAtTheMinimumBudget() {
        int[][] heads = {{1, 1}, {8, 8}, {32, 8}, {12, 1}, {7, 7}, {128, 1}, {40, 8}};
        for (int[] hh : heads) {
            for (int dK = 4; dK <= FlashAttnTiling.MAX_HEAD_DIM; dK += 4) {
                for (int dV : new int[]{dK, Math.max(4, dK - 64)}) {
                    for (int smem : new int[]{SMEM_MIN, SMEM_48K}) {
                        FlashAttnTiling dec = FlashAttnTiling.decode(hh[0], hh[1], dK, dV, 32768, smem, CU);
                        checkDecodeConstraints(dec);
                        FlashAttnTiling pre = FlashAttnTiling.prefill(hh[0], hh[1], dK, dV, 32768, smem, CU);
                        if (pre.plan() == FlashAttnTiling.Plan.TILED) {
                            checkTiledConstraints(pre);
                        } else {
                            checkDecodeConstraints(pre);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void decodeSplitsFollowTheLiveKeyCount() {
        // 8 workgroups per split on 40 compute units: wanted = 10 splits, split_len in [128, 2688].
        FlashAttnTiling t = FlashAttnTiling.decode(32, 8, 128, 128, N_KV, SMEM_48K, CU);
        assertEquals(128, t.minSplitLen());
        assertEquals(N_KV, t.keyCapacity());
        assertSplit(t.decodeSplit(0), 1, 128);
        FlashAttnTiling.Split one = t.decodeSplit(128);
        assertSplit(one, 1, 128);
        assertTrue(one.direct());
        FlashAttnTiling.Split two = t.decodeSplit(129);
        assertSplit(two, 2, 128);
        assertFalse(two.direct());
        assertSplit(t.decodeSplit(1000), 8, 128);
        assertSplit(t.decodeSplit(2000), 9, 224);
        assertSplit(t.decodeSplit(N_KV), 10, 416);
        assertThrows(IllegalArgumentException.class, () -> t.decodeSplit(N_KV + 1));
        assertThrows(IllegalArgumentException.class, () -> t.decodeSplit(-1));
        assertArrayEquals(new long[]{10 * 128, 8 * 1}, t.decodeGlobal(10));
        assertArrayEquals(new long[]{128, 1}, FlashAttnTiling.decodeLocal());
        assertEquals(32L * 10 * 130, t.decodePartialFloats());
    }

    @Test
    public void largeContextDecodeFillsTheDeviceAtShortPositions() {
        // Llama-3-8B shape on 80 compute units with a 131072-row cache: short live ranges still split.
        FlashAttnTiling t = FlashAttnTiling.decode(32, 8, 128, 128, 131072, SMEM_48K, 80);
        assertEquals(2688, t.maxSplitLen());
        assertEquals(49, t.maxSplits());
        assertSplit(t.decodeSplit(1000), 8, 128);
        assertSplit(t.decodeSplit(5000), 20, 256);
        assertSplit(t.decodeSplit(131072), 49, 2688);
        assertEquals(32L * 49 * 130, t.decodePartialFloats());
    }

    @Test
    public void everyDecodeSplitIsWithinThePlan() {
        int[][] shapes = {{32, 8, 128, 128}, {32, 32, 128, 128}, {128, 1, 576, 512}, {16, 2, 512, 512}};
        for (int[] sh : shapes) {
            for (int smem : new int[]{SMEM_MIN, SMEM_48K}) {
                for (int cu : new int[]{1, CU, 128}) {
                    FlashAttnTiling t = FlashAttnTiling.decode(sh[0], sh[1], sh[2], sh[3], N_KV, smem, cu);
                    for (int n = 0; n <= N_KV; n++) {
                        FlashAttnTiling.Split s = t.decodeSplit(n);
                        String where = t + " at " + n + " keys: " + s;
                        assertTrue(where, s.count() >= 1 && s.count() <= t.maxSplits());
                        assertTrue(where, s.length() >= t.minSplitLen() && s.length() <= t.maxSplitLen());
                        assertEquals(where, 0, s.length() % 32);
                        assertTrue(where, (long) s.count() * s.length() >= n);
                        assertTrue(where, (long) (s.count() - 1) * s.length() < Math.max(1, n));
                        assertTrue(where, (long) t.gTile() * (t.dK() + s.length()) <= t.smemFloats());
                    }
                }
            }
        }
    }

    @Test
    public void prefillSplitsOnlyWhenTheGridIsSmall() {
        FlashAttnTiling t = FlashAttnTiling.prefill(32, 8, 128, 128, N_KV, SMEM_48K, CU);
        // 512 tokens: 64 token tiles x 8 KV heads = 512 workgroups >= 2 x 40, so one split.
        assertEquals(1, t.prefillSplit(512, 4096, Long.MAX_VALUE).count());
        // 8 tokens: 1 x 8 = 8 workgroups; wanted ceil(80 / 8) = 10, keys allow 4096 / (4 * 32) = 32 splits.
        FlashAttnTiling.Split s = t.prefillSplit(8, 4096, Long.MAX_VALUE);
        assertEquals(10, s.count());
        assertEquals(0, s.length() % t.bk());
        assertTrue((long) s.count() * s.length() >= 4096);
        // capacity for 3 splits of 8 tokens x 32 heads x 130 floats
        assertEquals(3, t.prefillSplit(8, 4096, 3L * 8 * 32 * 130).count());
        assertEquals(1, t.prefillSplit(8, 4096, 0).count());
        assertArrayEquals(new long[]{128, 8, 3}, t.prefillGlobal(8, 3));
        assertArrayEquals(new long[]{128, 1, 1}, FlashAttnTiling.prefillLocal());
        assertArrayEquals(new long[]{5 * 128}, FlashAttnTiling.combineGlobal(5));
        assertThrows(IllegalStateException.class, () -> t.decodeSplit(10));
    }

    @Test
    public void invalidShapesAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> FlashAttnTiling.decode(32, 6, 128, 128, N_KV, SMEM_MIN, CU));
        assertThrows(IllegalArgumentException.class, () -> FlashAttnTiling.decode(32, 8, 130, 128, N_KV, SMEM_MIN, CU));
        assertThrows(IllegalArgumentException.class, () -> FlashAttnTiling.decode(32, 8, 128, 0, N_KV, SMEM_MIN, CU));
        assertThrows(IllegalArgumentException.class, () -> FlashAttnTiling.prefill(32, 8, 128, 128, 0, SMEM_MIN, CU));
        assertThrows(IllegalArgumentException.class, () -> FlashAttnTiling.decode(32, 8, 128, 128, N_KV, 100, CU));
    }

    private static void assertSplit(FlashAttnTiling.Split s, int count, int length) {
        assertEquals("split count of " + s, count, s.count());
        assertEquals("split length of " + s, length, s.length());
    }

    private static void assertDecode(FlashAttnTiling t, int gTile, int maxSplitLen, int maxSplits) {
        assertEquals(FlashAttnTiling.Plan.DECODE, t.plan());
        assertEquals("g_tile of " + t, gTile, t.gTile());
        assertEquals("max split_len of " + t, maxSplitLen, t.maxSplitLen());
        assertEquals("max splits of " + t, maxSplits, t.maxSplits());
        checkDecodeConstraints(t);
    }

    private static void assertTiled(FlashAttnTiling t, int gTile, int bq, int bk, long smem) {
        assertEquals(FlashAttnTiling.Plan.TILED, t.plan());
        assertEquals("g_tile of " + t, gTile, t.gTile());
        assertEquals("bq of " + t, bq, t.bq());
        assertEquals("bk of " + t, bk, t.bk());
        assertEquals("shared floats of " + t, smem, t.smemUsed());
        checkTiledConstraints(t);
    }

    /** The checks of nf_fd_plan_ok in flash_decode.cl. */
    private static void checkDecodeConstraints(FlashAttnTiling t) {
        assertTrue(t.toString(), t.gTile() >= 1 && t.gTile() <= FlashAttnTiling.MAX_G_TILE);
        assertTrue(t.toString(), t.maxSplitLen() >= 32 && t.maxSplitLen() % 32 == 0);
        assertTrue(t.toString(), t.minSplitLen() >= 32 && t.minSplitLen() % 32 == 0
                && t.minSplitLen() <= t.maxSplitLen());
        assertTrue(t.toString(), (long) t.gTile() * (t.dK() + t.maxSplitLen()) <= t.smemFloats());
        assertTrue(t.toString(), (long) t.gTile() * t.dV() <= FlashAttnTiling.ACC_PER_GROUP);
        assertTrue(t.toString(), (long) t.maxSplits() * t.maxSplitLen() >= t.keyCapacity());
    }

    /** The checks of nf_fa_plan_ok in flash_prefill.cl. */
    private static void checkTiledConstraints(FlashAttnTiling t) {
        int rows = t.bq() * t.gTile();
        assertTrue(t.toString(), t.gTile() >= 1 && t.gTile() <= FlashAttnTiling.MAX_G_TILE);
        assertTrue(t.toString(), t.bk() >= 16 && t.bq() >= 1);
        assertTrue(t.toString(), rows * 64L + t.bk() * 64L + (long) rows * t.bk() + 3L * rows <= t.smemFloats());
        assertTrue(t.toString(), (long) rows * t.dV() <= FlashAttnTiling.ACC_PER_GROUP);
    }
}
