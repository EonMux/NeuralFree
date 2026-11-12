package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The subgroup flash-attention gate ({@link SubgroupFlashAttention}) and its plans ({@link FlashAttnTiling#decodeSubgroup},
 * {@link FlashAttnTiling#decodePagedSubgroup}, {@link FlashAttnTiling#prefillSubgroup}).
 */
public class SubgroupFlashAttentionTest {

    private static final int SMEM = 11264;
    private static final int CU = 64;

    @Test
    public void shapeGateMirrorsTheKernelPlanCheck() {
        for (KvDType t : new KvDType[]{KvDType.F32, KvDType.F16, KvDType.BF16}) {
            assertTrue(t + " d=128", SubgroupFlashAttention.shapeServes(t, t, 128, 128));
            assertTrue(t + " d=64", SubgroupFlashAttention.shapeServes(t, t, 64, 64));
            assertTrue(t + " d=80", SubgroupFlashAttention.shapeServes(t, t, 80, 80));
        }
        assertFalse("wider than one subgroup's lanes", SubgroupFlashAttention.shapeServes(KvDType.F16, KvDType.F16, 256, 256));
        assertFalse("not a multiple of the dims per lane", SubgroupFlashAttention.shapeServes(KvDType.F16, KvDType.F16, 68, 68));
        assertFalse("unequal head dims", SubgroupFlashAttention.shapeServes(KvDType.F16, KvDType.F16, 128, 64));
        assertFalse("mixed cache types", SubgroupFlashAttention.shapeServes(KvDType.F32, KvDType.F16, 128, 128));
        assertTrue("Q8_0 d=128", SubgroupFlashAttention.shapeServes(KvDType.Q8_0, KvDType.Q8_0, 128, 128));
        assertTrue("Q8_0 d=64", SubgroupFlashAttention.shapeServes(KvDType.Q8_0, KvDType.Q8_0, 64, 64));
        assertFalse("Q8_0 rows of part blocks", SubgroupFlashAttention.shapeServes(KvDType.Q8_0, KvDType.Q8_0, 80, 80));
        assertFalse("mixed Q8_0 and F16", SubgroupFlashAttention.shapeServes(KvDType.Q8_0, KvDType.F16, 128, 128));
        assertFalse("other quantized cache", SubgroupFlashAttention.shapeServes(KvDType.Q4_0, KvDType.Q4_0, 128, 128));
    }

    /** Row strides must keep every load on the kernels' alignment: whole 8-value groups, whole Q8_0 blocks. */
    @Test
    public void strideGateMirrorsTheKernelPlanCheck() {
        assertTrue(SubgroupFlashAttention.stridesServe(KvDType.F16, 1024, 1024));
        assertTrue(SubgroupFlashAttention.stridesServe(KvDType.F16, 1032, 1032));
        assertFalse(SubgroupFlashAttention.stridesServe(KvDType.F16, 1028, 1024));
        assertTrue(SubgroupFlashAttention.stridesServe(KvDType.Q8_0, 1024, 1024));
        assertFalse("K rows of part blocks", SubgroupFlashAttention.stridesServe(KvDType.Q8_0, 1032, 1024));
        assertFalse("V rows of part blocks", SubgroupFlashAttention.stridesServe(KvDType.Q8_0, 1024, 1040));
    }

    @Test
    public void planRunsOneHeadPerSixtyFourItemWorkGroup() {
        FlashAttnTiling p = FlashAttnTiling.decodeSubgroup(32, 8, 128, 128, 4096, SMEM, CU);
        assertTrue(p.subgroup());
        assertEquals(FlashAttnTiling.Plan.DECODE, p.plan());
        assertEquals(1, p.gTile());
        assertEquals(SubgroupFlashAttention.WORK_GROUP, 64);
        assertArrayEquals(new long[]{64, 1}, p.decodeLocalSize());
        assertArrayEquals(new long[]{3 * 64, 32}, p.decodeGlobal(3));
        assertArrayEquals(new long[]{2 * 64, 5 * 32}, p.decodePagedGlobal(5, 2));
        assertEquals(64, p.minSplitLen());

        FlashAttnTiling portable = FlashAttnTiling.decode(32, 8, 128, 128, 4096, SMEM, CU, false);
        assertFalse(portable.subgroup());
        assertArrayEquals(FlashAttnTiling.decodeLocal(), portable.decodeLocalSize());
    }

    @Test
    public void splitsAimAtTwoWorkGroupsPerComputeUnitAndCoverTheKeys() {
        FlashAttnTiling p = FlashAttnTiling.decodeSubgroup(16, 8, 128, 128, 40960, SMEM, CU);
        // 2 * 64 CUs over 16 heads: 8 splits wanted, never shorter than one work-group's first scan of 64 keys.
        assertEquals(1, p.decodeSplit(5).count());
        assertEquals(64, p.decodeSplit(5).length());
        assertEquals(2, p.decodeSplit(128).count());
        FlashAttnTiling.Split s = p.decodeSplit(40960);
        assertEquals(8, s.count());
        assertTrue(s.count() * s.length() >= 40960);
        assertTrue(p.maxSplits() >= s.count());
        assertTrue(p.decodePartialFloats() >= 16L * s.count() * (128 + 2));
    }

    /** A layer's single-row and paged batch rows over the same keys must keep one summation order. */
    @Test
    public void denseAndPagedPlansSplitAStepAlike() {
        FlashAttnTiling dense = FlashAttnTiling.decodeSubgroup(16, 8, 128, 128, 4096, SMEM, CU);
        FlashAttnTiling paged = FlashAttnTiling.decodePagedSubgroup(4, 16, 8, 128, 128, 1 << 15, SMEM, CU);
        for (int n : new int[]{0, 1, 63, 64, 65, 500, 2048, 4096}) {
            assertEquals("n=" + n, dense.decodeSplit(n).toString(), paged.decodeSplit(n).toString());
        }
    }

    @Test
    public void prefillPlanPacksEightRowsWithTheWidestPowerOfTwoHeadTile() {
        int[][] groups = {{16, 16, 1}, {16, 8, 2}, {12, 4, 2}, {32, 8, 4}, {64, 8, 8}, {64, 4, 8}};
        for (int[] g : groups) {
            FlashAttnTiling p = FlashAttnTiling.prefillSubgroup(g[0], g[1], 128, 128, 4096, SMEM, CU);
            String what = g[0] + "/" + g[1] + ": " + p;
            assertTrue(what, p.subgroup());
            assertEquals(what, FlashAttnTiling.Plan.TILED, p.plan());
            assertEquals(what, g[2], p.gTile());
            assertEquals(what, SubgroupFlashAttention.PREFILL_ROWS, p.rows());
            assertEquals(what, SubgroupFlashAttention.PREFILL_BLOCK, p.bk());
        }
        FlashAttnTiling p = FlashAttnTiling.prefillSubgroup(32, 8, 128, 128, 4096, SMEM, CU);
        assertArrayEquals(new long[]{64, 1, 1}, p.prefillLocalSize());
        // 512 tokens in tiles of bq = 2 over 8 KV heads of one head tile each
        assertArrayEquals(new long[]{256L * 64, 8, 1}, p.prefillGlobal(512, 1));
        FlashAttnTiling portable = FlashAttnTiling.prefill(32, 8, 128, 128, 4096, SMEM, CU);
        assertFalse(portable.subgroup());
        assertArrayEquals(FlashAttnTiling.prefillLocal(), portable.prefillLocalSize());
    }

    @Test
    public void prefillSplitsAShortStepOverTheKeyBlocks() {
        FlashAttnTiling p = FlashAttnTiling.prefillSubgroup(16, 8, 128, 128, 4096, SMEM, CU);
        // one 4-token tile per KV head: 8 work-groups, far below two per compute unit, so the keys split
        FlashAttnTiling.Split s = p.prefillSplit(4, 4096, Long.MAX_VALUE);
        assertTrue(s.toString(), s.count() > 1 && s.length() % SubgroupFlashAttention.PREFILL_BLOCK == 0);
        assertTrue(s.toString(), (long) s.count() * s.length() >= 4096);
        // a 512-token prefill fills the device without splitting
        assertEquals(1, p.prefillSplit(512, 4096, Long.MAX_VALUE).count());
    }

    @Test
    public void shapesTheKernelsDoNotServeAreRefused() {
        int[][] dims = {{256, 256}, {128, 64}, {68, 68}};
        for (int[] d : dims) {
            try {
                FlashAttnTiling.decodeSubgroup(8, 2, d[0], d[1], 1024, SMEM, CU);
                fail("decode d=" + d[0] + "x" + d[1] + " must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("subgroup attention"));
            }
            try {
                FlashAttnTiling.prefillSubgroup(8, 2, d[0], d[1], 1024, SMEM, CU);
                fail("prefill d=" + d[0] + "x" + d[1] + " must be refused");
            } catch (IllegalArgumentException expected) {
                assertTrue(expected.getMessage(), expected.getMessage().contains("subgroup attention"));
            }
        }
        try {
            FlashAttnTiling.decodePagedSubgroup(0, 8, 2, 128, 128, 1024, SMEM, CU);
            fail("a paged plan needs a sequence");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("sequence"));
        }
    }
}
