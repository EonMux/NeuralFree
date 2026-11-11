package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The split plan of the grouped-query dense decode kernel ({@link FlashAttnTiling#decodeOnline}) and the choice of
 * the grouped-query dense kernel for the plain one-row step ({@link FlashPlan#gqaRowKernelName}): the split count
 * aims at a fixed number of work-groups per compute unit at every depth instead of growing with the key range, every
 * step's splits cover its keys inside the partial capacity, and the grid follows the work-group's head tile.
 */
public class FlashOnlineDecodePlanTest {

    private static final int SMEM = 11264;
    private static final int CU = 60;

    @Test
    public void theGroupedPlanHoldsTheWholeGroupAndAimsAtTheUnits() {
        FlashAttnTiling p = FlashAttnTiling.decodeOnline(32, 8, 128, 128, 40960, SMEM, CU);
        assertTrue(p.online());
        assertEquals(FlashAttnTiling.Plan.DECODE, p.plan());
        assertEquals(4, p.gTile());
        assertEquals(1, p.groupTiles());
        assertEquals(0, p.smemUsed());
        int wanted = (FlashAttnTiling.ONLINE_GROUPS_PER_UNIT * CU + 7) / 8;
        assertEquals(wanted, p.maxSplits());
        assertArrayEquals(new long[]{3L * FlashAttnTiling.WG, 8}, p.decodeGlobal(3));
        assertEquals(32L * wanted * (128 + 2), p.decodePartialFloats());
        // deep steps keep about the wanted split count (the 32-key rounding of the length takes a few off): only
        // the split length grows with the depth
        for (int keys : new int[]{4096, 16384, 40960}) {
            int count = p.decodeSplit(keys).count();
            assertTrue(keys + " keys: " + count + " splits", count <= wanted && count > wanted * 2 / 3);
        }
    }

    /** The decode plan caps the fast path's split at the work-group, so its split count grows with depth. */
    @Test
    public void theOnlinePlanNeedsFarFewerSplitsThanTheFastPathPlanAtDepth() {
        FlashAttnTiling fast = FlashAttnTiling.decode(32, 8, 128, 128, 40960, SMEM, CU, true);
        FlashAttnTiling online = FlashAttnTiling.decodeOnline(32, 8, 128, 128, 40960, SMEM, CU);
        assertEquals(128, fast.decodeSplit(16384).count());
        assertTrue(online.decodeSplit(16384).count() <= (FlashAttnTiling.ONLINE_GROUPS_PER_UNIT * CU + 7) / 8);
    }

    @Test
    public void onlyTheGroupedKernelRunsTheOnlinePlan() {
        FlashAttnTiling decode = FlashAttnTiling.decode(16, 8, 128, 128, 40960, SMEM, CU, true);
        assertSame(decode, FlashPlan.rowPlan(AttnKernels.FLASH_DECODE_ROW_F16, decode, 16, 8, 128, 128, 40960, SMEM,
                CU));
        FlashAttnTiling gqa = FlashPlan.rowPlan(AttnKernels.flashDecodeGqa(KvDType.F16, KvDType.F16), decode, 16, 8,
                128, 128, 40960, SMEM, CU);
        assertTrue(gqa.online());
        assertEquals(2, gqa.gTile());
        assertFalse(decode.online());
    }

    @Test
    public void everyStepsSplitsCoverItsKeysInsideTheCapacity() {
        for (int nHead : new int[]{8, 16, 32}) {
            for (int units : new int[]{1, 7, 60, 132}) {
                for (int keys : new int[]{1, 31, 32, 500, 4097}) {
                    FlashAttnTiling p = FlashAttnTiling.decodeOnline(nHead, 8, 64, 64, keys, SMEM, units);
                    for (int n = 0; n <= keys; n++) {
                        FlashAttnTiling.Split s = p.decodeSplit(n);
                        String at = "n_head=" + nHead + " units=" + units + " keys=" + keys + " n=" + n + " " + s;
                        assertTrue(at, s.count() >= 1 && s.count() <= p.maxSplits());
                        assertEquals(at, 0, s.length() % 32);
                        assertTrue(at, s.length() >= p.minSplitLen() && s.length() <= p.maxSplitLen());
                        assertTrue(at, (long) s.count() * s.length() >= n);
                        assertTrue(at, n == 0 || (long) (s.count() - 1) * s.length() < n);
                    }
                }
            }
        }
    }

    @Test
    public void invalidShapesAreRefused() {
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodeOnline(30, 8, 128, 128, 4096, SMEM, CU));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodeOnline(32, 8, 128, 128, 0, SMEM, CU));
        assertThrows(IllegalArgumentException.class,
                () -> FlashAttnTiling.decodeOnline(32, 8, 128, 128, 4096, SMEM, 0));
    }

    @Test
    public void theGroupedKernelServesGroupedF16Bf16AndQ80DenseCaches() {
        for (KvDType t : new KvDType[]{KvDType.F16, KvDType.BF16, KvDType.Q8_0}) {
            for (int d : new int[]{64, 128}) {
                for (int g : new int[]{2, 3, 4}) {
                    assertEquals(AttnKernels.flashDecodeGqa(t, t),
                            FlashPlan.gqaRowKernelName(t, t, false, 8 * g, 8, d, d, 8 * d, 8 * d, false));
                }
            }
        }
        assertEquals("LLM_FlashDecodeGqa_F16", AttnKernels.flashDecodeGqa(KvDType.F16, KvDType.F16));
    }

    @Test
    public void theGroupedKernelRefusesWhatItDoesNotHold() {
        KvDType f16 = KvDType.F16;
        // one query head per KV head: the per-head kernel reads each row once already
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 8, 8, 128, 128, 1024, 1024, false));
        // more query heads per KV head than the registers hold
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 64, 8, 64, 64, 512, 512, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 32, 8, 256, 256, 2048, 2048, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 32, 8, 96, 96, 768, 768, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, true, 32, 8, 128, 128, 1024, 1024, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 32, 8, 128, 128, 1028, 1024, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 32, 8, 128, 128, 1024, 1028, false));
        assertNull(FlashPlan.gqaRowKernelName(f16, f16, false, 32, 8, 128, 128, 1024, 1024, true));
        assertNull(FlashPlan.gqaRowKernelName(KvDType.Q4_0, KvDType.Q4_0, false, 32, 8, 128, 128, 1024, 1024,
                false));
        // a Q8_0 row stride must hold whole 32-value blocks
        assertNull(FlashPlan.gqaRowKernelName(KvDType.Q8_0, KvDType.Q8_0, false, 32, 8, 128, 128, 1040, 1024,
                false));
        assertNull(FlashPlan.gqaRowKernelName(KvDType.Q8_0, KvDType.Q8_0, false, 32, 8, 128, 128, 1024, 1040,
                false));
        assertNull(FlashPlan.gqaRowKernelName(f16, KvDType.BF16, false, 32, 8, 128, 128, 1024, 1024, false));
        assertFalse(FlashAttnTiling.gqaDecodeServes(32, 8, 128, 64));
    }
}
