package com.eonmux.neuralfree.gguf.gpu.weights;

import com.eonmux.neuralfree.gguf.gpu.weights.MoeCacheSlru.Fill;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The admission policy of {@link MoeCacheSlru} against the semantics of llama.cpp's {@code moe_cache_slru}
 * (src/llama-moe-cache.cpp:17-155): hits promote to the protected half, misses take probation victims, a pending
 * slot is never a victim, and a demand wider than the pool assigns nothing. Lent slots (Avalon's prefetch zones)
 * evict their expert, are never victims, and come back empty as the next victim.
 */
public class MoeCacheSlruTest {

    @Test
    public void firstPlanFillsThePoolInExpertOrder() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 8, 4);
        List<Fill> fills = new ArrayList<>();
        int hits = slru.plan(0, new int[]{1, 3, 5, 7}, 4, fills);

        assertEquals(0, hits);
        assertEquals(4, fills.size());
        for (int i = 0; i < 4; i++) {
            assertEquals(2 * i + 1, fills.get(i).expert);
            assertEquals(i, fills.get(i).slot);
            assertEquals(fills.get(i).slot, slru.slotOf(0, fills.get(i).expert));
        }
        assertEquals(-1, slru.slotOf(0, 0));
    }

    @Test
    public void hitsPromoteAndMissesEvictTheProbationLru() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 8, 4);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1, 2, 3}, 4, fills);
        fills.clear();
        // expert 0 is hit and protected; the miss evicts expert 1, the probation LRU
        int hits = slru.plan(0, new int[]{0, 0, 4}, 3, fills);

        assertEquals(1, hits);
        assertEquals(1, fills.size());
        assertEquals(4, fills.get(0).expert);
        assertEquals(1, fills.get(0).slot);
        assertEquals(0, slru.slotOf(0, 0));
        assertEquals(-1, slru.slotOf(0, 1));
        assertEquals(2, slru.slotOf(0, 2));
        assertEquals(3, slru.slotOf(0, 3));
        assertEquals(1, slru.slotOf(0, 4));
    }

    @Test
    public void aProtectedDemandSurvivesWhileAnUndemandedExpertIsEvicted() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 2);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1}, 2, fills);
        fills.clear();
        slru.plan(0, new int[]{0, 2}, 2, fills);

        assertEquals(0, slru.slotOf(0, 0));
        assertEquals(-1, slru.slotOf(0, 1));
        assertEquals(1, slru.slotOf(0, 2));
    }

    @Test
    public void aDemandWiderThanThePoolAssignsNothing() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 8, 2);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1}, 2, fills);
        fills.clear();
        int hits = slru.plan(0, new int[]{0, 2, 3}, 3, fills);

        assertEquals(-1, hits);
        assertTrue(fills.isEmpty());
        assertEquals(0, slru.slotOf(0, 0));
        assertEquals(1, slru.slotOf(0, 1));
    }

    @Test
    public void slotsAreSharedAcrossLayersOfOnePool() {
        MoeCacheSlru slru = new MoeCacheSlru(2, 4, 2);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1}, 2, fills);
        fills.clear();
        slru.plan(1, new int[]{2, 3}, 2, fills);

        assertEquals(2, fills.size());
        assertEquals(-1, slru.slotOf(0, 0));
        assertEquals(-1, slru.slotOf(0, 1));
        assertEquals(0, slru.slotOf(1, 2));
        assertEquals(1, slru.slotOf(1, 3));
    }

    @Test
    public void pendingSlotsAreNotVictimsUntilDone() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 1);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0}, 1, fills);
        slru.fillStarted(0);
        fills.clear();
        // the only slot is pending: the plan assigns nothing and the resident expert stays
        int hits = slru.plan(0, new int[]{1}, 1, fills);

        assertEquals(0, hits);
        assertTrue(fills.isEmpty());
        assertEquals(0, slru.slotOf(0, 0));
        assertEquals(-1, slru.slotOf(0, 1));
        assertTrue(slru.pending(0));
        slru.fillDone(0);
        assertEquals(0, slru.admit(0, 1));
        assertEquals(-1, slru.slotOf(0, 0));
        assertEquals(0, slru.slotOf(0, 1));
    }

    @Test
    public void admitPlacesAtTheProbationMrUEnd() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 2);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0}, 1, fills);
        int slot = slru.admit(0, 2);

        assertEquals(1, slot);
        assertEquals(1, slru.slotOf(0, 2));
        // the second admission must evict expert 2 (the probation MRU), not expert 0
        assertEquals(1, slru.admit(0, 3));
        assertEquals(-1, slru.slotOf(0, 2));
        assertEquals(0, slru.slotOf(0, 0));
    }

    @Test
    public void outOfRangeArgumentsAreRefusedByName() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 2);
        List<Fill> fills = new ArrayList<>();
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> slru.plan(1, new int[]{0}, 1, fills));
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> slru.plan(0, new int[]{4}, 1, fills));
        org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> slru.plan(0, new int[]{0}, 2, fills));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> slru.slotOf(0, -1));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> slru.fillStarted(2));
    }

    @Test
    public void sizesMustBePositive() {
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> new MoeCacheSlru(0, 4, 2));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> new MoeCacheSlru(1, 0, 2));
        org.junit.Assert.assertThrows(IllegalArgumentException.class, () -> new MoeCacheSlru(1, 4, 0));
    }

    @Test
    public void aLentSlotEvictsItsExpertAndIsNeverAVictim() {
        MoeCacheSlru slru = new MoeCacheSlru(2, 4, 3);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1, 2}, 3, fills);

        assertEquals(2, slru.lend(2));
        assertTrue(slru.lent(2));
        assertEquals(-1, slru.slotOf(0, 2));
        assertEquals(1, slru.lend(1));
        assertEquals(-1, slru.slotOf(0, 1));
        // only slot 0 is left for the other layer's two misses: one fill, never into a lent slot
        fills.clear();
        slru.plan(1, new int[]{0, 1}, 2, fills);
        assertEquals(1, fills.size());
        assertEquals(0, fills.get(0).slot);
        assertEquals(-1, slru.admit(1, 3));
    }

    @Test
    public void aRestoredSlotComesBackEmptyAsTheNextVictim() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 8, 4);
        List<Fill> fills = new ArrayList<>();
        slru.plan(0, new int[]{0, 1, 2, 3}, 4, fills);
        slru.lend(3);

        slru.restore(3);
        assertFalse(slru.lent(3));
        fills.clear();
        slru.plan(0, new int[]{6}, 1, fills);
        assertEquals("the restored slot is taken before any resident expert", 3, fills.get(0).slot);
        for (int e = 0; e < 3; e++) {
            assertEquals(e, slru.slotOf(0, e));
        }
    }

    @Test
    public void lendingRefusesAPendingOrLentSlotAndRestoreAnUnlentOne() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 2);
        slru.fillStarted(0);
        assertThrows(IllegalStateException.class, () -> slru.lend(0));
        slru.lend(1);
        assertThrows(IllegalStateException.class, () -> slru.lend(1));
        assertThrows(IllegalStateException.class, () -> slru.restore(0));
        assertThrows(IllegalArgumentException.class, () -> slru.lend(2));
    }

    @Test
    public void aLentSlotIsNotCountedAsEmpty() {
        MoeCacheSlru slru = new MoeCacheSlru(1, 4, 3);
        assertEquals(3, slru.emptySlots());
        slru.lend(2);
        assertEquals("a lent slot is no free slot of the pool", 2, slru.emptySlots());
        slru.restore(2);
        assertEquals(3, slru.emptySlots());
    }
}
