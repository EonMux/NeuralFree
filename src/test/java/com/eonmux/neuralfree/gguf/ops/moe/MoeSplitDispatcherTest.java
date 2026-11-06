package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The hybrid dispatcher's cost model ({@link MoeSplitDispatcher}, Avalon moe-cache.cu moe_hosted_split): steps
 * with one token never split, a move happens only when the copy ends the MoE window sooner than the host rows,
 * an expert already in the cache is never a candidate, the plan never takes more slots than the cache has free,
 * and two routed experts move together when both copies pay off.
 */
public class MoeSplitDispatcherTest {

    private static final int EXPERTS = 8;
    private static final int TENSORS = 3;
    private static final long SMALL_EXPERT_BYTES = 100_000L;

    private static int[] rows(int expertWithRows, int nRows) {
        int[] r = new int[EXPERTS];
        r[expertWithRows] = nRows;
        return r;
    }

    private static MoeSplitDispatcher dispatcher() {
        return new MoeSplitDispatcher(MoeSplitDispatcher.Model.of(8));
    }

    @Test
    public void oneTokenStepsNeverSplit() {
        MoeSplitDispatcher.Plan p = dispatcher().decide(1, rows(3, 200), new boolean[EXPERTS], 2,
                SMALL_EXPERT_BYTES, TENSORS);
        assertTrue(p.isEmpty());
    }

    @Test
    public void noFreeSlotsMeansNoMove() {
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, rows(3, 200), new boolean[EXPERTS], 0,
                SMALL_EXPERT_BYTES, TENSORS);
        assertTrue(p.isEmpty());
    }

    @Test
    public void noRoutedRowsMeansNoMove() {
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, new int[EXPERTS], new boolean[EXPERTS], 2,
                SMALL_EXPERT_BYTES, TENSORS);
        assertTrue(p.isEmpty());
    }

    @Test
    public void aHeavilyRoutedExpertMovesWhenTheCopyIsCheaper() {
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, rows(3, 200), new boolean[EXPERTS], 2,
                SMALL_EXPERT_BYTES, TENSORS);
        assertArrayEquals(new int[]{3}, p.migrate());
    }

    @Test
    public void aHugeExpertStaysOnTheHost() {
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, rows(3, 200), new boolean[EXPERTS], 2,
                10_000_000_000L, TENSORS);
        assertTrue(p.isEmpty());
    }

    @Test
    public void cachedExpertsAreNeverCandidates() {
        boolean[] cached = new boolean[EXPERTS];
        cached[3] = true;
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, rows(3, 200), cached, 2, SMALL_EXPERT_BYTES, TENSORS);
        assertTrue(p.isEmpty());
    }

    @Test
    public void twoRoutedExpertsMoveTogether() {
        int[] r = new int[EXPERTS];
        r[2] = 100;
        r[5] = 100;
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, r, new boolean[EXPERTS], 2, SMALL_EXPERT_BYTES,
                TENSORS);
        assertArrayEquals(new int[]{2, 5}, p.migrate());
    }

    @Test
    public void thePlanNeverExceedsTheFreeSlots() {
        int[] r = new int[EXPERTS];
        r[2] = 100;
        r[5] = 100;
        MoeSplitDispatcher.Plan p = dispatcher().decide(32, r, new boolean[EXPERTS], 1, SMALL_EXPERT_BYTES,
                TENSORS);
        assertEquals(1, p.migrate().length);
        assertEquals(2, p.migrate()[0]);
    }

    @Test
    public void modelCostsFollowTheFittedTerms() {
        MoeSplitDispatcher.Model m = MoeSplitDispatcher.Model.of(8);
        assertEquals(0.0, m.cpuUs(0, 100, 0), 0.0);
        assertEquals(MoeSplitDispatcher.CPU_FIXED_US + MoeSplitDispatcher.CPU_EXPERT_US
                        + 100 * MoeSplitDispatcher.CPU_ROW_US,
                m.cpuUs(1, 100, 0), 1e-9);
        MoeSplitDispatcher.Model half = MoeSplitDispatcher.Model.of(MoeSplitDispatcher.PCIE_BYTES_PER_US, 0.5, 4);
        double share = MoeSplitDispatcher.CPU_FIT_THREADS / 4.0;
        assertEquals(0.5 * (MoeSplitDispatcher.CPU_FIXED_US + (MoeSplitDispatcher.CPU_EXPERT_US
                        + MoeSplitDispatcher.CPU_ROW_US * 100) * share),
                half.cpuUs(1, 100, 0), 1e-9);
        assertEquals(MoeSplitDispatcher.COPY_TENSOR_US * (1 + TENSORS) + SMALL_EXPERT_BYTES / 13600.0,
                m.copiesUs(SMALL_EXPERT_BYTES, TENSORS), 1e-9);
        assertFalse(m.toString().isEmpty());
    }

    @Test
    public void argumentsAreChecked() {
        MoeSplitDispatcher d = dispatcher();
        assertThrows(IllegalArgumentException.class,
                () -> d.decide(2, new int[3], new boolean[4], 1, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> d.decide(2, rows(0, -1), new boolean[EXPERTS], 1, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> d.decide(2, new int[EXPERTS], new boolean[EXPERTS], -1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> MoeSplitDispatcher.Model.of(0, 1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> MoeSplitDispatcher.Model.of(13600.0, -1.0, 1));
        assertThrows(IllegalArgumentException.class, () -> MoeSplitDispatcher.Model.of(13600.0, 1.0, 0));
    }
}
