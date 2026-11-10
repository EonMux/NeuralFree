package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The hybrid dispatcher's cost model ({@link MoeSplitDispatcher}, Avalon moe-cache.cu moe_hosted_split): steps
 * with one token never split, a move happens only when the copy ends the MoE window sooner than the host rows,
 * an expert already in the cache is never a candidate, the plan never takes more slots than the cache has free,
 * two routed experts move together when both copies pay off, and the greedy extension past the best pair
 * scores the experts it would actually move, taking the one that ends the window soonest while that beats the best
 * move so far.
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

    @Test
    public void theExtensionScoresTheExpertsItWouldMove() {
        // The pair 5+6 makes the step GPU-bound: any further copy only delays the moved rows. The tiny experts at
        // the lowest ids would gain as a set of three against no move at all, which must not count for one of them.
        int[] r = new int[EXPERTS];
        r[0] = 1;
        r[1] = 1;
        r[2] = 1;
        r[5] = 100;
        r[6] = 100;
        boolean[] cached = new boolean[EXPERTS];
        MoeSplitDispatcher.Model model = MoeSplitDispatcher.Model.of(8);
        MoeSplitDispatcher.Plan p = new MoeSplitDispatcher(model).decide(32, r, cached, 4, SMALL_EXPERT_BYTES,
                TENSORS);
        assertArrayEquals(new int[]{5, 6}, p.migrate());
        assertTrue(windowOf(model, r, cached, new int[]{5, 6, 0}, SMALL_EXPERT_BYTES)
                > windowOf(model, r, cached, new int[]{5, 6}, SMALL_EXPERT_BYTES));
    }

    @Test
    public void everyExtensionStepEndsTheWindowSooner() {
        // Six hot experts on a fast link: the pair alone leaves four hot experts on the host, so the extension
        // takes the hottest remaining one each round.
        int[] r = new int[16];
        r[1] = 2;
        r[9] = 120;
        r[10] = 150;
        r[11] = 90;
        r[12] = 140;
        r[13] = 100;
        r[14] = 130;
        boolean[] cached = new boolean[16];
        MoeSplitDispatcher.Model model = MoeSplitDispatcher.Model.of(MoeSplitDispatcher.PCIE_BYTES_PER_US * 4, 1.0, 8);
        int[] plan = new MoeSplitDispatcher(model).decide(64, r, cached, 8, SMALL_EXPERT_BYTES, TENSORS).migrate();
        assertTrue("the extension runs: " + Arrays.toString(plan), plan.length > 2);
        assertArrayEquals(reference(model, 64, r, cached, 8, SMALL_EXPERT_BYTES), plan);
        for (int k = 3; k <= plan.length; k++) {
            assertTrue("expert " + plan[k - 1] + " ends the window sooner",
                    windowOf(model, r, cached, Arrays.copyOf(plan, k), SMALL_EXPERT_BYTES)
                            < windowOf(model, r, cached, Arrays.copyOf(plan, k - 1), SMALL_EXPERT_BYTES));
        }
    }

    @Test
    public void randomStepsMatchTheReferenceGreedyAndNeverLengthenTheWindow() {
        Random rnd = new Random(42);
        long expertBytes = 3L * 2048 * 768 * 144 / 256;
        int extended = 0;
        for (int trial = 0; trial < 4000; trial++) {
            int nExperts = 8 + rnd.nextInt(121);
            int nTok = 2 + rnd.nextInt(255);
            int[] r = new int[nExperts];
            for (int i = 0; i < nTok * 8; i++) {
                int e = (int) Math.min(nExperts - 1, Math.abs(rnd.nextGaussian()) * nExperts / 4.0);
                r[(e * 37) % nExperts]++;
            }
            boolean[] cached = new boolean[nExperts];
            for (int e = 0; e < nExperts; e++) {
                cached[e] = rnd.nextInt(4) == 0;
            }
            int free = 1 + rnd.nextInt(16);
            MoeSplitDispatcher.Model model = MoeSplitDispatcher.Model.of(13600.0 * (0.5 + 4 * rnd.nextDouble()),
                    0.5 + rnd.nextDouble(), 1 + rnd.nextInt(32));
            int[] plan = new MoeSplitDispatcher(model).decide(nTok, r, cached, free, expertBytes, TENSORS)
                    .migrate();
            String at = "trial " + trial;
            assertArrayEquals(at, reference(model, nTok, r, cached, free, expertBytes), plan);
            assertTrue(at, plan.length <= free);
            if (plan.length > 0) {
                assertTrue(at, windowOf(model, r, cached, plan, expertBytes)
                        < windowOf(model, r, cached, new int[0], expertBytes));
            }
            if (plan.length > 2) {
                extended++;
            }
        }
        assertTrue("the extension ran in " + extended + " trials", extended > 50);
    }

    @Test
    public void aPairThatOnlyTiesASingleExpertDoesNotMove() {
        // The GPU path dominates behind 1000 resident rows, so a move's window depends on its moved rows alone:
        // the pair of experts 1 and 2 (4 + 3 rows) ties expert 5 (7 rows), and the single expert wins, as in the
        // reference search, which scores every single expert before any pair.
        int[] r = {12, 4, 3, 9, 5, 7, 1000};
        boolean[] cached = new boolean[r.length];
        cached[6] = true;
        MoeSplitDispatcher.Model model = MoeSplitDispatcher.Model.of(54400.0, 0.5, 1);
        int[] plan = new MoeSplitDispatcher(model).decide(64, r, cached, 3, SMALL_EXPERT_BYTES, TENSORS).migrate();
        assertEquals(windowOf(model, r, cached, new int[]{1, 2}, SMALL_EXPERT_BYTES),
                windowOf(model, r, cached, new int[]{5}, SMALL_EXPERT_BYTES), 0.0);
        assertArrayEquals(new int[]{5}, plan);
        assertArrayEquals(reference(model, 64, r, cached, 3, SMALL_EXPERT_BYTES), plan);
    }

    @Test
    public void fewRowsBehindAResidentBacklogMatchTheReference() {
        // Small row counts behind a resident backlog make equal-window moves common.
        Random rnd = new Random(7);
        double[] rates = {13600.0, 54400.0, 1e6};
        int[] residents = {0, 10, 50, 200, 1000};
        long[] sizes = {1000L, SMALL_EXPERT_BYTES, 1_000_000L};
        for (int trial = 0; trial < 20000; trial++) {
            int n = 2 + rnd.nextInt(5);
            int[] r = new int[n + 1];
            for (int e = 0; e < n; e++) {
                r[e] = 1 + rnd.nextInt(12);
            }
            r[n] = residents[rnd.nextInt(residents.length)];
            boolean[] cached = new boolean[n + 1];
            cached[n] = true;
            int free = 1 + rnd.nextInt(6);
            long bytes = sizes[rnd.nextInt(sizes.length)];
            MoeSplitDispatcher.Model model = MoeSplitDispatcher.Model.of(rates[rnd.nextInt(rates.length)],
                    0.5 * (1 + rnd.nextInt(3)), 1 << rnd.nextInt(5));
            int[] plan = new MoeSplitDispatcher(model).decide(64, r, cached, free, bytes, TENSORS).migrate();
            assertArrayEquals("trial " + trial, reference(model, 64, r, cached, free, bytes), plan);
        }
    }

    /**
     * Avalon's moe_hosted_split search over expert sets, scored by {@link #windowOf}: every single expert, then every
     * pair, then, only past a best pair, one more expert at a time while one ends the window sooner. The cache's free
     * slots bound the move: a pair needs two, the extension stops when they are taken.
     */
    private static int[] reference(MoeSplitDispatcher.Model model, int nTokens, int[] r, boolean[] cached,
                                   int free, long bytes) {
        if (nTokens < 2 || free == 0) {
            return new int[0];
        }
        java.util.List<Integer> cand = new java.util.ArrayList<>();
        for (int e = 0; e < r.length; e++) {
            if (!cached[e] && r[e] > 0) {
                cand.add(e);
            }
        }
        double all = windowOf(model, r, cached, new int[0], bytes);
        double bestGain = 0.0;
        int[] best = new int[0];
        for (int i = 0; i < cand.size(); i++) {
            double g = all - windowOf(model, r, cached, new int[]{cand.get(i)}, bytes);
            if (g > bestGain) {
                bestGain = g;
                best = new int[]{cand.get(i)};
            }
        }
        for (int i = 0; i < cand.size() && free >= 2; i++) {
            for (int j = i + 1; j < cand.size(); j++) {
                double g = all - windowOf(model, r, cached, new int[]{cand.get(i), cand.get(j)}, bytes);
                if (g > bestGain) {
                    bestGain = g;
                    best = new int[]{cand.get(i), cand.get(j)};
                }
            }
        }
        if (best.length != 2) {
            return best;
        }
        int[] plan = best;
        while (plan.length < free) {
            int add = -1;
            for (int e : cand) {
                if (Arrays.stream(plan).anyMatch(x -> x == e)) {
                    continue;
                }
                int[] next = Arrays.copyOf(plan, plan.length + 1);
                next[plan.length] = e;
                double g = all - windowOf(model, r, cached, next, bytes);
                if (g > bestGain) {
                    bestGain = g;
                    add = e;
                }
            }
            if (add < 0) {
                break;
            }
            plan = Arrays.copyOf(plan, plan.length + 1);
            plan[plan.length - 1] = add;
        }
        return plan;
    }

    /** The window when exactly the experts of {@code move}, of {@code bytes} each, copy into the cache. */
    private static double windowOf(MoeSplitDispatcher.Model model, int[] r, boolean[] cached, int[] move,
                                   long bytes) {
        boolean[] moved = new boolean[r.length];
        for (int e : move) {
            moved[e] = true;
        }
        int d = 0;
        int cpuRows = 0;
        int movedRows = 0;
        int resident = 0;
        for (int e = 0; e < r.length; e++) {
            if (cached[e]) {
                resident += r[e];
            } else if (moved[e]) {
                movedRows += r[e];
            } else if (r[e] > 0) {
                d++;
                cpuRows += r[e];
            }
        }
        int m = move.length;
        return model.windowUs(d, cpuRows, movedRows, TENSORS * m,
                model.copiesUs(bytes * m, TENSORS * m), resident);
    }
}
