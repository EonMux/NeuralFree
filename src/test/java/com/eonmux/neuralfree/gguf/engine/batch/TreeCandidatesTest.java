package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftVocabMapping;
import com.eonmux.neuralfree.gguf.engine.RowTopK;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The children of a tree draft's row: the host selection over a whole row picks the ids a stable full sort of the
 * row picks (larger logit first, the lower id on ties, NaN and {@code -inf} never, each target id once) and scores
 * them by their log-softmax over the row; a normalized top-k readout of {@link TreeCandidates#deviceWidth} candidates
 * gives the same children and scores over an identity vocabulary whose draft side is wider than the target's.
 */
public class TreeCandidatesTest {

    private static final int TARGET = 40;
    private static final int DRAFT = 44;

    @Test
    public void theHostSelectionPicksWhatAStableFullSortPicks() {
        Random random = new Random(11);
        DraftVocabMapping identity = identity(TARGET, DRAFT);
        for (int trial = 0; trial < 500; trial++) {
            float[] row = row(random, DRAFT);
            for (int branches = 1; branches <= 6; branches++) {
                TreeCandidates c = TreeCandidates.ofRow(row.clone(), branches, identity);
                int[] expected = sortedChildren(row, branches, identity);
                assertArrayEquals("trial " + trial + " branches " + branches, expected, ids(c));
                double lse = RowTopK.logSumExp(row);
                for (int i = 0; i < c.size(); i++) {
                    assertEquals(row[c.id(i)] - lse, c.logProb(i), 1e-12);
                }
            }
        }
    }

    @Test
    public void aNormalizedReadoutOfTheDeviceWidthGivesTheWholeRowsChildren() {
        Random random = new Random(5);
        DraftVocabMapping identity = identity(TARGET, DRAFT);
        for (int trial = 0; trial < 300; trial++) {
            // the device readout never ranks a NaN; the host top-k standing in for it here takes rows without one
            float[][] rows = {noNaN(row(random, DRAFT)), noNaN(row(random, DRAFT)), noNaN(row(random, DRAFT))};
            for (int branches = 1; branches <= 5; branches++) {
                int width = TreeCandidates.deviceWidth(branches, identity);
                assertEquals(branches + DRAFT - TARGET, width);
                RowTopK top = RowTopK.ofNormalized(rows, width);
                for (int r = 0; r < rows.length; r++) {
                    TreeCandidates host = TreeCandidates.ofRow(rows[r].clone(), branches, identity);
                    TreeCandidates device = TreeCandidates.ofTopK(top, r, branches, identity);
                    assertArrayEquals("trial " + trial + " row " + r, ids(host), ids(device));
                    for (int i = 0; i < host.size(); i++) {
                        assertEquals(host.logProb(i), device.logProb(i), 1e-12);
                    }
                }
            }
        }
    }

    @Test
    public void aReadoutTooWideForTheDeviceOrAnIntersectionReadsWholeRows() {
        assertEquals(-1, TreeCandidates.deviceWidth(3, identity(10, 10 + RowTopK.MAX_DEVICE_CANDIDATES)));
        assertEquals(3, TreeCandidates.deviceWidth(3, identity(10, 10)));
    }

    /** The children of the reference selection: a stable sort of the ranking ids, then each target id once. */
    private static int[] sortedChildren(float[] row, int k, DraftVocabMapping vocab) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < row.length; i++) {
            if (!Float.isNaN(row[i]) && row[i] != Float.NEGATIVE_INFINITY) {
                order.add(i);
            }
        }
        order.sort((a, b) -> row[a] == row[b] ? 0 : row[a] > row[b] ? -1 : 1);
        Set<Integer> seen = new HashSet<>();
        List<Integer> kept = new ArrayList<>();
        for (int id : order) {
            int target = vocab.toTarget(id);
            if (kept.size() < k && target >= 0 && seen.add(target)) {
                kept.add(id);
            }
        }
        return kept.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Logits with ties, NaN, {@code -inf} and the strongest values often on ids past the target vocabulary. */
    private static float[] row(Random random, int n) {
        float[] row = new float[n];
        for (int i = 0; i < n; i++) {
            int kind = random.nextInt(20);
            row[i] = kind == 0 ? Float.NaN : kind == 1 ? Float.NEGATIVE_INFINITY
                    : kind < 6 ? random.nextInt(3) : (float) random.nextGaussian();
            if (i >= TARGET && random.nextBoolean()) {
                row[i] = 9f + random.nextInt(2);
            }
        }
        return row;
    }

    private static float[] noNaN(float[] row) {
        for (int i = 0; i < row.length; i++) {
            if (Float.isNaN(row[i])) {
                row[i] = -1f;
            }
        }
        return row;
    }

    private static int[] ids(TreeCandidates c) {
        int[] ids = new int[c.size()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = c.id(i);
        }
        return ids;
    }

    private static DraftVocabMapping identity(int target, int draft) {
        return DraftVocabMapping.identity(target, draft);
    }
}
