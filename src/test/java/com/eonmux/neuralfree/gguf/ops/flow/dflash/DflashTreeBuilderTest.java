package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.engine.DraftTree;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The best-first dflash tree builder: the strongest chain pops first with its ancestors ahead of it, siblings
 * carry distinct tokens, and the node, leaf and layout-row budgets bound the proposal (SpecPilot's
 * {@code _tree_build_kernel} over a plain DFlash top-k table).
 */
public class DflashTreeBuilderTest {

    /** Three block rows of two candidates each; row scores keep the chain 10, 20, 30 strictly strongest. */
    private static DflashTreeBuilder.TopKTable table() {
        return new DflashTreeBuilder.TopKTable(1, 3, 2,
                new int[] {10, 11, 20, 21, 30, 31},
                new float[] {2.0f, 1.2f, 3.0f, 2.0f, 0.5f, -0.5f});
    }

    private static int[] tokens(DraftTree t) {
        int[] out = new int[t.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = t.nodes().get(i).token();
        }
        return out;
    }

    private static int[] parents(DraftTree t) {
        int[] out = new int[t.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = t.nodes().get(i).parent();
        }
        return out;
    }

    @Test
    public void theStrongestChainPopsFirstAndParentsPrecedeChildren() {
        DraftTree t = DflashTreeBuilder.build(table(), 7, 6, 3, 64, 1.0f);
        // pop order: the greedy chain 10, 20, 30 pops whole before the secondary root 11, whose row-2 candidate
        // 31 follows; the weakest path's candidates come last
        assertEquals("pop order over the candidate pool", "[10, 20, 30, 11, 20, 21]",
                java.util.Arrays.toString(tokens(t)));
        assertTrue("parents precede children", t.nodes().stream().allMatch(n -> n.parent() < 0
                || t.nodes().get(n.parent()).depth() == n.depth() - 1));
        for (int i = 0; i < t.size(); i++) {
            assertTrue("every parent precedes its child", t.nodes().get(i).parent() < i);
        }
        assertFalse("the proposal branches", t.linear());
        assertEquals("the tree reaches every row", 3, t.depth());
    }

    @Test
    public void siblingsCarryDistinctTokens() {
        DraftTree t = DflashTreeBuilder.build(table(), 7, 64, 4, 64, 1.0f);
        List<Set<Integer>> siblingTokens = new ArrayList<>();
        for (int i = 0; i < t.size(); i++) {
            Set<Integer> s = new HashSet<>();
            for (int j = 0; j < t.size(); j++) {
                if (t.nodes().get(j).parent() == i && !s.add(t.nodes().get(j).token())) {
                    siblingTokens.add(s);
                }
            }
        }
        assertEquals("no node has two children with one token", 0, siblingTokens.size());
        // the duplicate id of one conditioned row keeps only its best-ranked occurrence
        DflashTreeBuilder.TopKTable duplicate = new DflashTreeBuilder.TopKTable(1, 2, 2,
                new int[] {10, 11, 20, 20}, new float[] {2.0f, 1.0f, 3.0f, 2.5f});
        DraftTree td = DflashTreeBuilder.build(duplicate, 7, 8, 4, 64, 1.0f);
        long childrenOfTen = td.nodes().stream().filter(n -> n.parent() == 0).count();
        assertEquals("the duplicated candidate id is proposed once", 1, childrenOfTen);
    }

    @Test
    public void theNodeBudgetCapsTheProposal() {
        DraftTree t = DflashTreeBuilder.build(table(), 7, 3, 8, 64, 1.0f);
        assertEquals("only the first three pooled candidates", 3, t.size());
        assertEquals("[10, 20, 30]", java.util.Arrays.toString(tokens(t)));
        assertEquals("an empty budget proposes nothing", 0,
                DflashTreeBuilder.build(table(), 7, 0, 8, 64, 1.0f).size());
    }

    @Test
    public void theLeafBudgetCapsTheBranches() {
        DraftTree t = DflashTreeBuilder.build(table(), 7, 8, 1, 64, 1.0f);
        assertTrue("one leaf means one chain", t.linear());
        assertEquals("the chain of best candidates", "[10, 20, 30]", java.util.Arrays.toString(tokens(t)));
        assertArrayEquals("every node is a single chain", new int[] {-1, 0, 1}, parents(t));
    }

    @Test
    public void theRowBudgetBoundsTheLayoutPaths() {
        // the chains [10, 20, 30] and the fresh path [11]: three rows plus one, and the pool's weaker candidates
        // would each open or extend a path past the budget of four
        DraftTree t = DflashTreeBuilder.build(table(), 7, 8, 8, 4, 1.0f);
        assertEquals("[10, 20, 30, 11]", java.util.Arrays.toString(tokens(t)));
        int leaves = 0;
        boolean[] hasChild = new boolean[t.size()];
        for (DraftTree.Node n : t.nodes()) {
            if (n.parent() >= 0) {
                hasChild[n.parent()] = true;
            }
        }
        for (boolean b : hasChild) {
            leaves += b ? 0 : 1;
        }
        assertEquals("two leaf paths", 2, leaves);
        assertEquals("the budget stops the weaker candidates", 4, t.size());
    }

    @Test
    public void theTemperatureShiftsTheRankingFromChainToBreadth() {
        // the secondary root 11 nearly ties 10; a cold temperature keeps the chain ahead of it, a hot one lets
        // every candidate's log-softmax near -log 2, so the insertion order (breadth) wins
        DraftTree cold = DflashTreeBuilder.build(table(), 7, 6, 4, 64, 0.01f);
        assertEquals("a cold tree is the greedy chain first", 10, cold.nodes().get(0).token());
        assertEquals("the chain's depth-2 extension outranks the secondary root", 20, cold.nodes().get(1).token());
        DraftTree hot = DflashTreeBuilder.build(table(), 7, 6, 4, 64, 1000.0f);
        assertEquals("a hot tree proposes the first row's candidates first", 11, hot.nodes().get(1).token());
    }

    @Test
    public void anAnchorFirstTableRenumbersTheDepthsFromOne() {
        DflashTreeBuilder.TopKTable anchorFirst = new DflashTreeBuilder.TopKTable(0, 2, 2,
                new int[] {10, 11, 20, 21}, new float[] {2.0f, 1.0f, 3.0f, 2.9f});
        DraftTree t = DflashTreeBuilder.build(anchorFirst, 7, 8, 4, 64, 1.0f);
        assertEquals("the first-depth nodes sit at depth 1", 1, t.nodes().get(0).depth());
        assertEquals("their parents are the pending token", -1, t.nodes().get(0).parent());
        assertEquals(2, t.nodes().get(2).depth());
    }

    @Test
    public void aNonPositiveTemperatureIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> DflashTreeBuilder.build(table(), 7, 8, 4, 64, 0.0f));
        assertThrows(IllegalArgumentException.class, () -> DflashTreeBuilder.build(table(), 7, 8, 4, 64, -1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> DflashTreeBuilder.build(table(), 7, 8, 4, 64, Float.NaN));
        assertThrows(IllegalArgumentException.class,
                () -> DflashTreeBuilder.build(table(), 7, 8, 4, 64, Float.POSITIVE_INFINITY));
    }

    @Test
    public void aRowOfOneScoreStillBuildsAndANegativeAnchorProposesNothing() {
        DflashTreeBuilder.TopKTable one = new DflashTreeBuilder.TopKTable(1, 2, 1,
                new int[] {10, 20}, new float[] {2.0f, 3.0f});
        DraftTree t = DflashTreeBuilder.build(one, 7, 8, 4, 64, 1.0f);
        assertTrue("a single candidate per row is a chain", t.linear());
        assertEquals("[10, 20]", java.util.Arrays.toString(tokens(t)));
        assertEquals("no anchor, no tree", 0, DflashTreeBuilder.build(table(), -1, 8, 4, 64, 1.0f).size());
    }
}
