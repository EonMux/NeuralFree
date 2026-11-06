package com.eonmux.neuralfree.gguf.engine.batch;

import com.eonmux.neuralfree.gguf.engine.DraftTree;
import com.eonmux.neuralfree.gguf.engine.SpeculativeOptions;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The batch engine's verify budget ({@link TreeBudget}): the tiered rows total clamps like the reference's
 * {@code tokens_for_bs}, and every policy's cut keeps per request a node prefix of its proposal — the property that
 * makes a budget cut lossless, since the first {@code n} best-first nodes are the {@code n}-node proposal.
 */
public class TreeBudgetTest {

    private static SpeculativeOptions options(String rows, TreeBudgetPolicy policy) {
        SpeculativeOptions.Builder b = SpeculativeOptions.builder();
        if (rows != null) {
            b.treeRows(rows);
        }
        b.treeBudgetPolicy(policy);
        return b.build();
    }

    /** Two proposals: the strong chain scores {@code [0, -0.1, -0.2]}, the weak one {@code [0, -0.5, -0.9]}. */
    private static double[][] strongAndWeak() {
        return new double[][] {{0.0, -0.1, -0.2}, {0.0, -0.5, -0.9}};
    }

    @Test
    public void aConstantRowTotalCapsEveryStep() {
        TreeBudget budget = TreeBudget.of(options("6", TreeBudgetPolicy.FULL));
        assertEquals(TreeBudgetPolicy.FULL, budget.policy());
        assertEquals(6, budget.totalFor(2, 8));
        assertEquals(6, budget.totalFor(1, 8));
        // every sequence verifies at least its pending row: the total clamps up to the batch size
        assertEquals(10, budget.totalFor(10, 8));
    }

    @Test
    public void withoutTiersTheTotalIsTheFullWidthTrees() {
        TreeBudget budget = TreeBudget.of(options(null, TreeBudgetPolicy.FULL));
        assertEquals(15, budget.totalFor(3, 5));
        assertEquals(32, budget.totalFor(8, 4));
        // a total above the full-width trees still clamps to them plus the reference's padding allowance
        TreeBudget padded = TreeBudget.of(options("999", TreeBudgetPolicy.FULL));
        assertEquals(8 * 4 + TreeBudget.PAD_MAX, padded.totalFor(8, 4));
    }

    @Test
    public void aTierTableGivesTheFirstCoveringTier() {
        TreeBudget budget = TreeBudget.of(options("4:8;16:24", TreeBudgetPolicy.FULL));
        assertEquals(8, budget.totalFor(1, 4));
        assertEquals(8, budget.totalFor(4, 4));
        assertEquals(24, budget.totalFor(5, 4));
        assertEquals(24, budget.totalFor(16, 4));
        assertEquals(24, budget.totalFor(17, 4));
        // a batch size above the tier's total clamps the total up to the batch size
        assertEquals(64, budget.totalFor(64, 4));
    }

    @Test
    public void aScaledTierCeilsTheRowsPerRequest() {
        TreeBudget budget = TreeBudget.of(options("8:x2.5", TreeBudgetPolicy.FULL));
        assertEquals(8, budget.totalFor(3, 8));
        assertEquals(10, budget.totalFor(4, 8));
        assertEquals(20, budget.totalFor(8, 8));
    }

    @Test
    public void badRowsTablesRefuseNamingTheProperty() {
        assertThrows(IllegalArgumentException.class, () -> TreeBudget.of(options("0", TreeBudgetPolicy.FULL)));
        assertThrows(IllegalArgumentException.class, () -> TreeBudget.of(options("four:8", TreeBudgetPolicy.FULL)));
        assertThrows(IllegalArgumentException.class, () -> TreeBudget.of(options("4:8;;16:24",
                TreeBudgetPolicy.FULL)));
        assertThrows(IllegalArgumentException.class, () -> TreeBudget.of(options("8:x-1", TreeBudgetPolicy.FULL)));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> TreeBudget.of(options("8:-2", TreeBudgetPolicy.FULL)));
        assertTrue(refused.getMessage(), refused.getMessage().contains(SpeculativeOptions.PROP_TREE_ROWS));
    }

    @Test
    public void theFullPolicyKeepsEverythingWhenTheBudgetFits() {
        TreeBudget budget = TreeBudget.of(options(null, TreeBudgetPolicy.FULL));
        assertArrayEquals(new int[] {3, 3}, budget.cut(strongAndWeak(), 6));
        assertArrayEquals(new int[] {3, 3}, budget.cut(strongAndWeak(), 100));
    }

    @Test
    public void theFullPolicyDropsTheWeakestPrefixesWhenTheBudgetBinds() {
        TreeBudget budget = TreeBudget.of(options("4", TreeBudgetPolicy.FULL));
        // the global top two nodes past the roots sit on the strong chain's prefix, so the weak proposal keeps its
        // root only
        assertArrayEquals(new int[] {3, 1}, budget.cut(strongAndWeak(), 4));
    }

    @Test
    public void aBudgetAtTheBatchSizeKeepsOneNodeEach() {
        TreeBudget budget = TreeBudget.of(options("2", TreeBudgetPolicy.FULL));
        assertArrayEquals(new int[] {1, 1}, budget.cut(strongAndWeak(), 2));
    }

    @Test
    public void theToptPolicySelectsTheStrongestNodesOverAllRequests() {
        TreeBudget budget = TreeBudget.of(options("4", TreeBudgetPolicy.TOPT));
        assertArrayEquals(new int[] {3, 1}, budget.cut(strongAndWeak(), 4));
    }

    @Test
    public void theToptTotalWithoutTiersIsTheMeanTimesTheBatch() {
        TreeBudget budget = TreeBudget.of(options(null, TreeBudgetPolicy.TOPT));
        assertEquals(2, budget.toptTotalFor(2, 8)); // the default mean 1.0
        TreeBudget tabled = TreeBudget.of(SpeculativeOptions.builder().treeBudgetPolicy(TreeBudgetPolicy.TOPT)
                .treeMean("1:1.5;8:2.5").build());
        assertEquals(3, tabled.toptTotalFor(2, 8));
        assertEquals(20, tabled.toptTotalFor(8, 8));
        // clamped into [bs, bs * width]
        assertEquals(2, tabled.toptTotalFor(2, 1));
    }

    @Test
    public void theThreshPolicyKeepsNodesWhileGammaScoreStaysAboveLogLambda() {
        TreeBudget budget = TreeBudget.of(SpeculativeOptions.builder()
                .treeBudgetPolicy(TreeBudgetPolicy.THRESH).treeThreshGamma(1.0).treeThreshLambda(0.5).build());
        // every score of the strong chain stays above log(0.5) ~ -0.69; the weak one's -0.9 falls below it
        assertArrayEquals(new int[] {3, 2}, budget.cut(strongAndWeak(), 100));
    }

    @Test
    public void aNonPositiveLambdaKeepsEverything() {
        TreeBudget budget = TreeBudget.of(SpeculativeOptions.builder()
                .treeBudgetPolicy(TreeBudgetPolicy.THRESH).treeThreshLambda(0.0).build());
        assertArrayEquals(new int[] {3, 3}, budget.cut(strongAndWeak(), 100));
    }

    @Test
    public void theFixedPolicyCapsEveryRequest() {
        TreeBudget budget = TreeBudget.of(SpeculativeOptions.builder()
                .treeBudgetPolicy(TreeBudgetPolicy.FIXED).treeFixed(2).build());
        assertArrayEquals(new int[] {2, 2}, budget.cut(strongAndWeak(), 100));
    }

    @Test
    public void theMinFloorLiftsShortCountsAndBreaksTheSumLikeTheReference() {
        TreeBudget budget = TreeBudget.of(SpeculativeOptions.builder()
                .treeBudgetPolicy(TreeBudgetPolicy.FIXED).treeFixed(1).treeMinNodes(3).build());
        assertArrayEquals(new int[] {3, 3}, budget.cut(strongAndWeak(), 100));
        // a floor above what the total fills keeps the floor: the policy-studies setting
        assertArrayEquals(new int[] {3, 3}, budget.cut(strongAndWeak(), 2));
    }

    @Test
    public void everyCutKeepsPerRequestAPrefixWithinItsNodes() {
        TreeBudget budget = TreeBudget.of(options("5", TreeBudgetPolicy.FULL));
        double[][] scores = {{0.0, -0.05, -0.3, -0.31}, {0.0, -0.2, -0.21}};
        int[] out = budget.cut(scores, 5);
        assertEquals(scores.length, out.length);
        int sum = 0;
        for (int i = 0; i < out.length; i++) {
            assertTrue("kept " + out[i] + " of " + scores[i].length + " nodes", out[i] >= 1
                    && out[i] <= scores[i].length);
            sum += out[i];
        }
        assertTrue("the cut sum " + sum + " exceeds the total 5", sum <= 5);
    }

    @Test
    public void thePolicyParseAcceptsItsNamesAndRefusesOthers() {
        assertEquals(TreeBudgetPolicy.FULL, TreeBudgetPolicy.parse("k", "full"));
        assertEquals(TreeBudgetPolicy.TOPT, TreeBudgetPolicy.parse("k", "TOPT"));
        assertEquals(TreeBudgetPolicy.THRESH, TreeBudgetPolicy.parse("k", "thresh"));
        assertEquals(TreeBudgetPolicy.FIXED, TreeBudgetPolicy.parse("k", "fixed"));
        IllegalArgumentException refused = assertThrows(IllegalArgumentException.class,
                () -> TreeBudgetPolicy.parse("k", "weird"));
        assertTrue(refused.getMessage(), refused.getMessage().contains("k=weird"));
    }

    /** A branched proposal: the true chain plus a wrong sibling off its root. */
    private static TreeProposal branched() {
        List<DraftTree.Node> nodes = new ArrayList<>();
        // the true chain: tokens 11 -> 12 -> 13, depths 1..3
        nodes.add(new DraftTree.Node(11, -1, 1, 0));
        nodes.add(new DraftTree.Node(12, 0, 2, 0));
        // the wrong sibling under the root
        nodes.add(new DraftTree.Node(99, 0, 2, 0));
        nodes.add(new DraftTree.Node(13, 1, 3, 0));
        return new TreeProposal(new DraftTree(nodes), new double[] {-0.1, -0.2, -0.3, -0.4});
    }

    @Test
    public void aProposalCarriesOneFiniteScorePerNodeAndDescendsAlongTheList() {
        assertThrows(IllegalArgumentException.class, () -> new TreeProposal(branched().tree(),
                new double[] {-0.1, -0.2}));
        assertThrows(IllegalArgumentException.class, () -> new TreeProposal(branched().tree(),
                new double[] {-0.1, -0.2, -0.3, Double.NaN}));
        assertThrows(IllegalArgumentException.class, () -> new TreeProposal(branched().tree(),
                new double[] {-0.1, -0.3, -0.2, -0.4}));
        assertThrows(NullPointerException.class, () -> new TreeProposal(null, new double[] {-0.1}));
    }

    @Test
    public void aProposalCutIsItsNodePrefixAndStillAValidTree() {
        TreeProposal proposal = branched();
        assertEquals(4, proposal.tree().size());
        TreeProposal cut = proposal.cut(3);
        assertEquals(3, cut.tree().size());
        assertEquals(11, cut.tree().nodes().get(0).token());
        assertEquals(12, cut.tree().nodes().get(1).token());
        assertEquals(99, cut.tree().nodes().get(2).token());
        assertArrayEquals(Arrays.copyOf(new double[] {-0.1, -0.2, -0.3, -0.4}, 3), cut.pathScores(), 0.0);
        // the root alone, and the empty cut
        assertEquals(1, proposal.cut(1).tree().size());
        assertEquals(0, proposal.cut(0).tree().size());
        assertEquals(0, proposal.cut(-3).tree().size());
        // cutting at or above the size returns the proposal itself
        assertEquals(proposal.tree().size(), proposal.cut(9).tree().size());
    }
}
