package com.eonmux.neuralfree.gguf.gpu.parallel;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The expert-placement policies against hand-worked {@code DefaultEplbPolicy} results: the weighted packing heaviest
 * first into the lightest pack, replication of the largest load per replica, the hierarchical rearrangement's
 * group-to-node packing over one node, the slot preservation of experts that stay on a rank, and the per-rank load
 * figures. No GPU is needed.
 */
public class EplbPolicyTest {

    @Test
    public void thePackingPutsTheHeaviestItemsFirstIntoTheLightestPack() {
        // vLLM doc example: weight {5,1,1,1} into two packs gives {0,1,1,0}.
        EplbPolicy.Packing packed = EplbPolicy.balancedPacking(new double[][]{{5, 1, 1, 1}}, 2);
        assertArrayEquals(new int[]{0, 1, 1, 0}, packed.packIndex()[0]);
        assertArrayEquals(new int[]{0, 0, 1, 1}, packed.rankInPack()[0]);
        // Equal weights take their id order: the first goes to pack 0.
        assertArrayEquals(new int[]{0, 1, 0, 1},
                EplbPolicy.balancedPacking(new double[][]{{5, 5, 5, 5}}, 2).packIndex()[0]);
        // Two layers pack independently.
        EplbPolicy.Packing two = EplbPolicy.balancedPacking(new double[][]{{7, 1, 1, 3, 2, 1}, {1, 9, 1, 1, 1, 1}}, 2);
        assertArrayEquals(new int[]{0, 1, 0, 1, 1, 0}, two.packIndex()[0]);
        assertArrayEquals(new int[]{0, 2, 1, 0, 1, 2}, two.rankInPack()[0]);
        assertArrayEquals(new int[]{1, 0, 1, 1, 0, 0}, two.packIndex()[1]);
    }

    @Test
    public void oneItemPerPackKeepsItsOrder() {
        // num_packs == items packs the items as they are.
        EplbPolicy.Packing packed = EplbPolicy.balancedPacking(new double[][]{{3, 1, 2}}, 3);
        assertArrayEquals(new int[]{0, 1, 2}, packed.packIndex()[0]);
        assertArrayEquals(new int[]{0, 0, 0}, packed.rankInPack()[0]);
    }

    @Test
    public void thePackingRefusesBadShapesAndWeights() {
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balancedPacking(new double[][]{{1, 2, 3}}, 2));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balancedPacking(new double[][]{{-1, 2}}, 1));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.balancedPacking(new double[][]{{Double.NaN, 2}}, 1));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.balancedPacking(new double[][]{{1, 2}, {1}}, 1));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balancedPacking(new double[0][], 1));
    }

    @Test
    public void replicationAddsSlotsToTheLargestLoadPerReplica() {
        // vLLM doc example: {5,1,1} into five slots gives {0,1,2,0,0}.
        assertArrayEquals(new int[][]{{0, 1, 2, 0, 0}},
                EplbPolicy.replicateExperts(new double[][]{{5, 1, 1}}, 5));
        // A tie of load per replica keeps the lower expert (8/2 == 4/1): expert 0 gains twice.
        assertArrayEquals(new int[][]{{0, 1, 2, 3, 0, 0}},
                EplbPolicy.replicateExperts(new double[][]{{8, 4, 2, 1}}, 6));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.replicateExperts(new double[][]{{1, 2}}, 1));
    }

    @Test
    public void theHierarchicalPolicyPacksTheGroupsNodeLocallyThenSpreadsTheSlots() {
        // Groups {7,1,1} and {3,2,1} over one node: the local order is the logical order, the six slots pack into two
        // ranks as heaviest first, which spreads them as {0,2,5} and {3,4,1}.
        assertArrayEquals(new int[][]{{0, 2, 5, 3, 4, 1}},
                EplbPolicy.rebalanceHierarchical(new double[][]{{7, 1, 1, 3, 2, 1}}, 6, 2, 1, 2));
        // The global policy is the hierarchical one over one group and one node.
        assertArrayEquals(new int[][]{{0, 2, 5, 3, 4, 1}},
                EplbPolicy.rebalanceHierarchical(new double[][]{{7, 1, 1, 3, 2, 1}}, 6, 1, 1, 2));
    }

    @Test
    public void theHierarchicalPolicyRefusesShapesItCannotSatisfy() {
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rebalanceHierarchical(new double[][]{{1, 1, 1}}, 4, 2, 1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rebalanceHierarchical(new double[][]{{1, 1, 1, 1}}, 4, 3, 1, 2));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rebalanceHierarchical(new double[][]{{1, 1, 1, 1}}, 4, 2, 3, 2));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rebalanceHierarchical(new double[][]{{1, 1, 1, 1}}, 4, 2, 2, 3));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rebalanceHierarchical(new double[][]{{1, 1, 1, 1}}, 5, 2, 1, 2));
    }

    @Test
    public void theSlotPreservationKeepsTheStayingExpertsSlots() {
        // Rank 0 keeps experts 0 and 2 in their slots and takes 5 into the freed one; rank 1 keeps 3 and 4 and takes
        // 1; the mapping stays a permutation of the experts.
        assertArrayEquals(new int[][]{{0, 5, 2, 3, 4, 1}},
                EplbPolicy.preserveIntraRankSlots(new int[][]{{0, 2, 5, 3, 4, 1}}, 2,
                        new int[][]{{0, 1, 2, 3, 4, 5}}));
        // Without an old mapping the new one is kept as computed.
        assertArrayEquals(new int[][]{{0, 2, 5, 3, 4, 1}},
                EplbPolicy.preserveIntraRankSlots(new int[][]{{0, 2, 5, 3, 4, 1}}, 2, null));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.preserveIntraRankSlots(new int[][]{{0, 1}}, 2, new int[][]{{0, 1, 2}}));
    }

    @Test
    public void theEntryPointsHidesThePoliciesBehindTheirDivisibilityRule() {
        // num_groups % num_nodes == 0 keeps the hierarchical packing; the preservation keeps the staying experts'
        // slots (0, 2, 3, 4) and fills the freed ones with 5 and 1.
        assertArrayEquals(new int[][]{{0, 5, 2, 3, 4, 1}},
                EplbPolicy.rebalance(new double[][]{{7, 1, 1, 3, 2, 1}}, 6, 2, 1, 2,
                        new int[][]{{0, 1, 2, 3, 4, 5}}));
        // A group count that does not divide falls back to the global policy; without an old mapping the layout is
        // kept as computed.
        assertArrayEquals(new int[][]{{0, 2, 5, 3, 4, 1}},
                EplbPolicy.rebalance(new double[][]{{7, 1, 1, 3, 2, 1}}, 6, 3, 2, 2, null));
    }

    @Test
    public void theLayoutMapsLogicalExpertsToTheirSlotsAndCountsTheReplicas() {
        EplbPolicy.Layout layout = EplbPolicy.layout(new int[][]{{0, 0, 2, 1, 1, 3}}, 4);
        assertArrayEquals(new int[]{0, 0, 2, 1, 1, 3}, layout.phyToLog()[0]);
        assertArrayEquals(new int[][]{{0, 1}, {3, 4}, {2, -1}, {5, -1}}, layout.logicalToPhysical(0));
        assertArrayEquals(new int[]{2, 2, 1, 1}, layout.replicaCount()[0]);
        assertEquals(6, layout.physical());
        assertEquals(4, layout.logical());
        assertEquals(2, EplbPolicy.maxReplicas(new int[][]{{0, 0, 2, 1, 1, 3}}, 4));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.layout(new int[][]{{0, 4}}, 4));
    }

    @Test
    public void theRankLoadsSplitEachReplicasShareOverItsRanks() {
        // A slot holding a replica of a twice-replicated expert counts half its load.
        assertArrayEquals(new double[]{8, 0},
                EplbPolicy.rankLoads(new double[]{8, 0, 0, 0}, new int[]{0, 0, 1, 1}, 2), 1e-9);
        // The balanced {0,2} and {1,3} packing of loads {100,100,1,1} puts 101 on each rank.
        assertArrayEquals(new double[]{101, 101},
                EplbPolicy.rankLoads(new double[]{100, 100, 1, 1}, new int[]{0, 2, 1, 3}, 2), 1e-9);
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rankLoads(new double[]{1, 1, 1}, new int[]{0, 1, 2}, 2));
        assertThrows(IllegalArgumentException.class,
                () -> EplbPolicy.rankLoads(new double[]{1, 1}, new int[]{0, 2}, 2));
    }

    @Test
    public void theBalanceLineReportsTheReferenceFigures() {
        assertEquals("avg_tokens=101.00, max_tokens=101, balancedness=1.0000",
                EplbPolicy.balanceLine(new double[][]{{101, 101}}));
        assertEquals("avg_tokens=4.00, max_tokens=8, balancedness=0.5000",
                EplbPolicy.balanceLine(new double[][]{{8, 0}}));
        // The layers sum their averages and maxima; an idle split counts as fully balanced.
        assertEquals("avg_tokens=6.00, max_tokens=10, balancedness=0.6000",
                EplbPolicy.balanceLine(new double[][]{{8, 2}, {0, 2}}));
        assertEquals("avg_tokens=0.00, max_tokens=0, balancedness=1.0000",
                EplbPolicy.balanceLine(new double[][]{{0, 0}}));
        assertTrue(EplbPolicy.balanceLine(new double[][]{{1, 2}}).endsWith("balancedness=0.7500"));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balanceLine(new double[0][]));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balanceLine(new double[][]{{1}, {1, 2}}));
        assertThrows(IllegalArgumentException.class, () -> EplbPolicy.balanceLine(new double[][]{{-1, 0}}));
    }
}
