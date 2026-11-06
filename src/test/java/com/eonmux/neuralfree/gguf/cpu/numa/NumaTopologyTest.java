package com.eonmux.neuralfree.gguf.cpu.numa;

import com.sun.jna.Platform;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link NumaTopology}: the host's sysfs layout reads as nodes of plausible CPUs, and an explicit topology maps
 * workers' CPUs, the starting node and its fallbacks.
 */
public class NumaTopologyTest {

    @Test
    public void readsTheHostLayout() {
        Assume.assumeTrue(Platform.isLinux());
        NumaTopology topology = NumaTopology.read();
        assertTrue("a Linux host reports sysfs NUMA nodes", topology.available());
        assertTrue(topology.nodeCount() >= 1);
        assertTrue(topology.cpuNumbers().length >= 1);
        assertTrue("the starting node exists", topology.currentNode() >= 0
                && topology.currentNode() < topology.nodeCount());
        assertTrue(topology.affinityCpus().length >= 1);
        assertFalse(topology.describe().isEmpty());
    }

    @Test
    public void anExplicitTopologyMapsNodesAndTheStartingNode() {
        NumaTopology topology = NumaTopology.of(new int[][] {{0, 1}, {2, 3}},
                new int[] {0, 1, 2, 3}, 3);
        assertTrue(topology.available());
        assertTrue(topology.isNuma());
        assertEquals(2, topology.nodeCount());
        assertArrayEquals(new int[] {0, 1}, topology.cpusOf(0));
        assertArrayEquals(new int[] {2, 3}, topology.cpusOf(1));
        assertArrayEquals(new int[] {0, 1, 2, 3}, topology.cpuNumbers());
        assertEquals(3, topology.currentCpu());
        assertEquals("CPU 3 is on node 1", 1, topology.currentNode());
        assertTrue(topology.describe(), topology.describe().contains("2 NUMA node(s)"));
        assertTrue(topology.describe(), topology.describe().contains("node1 CPUs 2-3"));
        assertTrue(topology.describe(), topology.describe().contains("started on node 1"));
    }

    @Test
    public void anUnknownStartingCpuFallsBackToTheFirstAffinityCpu() {
        NumaTopology topology = NumaTopology.of(new int[][] {{0, 1}, {2, 3}}, new int[] {2, 3}, -1);
        assertEquals(1, topology.currentNode());

        NumaTopology noAffinity = NumaTopology.of(new int[][] {{0, 1}, {2, 3}}, new int[0], -1);
        assertEquals(0, noAffinity.currentNode());
    }

    @Test
    public void oneNodeIsNotNuma() {
        NumaTopology topology = NumaTopology.of(new int[][] {{0, 1, 2}}, new int[] {0, 1, 2}, 1);
        assertTrue(topology.available());
        assertFalse(topology.isNuma());
        assertEquals(1, topology.nodeCount());
    }

    @Test
    public void noTopologyHasNothingToPlace() {
        NumaTopology none = NumaTopology.none();
        assertFalse(none.available());
        assertFalse(none.isNuma());
        assertEquals(0, none.nodeCount());
        assertEquals(-1, none.currentNode());
        assertEquals(-1, none.currentCpu());
        assertEquals(0, none.cpuNumbers().length);
        assertEquals("no NUMA nodes", none.describe());
        assertThrows(IllegalArgumentException.class, () -> none.cpusOf(0));
    }

    @Test
    public void refusesDuplicateAndNegativeCpus() {
        IllegalArgumentException duplicate = assertThrows(IllegalArgumentException.class,
                () -> NumaTopology.of(new int[][] {{0, 1}, {1, 2}}, new int[0], 0));
        assertTrue(duplicate.getMessage(), duplicate.getMessage().contains("CPU 1 is listed by more than one"));

        IllegalArgumentException negative = assertThrows(IllegalArgumentException.class,
                () -> NumaTopology.of(new int[][] {{-1}}, new int[0], 0));
        assertTrue(negative.getMessage(), negative.getMessage().contains("is negative"));
    }

    @Test
    public void cpusAreReturnedAsCopiesInIncreasingOrder() {
        NumaTopology topology = NumaTopology.of(new int[][] {{3, 1}, {0, 2}}, new int[] {3}, 0);
        assertArrayEquals("node CPUs are sorted", new int[] {1, 3}, topology.cpusOf(0));
        int[] cpus = topology.cpusOf(0);
        cpus[0] = 99;
        assertArrayEquals("the caller cannot change the layout", new int[] {1, 3}, topology.cpusOf(0));
        assertArrayEquals("every CPU in increasing order", new int[] {0, 1, 2, 3}, topology.cpuNumbers());
        assertThrows(IllegalArgumentException.class, () -> topology.cpusOf(2));
    }
}
