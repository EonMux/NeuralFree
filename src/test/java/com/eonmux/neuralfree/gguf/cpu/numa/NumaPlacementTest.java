package com.eonmux.neuralfree.gguf.cpu.numa;

import com.sun.jna.Platform;
import org.junit.Assume;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link NumaPlacement}: the three strategies pick a worker's CPUs (round-robin over the nodes, the starting node,
 * the inherited CPU set), placement is inactive without several nodes, and the calling thread is really pinned when
 * the host's affinity calls are available.
 */
public class NumaPlacementTest {

    private static NumaTopology twoNodes() {
        return NumaTopology.of(new int[][] {{0, 1}, {2, 3}}, new int[] {0, 1, 2, 3}, 3);
    }

    @Test
    public void disabledHasNoTopologyAndPlacesNothing() {
        NumaPlacement placement = NumaPlacement.disabled();
        assertEquals(NumaStrategy.DISABLED, placement.strategy());
        assertNull(placement.topology());
        assertFalse(placement.active());
        assertNull(placement.cpusForWorker(0));
        placement.pinCurrentThread(0);

        NumaPlacement fromProperties = NumaPlacement.fromProperties(new Properties());
        assertFalse(fromProperties.active());
        assertNull(fromProperties.cpusForWorker(3));
    }

    @Test
    public void distributeSpreadsWorkersOverTheNodes() {
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.DISTRIBUTE, twoNodes());
        assertTrue(placement.active());
        assertArrayEquals(new int[] {0, 1}, placement.cpusForWorker(0));
        assertArrayEquals(new int[] {2, 3}, placement.cpusForWorker(1));
        assertArrayEquals("the next round starts at node 0", new int[] {0, 1}, placement.cpusForWorker(2));
        assertArrayEquals(new int[] {2, 3}, placement.cpusForWorker(3));
    }

    @Test
    public void isolateKeepsEveryWorkerOnTheStartingNode() {
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.ISOLATE, twoNodes());
        assertArrayEquals(new int[] {2, 3}, placement.cpusForWorker(0));
        assertArrayEquals(new int[] {2, 3}, placement.cpusForWorker(7));
    }

    @Test
    public void numactlKeepsTheInheritedCpuSet() {
        NumaTopology topology = NumaTopology.of(new int[][] {{0, 1}, {2, 3}}, new int[] {1, 2}, 1);
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.NUMACTL, topology);
        assertArrayEquals(new int[] {1, 2}, placement.cpusForWorker(0));
        assertArrayEquals(new int[] {1, 2}, placement.cpusForWorker(5));
    }

    @Test
    public void oneNodeLeavesPlacementInactive() {
        NumaTopology one = NumaTopology.of(new int[][] {{0, 1, 2, 3}}, new int[] {0, 1, 2, 3}, 0);
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.DISTRIBUTE, one);
        assertFalse(placement.active());
        assertNull(placement.cpusForWorker(0));
    }

    @Test
    public void fromPropertiesParsesTheStrategyAndRefusesUnknownNames() {
        Properties p = new Properties();
        p.setProperty(NumaStrategy.PROPERTY, "isolate");
        NumaPlacement placement = NumaPlacement.fromProperties(p);
        assertEquals(NumaStrategy.ISOLATE, placement.strategy());
        p.setProperty(NumaStrategy.PROPERTY, "sideways");
        assertThrows(IllegalArgumentException.class, () -> NumaPlacement.fromProperties(p));
    }

    @Test
    public void activeInFollowsTheStrategyAndTheHostLayout() {
        Properties p = new Properties();
        assertFalse("no strategy places nothing", NumaPlacement.activeIn(p));
        p.setProperty(NumaStrategy.PROPERTY, "distribute");
        assertEquals("a strategy places threads exactly when the host has several nodes",
                NumaTopology.host().isNuma(), NumaPlacement.activeIn(p));
        p.setProperty(NumaStrategy.PROPERTY, "off");
        assertFalse(NumaPlacement.activeIn(p));
    }

    @Test
    public void pinsTheCallingThreadToItsNode() {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue("the host exposes sched_setaffinity", CpuAffinity.available());
        int[] before = CpuAffinity.currentThreadCpus();
        Assume.assumeTrue("the test thread may run on at least two CPUs", before.length >= 2);
        int first = before[0];
        int second = before[before.length - 1];
        NumaTopology topology = NumaTopology.of(new int[][] {{first}, {second}}, before, first);
        NumaPlacement placement = NumaPlacement.of(NumaStrategy.DISTRIBUTE, topology);
        assertTrue(placement.active());
        try {
            placement.pinCurrentThread(1);
            assertArrayEquals("worker 1 runs on node 1's CPU", new int[] {second},
                    CpuAffinity.currentThreadCpus());
            placement.pinCurrentThread(0);
            assertArrayEquals("worker 0 runs on node 0's CPU", new int[] {first},
                    CpuAffinity.currentThreadCpus());
        } finally {
            CpuAffinity.setCurrentThreadCpus(before);
        }
        assertArrayEquals("the original affinity is back", before, CpuAffinity.currentThreadCpus());
    }
}
