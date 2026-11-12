package com.eonmux.neuralfree.gguf.cpu.numa;

import com.eonmux.neuralfree.training.LogCapture;
import com.sun.jna.Platform;
import org.apache.logging.log4j.Level;
import org.junit.Assume;
import org.junit.Test;

import java.util.Collections;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link NumaPlacement}: the three strategies pick a worker's CPUs (round-robin over the nodes, the starting node,
 * the inherited CPU set), placement is inactive without several nodes, the pin property gives every worker one CPU
 * set, and the calling thread is really pinned when the host's affinity calls are available.
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

    @Test
    public void thePinPropertyNamesOneCpuSetForEveryWorker() {
        Properties p = new Properties();
        assertNull("absent", NumaPlacement.pinCpus(p));
        p.setProperty(NumaPlacement.PIN_PROPERTY, " OFF ");
        assertNull("off", NumaPlacement.pinCpus(p));
        assertFalse(NumaPlacement.fromProperties(p).active());
        p.setProperty(NumaPlacement.PIN_PROPERTY, "16-19,3");
        assertArrayEquals(new int[] {3, 16, 17, 18, 19}, NumaPlacement.pinCpus(p));
        NumaPlacement placement = NumaPlacement.fromProperties(p);
        assertTrue("a pinned set is an active placement, so the caller only waits", placement.active());
        assertEquals(NumaStrategy.DISABLED, placement.strategy());
        assertArrayEquals(new int[] {3, 16, 17, 18, 19}, placement.pinnedCpus());
        for (int worker : new int[] {0, 1, 7, 12}) {
            assertArrayEquals("worker " + worker, new int[] {3, 16, 17, 18, 19}, placement.cpusForWorker(worker));
        }
        p.setProperty(NumaPlacement.PIN_PROPERTY, "policy");
        int[] policy = CpuThreadPolicy.host().cpus(CpuThreadPolicy.configuredPolicy());
        if (policy.length == 0) {
            assertNull("no topology, no set", NumaPlacement.pinCpus(p));
        } else {
            assertArrayEquals("the configured thread policy's CPUs", policy, NumaPlacement.pinCpus(p));
        }
        assertNull(NumaPlacement.disabled().pinnedCpus());
    }

    @Test
    public void aMalformedOrEmptyPinSetIsRefusedByName() {
        Properties p = new Properties();
        for (String bad : new String[] {"7-3", "sideways", "1,,2", "-1"}) {
            p.setProperty(NumaPlacement.PIN_PROPERTY, bad);
            IllegalArgumentException e = assertThrows(bad, IllegalArgumentException.class,
                    () -> NumaPlacement.fromProperties(p));
            assertTrue(e.getMessage(), e.getMessage().contains(NumaPlacement.PIN_PROPERTY));
        }
        assertThrows(IllegalArgumentException.class, () -> NumaPlacement.pinned(new int[0]));
        assertThrows(IllegalArgumentException.class, () -> NumaPlacement.pinned(new int[] {-2, 1}));
    }

    @Test
    public void anInactiveNumaStrategyLeavesThePinSetInCharge() {
        Assume.assumeFalse("a single-node host leaves the NUMA strategy inactive", NumaTopology.host().isNuma());
        Properties p = new Properties();
        p.setProperty(NumaStrategy.PROPERTY, "distribute");
        p.setProperty(NumaPlacement.PIN_PROPERTY, "0");
        NumaPlacement placement = NumaPlacement.fromProperties(p);
        assertTrue(placement.active());
        assertArrayEquals(new int[] {0}, placement.cpusForWorker(3));
    }

    /**
     * An active NUMA placement places the pool itself, so a pin set beside it has no effect, and the placement says
     * so once, when it is resolved; without a pin set it says nothing about one.
     */
    @Test
    public void aPinSetUnderAnActiveNumaPlacementIsLoggedAsIgnored() {
        Properties p = new Properties();
        p.setProperty(NumaStrategy.PROPERTY, "distribute");
        p.setProperty(NumaPlacement.PIN_PROPERTY, "16-23");
        try (LogCapture log = new LogCapture(NumaPlacement.class)) {
            NumaPlacement placement = NumaPlacement.fromProperties(p, NumaPlacementTest::twoNodes);
            assertEquals(NumaStrategy.DISTRIBUTE, placement.strategy());
            assertNull("the NUMA placement wins", placement.pinnedCpus());
            assertArrayEquals(new int[] {2, 3}, placement.cpusForWorker(1));
            assertEquals(Collections.singletonList(NumaPlacement.PIN_PROPERTY + "=16-23 has no effect: "
                    + NumaStrategy.PROPERTY + "=distribute places the CPU pool on its NUMA nodes"),
                    log.messages(Level.WARN, NumaPlacement.PIN_PROPERTY));

            p.remove(NumaPlacement.PIN_PROPERTY);
            NumaPlacement.fromProperties(p, NumaPlacementTest::twoNodes);
            assertEquals("no pin set, nothing to report", 1,
                    log.messages(Level.WARN, NumaPlacement.PIN_PROPERTY).size());
        }
    }

    @Test
    public void pinsTheCallingThreadToThePinnedSet() throws Exception {
        Assume.assumeTrue(Platform.isLinux());
        Assume.assumeTrue("the host exposes sched_setaffinity", CpuAffinity.available());
        int[] allowed = CpuAffinity.currentThreadCpus();
        Assume.assumeTrue("the test thread may run on at least two CPUs", allowed.length >= 2);
        int[] set = {allowed[0], allowed[allowed.length - 1]};
        NumaPlacement placement = NumaPlacement.pinned(set);
        int[][] seen = new int[1][];
        Thread worker = new Thread(() -> {
            placement.pinCurrentThread(5);
            seen[0] = CpuAffinity.currentThreadCpus();
        });
        worker.start();
        worker.join();
        assertArrayEquals("the worker runs on the pinned set", set, seen[0]);
        assertArrayEquals("the caller keeps its own affinity", allowed, CpuAffinity.currentThreadCpus());
    }
}
