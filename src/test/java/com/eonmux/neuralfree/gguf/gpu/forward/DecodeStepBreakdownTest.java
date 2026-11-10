package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.KernelTiming;
import org.junit.Assume;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The decode-step breakdown's bookkeeping: each lap lands in its phase, a step counts under the trunk path it
 * reported, the gap before a call counts only when the call continues the previous position, an interrupted call
 * leaves nothing behind, the device time of a step is the timing report's growth over the call, the bytes the
 * decode thread allocates inside a call count toward its step, and the property yields the disabled breakdown
 * unless it is on.
 */
public class DecodeStepBreakdownTest {

    private static void sleepMillis(long ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
    }

    private static void step(DecodeStepBreakdown b, int pos, DecodeStepBreakdown.TrunkPath path) {
        b.begin(pos);
        b.lap(DecodeStepBreakdown.Phase.CALL);
        b.lap(DecodeStepBreakdown.Phase.INPUTS);
        b.trunkPath(path);
        b.end();
    }

    @Test
    public void aLapLandsInItsPhaseAndTheStepUnderItsPath() {
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(null);
        b.begin(0);
        sleepMillis(3);
        b.lap(DecodeStepBreakdown.Phase.WAIT);
        b.trunkPath(DecodeStepBreakdown.TrunkPath.REPLAYED);
        b.end();
        DecodeStepBreakdown.Report r = b.report();
        assertEquals(1, r.steps());
        assertEquals(1, r.steps(DecodeStepBreakdown.TrunkPath.REPLAYED));
        assertTrue("the slept lap is in the wait phase",
                r.phaseNanos(DecodeStepBreakdown.TrunkPath.REPLAYED, DecodeStepBreakdown.Phase.WAIT) >= 3_000_000L);
        assertTrue("nothing else took the sleep",
                r.phaseNanos(DecodeStepBreakdown.TrunkPath.REPLAYED, DecodeStepBreakdown.Phase.CALL) < 3_000_000L);
        assertEquals(r.meanNanos(DecodeStepBreakdown.TrunkPath.REPLAYED, DecodeStepBreakdown.Phase.WAIT),
                r.medianNanos(DecodeStepBreakdown.TrunkPath.REPLAYED, DecodeStepBreakdown.Phase.WAIT), 0.0);
        assertTrue("the first call follows no call", Double.isNaN(r.meanPeriodNanos(null)));
    }

    @Test
    public void theGapCountsOnlyBeforeTheNextPosition() {
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(null);
        step(b, 4, DecodeStepBreakdown.TrunkPath.DIRECT);
        sleepMillis(2);
        step(b, 5, DecodeStepBreakdown.TrunkPath.DIRECT);
        step(b, 9, DecodeStepBreakdown.TrunkPath.DIRECT);
        DecodeStepBreakdown.Report r = b.report();
        assertEquals(3, r.steps(DecodeStepBreakdown.TrunkPath.DIRECT));
        double between = r.meanNanos(DecodeStepBreakdown.TrunkPath.DIRECT, DecodeStepBreakdown.Phase.BETWEEN);
        assertTrue("one gap, the slept one: " + between, between >= 2_000_000.0);
        assertTrue("the period holds the gap", r.meanPeriodNanos(DecodeStepBreakdown.TrunkPath.DIRECT) >= between);
        assertEquals(r.meanPeriodNanos(DecodeStepBreakdown.TrunkPath.DIRECT),
                r.medianPeriodNanos(DecodeStepBreakdown.TrunkPath.DIRECT), 0.0);
    }

    @Test
    public void anInterruptedCallIsDroppedAndBreaksTheGap() {
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(null);
        step(b, 0, DecodeStepBreakdown.TrunkPath.REPLAYED);
        b.begin(1);
        b.lap(DecodeStepBreakdown.Phase.INPUTS);
        b.interrupt();
        b.end();
        step(b, 2, DecodeStepBreakdown.TrunkPath.REPLAYED);
        b.interrupt();
        step(b, 3, DecodeStepBreakdown.TrunkPath.REPLAYED);
        DecodeStepBreakdown.Report r = b.report();
        assertEquals("the interrupted call is not a step", 3, r.steps());
        assertTrue("no step followed a contiguous one", Double.isNaN(r.meanNanos(null,
                DecodeStepBreakdown.Phase.BETWEEN)));
    }

    @Test
    public void theDeviceTimeIsTheReportsGrowthOverTheCall() {
        KernelTiming timing = new KernelTiming();
        timing.add("before", 9_000);
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(timing);
        for (int pos = 0; pos < 3; pos++) {
            b.begin(pos);
            timing.add(KernelTiming.graphKey(10), 1_000);
            timing.add("head", 500);
            b.trunkPath(DecodeStepBreakdown.TrunkPath.REPLAYED);
            b.end();
        }
        DecodeStepBreakdown.Report r = b.report();
        assertTrue(r.timed());
        assertEquals(1_500.0, r.deviceNanos(DecodeStepBreakdown.TrunkPath.REPLAYED), 0.0);
        assertEquals(1_000.0, r.graphNanos(null), 0.0);
        assertEquals(1_500.0, r.medianDeviceNanos(DecodeStepBreakdown.TrunkPath.REPLAYED), 0.0);
        List<String> lines = r.format();
        assertTrue(lines.get(0), lines.get(0).startsWith("3 decode steps (replayed 3,"));
        assertTrue(String.join("\n", lines), lines.stream().anyMatch(l -> l.startsWith("device busy")));
        assertTrue(String.join("\n", lines), lines.stream().anyMatch(l -> l.startsWith("device idle")));
    }

    @Test
    public void theIdleTimeSubtractsTheDeviceTimeOfTheSameContiguousSteps() {
        KernelTiming timing = new KernelTiming();
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(timing);
        long[] device = {1_000_000L, 100_000L, 100_000L};
        for (int pos = 0; pos < device.length; pos++) {
            b.begin(pos);
            timing.add("step", device[pos]);
            sleepMillis(2);
            b.lap(DecodeStepBreakdown.Phase.WAIT);
            b.end();
        }
        DecodeStepBreakdown.Report r = b.report();
        DecodeStepBreakdown.TrunkPath direct = DecodeStepBreakdown.TrunkPath.DIRECT;
        assertEquals("the first step follows no call", 2, r.contiguousSteps(direct));
        assertEquals("the device time of the two contiguous steps alone", r.meanPeriodNanos(direct) - 100_000.0,
                r.meanIdleNanos(direct), 0.0);
        assertEquals("two contiguous steps: their mean is their median", r.medianIdleNanos(direct),
                r.meanIdleNanos(direct), 0.0);
        assertEquals(r.meanIdleNanos(direct), r.meanIdleNanos(null), 0.0);
        String idle = r.format().stream().filter(l -> l.startsWith("device idle")).findFirst().orElse("");
        assertTrue(idle, idle.contains(String.format(java.util.Locale.ROOT, "%.1f", r.meanIdleNanos(direct) / 1e3)));
    }

    @Test
    public void aGraphLaunchTheQueueDoesNotTimeLeavesItsStepOutOfTheDeviceRows() {
        KernelTiming timing = new KernelTiming();
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(timing, false);
        DecodeStepBreakdown.TrunkPath[] paths = {DecodeStepBreakdown.TrunkPath.DIRECT,
            DecodeStepBreakdown.TrunkPath.DIRECT, DecodeStepBreakdown.TrunkPath.CAPTURED,
            DecodeStepBreakdown.TrunkPath.REPLAYED, DecodeStepBreakdown.TrunkPath.REPLAYED,
            DecodeStepBreakdown.TrunkPath.REPLAYED};
        for (int pos = 0; pos < paths.length; pos++) {
            b.begin(pos);
            // the untimed launch adds only the commands around the trunk, as an OpenCL command buffer leaves it
            timing.add("head", paths[pos] == DecodeStepBreakdown.TrunkPath.DIRECT ? 900_000L : 50_000L);
            b.trunkPath(paths[pos]);
            b.end();
        }
        DecodeStepBreakdown.Report r = b.report();
        DecodeStepBreakdown.TrunkPath replayed = DecodeStepBreakdown.TrunkPath.REPLAYED;
        assertEquals(replayed, r.dominantPath());
        assertEquals(4, r.untimedSteps(null));
        assertEquals(3, r.untimedSteps(replayed));
        assertTrue("no device time for the replayed steps", Double.isNaN(r.deviceNanos(replayed)));
        assertTrue("and no idle time", Double.isNaN(r.meanIdleNanos(replayed)));
        assertEquals("the direct steps keep theirs", 900_000.0, r.deviceNanos(null), 0.0);
        List<String> lines = r.format();
        assertTrue(String.join("\n", lines), lines.stream().anyMatch(l -> l.equals("(the device rows leave out 4"
                + " steps whose trunk ran as a graph the queue does not time)")));
        DecodeStepBreakdown timed = DecodeStepBreakdown.recording(timing, true);
        timed.begin(0);
        timing.add("graph", 800_000L);
        timed.trunkPath(replayed);
        timed.end();
        assertEquals(800_000.0, timed.report().deviceNanos(replayed), 0.0);
        assertEquals(0, timed.report().untimedSteps(null));
    }

    @Test
    public void theReportFollowsThePathMostStepsTook() {
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(null);
        step(b, 0, DecodeStepBreakdown.TrunkPath.CAPTURED);
        for (int pos = 1; pos < 4; pos++) {
            step(b, pos, DecodeStepBreakdown.TrunkPath.DIRECT);
        }
        DecodeStepBreakdown.Report r = b.report();
        assertEquals(DecodeStepBreakdown.TrunkPath.DIRECT, r.dominantPath());
        List<String> lines = r.format();
        assertTrue(lines.get(1), lines.get(1).contains("direct mean"));
        assertFalse("no device rows without timing", lines.stream().anyMatch(l -> l.startsWith("device busy")));
        b.reset();
        assertEquals(0, b.report().steps());
    }

    /** Where a step's allocation lands, so the escape analysis cannot drop it. */
    private static volatile byte[] sink;

    @Test
    public void theBytesAllocatedInsideACallCountTowardItsStep() {
        DecodeStepBreakdown b = DecodeStepBreakdown.recording(null);
        Assume.assumeTrue("the JVM counts allocations per thread", b.report().allocationsCounted());
        int size = 4 << 20;
        for (int pos = 0; pos < 3; pos++) {
            b.begin(pos);
            sink = new byte[size];
            b.lap(DecodeStepBreakdown.Phase.INPUTS);
            b.end();
            sink = new byte[8 * size];
        }
        assertNotNull(sink);
        DecodeStepBreakdown.Report r = b.report();
        double mean = r.meanAllocatedBytes(DecodeStepBreakdown.TrunkPath.DIRECT);
        assertTrue("each step allocated its array: " + mean, mean >= size);
        assertTrue("the allocation between calls is not counted: " + mean, mean < 2.0 * size);
        assertTrue(r.medianAllocatedBytes(DecodeStepBreakdown.TrunkPath.DIRECT) >= size);
        assertTrue(String.join("\n", r.format()),
                r.format().stream().anyMatch(l -> l.startsWith("allocated bytes per step")));
        b.reset();
        assertTrue(Double.isNaN(b.report().meanAllocatedBytes(null)));
    }

    @Test
    public void thePropertyChoosesTheBreakdown() {
        String prior = System.getProperty(DecodeStepBreakdown.PROPERTY);
        try {
            System.clearProperty(DecodeStepBreakdown.PROPERTY);
            assertSame(DecodeStepBreakdown.DISABLED, DecodeStepBreakdown.configured(null, false));
            System.setProperty(DecodeStepBreakdown.PROPERTY, "true");
            assertTrue(DecodeStepBreakdown.configured(null, false).enabled());
            System.setProperty(DecodeStepBreakdown.PROPERTY, "on");
            assertThrows(IllegalArgumentException.class, () -> DecodeStepBreakdown.configured(null, false));
        } finally {
            if (prior == null) {
                System.clearProperty(DecodeStepBreakdown.PROPERTY);
            } else {
                System.setProperty(DecodeStepBreakdown.PROPERTY, prior);
            }
        }
    }

    @Test
    public void theDisabledBreakdownRecordsNothing() {
        DecodeStepBreakdown b = DecodeStepBreakdown.DISABLED;
        assertFalse(b.enabled());
        step(b, 0, DecodeStepBreakdown.TrunkPath.REPLAYED);
        b.reset();
        assertEquals(0, b.report().steps());
    }
}
