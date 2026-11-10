package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The device timing report and its mode: {@code -Dneuralfree.gpu.profile} reads as off, per dispatch or per step
 * and refuses anything else, graph launches report under their own key, and the graph total sums those entries
 * alone.
 */
public class KernelTimingTest {

    @Test
    public void theProfilePropertyNamesAMode() {
        assertEquals(KernelTiming.Mode.OFF, KernelTiming.Mode.parse(null));
        assertEquals(KernelTiming.Mode.OFF, KernelTiming.Mode.parse(""));
        assertEquals(KernelTiming.Mode.OFF, KernelTiming.Mode.parse("false"));
        assertEquals(KernelTiming.Mode.DISPATCH, KernelTiming.Mode.parse("true"));
        assertEquals(KernelTiming.Mode.DISPATCH, KernelTiming.Mode.parse("TRUE"));
        assertEquals(KernelTiming.Mode.STEP, KernelTiming.Mode.parse("step"));
        assertEquals(KernelTiming.Mode.STEP, KernelTiming.Mode.parse(" Step "));
    }

    @Test
    public void anUnknownProfileValueIsRefusedByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> KernelTiming.Mode.parse("yes"));
        assertTrue(e.getMessage(), e.getMessage().contains(KernelTiming.PROFILE_PROPERTY));
        assertTrue(e.getMessage(), e.getMessage().contains("'yes'"));
    }

    @Test
    public void theConfiguredModeReadsTheProperty() {
        String prior = System.getProperty(KernelTiming.PROFILE_PROPERTY);
        try {
            System.setProperty(KernelTiming.PROFILE_PROPERTY, "step");
            assertEquals(KernelTiming.Mode.STEP, KernelTiming.Mode.configured());
            System.clearProperty(KernelTiming.PROFILE_PROPERTY);
            assertEquals(KernelTiming.Mode.OFF, KernelTiming.Mode.configured());
        } finally {
            if (prior == null) {
                System.clearProperty(KernelTiming.PROFILE_PROPERTY);
            } else {
                System.setProperty(KernelTiming.PROFILE_PROPERTY, prior);
            }
        }
    }

    @Test
    public void graphLaunchesSumApartFromTheOtherCommands() {
        KernelTiming timing = new KernelTiming();
        String graph = KernelTiming.graphKey(421);
        assertEquals("graph[421 dispatches]", graph);
        assertTrue(KernelTiming.isGraphKey(graph));
        assertFalse(KernelTiming.isGraphKey("memcpy HtoD"));
        assertFalse(KernelTiming.isGraphKey(KernelTiming.SUBMISSION_KEY));
        timing.add(graph, 1_000);
        timing.add(graph, 3_000);
        timing.add(KernelTiming.graphKey(7), 500);
        timing.add("memcpy HtoD", 200);
        assertEquals(4_500, timing.graphNanos());
        assertEquals(4_700, timing.totalNanos());
        assertEquals(4, timing.totalCount());
        assertEquals(graph, timing.snapshot().get(0).name());
        assertEquals(2_000, timing.snapshot().get(0).meanNanos());
        timing.reset();
        assertEquals(0, timing.graphNanos());
        assertTrue(timing.isEmpty());
    }
}
