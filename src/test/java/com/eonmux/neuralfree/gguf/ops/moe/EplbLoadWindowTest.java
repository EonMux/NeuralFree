package com.eonmux.neuralfree.gguf.ops.moe;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@link EplbLoadWindow} of the runtime expert rearrangement: routed rows are counted only while the window
 * records, draining returns the counted rows hottest first and empties the window, and a window below one step is
 * refused.
 */
public class EplbLoadWindowTest {

    @Test
    public void theWindowCountsOnlyWhileItRecords() {
        EplbLoadWindow window = new EplbLoadWindow(2);
        window.note(0, new int[]{1, 1}, 2);
        assertTrue("a fresh window counts nothing", window.isEmpty());
        window.record(true);
        assertTrue(window.recording());
        window.note(3, new int[]{4, 4, 5}, 3);
        assertFalse(window.isEmpty());
        window.record(false);
        assertFalse(window.recording());
        window.note(3, new int[]{9, 9}, 2);
        List<RankedExpert> drained = window.drain();
        assertEquals(Arrays.asList(new RankedExpert(3, 4, 2), new RankedExpert(3, 5, 1)), drained);
    }

    @Test
    public void drainingClearsTheWindow() {
        EplbLoadWindow window = new EplbLoadWindow(1);
        window.record(true);
        window.note(2, new int[]{7}, 1);
        window.drain();
        assertTrue("the next boundary reads an empty window", window.isEmpty());
        window.note(2, new int[]{8}, 1);
        assertEquals(Arrays.asList(new RankedExpert(2, 8, 1)), window.drain());
    }

    @Test
    public void aWindowBelowOneStepIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> new EplbLoadWindow(0));
        assertTrue(e.getMessage(), e.getMessage().contains("0"));
    }

    @Test
    public void theWindowSizeIsKept() {
        assertEquals(5, new EplbLoadWindow(5).windowSize());
    }
}
