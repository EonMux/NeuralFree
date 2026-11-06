package com.eonmux.neuralfree.gguf.gpu.pipeline;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

/**
 * The back-to-front fill of the layer split's automatic fit ({@link BackFill}): the last device takes the largest
 * run of trailing layers it holds, then the one before it, the leftover leading layers stay on the host, every device
 * keeps at least one layer, and a run whose routed experts can move to the host is preferred over a shorter run.
 */
public class BackFillTest {

    @Test
    public void theLastDeviceFillsFirstAndTheLeadingLayersStayOnTheHost() {
        BackFill.Layout layout = BackFill.fill(2, 8, (device, start, end) -> end - start <= (device == 1 ? 2 : 5)
                ? 0 : -1);
        assertEquals("the last device takes the two trailing layers it holds", 2, layout.layers(1));
        assertEquals("the first device takes the next five", 5, layout.layers(0));
        assertEquals("the leftover layer runs on the host", 1, layout.firstGpuLayer());
        assertEquals(0, layout.experts(0));
        assertEquals(0, layout.experts(1));
    }

    @Test
    public void everyEarlierDeviceKeepsOneLayerWhenTheLastDeviceWouldTakeThemAll() {
        BackFill.Layout layout = BackFill.fill(3, 5, (device, start, end) -> 0);
        assertEquals(1, layout.layers(0));
        assertEquals(1, layout.layers(1));
        assertEquals(3, layout.layers(2));
        assertEquals(0, layout.firstGpuLayer());
    }

    @Test
    public void aRunWhoseExpertsMoveToTheHostIsPreferredOverAShorterRun() {
        BackFill.Layout layout = BackFill.fill(1, 8, (device, start, end) -> {
            int count = end - start;
            if (count <= 2) {
                return 0;
            }
            return count <= 4 ? count : -1;
        });
        assertEquals("four layers fit, with all four experts in RAM, rather than two resident", 4, layout.layers(0));
        assertEquals(4, layout.experts(0));
        assertEquals(4, layout.firstGpuLayer());
    }

    @Test
    public void noLayoutWhenADeviceCannotHoldOneLayer() {
        assertNull(BackFill.fill(2, 8, (device, start, end) -> device == 1 ? -1 : 0));
        assertNull("fewer layers than devices", BackFill.fill(3, 2, (device, start, end) -> 0));
    }
}
