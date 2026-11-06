package com.eonmux.neuralfree.gguf;

import java.lang.management.ManagementFactory;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link HostMemory}: the cached physical-RAM figure matches the runtime's
 * {@link com.sun.management.OperatingSystemMXBean#getTotalPhysicalMemorySize()}, with non-positive figures
 * reported as unknown (-1).
 */
public class HostMemoryTest {

    @Test
    public void totalBytesMatchesTheRuntimeFigureAndStaysCached() {
        java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
        long expected = -1;
        if (bean instanceof com.sun.management.OperatingSystemMXBean) {
            long reported = ((com.sun.management.OperatingSystemMXBean) bean).getTotalPhysicalMemorySize();
            expected = reported > 0 ? reported : -1;
        }
        long total = HostMemory.totalBytes();
        assertEquals(expected, total);
        assertEquals("the figure is read once and cached", total, HostMemory.totalBytes());
        assertTrue("the test host reports its physical RAM", total > 0);
    }
}
