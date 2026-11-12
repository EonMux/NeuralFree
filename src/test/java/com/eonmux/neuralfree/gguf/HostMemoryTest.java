package com.eonmux.neuralfree.gguf;

import java.lang.management.ManagementFactory;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link HostMemory}: the cached physical-RAM figure matches the runtime's
 * {@link com.sun.management.OperatingSystemMXBean#getTotalPhysicalMemorySize()}, with non-positive figures
 * reported as unknown (-1); the available RAM is {@code /proc/meminfo}'s {@code MemAvailable}.
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

    @Test
    public void theAvailableRamIsTheMemAvailableLineInBytes() {
        String meminfo = "MemTotal:       65536000 kB\nMemFree:         1000000 kB\nMemAvailable:   46157572 kB\n"
                + "Buffers:          123456 kB\n";
        assertEquals(46157572L * 1024, HostMemory.parseMemAvailable(meminfo));
        assertEquals("no line", -1L, HostMemory.parseMemAvailable("MemTotal: 1 kB\n"));
        assertEquals("not kB", -1L, HostMemory.parseMemAvailable("MemAvailable: 12 MB\n"));
        assertEquals("not a number", -1L, HostMemory.parseMemAvailable("MemAvailable: lots kB\n"));
        long available = HostMemory.availableBytes();
        assertTrue("this Linux host reports its available RAM, below the total",
                available > 0 && available <= HostMemory.totalBytes());
    }
}
