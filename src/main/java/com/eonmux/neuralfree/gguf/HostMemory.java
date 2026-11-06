package com.eonmux.neuralfree.gguf;

import java.lang.management.ManagementFactory;

/**
 * Total physical RAM of the host, the figure the larger-than-RAM decisions compare a model against: Avalon's
 * server disables its prompt cache when the model bytes exceed {@code ram_total}
 * (tools/server/server-context.cpp:1550-1566), and NeuralFree sizes its host weight cache the same way. Read
 * through {@link com.sun.management.OperatingSystemMXBean#getTotalPhysicalMemorySize()} once and cached; -1 when
 * the runtime does not report it.
 */
public final class HostMemory {

    private static volatile Long cachedTotal;

    private HostMemory() {
    }

    /** Total physical RAM in bytes, or -1 when unknown. */
    public static long totalBytes() {
        Long cached = cachedTotal;
        if (cached == null) {
            synchronized (HostMemory.class) {
                cached = cachedTotal;
                if (cached == null) {
                    cachedTotal = cached = queryTotalBytes();
                }
            }
        }
        return cached;
    }

    private static long queryTotalBytes() {
        try {
            java.lang.management.OperatingSystemMXBean bean = ManagementFactory.getOperatingSystemMXBean();
            if (bean instanceof com.sun.management.OperatingSystemMXBean) {
                long bytes = ((com.sun.management.OperatingSystemMXBean) bean).getTotalPhysicalMemorySize();
                if (bytes > 0) {
                    return bytes;
                }
            }
        } catch (RuntimeException e) {
            // non-HotSpot runtime or a denied MXBean: fall through to unknown
        }
        return -1;
    }
}
