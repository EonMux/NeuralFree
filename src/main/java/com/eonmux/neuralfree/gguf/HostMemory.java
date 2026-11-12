package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Total physical RAM of the host, the figure the larger-than-RAM decisions compare a model against: Avalon's
 * server disables its prompt cache when the model bytes exceed {@code ram_total}
 * (tools/server/server-context.cpp:1550-1566), and NeuralFree sizes its host weight cache the same way. Read
 * through {@link com.sun.management.OperatingSystemMXBean#getTotalPhysicalMemorySize()} once and cached; -1 when
 * the runtime does not report it. {@link #availableBytes()} is the RAM the host can still hand out now, read on every
 * call.
 */
public final class HostMemory {

    private static final Path MEMINFO = Paths.get("/proc/meminfo");
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

    /**
     * RAM the host can give new allocations now without swapping, in bytes: the kernel's {@code MemAvailable} estimate
     * (free memory plus the page cache and slab it can reclaim) on Linux; -1 where the host does not report it. Read
     * on every call: it changes with the other processes' use.
     */
    public static long availableBytes() {
        if (!Files.isReadable(MEMINFO)) {
            return -1;
        }
        try {
            return parseMemAvailable(new String(Files.readAllBytes(MEMINFO), StandardCharsets.US_ASCII));
        } catch (IOException e) {
            return -1;
        }
    }

    /**
     * The {@code MemAvailable} line of a {@code /proc/meminfo} text, in bytes; -1 when it has none or the value is not
     * a size in kB.
     */
    static long parseMemAvailable(String meminfo) {
        for (String line : meminfo.split("\n")) {
            if (!line.startsWith("MemAvailable:")) {
                continue;
            }
            String[] parts = line.substring("MemAvailable:".length()).trim().split("\\s+");
            if (parts.length != 2 || !parts[1].equals("kB")) {
                return -1;
            }
            try {
                long kb = Long.parseLong(parts[0]);
                return kb < 0 || kb > Long.MAX_VALUE / 1024 ? -1 : kb * 1024;
            } catch (NumberFormatException e) {
                return -1;
            }
        }
        return -1;
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
