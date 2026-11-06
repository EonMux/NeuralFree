package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Free VRAM from the amdgpu driver's sysfs counters.
 *
 * <p>The driver publishes {@code mem_info_vram_total} and {@code mem_info_vram_used} for every PCI
 * device under {@code /sys/bus/pci/devices/<bus id>}; both count the allocations of every process.
 * On ROCm {@code hipMemGetInfo} can report noticeably less free memory than these counters, so a
 * cache or streaming budget sized from the HIP figure leaves usable VRAM unused.</p>
 */
public final class AmdgpuVramCounters {

    private static final Logger LOGGER = LogManager.getLogger(AmdgpuVramCounters.class);

    /** Directory the amdgpu driver exposes one entry per PCI device under. */
    public static final Path SYSFS_PCI_DEVICES = Paths.get("/sys/bus/pci/devices");

    private AmdgpuVramCounters() {
    }

    /**
     * Free VRAM of {@code identity}'s PCI device as the amdgpu driver reports it, or -1 when the
     * identity has no PCI bus id, the counters are missing or unreadable, or their values are
     * inconsistent (used greater than total). A completed reading may legitimately be 0.
     */
    public static long freeBytes(DeviceIdentity identity) {
        if (identity == null) {
            return -1;
        }
        return freeBytes(SYSFS_PCI_DEVICES, identity.pciBusId());
    }

    /** {@link #freeBytes(DeviceIdentity)} against an explicit sysfs device directory, for tests. */
    static long freeBytes(Path pciDevices, String pciBusId) {
        if (pciDevices == null || pciBusId == null || pciBusId.trim().isEmpty()) {
            return -1;
        }
        Path device;
        try {
            Path name = pciDevices.getFileSystem().getPath(pciBusId.trim());
            if (name.isAbsolute() || name.getNameCount() != 1) {
                LOGGER.debug("amdgpu: '{}' is not a single PCI device name", pciBusId);
                return -1;
            }
            device = pciDevices.resolve(name);
        } catch (RuntimeException e) {
            LOGGER.debug("amdgpu: '{}' is not a usable PCI bus id: {}", pciBusId, e.toString());
            return -1;
        }
        long total = counter(device.resolve("mem_info_vram_total"));
        long used = counter(device.resolve("mem_info_vram_used"));
        if (total <= 0 || used < 0 || used > total) {
            LOGGER.debug("amdgpu: no consistent VRAM counters for {} (total={}, used={})", device, total, used);
            return -1;
        }
        long free = total - used;
        LOGGER.debug("amdgpu: {} reports {} MiB free of {} MiB", device, free >> 20, total >> 20);
        return free;
    }

    /** One counter file's value, or -1 when it is missing, unreadable or not a non-negative integer. */
    private static long counter(Path path) {
        try {
            String text = new String(Files.readAllBytes(path), StandardCharsets.US_ASCII).trim();
            long value = Long.parseLong(text);
            return value >= 0 ? value : -1;
        } catch (IOException | NumberFormatException | SecurityException e) {
            LOGGER.debug("amdgpu: counter {} is unavailable: {}", path, e.toString());
            return -1;
        }
    }
}
