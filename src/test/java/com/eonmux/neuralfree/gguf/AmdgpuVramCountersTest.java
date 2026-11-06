package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;

/** Reads the amdgpu driver's {@code mem_info_vram_*} counters from a synthetic sysfs tree. */
public class AmdgpuVramCountersTest {

    private static final String BUS_ID = "0000:03:00.0";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void readsFreeFromTotalAndUsed() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "17163091968\n");
        write(device, "mem_info_vram_used", "2147483648\n");

        assertEquals(17163091968L - 2147483648L,
                AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void fullyUsedDeviceReportsZeroNotUnavailable() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "8589934592");
        write(device, "mem_info_vram_used", "8589934592");

        assertEquals(0L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void missingUsedCounterIsUnavailable() throws IOException {
        write(device(), "mem_info_vram_total", "8589934592");

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void usedGreaterThanTotalIsUnavailable() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "1024");
        write(device, "mem_info_vram_used", "2048");

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void malformedCounterIsUnavailable() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "not-a-number");
        write(device, "mem_info_vram_used", "0");

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void negativeCounterIsUnavailable() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "-1");
        write(device, "mem_info_vram_used", "0");

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), BUS_ID));
    }

    @Test
    public void unknownBusIdIsUnavailable() throws IOException {
        Path device = device();
        write(device, "mem_info_vram_total", "1024");
        write(device, "mem_info_vram_used", "0");

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), "0000:04:00.0"));
        assertEquals(-1L, AmdgpuVramCounters.freeBytes(tmp.getRoot().toPath(), null));
    }

    @Test
    public void aBusIdThatLeavesTheSysfsDirectoryIsUnavailable() {
        Path root = tmp.getRoot().toPath();

        assertEquals(-1L, AmdgpuVramCounters.freeBytes(root, "../outside"));
        assertEquals(-1L, AmdgpuVramCounters.freeBytes(root, "/sys/bus/pci/devices/0000:03:00.0"));
    }

    @Test
    public void identityWithoutPciAddressIsUnavailable() {
        assertEquals(-1L, AmdgpuVramCounters.freeBytes((DeviceIdentity) null));
        assertEquals(-1L, AmdgpuVramCounters.freeBytes(DeviceIdentity.ofName("AMD Radeon")));
    }

    private Path device() throws IOException {
        return tmp.newFolder(BUS_ID).toPath();
    }

    private static void write(Path device, String counter, String value) throws IOException {
        Files.write(device.resolve(counter), value.getBytes(StandardCharsets.US_ASCII));
    }
}
