package com.eonmux.neuralfree.cli;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The device listing's printed form, llama.cpp's {@code Available devices:} header with one line per
 * device and the {@code (none)} line for an empty enumeration; a device whose memory the API does not
 * report prints without a size. The enumeration itself runs the engine's own
 * {@code HWAcceleration}, so the tests cover the formatting, never the machine's devices.
 */
public class DeviceListingTest {

    private static final long MIB = 1024L * 1024L;

    @Test
    public void anEmptyEnumerationPrintsTheNoneLine() {
        assertEquals("Available devices:\n  (none)\n", DeviceListing.lines(Collections.emptyList()));
    }

    @Test
    public void eachDevicePrintsWithItsApiSelectorIndexAndTotalMemory() {
        String lines = DeviceListing.lines(Arrays.asList(
                new DeviceListing.Entry("OpenCL", 0, "NVIDIA RTX 5080 Laptop GPU", 15 * MIB),
                new DeviceListing.Entry("CUDA", 0, "NVIDIA RTX 5080 Laptop GPU", 15 * MIB),
                new DeviceListing.Entry("Vulkan", 1, "AMD Radeon", 8 * MIB),
                new DeviceListing.Entry("HIP", 0, "AMD Radeon", 8 * MIB)));
        assertTrue(lines, lines.startsWith("Available devices:\n"));
        assertTrue(lines, lines.contains("  OpenCL 0: NVIDIA RTX 5080 Laptop GPU (15 MiB)\n"));
        assertTrue(lines, lines.contains("  CUDA 0: NVIDIA RTX 5080 Laptop GPU (15 MiB)\n"));
        assertTrue(lines, lines.contains("  Vulkan 1: AMD Radeon (8 MiB)\n"));
        assertTrue(lines, lines.contains("  HIP 0: AMD Radeon (8 MiB)\n"));
        assertEquals(5, lines.split("\n", -1).length - 1);
    }

    @Test
    public void anUnknownMemoryPrintsNoSize() {
        String lines = DeviceListing.lines(Arrays.asList(
                new DeviceListing.Entry("OpenCL", 0, "Unknown GPU", -1)));
        assertTrue(lines, lines.contains("  OpenCL 0: Unknown GPU\n"));
        assertFalse(lines, lines.contains("MiB"));
    }

    /**
     * The printed index is the one {@code -dev} selects: every API's entries are numbered from 0 in
     * enumeration order, so the first device of each API is device 0 of that backend.
     */
    @Test
    public void everyApisDevicesAreNumberedFromZero() {
        int[] next = new int[4];
        String[] apis = {"OpenCL", "CUDA", "Vulkan", "HIP"};
        for (DeviceListing.Entry e : DeviceListing.available()) {
            int a = Arrays.asList(apis).indexOf(e.api());
            assertTrue("a known API: " + e.api(), a >= 0);
            assertEquals("the " + e.api() + " devices are numbered from 0", next[a]++, e.index());
        }
    }

    @Test
    public void theEntriesExposeTheirParts() {
        DeviceListing.Entry e = new DeviceListing.Entry("CUDA", 2, "A100", 40L * MIB);
        assertEquals("CUDA", e.api());
        assertEquals(2, e.index());
        assertEquals("A100", e.name());
        assertEquals(40L * MIB, e.totalBytes());
    }
}
