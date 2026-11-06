package com.eonmux.neuralfree.gguf;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend.BackendType;
import com.eonmux.neuralfree.acceleration.backend.DeviceCapabilities;
import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import com.eonmux.neuralfree.gguf.VRAMBudget.FreeVram;
import com.eonmux.neuralfree.gguf.VRAMBudget.FreeVramSource;
import com.eonmux.neuralfree.gguf.testutil.FakeComputeBackend;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

/** Selection of the AMD platform free-VRAM sources and the zero-free reading they can produce. */
public class VRAMBudgetFreeSourcesTest {

    private static final long MIB = 1024L * 1024L;
    private static final long GIB = 1024L * MIB;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void platformCountersMeasureEveryProcess() {
        assertTrue(FreeVramSource.AMDGPU_SYSFS.measured());
        assertTrue(FreeVramSource.WDDM_D3DKMT.measured());
        assertFalse(FreeVramSource.ESTIMATE.measured());
        assertFalse(FreeVramSource.UNKNOWN.measured());
    }

    @Test
    public void zeroFromAMeasuredSourceIsAReadingNotUnknown() {
        FreeVram zero = FreeVram.of(0, FreeVramSource.AMDGPU_SYSFS);

        assertEquals(0L, zero.bytes());
        assertEquals(FreeVramSource.AMDGPU_SYSFS, zero.source());
        assertEquals(0L, VRAMBudget.memoryBudgetBytes(8 * GIB, zero, 0, 0, 0));
    }

    @Test
    public void negativeFreeIsUnknown() {
        assertSame(FreeVram.UNKNOWN, FreeVram.of(-1, FreeVramSource.WDDM_D3DKMT));
        assertSame(FreeVram.UNKNOWN, FreeVram.of(5, FreeVramSource.UNKNOWN));
    }

    @Test
    public void linuxUsesTheSysfsCounters() {
        FreeVram free = VRAMBudget.selectLinuxAmd(8 * GIB, 3 * GIB);

        assertEquals(FreeVramSource.AMDGPU_SYSFS, free.source());
        assertEquals(3 * GIB, free.bytes());
        assertTrue(free.source().measured());
    }

    @Test
    public void linuxClampsTheCountersToTheDeviceSize() {
        FreeVram free = VRAMBudget.selectLinuxAmd(8 * GIB, 9 * GIB);

        assertEquals(8 * GIB, free.bytes());
    }

    @Test
    public void linuxWithoutCountersFallsBackToTheRuntime() {
        assertNull(VRAMBudget.selectLinuxAmd(8 * GIB, -1));
    }

    @Test
    public void linuxFullDeviceIsMeasuredZero() {
        FreeVram free = VRAMBudget.selectLinuxAmd(8 * GIB, 0);

        assertEquals(FreeVramSource.AMDGPU_SYSFS, free.source());
        assertEquals(0L, free.bytes());
    }

    @Test
    public void windowsTakesTheSmallerOfTheRuntimeAndWddm() {
        FreeVram wddmSmaller = VRAMBudget.selectWindowsAmd(24 * GIB, 20 * GIB, 18 * GIB);
        assertEquals(FreeVramSource.WDDM_D3DKMT, wddmSmaller.source());
        assertEquals(18 * GIB, wddmSmaller.bytes());

        FreeVram runtimeSmaller = VRAMBudget.selectWindowsAmd(24 * GIB, 10 * GIB, 18 * GIB);
        assertEquals(FreeVramSource.WDDM_D3DKMT, runtimeSmaller.source());
        assertEquals(10 * GIB, runtimeSmaller.bytes());
    }

    @Test
    public void windowsWithoutWddmWithholdsOtherProcessesFromTheRuntimeFigure() {
        FreeVram free = VRAMBudget.selectWindowsAmd(24 * GIB, 20 * GIB, -1);

        assertEquals(FreeVramSource.ESTIMATE, free.source());
        assertEquals(20 * GIB - 3072 * MIB, free.bytes());
    }

    @Test
    public void windowsWithoutWddmNeverGoesNegative() {
        FreeVram free = VRAMBudget.selectWindowsAmd(24 * GIB, 1 * GIB, -1);

        assertEquals(0L, free.bytes());
    }

    @Test
    public void windowsWithoutAnyFigureFallsBack() {
        assertNull(VRAMBudget.selectWindowsAmd(24 * GIB, -1, -1));
    }

    @Test
    public void windowsFullDeviceIsMeasuredZero() {
        FreeVram free = VRAMBudget.selectWindowsAmd(24 * GIB, 0, 0);

        assertEquals(FreeVramSource.WDDM_D3DKMT, free.source());
        assertEquals(0L, free.bytes());
    }

    @Test
    public void queryFreeVramUsesTheAmdgpuSysfsCounters() throws IOException {
        Assume.assumeFalse(System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("win"));
        Path device = tmp.newFolder("0000:03:00.0").toPath();
        Files.write(device.resolve("mem_info_vram_total"), "8589934592".getBytes(StandardCharsets.US_ASCII));
        Files.write(device.resolve("mem_info_vram_used"), "1073741824".getBytes(StandardCharsets.US_ASCII));
        FakeComputeBackend backend = amdBackend(DeviceClass.DISCRETE, "0000:03:00.0");

        FreeVram free = VRAMBudget.queryFreeVram(backend, tmp.getRoot().toPath());

        assertEquals(FreeVramSource.AMDGPU_SYSFS, free.source());
        assertEquals(7 * GIB, free.bytes());
    }

    @Test
    public void integratedAmdDevicesSkipTheVramCounters() throws IOException {
        Path device = tmp.newFolder("0000:03:00.0").toPath();
        Files.write(device.resolve("mem_info_vram_total"), "8589934592".getBytes(StandardCharsets.US_ASCII));
        Files.write(device.resolve("mem_info_vram_used"), "1073741824".getBytes(StandardCharsets.US_ASCII));
        FakeComputeBackend backend = amdBackend(DeviceClass.INTEGRATED, "0000:03:00.0");

        FreeVram free = VRAMBudget.queryFreeVram(backend, tmp.getRoot().toPath());

        assertEquals(FreeVramSource.ESTIMATE, free.source());
        assertEquals(8 * GIB, free.bytes());
    }

    @Test
    public void queryFreeVramFallsBackWhenThePciDeviceIsNotInSysfs() {
        FakeComputeBackend backend = amdBackend(DeviceClass.DISCRETE, "ffff:ff:1f.7");

        FreeVram free = VRAMBudget.queryFreeVram(backend);

        assertEquals(FreeVramSource.ESTIMATE, free.source());
        assertEquals(8 * GIB, free.bytes());
    }

    /** An 8 GiB OpenCL-typed AMD backend whose runtime free figure equals the device size. */
    private static FakeComputeBackend amdBackend(DeviceClass deviceClass, String pciBusId) {
        DeviceIdentity identity = new DeviceIdentity("AMD Radeon RX Fake", 0x1002, pciBusId, null);
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("AMD Radeon RX Fake")
                .totalVramBytes(8 * GIB)
                .maxAllocSizeBytes(2 * GIB)
                .deviceClass(deviceClass)
                .identity(identity)
                .build();
        return FakeComputeBackend.builder()
                .backendType(BackendType.OPENCL)
                .capabilities(caps)
                .build();
    }
}
