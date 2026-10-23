package com.eonmux.neuralfree.acceleration.vulkan;

import com.eonmux.neuralfree.acceleration.backend.DeviceClass;
import com.eonmux.neuralfree.acceleration.backend.DeviceIdentity;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

/**
 * The Vulkan device-ranking policy over synthetic profiles: device class first, then the
 * OpenCL-measured hint within the best class, then device-local memory, then shader cores, then
 * enumeration order, with the forced choice by name ahead of everything. No device is needed.
 */
public class VkDeviceRankingTest {

    private String savedDevice;
    private String savedGlobal;

    @Before
    public void clearDeviceFilters() {
        savedDevice = System.getProperty(VkDeviceRanking.DEVICE_PROPERTY);
        savedGlobal = System.getProperty(VkDeviceRanking.GLOBAL_DEVICE_PROPERTY);
        System.clearProperty(VkDeviceRanking.DEVICE_PROPERTY);
        System.clearProperty(VkDeviceRanking.GLOBAL_DEVICE_PROPERTY);
    }

    @After
    public void restoreDeviceFilters() {
        restore(VkDeviceRanking.DEVICE_PROPERTY, savedDevice);
        restore(VkDeviceRanking.GLOBAL_DEVICE_PROPERTY, savedGlobal);
    }

    @Test
    public void emptyOrNullListHasNoWinner() {
        assertEquals(-1, VkDeviceRanking.rank(Collections.<VkDeviceRanking.Profile>emptyList(), null));
        assertEquals(-1, VkDeviceRanking.rank(null, null));
        assertTrue(VkDeviceRanking.eligible(null).isEmpty());
        assertNull(VkDeviceRanking.fastest(Collections.<VkPhysicalDeviceInfo>emptyList()));
    }

    @Test
    public void classOrderIsDiscreteIntegratedVirtualUnknownCpu() {
        assertTrue(VkDeviceRanking.classRank(DeviceClass.DISCRETE) < VkDeviceRanking.classRank(DeviceClass.INTEGRATED));
        assertTrue(VkDeviceRanking.classRank(DeviceClass.INTEGRATED) < VkDeviceRanking.classRank(DeviceClass.VIRTUAL));
        assertTrue(VkDeviceRanking.classRank(DeviceClass.VIRTUAL) < VkDeviceRanking.classRank(DeviceClass.UNKNOWN));
        assertTrue(VkDeviceRanking.classRank(DeviceClass.UNKNOWN) < VkDeviceRanking.classRank(DeviceClass.CPU));
    }

    @Test
    public void discreteBeatsALargerIntegratedDevice() {
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("iGPU", DeviceClass.INTEGRATED, null, 64L << 30, 0),
                profile("dGPU", DeviceClass.DISCRETE, null, 8L << 30, 0));

        assertEquals(1, VkDeviceRanking.rank(p, null));
    }

    @Test
    public void measuredHintWinsWithinTheBestClass() {
        DeviceIdentity slowCard = new DeviceIdentity("Card B", 0x1002, "0000:03:00.0", null);
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("Card A", DeviceClass.DISCRETE, new DeviceIdentity("Card A", 0x10de, "0000:01:00.0", null),
                        24L << 30, 0),
                profile("Card B", DeviceClass.DISCRETE, slowCard, 16L << 30, 0));
        DeviceIdentity hint = new DeviceIdentity("Card B (OpenCL)", 0x1002, "0:3:0.0", null);

        assertEquals(1, VkDeviceRanking.rank(p, hint));
        assertEquals(0, VkDeviceRanking.rank(p, null));
    }

    @Test
    public void measuredHintDoesNotLiftALowerClass() {
        DeviceIdentity integrated = new DeviceIdentity("iGPU", 0x8086, "0000:00:02.0", null);
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("dGPU", DeviceClass.DISCRETE, new DeviceIdentity("dGPU", 0x10de, "0000:01:00.0", null),
                        8L << 30, 0),
                profile("iGPU", DeviceClass.INTEGRATED, integrated, 32L << 30, 0));

        assertEquals(0, VkDeviceRanking.rank(p, integrated));
    }

    @Test
    public void memoryThenShaderCoresThenEnumerationOrderBreakTies() {
        List<VkDeviceRanking.Profile> byMemory = Arrays.asList(
                profile("a", DeviceClass.DISCRETE, null, 8L << 30, 9000),
                profile("b", DeviceClass.DISCRETE, null, 16L << 30, 100));
        List<VkDeviceRanking.Profile> byCores = Arrays.asList(
                profile("a", DeviceClass.DISCRETE, null, 16L << 30, 4000),
                profile("b", DeviceClass.DISCRETE, null, 16L << 30, 8000));
        List<VkDeviceRanking.Profile> tie = Arrays.asList(
                profile("a", DeviceClass.DISCRETE, null, 16L << 30, 0),
                profile("b", DeviceClass.DISCRETE, null, 16L << 30, 0));

        assertEquals(1, VkDeviceRanking.rank(byMemory, null));
        assertEquals(1, VkDeviceRanking.rank(byCores, null));
        assertEquals(0, VkDeviceRanking.rank(tie, null));
    }

    @Test
    public void forcedNameWinsOverClass() {
        System.setProperty(VkDeviceRanking.DEVICE_PROPERTY, "LLVMPIPE");
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("NVIDIA GeForce RTX 5080", DeviceClass.DISCRETE, null, 16L << 30, 0),
                profile("llvmpipe (LLVM 19.1.7, 256 bits)", DeviceClass.CPU, null, 0, 0));

        assertEquals(1, VkDeviceRanking.rank(p, null));
    }

    @Test
    public void globalFilterAppliesWhenTheVulkanFilterIsUnset() {
        System.setProperty(VkDeviceRanking.GLOBAL_DEVICE_PROPERTY, "radeon");
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("NVIDIA GeForce RTX 4090", DeviceClass.DISCRETE, null, 24L << 30, 0),
                profile("AMD Radeon 780M", DeviceClass.INTEGRATED, null, 4L << 30, 0));

        assertEquals(1, VkDeviceRanking.rank(p, null));
    }

    @Test
    public void filterMatchingNothingFallsBackToTheRanking() {
        System.setProperty(VkDeviceRanking.DEVICE_PROPERTY, "no such device");
        List<VkDeviceRanking.Profile> p = Arrays.asList(
                profile("iGPU", DeviceClass.INTEGRATED, null, 4L << 30, 0),
                profile("dGPU", DeviceClass.DISCRETE, null, 8L << 30, 0));

        assertEquals(1, VkDeviceRanking.rank(p, null));
    }

    @Test
    public void profileDefaultsMissingNameAndClass() {
        VkDeviceRanking.Profile p = new VkDeviceRanking.Profile(null, null, null, 0, 0);

        assertEquals("<unnamed>", p.name);
        assertEquals(DeviceClass.UNKNOWN, p.deviceClass);
    }

    private static VkDeviceRanking.Profile profile(String name, DeviceClass c, DeviceIdentity id, long memory,
                                                   int cores) {
        return new VkDeviceRanking.Profile(name, c, id, memory, cores);
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
