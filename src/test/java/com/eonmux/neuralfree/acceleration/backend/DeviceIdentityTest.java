package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public class DeviceIdentityTest {

    private static final String RTX = "NVIDIA GeForce RTX 5080 Laptop GPU";
    private static final String UUID_A = "0123456789abcdef0123456789abcdef";
    private static final String UUID_B = "fedcba9876543210fedcba9876543210";

    @Test
    public void pciBusIdIsNormalised() {
        assertEquals("0000:01:00.0", new DeviceIdentity(RTX, 0x10de, "0000:01:00.0", null).pciBusId());
        assertEquals("0000:01:00.0", new DeviceIdentity(RTX, 0x10de, "00000000:01:00.0", null).pciBusId());
        assertEquals("0000:0a:1f.7", new DeviceIdentity(RTX, 0x10de, "0A:1F.7", null).pciBusId());
        assertEquals("0000:01:00.0", DeviceIdentity.formatPciBusId(0, 1, 0, 0));
        assertNull(new DeviceIdentity(RTX, 0x10de, "  ", null).pciBusId());
    }

    @Test
    public void uuidIsNormalised() {
        assertEquals(UUID_A, new DeviceIdentity(RTX, -1, null, "01234567-89AB-CDEF-0123-456789ABCDEF").uuid());
        assertNull("all-zero UUID means unknown",
                new DeviceIdentity(RTX, -1, null, "00000000000000000000000000000000").uuid());
        byte[] raw = new byte[16];
        raw[0] = (byte) 0xAB;
        raw[15] = 0x01;
        assertEquals("ab000000000000000000000000000001", DeviceIdentity.hexUuid(raw));
    }

    @Test
    public void khronosVendorIdsAboveSixteenBitsAreAccepted() {
        DeviceIdentity llvmpipe = new DeviceIdentity("llvmpipe (LLVM 19.1.0, 256 bits)", 0x10005, null, null);
        assertEquals(0x10005, llvmpipe.vendorId());
    }

    @Test
    public void malformedValuesAreRejected() {
        expectIllegal(() -> new DeviceIdentity(null, -1, null, null));
        expectIllegal(() -> new DeviceIdentity(RTX, -1, "01-00-0", null));
        expectIllegal(() -> new DeviceIdentity(RTX, -1, "0000:01:00.8", null));
        expectIllegal(() -> new DeviceIdentity(RTX, -1, null, "abc"));
        expectIllegal(() -> new DeviceIdentity(RTX, -2, null, null));
        expectIllegal(() -> DeviceIdentity.hexUuid(new byte[8]));
        expectIllegal(() -> DeviceIdentity.formatPciBusId(0, 256, 0, 0));
    }

    @Test
    public void pciBusIdDecidesWhenBothKnown() {
        DeviceIdentity ocl = new DeviceIdentity(RTX, 0x10de, "0000:01:00.0", UUID_A);
        DeviceIdentity vk = new DeviceIdentity("other name", 0x10de, "0000:01:00.0", UUID_B);
        DeviceIdentity otherSlot = new DeviceIdentity(RTX, 0x10de, "0000:02:00.0", UUID_A);
        assertTrue(ocl.samePhysicalDevice(vk));
        assertFalse("different PCI address wins over equal UUID and name", ocl.samePhysicalDevice(otherSlot));
    }

    @Test
    public void uuidDecidesWithoutPciBusId() {
        DeviceIdentity a = new DeviceIdentity(RTX, 0x10de, "0000:01:00.0", UUID_A);
        DeviceIdentity b = new DeviceIdentity("renamed", 0x10de, null, UUID_A);
        DeviceIdentity c = new DeviceIdentity(RTX, 0x10de, null, UUID_B);
        assertTrue(a.samePhysicalDevice(b));
        assertFalse(a.samePhysicalDevice(c));
    }

    @Test
    public void nameDecidesAsLastResort() {
        DeviceIdentity a = DeviceIdentity.ofName(RTX);
        DeviceIdentity b = new DeviceIdentity(RTX, 0x10de, "0000:01:00.0", null);
        assertTrue(a.samePhysicalDevice(b));
        assertFalse(a.samePhysicalDevice(DeviceIdentity.ofName("Intel(R) Graphics")));
        assertFalse("empty names never match", DeviceIdentity.ofName("").samePhysicalDevice(DeviceIdentity.ofName("")));
        assertFalse(a.samePhysicalDevice(null));
    }

    @Test
    public void equalityCoversAllFields() {
        DeviceIdentity a = new DeviceIdentity(RTX, 0x10de, "0000:01:00.0", UUID_A);
        DeviceIdentity b = new DeviceIdentity(RTX, 0x10de, "01:00.0", UUID_A.toUpperCase());
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
        assertFalse(a.equals(new DeviceIdentity(RTX, 0x1002, "0000:01:00.0", UUID_A)));
    }

    private static void expectIllegal(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // malformed input is rejected
        }
    }
}
