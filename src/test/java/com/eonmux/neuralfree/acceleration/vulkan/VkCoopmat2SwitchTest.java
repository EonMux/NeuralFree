package com.eonmux.neuralfree.acceleration.vulkan;

import org.junit.After;
import org.junit.Assume;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@value VkPhysicalDeviceInfo#COOPMAT2_PROPERTY} switch: its values, and on a device that reports
 * {@code VK_NV_cooperative_matrix2} (skipped elsewhere) the enumeration under {@code off}, which reports none of the
 * CM2 features while keeping the KHR cooperative matrices; and the values of
 * {@value VkPhysicalDeviceInfo#COOPMAT2_DECODE_VECTOR_PROPERTY}.
 */
public class VkCoopmat2SwitchTest {

    @After
    public void clearProperty() {
        System.clearProperty(VkPhysicalDeviceInfo.COOPMAT2_PROPERTY);
    }

    @Test
    public void autoAndUnsetAllowTheFeaturesAndOffRefusesThem() {
        assertTrue(VkPhysicalDeviceInfo.coopMat2Allowed(null));
        assertTrue(VkPhysicalDeviceInfo.coopMat2Allowed(" "));
        assertTrue(VkPhysicalDeviceInfo.coopMat2Allowed("auto"));
        assertTrue(VkPhysicalDeviceInfo.coopMat2Allowed("AUTO"));
        assertFalse(VkPhysicalDeviceInfo.coopMat2Allowed("off"));
        assertFalse(VkPhysicalDeviceInfo.coopMat2Allowed(" Off "));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VkPhysicalDeviceInfo.coopMat2Allowed("on"));
        assertTrue(e.getMessage(), e.getMessage().contains(VkPhysicalDeviceInfo.COOPMAT2_PROPERTY));
    }

    @Test
    public void theDecodeVectorSwitchTakesAutoAndOff() {
        assertTrue(VkPhysicalDeviceInfo.decodeVectorAllowed(null));
        assertTrue(VkPhysicalDeviceInfo.decodeVectorAllowed(""));
        assertTrue(VkPhysicalDeviceInfo.decodeVectorAllowed(" Auto "));
        assertFalse(VkPhysicalDeviceInfo.decodeVectorAllowed("OFF"));
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VkPhysicalDeviceInfo.decodeVectorAllowed("true"));
        assertTrue(e.getMessage(), e.getMessage().contains(VkPhysicalDeviceInfo.COOPMAT2_DECODE_VECTOR_PROPERTY));
    }

    @Test
    public void offEnumeratesTheDeviceWithoutTheCooperativeMatrix2Features() {
        VkInstanceManager first;
        try {
            first = VkInstanceManager.acquire();
        } catch (RuntimeException e) {
            Assume.assumeNoException(e);
            return;
        }
        String name = null;
        try {
            for (VkPhysicalDeviceInfo info : first.devices()) {
                if (name == null && info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2)) {
                    name = info.name();
                }
            }
        } finally {
            first.release();
        }
        Assume.assumeTrue("no device reports VK_NV_cooperative_matrix2", name != null);
        System.setProperty(VkPhysicalDeviceInfo.COOPMAT2_PROPERTY, "off");
        VkInstanceManager manager = VkInstanceManager.acquire();
        try {
            Assume.assumeTrue("another holder keeps the first enumeration alive", manager != first);
            boolean found = false;
            for (VkPhysicalDeviceInfo info : manager.devices()) {
                if (info.name().equals(name)) {
                    found = true;
                    assertFalse(name, info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2));
                    assertFalse(name, info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2_ATTENTION));
                    assertFalse(name,
                            info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX_2_DECODE_VECTOR));
                    assertTrue(name, info.supports(VkPhysicalDeviceInfo.Feature.COOPERATIVE_MATRIX));
                }
            }
            assertTrue("device " + name + " enumerated again", found);
        } finally {
            manager.release();
        }
    }
}
