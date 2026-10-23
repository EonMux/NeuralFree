package com.eonmux.neuralfree.acceleration;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import org.junit.Test;

import static org.junit.Assert.*;

public class BackendAutoDetectionTest {

    @Test
    public void testAutoModeExistsInEnum() {
        ACCELERATION_MODE auto = ACCELERATION_MODE.AUTO;
        assertNotNull("AUTO mode should exist in ACCELERATION_MODE enum", auto);
        assertEquals("AUTO", auto.name());
    }

    @Test
    public void testAllModesExistInEnum() {
        ACCELERATION_MODE[] modes = ACCELERATION_MODE.values();
        assertEquals("Should have 4 modes", 4, modes.length);
        
        boolean hasAuto = false, hasCuda = false, hasOpencl = false, hasVulkan = false;
        for (ACCELERATION_MODE mode : modes) {
            switch (mode) {
                case AUTO: hasAuto = true; break;
                case CUDA: hasCuda = true; break;
                case OPENCL: hasOpencl = true; break;
                case VULKAN: hasVulkan = true; break;
            }
        }
        assertTrue("Should have AUTO mode", hasAuto);
        assertTrue("Should have CUDA mode", hasCuda);
        assertTrue("Should have OPENCL mode", hasOpencl);
        assertTrue("Should have VULKAN mode", hasVulkan);
    }

    @Test
    public void testNoBackendAvailableExceptionExists() {
        NoBackendAvailableException ex = new NoBackendAvailableException("Test message");
        assertEquals("Test message", ex.getMessage());
        assertTrue("Should extend RuntimeException", ex instanceof RuntimeException);
    }

    @Test
    public void testNoBackendAvailableExceptionWithCause() {
        Exception cause = new RuntimeException("Root cause");
        NoBackendAvailableException ex = new NoBackendAvailableException("Test message", cause);
        assertEquals("Test message", ex.getMessage());
        assertEquals(cause, ex.getCause());
    }

    @Test
    public void testPriorityOrderConstants() {
        assertEquals("AUTO should be first in enum for default behavior", 
                0, ACCELERATION_MODE.AUTO.ordinal());
    }

    @Test
    public void testNoBackendAvailableExceptionMessageContent() {
        NoBackendAvailableException ex = new NoBackendAvailableException(
            "No GPU backend available. Troubleshooting: Test reason");
        assertTrue("Should contain 'No GPU backend available'", 
                ex.getMessage().contains("No GPU backend available"));
        assertTrue("Should contain troubleshooting info", 
                ex.getMessage().contains("Troubleshooting"));
    }

    @Test
    public void testExceptionMessagesAreInformative() {
        NoBackendAvailableException ex1 = new NoBackendAvailableException(
            "Vulkan backend requested but no Vulkan devices available. " +
            "Please install Vulkan drivers from your GPU vendor (NVIDIA/AMD/Intel).");
        assertTrue("Vulkan exception should be informative", ex1.getMessage().contains("Vulkan"));
        assertTrue("Vulkan exception should mention drivers", ex1.getMessage().contains("drivers"));

        NoBackendAvailableException ex2 = new NoBackendAvailableException(
            "CUDA backend requested but no CUDA devices available. " +
            "Please install CUDA Toolkit from NVIDIA and ensure you have an NVIDIA GPU.");
        assertTrue("CUDA exception should be informative", ex2.getMessage().contains("CUDA"));
        assertTrue("CUDA exception should mention NVIDIA", ex2.getMessage().contains("NVIDIA"));

        NoBackendAvailableException ex3 = new NoBackendAvailableException(
            "OpenCL backend requested but no OpenCL devices available. " +
            "Please install OpenCL runtime from your GPU vendor (NVIDIA/AMD/Intel).");
        assertTrue("OpenCL exception should be informative", ex3.getMessage().contains("OpenCL"));
        assertTrue("OpenCL exception should mention runtime", ex3.getMessage().contains("runtime"));
    }

    @Test
    public void testEnumValueOfAuto() {
        ACCELERATION_MODE mode = ACCELERATION_MODE.valueOf("AUTO");
        assertEquals(ACCELERATION_MODE.AUTO, mode);
    }

    @Test
    public void testEnumValueOfAllModes() {
        assertEquals(ACCELERATION_MODE.AUTO, ACCELERATION_MODE.valueOf("AUTO"));
        assertEquals(ACCELERATION_MODE.CUDA, ACCELERATION_MODE.valueOf("CUDA"));
        assertEquals(ACCELERATION_MODE.OPENCL, ACCELERATION_MODE.valueOf("OPENCL"));
        assertEquals(ACCELERATION_MODE.VULKAN, ACCELERATION_MODE.valueOf("VULKAN"));
    }

    @Test
    public void testBackendPriorityOrder() {
        ACCELERATION_MODE[] expectedOrder = {
            ACCELERATION_MODE.AUTO,
            ACCELERATION_MODE.CUDA,
            ACCELERATION_MODE.OPENCL,
            ACCELERATION_MODE.VULKAN
        };
        
        ACCELERATION_MODE[] actualOrder = ACCELERATION_MODE.values();
        assertArrayEquals("Enum order should match expected priority", expectedOrder, actualOrder);
    }
}
