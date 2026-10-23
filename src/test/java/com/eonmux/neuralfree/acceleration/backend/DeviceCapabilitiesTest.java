package com.eonmux.neuralfree.acceleration.backend;

import org.junit.Test;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.*;

public class DeviceCapabilitiesTest {

    @Test
    public void testBuilderCreatesFullCapabilities() {
        Set<String> activations = new HashSet<>(Arrays.asList("SIGMOID", "RELU", "TANH"));
        Set<String> costs = new HashSet<>(Arrays.asList("MSE", "MAE"));
        Set<String> blasLevels = new HashSet<>(Arrays.asList("1", "2", "3"));

        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("TestDevice")
                .totalVramBytes(8L * 1024 * 1024 * 1024)
                .computeUnits(16)
                .maxClockFrequencyMhz(1500)
                .maxWorkGroupSize(256)
                .supportsAsyncTransfer(true)
                .driverVersion("1.0.0")
                .supportsFloat64(true)
                .supportsFloat16(false)
                .maxLocalMemoryBytes(48 * 1024)
                .supportedActivationFunctions(activations)
                .supportedCostFunctions(costs)
                .supportedBLASLevels(blasLevels)
                .build();

        assertEquals("TestDevice", caps.getDeviceName());
        assertEquals(8L * 1024 * 1024 * 1024, caps.getTotalVramBytes());
        assertEquals(16, caps.getComputeUnits());
        assertEquals(1500, caps.getMaxClockFrequencyMhz());
        assertEquals(256, caps.getMaxWorkGroupSize());
        assertTrue(caps.supportsAsyncTransfer());
        assertEquals("1.0.0", caps.getDriverVersion());
        assertTrue(caps.supportsFloat64());
        assertFalse(caps.supportsFloat16());
        assertEquals(48 * 1024, caps.getMaxLocalMemoryBytes());
        assertEquals(3, caps.getSupportedActivationFunctions().size());
        assertEquals(2, caps.getSupportedCostFunctions().size());
        assertEquals(3, caps.getSupportedBLASLevels().size());
    }

    @Test
    public void testBuilderDefaults() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("TestDevice")
                .totalVramBytes(1024 * 1024 * 1024)
                .computeUnits(4)
                .build();

        assertTrue(caps.supportsFloat64());
        assertFalse(caps.supportsFloat16());
        assertEquals(32768, caps.getMaxLocalMemoryBytes());
        assertEquals(18, caps.getSupportedActivationFunctions().size());
        assertEquals(6, caps.getSupportedCostFunctions().size());
        assertEquals(3, caps.getSupportedBLASLevels().size());
    }

    @Test
    public void testActivationFunctionsImmutable() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("TestDevice")
                .build();

        Set<String> functions = caps.getSupportedActivationFunctions();
        try {
            functions.add("NEW");
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
        }
    }

    @Test
    public void testSupportsActivationFunction() {
        Set<String> activations = new HashSet<>(Arrays.asList("SIGMOID", "RELU"));

        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedActivationFunctions(activations)
                .build();

        assertTrue(caps.supportsActivationFunction("SIGMOID"));
        assertTrue(caps.supportsActivationFunction("RELU"));
        assertFalse(caps.supportsActivationFunction("TANH"));
    }

    @Test
    public void testSupportsCostFunction() {
        Set<String> costs = new HashSet<>(Arrays.asList("MSE", "HUBER"));

        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedCostFunctions(costs)
                .build();

        assertTrue(caps.supportsCostFunction("MSE"));
        assertTrue(caps.supportsCostFunction("HUBER"));
        assertFalse(caps.supportsCostFunction("MAE"));
    }

    @Test
    public void testSupportsBLASLevel() {
        Set<String> levels = new HashSet<>(Arrays.asList("1", "3"));

        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedBLASLevels(levels)
                .build();

        assertTrue(caps.supportsBLASLevel("1"));
        assertFalse(caps.supportsBLASLevel("2"));
        assertTrue(caps.supportsBLASLevel("3"));
    }

    @Test
    public void testDefaultActivationFunctions() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder().build();

        String[] expected = {"SIGMOID", "RELU", "TANH", "LEAKY_RELU", "ELU", "SOFTPLUS",
                "SOFTSIGN", "ISRU", "GAUSSIAN", "SINC", "SWISH", "SELU", "HARD_SIGMOID",
                "BENT_IDENTITY", "PRELU", "SRELU", "SOFTEXPONENTIAL", "IDENTITY"};

        for (String expectedFunction : expected) {
            assertTrue("Missing activation function: " + expectedFunction,
                    caps.supportsActivationFunction(expectedFunction));
        }
    }

    @Test
    public void testDefaultCostFunctions() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder().build();

        String[] expected = {"MSE", "MAE", "HUBER", "LOG_COSH", "KL_DIVERGENCE", "CROSS_ENTROPY"};

        for (String expectedFunction : expected) {
            assertTrue("Missing cost function: " + expectedFunction,
                    caps.supportsCostFunction(expectedFunction));
        }
    }

    @Test
    public void testToString() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .deviceName("TestDevice")
                .totalVramBytes(1024 * 1024 * 1024)
                .supportsFloat64(true)
                .maxLocalMemoryBytes(32768)
                .build();

        String str = caps.toString();
        assertTrue(str.contains("TestDevice"));
        assertTrue(str.contains("1024 MB"));
        assertTrue(str.contains("float64=true"));
        assertTrue(str.contains("localMem=32 KB"));
    }

    @Test
    public void testNullActivationFunctions() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedActivationFunctions(null)
                .build();

        assertTrue(caps.getSupportedActivationFunctions().isEmpty());
    }

    @Test
    public void testNullCostFunctions() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedCostFunctions(null)
                .build();

        assertTrue(caps.getSupportedCostFunctions().isEmpty());
    }

    @Test
    public void testNullBLASLevels() {
        DeviceCapabilities caps = new DeviceCapabilities.Builder()
                .supportedBLASLevels(null)
                .build();

        assertTrue(caps.getSupportedBLASLevels().isEmpty());
    }
}
