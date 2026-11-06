package com.eonmux.neuralfree;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The training framework's backend names: accelerator APIs and their runtime spellings are accepted,
 * CPU keeps its "not here" refusal, and TPU is refused with the reason it cannot be driven.
 */
public class NeuralFreeBackendParseTest {

    @Test
    public void acceleratorNamesReachTheirModes() {
        assertEquals(ACCELERATION_MODE.SYCL, NeuralFree.parseBackend("sycl"));
        assertEquals(ACCELERATION_MODE.SYCL, NeuralFree.parseBackend("level_zero"));
        assertEquals(ACCELERATION_MODE.XPU, NeuralFree.parseBackend("XPU"));
        assertEquals(ACCELERATION_MODE.OPENVINO, NeuralFree.parseBackend("openvino"));
        assertEquals(ACCELERATION_MODE.WEBGPU, NeuralFree.parseBackend("web-gpu"));
        assertEquals(ACCELERATION_MODE.ZENDNN, NeuralFree.parseBackend("zen-dnn"));
        assertEquals(ACCELERATION_MODE.METAL, NeuralFree.parseBackend("METAL"));
        assertEquals(ACCELERATION_MODE.AUTO, NeuralFree.parseBackend("auto"));
    }

    @Test
    public void tpuIsRefusedWithItsReason() {
        try {
            NeuralFree.parseBackend("tpu");
            fail("TPU is not reachable from this engine");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("tpu"));
            assertTrue(e.getMessage(), e.getMessage().contains("not supported"));
            assertTrue(e.getMessage(), e.getMessage().contains("torch_xla"));
        }
    }

    @Test
    public void cpuKeepsTheTrainingFrameworkRefusal() {
        try {
            NeuralFree.parseBackend("cpu");
            fail("the training framework has no CPU backend mode");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("training framework"));
        }
    }

    @Test
    public void anUnknownNameIsRefusedWithTheAcceptedList() {
        try {
            NeuralFree.parseBackend("not-an-api");
            fail("not-an-api names no API of this engine");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("not-an-api"));
            assertTrue(e.getMessage(), e.getMessage().contains("metal"));
            assertTrue(e.getMessage(), e.getMessage().contains("sycl"));
            assertTrue(e.getMessage(), e.getMessage().contains("zendnn"));
        }
    }
}
