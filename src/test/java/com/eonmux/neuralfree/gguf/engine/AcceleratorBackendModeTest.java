package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.Config.ACCELERATION_MODE;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The backend names the engine configuration accepts: the accelerator APIs and their runtime
 * spellings reach the matching mode, TPU is refused with its reason, and an unknown name is refused
 * with the accepted list. {@code neuralfree.backend} reaches the same parser through
 * {@link EngineProperties}.
 */
public class AcceleratorBackendModeTest {

    @Test
    public void acceleratorNamesReachTheirModes() {
        assertEquals(EngineConfig.BackendMode.SYCL, EngineConfig.BackendMode.parse("sycl"));
        assertEquals(EngineConfig.BackendMode.SYCL, EngineConfig.BackendMode.parse(" LEVEL-ZERO "));
        assertEquals(EngineConfig.BackendMode.XPU, EngineConfig.BackendMode.parse("xpu"));
        assertEquals(EngineConfig.BackendMode.CANN, EngineConfig.BackendMode.parse("cann"));
        assertEquals(EngineConfig.BackendMode.HEXAGON, EngineConfig.BackendMode.parse("hexagon"));
        assertEquals(EngineConfig.BackendMode.OPENVINO, EngineConfig.BackendMode.parse("openvino"));
        assertEquals(EngineConfig.BackendMode.WEBGPU, EngineConfig.BackendMode.parse("web-gpu"));
        assertEquals(EngineConfig.BackendMode.MUSA, EngineConfig.BackendMode.parse("musa"));
        assertEquals(EngineConfig.BackendMode.ZDNN, EngineConfig.BackendMode.parse("zdnn"));
        assertEquals(EngineConfig.BackendMode.ZENDNN, EngineConfig.BackendMode.parse("zen-dnn"));
        assertEquals(EngineConfig.BackendMode.METAL, EngineConfig.BackendMode.parse("metal"));
    }

    @Test
    public void acceleratorModesUseTheGpuPath() {
        assertTrue(EngineConfig.BackendMode.SYCL.usesGpu());
        assertEquals(ACCELERATION_MODE.SYCL, EngineConfig.BackendMode.SYCL.toAccelerationMode());
        assertEquals(ACCELERATION_MODE.ZENDNN, EngineConfig.BackendMode.ZENDNN.toAccelerationMode());
        assertEquals(ACCELERATION_MODE.XPU, EngineConfig.BackendMode.XPU.toAccelerationMode());
        assertFalse(EngineConfig.BackendMode.CPU.usesGpu());
    }

    @Test
    public void cpuHasNoAccelerationMode() {
        try {
            EngineConfig.BackendMode.CPU.toAccelerationMode();
            fail("CPU has no GPU acceleration mode");
        } catch (IllegalStateException expected) {
            assertTrue(expected.getMessage(), expected.getMessage().contains("CPU"));
        }
    }

    @Test
    public void tpuIsRefusedByNameWithItsReason() {
        try {
            EngineConfig.BackendMode.parse("tpu");
            fail("TPU is not reachable from this engine");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_BACKEND));
            assertTrue(e.getMessage(), e.getMessage().contains("tpu"));
            assertTrue(e.getMessage(), e.getMessage().contains("not supported"));
            assertTrue(e.getMessage(), e.getMessage().contains("torch_xla"));
        }
    }

    @Test
    public void anUnknownNameIsRefusedWithTheAcceptedList() {
        try {
            EngineConfig.BackendMode.parse("not-an-api");
            fail("not-an-api names no backend of this engine");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("not-an-api"));
            assertTrue(e.getMessage(), e.getMessage().contains("sycl"));
            assertTrue(e.getMessage(), e.getMessage().contains("metal"));
            assertTrue(e.getMessage(), e.getMessage().contains("zendnn"));
            assertTrue(e.getMessage(), e.getMessage().contains("cpu"));
        }
    }

    @Test
    public void thePropertyReachesTheAcceleratorMode() {
        Properties properties = new Properties();
        properties.setProperty(EngineConfig.PROP_BACKEND, "level-zero");
        assertEquals(EngineConfig.BackendMode.SYCL, EngineProperties.read(properties).backend());

        properties.setProperty(EngineConfig.PROP_BACKEND, "tpu");
        try {
            EngineProperties.read(properties);
            fail("the property must refuse tpu");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains("torch_xla"));
        }
    }
}
