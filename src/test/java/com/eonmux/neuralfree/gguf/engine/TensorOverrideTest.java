package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Locale;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link TensorOverride#parse} takes {@code cpu}, {@code gpu} and every GPU backend of
 * {@link EngineConfig.BackendMode} ({@code opencl}, {@code cuda}, {@code vulkan}, {@code hip}, {@code metal}) as
 * its {@code -ot} target, case-insensitively, and refuses anything else naming the valid targets; the target lists
 * derive from {@link EngineConfig.BackendMode}, and {@code hip} and {@code metal} reach the engine's placement the
 * same way the older backend targets do.
 */
public class TensorOverrideTest {

    @Test
    public void hipAndMetalAreDeviceTargets() {
        for (String target : new String[] {"hip", "HIP", "metal", "Metal", " metal "}) {
            TensorOverride o = TensorOverride.parse("blk\\.3\\..*", target);
            assertFalse("target '" + target + "' keeps the tensors on the device", o.host());
            assertEquals(target.trim().toLowerCase(Locale.ROOT), o.target());
        }
    }

    @Test
    public void theEstablishedTargetsStillParse() {
        assertTrue("cpu is the host", TensorOverride.parse("ffn_up_exps", "cpu").host());
        for (String target : new String[] {"gpu", "opencl", "cuda", "vulkan"}) {
            TensorOverride o = TensorOverride.parse("ffn_up_exps", target);
            assertFalse("'" + target + "' is a device target", o.host());
            assertEquals(target, o.target());
        }
    }

    @Test
    public void anUnknownTargetIsRefusedNamingTheValidOnes() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> TensorOverride.parse("ffn_up_exps", "not-a-gpu"));
        assertTrue("the refusal names the value: " + e.getMessage(), e.getMessage().contains("not-a-gpu"));
        assertTrue("the refusal lists cpu: " + e.getMessage(), e.getMessage().contains("cpu"));
        assertTrue("the refusal lists gpu: " + e.getMessage(), e.getMessage().contains("gpu"));
        assertTrue("the refusal lists hip: " + e.getMessage(), e.getMessage().contains("hip"));
        assertTrue("the refusal lists metal: " + e.getMessage(), e.getMessage().contains("metal"));
        assertTrue("the refusal lists sycl: " + e.getMessage(), e.getMessage().contains("sycl"));
    }

    @Test
    public void theTargetsDeriveFromBackendMode() {
        for (EngineConfig.BackendMode mode : EngineConfig.BackendMode.values()) {
            String name = mode.name().toLowerCase(Locale.ROOT);
            if (mode != EngineConfig.BackendMode.AUTO && mode.usesGpu()) {
                assertTrue("targets() names " + name, TensorOverride.targets().contains(name));
                assertTrue("deviceTargets() names " + name, TensorOverride.deviceTargets().contains(name));
                assertFalse(name + " parses as a device rule", TensorOverride.parse("x", name).host());
            }
        }
        assertTrue("cpu is a target", TensorOverride.targets().contains("cpu"));
        assertTrue("gpu is a target", TensorOverride.targets().contains("gpu"));
        assertFalse("auto is no target", TensorOverride.targets().contains("auto"));
        assertFalse("cpu is no device target", TensorOverride.deviceTargets().contains("cpu"));
        assertFalse("gpu is no device target", TensorOverride.deviceTargets().contains("gpu"));
    }

    @Test
    public void hipAndMetalReachTheBuilder() {
        EngineConfig c = EngineConfig.builder()
                .addTensorOverride("ffn_up_exps", "hip")
                .addTensorOverride("blk\\.0\\..*", "Metal")
                .build();
        assertEquals(2, c.tensorOverrides().size());
        assertEquals("hip", c.tensorOverrides().get(0).target());
        assertFalse("the hip rule is a device rule", c.tensorOverrides().get(0).host());
        assertEquals("metal", c.tensorOverrides().get(1).target());
        assertFalse(c.tensorOverrides().get(1).host());
    }

    @Test
    public void theOverrideTensorPropertyAcceptsHipAndMetal() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_OVERRIDES, "exps=hip, blk\\.0\\..*=METAL");
        EngineConfig c = EngineConfig.fromProperties(p);
        assertEquals(2, c.tensorOverrides().size());
        assertEquals("hip", c.tensorOverrides().get(0).target());
        assertEquals("metal", c.tensorOverrides().get(1).target());
    }

    @Test
    public void theOverrideTensorPropertyErrorNamesEveryTarget() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_OVERRIDES, "exps");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue("the refusal shows the target list: " + e.getMessage(),
                e.getMessage().contains("cpu | gpu | opencl | cuda | vulkan | hip | metal"));
    }
}
