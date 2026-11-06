package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.gpu.weights.TensorPlacementOverrides;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The placement properties of {@link EngineConfig}: {@code neuralfree.cpu-moe}, {@code neuralfree.n-cpu-moe},
 * {@code neuralfree.n-cpu-ffn}, {@code neuralfree.override-tensor}, {@code neuralfree.moe.cache.mb} and
 * {@code neuralfree.fit} parse with their defaults and refuse malformed values naming the property. The
 * flag-shaped keys are aliases of the dotted settings, and {@code neuralfree.override-tensor} entries reach the
 * same {@link EngineConfig#tensorOverrides()} list as {@code neuralfree.override.tensor}.
 */
public class EngineConfigPlacementTest {

    private static EngineConfig from(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i + 1 < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return EngineConfig.fromProperties(p);
    }

    @Test
    public void placementSettingsDefaultToNone() {
        EngineConfig c = from();
        assertFalse(c.cpuMoe());
        assertEquals(EngineConfig.N_CPU_LAYERS_UNSET, c.nCpuMoe());
        assertEquals(EngineConfig.N_CPU_LAYERS_UNSET, c.nCpuFfn());
        assertTrue(c.tensorOverrides().isEmpty());
        assertEquals(0L, c.moeCacheMb());
        assertFalse(c.fit());
    }

    @Test
    public void placementSettingsParse() {
        EngineConfig c = from(EngineConfig.PROP_CPU_MOE, "true",
                EngineConfig.PROP_N_CPU_MOE, "3",
                EngineConfig.PROP_N_CPU_FFN, "1",
                EngineConfig.PROP_OVERRIDE_TENSOR, "blk\\.2\\..*=GPU",
                EngineConfig.PROP_MOE_CACHE_MB, "512",
                EngineConfig.PROP_FIT, "true");
        assertTrue(c.cpuMoe());
        assertEquals(3, c.nCpuMoe());
        assertEquals(1, c.nCpuFfn());
        assertEquals(1, c.tensorOverrides().size());
        assertFalse("blk\\.2\\..*=GPU reaches the override list as a device rule",
                c.tensorOverrides().get(0).host());
        assertEquals(512L, c.moeCacheMb());
        assertTrue(c.fit());
        assertTrue(c.toBuilder().build().cpuMoe());
        assertEquals(3, c.toBuilder().build().nCpuMoe());
        assertEquals(512L, c.toBuilder().build().moeCacheMb());
    }

    @Test
    public void malformedValuesFailNamingTheProperty() {
        IllegalArgumentException layers = assertThrows(IllegalArgumentException.class,
                () -> from(EngineConfig.PROP_N_CPU_MOE, "-2"));
        assertTrue(layers.getMessage().contains(EngineConfig.PROP_N_CPU_MOE));
        IllegalArgumentException cache = assertThrows(IllegalArgumentException.class,
                () -> from(EngineConfig.PROP_MOE_CACHE_MB, "-1"));
        assertTrue(cache.getMessage().contains(EngineConfig.PROP_MOE_CACHE_MB));
        IllegalArgumentException rules = assertThrows(IllegalArgumentException.class,
                () -> from(EngineConfig.PROP_OVERRIDE_TENSOR, "exps=CPU0"));
        assertTrue(rules.getMessage().contains(EngineConfig.PROP_OVERRIDE_TENSOR));
        assertTrue(rules.getMessage().contains("CPU0"));
    }

    @Test
    public void aNegativeLayerCountClearsTheBuilderValue() {
        EngineConfig c = EngineConfig.builder().nCpuMoe(4).nCpuMoe(-1).build();
        assertEquals(EngineConfig.N_CPU_LAYERS_UNSET, c.nCpuMoe());
    }

    @Test
    public void theOverridesParseIntoPlacementRules() {
        TensorPlacementOverrides rules = TensorPlacementOverrides.parse("blk\\.0\\..*=GPU, ffn_up_exps=CPU");
        assertTrue(rules.host("blk.1.ffn_up_exps.weight"));
        assertFalse("the GPU rule shields layer 0", rules.host("blk.0.ffn_up_exps.weight"));
    }

    @Test
    public void theFlagShapedKeysAliasTheDottedSettings() {
        EngineConfig dotted = from(EngineConfig.PROP_CPU_MOE, "true", EngineConfig.PROP_N_CPU_MOE, "2",
                EngineConfig.PROP_TENSOR_OVERRIDES, "blk\\.1\\..*=cpu");
        EngineConfig flagged = from(EngineConfig.PROP_CPU_MOE_ALIAS, "true", EngineConfig.PROP_N_CPU_MOE_ALIAS, "2",
                EngineConfig.PROP_OVERRIDE_TENSOR, "blk\\.1\\..*=CPU");
        assertEquals(dotted.cpuMoe(), flagged.cpuMoe());
        assertEquals(dotted.nCpuMoe(), flagged.nCpuMoe());
        assertEquals(dotted.tensorOverrides().get(0).matches("blk.1.ffn_up_exps.weight"),
                flagged.tensorOverrides().get(0).matches("blk.1.ffn_up_exps.weight"));
        assertTrue(flagged.tensorOverrides().get(0).host());
    }
}
