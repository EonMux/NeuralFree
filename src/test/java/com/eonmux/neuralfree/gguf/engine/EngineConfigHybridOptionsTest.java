package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Parsing and defaults of the hybrid CPU+GPU load options: {@code -ngl}, {@code -ot} and the MoE expert settings. */
public class EngineConfigHybridOptionsTest {

    @Test
    public void gpuLayersDefaultsToAuto() {
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, EngineConfig.defaults().gpuLayers());
    }

    @Test
    public void gpuLayersParsesAutoAllAndCounts() {
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, parse(EngineConfig.PROP_N_GPU_LAYERS, "auto"));
        assertEquals(EngineConfig.GPU_LAYERS_ALL, parse(EngineConfig.PROP_N_GPU_LAYERS, "all"));
        assertEquals(EngineConfig.GPU_LAYERS_ALL, parse(EngineConfig.PROP_N_GPU_LAYERS, "ALL"));
        assertEquals(5, parse(EngineConfig.PROP_N_GPU_LAYERS, "5"));
        assertEquals(0, parse(EngineConfig.PROP_N_GPU_LAYERS, "0"));
    }

    @Test
    public void gpuLayersRefusesOtherValues() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> parse(EngineConfig.PROP_N_GPU_LAYERS, "-3"));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_N_GPU_LAYERS));
        assertThrows(IllegalArgumentException.class, () -> parse(EngineConfig.PROP_N_GPU_LAYERS, "many"));
    }

    @Test
    public void tensorOverridesParseEntriesInOrder() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_OVERRIDES, "blk\\.[012]\\.=cpu , attn_k=GPU, output.*=opencl");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertEquals(3, cfg.tensorOverrides().size());
        assertTrue(cfg.tensorOverrides().get(0).host());
        assertTrue(cfg.tensorOverrides().get(0).matches("blk.1.attn_norm.weight"));
        assertFalse(cfg.tensorOverrides().get(0).matches("blk.3.attn_norm.weight"));
        assertFalse(cfg.tensorOverrides().get(1).host());
        assertTrue(cfg.tensorOverrides().get(1).matches("model.layers.4.self_attn.attn_k.weight"));
        assertEquals("gpu", cfg.tensorOverrides().get(1).target());
        assertFalse(cfg.tensorOverrides().get(2).host());
        assertEquals("opencl", cfg.tensorOverrides().get(2).target());
    }

    @Test
    public void tensorOverridesRefuseMalformedEntriesByName() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parseList("blk.1.=video"));
        assertTrue(e.getMessage().contains("video"));
        e = assertThrows(IllegalArgumentException.class, () -> parseList("blk.1."));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_TENSOR_OVERRIDES));
        e = assertThrows(IllegalArgumentException.class, () -> parseList("blk.[1.=cpu"));
        assertTrue(e.getMessage().contains("regular expression"));
        e = assertThrows(IllegalArgumentException.class, () -> parseList("=cpu"));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_TENSOR_OVERRIDES));
    }

    @Test
    public void moeExpertSettingsParseAndValidate() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_CPU_MOE, "true");
        p.setProperty(EngineConfig.PROP_N_CPU_MOE, "4");
        EngineConfig cfg = EngineConfig.fromProperties(p);
        assertTrue(cfg.cpuMoe());
        assertEquals(4, cfg.nCpuMoe());
        assertFalse(EngineConfig.defaults().cpuMoe());
        assertEquals(0, EngineConfig.defaults().nCpuMoe());
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> build(EngineConfig.PROP_N_CPU_MOE, "-1"));
        assertTrue(e.getMessage().contains(EngineConfig.PROP_N_CPU_MOE));
    }

    @Test
    public void builderRoundTripKeepsTheOptions() {
        EngineConfig cfg = EngineConfig.builder()
                .gpuLayers(7)
                .addTensorOverride("blk\\.[01]\\.", "cpu")
                .cpuMoe(true)
                .nCpuMoe(3)
                .build();
        EngineConfig copy = cfg.toBuilder().build();
        assertEquals(7, copy.gpuLayers());
        assertEquals(cfg.tensorOverrides(), copy.tensorOverrides());
        assertTrue(copy.cpuMoe());
        assertEquals(3, copy.nCpuMoe());
        assertTrue(cfg.toString().contains("gpuLayers=7"));
        assertTrue(cfg.toString().contains("cpuMoe"));
    }

    private static int parse(String key, String value) {
        return build(key, value).gpuLayers();
    }

    private static EngineConfig build(String key, String value) {
        Properties p = new Properties();
        p.setProperty(key, value);
        return EngineConfig.fromProperties(p);
    }

    private static EngineConfig parseList(String value) {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_TENSOR_OVERRIDES, value);
        return EngineConfig.fromProperties(p);
    }
}
