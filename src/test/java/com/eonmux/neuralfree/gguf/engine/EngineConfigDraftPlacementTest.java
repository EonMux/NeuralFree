package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The draft model's own placement settings of {@link EngineConfig}: {@code neuralfree.draft.gpu.devices}
 * (llama.cpp {@code -devd}, {@code none} alone keeps the draft on the host), {@code neuralfree.draft.n.gpu.layers}
 * ({@code -ngld}), {@code neuralfree.draft.override.tensor} ({@code -otd}), {@code neuralfree.draft.cpu.moe}
 * ({@code -cmoed}) and {@code neuralfree.draft.n.cpu.moe} ({@code -ncmoed}) parse with their defaults and refuse
 * malformed values naming the property. They apply to the load of a separate draft model only, so without
 * {@code neuralfree.draft} the builder refuses them naming the properties, and {@code none} beside devices is
 * refused.
 */
public class EngineConfigDraftPlacementTest {

    private static EngineConfig from(String... kv) {
        Properties p = new Properties();
        for (int i = 0; i + 1 < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return EngineConfig.fromProperties(p);
    }

    private static EngineConfig withDraft(String... kv) {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_DRAFT, "draft.gguf");
        for (int i = 0; i + 1 < kv.length; i += 2) {
            p.setProperty(kv[i], kv[i + 1]);
        }
        return EngineConfig.fromProperties(p);
    }

    @Test
    public void draftPlacementDefaultsToTheTargetsPlacement() {
        EngineConfig c = from();
        assertEquals(EngineConfig.GPU_LAYERS_AUTO, c.draftGpuLayers());
        assertTrue(c.draftTensorOverrides().isEmpty());
        assertFalse(c.draftCpuMoe());
        assertEquals(EngineConfig.N_CPU_LAYERS_UNSET, c.draftNCpuMoe());
        assertTrue(c.draftDevices().isEmpty());
        assertFalse(c.hasDraftPlacement());
    }

    @Test
    public void draftPlacementSettingsParse() {
        EngineConfig c = withDraft(EngineConfig.PROP_DRAFT_GPU_DEVICES, "0, A100",
                EngineConfig.PROP_DRAFT_N_GPU_LAYERS, "all",
                EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES, "exps=cpu, blk\\.3\\..*=gpu",
                EngineConfig.PROP_DRAFT_CPU_MOE, "true",
                EngineConfig.PROP_DRAFT_N_CPU_MOE, "2");
        assertEquals(List.of("0", "A100"), c.draftDevices());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, c.draftGpuLayers());
        assertEquals(2, c.draftTensorOverrides().size());
        assertTrue(c.draftTensorOverrides().get(0).host());
        assertFalse(c.draftTensorOverrides().get(1).host());
        assertTrue(c.draftCpuMoe());
        assertEquals(2, c.draftNCpuMoe());
        assertTrue(c.hasDraftPlacement());
        assertTrue("the settings survive a toBuilder round trip", c.toBuilder().build().hasDraftPlacement());
        assertEquals(EngineConfig.GPU_LAYERS_ALL, c.toBuilder().build().draftGpuLayers());
        assertEquals(List.of("0", "A100"), c.toBuilder().build().draftDevices());
    }

    @Test
    public void noneAloneKeepsTheDraftOnTheHost() {
        EngineConfig c = withDraft(EngineConfig.PROP_DRAFT_GPU_DEVICES, " none ");
        assertEquals(List.of(EngineConfig.DRAFT_NO_DEVICE), c.draftDevices());
        assertTrue(c.hasDraftPlacement());
    }

    @Test
    public void malformedValuesFailNamingTheProperty() {
        IllegalArgumentException devices = assertThrows(IllegalArgumentException.class,
                () -> withDraft(EngineConfig.PROP_DRAFT_GPU_DEVICES, "0,,1"));
        assertTrue(devices.getMessage(), devices.getMessage().contains(EngineConfig.PROP_DRAFT_GPU_DEVICES));
        IllegalArgumentException ngl = assertThrows(IllegalArgumentException.class,
                () -> withDraft(EngineConfig.PROP_DRAFT_N_GPU_LAYERS, "many"));
        assertTrue(ngl.getMessage(), ngl.getMessage().contains(EngineConfig.PROP_DRAFT_N_GPU_LAYERS));
        IllegalArgumentException rules = assertThrows(IllegalArgumentException.class,
                () -> withDraft(EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES, "exps"));
        assertTrue(rules.getMessage(), rules.getMessage().contains(EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES));
        IllegalArgumentException moe = assertThrows(IllegalArgumentException.class,
                () -> withDraft(EngineConfig.PROP_DRAFT_CPU_MOE, "perhaps"));
        assertTrue(moe.getMessage(), moe.getMessage().contains(EngineConfig.PROP_DRAFT_CPU_MOE));
        IllegalArgumentException nMoe = assertThrows(IllegalArgumentException.class,
                () -> withDraft(EngineConfig.PROP_DRAFT_N_CPU_MOE, "-1"));
        assertTrue(nMoe.getMessage(), nMoe.getMessage().contains(EngineConfig.PROP_DRAFT_N_CPU_MOE));
    }

    @Test
    public void noneBesideDevicesIsRefused() {
        IllegalArgumentException refusal = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().draftModelPath("draft.gguf").addDraftDevice("none").addDraftDevice("0")
                        .build());
        assertTrue(refusal.getMessage(), refusal.getMessage().contains(EngineConfig.PROP_DRAFT_GPU_DEVICES));
        assertTrue(refusal.getMessage(), refusal.getMessage().contains(EngineConfig.DRAFT_NO_DEVICE));
    }

    @Test
    public void aNegativeDraftLayerCountClearsTheBuilderValue() {
        EngineConfig c = EngineConfig.builder().draftModelPath("d.gguf").draftNCpuMoe(3).draftNCpuMoe(-1).build();
        assertEquals(EngineConfig.N_CPU_LAYERS_UNSET, c.draftNCpuMoe());
    }

    @Test
    public void draftPlacementWithoutADraftModelIsRefusedNamingTheSettings() {
        for (String[] kv : new String[][] {
                {EngineConfig.PROP_DRAFT_N_GPU_LAYERS, "8"},
                {EngineConfig.PROP_DRAFT_TENSOR_OVERRIDES, "exps=cpu"},
                {EngineConfig.PROP_DRAFT_CPU_MOE, "true"},
                {EngineConfig.PROP_DRAFT_N_CPU_MOE, "1"},
                {EngineConfig.PROP_DRAFT_GPU_DEVICES, "none"},
                {EngineConfig.PROP_DRAFT_GPU_DEVICES, "0,1"}}) {
            IllegalArgumentException refusal = assertThrows(IllegalArgumentException.class, () -> from(kv[0], kv[1]));
            assertTrue(refusal.getMessage(), refusal.getMessage().contains(kv[0]));
            assertTrue(refusal.getMessage(), refusal.getMessage().contains(EngineConfig.PROP_DRAFT));
        }
    }
}
