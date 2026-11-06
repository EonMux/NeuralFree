package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The control-vector settings of {@link EngineConfig} ({@code --control-vector}, {@code --control-vector-scaled}
 * and {@code --control-vector-layer-range} as properties): the {@code path[:scale]} list, the {@code START:END}
 * layer range with the llama.cpp defaults for missing sides, the refusal of invalid values naming the property, and
 * the builder round trip.
 */
public class EngineConfigControlVectorTest {

    @Test
    public void thePropertyHoldsAPathScaleList() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_CONTROL_VECTOR, " /a/v1.gguf:2.5, /b/v2.gguf ,,/c/v3.gguf:0.25");
        List<ControlVectorSpec> v = EngineConfig.fromProperties(p).controlVectors();
        assertEquals(3, v.size());
        assertEquals("/a/v1.gguf", v.get(0).path());
        assertEquals(2.5f, v.get(0).scale(), 0f);
        assertEquals("/b/v2.gguf", v.get(1).path());
        assertEquals(1.0f, v.get(1).scale(), 0f);
        assertEquals("/c/v3.gguf", v.get(2).path());
        assertEquals(0.25f, v.get(2).scale(), 0f);
    }

    @Test
    public void aWindowsStylePathWithoutAScaleKeepsItsColon() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_CONTROL_VECTOR, "C:\\vectors\\down.gguf");
        assertEquals("C:\\vectors\\down.gguf", EngineConfig.fromProperties(p).controlVectors().get(0).path());
    }

    @Test
    public void theLayerRangePropertyReadsBothSidesAndDefaultsTheMissingOnes() {
        assertEquals(0, EngineConfig.fromProperties(new Properties()).controlVectorLayerStart());
        assertEquals(0, EngineConfig.fromProperties(new Properties()).controlVectorLayerEnd());
        Properties both = new Properties();
        both.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "3:7");
        assertEquals(3, EngineConfig.fromProperties(both).controlVectorLayerStart());
        assertEquals(7, EngineConfig.fromProperties(both).controlVectorLayerEnd());
        Properties endOnly = new Properties();
        endOnly.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, ":7");
        assertEquals(0, EngineConfig.fromProperties(endOnly).controlVectorLayerStart());
        assertEquals(7, EngineConfig.fromProperties(endOnly).controlVectorLayerEnd());
        Properties startOnly = new Properties();
        startOnly.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "5:");
        assertEquals(5, EngineConfig.fromProperties(startOnly).controlVectorLayerStart());
        assertEquals(0, EngineConfig.fromProperties(startOnly).controlVectorLayerEnd());
        Properties lone = new Properties();
        lone.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "5");
        assertEquals(5, EngineConfig.fromProperties(lone).controlVectorLayerStart());
        assertEquals(0, EngineConfig.fromProperties(lone).controlVectorLayerEnd());
    }

    @Test
    public void invalidValuesAreRefusedNamingTheProperty() {
        Properties badRange = new Properties();
        badRange.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "3:7:9");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(badRange));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE));

        Properties badLayer = new Properties();
        badLayer.setProperty(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE, "two:3");
        IllegalArgumentException f = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(badLayer));
        assertTrue(f.getMessage(), f.getMessage().contains(EngineConfig.PROP_CONTROL_VECTOR_LAYER_RANGE));

        assertThrows(IllegalArgumentException.class,
                () -> new ControlVectorSpec(" ", 1.0f));
        assertThrows(IllegalArgumentException.class,
                () -> new ControlVectorSpec("v.gguf", Float.POSITIVE_INFINITY));
    }

    @Test
    public void theBuilderCarriesTheSettingsThroughToBuilderRoundTrips() {
        EngineConfig cfg = EngineConfig.builder()
                .addControlVector("v.gguf", 2.0f)
                .controlVectorLayerRange(2, 5)
                .build();
        assertEquals(1, cfg.controlVectors().size());
        assertEquals("v.gguf", cfg.controlVectors().get(0).path());
        assertEquals(2.0f, cfg.controlVectors().get(0).scale(), 0f);
        assertEquals(2, cfg.controlVectorLayerStart());
        assertEquals(5, cfg.controlVectorLayerEnd());

        EngineConfig copy = cfg.toBuilder().build();
        assertEquals(cfg.controlVectors(), copy.controlVectors());
        assertEquals(cfg.controlVectorLayerStart(), copy.controlVectorLayerStart());
        assertEquals(cfg.controlVectorLayerEnd(), copy.controlVectorLayerEnd());
        assertTrue(cfg.toString().contains("controlVector="));
        assertTrue(cfg.toString().contains("controlVectorRange=2:5"));
        assertTrue(EngineConfig.defaults().controlVectors().isEmpty());
    }
}
