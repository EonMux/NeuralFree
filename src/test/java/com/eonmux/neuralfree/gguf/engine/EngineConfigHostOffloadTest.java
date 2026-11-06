package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The host-mapped (UVA) offload and selective-streaming settings of {@link EngineConfig}: the properties parse
 * into the configuration, the parameter segments keep their order, and contradictory or malformed values are
 * refused by the property they came from.
 */
public class EngineConfigHostOffloadTest {

    @Test
    public void theOffloadIsOffByDefault() {
        EngineConfig c = EngineConfig.defaults();

        assertEquals(0L, c.cpuOffloadMb());
        assertEquals(0L, c.cpuOffloadBytes());
        assertTrue(c.cpuOffloadParams().isEmpty());
        assertTrue(c.streamParams().isEmpty());
    }

    @Test
    public void thePropertiesParseIntoTheConfiguration() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_CPU_OFFLOAD_MB, "512");
        p.setProperty(EngineConfig.PROP_CPU_OFFLOAD_PARAMS, "experts, ffn_gate_exps");
        p.setProperty(EngineConfig.PROP_STREAM_PARAMS, "ffn_up_exps");

        EngineConfig c = EngineConfig.fromProperties(p);

        assertEquals(512L, c.cpuOffloadMb());
        assertEquals(512L * 1024 * 1024, c.cpuOffloadBytes());
        assertEquals(Arrays.asList("experts", "ffn_gate_exps"), c.cpuOffloadParams());
        assertEquals(Collections.singletonList("ffn_up_exps"), c.streamParams());
    }

    @Test
    public void aToBuilderRoundTripKeepsBothSelections() {
        EngineConfig c = EngineConfig.builder()
                .cpuOffloadMb(64)
                .cpuOffloadParams(Collections.singletonList("ffn_gate_exps"))
                .streamParams(Collections.singletonList("experts"))
                .build();

        EngineConfig copy = c.toBuilder().build();

        assertEquals(64L, copy.cpuOffloadMb());
        assertEquals(Collections.singletonList("ffn_gate_exps"), copy.cpuOffloadParams());
        assertEquals(Collections.singletonList("experts"), copy.streamParams());
    }

    @Test
    public void offloadParametersWithoutABudgetAreRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .cpuOffloadParams(Collections.singletonList("experts"))
                .build());

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_PARAMS));
        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_MB));
    }

    @Test
    public void aNegativeBudgetIsRefused() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.builder().cpuOffloadMb(-1).build());

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_MB));
    }

    @Test
    public void aBlankOrPartlyEmptySegmentIsRefusedByName() {
        IllegalArgumentException blank = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .cpuOffloadMb(1)
                .streamParams(Collections.singletonList(" ")));
        assertTrue(blank.getMessage(), blank.getMessage().contains(EngineConfig.PROP_STREAM_PARAMS));

        IllegalArgumentException emptyPart = assertThrows(IllegalArgumentException.class, () -> EngineConfig.builder()
                .cpuOffloadMb(1)
                .cpuOffloadParams(Collections.singletonList("experts..weight")));
        assertTrue(emptyPart.getMessage(), emptyPart.getMessage().contains(EngineConfig.PROP_CPU_OFFLOAD_PARAMS));
        assertTrue(emptyPart.getMessage(), emptyPart.getMessage().contains("empty dotted part"));
    }

    @Test
    public void anEmptyListEntryIsRefusedByThePropertyReader() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_STREAM_PARAMS, "experts,,ffn_gate_exps");

        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));

        assertTrue(e.getMessage(), e.getMessage().contains(EngineConfig.PROP_STREAM_PARAMS));
        assertTrue(e.getMessage(), e.getMessage().contains("empty entry"));
    }
}
