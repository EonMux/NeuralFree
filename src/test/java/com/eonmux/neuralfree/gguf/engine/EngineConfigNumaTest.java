package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.cpu.numa.NumaStrategy;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineConfig}'s NUMA strategy: the property parser, the builder default, the round trip and the report.
 */
public class EngineConfigNumaTest {

    @Test
    public void thePropertySelectsTheStrategy() {
        Properties p = new Properties();
        assertEquals(NumaStrategy.DISABLED, EngineConfig.fromProperties(p).numaStrategy());
        p.setProperty(EngineConfig.PROP_NUMA, "distribute");
        assertEquals(NumaStrategy.DISTRIBUTE, EngineConfig.fromProperties(p).numaStrategy());
        p.setProperty(EngineConfig.PROP_NUMA, "NUMACTL");
        assertEquals(NumaStrategy.NUMACTL, EngineConfig.fromProperties(p).numaStrategy());
    }

    @Test
    public void anInvalidStrategyIsRefusedNamingTheProperty() {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_NUMA, "mirror");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(p));
        assertTrue(e.getMessage(), e.getMessage().startsWith(EngineConfig.PROP_NUMA + "=mirror"));
    }

    @Test
    public void theBuilderRoundTripsAndReportsTheStrategy() {
        EngineConfig config = EngineConfig.builder().numa(NumaStrategy.ISOLATE).build();
        assertEquals(NumaStrategy.ISOLATE, config.numaStrategy());
        assertEquals(NumaStrategy.ISOLATE, config.toBuilder().build().numaStrategy());
        assertTrue(config.toString(), config.toString().contains("numa=isolate"));
        assertFalse(EngineConfig.defaults().toString().contains("numa="));
        assertEquals("a null strategy is the default", NumaStrategy.DISABLED,
                EngineConfig.builder().numa(null).build().numaStrategy());
    }
}
