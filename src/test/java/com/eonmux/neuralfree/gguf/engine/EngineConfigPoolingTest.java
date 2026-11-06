package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineConfig#pooling()}: the {@code neuralfree.pooling} override (llama.cpp {@code --pooling}, vLLM's
 * pooler {@code pooling_type} names), unset by default, refused by property when the name is unknown.
 */
public class EngineConfigPoolingTest {

    private static EngineConfig fromProperty(String value) {
        Properties p = new Properties();
        p.setProperty(EngineConfig.PROP_POOLING, value);
        return EngineConfig.fromProperties(p);
    }

    @Test
    public void unsetFollowsTheGguf() {
        assertNull(EngineConfig.defaults().pooling());
        assertNull(EngineConfig.fromProperties(new Properties()).pooling());
        assertNull(EngineConfig.builder().build().pooling());
    }

    @Test
    public void llamaCppNamesParseIgnoringCase() {
        assertEquals(PoolingKind.NONE, fromProperty("none").pooling());
        assertEquals(PoolingKind.MEAN, fromProperty("mean").pooling());
        assertEquals(PoolingKind.CLS, fromProperty("cls").pooling());
        assertEquals(PoolingKind.LAST, fromProperty("last").pooling());
        assertEquals(PoolingKind.RANK, fromProperty("rank").pooling());
        assertEquals(PoolingKind.CLS, fromProperty(" CLS ").pooling());
    }

    @Test
    public void vllmPoolerNamesParse() {
        assertEquals(PoolingKind.NONE, PoolingKind.parse("ALL"));
        assertEquals(PoolingKind.MEAN, PoolingKind.parse("Mean"));
        assertEquals(PoolingKind.LAST, PoolingKind.parse("last"));
        IllegalArgumentException step = assertThrows(IllegalArgumentException.class,
                () -> PoolingKind.parse("STEP"));
        assertTrue(step.getMessage(), step.getMessage().contains("STEP"));
    }

    @Test
    public void unknownNamesAreRefusedNamingTheProperty() {
        IllegalArgumentException bad = assertThrows(IllegalArgumentException.class, () -> fromProperty("avg"));
        assertTrue(bad.getMessage(), bad.getMessage().contains(EngineConfig.PROP_POOLING));
        assertTrue(bad.getMessage(), bad.getMessage().contains("mean"));
        IllegalArgumentException flagValue = assertThrows(IllegalArgumentException.class,
                () -> PoolingKind.parse("avg"));
        assertTrue(flagValue.getMessage(), flagValue.getMessage().contains("avg"));
    }

    @Test
    public void builderRoundTripsAndClears() {
        EngineConfig c = EngineConfig.builder().pooling(PoolingKind.LAST).build();
        assertEquals(PoolingKind.LAST, c.pooling());
        assertEquals(PoolingKind.LAST, c.toBuilder().build().pooling());
        assertNull(c.toBuilder().pooling((PoolingKind) null).build().pooling());
        assertTrue(c.toString().contains("pooling=last"));
    }

    @Test
    public void toStringNamesTheOverride() {
        assertTrue(EngineConfig.builder().pooling(PoolingKind.CLS).build().toString().contains("pooling=cls"));
    }
}
