package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.lora.LoraConfig;
import org.junit.Test;

import java.util.Properties;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The registry settings of {@link EngineConfig} ({@code neuralfree.lora.max}, {@code neuralfree.lora.max.rank}
 * and {@code neuralfree.lora.cache} as properties, vLLM's {@code --max-loras}, {@code --max-lora-rank} and
 * {@code --max-cpu-loras}): the property wiring into {@link EngineConfig#loraConfig()}, the defaults when no key
 * is set, the refusal of an invalid value naming the property, and the builder round trip.
 */
public class EngineConfigLoraConfigTest {

    @Test
    public void theDefaultsHoldWithoutAnyProperty() {
        assertEquals(LoraConfig.defaults(), EngineConfig.defaults().loraConfig());
        assertEquals(LoraConfig.defaults(), EngineConfig.fromProperties(new Properties()).loraConfig());
    }

    @Test
    public void theRegistryPropertiesFillTheConfig() {
        Properties p = new Properties();
        p.setProperty(LoraConfig.PROP_MAX, "2");
        p.setProperty(LoraConfig.PROP_MAX_RANK, "16");
        p.setProperty(LoraConfig.PROP_CACHE, "0");
        LoraConfig lora = EngineConfig.fromProperties(p).loraConfig();
        assertEquals(2, lora.maxAdapters());
        assertEquals(16, lora.maxRank());
        assertEquals(0, lora.cacheEntries());
    }

    @Test
    public void anInvalidRegistryValueIsRefusedNamingTheProperty() {
        Properties bad = new Properties();
        bad.setProperty(LoraConfig.PROP_MAX, "many");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> EngineConfig.fromProperties(bad));
        assertTrue(e.getMessage(), e.getMessage().contains(LoraConfig.PROP_MAX));
    }

    @Test
    public void theBuilderCarriesTheSettingsThroughToBuilderRoundTrips() {
        EngineConfig cfg = EngineConfig.builder()
                .loraConfig(LoraConfig.builder().maxAdapters(3).maxRank(8).cacheEntries(5).build())
                .build();
        assertEquals(3, cfg.loraConfig().maxAdapters());
        assertEquals(8, cfg.loraConfig().maxRank());
        assertEquals(5, cfg.loraConfig().cacheEntries());

        EngineConfig copy = cfg.toBuilder().build();
        assertEquals(cfg.loraConfig(), copy.loraConfig());
        assertEquals(LoraConfig.defaults(), EngineConfig.builder().build().loraConfig());
    }
}
