package com.eonmux.neuralfree.gguf.engine;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link EngineConfig#legacySampler}: the one mapping of the legacy {@code generate} parameters both orchestrators
 * use. The given values replace the configured defaults' and pass through unchanged (window 0 meaning the whole
 * context); every other setting of the defaults is kept, and out-of-range values are rejected rather than clamped.
 */
public class EngineConfigLegacySamplerTest {

    private static EngineConfig configured() {
        SamplerConfig defaults = SamplerConfig.builder().typicalP(0.9f).presencePenalty(0.5f).frequencyPenalty(0.25f)
                .seed(42L).build();
        return EngineConfig.builder().samplerDefaults(defaults).build();
    }

    @Test
    public void givenValuesReplaceTheDefaultsAndTheRestIsKept() {
        SamplerConfig s = configured().legacySampler(0.7f, 0.9f, 1.1f, 64, 0.05f, 40);
        assertEquals(0.7f, s.temperature(), 0f);
        assertEquals(0.9f, s.topP(), 0f);
        assertEquals(1.1f, s.repeatPenalty(), 0f);
        assertEquals(64, s.penaltyLastN());
        assertEquals(0.05f, s.minP(), 0f);
        assertEquals(40, s.topK());
        assertEquals(0.9f, s.typicalP(), 0f);
        assertEquals(0.5f, s.presencePenalty(), 0f);
        assertEquals(0.25f, s.frequencyPenalty(), 0f);
        assertEquals(42L, s.seed());
    }

    @Test
    public void windowZeroAndMinusOnePenalizeTheWholeContext() {
        EngineConfig c = configured();
        assertEquals(SamplerConfig.PENALTY_WHOLE_CONTEXT, c.legacySampler(0.8f, 1f, 1f, 0, 0f, 0).penaltyLastN());
        assertEquals(SamplerConfig.PENALTY_WHOLE_CONTEXT, c.legacySampler(0.8f, 1f, 1f, -1, 0f, 0).penaltyLastN());
    }

    @Test
    public void penaltiesBelowOneApplyAndConfiguredPenaltiesStayActiveAtRepeatOne() {
        EngineConfig c = configured();
        assertEquals(0.9f, c.legacySampler(0.8f, 1f, 0.9f, 0, 0f, 0).repeatPenalty(), 0f);
        assertTrue(c.legacySampler(0.8f, 1f, 1f, 0, 0f, 0).penaltiesEnabled());
    }

    @Test
    public void greedyDefaultsStayGreedy() {
        EngineConfig c = EngineConfig.builder().samplerDefaults(SamplerConfig.greedyDecoding()).build();
        assertTrue(c.legacySampler(0.8f, 0.95f, 1f, 0, 0.05f, 40).isGreedy());
    }

    @Test
    public void outOfRangeValuesAreRejected() {
        EngineConfig c = EngineConfig.defaults();
        assertThrows(IllegalArgumentException.class, () -> c.legacySampler(0.8f, 1.5f, 1f, 0, 0f, 0));
        assertThrows(IllegalArgumentException.class, () -> c.legacySampler(0.8f, 1f, 1f, -2, 0f, 0));
        assertThrows(IllegalArgumentException.class, () -> c.legacySampler(0.8f, 1f, 0f, 0, 0f, 0));
        assertThrows(IllegalArgumentException.class, () -> c.legacySampler(Float.NaN, 1f, 1f, 0, 0f, 0));
    }
}
