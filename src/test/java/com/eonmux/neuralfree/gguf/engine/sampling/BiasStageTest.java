package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import org.junit.Test;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link BiasStage} with additive biases (llama.cpp {@code llama_sampler_init_logit_bias} + the request's
 * {@code logit_bias}) through {@link Sampler#create(SamplerConfig, int[], Map)}: the bias lands at the chain's
 * first stage — a candidate's index is still its token id there — so a positive bias lifts the token ahead of
 * the peaks under the same seed, and a bias of {@code -100} or below removes the token like a ban.
 */
public class BiasStageTest {

    /** Flat field, no sort-reducing stage: temperature 1, everything else off. */
    private static SamplerConfig seeded() {
        return SamplerConfig.builder().temperature(1f).topK(0).topP(1f).minP(0f).typicalP(1f)
                .penaltyLastN(0).seed(11L).build();
    }

    /** A flat logit row with three moderate peaks; id 0 sits mid-field. */
    private static float[] row() {
        float[] row = new float[32];
        row[5] = 2.0f;
        row[9] = 1.8f;
        row[14] = 1.5f;
        return row;
    }

    @Test
    public void theBiasLandsAtTheFirstStageAndMinusOneHundredBans() {
        Map<Integer, Float> bias = new LinkedHashMap<>();
        bias.put(0, 1.0f);
        bias.put(1, -100f);
        Sampler.Candidates c = Sampler.create(seeded(), new int[]{31}, bias).apply(row());
        // the set is unsorted (no sort stage), so the candidate index is still the token id
        assertEquals("the bias adds to the flat logit", row()[0] + 1.0f, c.logit(0), 0f);
        assertEquals("the -100 bias bans", Float.NEGATIVE_INFINITY, c.logit(1), 0f);
        assertEquals("the explicit ban still bans", Float.NEGATIVE_INFINITY, c.logit(31), 0f);
        assertEquals("untouched ids keep their logit", row()[5], c.logit(5), 0f);
    }

    @Test
    public void aBiasMovesTheSeededDraw() {
        Map<Integer, Float> bias = new LinkedHashMap<>();
        bias.put(0, 8f);
        Sampler biased = Sampler.create(seeded(), new int[0], bias);
        assertEquals("the biased token wins the draw", 0, biased.sample(row()));

        Sampler plain = Sampler.create(seeded(), new int[0]);
        assertNotEquals("the unbiased chain does not draw the flat token", 0, plain.sample(row()));
    }

    @Test
    public void aMinusOneHundredBiasRemovesTheTokenFromATopKChain() {
        SamplerConfig topTwo = SamplerConfig.builder().temperature(1f).topK(2).topP(1f).minP(0f).typicalP(1f)
                .penaltyLastN(0).seed(11L).build();
        Map<Integer, Float> ban = new LinkedHashMap<>();
        ban.put(9, -100f);
        Sampler.Candidates c = Sampler.create(topTwo, new int[0], ban).apply(row());
        assertEquals(2, c.size());
        for (int i = 0; i < c.size(); i++) {
            assertTrue("the banned token left the chain", c.id(i) != 9);
        }
    }

    @Test
    public void theStageAndTheRequestValidateTheirArguments() {
        BiasStage stage = new BiasStage(new int[]{31}, new int[]{0, 1}, new float[]{1.0f, -100f});
        assertArrayEquals(new int[]{31}, stage.banned());
        assertArrayEquals(new int[]{0, 1}, stage.biasedIds());
        assertArrayEquals(new float[]{1.0f, -100f}, stage.biases(), 0f);
        assertThrows(IllegalArgumentException.class,
                () -> new BiasStage(new int[]{-1}, new int[0], new float[0]));
        assertThrows(IllegalArgumentException.class,
                () -> new BiasStage(new int[0], new int[]{0}, new float[]{Float.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> new BiasStage(new int[0], new int[1], new float[0]));

        assertThrows(IllegalArgumentException.class,
                () -> Sampler.create(seeded(), new int[0], Map.of(-1, 1f)));
        assertThrows(NullPointerException.class,
                () -> Sampler.create(seeded(), new int[0], Map.of(0, null)));
    }
}
