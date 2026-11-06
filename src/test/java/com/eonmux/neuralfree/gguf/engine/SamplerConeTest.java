package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.AllowedTokensStage;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * A truncated sampler chain samples exactly what the whole row samples: {@link Sampler#applyTopK} over the
 * {@link RowTopK} candidates of a row returns the same selection, probabilities and kept set as
 * {@link Sampler#apply} over the row whenever {@link Sampler#coneWidth()} covers them, for seeded chains over
 * random rows.
 */
public class SamplerConeTest {

    private static final int VOCAB = 2000;

    private static SamplerConfig config(long seed) {
        return SamplerConfig.builder().temperature(0.8f).topK(40).topP(0.9f).minP(0.05f).typicalP(1f)
                .penaltyLastN(0).seed(seed).build();
    }

    private static float[] row(long seed) {
        Random r = new Random(seed);
        float[] x = new float[VOCAB];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (r.nextGaussian() * 2.0);
        }
        return x;
    }

    private static void assertSameSelection(Sampler sampler, float[] logits) {
        int cone = sampler.coneWidth();
        RowTopK top = RowTopK.of(new float[][]{logits}, cone);
        Sampler.Candidates whole = sampler.apply(logits.clone());
        Sampler.Candidates coneSet = sampler.applyTopK(top.ids(0), top.logits(0));
        assertEquals("the chain keeps the same candidates", whole.size(), coneSet.size());
        for (int i = 0; i < whole.size(); i++) {
            assertEquals("candidate " + i + " token", whole.id(i), coneSet.id(i));
            assertEquals("candidate " + i + " logit", whole.logit(i), coneSet.logit(i), 0f);
            assertEquals("candidate " + i + " probability", whole.prob(i), coneSet.prob(i), 0f);
        }
        assertEquals("the draw", whole.id(whole.selected()), coneSet.id(coneSet.selected()));
    }

    /** The greedy chain over the cone selects the token the chain selects over the whole row. */
    private static void assertSameToken(Sampler sampler, float[] logits) {
        int cone = sampler.coneWidth();
        assertTrue("the chain carries a cone", cone > 0);
        RowTopK top = RowTopK.of(new float[][]{logits}, cone);
        int whole = sampler.apply(logits.clone()).selectedId();
        int coneId = sampler.applyTopK(top.ids(0), top.logits(0)).selectedId();
        assertEquals("the greedy selection", whole, coneId);
    }

    private static SamplerConfig greedy() {
        return SamplerConfig.builder().greedy(true).topK(0).topP(1f).minP(0f).typicalP(1f).penaltyLastN(0).seed(1)
                .build();
    }

    @Test
    public void theConeCoversTheDefaultChain() {
        SamplerConfig cfg = config(42);
        assertEquals(40, cfg.topK());
        assertEquals(40, Sampler.create(cfg).coneWidth());
    }

    @Test
    public void bannedTokensWidenTheCone() {
        assertEquals(42, Sampler.create(config(7), new int[]{5, 900}).coneWidth());
    }

    @Test
    public void theSeededDrawMatchesOverTheCone() {
        for (long seed = 0; seed < 5; seed++) {
            assertSameSelection(Sampler.create(config(seed)), row(seed));
        }
    }

    @Test
    public void bannedTokensKeepTheDrawTheSame() {
        for (long seed = 0; seed < 5; seed++) {
            SamplerConfig cfg = config(seed);
            int[] banned = {(int) (Math.abs(new Random(seed).nextLong()) % VOCAB), VOCAB - 1};
            assertSameSelection(Sampler.create(cfg, banned), row(seed));
        }
    }

    @Test
    public void aNegativeLogitBiasKeepsTheDrawTheSame() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(17, -5f);
        bias.put(3, -0.5f);
        for (long seed = 0; seed < 5; seed++) {
            assertSameSelection(Sampler.create(config(seed), new int[0], bias), row(seed));
        }
    }

    @Test
    public void penaltiesRefuseTheCone() {
        SamplerConfig cfg = SamplerConfig.builder().temperature(0.8f).topK(40).penaltyLastN(8)
                .repeatPenalty(1.1f).seed(1).build();
        assertEquals(-1, Sampler.create(cfg).coneWidth());
    }

    @Test
    public void aPositiveLogitBiasRefusesTheCone() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(17, 5f);
        assertEquals(-1, Sampler.create(config(1), new int[0], bias).coneWidth());
    }

    @Test
    public void typicalSamplingBeforeTopKRefusesTheCone() {
        SamplerConfig cfg = SamplerConfig.builder().temperature(0.8f).topK(40).typicalP(0.5f).penaltyLastN(0)
                .seed(1)
                .samplers(java.util.List.of(StageKind.TYPICAL_P, StageKind.TOP_K, StageKind.TEMPERATURE))
                .build();
        assertEquals(-1, Sampler.create(cfg).coneWidth());
    }

    @Test
    public void mirostatRefusesTheCone() {
        SamplerConfig cfg = SamplerConfig.builder().mirostat(1).mirostatTau(5f).seed(1).build();
        assertEquals(-1, Sampler.create(cfg).coneWidth());
    }

    @Test
    public void withoutATopKStageRefusesTheCone() {
        SamplerConfig cfg = SamplerConfig.builder().temperature(0.8f).topK(0).topP(0.9f).penaltyLastN(0).seed(1)
                .build();
        assertEquals(-1, Sampler.create(cfg).coneWidth());
    }

    /** The request's masks run from the constraint head over the whole row, so no device cone carries them. */
    @Test
    public void allowedTokenIdsRefuseTheCone() {
        Sampler sampler = Sampler.create(config(1), new int[0],
                List.of(new AllowedTokensStage(new int[]{3, 7, 900})));
        assertTrue(sampler.hasConstraints());
        assertEquals(-1, sampler.coneWidth());
        try {
            sampler.applyTopK(new int[]{900, 7, 3}, new float[]{1f, 0.5f, 0.4f});
            fail("applyTopK over a chain with constraint stages must refuse");
        } catch (IllegalStateException e) {
            assertTrue("the refusal names the masks: " + e.getMessage(),
                    e.getMessage().contains("allowed_token_ids"));
        }
    }

    /** A chain without constraint stages keeps the cone; the flag only reports the constraint head. */
    @Test
    public void aPlainChainCarriesNoConstraints() {
        Sampler sampler = Sampler.create(config(1));
        assertFalse(sampler.hasConstraints());
        assertEquals(40, sampler.coneWidth());
    }

    /** The host path over the same masked chain still runs the head: an empty admitted set is a dead end. */
    @Test
    public void theMaskedHostPathStillRunsTheHead() {
        Sampler sampler = Sampler.create(config(1), new int[0],
                List.of(new AllowedTokensStage(new int[]{3, 7, 900})));
        float[] logits = row(1);
        logits[900] = 100f;
        Sampler.Candidates cur = sampler.apply(logits);
        assertEquals(900, cur.id(cur.selected()));
        Arrays.fill(logits, Float.NEGATIVE_INFINITY);
        try {
            sampler.apply(logits);
            fail("a mask that admits nothing must dead-end");
        } catch (ConstraintDeadEndException e) {
            // the host path's named refusal
        }
    }

    /**
     * A greedy chain without a top-k stage reads back only the prefix that can hold its selection: the raw argmax,
     * plus one candidate per banned or biased token, since those edits only lower a logit (Avalon's
     * {@code greedy_argmax} fast path over a device readback).
     */
    @Test
    public void aGreedyChainWithoutATopKStageReadsItsEditedPrefix() {
        Sampler plain = Sampler.create(greedy());
        assertTrue(plain.argmaxOnly());
        assertEquals("the raw argmax alone", 1, plain.coneWidth());

        Sampler banned = Sampler.create(greedy(), new int[]{5, 900});
        assertFalse(banned.argmaxOnly());
        assertEquals(3, banned.coneWidth());

        Map<Integer, Float> bias = new HashMap<>();
        bias.put(7, -2f);
        assertEquals(2, Sampler.create(greedy(), new int[0], bias).coneWidth());
    }

    @Test
    public void theGreedyPrefixSelectsTheWholeRowsToken() {
        SamplerConfig cfg = greedy();
        for (long seed = 0; seed < 8; seed++) {
            float[] logits = row(seed);
            int top = ForwardBackend.argmax(logits);
            int second = ForwardBackend.argmax(banOne(logits, top));
            // the raw argmax banned, and a ban well behind it
            assertSameToken(Sampler.create(cfg, new int[]{top, 1999 - (int) seed}), logits);
            // the raw argmax pushed below a token the cone also carries
            Map<Integer, Float> bias = new HashMap<>();
            bias.put(top, -3f);
            assertSameToken(Sampler.create(cfg, new int[0], bias), logits);
            // the raw argmax banned and another token lowered
            Map<Integer, Float> mixed = new HashMap<>();
            mixed.put(second, -1f);
            assertSameToken(Sampler.create(cfg, new int[]{top}, mixed), logits);
        }
    }

    /** Every banned token in the prefix pushes the selection one rank down, so the prefix widens with them. */
    @Test
    public void theGreedyPrefixWidensPastEveryBan() {
        float[] logits = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            logits[i] = VOCAB - i;
        }
        Sampler sampler = Sampler.create(greedy(), new int[]{0, 1, 2});
        assertEquals(4, sampler.coneWidth());
        assertSameToken(sampler, logits);
        RowTopK top = RowTopK.of(new float[][]{logits}, sampler.coneWidth());
        assertEquals(3, sampler.applyTopK(top.ids(0), top.logits(0)).selectedId());
    }

    /**
     * A bias that lowers the raw argmax into a tie with a lower-id token keeps the tie the whole-row chain takes
     * (the vocabulary order's lower id), not the readback's rank order.
     */
    @Test
    public void theGreedyPrefixBreaksTiesByTokenId() {
        float[] logits = new float[200];
        logits[100] = 5f;
        logits[50] = 4f;
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(100, -1f);
        Sampler sampler = Sampler.create(greedy(), new int[0], bias);
        assertEquals(2, sampler.coneWidth());
        assertEquals(50, sampler.apply(logits.clone()).selectedId());
        RowTopK top = RowTopK.of(new float[][]{logits}, sampler.coneWidth());
        assertEquals(50, sampler.applyTopK(top.ids(0), top.logits(0)).selectedId());
    }

    /** A positive bias raises a token from anywhere in the row, so the sparse greedy prefix cannot carry it. */
    @Test
    public void aPositiveBiasKeepsTheGreedyChainOffThePrefix() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(7, 2f);
        assertEquals(-1, Sampler.create(greedy(), new int[0], bias).coneWidth());
    }

    @Test
    public void aTopKOfOneSamplesItsBiasedTokenOverThePrefix() {
        SamplerConfig cfg = SamplerConfig.builder().temperature(0.8f).topK(1).penaltyLastN(0).seed(3).build();
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(17, -1f);
        Sampler sampler = Sampler.create(cfg, new int[0], bias);
        assertEquals(2, sampler.coneWidth());
        for (long seed = 0; seed < 5; seed++) {
            assertSameSelection(sampler, row(seed));
        }
    }

    @Test
    public void greedyPenaltiesKeepTheWholeRow() {
        SamplerConfig cfg = SamplerConfig.builder().greedy(true).topK(0).repeatPenalty(1.1f).penaltyLastN(8)
                .seed(1).build();
        assertEquals(-1, Sampler.create(cfg).coneWidth());
    }

    /** {@code logits} with token {@code id} set to {@code -inf}; used to rank the second-best token. */
    private static float[] banOne(float[] logits, int id) {
        float[] copy = logits.clone();
        copy[id] = Float.NEGATIVE_INFINITY;
        return copy;
    }

    @Test
    public void candidateRowsRankByLogitThenId() {
        float[] logits = {1f, 3f, 3f, -1f};
        RowTopK top = RowTopK.of(new float[][]{logits}, 3);
        assertArrayEquals(new int[]{1, 2, 0}, top.ids(0));
        assertArrayEquals(new float[]{3f, 3f, 1f}, top.logits(0), 0f);
    }
}
