package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.Mt19937;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkAlgorithm;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkConfig;
import com.eonmux.neuralfree.gguf.engine.watermark.WatermarkContextScope;
import com.eonmux.neuralfree.gguf.engine.watermark.Watermarker;
import org.junit.Test;

import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** The {@link Sampler} chain with a watermark: the keyed draw, its fallbacks, greedy bypass and state rollback. */
public class SamplerWatermarkTest {

    private static final long SEED = 42L;
    private static final float[] LOGITS = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f};

    private static SamplerConfig.Builder off() {
        return SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f).temperature(1f).seed(SEED);
    }

    private static WatermarkConfig watermark(int contextWidth) {
        return WatermarkConfig.builder(1234L).contextWidth(contextWidth).build();
    }

    /** The softmax probabilities of a row, in token-id order. */
    private static float[] probabilities(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) {
            max = Math.max(max, v);
        }
        double sum = 0.0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            double e = Math.exp(logits[i] - max);
            probs[i] = (float) e;
            sum += e;
        }
        for (int i = 0; i < probs.length; i++) {
            probs[i] = (float) (probs[i] / sum);
        }
        return probs;
    }

    @Test
    public void theChainSelectsTheWatermarkedTokenAndReportsTheOrdinaryDistribution() {
        WatermarkConfig config = watermark(4);
        Sampler sampler = Sampler.create(off().watermark(config).build());
        int[] prompt = {10, 11, 12, 13, 14, 15};
        sampler.acceptPrompt(prompt, 0, prompt.length);

        Watermarker expected = Watermarker.create(config);
        int expectedToken = expected.sample(LOGITS, new int[]{12, 13, 14, 15}, () -> 0.5);

        Sampler.Candidates candidates = sampler.apply(LOGITS);
        assertEquals(expectedToken, candidates.selectedId());

        float[] probs = probabilities(LOGITS);
        for (int i = 0; i < candidates.size(); i++) {
            assertEquals("probability of token " + candidates.id(i), probs[candidates.id(i)],
                    candidates.prob(i), 1e-6f);
        }
    }

    @Test
    public void greedyDecodingIgnoresTheWatermark() {
        Sampler sampler = Sampler.create(off().greedy(true).watermark(watermark(4)).build());
        sampler.acceptPrompt(new int[]{5, 6, 7, 8}, 0, 4);
        Sampler.Candidates candidates = sampler.apply(LOGITS);
        assertEquals(3, candidates.selectedId());
        assertTrue(sampler.argmaxOnly());
    }

    @Test
    public void aWatermarkedChainIsNeverArgmaxOnly() {
        Sampler sampler = Sampler.create(off().topK(1).watermark(watermark(4)).build());
        assertFalse(sampler.argmaxOnly());
    }

    @Test
    public void aTerminalSelectorChainRefusesTheWatermark() {
        IllegalArgumentException error = org.junit.Assert.assertThrows(IllegalArgumentException.class,
                () -> Sampler.create(off().mirostat(1).watermark(watermark(4)).build()));
        assertTrue(error.getMessage().contains("watermarking"));
    }

    @Test
    public void aRepeatedContextFallsBackToTheOrdinarySeededDraw() {
        WatermarkConfig config = watermark(1);
        Sampler sampler = Sampler.create(off().watermark(config).build());
        sampler.acceptPrompt(new int[]{9}, 0, 1);
        sampler.accept(5);
        int firstWatermarked = sampler.apply(LOGITS).selectedId();

        // the generated scope is now [firstWatermarked, 5], repeating the context [5] of the first generated token
        sampler.accept(firstWatermarked);
        sampler.accept(5);
        Sampler.Candidates repeated = sampler.apply(LOGITS);
        float[] probs = probabilities(LOGITS);
        Mt19937 reference = new Mt19937(SEED);
        assertEquals(reference.distDraw(probs), repeated.selectedId());
    }

    @Test
    public void savingAndRestoringRollsTheWatermarkContextBack() {
        WatermarkConfig config = watermark(2);
        Sampler sampler = Sampler.create(off().watermark(config).build());
        sampler.acceptPrompt(new int[]{4, 5, 6}, 0, 3);
        Object saved = sampler.saveState();
        int before = sampler.apply(LOGITS).selectedId();
        sampler.accept(before);
        sampler.accept(before);
        sampler.restoreState(saved);
        assertEquals(before, sampler.apply(LOGITS).selectedId());
    }

    @Test
    public void theDeviceTopKConeKeepsTheWatermarkedSelection() {
        WatermarkConfig config = watermark(4);
        float[] logits = new float[32];
        SplittableRandom random = new SplittableRandom(7);
        for (int i = 0; i < logits.length; i++) {
            logits[i] = (float) (random.nextDouble() * 4.0);
        }
        int k = 5;
        Sampler full = Sampler.create(off().topK(k).watermark(config).build());
        full.acceptPrompt(new int[]{1, 2, 3, 4}, 0, 4);
        int fullToken = full.apply(logits).selectedId();

        int[] ranked = rank(logits, k);
        float[] rankedLogits = new float[k];
        for (int i = 0; i < k; i++) {
            rankedLogits[i] = logits[ranked[i]];
        }
        Sampler cone = Sampler.create(off().topK(k).watermark(config).build());
        cone.acceptPrompt(new int[]{1, 2, 3, 4}, 0, 4);
        Sampler.Candidates candidates = cone.applyTopK(ranked, rankedLogits);
        assertEquals(fullToken, candidates.selectedId());
    }

    /** The {@code k} largest logits' token ids, in rank order (larger first, lower id on ties). */
    private static int[] rank(float[] logits, int k) {
        Integer[] order = new Integer[logits.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (a, b) -> logits[a] != logits[b]
                ? Float.compare(logits[b], logits[a]) : Integer.compare(a, b));
        int[] ranked = new int[k];
        for (int i = 0; i < k; i++) {
            ranked[i] = order[i];
        }
        return ranked;
    }

    @Test
    public void dualKeyRoutesWithTheSamplersOwnRandomStream() {
        WatermarkConfig config = WatermarkConfig.builder(77L).contextWidth(2)
                .algorithm(WatermarkAlgorithm.DUAL_KEY_GUMBEL).alpha(0.5).build();
        Sampler first = Sampler.create(off().watermark(config).build());
        Sampler second = Sampler.create(off().watermark(config).build());
        int[] prompt = {1, 2, 3, 4};
        first.acceptPrompt(prompt, 0, prompt.length);
        second.acceptPrompt(prompt, 0, prompt.length);
        // the same seed and context route the same way
        assertEquals(first.apply(LOGITS).selectedId(), second.apply(LOGITS).selectedId());
        assertTrue(config.supportsSpeculativeDecoding());
    }

    @Test
    public void allScopeSkipsOnlyThePartialContext() {
        WatermarkConfig config = WatermarkConfig.builder(9L).contextWidth(3)
                .deduplicateContexts(WatermarkContextScope.ALL).build();
        Sampler sampler = Sampler.create(off().watermark(config).build());
        sampler.acceptPrompt(new int[]{1, 2, 3}, 0, 3);
        // with no generated token yet the context is partial, so the draw is ordinary
        Sampler.Candidates partial = sampler.apply(LOGITS);
        float[] probs = probabilities(LOGITS);
        assertEquals(new Mt19937(SEED).distDraw(probs), partial.selectedId());
    }
}
