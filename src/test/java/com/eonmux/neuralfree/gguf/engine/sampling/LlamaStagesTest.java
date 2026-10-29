package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The llama.cpp sampler stages SA-2 adds, each as the only configured entry of the {@link Sampler} chain. Every
 * expectation is hand-derived from the llama.cpp code paths:
 *
 * <ul>
 *   <li>mt19937 seed 42: the first {@code generate_canonical&lt;float, 24&gt;} draw is
 *       {@code 1608637542 / 2^32 = 0.37454012} (XTC's chance, llama_sample_xtc_apply); the
 *       {@code generate_canonical&lt;double, 53&gt;} pairs are {@code 0.79654298}, {@code 0.18343479} and
 *       {@code 0.77969100} (the llama_sample_dist targets, src/llama-sampler.cpp:217).</li>
 *   <li>The softmax of the row {@code {2, 1, 0.5, 3, -1}} is
 *       {@code [0.22941, 0.08439, 0.05119, 0.62359, 0.01142]} for ids 0, 1, 2, 3, 4
 *       (sum exp(l - 3) = 1.6035937).</li>
 * </ul>
 */
public class LlamaStagesTest {

    private static final float[] LOGITS = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f};
    private static final long SEED = 42L;

    /** The llama.cpp defaults with every stage the chain inherits off. */
    private static SamplerConfig.Builder base() {
        return SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f).temperature(1f).seed(SEED);
    }

    private static List<String> labels(List<StageKind> order) {
        List<String> out = new ArrayList<>();
        for (StageKind kind : order) {
            out.add(kind.label());
        }
        return out;
    }

    private static int[] ids(Sampler.Candidates c) {
        int[] out = new int[c.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = c.id(i);
        }
        return out;
    }

    // ------------------------------------------------------------------ top-n-sigma

    private static Sampler topNSigmaSampler(float n) {
        return Sampler.create(base().greedy(true).samplers("top_n_sigma").topNSigma(n).build());
    }

    @Test
    public void topNSigmaMasksTheTailBelowMaxMinusNStdDev() {
        // mean 1.1, population std sqrt(9.2/5) = 1.356466: n = 0.5 cuts below 3 - 0.5*1.356466 = 2.32177,
        // so only the maximum 3.0 survives (llama_sampler_top_n_sigma_apply).
        Sampler.Candidates c = apply(topNSigmaSampler(0.5f));
        assertEquals(5, c.size());
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(0), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(1), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(2), 0f);
        assertEquals(3.0f, c.logit(3), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(4), 0f);
        assertEquals(3, c.selectedId());
        assertFalse(c.sorted());

        // n = 2 cuts below 3 - 2*1.356466 = 0.28707: the 0.5 survives with the 1.0, 2.0 and the maximum, and
        // only the -1.0 is masked.
        Sampler.Candidates wide = apply(topNSigmaSampler(2f));
        assertEquals(Float.NEGATIVE_INFINITY, wide.logit(4), 0f);
        assertEquals(1.0f, wide.logit(1), 0f);
        assertEquals(2.0f, wide.logit(0), 0f);
        assertEquals(0.5f, wide.logit(2), 0f);
        assertEquals(3, wide.selectedId());
    }

    @Test
    public void topNSigmaIgnoresAlreadyMaskedLogits() {
        // {3, 2, -inf, 4}: the mean is over the three finite logits (3), the std is 1, so n = 0.5 cuts below 3.5;
        // the -inf stays -inf and never enters the sums.
        float[] row = {3.0f, 2.0f, Float.NEGATIVE_INFINITY, 4.0f};
        Sampler.Candidates c = apply(topNSigmaSampler(0.5f), row);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(0), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(1), 0f);
        assertEquals(Float.NEGATIVE_INFINITY, c.logit(2), 0f);
        assertEquals(4.0f, c.logit(3), 0f);
        assertEquals(3, c.selectedId());
    }

    @Test
    public void topNSigmaZeroAppliesNothing() {
        Sampler.Candidates c = apply(topNSigmaSampler(0f));
        for (int i = 0; i < LOGITS.length; i++) {
            assertEquals(LOGITS[i], c.logit(i), 0f);
        }
        assertEquals(3, c.selectedId());
        assertTrue(Sampler.create(base().greedy(true).samplers("top_n_sigma").build()).argmaxOnly());
        assertFalse(Sampler.create(base().greedy(true).samplers("top_n_sigma").topNSigma(1f).build()).argmaxOnly());
    }

    // ------------------------------------------------------------------ xtc

    /**
     * XTC with seed 42: the chance is 0.37454012, so probability 0.1 misses the trim and 0.5 / 0.9 trim; the chain
     * is {@code xtc} alone under greedy decoding.
     */
    private static Sampler xtcSampler(float probability, float threshold, int minKeep) {
        SamplerConfig config = base().seed(SEED).greedy(true).minKeep(minKeep).samplers("xtc")
                .xtcProbability(probability).xtcThreshold(threshold).build();
        return Sampler.create(config);
    }

    @Test
    public void xtcTrimsCandidatesOverTheThreshold() {
        // Softmax sorted: [0.62359 id3, 0.22941 id0, 0.08439 id1, 0.05119 id2, 0.01142 id4]. Threshold 0.2 keeps
        // ids 3 and 0 (pos_last = 2 of 0..1), so the first two are removed:
        // llama.cpp does data += pos_last, size -= pos_last after llama_sample_xtc_apply's threshold scan.
        Sampler.Candidates c = apply(xtcSampler(0.9f, 0.2f, 0));
        assertArrayEquals(new int[]{0, 1, 2, 4}, ids(c));
        assertEquals(0.22941f, c.prob(0), 1e-4f);
        assertEquals(0.08439f, c.prob(1), 1e-4f);
        assertEquals(0.05119f, c.prob(2), 1e-4f);
        assertEquals(0.01142f, c.prob(3), 1e-4f);
    }

    @Test
    public void xtcTrimsNothingWhenOnlyTheTopIsOverTheThreshold() {
        // Threshold 0.3: only the top candidate 0.62359 is over it, pos_last = 0, so nothing is removed.
        Sampler.Candidates c = apply(xtcSampler(0.9f, 0.3f, 0), LOGITS);
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(c));
        assertEquals(0.62359f, c.prob(0), 1e-4f);
        assertEquals(0.01142f, c.prob(4), 1e-4f);
    }

    @Test
    public void xtcLowerThresholdKeepsMoreOfTheSmallMass() {
        // Threshold 0.1: the scan advances while the sorted probability reaches it, so id0 (0.22941) is inside
        // the trim and id1 (0.08439) stops it: pos_last = 1, and the front candidate id3 is removed.
        Sampler.Candidates c = apply(xtcSampler(0.9f, 0.1f, 0));
        assertArrayEquals(new int[]{0, 1, 2, 4}, ids(c));
        assertEquals(0.22941f, c.prob(0), 1e-4f);
        assertEquals(0.08439f, c.prob(1), 1e-4f);
    }

    @Test
    public void xtcSkipsTheTrimWhenTheChanceMisses() {
        // probability 0.1 < chance 0.37454012: no trim, and not even the softmax runs, so the raw set stays in
        // vocabulary order with its logits untouched.
        Sampler.Candidates c = apply(xtcSampler(0.1f, 0.1f, 0));
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, ids(c));
        assertEquals(2.0f, c.logit(0), 0f);
        assertEquals(3.0f, c.logit(3), 0f);
    }

    @Test
    public void xtcMinKeepGuardsTheTail() {
        // min_keep 5 with threshold 0.2: the trim would keep 4, so llama.cpp keeps the whole sorted set.
        Sampler.Candidates kept = apply(xtcSampler(0.9f, 0.2f, 5));
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(kept));
        assertEquals(0.62359f, kept.prob(0), 1e-4f);
        assertEquals(0.22941f, kept.prob(1), 1e-4f);
        // min_keep 4 is exactly the survivor count: the trim runs.
        Sampler.Candidates trimmed = apply(xtcSampler(0.9f, 0.2f, 4));
        assertArrayEquals(new int[]{0, 1, 2, 4}, ids(trimmed));
    }

    @Test
    public void xtcResetRestartsItsDraw() {
        Sampler s = xtcSampler(0.5f, 0.1f, 0);
        int[] first = ids(apply(s));
        assertArrayEquals(new int[]{0, 1, 2, 4}, first);
        s.reset();
        // The draw restarts at the same chance 0.37454012, so the reset chain trims the same set again.
        assertArrayEquals(first, ids(apply(s)));
    }

    @Test
    public void xtcDisabledGatesNeverTouchTheSet() {
        // probability 0 and a threshold above 0.5 are llama.cpp's empty samplers: they return before the softmax,
        // so the raw set stays in vocabulary order.
        Sampler.Candidates off = apply(xtcSampler(0f, 0.1f, 0));
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, ids(off));
        Sampler.Candidates over = apply(xtcSampler(0.9f, 0.6f, 0));
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, ids(over));
        // both gates keep the chain argmax-only, the enabled stage does not
        assertTrue(Sampler.create(base().greedy(true).samplers("xtc").build()).argmaxOnly());
        assertTrue(Sampler.create(base().greedy(true).samplers("xtc").xtcProbability(0.5f).xtcThreshold(0.6f)
                .build()).argmaxOnly());
        assertFalse(Sampler.create(base().greedy(true).samplers("xtc").xtcProbability(0.5f).xtcThreshold(0.1f)
                .build()).argmaxOnly());
    }

    // ------------------------------------------------------------------ mirostat

    @Test
    public void mirostatV2TruncatesAtSurpriseMuAndDraws() {
        // Uniform row: every probability is 0.2 (surprise -log2(0.2) = 2.32193). mu = 2*tau = 2.4 with tau 1.2:
        // nothing truncates, and the draw target 0.79654298 picks the 4th candidate (cumulative 0.8), id 3 because
        // equal logits keep their id order (llama.cpp ties sort stable).
        SamplerConfig config = base().seed(SEED).mirostat(2).mirostatTau(1.2f).build();
        Sampler.Candidates c = apply(Sampler.create(config), new float[]{0f, 0f, 0f, 0f, 0f});
        assertEquals(3, c.selectedId());
        assertEquals(5, c.size());
        for (int i = 0; i < 5; i++) {
            assertEquals(0.2f, c.prob(i), 1e-6f);
        }
    }

    @Test
    public void mirostatV2CollapsesToTheTopWhenMuIsLow() {
        // tau 1.0 → mu = 2: the top candidate's surprise -log2(0.62359) = 0.68133 stays, but the next (surprise
        // 2.12402 > 2) truncates; the surviving set collapses to the top candidate and renormalizes to 1.
        SamplerConfig config = base().seed(SEED).mirostat(2).mirostatTau(1.0f).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertArrayEquals(new int[]{3}, ids(c));
        assertEquals(1.0f, c.prob(0), 1e-6f);
        assertEquals(3, c.selectedId());
    }

    @Test
    public void mirostatV1KeepsTheWholeSetAtTheDefaultTauAndDraws() {
        // tau 5 (llama.cpp default) → mu = 10. Over the four probability ratios the estimate is s_hat = 2.02875,
        // epsilon_hat = 1.02875, so k = ((1.02875 * 2^10) / (1 - 5^-1.02875))^(1/2.02875) = 34.29 and the top-k
        // clamps to the whole set of 5. The draw target 0.79654298 crosses at the second cumulative probability
        // (0.62359 + 0.22941 = 0.85300), whose token is id 0 (llama_sampler_mirostat_apply).
        SamplerConfig config = base().seed(SEED).mirostat(1).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertEquals(5, c.size());
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(c));
        assertEquals(0.62359f, c.prob(0), 1e-4f);
        assertEquals(0.22941f, c.prob(1), 1e-4f);
        assertEquals(0, c.selectedId());
        assertFalse(Sampler.create(config).argmaxOnly());
    }

    @Test
    public void mirostatReplacesTheWholeConfiguredOrder() {
        // llama.cpp builds temp + mirostat only: a configured top_k order is ignored, as is the fixed dist.
        SamplerConfig config = base().seed(SEED).mirostat(1).samplers("top_k;typ_p;min_p").topK(2).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(c));
        assertEquals(0, c.selectedId());
    }

    // ------------------------------------------------------------------ dynatemp

    @Test
    public void dynatempRescalesByTheEntropyMappedTemperature() {
        // The row's softmax has entropy 1.04411 against maximum ln 5 = 1.60944, so norm 0.64874; with exponent 1
        // and range 0.5 around temperature 1 the dynamic temperature is 1.14874, and the dist over the rescaled
        // logits gives probs [0.24089, 0.10087, 0.06527, 0.57528, 0.01769] for ids 0..4. The stage sorted the set
        // (softmax_impl with sort), so the sorted order is [id3 0.57528, id0 0.24089, id1 0.10087, id2 0.06527,
        // id4 0.01769], and the draw target 0.79654298 lands past the first two (cumulative 0.81617) on id 0
        // (llama_sampler_temp_ext_apply).
        SamplerConfig config = base().seed(SEED).samplers("temperature").temperature(1f).dynatempRange(0.5f)
                .dynatempExponent(1f).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertEquals(0, c.selectedId());
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(c));
        assertEquals(0.57528f, c.prob(0), 1e-3f);
        assertEquals(0.24089f, c.prob(1), 1e-3f);
        assertEquals(0.10087f, c.prob(2), 1e-3f);
        assertEquals(0.06527f, c.prob(3), 1e-3f);
        assertEquals(0.01769f, c.prob(4), 1e-3f);
    }

    @Test
    public void dynatempExponentOfZeroMapsToTheMaximumTemperature() {
        // pow(norm, 0) = 1 → dyn_temp = min + (max - min) = 1.5: every logit is divided by 1.5, which gives the
        // softmax [0.25225, 0.12951, 0.09280, 0.49131, 0.03414] for ids 0..4. The sorted set is [id3, id0, id1,
        // id2, id4], and the draw target 0.79654298 lands on id1 (cumulative 0.87307 crosses it).
        SamplerConfig config = base().seed(SEED).samplers("temperature").temperature(1f).dynatempRange(0.5f)
                .dynatempExponent(0f).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertEquals(1, c.selectedId());
        assertArrayEquals(new int[]{3, 0, 1, 2, 4}, ids(c));
        assertEquals(0.49131f, c.prob(0), 1e-3f);
        assertEquals(0.25225f, c.prob(1), 1e-3f);
        assertEquals(0.12951f, c.prob(2), 1e-3f);
        assertEquals(0.09280f, c.prob(3), 1e-3f);
        assertEquals(0.03414f, c.prob(4), 1e-3f);
    }

    @Test
    public void dynatempRangeOfZeroKeepsThePlainScaling() {
        // temperature 2 with no range: plain /2 scaling over [1, 0.5, 0.25, 1.5, -0.5], whose softmax is
        // [0.25310, 0.15352, 0.11956, 0.41732, 0.05649]. The plain temperature never sorts, so the set stays in
        // vocabulary order, and the draw target 0.79654298 crosses at id3 (cumulative 0.94350).
        SamplerConfig config = base().seed(SEED).samplers("temperature").temperature(2f).build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, ids(c));
        assertEquals(3, c.selectedId());
        assertEquals(0.41732f, c.prob(3), 1e-3f);
    }

    // ------------------------------------------------------------------ dry

    /** DRY with an empty breakers list (no vocabulary needed): multiplier 0.8, base 2, window 8. */
    private static Sampler drySampler(int allowedLength) {
        SamplerConfig config = base().seed(SEED).greedy(true).samplers("dry").dryMultiplier(0.8f).dryBase(2f)
                .dryAllowedLength(allowedLength).dryPenaltyLastN(8)
                .drySequenceBreakers(Collections.emptyList()).build();
        return Sampler.create(config);
    }

    @Test
    public void dryPenalizesTheTokenThatExtendsARepeat() {
        // Window [0, 1, 0, 1] (rat = 1, 0, 1, 0): the reverse Z-array is [0, 2, 0, 0], so only emitting a 0 would
        // extend the repeated "1 0" of length 2: penalty 0.8 * base^(2-2) = 0.8 on token 0
        // (llama.cpp steps 3/4 of llama_sampler_dry_apply).
        Sampler s = drySampler(2);
        s.accept(0);
        s.accept(1);
        s.accept(0);
        s.accept(1);
        Sampler.Candidates c = apply(s);
        assertEquals(1.2f, c.logit(0), 1e-6f);
        assertEquals(1.0f, c.logit(1), 0f);
        assertEquals(3.0f, c.logit(3), 0f);
        assertEquals(3, c.selectedId());
    }

    @Test
    public void dryLongerRepeatsGetTheStrongerPenalty() {
        // Window [0, 1] three times (rat = 1, 0, 1, 0, 1, 0): the Z-entry 4 marks the 0 as extending a repeat of
        // length 4: penalty 0.8 * 2^(4-2) = 3.2.
        Sampler s = drySampler(2);
        s.accept(0);
        s.accept(1);
        s.accept(0);
        s.accept(1);
        s.accept(0);
        s.accept(1);
        Sampler.Candidates c = apply(s);
        assertEquals(-1.2f, c.logit(0), 1e-6f);
        assertEquals(1.0f, c.logit(1), 0f);
    }

    @Test
    public void dryAllowedLengthGuardsShortRepeatSuffixes() {
        // allowed_length 3: the longest repeat in the window is only 2, so nothing is penalized.
        Sampler s = drySampler(3);
        s.accept(0);
        s.accept(1);
        s.accept(0);
        s.accept(1);
        Sampler.Candidates c = apply(s);
        for (int i = 0; i < LOGITS.length; i++) {
            assertEquals(LOGITS[i], c.logit(i), 0f);
        }
    }

    @Test
    public void dryDisabledStagesKeepNoHistoryAndStayArgmaxOnly() {
        // multiplier 0 is llama.cpp's empty dry sampler: no history, no penalty.
        SamplerConfig config = base().greedy(true).samplers("dry").build();
        Sampler s = Sampler.create(config);
        s.accept(0);
        s.accept(1);
        s.accept(0);
        s.accept(1);
        Sampler.Candidates c = apply(s);
        for (int i = 0; i < LOGITS.length; i++) {
            assertEquals(LOGITS[i], c.logit(i), 0f);
        }
        assertTrue(Sampler.create(config).argmaxOnly());
        // an enabled stage with breakers needs the vocabulary; an enabled stage with no breakers builds without one
        assertFalse(Sampler.create(config.toBuilder().dryMultiplier(0.5f)
                .drySequenceBreakers(Collections.emptyList()).build()).argmaxOnly());
        // base < 1 disables again
        assertTrue(Sampler.create(config.toBuilder().dryMultiplier(0.5f).dryBase(0.5f).build()).argmaxOnly());
    }

    @Test
    public void dryTokenizesItsBreakersWithTheModelVocabulary() {
        SamplerConfig config = base().greedy(true).samplers("dry").dryMultiplier(0.1f).dryBase(2f)
                .dryPenaltyLastN(8).drySequenceBreakers(Collections.emptyList()).build();
        Sampler.create(config, new int[0], (SamplerVocab) null);
        // an enabled version with breakers cannot build without the model's vocabulary
        SamplerConfig withBreakers = config.toBuilder().drySequenceBreakers(List.of(".")).build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Sampler.create(withBreakers, new int[0], (SamplerVocab) null));
        assertTrue(e.getMessage(), e.getMessage().contains("vocabulary"));
        assertThrows(IllegalArgumentException.class, () -> Sampler.create(withBreakers));
    }

    // ------------------------------------------------------------------ adaptive-p

    @Test
    public void adaptivePReshapesAroundTheEmaOfAcceptedProbabilities() {
        // target 0.5, decay 0: the EMA starts at the target, so the adapted target is 2*0.5 - 0.5 = 0.5. A uniform
        // row (p = 0.2) sits at distance 1.0 = |0.2 - 0.5|/0.3, so every transform is 5 - 10/2 = 0: the set stays
        // uniform and the draw target 0.79654298 selects id 3. accept(3) folds the original 0.2 (decay 0), so the
        // next adapted target is 2*0.5 - 0.2 = 0.8: distance 2.0, transform 5 - 10*4/3 = -8.3333, again uniform.
        // Draw target 0.18343479 selects id 0, then target 0.77969100 selects id 3 again
        // (llama_sampler_adaptive_p_apply, constants DISTRIBUTION_WIDTH 0.3 / PEAK_LOGIT 5 / SHARPNESS 10).
        SamplerConfig config = base().seed(SEED).samplers("adaptive_p").adaptiveTarget(0.5f).adaptiveDecay(0f)
                .build();
        Sampler s = Sampler.create(config);
        float[] row = {0f, 0f, 0f, 0f, 0f};
        assertEquals(3, s.apply(row).selectedId());
        s.accept(3);
        assertEquals(0, s.apply(row).selectedId());
        s.accept(0);
        assertEquals(3, s.apply(row).selectedId());
        assertFalse(Sampler.create(config).argmaxOnly());
    }

    @Test
    public void adaptivePNegativeTargetJustDraws() {
        // target -1 (the llama.cpp default) is the no-op branch: the draw runs over the plain softmax of the raw
        // logits, target 0.79654298 crosses inside the top probability 0.62359 → id 3; the second draw 0.18343479
        // crosses within the first candidate 0.22941 → id 0.
        SamplerConfig config = base().seed(SEED).samplers("adaptive_p").build();
        Sampler s = Sampler.create(config);
        assertEquals(3, s.apply(LOGITS).selectedId());
        s.accept(3);
        assertEquals(0, s.apply(LOGITS).selectedId());
    }

    // ------------------------------------------------------------------ infill

    /** A five-token vocabulary: ids 0-3 text, 4 the end-of-text end-of-generation id. */
    private static SamplerVocab eogVocab() {
        return new SamplerVocab() {
            @Override
            public int vocabSize() {
                return 5;
            }

            @Override
            public boolean isEog(int id) {
                return id == 4;
            }

            @Override
            public int eotToken() {
                return 4;
            }

            @Override
            public int eosToken() {
                return -1;
            }

            @Override
            public byte[] piece(int id) {
                return pieceText(id).getBytes(StandardCharsets.UTF_8);
            }

            @Override
            public String pieceText(int id) {
                String[] pieces = {"ab", "c", "a", "!", "<eot>"};
                return pieces[id];
            }

            @Override
            public int[] tokenize(String text) {
                return new int[0];
            }
        };
    }

    /** A 40-token vocabulary: 39 distinct single-character pieces and id 39 the EOT/EOG. */
    private static SamplerVocab wideVocab() {
        return new SamplerVocab() {
            @Override
            public int vocabSize() {
                return 40;
            }

            @Override
            public boolean isEog(int id) {
                return id == 39;
            }

            @Override
            public int eotToken() {
                return 39;
            }

            @Override
            public int eosToken() {
                return -1;
            }

            @Override
            public byte[] piece(int id) {
                return pieceText(id).getBytes(StandardCharsets.UTF_8);
            }

            @Override
            public String pieceText(int id) {
                return id == 39 ? "<eot>" : String.valueOf((char) ('A' + id));
            }

            @Override
            public int[] tokenize(String text) {
                return new int[0];
            }
        };
    }

    @Test
    public void infillCombinesPrefixTokensThenDropsTheSmallMass() {
        // Row {1, 1, 2, 0, -2} with pieces ab / c / a / ! / <eot>(EOG): the softmax is [ab 0.1946, c 0.1946,
        // a 0.5288, ! 0.0738, <eot> 0.0097]. The EOG ratio check 3*0.0097*5 = 0.1458 < 0.9903 does not trigger,
        // the "a" prefix absorbs "ab" (0.5288 + 0.1946 = 0.7234), the 0.2 threshold keeps "a" and the EOG token,
        // and the renormalized pair keeps both survivors through the 1/(1+1) = 0.5 second threshold
        // (llama_sampler_infill_apply).
        SamplerConfig config = base().greedy(true).samplers("infill").build();
        float[] logits = {1f, 1f, 2f, 0f, -2f};
        Sampler.Candidates c = apply(Sampler.create(config, new int[0], eogVocab()), logits);
        assertArrayEquals(new int[]{2, 4}, ids(c));
        assertEquals(0.9868f, c.prob(0), 1e-3f);
        assertEquals(0.0132f, c.prob(1), 1e-3f);
        assertEquals(2, c.selectedId());
    }

    @Test
    public void infillCollapsesToTheEogTokensWhenTheRatioIsLow() {
        // Row {-5, -5, -5, 2} with the EOG at logit 2: p_eog = 0.99727 vs p_text 0.00273; 3*0.99727*4 = 11.97
        // dominates, so the set reduces to the EOG candidates renormalized to 1.
        SamplerConfig config = base().greedy(true).samplers("infill").build();
        float[] logits = {-5f, -5f, -5f, 2f};
        Sampler.Candidates c = apply(Sampler.create(config, new int[0], eogVocab()), logits);
        assertArrayEquals(new int[]{3}, ids(c));
        assertEquals(1f, c.prob(0), 1e-5f);
        assertEquals(3, c.selectedId());
    }

    @Test
    public void infillFallsBackToTheEotTokenWhenNoTextSurvives() {
        // 40 uniform text logits (p = 0.02563 each) and one deep EOG (logit -4, p = 0.00047): the EOG ratio check
        // 3*0.00047*40 = 0.056 < 0.99953 holds, the 0.2 threshold drops every text candidate, and llama.cpp
        // replaces the set by the single token_eot at logit 1.
        SamplerConfig config = base().greedy(true).samplers("infill").build();
        float[] logits = new float[40];
        logits[39] = -4f;
        Sampler.Candidates c = apply(Sampler.create(config, new int[0], wideVocab()), logits);
        assertArrayEquals(new int[]{39}, ids(c));
        assertEquals(1.0f, c.logit(0), 0f);
        assertEquals(39, c.selectedId());
    }

    @Test
    public void infillNeedsTheModelVocabularyToBuild() {
        SamplerConfig config = base().greedy(true).samplers("infill").build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Sampler.create(config, new int[0], (SamplerVocab) null));
        assertTrue(e.getMessage(), e.getMessage().contains("vocabulary"));
        // an order without the infill kind still builds without one
        Sampler.create(config.toBuilder().samplers("top_k").topK(1).build(), new int[0], (SamplerVocab) null);
    }

    // ------------------------------------------------------------------ config surface

    @Test
    public void stageKnobsCarryLlamaCppDefaults() {
        SamplerConfig d = SamplerConfig.defaults();
        assertEquals(0f, d.dryMultiplier(), 0f);
        assertEquals(1.75f, d.dryBase(), 0f);
        assertEquals(2, d.dryAllowedLength());
        assertEquals(64, d.dryPenaltyLastN());
        assertEquals(Arrays.asList("\n", ":", "\"", "*"), d.drySequenceBreakers());
        assertEquals(0f, d.topNSigma(), 0f);
        assertEquals(0f, d.xtcProbability(), 0f);
        assertEquals(0.10f, d.xtcThreshold(), 0f);
        assertEquals(0, d.mirostat());
        assertEquals(5.0f, d.mirostatTau(), 0f);
        assertEquals(0.10f, d.mirostatEta(), 0f);
        assertEquals(0f, d.dynatempRange(), 0f);
        assertEquals(1f, d.dynatempExponent(), 0f);
        assertEquals(-1f, d.adaptiveTarget(), 0f);
        assertEquals(0.9f, d.adaptiveDecay(), 0f);
        assertFalse(d.isExplicit(SamplerConfig.Setting.MIROSTAT));
    }

    @Test
    public void stageKnobsParseFromProperties() {
        Properties p = new Properties();
        p.setProperty(SamplerConfig.PROP_DRY_MULTIPLIER, "0.8");
        p.setProperty(SamplerConfig.PROP_DRY_BASE, "2.0");
        p.setProperty(SamplerConfig.PROP_DRY_ALLOWED_LENGTH, "3");
        p.setProperty(SamplerConfig.PROP_DRY_PENALTY_LAST_N, "128");
        p.setProperty(SamplerConfig.PROP_DRY_SEQUENCE_BREAKERS, "\n,:");
        p.setProperty(SamplerConfig.PROP_TOP_N_SIGMA, "1.5");
        p.setProperty(SamplerConfig.PROP_XTC_PROBABILITY, "0.7");
        p.setProperty(SamplerConfig.PROP_XTC_THRESHOLD, "0.2");
        p.setProperty(SamplerConfig.PROP_MIROSTAT, "2");
        p.setProperty(SamplerConfig.PROP_MIROSTAT_TAU, "3.0");
        p.setProperty(SamplerConfig.PROP_MIROSTAT_ETA, "0.2");
        p.setProperty(SamplerConfig.PROP_DYNATEMP_RANGE, "0.5");
        p.setProperty(SamplerConfig.PROP_DYNATEMP_EXPONENT, "2");
        p.setProperty(SamplerConfig.PROP_ADAPTIVE_TARGET, "0.4");
        p.setProperty(SamplerConfig.PROP_ADAPTIVE_DECAY, "0.5");
        SamplerConfig c = SamplerConfig.fromProperties(p);
        assertEquals(0.8f, c.dryMultiplier(), 0f);
        assertEquals(2.0f, c.dryBase(), 0f);
        assertEquals(3, c.dryAllowedLength());
        assertEquals(128, c.dryPenaltyLastN());
        assertEquals(Arrays.asList("\n", ":"), c.drySequenceBreakers());
        assertEquals(1.5f, c.topNSigma(), 0f);
        assertEquals(0.7f, c.xtcProbability(), 0f);
        assertTrue(c.isExplicit(SamplerConfig.Setting.XTC_PROBABILITY));
        assertEquals(0.2f, c.xtcThreshold(), 0f);
        assertTrue(c.isExplicit(SamplerConfig.Setting.XTC_THRESHOLD));
        assertEquals(2, c.mirostat());
        assertTrue(c.isExplicit(SamplerConfig.Setting.MIROSTAT));
        assertEquals(3.0f, c.mirostatTau(), 0f);
        assertTrue(c.isExplicit(SamplerConfig.Setting.MIROSTAT_TAU));
        assertEquals(0.2f, c.mirostatEta(), 0f);
        assertTrue(c.isExplicit(SamplerConfig.Setting.MIROSTAT_ETA));
        assertEquals(0.5f, c.dynatempRange(), 0f);
        assertEquals(2f, c.dynatempExponent(), 0f);
        assertEquals(0.4f, c.adaptiveTarget(), 0f);
        assertEquals(0.5f, c.adaptiveDecay(), 0f);
    }

    @Test
    public void stageKnobValuesAreValidated() {
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().mirostat(3).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().mirostat(-1).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().mirostatTau(0f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().mirostatEta(0f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().xtcProbability(1.5f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().xtcThreshold(1.5f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().dynatempRange(-1f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().dynatempExponent(-1f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().adaptiveDecay(1.5f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().dryAllowedLength(-1).build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().dryPenaltyLastN(-1).build());
        // the disabling values themselves are accepted
        assertEquals(0, SamplerConfig.builder().mirostat(0).build().mirostat());
        assertEquals(0.6f, SamplerConfig.builder().xtcThreshold(0.6f).build().xtcThreshold(), 0f);
        assertEquals(-2f, SamplerConfig.builder().adaptiveTarget(-2f).build().adaptiveTarget(), 0f);
        assertEquals(0f, SamplerConfig.builder().adaptiveDecay(0f).build().adaptiveDecay(), 0f);
    }

    @Test
    public void toBuilderKeepsTheStageKnobs() {
        SamplerConfig c = base().dryMultiplier(0.8f).dryBase(2f).dryAllowedLength(3).dryPenaltyLastN(9)
                .drySequenceBreakers(List.of("x")).topNSigma(1f).xtcProbability(0.4f).xtcThreshold(0.3f)
                .mirostat(1).mirostatTau(2f).mirostatEta(0.3f).dynatempRange(0.2f).dynatempExponent(2f)
                .adaptiveTarget(0.3f).adaptiveDecay(0.4f).build();
        SamplerConfig copy = c.toBuilder().build();
        assertEquals(c.dryMultiplier(), copy.dryMultiplier(), 0f);
        assertEquals(c.dryBase(), copy.dryBase(), 0f);
        assertEquals(c.dryAllowedLength(), copy.dryAllowedLength());
        assertEquals(c.dryPenaltyLastN(), copy.dryPenaltyLastN());
        assertEquals(c.drySequenceBreakers(), copy.drySequenceBreakers());
        assertEquals(c.topNSigma(), copy.topNSigma(), 0f);
        assertEquals(c.xtcProbability(), copy.xtcProbability(), 0f);
        assertEquals(c.xtcThreshold(), copy.xtcThreshold(), 0f);
        assertEquals(c.mirostat(), copy.mirostat());
        assertEquals(c.mirostatTau(), copy.mirostatTau(), 0f);
        assertEquals(c.mirostatEta(), copy.mirostatEta(), 0f);
        assertEquals(c.dynatempRange(), copy.dynatempRange(), 0f);
        assertEquals(c.dynatempExponent(), copy.dynatempExponent(), 0f);
        assertEquals(c.adaptiveTarget(), copy.adaptiveTarget(), 0f);
        assertEquals(c.adaptiveDecay(), copy.adaptiveDecay(), 0f);
        assertTrue(copy.isExplicit(SamplerConfig.Setting.MIROSTAT));
    }

    @Test
    public void modelDefaultsFillTheMirostatAndXtcSettings() {
        SamplingDefaults model = SamplingDefaults.builder().mirostat(2).mirostatTau(1.0f).mirostatEta(0.2f)
                .xtcProbability(0.5f).xtcThreshold(0.3f).build();
        SamplerConfig filled = base().build().withModelDefaults(model);
        assertEquals(2, filled.mirostat());
        assertEquals(1.0f, filled.mirostatTau(), 0f);
        assertEquals(0.2f, filled.mirostatEta(), 0f);
        assertEquals(0.5f, filled.xtcProbability(), 0f);
        assertEquals(0.3f, filled.xtcThreshold(), 0f);
        assertFalse(filled.isExplicit(SamplerConfig.Setting.MIROSTAT));
        assertFalse(filled.isExplicit(SamplerConfig.Setting.XTC_THRESHOLD));

        // request-explicit settings are never replaced
        SamplerConfig explicit = base().mirostat(1).xtcProbability(0.9f).build().withModelDefaults(model);
        assertEquals(1, explicit.mirostat());
        assertEquals(0.9f, explicit.xtcProbability(), 0f);
        assertTrue(explicit.isExplicit(SamplerConfig.Setting.MIROSTAT));

        // the model's mirostat version also builds the temp + mirostat chain: mu = 2*1 = 2 under the uniform
        // surprise 2.32193 collapses the uniform row to its first candidate, which id-order ties put at id 0
        SamplerConfig chain = base().seed(SEED).build().withModelDefaults(
                SamplingDefaults.builder().mirostat(2).mirostatTau(1.0f).build());
        Sampler.Candidates c = apply(Sampler.create(chain), new float[]{0f, 0f, 0f, 0f, 0f});
        assertArrayEquals(new int[]{0}, ids(c));
        assertEquals(1.0f, c.prob(0), 1e-6f);
        assertEquals(0, c.selectedId());
    }

    @Test
    public void modelDefaultValuesAreValidated() {
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().mirostat(3).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().mirostatTau(0f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().mirostatEta(-0.1f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().xtcProbability(2f).build());
        assertThrows(IllegalArgumentException.class, () -> SamplingDefaults.builder().xtcThreshold(-1f).build());
        assertEquals(0, SamplingDefaults.builder().mirostat(0).build().mirostat().intValue());
    }

    @Test
    public void stageKindsPlusDisableGatesKeepTheDefaultChainIdentical() {
        // The default order carries the new dry/top_n_sigma/xtc entries; with the llama.cpp default knob values
        // all three are empty samplers, so the default chain leaves every logit untouched (the same assertion the
        // pre-split chain comparison of SamplerOrderTest makes on this row).
        SamplerConfig config = base().build();
        Sampler.Candidates c = apply(Sampler.create(config), LOGITS);
        for (int i = 0; i < LOGITS.length; i++) {
            assertEquals(LOGITS[i], c.logit(i), 0f);
        }
        assertEquals(Arrays.asList("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc",
                "temperature"), labels(config.order()));
    }

    private static Sampler.Candidates apply(Sampler s) {
        return s.apply(LOGITS);
    }

    private static Sampler.Candidates apply(Sampler s, float[] logits) {
        return s.apply(logits);
    }
}
