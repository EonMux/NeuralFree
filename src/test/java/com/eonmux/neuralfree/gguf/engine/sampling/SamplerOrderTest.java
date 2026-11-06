package com.eonmux.neuralfree.gguf.engine.sampling;

import com.eonmux.neuralfree.gguf.engine.Sampler;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.SamplingDefaults;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The configurable stage order. The default chain ({@link StageKind#DEFAULT_ORDER} with the stages SA-2 adds off)
 * gives the candidates of the pre-split chain - the fixed {@code bias → penalties → top-k → typical-p → top-p →
 * min-p → temperature → dist} implementation Sampler ran before the stage split - compared candidate per candidate
 * on fixed logits and seeded random rows. {@code neuralfree.samplers} and {@code neuralfree.min.keep} parse and
 * apply; a model's {@code general.sampling.sequence} fills the order of a chain the request does not configure.
 */
public class SamplerOrderTest {

    private static final float[] LOGITS = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f};
    private static final long SEED = 42L;

    /** The default request: the explicit llama.cpp order, everything else off. */
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

    /** Applies {@code logits} to the reference chain first, then to {@code sampler}, and compares both sets. */
    private static void assertSameCandidates(Sampler sampler, PreSplitChain reference, float[] logits) {
        reference.apply(logits);
        Sampler.Candidates c = sampler.apply(logits);
        int n = reference.size();
        assertEquals("candidates left", n, c.size());
        for (int i = 0; i < n; i++) {
            assertEquals("token id " + i, reference.id(i), c.id(i));
            assertEquals("logit " + i, reference.logit(i), c.logit(i), 0f);
            assertEquals("prob " + i, reference.prob(i), c.prob(i), 0f);
        }
        assertEquals("sorted", reference.sorted(), c.sorted());
    }

    // ------------------------------------------------------------------ the tests

    @Test
    public void defaultChainMatchesThePreSplitChain() {
        List<SamplerConfig> configs = Arrays.asList(
                base().topK(3).topP(0.9f).minP(0.1f).build(),
                base().build(),
                base().greedy(true).build(),
                base().temperature(0f).build(),
                base().greedy(true).repeatPenalty(1.5f).penaltyLastN(64).build(),
                base().typicalP(0.5f).build(),
                base().topP(0.7f).minP(0.05f).build(),
                base().topK(2).temperature(0.5f).build(),
                base().minP(0.3f).build());
        for (SamplerConfig config : configs) {
            Sampler s = Sampler.create(config);
            assertSameCandidates(s, new PreSplitChain(config, new int[0], s.seed()), LOGITS);
        }
    }

    @Test
    public void seededDrawsAndAcceptedTokensMatchAcrossSamples() {
        SamplerConfig config = base().topK(3).topP(0.9f).minP(0.1f).typicalP(0.6f).build();
        Sampler s = Sampler.create(config);
        PreSplitChain reference = new PreSplitChain(config, new int[0], s.seed());
        for (int step = 0; step < 16; step++) {
            reference.accept(step % 5);
            s.accept(step % 5);
            assertSameCandidates(s, reference, LOGITS);
        }
        s.reset();
        reference = new PreSplitChain(config, new int[0], s.seed());
        assertSameCandidates(s, reference, LOGITS);
    }

    @Test
    public void bannedTokensMatchThePreSplitChain() {
        SamplerConfig config = base().topK(2).minP(0.4f).greedy(true).build();
        int[] banned = {1, 4};
        Sampler s = Sampler.create(config, banned);
        assertSameCandidates(s, new PreSplitChain(config, banned, s.seed()), LOGITS);
    }

    @Test
    public void randomRowsMatchThePreSplitChain() {
        SplittableRandom r = new SplittableRandom(9);
        List<SamplerConfig> configs = Arrays.asList(
                base().topK(3).topP(0.9f).minP(0.1f).build(),
                base().greedy(true).repeatPenalty(1.2f).penaltyLastN(-1).build(),
                base().typicalP(0.5f).topK(4).topP(0.6f).build());
        for (int trial = 0; trial < 24; trial++) {
            float[] row = new float[64];
            for (int i = 0; i < row.length; i++) {
                row[i] = (float) (r.nextDouble() * 8.0 - 4.0);
            }
            for (SamplerConfig config : configs) {
                Sampler s = Sampler.create(config);
                assertSameCandidates(s, new PreSplitChain(config, new int[0], s.seed()), row);
            }
        }
    }

    @Test
    public void diffusionChainCandidatesAreInVocabularyOrder() {
        Sampler s = Sampler.diffusionChain(0, 1f, 0f, SEED);
        assertFalse(s.greedy());
        Sampler.Candidates c = s.apply(LOGITS);
        assertEquals(5, c.size());
        assertArrayEquals(new int[]{0, 1, 2, 3, 4}, ids(c));
        float sum = 0f;
        for (int i = 0; i < c.size(); i++) {
            sum += c.prob(i);
        }
        assertEquals(1f, sum, 1e-4f);
        assertEquals(0.62359f, c.prob(3), 1e-4f);
    }

    // ------------------------------------------------------------------ the order

    @Test
    public void defaultOrderIsLlamaCpps() {
        assertEquals(Arrays.asList("penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc",
                "temperature"), labels(SamplerConfig.defaults().order()));
        assertEquals(StageKind.DEFAULT_ORDER, SamplerConfig.defaults().order());
        assertEquals(0, SamplerConfig.defaults().minKeep());
    }

    @Test
    public void sa2KindsBuildNothingYet() {
        // dry, top_n_sigma and xtc are accepted llama.cpp names; without their stage classes (SA-2) they apply
        // nothing, which keeps the default chain identical to the pre-split chain as well.
        SamplerConfig config = base().samplers("penalties;dry;top_k").build();
        assertEquals(Arrays.asList("penalties", "dry", "top_k"), labels(config.order()));
        Sampler s = Sampler.create(config);
        assertSameCandidates(s, new PreSplitChain(config, new int[0], s.seed()), LOGITS);
    }

    @Test
    public void samplersPropertyParsesAndValidates() {
        Properties p = new Properties();
        p.setProperty(SamplerConfig.PROP_SAMPLERS, "typ_p;top_k");
        SamplerConfig c = SamplerConfig.fromProperties(p);
        assertEquals(Arrays.asList("typ_p", "top_k"), labels(c.order()));
        assertTrue(c.isExplicit(SamplerConfig.Setting.SEQUENCE));

        p.setProperty(SamplerConfig.PROP_SAMPLERS, "top_k; top_p ;typ_p");
        assertEquals(Arrays.asList("top_k", "top_p", "typ_p"),
                labels(SamplerConfig.fromProperties(p).order()));

        p.setProperty(SamplerConfig.PROP_SAMPLERS, "top_k;top_k");
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> SamplerConfig.fromProperties(p));
        assertTrue(e.getMessage().contains(SamplerConfig.PROP_SAMPLERS));

        p.setProperty(SamplerConfig.PROP_SAMPLERS, "top-k");
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.fromProperties(p));

        p.setProperty(SamplerConfig.PROP_SAMPLERS, "top_k;;typ_p");
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.fromProperties(p));
    }

    @Test
    public void aCustomOrderRunsItsStagesInOrder() {
        // temperature first, then top-k: every survivor is scaled by 1/0.5 and the scaled top-2 wins.
        SamplerConfig config = base().topK(2).temperature(0.5f).samplers("temperature;top_k").build();
        Sampler.Candidates c = Sampler.create(config).apply(LOGITS);
        assertArrayEquals(new int[]{3, 0}, ids(c));
        assertTrue(c.sorted());
        assertEquals(6.0f, c.logit(0), 0f);
        assertEquals(4.0f, c.logit(1), 0f);
    }

    @Test
    public void minKeepKeepsThatManyCandidates() {
        // top-k: keep 3 although topK is 2.
        assertArrayEquals(new int[]{3, 0, 1}, ids(Sampler.create(
                base().topK(2).minKeep(3).build()).apply(LOGITS)));
        // top-p 0.5 keeps [3]; min_keep 2 keeps the next candidate too.
        assertArrayEquals(new int[]{3, 0}, ids(Sampler.create(
                base().topP(0.5f).minKeep(2).build()).apply(LOGITS)));
        // typical 0.2 keeps [0] alone (its cumulative 0.2294 already exceeds it); min_keep 2 keeps [0, 3].
        assertArrayEquals(new int[]{0, 3}, ids(Sampler.create(
                base().typicalP(0.2f).minKeep(2).build()).apply(LOGITS)));
        // min-p 0.9 over [2, 1, 0.5, 3, -1]: only id 3 is over 3 + ln 0.9, so the unsorted pass fails llama.cpp's
        // min_keep test and the sorted fallback stops scanning only once min_keep candidates are kept.
        assertArrayEquals(new int[]{3, 0, 1}, ids(Sampler.create(
                base().minP(0.9f).minKeep(3).build()).apply(LOGITS)));
        // min-p 0.3 without a min_keep keeps the vocabulary-order pair (the unsorted path), as before the split.
        assertArrayEquals(new int[]{0, 3}, ids(Sampler.create(
                base().minP(0.3f).build()).apply(LOGITS)));
    }

    @Test
    public void minKeepMustNotBeNegative() {
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().minKeep(-1).build());
    }

    @Test
    public void extraStagesRunImmediatelyBeforeTemperature() {
        final List<Float> seen = new ArrayList<>();
        final List<Integer> sizes = new ArrayList<>();
        SamplerStage recorder = new SamplerStage() {
            @Override
            public void apply(Sampler.Candidates candidates) {
                seen.add(candidates.logit(0));
                sizes.add(candidates.size());
            }
        };
        Sampler s = Sampler.create(base().topK(3).temperature(0.5f).build(), new int[0],
                Collections.singletonList(recorder));
        s.apply(LOGITS);
        assertEquals(1, seen.size());
        // The recorder runs after the top-k truncation and before the temperature: the 3 survivors, unscaled.
        assertEquals(3.0f, seen.get(0), 0f);
        assertEquals(3, sizes.get(0).intValue());

        // Greedy skips the temperature stage but keeps the custom stage running.
        seen.clear();
        sizes.clear();
        Sampler greedy = Sampler.create(base().greedy(true).build(), new int[0],
                Collections.singletonList(recorder));
        assertEquals(3, greedy.apply(LOGITS).selectedId());
        assertEquals(1, seen.size());
        assertEquals(2.0f, seen.get(0), 0f);
        assertEquals(5, sizes.get(0).intValue());
    }

    /**
     * Custom logits processors run at the chain head, after the constraint head and before penalties and the
     * truncating stages (vLLM applies them to the logits before its sampling stages), so a mask can keep a token
     * top-k would otherwise prune.
     */
    @Test
    public void logitsProcessorsRunAtTheChainHeadBeforeTopK() {
        final List<Integer> sizes = new ArrayList<>();
        final List<Float> logits = new ArrayList<>();
        SamplerStage recorder = new SamplerStage() {
            @Override
            public void apply(Sampler.Candidates candidates) {
                sizes.add(candidates.size());
                logits.add(candidates.logit(0));
            }
        };
        Sampler s = Sampler.create(base().topK(2).temperature(0.5f).build(), new int[0],
                Collections.<SamplerStage>emptyList(), Collections.singletonList(recorder));
        s.apply(LOGITS);
        assertEquals("the processor sees the whole row, before top-k", 5, sizes.get(0).intValue());
        assertEquals(2.0f, logits.get(0), 0f);

        SamplerStage maskIdZero = new SamplerStage() {
            @Override
            public void apply(Sampler.Candidates candidates) {
                for (int i = 0; i < candidates.size(); i++) {
                    if (candidates.id(i) != 0) {
                        candidates.setLogit(i, Float.NEGATIVE_INFINITY);
                    }
                }
                candidates.setSorted(false);
            }
        };
        Sampler forced = Sampler.create(base().topK(1).temperature(0.5f).build(), new int[0],
                Collections.<SamplerStage>emptyList(), Collections.singletonList(maskIdZero));
        assertEquals(0, forced.apply(LOGITS).selectedId());
    }

    @Test
    public void extraStageArgumentsAreChecked() {        assertThrows(NullPointerException.class,
                () -> Sampler.create(base().build(), new int[0], (List<SamplerStage>) null));
        List<SamplerStage> withNull = new ArrayList<>();
        withNull.add(null);
        assertThrows(IllegalArgumentException.class,
                () -> Sampler.create(base().build(), new int[0], withNull));
        assertThrows(NullPointerException.class,
                () -> Sampler.create(base().build(), new int[0], Collections.singletonList(new SamplerStage() {
                    @Override
                    public void apply(Sampler.Candidates candidates) {
                    }
                }), (SamplerVocab) null));
    }

    @Test
    public void modelSequenceFillsTheOrderOrIsIgnored() {
        // A model's sequence becomes the order of a chain the request does not configure.
        SamplerConfig config = base().build();
        SamplerConfig filled = config.withModelDefaults(
                SamplingDefaults.builder().sequence("top_k;temperature").build());
        assertEquals(Arrays.asList("top_k", "temperature"), labels(filled.order()));
        assertFalse(filled.isExplicit(SamplerConfig.Setting.SEQUENCE));

        // An order the request itself configured is never replaced by the model's.
        SamplerConfig explicit = base().samplers("typ_p").build();
        SamplerConfig kept = explicit.withModelDefaults(
                SamplingDefaults.builder().sequence("top_k;temperature").build());
        assertEquals(Arrays.asList("typ_p"), labels(kept.order()));
        assertTrue(kept.isExplicit(SamplerConfig.Setting.SEQUENCE));

        // Other model defaults still fill fallbacks while the model's sequence is taken when SEQUENCE is unset; the
        // temperature is left a fallback here, since base() gives it explicitly and an explicit value always wins.
        SamplerConfig fallbackTemperature = SamplerConfig.builder().topK(0).topP(1f).minP(0f).typicalP(1f)
                .seed(SEED).build();
        SamplerConfig both = fallbackTemperature.withModelDefaults(SamplingDefaults.builder().temp(0.3f)
                .sequence("temperature").build());
        assertEquals(Arrays.asList("temperature"), labels(both.order()));
        assertFalse(both.isExplicit(SamplerConfig.Setting.SEQUENCE));
        assertEquals(0.3f, both.temperature(), 0f);
        assertEquals(1f, config.withModelDefaults(SamplingDefaults.builder().temp(0.3f).build()).temperature(), 0f);
    }

    @Test
    public void builderSamplersParseLikeTheProperty() {
        SamplerConfig c = SamplerConfig.builder().samplers("top_k").build();
        assertEquals(Arrays.asList("top_k"), labels(c.order()));
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().samplers("top_k;top_k").build());
        assertThrows(IllegalArgumentException.class, () -> SamplerConfig.builder().samplers("top-k").build());
        SamplerConfig model = SamplerConfig.builder().build().withModelDefaults(
                SamplingDefaults.builder().sequence(Arrays.asList(StageKind.TOP_P, StageKind.MIN_P)).build());
        assertEquals(Arrays.asList("top_p", "min_p"), labels(model.order()));
    }
}
