package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.SamplerVocab;
import com.eonmux.neuralfree.gguf.engine.sampling.StageKind;
import org.junit.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The device prefilter path selects exactly what the whole-row chain selects: applying {@link
 * LogitPrefilter.Row} to a row (the edits {@code LLM_LogitsPrefilter} makes) and running {@link
 * Sampler#applyTopK(int[], float[], boolean)} over its {@link Sampler#deviceConeWidth()} top candidates
 * returns the same candidates, logits, probabilities and draw as {@link Sampler#apply(float[])} over the
 * whole row, for seeded chains with penalties, DRY and logit biases.
 */
public class SamplerPrefilterTest {

    private static final int VOCAB = 600;
    private static final float NEG_INF = Float.NEGATIVE_INFINITY;

    /** A vocabulary of single-character pieces, plus "ab" pieces and a "." token for DRY's breakers. */
    private static final class FakeVocab implements SamplerVocab {

        private final Map<String, Integer> byText = new HashMap<>();

        FakeVocab() {
            byText.put(".", 0);
            byText.put("a", 1);
            byText.put("b", 2);
            byText.put("c", 3);
            for (int id = 4; id < VOCAB; id++) {
                byText.put("t" + id, id);
            }
        }

        @Override
        public int vocabSize() {
            return VOCAB;
        }

        @Override
        public boolean isEog(int id) {
            return false;
        }

        @Override
        public int eotToken() {
            return -1;
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
            for (Map.Entry<String, Integer> e : byText.entrySet()) {
                if (e.getValue() == id) {
                    return e.getKey();
                }
            }
            return "";
        }

        @Override
        public int[] tokenize(String text) {
            if (byText.containsKey(text)) {
                return new int[]{byText.get(text)};
            }
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < text.length(); i++) {
                Integer id = byText.get(String.valueOf(text.charAt(i)));
                if (id != null) {
                    ids.add(id);
                }
            }
            int[] out = new int[ids.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = ids.get(i);
            }
            return out;
        }
    }

    private static SamplerConfig.Builder penalties() {
        return SamplerConfig.builder().temperature(0.8f).topK(16).topP(0.95f).minP(0.05f).typicalP(1f)
                .penaltyLastN(12).repeatPenalty(1.25f).frequencyPenalty(0.4f).presencePenalty(0.2f);
    }

    private static float[] row(long seed) {
        Random r = new Random(seed);
        float[] x = new float[VOCAB];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (r.nextGaussian() * 2.0);
        }
        return x;
    }

    /**
     * The kernel's edits over one row, written independently of the sampler stage classes; the GPU tests use it
     * as the host expectation of {@code LLM_LogitsPrefilter}.
     */
    public static float[] applyPrefilter(float[] source, LogitPrefilter.Row row) {
        float[] x = source.clone();
        int[] ids = row.biasIds();
        float[] values = row.biasValues();
        for (int i = 0; i < ids.length; i++) {
            x[ids[i]] = Float.isNaN(values[i]) ? NEG_INF : x[ids[i]] + values[i];
        }
        if (row.penalties() && row.penaltiesFirst()) {
            penalties(x, row);
        }
        if (row.dry()) {
            dry(x, row);
        }
        if (row.penalties() && !row.penaltiesFirst()) {
            penalties(x, row);
        }
        return x;
    }

    private static void penalties(float[] x, LogitPrefilter.Row row) {
        int[] ids = row.penaltyTokens();
        int[] counts = row.penaltyCounts();
        for (int i = 0; i < ids.length; i++) {
            float l = x[ids[i]];
            l = l <= 0f ? l * row.repeatPenalty() : l / row.repeatPenalty();
            x[ids[i]] = l - ((float) counts[i] * row.frequencyPenalty() + row.presencePenalty());
        }
    }

    /** DRY over the window, oldest first, matching llama_sampler_dry_apply. */
    private static void dry(float[] x, LogitPrefilter.Row row) {
        int[] window = row.dryTokens();
        int n = window.length;
        if (n == 0) {
            return;
        }
        int[][] breakers = row.dryBreakers();
        int limit = n;
        for (int i = 0; i < n; i++) {
            int head = rat(window, i);
            int longest = -1;
            for (int[] breaker : breakers) {
                if (breaker[0] != head) {
                    continue;
                }
                int tailLen = breaker.length - 1;
                if (tailLen > longest && tailLen <= i) {
                    boolean match = true;
                    for (int o = 0; o < tailLen; o++) {
                        if (breaker[1 + o] != rat(window, i - o - 1)) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        longest = tailLen;
                    }
                }
            }
            if (longest >= 0) {
                limit = i - longest;
                break;
            }
        }
        if (limit < row.dryAllowedLength()) {
            return;
        }
        Map<Integer, Integer> maxRepeat = new HashMap<>();
        for (int k = 1; k < n; k++) {
            int z = 0;
            while (z + k < n && rat(window, z) == rat(window, z + k)) {
                z++;
            }
            int len = Math.min(z, limit);
            if (len >= row.dryAllowedLength()) {
                int token = rat(window, k - 1);
                Integer previous = maxRepeat.get(token);
                if (previous == null || previous < len) {
                    maxRepeat.put(token, len);
                }
            }
        }
        float maxExponentFloat = 0f;
        if (row.dryBase() > 1.000001f) {
            maxExponentFloat = 88.7228391f / (float) Math.log(row.dryBase());
        }
        int maxExponent = (int) maxExponentFloat > 0 ? (int) maxExponentFloat : 0;
        for (Map.Entry<Integer, Integer> e : maxRepeat.entrySet()) {
            int token = e.getKey();
            if (isSingleTokenBreaker(token, breakers)) {
                continue;
            }
            int repeatExp = e.getValue() - row.dryAllowedLength();
            if (maxExponent > 0 && repeatExp > maxExponent) {
                repeatExp = maxExponent;
            }
            x[token] -= row.dryMultiplier() * (float) Math.pow(row.dryBase(), repeatExp);
        }
    }

    private static boolean isSingleTokenBreaker(int token, int[][] breakers) {
        for (int[] breaker : breakers) {
            if (breaker.length == 1 && breaker[0] == token) {
                return true;
            }
        }
        return false;
    }

    private static int rat(int[] window, int i) {
        return window[window.length - 1 - i];
    }

    private static void assertSameSelection(SamplerConfig config, Map<Integer, Float> bias, int[] history,
                                            long seed, SamplerVocab vocab) {
        Sampler whole = Sampler.create(config, new int[0], bias, vocab);
        Sampler cone = Sampler.create(config, new int[0], bias, vocab);
        whole.accept(history, 0, history.length);
        cone.accept(history, 0, history.length);
        float[] logits = row(seed);
        LogitPrefilter.Row row = cone.prefilterRow();
        float[] edited = applyPrefilter(logits, row);
        int width = cone.deviceConeWidth();
        assertTrue("the device cone covers the chain", width >= 1);
        RowTopK top = RowTopK.of(new float[][]{edited}, width);
        Sampler.Candidates expected = whole.apply(logits);
        Sampler.Candidates actual = cone.applyTopK(top.ids(0), top.logits(0), true);
        assertEquals("the chain keeps the same candidates", expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals("candidate " + i + " token", expected.id(i), actual.id(i));
            assertEquals("candidate " + i + " logit", expected.logit(i), actual.logit(i), 0f);
            assertEquals("candidate " + i + " probability", expected.prob(i), actual.prob(i), 0f);
        }
        assertEquals("the draw", expected.id(expected.selected()), actual.id(actual.selected()));
    }

    @Test
    public void penaltiesKeepTheWholeRowSelection() {
        int[] history = {1, 2, 1, 1, 3, 2, 1, 0, 1, 2};
        for (long seed = 0; seed < 6; seed++) {
            assertSameSelection(penalties().seed(seed).build(), null, history, seed, null);
        }
    }

    @Test
    public void aPositiveLogitBiasKeepsTheWholeRowSelection() {
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(7, 4.5f);
        bias.put(11, -0.75f);
        int[] history = {1, 1, 1};
        for (long seed = 0; seed < 6; seed++) {
            assertSameSelection(SamplerConfig.builder().temperature(0.8f).topK(16).penaltyLastN(0).seed(seed)
                    .build(), bias, history, seed, null);
        }
    }

    @Test
    public void dryKeepsTheWholeRowSelection() {
        SamplerConfig config = SamplerConfig.builder().temperature(0.8f).topK(16).penaltyLastN(0)
                .dryMultiplier(0.8f).dryBase(2f).dryAllowedLength(2).dryPenaltyLastN(12)
                .samplers("dry;top_k;temperature")
                .build();
        // window "a b a b a b . b a b" exercises a repeated suffix and the single-token "." breaker
        int[] history = {1, 2, 1, 2, 1, 2, 0, 2, 1, 2};
        FakeVocab vocab = new FakeVocab();
        SamplerConfig withBreakers = config.toBuilder().drySequenceBreakers(Arrays.asList("ab", ".")).build();
        for (long seed = 0; seed < 6; seed++) {
            assertSameSelection(withBreakers.toBuilder().seed(seed).build(), null, history, seed, vocab);
        }
    }

    @Test
    public void penaltiesBeforeDryAndDryBeforePenaltiesBothKeepTheSelection() {
        int[] history = {1, 2, 1, 2, 1, 2, 0, 2, 1, 2};
        FakeVocab vocab = new FakeVocab();
        for (String order : new String[]{"penalties;dry;top_k;temperature", "dry;penalties;top_k;temperature"}) {
            SamplerConfig config = penalties().penaltyLastN(12).dryMultiplier(0.8f).dryBase(2f)
                    .dryAllowedLength(2).dryPenaltyLastN(12).drySequenceBreakers(Arrays.asList("ab", "."))
                    .samplers(order).seed(3).build();
            assertSameSelection(config, null, history, 3, vocab);
        }
    }

    @Test
    public void aPenaltyStageAfterTopKStillKeepsTheSelection() {
        SamplerConfig config = penalties().samplers("top_k;penalties;temperature").seed(5).build();
        int[] history = {1, 2, 1, 1, 3, 2, 1, 0, 1, 2};
        assertSameSelection(config, null, history, 5, null);
    }

    @Test
    public void thePrefilterReproducesTheBannedAndBiasedTable() {
        SamplerConfig config = SamplerConfig.builder().temperature(0.8f).topK(8).penaltyLastN(0).seed(1)
                .build();
        Map<Integer, Float> bias = new HashMap<>();
        bias.put(5, 1.5f);
        bias.put(6, -100f);
        bias.put(7, 1.5f);
        Sampler sampler = Sampler.create(config, new int[]{5, 9}, bias);
        LogitPrefilter.Row row = sampler.prefilterRow();
        int[] ids = row.biasIds();
        float[] values = row.biasValues();
        assertEquals(4, ids.length);
        Map<Integer, Float> table = new HashMap<>();
        for (int i = 0; i < ids.length; i++) {
            table.put(ids[i], values[i]);
        }
        assertTrue("a banned token carries the -inf marker, its bias ignored", Float.isNaN(table.get(5)));
        assertTrue("a -100 bias carries the -inf marker", Float.isNaN(table.get(6)));
        assertTrue(Float.isNaN(table.get(9)));
        assertEquals(1.5f, table.get(7), 0f);
    }

    @Test
    public void aPlainChainNeedsNoPrefilter() {
        Sampler sampler = Sampler.create(
                SamplerConfig.builder().temperature(0.8f).topK(16).penaltyLastN(0).seed(1).build());
        assertFalse(sampler.prefilterRequired());
        assertTrue(sampler.prefilterRow().isEmpty());
        assertEquals(sampler.coneWidth(), sampler.deviceConeWidth());
    }

    @Test
    public void aPenaltyChainKeepsTheStrictConeRefusalButHasADeviceCone() {
        Sampler sampler = Sampler.create(penalties().seed(1).build());
        assertEquals(-1, sampler.coneWidth());
        assertEquals(16, sampler.deviceConeWidth());
        assertTrue(sampler.prefilterRequired());
    }

    @Test
    public void aPenaltyAfterTopKIsNotPartOfThePrefilterRow() {
        SamplerConfig config = penalties().samplers("top_k;penalties;temperature").seed(1).build();
        Sampler sampler = Sampler.create(config);
        sampler.accept(new int[]{1, 2, 3}, 0, 3);
        assertFalse(sampler.prefilterRow().penalties());
        assertTrue(sampler.prefilterRow().isEmpty());
    }

    @Test
    public void aWideDryWindowKeepsTheHostPath() {
        SamplerConfig config = SamplerConfig.builder().temperature(0.8f).topK(16).penaltyLastN(0)
                .dryMultiplier(0.8f).dryBase(2f).dryAllowedLength(2)
                .dryPenaltyLastN(Sampler.MAX_PREFILTER_DRY_WINDOW + 1)
                .drySequenceBreakers(Arrays.asList("ab")).samplers("dry;top_k;temperature").seed(1).build();
        Sampler sampler = Sampler.create(config, new int[0], new FakeVocab());
        int[] history = new int[Sampler.MAX_PREFILTER_DRY_WINDOW + 1];
        Arrays.fill(history, 1);
        sampler.accept(history, 0, history.length);
        assertEquals(-1, sampler.deviceConeWidth());
    }
}
