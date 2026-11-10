package com.eonmux.neuralfree.gguf.engine;

import com.eonmux.neuralfree.gguf.engine.sampling.DryStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TopPStage;
import com.eonmux.neuralfree.gguf.engine.sampling.TypicalStage;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * The whole-row host stages that avoid a full sort or boxed tables (top-p and locally typical over a selected prefix,
 * then over the order sorted bucket by bucket; DRY over primitive tables) leave the candidate set bit-identical to the
 * llama.cpp-shaped reference that sorts the whole row and boxes every lookup: the same survivors in the same order with
 * the same logits, probabilities and sorted flag, over peaked, flat, tied and banned rows, nuclei narrower and wider
 * than the first selection, {@code min_keep} beyond it, and DRY windows with repeats, restart sequences and
 * single-token breakers.
 */
public class SamplerTruncationParityTest {

    private static final int VOCAB = 6000;

    /** A row of {@code VOCAB} logits: scaled noise, quantized when {@code ties}, a few banned when {@code bans}. */
    private static float[] row(Random rnd, float scale, boolean ties, boolean bans) {
        float[] r = new float[VOCAB];
        for (int i = 0; i < VOCAB; i++) {
            float v = (float) rnd.nextGaussian() * scale;
            r[i] = ties ? Math.round(v * 2f) / 2f : v;
        }
        if (bans) {
            for (int i = 0; i < VOCAB; i += 7) {
                r[i] = Float.NEGATIVE_INFINITY;
            }
        }
        return r;
    }

    private static Sampler.Candidates load(float[] row, boolean sorted) {
        Sampler.Candidates c = new Sampler.Candidates();
        c.load(row);
        if (sorted) {
            c.sortDescending();
        }
        return c;
    }

    private static void assertSame(String what, Sampler.Candidates expected, Sampler.Candidates actual) {
        assertEquals(what + " size", expected.size(), actual.size());
        assertEquals(what + " sorted", expected.sorted(), actual.sorted());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(what + " id " + i, expected.id(i), actual.id(i));
            assertEquals(what + " logit " + i, Float.floatToIntBits(expected.logit(i)),
                    Float.floatToIntBits(actual.logit(i)));
            assertEquals(what + " prob " + i, Float.floatToIntBits(expected.prob(i)),
                    Float.floatToIntBits(actual.prob(i)));
        }
    }

    /** llama_sampler_top_p_apply over a whole sort. */
    private static void referenceTopP(Sampler.Candidates c, float topP, int minKeep) {
        c.softmax();
        if (!c.sorted()) {
            c.sortDescending();
        }
        float cum = 0f;
        int last = c.size();
        for (int i = 0; i < c.size(); i++) {
            cum += c.prob(i);
            if (cum >= topP && i + 1 >= minKeep) {
                last = i + 1;
                break;
            }
        }
        c.truncate(last);
    }

    /** llama_sampler_typical_apply over a whole sort of the shifted distances. */
    private static void referenceTypical(Sampler.Candidates c, float typicalP, int minKeep) {
        if (!c.sorted()) {
            c.sortDescending();
        }
        c.softmax();
        int n = c.size();
        float entropy = 0f;
        for (int i = 0; i < n; i++) {
            float p = c.prob(i);
            if (p > 0f) {
                entropy += -p * (float) Math.log(p);
            }
        }
        long[] k = new long[n];
        for (int i = 0; i < n; i++) {
            float p = c.prob(i);
            float shifted = p > 0f ? Math.abs(-(float) Math.log(p) - entropy) : Float.POSITIVE_INFINITY;
            k[i] = Sampler.Candidates.ascendingKey(shifted, i);
        }
        Arrays.sort(k);
        float cum = 0f;
        int last = n;
        for (int i = 0; i < n; i++) {
            cum += c.prob((int) k[i]);
            if (cum > typicalP && (minKeep == 0 || i >= minKeep - 1)) {
                last = i + 1;
                break;
            }
        }
        c.permute(k, last);
        c.truncate(last);
        c.setSorted(false);
    }

    @Test
    public void topPMatchesTheWholeSortOverEveryRowShape() {
        Random rnd = new Random(11);
        float[] scales = {0.01f, 0.3f, 1f, 4f, 12f, 40f};
        float[] tops = {0.05f, 0.5f, 0.9f, 0.99f, 0.9999999f};
        int[] keeps = {0, 1, 100, 700, 3000};
        int cases = 0;
        for (float scale : scales) {
            for (boolean ties : new boolean[]{false, true}) {
                float[] r = row(rnd, scale, ties, rnd.nextBoolean());
                for (float top : tops) {
                    for (int keep : keeps) {
                        for (boolean sorted : new boolean[]{false, true}) {
                            Sampler.Candidates expected = load(r, sorted);
                            referenceTopP(expected, top, keep);
                            Sampler.Candidates actual = load(r, sorted);
                            new TopPStage(top, keep).apply(actual);
                            assertSame("top-p " + top + " keep " + keep + " scale " + scale, expected, actual);
                            cases++;
                        }
                    }
                }
            }
        }
        assertEquals(600, cases);
    }

    /**
     * Rows the full sort orders by itself (a NaN logit, an infinite highest logit) and a row whose tail lies past
     * the bucketed range keep matching the whole sort.
     */
    @Test
    public void unusualRowsMatchTheWholeSort() {
        Random rnd = new Random(14);
        List<float[]> rows = new ArrayList<>();
        float[] nan = row(rnd, 2f, false, false);
        nan[17] = Float.NaN;
        rows.add(nan);
        float[] inf = row(rnd, 2f, false, false);
        inf[99] = Float.POSITIVE_INFINITY;
        rows.add(inf);
        float[] spread = row(rnd, 1f, false, true);
        for (int i = 0; i < VOCAB; i += 3) {
            spread[i] -= 200f;
        }
        rows.add(spread);
        for (int t = 0; t < rows.size(); t++) {
            for (float p : new float[]{0.5f, 0.99f, 0.9999999f}) {
                Sampler.Candidates expected = load(rows.get(t), false);
                referenceTopP(expected, p, 1);
                Sampler.Candidates actual = load(rows.get(t), false);
                new TopPStage(p, 1).apply(actual);
                assertSame("top-p " + p + " row " + t, expected, actual);
                expected = load(rows.get(t), false);
                referenceTypical(expected, p, 1);
                actual = load(rows.get(t), false);
                new TypicalStage(p, 1).apply(actual);
                assertSame("typical " + p + " row " + t, expected, actual);
            }
        }
    }

    @Test
    public void typicalMatchesTheWholeSortOverEveryRowShape() {
        Random rnd = new Random(12);
        float[] scales = {0.01f, 0.3f, 1f, 4f, 12f, 40f};
        float[] typicals = {0.05f, 0.5f, 0.9f, 0.99f, 0.9999999f};
        int[] keeps = {0, 1, 100, 700, 3000};
        for (float scale : scales) {
            for (boolean ties : new boolean[]{false, true}) {
                float[] r = row(rnd, scale, ties, rnd.nextBoolean());
                for (float typical : typicals) {
                    for (int keep : keeps) {
                        Sampler.Candidates expected = load(r, false);
                        referenceTypical(expected, typical, keep);
                        Sampler.Candidates actual = load(r, false);
                        new TypicalStage(typical, keep).apply(actual);
                        assertSame("typical " + typical + " keep " + keep + " scale " + scale, expected, actual);
                    }
                }
            }
        }
    }

    @Test
    public void aStageReusedAcrossRowsKeepsMatching() {
        Random rnd = new Random(13);
        TopPStage topP = new TopPStage(0.9f, 1);
        TypicalStage typical = new TypicalStage(0.8f, 1);
        for (int t = 0; t < 40; t++) {
            float[] r = row(rnd, 0.5f + t * 0.3f, t % 3 == 0, t % 2 == 0);
            Sampler.Candidates expected = load(r, false);
            referenceTopP(expected, 0.9f, 1);
            Sampler.Candidates actual = load(r, false);
            topP.apply(actual);
            assertSame("top-p row " + t, expected, actual);
            expected = load(r, false);
            referenceTypical(expected, 0.8f, 1);
            actual = load(r, false);
            typical.apply(actual);
            assertSame("typical row " + t, expected, actual);
        }
    }

    // ------------------------------------------------------------------ DRY

    /** llama_sampler_dry_apply with boxed tables, over the window {@code win} (newest first). */
    private static void referenceDry(Sampler.Candidates c, float multiplier, float base, int allowedLength,
                                     int[] win, List<int[]> tokenBreakers) {
        Map<Integer, List<List<Integer>>> breakers = new HashMap<>();
        for (int[] b : tokenBreakers) {
            List<Integer> tail = new ArrayList<>();
            for (int j = 1; j < b.length; j++) {
                tail.add(b[j]);
            }
            List<List<Integer>> tails = breakers.computeIfAbsent(b[0], k -> new ArrayList<>());
            if (!tails.contains(tail)) {
                tails.add(tail);
            }
        }
        int lastNRepeat = win.length;
        if (lastNRepeat <= allowedLength) {
            return;
        }
        int[] repeat = new int[lastNRepeat];
        int repLimit = lastNRepeat;
        for (int i = 0; i < lastNRepeat; i++) {
            List<List<Integer>> heads = breakers.get(win[i]);
            if (heads == null) {
                continue;
            }
            int longestMatch = -1;
            for (List<Integer> tail : heads) {
                int seqLen = tail.size();
                if (seqLen > longestMatch && seqLen <= i) {
                    boolean match = true;
                    for (int offset = 0; offset < seqLen; offset++) {
                        if (tail.get(offset) != win[i - offset - 1]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        longestMatch = seqLen;
                    }
                }
            }
            if (longestMatch >= 0) {
                repLimit = i - longestMatch;
                break;
            }
        }
        if (repLimit < allowedLength) {
            return;
        }
        int last = lastNRepeat - 1;
        int rt = 0;
        int lt = 0;
        for (int k = 1; k < lastNRepeat; k++) {
            if (k > rt) {
                int n = 0;
                while (n + k < lastNRepeat && win[n] == win[n + k]) {
                    n++;
                }
                repeat[last - k] = Math.min(n, repLimit);
                if (n > 0) {
                    lt = k;
                    rt = k + n - 1;
                }
            } else {
                int p = k - lt;
                int rightPartLen = rt - k + 1;
                if (repeat[last - p] < rightPartLen) {
                    repeat[last - k] = Math.min(repeat[last - p], repLimit);
                } else {
                    int i = rt + 1;
                    while (i < lastNRepeat && win[i] == win[i - k]) {
                        i++;
                    }
                    repeat[last - k] = Math.min(i - k, repLimit);
                    lt = k;
                    rt = i - 1;
                }
            }
        }
        Map<Integer, Integer> maxTokenRepeat = new HashMap<>();
        for (int i = 0; i < lastNRepeat - 1; i++) {
            int repeatLen = repeat[i];
            if (repeatLen >= allowedLength) {
                int token = win[lastNRepeat - 2 - i];
                Integer previous = maxTokenRepeat.get(token);
                if (previous == null || previous < repeatLen) {
                    maxTokenRepeat.put(token, repeatLen);
                }
            }
        }
        float maxExponentFloat = 0;
        if (base > 1.000001f) {
            maxExponentFloat = 88.7228391f / (float) Math.log(base);
        }
        int maxExponent = (int) maxExponentFloat > 0 ? (int) maxExponentFloat : 0;
        for (int i = 0; i < c.size(); i++) {
            Integer repeatLen = maxTokenRepeat.get(c.id(i));
            List<List<Integer>> heads = breakers.get(c.id(i));
            boolean single = heads != null && heads.stream().anyMatch(List::isEmpty);
            if (repeatLen != null && !single) {
                int repeatExp = repeatLen - allowedLength;
                if (maxExponent > 0 && repeatExp > maxExponent) {
                    repeatExp = maxExponent;
                }
                c.setLogit(i, c.logit(i) - multiplier * (float) Math.pow(base, repeatExp));
            }
        }
        c.setSorted(false);
    }

    @Test
    public void dryMatchesTheBoxedReferenceOverRepeatingWindows() {
        Random rnd = new Random(14);
        List<int[]> breakers = List.of(new int[]{5}, new int[]{9, 3}, new int[]{9, 3, 4}, new int[]{12, 7},
                new int[]{-3});
        int[] lastNs = {8, 64, 512};
        int[] alloweds = {0, 2, 4};
        int checked = 0;
        for (int lastN : lastNs) {
            for (int allowed : alloweds) {
                for (int trial = 0; trial < 6; trial++) {
                    DryStage stage = new DryStage(0.8f, 1.75f, allowed, lastN, breakers);
                    int accepted = lastN + rnd.nextInt(lastN);
                    int[] history = repeatingHistory(rnd, accepted, trial);
                    for (int t : history) {
                        stage.accept(t);
                    }
                    int window = Math.min(accepted, lastN);
                    int[] win = new int[window];
                    for (int i = 0; i < window; i++) {
                        win[i] = history[accepted - 1 - i];
                    }
                    float[] r = row(rnd, 2f, trial % 2 == 0, trial % 3 == 0);
                    Sampler.Candidates expected = load(r, trial % 2 == 1);
                    referenceDry(expected, 0.8f, 1.75f, allowed, win, breakers);
                    Sampler.Candidates actual = load(r, trial % 2 == 1);
                    stage.apply(actual);
                    assertSame("DRY lastN " + lastN + " allowed " + allowed + " trial " + trial, expected,
                            actual);
                    Sampler.Candidates again = load(r, trial % 2 == 1);
                    stage.apply(again);
                    assertSame("DRY reapplied", expected, again);
                    checked++;
                }
            }
        }
        assertEquals(54, checked);
    }

    /** Tokens from a small alphabet with copied stretches; some trials plant the restart sequences. */
    private static int[] repeatingHistory(Random rnd, int n, int trial) {
        int[] h = new int[n];
        for (int i = 0; i < n; i++) {
            if (i > 8 && rnd.nextInt(3) > 0) {
                h[i] = h[i - 1 - rnd.nextInt(Math.min(i, 6))];
            } else {
                h[i] = rnd.nextInt(16);
            }
        }
        if (trial % 3 == 2 && n > 6) {
            int at = n / 2 + rnd.nextInt(n / 2 - 2);
            h[at] = 9;
            h[at + 1] = 3;
        }
        return h;
    }
}
