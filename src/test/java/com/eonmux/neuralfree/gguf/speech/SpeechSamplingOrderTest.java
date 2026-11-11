package com.eonmux.neuralfree.gguf.speech;

import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * The primitive top-k order of {@link SpeechSampling} against the boxed stable sort under {@link Float#compare} it
 * replaces (kept here as the reference): the same kept indices in the same order, and the same drawn index, over
 * random rows with ties, signed zeros, infinities and NaN, for cuts of none, one, fifty and the whole row.
 */
public class SpeechSamplingOrderTest {

    @Test
    public void ordersLikeTheStableDescendingSort() {
        Random random = new Random(11);
        for (int trial = 0; trial < 300; trial++) {
            float[] row = row(random, 1 + random.nextInt(trial < 50 ? 20 : 2048));
            for (int k : new int[]{1, 2, 50, row.length, row.length + 3}) {
                int kept = Math.min(k, row.length);
                int[] want = Arrays.copyOf(referenceOrder(row), kept);
                assertArrayEquals("trial " + trial + " k " + k, want, SpeechSampling.topOrder(row, kept));
            }
        }
    }

    @Test
    public void drawsTheSameIndexAsTheBoxedSort() {
        Random random = new Random(12);
        for (int trial = 0; trial < 400; trial++) {
            float[] row = finiteRow(random, 1 + random.nextInt(2048));
            for (int topK : new int[]{0, 1, 50, row.length}) {
                for (float topP : new float[]{1f, 0.9f, 0.5f, 0.05f}) {
                    float rand = random.nextFloat();
                    assertEquals("trial " + trial + " k " + topK + " p " + topP,
                            referenceDraw(row, topK, topP, rand), SpeechSampling.draw(row, topK, topP, rand));
                }
            }
        }
    }

    private static float[] row(Random random, int n) {
        float[] specials = {Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 0f, -0f, 1f, -1f};
        float[] row = new float[n];
        for (int i = 0; i < n; i++) {
            int pick = random.nextInt(10);
            row[i] = pick < 2 ? specials[random.nextInt(specials.length)]
                    : pick < 5 ? (random.nextInt(8) - 4) * 0.5f : (float) random.nextGaussian() * 4f;
        }
        return row;
    }

    private static float[] finiteRow(Random random, int n) {
        float[] row = new float[n];
        for (int i = 0; i < n; i++) {
            row[i] = random.nextInt(4) == 0 ? (random.nextInt(6) - 3) : (float) random.nextGaussian() * 3f;
        }
        return row;
    }

    private static int[] referenceOrder(float[] values) {
        Integer[] order = new Integer[values.length];
        for (int i = 0; i < values.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int byValue = Float.compare(values[b], values[a]);
            return byValue != 0 ? byValue : Integer.compare(a, b);
        });
        int[] out = new int[order.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = order[i];
        }
        return out;
    }

    /** The draw over the boxed sort, as {@code do_sampling} transcribes it. */
    private static int referenceDraw(float[] values, int topK, float topP, float rand) {
        int n = values.length;
        int[] order = referenceOrder(values);
        int kept = topK > 0 && topK < n ? topK : n;
        float[] sorted = new float[kept];
        for (int i = 0; i < kept; i++) {
            sorted[i] = values[order[i]];
        }
        if (topP < 1f) {
            softmax(sorted);
            float cumulative = 0f;
            int keep = kept;
            for (int i = 0; i < kept; i++) {
                cumulative += sorted[i];
                if (cumulative >= topP) {
                    keep = i + 1;
                    break;
                }
            }
            for (int i = 0; i < keep; i++) {
                sorted[i] = values[order[i]];
            }
            kept = keep;
        }
        float[] probabilities = Arrays.copyOf(sorted, kept);
        softmax(probabilities);
        float cumulative = 0f;
        for (int i = 0; i < kept; i++) {
            cumulative += probabilities[i];
            if (cumulative > rand) {
                return order[i];
            }
        }
        return order[kept - 1];
    }

    private static void softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            if (v > max) {
                max = v;
            }
        }
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double e = Math.exp((double) x[i] - max);
            x[i] = (float) e;
            sum += e;
        }
        if (!(sum > 0.0) || Double.isInfinite(sum)) {
            Arrays.fill(x, 1.0f / x.length);
            return;
        }
        float scale = (float) (1.0 / sum);
        for (int i = 0; i < x.length; i++) {
            x[i] *= scale;
        }
    }
}
