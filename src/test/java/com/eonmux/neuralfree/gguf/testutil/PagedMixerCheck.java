package com.eonmux.neuralfree.gguf.testutil;

import java.util.SplittableRandom;
import java.util.function.Function;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertFalse;

/**
 * The equivalence check of a mixer's paged multi-sequence steps: two sequences run their rows in the same chunks
 * twice, once one sequence at a time on contiguous memories ({@link CpuMixerRig#contiguous}) and once as paged
 * two-segment steps ({@link CpuMixerRig#paged}), and every layer's outputs of every row must agree. A second check
 * runs a non-causal step: the two sequences batched equal each sequence's own non-causal step, and differ from the
 * causal step.
 */
public final class PagedMixerCheck {

    private PagedMixerCheck() {
    }

    /** Random inputs of {@code rows} rows of {@code width} values. */
    public static float[] inputs(long seed, int rows, int width) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] v = new float[rows * width];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (-1 + 2 * rng.nextDouble());
        }
        return v;
    }

    /**
     * Runs sequences A and B in chunks {@code chunksA[s]} / {@code chunksB[s]} rows per step and compares the paged
     * two-segment steps with the contiguous runs.
     *
     * @param rig opens a fresh rig: paged ({@code true}) or contiguous
     */
    public static void assertBatchMatchesSequences(Function<Boolean, CpuMixerRig> rig, int width, int layers,
                                                   int[] chunksA, int[] chunksB, float tol) {
        int totalA = sum(chunksA);
        int totalB = sum(chunksB);
        float[] inA = inputs(11, totalA, width);
        float[] inB = inputs(23, totalB, width);
        float[][] refA = contiguousRun(rig, width, chunksA, inA);
        float[][] refB = contiguousRun(rig, width, chunksB, inB);
        try (CpuMixerRig paged = rig.apply(true)) {
            int posA = 0;
            int posB = 0;
            for (int step = 0; step < chunksA.length; step++) {
                int a = chunksA[step];
                int b = chunksB[step];
                float[] in = new float[(a + b) * width];
                System.arraycopy(inA, posA * width, in, 0, a * width);
                System.arraycopy(inB, posB * width, in, a * width, b * width);
                float[] out = paged.paged(new int[]{0, 1}, new int[]{a, b}, new int[]{posA, posB}, in, false);
                for (int l = 0; l < layers; l++) {
                    String what = "step " + step + ", layer " + l + ", sequence ";
                    assertArrayEquals(what + "A", slice(refA[step], l, a, width),
                            paged.rows(out, l, a + b, 0, a), tol);
                    assertArrayEquals(what + "B", slice(refB[step], l, b, width),
                            paged.rows(out, l, a + b, a, b), tol);
                }
                posA += a;
                posB += b;
            }
        }
    }

    /**
     * A first non-causal step of {@code a} rows of sequence A and {@code b} rows of B: the batch equals each
     * sequence's own non-causal step, and A's non-causal outputs differ from its causal ones.
     */
    public static void assertNonCausalBatchKeepsSequencesApart(Function<Boolean, CpuMixerRig> rig, int width,
                                                               int layers, int a, int b, float tol) {
        float[] inA = inputs(31, a, width);
        float[] inB = inputs(47, b, width);
        float[] aloneA;
        float[] aloneB;
        float[] causalA;
        try (CpuMixerRig one = rig.apply(true)) {
            aloneA = one.paged(new int[]{0}, new int[]{a}, new int[]{0}, inA, true);
            aloneB = one.paged(new int[]{1}, new int[]{b}, new int[]{0}, inB, true);
        }
        try (CpuMixerRig one = rig.apply(true)) {
            causalA = one.paged(new int[]{0}, new int[]{a}, new int[]{0}, inA, false);
        }
        assertFalse("the non-causal step lets earlier rows see later ones", java.util.Arrays.equals(aloneA, causalA));
        try (CpuMixerRig two = rig.apply(true)) {
            float[] in = new float[(a + b) * width];
            System.arraycopy(inA, 0, in, 0, a * width);
            System.arraycopy(inB, 0, in, a * width, b * width);
            float[] out = two.paged(new int[]{0, 1}, new int[]{a, b}, new int[]{0, 0}, in, true);
            for (int l = 0; l < layers; l++) {
                assertArrayEquals("layer " + l + ", sequence A", slice(aloneA, l, a, width),
                        two.rows(out, l, a + b, 0, a), tol);
                assertArrayEquals("layer " + l + ", sequence B", slice(aloneB, l, b, width),
                        two.rows(out, l, a + b, a, b), tol);
            }
        }
    }

    private static float[][] contiguousRun(Function<Boolean, CpuMixerRig> rig, int width, int[] chunks, float[] in) {
        float[][] out = new float[chunks.length][];
        try (CpuMixerRig one = rig.apply(false)) {
            int pos = 0;
            for (int step = 0; step < chunks.length; step++) {
                float[] rows = new float[chunks[step] * width];
                System.arraycopy(in, pos * width, rows, 0, rows.length);
                out[step] = one.contiguous(pos, chunks[step], rows);
                pos += chunks[step];
            }
        }
        return out;
    }

    private static float[] slice(float[] out, int layer, int n, int width) {
        float[] r = new float[n * width];
        System.arraycopy(out, layer * n * width, r, 0, r.length);
        return r;
    }

    private static int sum(int[] v) {
        int s = 0;
        for (int x : v) {
            s += x;
        }
        return s;
    }
}
