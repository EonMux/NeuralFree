package com.eonmux.neuralfree.gguf.ops.sparse;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

/**
 * The Step5 sparse-attention math against independently written scalar references: zero-centered RMS and LayerNorm,
 * partial NeoX rope, the weighted-ReLU indexer score, top-k region selection, the region-summary update and the
 * sparse GQA attention with its per-head gate.
 */
public class CpuStep5SparseTest {

    private static final float EPS = 1e-5f;

    @Test
    public void zeroCenteredRmsNormScalesByOnePlusWeight() {
        float[] x = {1f, -2f, 3f, -4f, 5f, -6f};
        float[] w = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.05f};
        float[] out = new float[6];
        float rms = 0f;
        for (float v : x) {
            rms += v * v;
        }
        rms = (float) Math.sqrt(rms / x.length + EPS);
        float[] want = new float[6];
        for (int i = 0; i < x.length; i++) {
            want[i] = x[i] / rms * (1f + w[i]);
        }
        CpuStep5Sparse.rmsNormZeroCentered(x, 0, 6, w, EPS, out, 0);
        assertArrayEquals(want, out, 1e-5f);
    }

    @Test
    public void layerNormMatchesAThreePassReference() {
        float[] x = {2f, 4f, 4f, 4f, 5f, 5f, 7f, 9f};
        float[] w = {1f, 0.5f, -1f, 2f, 1f, 1f, 1f, 0.25f};
        float[] b = {0f, 1f, 0f, -1f, 0.5f, 0f, 0f, 2f};
        double mean = 0;
        for (float v : x) {
            mean += v;
        }
        mean /= x.length;
        double var = 0;
        for (float v : x) {
            var += (v - mean) * (v - mean);
        }
        var /= x.length;
        float[] want = new float[8];
        for (int i = 0; i < x.length; i++) {
            want[i] = (float) ((x[i] - mean) / Math.sqrt(var + EPS)) * w[i] + b[i];
        }
        float[] out = new float[8];
        CpuStep5Sparse.layerNorm(x, 0, 8, w, b, EPS, out, 0);
        assertArrayEquals(want, out, 1e-5f);
    }

    @Test
    public void neoxRopeRotatesLeadingPairsAndCopiesTheTail() {
        float[] x = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        float[] out = new float[8];
        int pos = 3;
        double theta = 10000.0;
        int ropeDim = 4;
        CpuStep5Sparse.neoxRope(x, 0, 8, ropeDim, pos, theta, out, 0);
        for (int i = 0; i < 2; i++) {
            double angle = pos / Math.pow(theta, 2.0 * i / ropeDim);
            float c = (float) Math.cos(angle);
            float s = (float) Math.sin(angle);
            assertEquals(x[i] * c - x[i + 2] * s, out[i], 1e-6f);
            assertEquals(x[i] * s + x[i + 2] * c, out[i + 2], 1e-6f);
        }
        assertEquals(5f, out[4], 0f);
        assertEquals(8f, out[7], 0f);
        float[] identity = new float[8];
        CpuStep5Sparse.neoxRope(x, 0, 8, ropeDim, 0, theta, identity, 0);
        for (int i = 0; i < 8; i++) {
            assertEquals(x[i], identity[i], 1e-6f);
        }
        assertThrows(IllegalArgumentException.class,
                () -> CpuStep5Sparse.neoxRope(x, 0, 8, 5, 1, theta, out, 0));
    }

    @Test
    public void indexerScoresAreReLUWeightedDots() {
        int heads = 2;
        int dim = 3;
        int regions = 3;
        float[] q = {1f, 2f, 3f, -1f, 0f, 1f};
        float[] w = {0.5f, 2f};
        float[] summary = {
                1f, 0f, 0f,
                -1f, -1f, -1f,
                0f, 1f, 1f};
        float[] out = new float[3];
        CpuStep5Sparse.indexerScores(heads, dim, q, 0, dim, w, 0, 1, summary, 0, dim, regions, out, 0);
        float r0 = (1f * 0.5f) + (0f * 2f);
        float r1 = 0f;
        float r2 = 5f * 0.5f + 1f * 2f;
        assertEquals(r0, out[0], 1e-6f);
        assertEquals(r1, out[1], 1e-6f);
        assertEquals(r2, out[2], 1e-6f);
    }

    @Test
    public void selectionLeftPadsWhenEveryClosedRegionFits() {
        int[] out = new int[4];
        int n = CpuStep5Sparse.selectRegions(new float[]{1f, 2f}, 0, 2, 3, out, 0);
        assertEquals(3, n);
        assertArrayEquals(new int[]{-1, 0, 1, 2}, out);
        int[] empty = new int[4];
        assertEquals(1, CpuStep5Sparse.selectRegions(new float[0], 0, 0, 3, empty, 0));
        assertArrayEquals(new int[]{-1, -1, -1, 0}, empty);
    }

    @Test
    public void selectionTakesTheTopKSortedButAlwaysTheOpenRegion() {
        float[] scores = {0.5f, 4f, 1f, -2f, 3f, 0.5f};
        int[] out = new int[4];
        int n = CpuStep5Sparse.selectRegions(scores, 0, 6, 3, out, 0);
        assertEquals(4, n);
        assertArrayEquals(new int[]{1, 2, 4, 6}, out);
    }

    @Test
    public void selectionKeepsLowestIdsOnATie() {
        float[] scores = {5f, 5f, 5f, 5f};
        int[] out = new int[3];
        CpuStep5Sparse.selectRegions(scores, 0, 4, 2, out, 0);
        assertArrayEquals(new int[]{0, 1, 4}, out);
        float[] nan = {1f, Float.NaN};
        assertThrows(IllegalArgumentException.class, () -> CpuStep5Sparse.selectRegions(nan, 0, 2, 1, out, 0));
    }

    @Test
    public void summaryIsThePerDimensionSoftmaxMeanOfTheClosedRegion() {
        int proxy = 3;
        int block = 4;
        int capacity = block;
        float[] key = new float[proxy];
        float[] z = new float[proxy];
        float[][] ring = new float[capacity][];
        float[][] mean = new float[2][];
        float[][] expectedKey = new float[block][proxy];
        float[][] expectedZ = new float[block][proxy];
        for (int p = 0; p < block; p++) {
            for (int d = 0; d < proxy; d++) {
                key[d] = (p + 1) * (d + 1) * 0.5f;
                z[d] = 0.25f * p - 0.5f * d;
                expectedKey[p][d] = key[d];
                expectedZ[p][d] = z[d];
            }
            CpuStep5Sparse.summaryUpdate(p, block, key, 0, z, 0, proxy, capacity, ring, mean);
            if (p < block - 1) {
                assertNull("region 0 must stay open at position " + p, mean[0]);
            }
        }
        float[] want = softmaxMean(expectedKey, expectedZ, proxy);
        assertArrayEquals(want, mean[0], 1e-5f);
    }

    @Test
    public void summaryRingWrapsAcrossRegionsAndOverwritesOldRows() {
        int proxy = 2;
        int block = 2;
        int capacity = 4;
        float[] key = new float[proxy];
        float[] z = new float[proxy];
        float[][] ring = new float[capacity][];
        float[][] mean = new float[4][];
        float[][] expectedKey = new float[6][proxy];
        float[][] expectedZ = new float[6][proxy];
        for (int p = 0; p < 6; p++) {
            for (int d = 0; d < proxy; d++) {
                key[d] = p - 0.5f * d;
                z[d] = 0.1f * p + d;
                expectedKey[p][d] = key[d];
                expectedZ[p][d] = z[d];
            }
            CpuStep5Sparse.summaryUpdate(p, block, key, 0, z, 0, proxy, capacity, ring, mean);
        }
        assertArrayEquals(softmaxMean(Arrays.copyOfRange(expectedKey, 0, 2),
                Arrays.copyOfRange(expectedZ, 0, 2), proxy), mean[0], 1e-5f);
        assertArrayEquals(softmaxMean(Arrays.copyOfRange(expectedKey, 2, 4),
                Arrays.copyOfRange(expectedZ, 2, 4), proxy), mean[1], 1e-5f);
        assertArrayEquals(softmaxMean(Arrays.copyOfRange(expectedKey, 4, 6),
                Arrays.copyOfRange(expectedZ, 4, 6), proxy), mean[2], 1e-5f);
        assertThrows(IllegalArgumentException.class,
                () -> CpuStep5Sparse.summaryUpdate(0, block, key, 0, z, 0, proxy, 1, ring, mean));
    }

    @Test
    public void sparseGqaAttendsOnlyWithinTheSelectionAndAppliesTheGate() {
        int tokens = 4;
        int heads = 4;
        int kvHeads = 2;
        int headDim = 2;
        int block = 2;
        int headsPerGroup = heads / kvHeads;
        int slots = 2;

        float[] q = new float[tokens * heads * headDim];
        float[] k = new float[tokens * kvHeads * headDim];
        float[] v = new float[tokens * kvHeads * headDim];
        float[] gate = new float[tokens * heads];
        for (int t = 0; t < tokens; t++) {
            for (int h = 0; h < heads; h++) {
                for (int d = 0; d < headDim; d++) {
                    q[(t * heads + h) * headDim + d] = 0.1f * (t + 1) + 0.05f * h - 0.03f * d;
                }
                gate[t * heads + h] = 0.2f * h - 0.4f * t;
            }
        }
        for (int t = 0; t < tokens; t++) {
            for (int g = 0; g < kvHeads; g++) {
                for (int d = 0; d < headDim; d++) {
                    k[(t * kvHeads + g) * headDim + d] = 0.2f * (t - g) + 0.1f * d;
                    v[(t * kvHeads + g) * headDim + d] = 1f / (1 + t + g + d);
                }
            }
        }
        int[] sel = new int[tokens * kvHeads * slots];
        int[] positions = {0, 1, 2, 3};
        for (int t = 0; t < tokens; t++) {
            for (int g = 0; g < kvHeads; g++) {
                int base = (t * kvHeads + g) * slots;
                if (t < 2) {
                    sel[base] = -1;
                    sel[base + 1] = 0;
                } else {
                    sel[base] = 0;
                    sel[base + 1] = 1;
                }
            }
        }
        float[] out = new float[tokens * heads * headDim];
        float scale = (float) (1.0 / Math.sqrt(headDim));
        CpuStep5Sparse.sparseGqa(tokens, heads, kvHeads, headDim, block, q, 0, heads * headDim,
                k, 0, kvHeads * headDim, v, 0, kvHeads * headDim, scale,
                sel, 0, kvHeads * slots, slots, positions, 0, gate, 0, heads, out, 0, heads * headDim);

        float[] want = reference(tokens, heads, kvHeads, headDim, block, q, k, v, scale, sel, slots, positions,
                gate, heads);
        assertArrayEquals(want, out, 1e-5f);

        CpuStep5Sparse.sparseGqa(tokens, heads, kvHeads, headDim, block, q, 0, heads * headDim,
                k, 0, kvHeads * headDim, v, 0, kvHeads * headDim, scale,
                sel, 0, kvHeads * slots, slots, positions, 0, null, 0, heads, out, 0, heads * headDim);
        float[] ungated = reference(tokens, heads, kvHeads, headDim, block, q, k, v, scale, sel, slots, positions,
                null, heads);
        assertArrayEquals(ungated, out, 1e-5f);
    }

    @Test
    public void sparseGqaSkipsClosedRegionsOutsideTheSelection() {
        float[] q = {1f, 2f};
        float[] k = {3f, -1f};
        float[] v = {7f, -2f};
        int[] sel = {0, 0};
        int[] positions = {0, 1};
        float[] out = new float[2];
        CpuStep5Sparse.sparseGqa(2, 1, 1, 1, 1, q, 0, 1, k, 0, 1, v, 0, 1, 1f,
                sel, 0, 1, 1, positions, 0, null, 0, 1, out, 0, 1);
        assertEquals(7f, out[0], 1e-6f);
        assertEquals(7f, out[1], 1e-6f);
    }

    private static float[] softmaxMean(float[][] keys, float[][] zs, int proxy) {
        int n = keys.length;
        float[] want = new float[proxy];
        for (int d = 0; d < proxy; d++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int p = 0; p < n; p++) {
                max = Math.max(max, zs[p][d]);
            }
            double num = 0;
            double den = 0;
            for (int p = 0; p < n; p++) {
                double w = Math.exp(zs[p][d] - max);
                num += w * keys[p][d];
                den += w;
            }
            want[d] = (float) (num / den);
        }
        return want;
    }

    private static float[] reference(int tokens, int heads, int kvHeads, int headDim, int block,
                                     float[] q, float[] k, float[] v, float scale, int[] sel, int slots,
                                     int[] positions, float[] gate, int gateStride) {
        int headsPerGroup = heads / kvHeads;
        float[] out = new float[tokens * heads * headDim];
        for (int t = 0; t < tokens; t++) {
            for (int h = 0; h < heads; h++) {
                int group = h / headsPerGroup;
                int base = (t * kvHeads + group) * slots;
                double max = Double.NEGATIVE_INFINITY;
                int count = 0;
                int[] attended = new int[slots * block];
                for (int s = 0; s < slots; s++) {
                    int region = sel[base + s];
                    if (region < 0) {
                        continue;
                    }
                    for (int p = region * block; p <= Math.min(region * block + block - 1, positions[t]); p++) {
                        attended[count++] = p;
                        max = Math.max(max, logit(q, k, t, h, p, group, heads, headDim, kvHeads, scale));
                    }
                }
                double sum = 0;
                for (int i = 0; i < count; i++) {
                    sum += Math.exp(logit(q, k, t, h, attended[i], group, heads, headDim, kvHeads, scale) - max);
                }
                float gateValue = gate == null ? 1f : CpuStep5Sparse.sigmoid(gate[t * gateStride + h]);
                for (int d = 0; d < headDim; d++) {
                    double acc = 0;
                    for (int i = 0; i < count; i++) {
                        double p = Math.exp(logit(q, k, t, h, attended[i], group, heads, headDim, kvHeads, scale)
                                - max) / sum;
                        acc += p * v[(attended[i] * kvHeads + group) * headDim + d];
                    }
                    out[(t * heads + h) * headDim + d] = (float) (acc * gateValue);
                }
            }
        }
        return out;
    }

    private static double logit(float[] q, float[] k, int t, int h, int p, int group, int heads, int headDim,
                                int kvHeads, float scale) {
        double dot = 0;
        for (int d = 0; d < headDim; d++) {
            dot += (double) q[(t * heads + h) * headDim + d] * k[(p * kvHeads + group) * headDim + d];
        }
        return dot * scale;
    }
}
