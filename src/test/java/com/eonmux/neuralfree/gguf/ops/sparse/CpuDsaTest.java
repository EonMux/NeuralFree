package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.ops.sparse.CpuDsaIndexer.RowReader;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the DSA CPU math ({@link CpuDsaIndexer}, {@link CpuTopK}, {@link CpuSparseMla}) against direct
 * evaluations of the llama.cpp formulas, and for the {@link SparseKernels} launch helpers.
 */
public class CpuDsaTest {

    private static final float EPS = 1e-5f;

    @Test
    public void visibilityFollowsLlamaKvMask() {
        assertFalse(CpuDsaIndexer.visible(true, SwaType.NONE, 0, -1, 5));
        assertTrue(CpuDsaIndexer.visible(true, SwaType.NONE, 0, 5, 5));
        assertFalse(CpuDsaIndexer.visible(true, SwaType.NONE, 0, 6, 5));
        assertTrue(CpuDsaIndexer.visible(false, SwaType.NONE, 0, 6, 5));
        // STANDARD masks p1 - p0 >= n_swa
        assertTrue(CpuDsaIndexer.visible(true, SwaType.STANDARD, 4, 7, 10));
        assertFalse(CpuDsaIndexer.visible(true, SwaType.STANDARD, 4, 6, 10));
        // CHUNKED masks p0 < (p1 / n) * n
        assertTrue(CpuDsaIndexer.visible(true, SwaType.CHUNKED, 4, 8, 10));
        assertFalse(CpuDsaIndexer.visible(true, SwaType.CHUNKED, 4, 7, 10));
        // SYMMETRIC masks |p1 - p0| > n / 2
        assertTrue(CpuDsaIndexer.visible(false, SwaType.SYMMETRIC, 4, 12, 10));
        assertFalse(CpuDsaIndexer.visible(false, SwaType.SYMMETRIC, 4, 13, 10));
        assertFalse(CpuDsaIndexer.visible(false, SwaType.SYMMETRIC, 4, 7, 10));
    }

    @Test
    public void indexerScoresMatchLightningIndexerFormula() {
        int nTokens = 2, nKv = 5, heads = 3, dim = 6;
        Random rnd = new Random(7);
        float[] iq = randoms(rnd, nTokens * heads * dim);
        float[] w = randoms(rnd, nTokens * heads);
        float[] keys = randoms(rnd, nKv * dim);
        int[] qPos = {3, 4};
        int[] kvPos = {0, 1, -1, 3, 4};
        float[] scores = new float[nTokens * nKv];
        CpuDsaIndexer.scores(nTokens, nKv, heads, dim, true, SwaType.NONE, 0, iq, 0, w, 0, keys, 0,
                qPos, 0, kvPos, 0, scores, 0);
        for (int t = 0; t < nTokens; t++) {
            for (int s = 0; s < nKv; s++) {
                float got = scores[t * nKv + s];
                boolean vis = kvPos[s] >= 0 && kvPos[s] <= qPos[t];
                if (!vis) {
                    assertEquals(Float.NEGATIVE_INFINITY, got, 0.0f);
                    continue;
                }
                double want = 0.0;
                for (int h = 0; h < heads; h++) {
                    double qk = 0.0;
                    for (int d = 0; d < dim; d++) {
                        qk += iq[(t * heads + h) * dim + d] * keys[s * dim + d];
                    }
                    want += Math.max(qk, 0.0) * w[t * heads + h];
                }
                assertEquals("t=" + t + " s=" + s, want, got, EPS);
            }
        }
        float[] viaReader = new float[nTokens * nKv];
        CpuDsaIndexer.scores(nTokens, nKv, heads, dim, true, SwaType.NONE, 0, iq, 0, w, 0,
                RowReader.ofArray(keys, 0, dim), qPos, 0, kvPos, 0, viaReader, 0);
        assertArrayEquals(scores, viaReader, 0.0f);
    }

    @Test
    public void indexerAppliesSlidingWindow() {
        int dim = 2;
        float[] iq = {1f, 0f};
        float[] w = {1f};
        float[] keys = {1f, 0f, 2f, 0f, 3f, 0f};
        int[] kvPos = {0, 1, 2};
        float[] scores = new float[3];
        CpuDsaIndexer.scores(1, 3, 1, dim, true, SwaType.STANDARD, 2, iq, 0, w, 0, keys, 0,
                new int[]{2}, 0, kvPos, 0, scores, 0);
        assertArrayEquals(new float[]{Float.NEGATIVE_INFINITY, 2f, 3f}, scores, 0.0f);
    }

    @Test
    public void topKSelectsLargestWithLowestIndexTies() {
        float[] row = {1f, 5f, 3f, 5f, Float.NEGATIVE_INFINITY, 3f, 3f, Float.NaN, 2f};
        int[] out = new int[4];
        int kk = CpuTopK.topKRow(row, 0, row.length, 4, out, 0);
        assertEquals(4, kk);
        // above the 4th value (3): columns 1, 3 in column order; then the first two 3s: columns 2, 5
        assertArrayEquals(new int[]{1, 3, 2, 5}, out);
    }

    @Test
    public void topKPadsWhenFewerEligibleThanK() {
        float[] row = {Float.NEGATIVE_INFINITY, 0.5f, Float.NaN, -2f};
        int[] out = new int[6];
        int kk = CpuTopK.topKRow(row, 0, row.length, 6, out, 0);
        assertEquals(2, kk);
        assertArrayEquals(new int[]{1, 3, -1, -1, -1, -1}, out);
        float[] none = {Float.NEGATIVE_INFINITY, Float.NaN};
        int[] out2 = new int[3];
        assertEquals(0, CpuTopK.topKRow(none, 0, none.length, 3, out2, 0));
        assertArrayEquals(new int[]{-1, -1, -1}, out2);
    }

    @Test
    public void topKTreatsSignedZerosAsEqual() {
        float[] row = {-0.0f, 0.0f, -0.0f, -1f};
        int[] out = new int[2];
        CpuTopK.topKRow(row, 0, row.length, 2, out, 0);
        assertArrayEquals(new int[]{0, 1}, out);
    }

    @Test
    public void topKMatchesSortedReferenceOnRandomRows() {
        Random rnd = new Random(42);
        int nRows = 6, nCols = 997;
        for (int k : new int[]{1, 7, 64, 500, 997, 2048}) {
            float[] scores = new float[nRows * nCols];
            for (int i = 0; i < scores.length; i++) {
                int pick = rnd.nextInt(10);
                if (pick == 0) {
                    scores[i] = Float.NEGATIVE_INFINITY;
                } else if (pick == 1) {
                    scores[i] = rnd.nextInt(5) * 0.25f;
                } else if (pick == 2) {
                    scores[i] = rnd.nextBoolean() ? 0.0f : -0.0f;
                } else {
                    scores[i] = (float) rnd.nextGaussian() * 1e3f;
                }
            }
            int[] out = new int[nRows * k];
            CpuTopK.topKRows(scores, 0, nRows, nCols, k, out, 0);
            for (int r = 0; r < nRows; r++) {
                int[] want = referenceTopK(scores, r * nCols, nCols, k);
                int[] got = Arrays.copyOfRange(out, r * k, (r + 1) * k);
                assertArrayEquals("k=" + k + " row=" + r, want, got);
            }
        }
    }

    @Test
    public void sparseMlaMatchesMaskedSoftmaxReference() {
        Random rnd = new Random(3);
        int nHead = 3, kvLora = 8, ropeDim = 4, dK = kvLora + ropeDim, nRows = 10;
        float[] cache = randoms(rnd, nRows * dK);
        float[] q = randoms(rnd, nHead * dK + 5);
        int[] sel = {4, -1, 0, 9, 2, -1};
        float[] sinks = {0.3f, -1.0f, 2.0f};
        for (float softcap : new float[]{0.0f, 1.5f}) {
            for (boolean hasSinks : new boolean[]{false, true}) {
                float kqScale = 0.37f;
                float[] out = new float[nHead * kvLora];
                CpuSparseMla.decode(nHead, kvLora, ropeDim, sel.length, kqScale, softcap, hasSinks, q, 5,
                        RowReader.ofArray(cache, 0, dK), sel, 0, sinks, 0, out, 0);
                for (int h = 0; h < nHead; h++) {
                    double[] want = referenceAttention(q, 5 + h * dK, cache, dK, kvLora, sel, 0, sel.length,
                            kqScale, softcap, hasSinks, sinks[h]);
                    for (int d = 0; d < kvLora; d++) {
                        assertEquals("softcap=" + softcap + " sinks=" + hasSinks + " h=" + h + " d=" + d,
                                want[d], out[h * kvLora + d], EPS);
                    }
                }
            }
        }
    }

    @Test
    public void sparseMlaWithNothingSelectedWritesZeros() {
        float[] cache = {1f, 2f, 3f};
        float[] q = {1f, 1f, 1f};
        float[] out = {9f, 9f};
        CpuSparseMla.decode(1, 2, 1, 3, 1.0f, 0.0f, true, q, 0, RowReader.ofArray(cache, 0, 3),
                new int[]{-1, -1, -1}, 0, new float[]{0.5f}, 0, out, 0);
        assertArrayEquals(new float[]{0f, 0f}, out, 0.0f);
    }

    @Test
    public void sparseMlaPrefillEqualsPerTokenDecode() {
        Random rnd = new Random(11);
        int nTokens = 3, nHead = 2, kvLora = 6, ropeDim = 2, dK = kvLora + ropeDim, nSel = 4, qStride = nHead * dK + 3;
        float[] cache = randoms(rnd, 7 * dK);
        float[] q = randoms(rnd, nTokens * qStride);
        int[] sel = {0, 3, -1, 6, 1, 2, 5, -1, 6, 4, 3, 0};
        float[] sinks = {0.1f, -0.2f};
        float[] prefill = new float[nTokens * nHead * kvLora];
        RowReader rows = RowReader.ofArray(cache, 0, dK);
        CpuSparseMla.prefill(nTokens, nHead, kvLora, ropeDim, nSel, 0.5f, 0.0f, true, q, 0, qStride, rows,
                sel, 0, sinks, 0, prefill, 0);
        for (int t = 0; t < nTokens; t++) {
            float[] decode = new float[nHead * kvLora];
            CpuSparseMla.decode(nHead, kvLora, ropeDim, nSel, 0.5f, 0.0f, true, q, t * qStride, rows,
                    sel, t * nSel, sinks, 0, decode, 0);
            assertArrayEquals("t=" + t, decode,
                    Arrays.copyOfRange(prefill, t * nHead * kvLora, (t + 1) * nHead * kvLora), 0.0f);
        }
    }

    @Test
    public void launchGeometryAndLimits() {
        assertArrayEquals(new long[]{3L * 2 * 128}, SparseKernels.indexerGlobal(3, 200, 128));
        assertArrayEquals(new long[]{5L * 256}, SparseKernels.topKGlobal(5, 256));
        assertArrayEquals(new long[]{16L * 128}, SparseKernels.sparseDecodeGlobal(16, 128));
        assertArrayEquals(new long[]{4L * 16 * 128}, SparseKernels.sparsePrefillGlobal(4, 16, 128));
        assertEquals(100, SparseKernels.selectedCount(100, 2048));
        assertEquals(2048, SparseKernels.selectedCount(5000, 2048));
        SparseKernels.checkTopK(SparseKernels.TOPK_MAX);
        SparseKernels.checkLatentCache(KvDType.Q8_0, 512, 64);
        SparseKernels.checkLatentCache(KvDType.F16, 500, 3);
        assertEquals("LLM_MLA_SparseDecode_Q8_0", SparseKernels.sparseDecode(KvDType.Q8_0));
        assertEquals("LLM_MLA_SparsePrefill_F32", SparseKernels.sparsePrefill(KvDType.F32));
        expectFailure(() -> SparseKernels.checkTopK(SparseKernels.TOPK_MAX + 1));
        expectFailure(() -> SparseKernels.checkIndexer(64, SparseKernels.INDEXER_MAX_HEAD_DIM + 1));
        SparseKernels.checkIndexerMask(SwaType.NONE, 0);
        SparseKernels.checkIndexerMask(SwaType.CHUNKED, 4);
        expectFailure(() -> SparseKernels.checkIndexerMask(SwaType.CHUNKED, 0));
        expectFailure(() -> SparseKernels.checkIndexerMask(null, 4));
        expectFailure(() -> SparseKernels.checkSparseMlaLocal(16));
        expectFailure(() -> SparseKernels.checkTopKLocal(96));
        expectFailure(() -> SparseKernels.checkLatentCache(KvDType.Q8_0, 500, 3));
        assertEquals("LLM_DSA_IndexerScore_F32", SparseKernels.indexerScore(KvDType.F32));
        assertEquals("LLM_DSA_IndexerScore_F16", SparseKernels.indexerScore(KvDType.F16));
        assertEquals("LLM_DSA_IndexerScore_Q8_0", SparseKernels.indexerScore(KvDType.Q8_0));
        SparseKernels.checkIndexerCache(KvDType.Q8_0, 128);
        SparseKernels.checkIndexerCache(KvDType.F16, 100);
        SparseKernels.checkIndexerCache(KvDType.F32, 7);
        expectFailure(() -> SparseKernels.checkIndexerCache(KvDType.Q8_0, 80));
        assertEquals(1 + 3 * KvDType.all().size(), SparseKernels.ALL.size());
        for (KvDType t : KvDType.all()) {
            assertTrue(SparseKernels.ALL.contains(SparseKernels.indexerScore(t)));
        }
        assertEquals("LLM_DSA_IndexerScore_Q5_0", SparseKernels.indexerScore(KvDType.Q5_0));
        expectFailure(() -> SparseKernels.checkIndexerCache(KvDType.Q4_1, 80));
    }

    private static void expectFailure(Runnable r) {
        try {
            r.run();
            fail("expected a rejection");
        } catch (IllegalArgumentException | UnsupportedOperationException expected) {
            assertTrue(expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }

    private static float[] randoms(Random rnd, int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) rnd.nextGaussian();
        }
        return v;
    }

    /** Top-k by descending value (ties: lower column first), emitted as the kernel orders it. */
    private static int[] referenceTopK(float[] scores, int off, int nCols, int k) {
        List<Integer> eligible = new ArrayList<>();
        for (int c = 0; c < nCols; c++) {
            if (scores[off + c] > Float.NEGATIVE_INFINITY) {
                eligible.add(c);
            }
        }
        eligible.sort((a, b) -> {
            float x = scores[off + a] + 0.0f;
            float y = scores[off + b] + 0.0f;
            int cmp = Float.compare(y, x);
            return cmp != 0 ? cmp : Integer.compare(a, b);
        });
        int kk = Math.min(k, eligible.size());
        int[] out = new int[k];
        Arrays.fill(out, -1);
        if (kk == 0) {
            return out;
        }
        float kth = scores[off + eligible.get(kk - 1)] + 0.0f;
        List<Integer> above = new ArrayList<>();
        List<Integer> equal = new ArrayList<>();
        for (int i = 0; i < kk; i++) {
            int c = eligible.get(i);
            if (scores[off + c] + 0.0f > kth) {
                above.add(c);
            } else {
                equal.add(c);
            }
        }
        above.sort(Integer::compare);
        int n = 0;
        for (int c : above) {
            out[n++] = c;
        }
        for (int c : equal) {
            out[n++] = c;
        }
        return out;
    }

    private static double[] referenceAttention(float[] q, int qOff, float[] cache, int dK, int kvLora, int[] sel,
                                               int selOff, int nSel, float kqScale, float softcap, boolean hasSinks,
                                               float sink) {
        double[] logits = new double[nSel];
        double m = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < nSel; j++) {
            int r = sel[selOff + j];
            if (r < 0) {
                logits[j] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double qk = 0.0;
            for (int d = 0; d < dK; d++) {
                qk += q[qOff + d] * cache[r * dK + d];
            }
            double s = qk * kqScale;
            logits[j] = softcap > 0 ? softcap * Math.tanh(s / softcap) : s;
            m = Math.max(m, logits[j]);
        }
        if (hasSinks) {
            m = Math.max(m, sink);
        }
        double l = hasSinks ? Math.exp(sink - m) : 0.0;
        double[] out = new double[kvLora];
        for (int j = 0; j < nSel; j++) {
            if (logits[j] == Double.NEGATIVE_INFINITY) {
                continue;
            }
            double p = Math.exp(logits[j] - m);
            l += p;
            int r = sel[selOff + j];
            for (int d = 0; d < kvLora; d++) {
                out[d] += p * cache[r * dK + d];
            }
        }
        for (int d = 0; d < kvLora; d++) {
            out[d] /= l;
        }
        return out;
    }
}
