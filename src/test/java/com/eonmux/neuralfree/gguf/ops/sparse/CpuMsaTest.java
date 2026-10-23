package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.memory.cpu.CpuIndexerCache;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Tests for the MSA and QSA CPU math ({@link CpuMsa}, {@link CpuQsa}) against direct evaluations of the llama.cpp
 * graphs (dense position mask, max pooling, masked softmax over all cells, pooled-key formulas), and for the
 * {@link MsaKernels} helpers.
 */
public class CpuMsaTest {

    private static final float EPS = 1e-4f;

    /** F32 heap cache with {@code nHead} heads per row; V aliases the first {@code headDimV} K values when latent. */
    private static final class ArrayKv implements KvAccessor {
        private final int rows;
        private final int nHead;
        private final int dimK;
        private final int dimV;
        private final boolean latent;
        private final float[] k;
        private final float[] v;

        ArrayKv(int rows, int nHead, int dimK, int dimV, boolean latent) {
            this.rows = rows;
            this.nHead = nHead;
            this.dimK = dimK;
            this.dimV = dimV;
            this.latent = latent;
            this.k = new float[rows * nHead * dimK];
            this.v = latent ? k : new float[rows * nHead * dimV];
        }

        @Override
        public int layer() {
            return 0;
        }

        @Override
        public KvLayout layout() {
            return latent
                    ? KvLayout.latent(rows, false, nHead * dimK, dimV, KvDType.F32, KvWindow.FULL_CAUSAL)
                    : KvLayout.of(rows, false, nHead * dimK, nHead * dimV, KvDType.F32, KvDType.F32, KvWindow.FULL_CAUSAL);
        }

        @Override
        public int group() {
            return 0;
        }

        @Override
        public int nHeadKv() {
            return nHead;
        }

        @Override
        public int headDimK() {
            return dimK;
        }

        @Override
        public int headDimV() {
            return dimV;
        }

        @Override
        public void readK(int row, int head, float[] dst, int off) {
            System.arraycopy(k, (row * nHead + head) * dimK, dst, off, dimK);
        }

        @Override
        public void readV(int row, int head, float[] dst, int off) {
            int stride = latent ? nHead * dimK : nHead * dimV;
            System.arraycopy(v, row * stride + head * dimV, dst, off, dimV);
        }

        @Override
        public void writeK(int row, float[] src, int off) {
            System.arraycopy(src, off, k, row * nHead * dimK, nHead * dimK);
        }

        @Override
        public void writeKSlice(int row, int from, int n, float[] src, int off) {
            System.arraycopy(src, off, k, row * nHead * dimK + from, n);
        }

        @Override
        public void writeV(int row, float[] src, int off) {
            if (latent) {
                throw new UnsupportedOperationException("latent V is part of the K row");
            }
            System.arraycopy(src, off, v, row * nHead * dimV, nHead * dimV);
        }
    }

    private static float[] random(Random rnd, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (float) rnd.nextGaussian();
        }
        return a;
    }

    private static ArrayKv filled(Random rnd, int rows, int nHead, int dimK, int dimV, boolean latent) {
        ArrayKv kv = new ArrayKv(rows, nHead, dimK, dimV, latent);
        for (int r = 0; r < rows; r++) {
            kv.writeK(r, random(rnd, nHead * dimK), 0);
            if (!latent) {
                kv.writeV(r, random(rnd, nHead * dimV), 0);
            }
        }
        return kv;
    }

    /** Row map after writing positions {@code [0, nWritten)} into a ring of {@code rows} rows. */
    private static int[] ringPositions(int rows, int nWritten) {
        int[] kvPos = new int[rows];
        Arrays.fill(kvPos, -1);
        for (int p = 0; p < nWritten; p++) {
            kvPos[p % rows] = p;
        }
        return kvPos;
    }

    /** The llama.cpp pos_slot map (set_input_pos_slot): cell of each position, -1 when not cached. */
    private static int[] posSlot(int nPos, int[] kvPos) {
        int[] slot = new int[nPos];
        Arrays.fill(slot, -1);
        for (int r = 0; r < kvPos.length; r++) {
            if (kvPos[r] >= 0 && kvPos[r] < nPos) {
                slot[kvPos[r]] = r;
            }
        }
        return slot;
    }

    /** Top-k columns of one row, highest first, ties to the lower column, -INF never selected, -1 padding. */
    private static int[] topK(float[] scores, int off, int n, int k) {
        int[] sel = new int[k];
        Arrays.fill(sel, -1);
        boolean[] used = new boolean[n];
        for (int s = 0; s < k; s++) {
            int best = -1;
            for (int c = 0; c < n; c++) {
                if (!used[c] && scores[off + c] > Float.NEGATIVE_INFINITY && (best < 0 || scores[off + c] > scores[off + best])) {
                    best = c;
                }
            }
            if (best < 0) {
                break;
            }
            used[best] = true;
            sel[s] = best;
        }
        return sel;
    }

    private static double dot(float[] a, int aOff, float[] b, int bOff, int n) {
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += (double) a[aOff + i] * b[bOff + i];
        }
        return acc;
    }

    /** minimax-m3 block scores: dense pos_mask scores, ggml_pool_2d MAX per block, local-force bias. */
    private static float[] referenceBlockScores(int nTokens, int nPos, int nH, int dim, int blk, int local,
                                                float[] iq, ArrayKv ik, int[] qPos, int[] kvPos) {
        int nblk = (nPos + blk - 1) / blk;
        int[] slot = posSlot(nPos, kvPos);
        float[] key = new float[dim];
        float[] out = new float[nTokens * nH * nblk];
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nH; h++) {
                float[] sc = new float[nPos];
                for (int p = 0; p < nPos; p++) {
                    boolean visible = slot[p] >= 0 && p <= qPos[t];
                    if (visible) {
                        ik.readK(slot[p], 0, key, 0);
                        sc[p] = (float) dot(iq, (t * nH + h) * dim, key, 0, dim);
                    } else {
                        sc[p] = Float.NEGATIVE_INFINITY;
                    }
                }
                int lBlk = qPos[t] / blk;
                for (int b = 0; b < nblk; b++) {
                    float m = Float.NEGATIVE_INFINITY;
                    for (int p = b * blk; p < Math.min(nPos, (b + 1) * blk); p++) {
                        m = Math.max(m, sc[p]);
                    }
                    boolean forced = false;
                    for (int l = 0; l < local; l++) {
                        forced |= lBlk - l == b && lBlk - l >= 0;
                    }
                    out[(t * nH + h) * nblk + b] = m + (forced ? 1e30f : 0f);
                }
            }
        }
        return out;
    }

    /** Attention over every cell with the given per-position selection mask (softmax in double precision). */
    private static void referenceAttn(float[] q, int qBase, int hd, float scale, ArrayKv kc, ArrayKv vc, int kvh,
                                      boolean[] selectedPos, int[] slot, int pQ, float[] out, int outBase) {
        float[] row = new float[hd];
        double m = Double.NEGATIVE_INFINITY;
        double[] s = new double[slot.length];
        for (int p = 0; p < slot.length; p++) {
            s[p] = Double.NEGATIVE_INFINITY;
            if (selectedPos[p] && slot[p] >= 0 && p <= pQ) {
                kc.readK(slot[p], kvh, row, 0);
                s[p] = dot(q, qBase, row, 0, hd) * scale;
                m = Math.max(m, s[p]);
            }
        }
        double l = 0;
        double[] acc = new double[hd];
        for (int p = 0; p < slot.length; p++) {
            if (s[p] == Double.NEGATIVE_INFINITY) {
                continue;
            }
            double w = Math.exp(s[p] - m);
            l += w;
            vc.readV(slot[p], kvh, row, 0);
            for (int d = 0; d < hd; d++) {
                acc[d] += w * row[d];
            }
        }
        for (int d = 0; d < hd; d++) {
            out[outBase + d] = l > 0 ? (float) (acc[d] / l) : 0f;
        }
    }

    @Test
    public void blockScoreMatchesMaskedMaxPoolWithLocalBias() {
        Random rnd = new Random(11);
        int nTokens = 3, nPos = 23, nH = 2, dim = 12, blk = 4, local = 2, rows = 16;
        int[] kvPos = ringPositions(rows, nPos);
        int[] qPos = {20, 21, 22};
        ArrayKv ik = filled(rnd, rows, 1, dim, dim, false);
        float[] iq = random(rnd, 5 + nTokens * nH * dim);
        float[] iqTok = Arrays.copyOfRange(iq, 5, iq.length);
        int nblk = CpuMsa.blockCount(nPos, blk);
        assertEquals(6, nblk);
        float[] got = new float[nTokens * nH * nblk];
        CpuMsa.blockScore(nTokens, nPos, nH, dim, blk, local, rows, iq, 5, ik, qPos, kvPos, got);
        float[] want = referenceBlockScores(nTokens, nPos, nH, dim, blk, local, iqTok, ik, qPos, kvPos);
        for (int i = 0; i < want.length; i++) {
            if (want[i] == Float.NEGATIVE_INFINITY) {
                assertEquals("block " + i, Float.NEGATIVE_INFINITY, got[i], 0f);
            } else {
                assertEquals("block " + i, want[i], got[i], Math.abs(want[i]) * 1e-6f + EPS);
            }
        }
        // Positions 0..6 left the 16-row ring, so block 0 (0..3) is empty and block 1 (4..7) keeps only position 7.
        assertEquals(Float.NEGATIVE_INFINITY, got[0], 0f);
        assertTrue(got[1] > Float.NEGATIVE_INFINITY);
        // Token at position 20: L = 5, the local blocks 4 and 5 carry the bias.
        assertTrue(got[4] > 1e29f && got[5] > 1e29f && got[3] < 1e29f);
    }

    @Test
    public void gatherAttnMatchesMaskedSoftmaxOverSelectedBlocks() {
        Random rnd = new Random(12);
        int nTokens = 2, nPos = 30, nHead = 4, nKv = 2, hd = 8, idxDim = 6, blk = 4, local = 1, k = 3, rows = 32;
        float scale = 1f / (float) Math.sqrt(hd);
        int[] kvPos = ringPositions(rows, nPos);
        kvPos[9] = -1;
        int[] qPos = {17, 29};
        ArrayKv ik = filled(rnd, rows, 1, idxDim, idxDim, false);
        float[] iq = random(rnd, nTokens * nKv * idxDim);
        int nblk = CpuMsa.blockCount(nPos, blk);
        float[] bs = new float[nTokens * nKv * nblk];
        CpuMsa.blockScore(nTokens, nPos, nKv, idxDim, blk, local, rows, iq, 0, ik, qPos, kvPos, bs);
        int[] blocks = new int[nTokens * nKv * k];
        for (int r = 0; r < nTokens * nKv; r++) {
            System.arraycopy(topK(bs, r * nblk, nblk, k), 0, blocks, r * k, k);
        }
        ArrayKv kc = filled(rnd, rows, nKv, hd, hd, false);
        int qStride = nHead * hd + 3;
        float[] q = random(rnd, 2 + nTokens * qStride);
        int outStride = nHead * hd + 5;
        float[] got = new float[1 + nTokens * outStride];
        CpuMsa.gatherAttn(nTokens, nHead, nKv, hd, blk, k, scale, nKv, false, rows, q, 2, qStride, kc, kc, qPos, kvPos,
                blocks, got, 1, outStride);
        int[] slot = posSlot(nPos, kvPos);
        float[] want = new float[got.length];
        for (int t = 0; t < nTokens; t++) {
            for (int h = 0; h < nHead; h++) {
                int kvh = h / (nHead / nKv);
                boolean[] sel = new boolean[nPos];
                for (int s = 0; s < k; s++) {
                    int b = blocks[(t * nKv + kvh) * k + s];
                    for (int p = b * blk; b >= 0 && p < Math.min(nPos, (b + 1) * blk); p++) {
                        sel[p] = true;
                    }
                }
                referenceAttn(q, 2 + t * qStride + h * hd, hd, scale, kc, kc, kvh, sel, slot, qPos[t], want,
                        1 + t * outStride + h * hd);
            }
        }
        assertArrayEquals(want, got, EPS);
        // The local block of each token (the block holding its own position) is always selected.
        for (int r = 0; r < nTokens * nKv; r++) {
            int t = r / nKv;
            assertTrue(Arrays.stream(blocks, r * k, r * k + k).anyMatch(b -> b == qPos[t] / blk));
        }
    }

    @Test
    public void gatherAttnPadsShortSelectionsAndZeroesEmptyRows() {
        Random rnd = new Random(13);
        int hd = 4, blk = 2, rows = 8;
        int[] kvPos = ringPositions(rows, 3);
        ArrayKv kc = filled(rnd, rows, 1, hd, hd, true);
        float[] q = random(rnd, 2 * hd);
        int[] blocks = {0, 1, -1, -1, -1, -1};
        float[] got = new float[2 * hd];
        CpuMsa.gatherAttn(2, 1, 1, hd, blk, 3, 0.5f, 1, false, rows, q, 0, hd, kc, kc, new int[]{2, 2}, kvPos,
                blocks, got, 0, hd);
        float[] want = new float[hd];
        boolean[] sel = {true, true, true};
        referenceAttn(q, 0, hd, 0.5f, kc, kc, 0, sel, posSlot(3, kvPos), 2, want, 0);
        assertArrayEquals(want, Arrays.copyOfRange(got, 0, hd), EPS);
        assertArrayEquals(new float[hd], Arrays.copyOfRange(got, hd, 2 * hd), 0f);
    }

    @Test
    public void gatherAttnWithTailMatchesKpoolSelection() {
        Random rnd = new Random(14);
        int nTokens = 3, nPos = 23, nHead = 4, nKv = 2, hd = 8, kpool = 4, k = 2, rows = 32;
        float scale = 0.3f;
        int[] kvPos = ringPositions(rows, nPos);
        int[] qPos = {15, 21, 22};
        int[] blocks = {1, 3, 0, 4, 2, -1};
        ArrayKv kc = filled(rnd, rows, nKv, hd, hd, false);
        float[] q = random(rnd, nTokens * nHead * hd);
        float[] got = new float[nTokens * nHead * hd];
        CpuMsa.gatherAttn(nTokens, nHead, nKv, hd, kpool, k, scale, 1, true, rows, q, 0, nHead * hd, kc, kc, qPos,
                kvPos, blocks, got, 0, nHead * hd);
        int[] slot = posSlot(nPos, kvPos);
        float[] want = new float[got.length];
        for (int t = 0; t < nTokens; t++) {
            boolean[] sel = new boolean[nPos];
            for (int s = 0; s < k; s++) {
                int b = blocks[t * k + s];
                for (int m = 0; b >= 0 && m < kpool; m++) {
                    sel[b * kpool + m] = true;
                }
            }
            // select_tail: positions p - j for j < (p + 1) % kpool (llama-memory-hybrid-idx.cpp:946-966).
            int nTail = (qPos[t] + 1) % kpool;
            for (int j = 0; j < nTail; j++) {
                sel[qPos[t] - j] = true;
            }
            for (int h = 0; h < nHead; h++) {
                referenceAttn(q, (t * nHead + h) * hd, hd, scale, kc, kc, h / (nHead / nKv), sel, slot, qPos[t], want,
                        (t * nHead + h) * hd);
            }
        }
        assertArrayEquals(want, got, EPS);
    }

    @Test
    public void kpoolPoolMatchesGatedAndMeanFormulas() {
        Random rnd = new Random(15);
        int idxDim = 5, kpool = 4, rows = 16;
        // Gated rows [key | gate | pooled].
        ArrayKv gated = filled(rnd, rows, 1, 3 * idxDim, 3 * idxDim, false);
        float[] ape = random(rnd, kpool * idxDim);
        float[] got = new float[2 * idxDim];
        CpuQsa.kpoolPool(2, 3, MsaKernels.KPOOL_MODE_GATED, kpool, idxDim, idxDim, rows, ape, gated, got);
        float[] row = new float[3 * idxDim];
        for (int j = 0; j < 2; j++) {
            for (int d = 0; d < idxDim; d++) {
                double[] x = new double[kpool];
                double[] key = new double[kpool];
                double mx = Double.NEGATIVE_INFINITY;
                for (int m = 0; m < kpool; m++) {
                    gated.readK(((3 + j) * kpool + m) % rows, 0, row, 0);
                    x[m] = row[idxDim + d] + ape[m * idxDim + d];
                    key[m] = row[d];
                    mx = Math.max(mx, x[m]);
                }
                double sum = 0;
                for (int m = 0; m < kpool; m++) {
                    sum += Math.exp(x[m] - mx);
                }
                double want = 0;
                for (int m = 0; m < kpool; m++) {
                    want += Math.exp(x[m] - mx) / sum * key[m];
                }
                assertEquals(want, got[j * idxDim + d], EPS);
            }
        }
        // Mean rows [raw key | pooled].
        ArrayKv mean = filled(rnd, rows, 1, 2 * idxDim, 2 * idxDim, false);
        float[] pooled = new float[idxDim];
        CpuQsa.kpoolPool(1, 1, MsaKernels.KPOOL_MODE_MEAN, kpool, idxDim, 0, rows, null, mean, pooled);
        float[] mrow = new float[2 * idxDim];
        for (int d = 0; d < idxDim; d++) {
            double want = 0;
            for (int m = 0; m < kpool; m++) {
                mean.readK(kpool + m, 0, mrow, 0);
                want += mrow[d];
            }
            assertEquals(want / kpool, pooled[d], EPS);
        }
    }

    @Test
    public void kpoolStoreWritesOnlyTheLastMemberPooledSlot() {
        Random rnd = new Random(16);
        int idxDim = 3, kpool = 4, rows = 8;
        ArrayKv idx = filled(rnd, rows, 1, 2 * idxDim, 2 * idxDim, false);
        float[][] before = new float[rows][2 * idxDim];
        for (int r = 0; r < rows; r++) {
            idx.readK(r, 0, before[r], 0);
        }
        float[] pooled = {1f, 2f, 3f, 4f, 5f, 6f};
        CpuQsa.kpoolStore(2, 1, kpool, idxDim, idxDim, rows, pooled, idx);
        // Pools 1 and 2 end at positions 7 and 11: rows 7 and 3 of the 8-row ring.
        float[] row = new float[2 * idxDim];
        for (int r = 0; r < rows; r++) {
            idx.readK(r, 0, row, 0);
            float[] want = before[r].clone();
            if (r == 7) {
                System.arraycopy(pooled, 0, want, idxDim, idxDim);
            } else if (r == 3) {
                System.arraycopy(pooled, idxDim, want, idxDim, idxDim);
            }
            assertArrayEquals("row " + r, want, row, 0f);
        }
    }

    /**
     * A Q8_0 indexer cache (the context's type_k, llama-memory-hybrid-idx.cpp:71): the pooled store rewrites the rep
     * row, the key and gate slots keep exactly the values they held and the pooled slot reads back as the Q8_0
     * round trip of the pooled key, as a ggml_set_rows into the pooled view does.
     */
    @Test
    public void kpoolStoreIntoQ8CacheKeepsTheOtherSlots() {
        Random rnd = new Random(18);
        int idxDim = 32, kpool = 4, rows = 8, width = 3 * idxDim;
        LayerMemoryPlan.IndexerPlan plan =
                LayerMemoryPlan.IndexerPlan.of(rows, width, KvDType.Q8_0, RotationPlan.NONE);
        CpuIndexerCache idx = CpuIndexerCache.allocate(0, plan,
                MemoryPlan.KvGroup.of(0, rows, false, KvWindow.FULL_CAUSAL));
        float[][] before = new float[rows][width];
        for (int r = 0; r < rows; r++) {
            idx.writeK(r, random(rnd, width), 0);
            idx.readK(r, 0, before[r], 0);
        }
        float[] pooled = random(rnd, idxDim);
        CpuQsa.kpoolStore(1, 0, kpool, idxDim, 2 * idxDim, rows, pooled, idx);

        float[] roundTrip = new float[width];
        System.arraycopy(pooled, 0, roundTrip, 2 * idxDim, idxDim);
        CpuIndexerCache q8 = CpuIndexerCache.allocate(0, LayerMemoryPlan.IndexerPlan.of(1, width, KvDType.Q8_0,
                RotationPlan.NONE), MemoryPlan.KvGroup.of(0, 1, false, KvWindow.FULL_CAUSAL));
        q8.writeK(0, roundTrip, 0);
        q8.readK(0, 0, roundTrip, 0);

        float[] row = new float[width];
        for (int r = 0; r < rows; r++) {
            idx.readK(r, 0, row, 0);
            float[] want = before[r].clone();
            if (r == kpool - 1) {
                System.arraycopy(roundTrip, 2 * idxDim, want, 2 * idxDim, idxDim);
            }
            assertArrayEquals("row " + r, want, row, 0f);
        }
    }

    @Test
    public void kpoolScoreMatchesReluWeightedIndexerWithPoolMask() {
        Random rnd = new Random(17);
        int nTokens = 2, nH = 3, idxDim = 6, kpool = 4, rows = 32, nPos = 18;
        int nPool = CpuQsa.poolCount(nPos, kpool);
        assertEquals(4, nPool);
        int[] kvPos = ringPositions(rows, nPos);
        int[] qPos = {10, 17};
        ArrayKv idx = filled(rnd, rows, 1, 3 * idxDim, 3 * idxDim, false);
        float[] iq = random(rnd, nTokens * nH * idxDim);
        float[] w = random(rnd, 1 + nTokens * nH);
        float wScale = (float) (1.0 / Math.sqrt(idxDim * nH));
        float[] got = new float[nTokens * nPool];
        CpuQsa.kpoolScore(nTokens, nPool, nH, idxDim, kpool, 2 * idxDim, rows, wScale, iq, 0, w, 1, idx, qPos, kvPos, got);
        float[] row = new float[3 * idxDim];
        for (int t = 0; t < nTokens; t++) {
            for (int i = 0; i < nPool; i++) {
                int end = (i + 1) * kpool - 1;
                if (end > qPos[t]) {
                    assertEquals(Float.NEGATIVE_INFINITY, got[t * nPool + i], 0f);
                    continue;
                }
                idx.readK(end % rows, 0, row, 0);
                double want = 0;
                for (int h = 0; h < nH; h++) {
                    double a = dot(iq, (t * nH + h) * idxDim, row, 2 * idxDim, idxDim);
                    want += Math.max(a, 0) * w[1 + t * nH + h] * wScale;
                }
                assertEquals(want, got[t * nPool + i], EPS);
            }
        }
        // Token at position 10 sees pools 0 and 1 (ends 3, 7) only.
        assertTrue(got[1] > Float.NEGATIVE_INFINITY && got[2] == Float.NEGATIVE_INFINITY);
    }

    @Test
    public void poolCountsFollowKpoolBuildState() {
        assertEquals(1, CpuQsa.firstNewPool(5, 4));
        assertEquals(2, CpuQsa.newPoolCount(5, 7, 4));
        assertEquals(0, CpuQsa.newPoolCount(4, 3, 4));
        assertEquals(1, CpuQsa.newPoolCount(4, 4, 4));
        assertEquals(3, CpuQsa.poolCount(12, 4));
        assertEquals(0, CpuQsa.poolCount(3, 4));
        assertEquals(4, MsaKernels.selectedPools(16, 4));
        assertEquals(2, MsaKernels.selectedPools(10, 4));
        assertEquals(0, MsaKernels.selectedPools(3, 4));
    }

    @Test
    public void kernelHelpersDescribeLayoutsAndGeometry() {
        assertEquals(2 + 2 * KvDType.all().size(), MsaKernels.ALL.size());
        assertEquals(MsaKernels.KPOOL_MODE_GATED, MsaKernels.kpoolMode(3));
        assertEquals(MsaKernels.KPOOL_MODE_MEAN, MsaKernels.kpoolMode(2));
        assertEquals(64, MsaKernels.kpoolGateOffset(3, 64));
        assertEquals(128, MsaKernels.kpoolPooledOffset(3, 64));
        assertEquals(64, MsaKernels.kpoolPooledOffset(2, 64));
        assertEquals(KvDType.Q8_0.code(), MsaKernels.kvTypeCode(KvDType.Q8_0));
        assertEquals(32, MsaKernels.blockScoreLocal(16));
        assertEquals(128, MsaKernels.blockScoreLocal(128));
        assertEquals(256, MsaKernels.blockScoreLocal(4096));
        assertArrayEquals(new long[]{3L * 6 * 32}, MsaKernels.blockScoreGlobal(3, 23, 4, 32));
        assertArrayEquals(new long[]{2L * 8 * 128}, MsaKernels.gatherGlobal(2, 8, 128));
        assertArrayEquals(new long[]{512}, MsaKernels.kpoolUpdateGlobal(3, 100, 256));
        assertArrayEquals(new long[]{2L * 2 * 128}, MsaKernels.kpoolScoreGlobal(2, 200, 128));
        MsaKernels.checkGatherAttn(32, 8, 128, 64, 16, 8, 4096, 1024, 1024, KvDType.Q8_0, KvDType.F16, 128);
        MsaKernels.checkKpool(4, 64, 192, 64, 128, 4096, MsaKernels.KPOOL_MODE_GATED);
        MsaKernels.checkKpoolCache(KvDType.Q8_0, 64, 192, 64, 128, MsaKernels.KPOOL_MODE_GATED);
        MsaKernels.checkKpoolCache(KvDType.Q8_0, 64, 128, 0, 64, MsaKernels.KPOOL_MODE_MEAN);
        MsaKernels.checkKpoolCache(KvDType.F16, 3, 9, 3, 6, MsaKernels.KPOOL_MODE_GATED);
        assertEquals("LLM_QSA_KpoolUpdate_F32", MsaKernels.kpoolUpdate(KvDType.F32));
        assertEquals("LLM_QSA_KpoolUpdate_F16", MsaKernels.kpoolUpdate(KvDType.F16));
        assertEquals("LLM_QSA_KpoolUpdate_Q8_0", MsaKernels.kpoolUpdate(KvDType.Q8_0));
        assertEquals("LLM_QSA_KpoolScore_F32", MsaKernels.kpoolScore(KvDType.F32));
        assertEquals("LLM_QSA_KpoolScore_F16", MsaKernels.kpoolScore(KvDType.F16));
        assertEquals("LLM_QSA_KpoolScore_Q8_0", MsaKernels.kpoolScore(KvDType.Q8_0));
        for (KvDType t : KvDType.all()) {
            assertEquals(t.code(), MsaKernels.kvTypeCode(t));
            assertTrue(MsaKernels.ALL.contains(MsaKernels.kpoolUpdate(t)));
            assertTrue(MsaKernels.ALL.contains(MsaKernels.kpoolScore(t)));
        }
        assertEquals("LLM_QSA_KpoolUpdate_IQ4_NL", MsaKernels.kpoolUpdate(KvDType.IQ4_NL));
        MsaKernels.checkKpoolCache(KvDType.Q5_1, 64, 128, 0, 64, MsaKernels.KPOOL_MODE_MEAN);
        MsaKernels.checkKpoolCache(KvDType.BF16, 3, 9, 3, 6, MsaKernels.KPOOL_MODE_GATED);
    }

    @Test
    public void kernelChecksRejectUnsupportedShapes() {
        expectRejected(() -> MsaKernels.kpoolMode(4));
        expectRejected(() -> MsaKernels.selectedPools(0, 4));
        expectRejected(() -> MsaKernels.selectedPools(16, 0));
        expectRejected(() -> MsaKernels.checkGatherAttn(32, 8, 1024, 64, 16, 8, 4096, 8192, 8192, KvDType.F16,
                KvDType.F16, 64));
        expectRejected(() -> MsaKernels.checkGatherAttn(32, 8, 128, 64, 16, 4, 4096, 1024, 1024, KvDType.F16,
                KvDType.F16, 128));
        expectRejected(() -> MsaKernels.checkGatherAttn(32, 8, 128, 64, 16, 8, 4096, 1000, 1024, KvDType.Q8_0,
                KvDType.F16, 128));
        expectRejected(() -> MsaKernels.checkBlockScore(4, 128, 64, 1, 4096, KvDType.F16, 48));
        expectRejected(() -> MsaKernels.checkKpool(4, 64, 160, 64, 128, 4096, MsaKernels.KPOOL_MODE_GATED));
        expectRejected(() -> MsaKernels.checkKpoolScore(4, 64, 4, 192, 128, 4096, 0f, 128));
        expectRejected(() -> MsaKernels.checkKpoolCache(KvDType.Q8_0, 48, 144, 48, 96, MsaKernels.KPOOL_MODE_GATED));
        expectRejected(() -> MsaKernels.checkKpoolCache(KvDType.Q8_0, 64, 192, 48, 128, MsaKernels.KPOOL_MODE_GATED));
    }

    private static void expectRejected(Runnable r) {
        try {
            r.run();
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            assertTrue(expected.getMessage() != null && !expected.getMessage().isEmpty());
        }
    }
}
