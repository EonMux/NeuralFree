package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.cpu.KvAccessor;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * CPU attention core against an independent double-precision softmax reference: GQA head mapping, sinks, softcap,
 * ALiBi slopes, T5 buckets, causal and the three window kinds, fully masked rows, ring caches through
 * {@link KvAccessor}, MLA latent aliasing, the multi-row form, ranges spanning several key blocks and head chunks,
 * and the key splits of decode rows with their combine.
 */
public class CpuAttentionCoreTest {

    private static final float TOL = 2e-5f;
    /** Rounding difference of the split-K combine against the single pass. */
    private static final float SPLIT_TOL = 1e-5f;

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int D_K = 8;
    private static final int D_V = 12;
    private static final int ROWS = 10;

    /** Reference options. */
    private static final class Ref {
        float scale = 1.0f;
        float softcap;
        boolean causal = true;
        SwaType swa = SwaType.NONE;
        int nSwa;
        float[] slopes;
        float[] relBias;
        int nBuckets;
        boolean bidir;
        float[] sinks;
    }

    @Test
    public void gqaCausalMatchesReference() {
        Fixture f = new Fixture(1);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.35f).build();
        Ref r = new Ref();
        r.scale = 0.35f;
        check(f, p, r, 6);
        check(f, p, r, 9);
        check(f, p, r, 0);
    }

    @Test
    public void sinksJoinTheDenominator() {
        Fixture f = new Fixture(2);
        float[] sinks = {0.5f, -1.0f, 2.0f, 0.1f};
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.5f).sinks(sinks).build();
        Ref r = new Ref();
        r.scale = 0.5f;
        r.sinks = sinks;
        check(f, p, r, 7);
    }

    @Test
    public void softcapBoundsTheLogits() {
        Fixture f = new Fixture(3);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(2.0f).softcap(1.5f).build();
        Ref r = new Ref();
        r.scale = 2.0f;
        r.softcap = 1.5f;
        check(f, p, r, 8);
        assertEquals(1.5f * (float) Math.tanh(10.0 * 2.0 / 1.5), CpuAttentionCore.score(10.0f, 2.0f, 1.5f), 1e-6f);
        assertEquals(20.0f, CpuAttentionCore.score(10.0f, 2.0f, 0.0f), 0.0f);
    }

    @Test
    public void alibiSlopesFollowGgml() {
        // n_head = 8: n_head_log2 = 8, m0 = 2^(-8/8) = 0.5, slope(h) = 0.5^(h+1)
        for (int h = 0; h < 8; h++) {
            assertEquals(Math.pow(0.5, h + 1), CpuAttentionCore.alibiSlope(h, 8, 8.0f), 1e-7);
        }
        // n_head = 12: n_head_log2 = 8, m1 = 2^(-4/8); heads >= 8 use m1^(2(h-8)+1)
        double m1 = Math.pow(2.0, -0.5);
        assertEquals(m1, CpuAttentionCore.alibiSlope(8, 12, 8.0f), 1e-6);
        assertEquals(Math.pow(m1, 3), CpuAttentionCore.alibiSlope(9, 12, 8.0f), 1e-6);
        assertEquals(Math.pow(0.5, 3), CpuAttentionCore.alibiSlope(2, 12, 8.0f), 1e-7);
        assertEquals(1.0f, CpuAttentionCore.alibiSlope(3, 12, 0.0f), 0.0f);

        Fixture f = new Fixture(4);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.3f).alibi(8.0f).build();
        Ref r = new Ref();
        r.scale = 0.3f;
        r.slopes = new float[N_HEAD];
        for (int h = 0; h < N_HEAD; h++) {
            r.slopes[h] = (float) Math.pow(2.0, -2.0 * (h + 1));   // n_head 4: m0 = 2^(-8/4)
        }
        check(f, p, r, 9);
    }

    @Test
    public void t5BucketsFollowLlamaCpp() {
        // encoder: bidirectional, 32 buckets -> 16 per direction, max_exact 8
        assertEquals(0, CpuAttentionCore.t5Bucket(0, 0, 32, 128, true));
        assertEquals(17, CpuAttentionCore.t5Bucket(1, 0, 32, 128, true));
        assertEquals(1, CpuAttentionCore.t5Bucket(0, 1, 32, 128, true));
        assertEquals(10, CpuAttentionCore.t5Bucket(0, 20, 32, 128, true));    // 8 + floor(ln(2.5) * 8 / ln(16))
        assertEquals(26, CpuAttentionCore.t5Bucket(20, 0, 32, 128, true));
        assertEquals(15, CpuAttentionCore.t5Bucket(0, 200, 32, 128, true));   // clamped to n_buckets - 1
        // decoder: unidirectional, max_exact 16; future keys map to 0
        assertEquals(0, CpuAttentionCore.t5Bucket(5, 0, 32, 128, false));
        assertEquals(3, CpuAttentionCore.t5Bucket(0, 3, 32, 128, false));
        assertEquals(17, CpuAttentionCore.t5Bucket(0, 20, 32, 128, false));   // 16 + floor(ln(1.25) * 16 / ln(8))
        assertEquals(31, CpuAttentionCore.t5Bucket(0, 1000, 32, 128, false));

        Fixture f = new Fixture(5);
        float[] rel = new float[32 * N_HEAD];
        Random rnd = new Random(55);
        for (int i = 0; i < rel.length; i++) {
            rel[i] = rnd.nextFloat() * 2 - 1;
        }
        for (boolean bidir : new boolean[]{false, true}) {
            CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD)
                    .window(bidir ? KvWindow.FULL_BIDIRECTIONAL : KvWindow.FULL_CAUSAL)
                    .t5(bidir, 32, 128, rel).build();
            Ref r = new Ref();
            r.causal = !bidir;
            r.relBias = rel;
            r.nBuckets = 32;
            r.bidir = bidir;
            check(f, p, r, 5);
        }
    }

    @Test
    public void windowKindsMaskLikeIsMaskedSwa() {
        Fixture f = new Fixture(6);
        SwaType[] types = {SwaType.STANDARD, SwaType.CHUNKED, SwaType.SYMMETRIC, SwaType.SYMMETRIC};
        boolean[] causalFlags = {true, true, false, true};
        for (int i = 0; i < types.length; i++) {
            SwaType t = types[i];
            boolean causal = causalFlags[i];
            CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.4f)
                    .window(KvWindow.of(t, 4, causal)).build();
            Ref r = new Ref();
            r.scale = 0.4f;
            r.causal = causal;
            r.swa = t;
            r.nSwa = 4;
            check(f, p, r, 6);
        }
        // STANDARD n_swa = 4 at position 6 sees exactly positions 3..6 (the query and the 3 before it)
        CpuAttentionCore.Params swa = CpuAttentionCore.Params.builder(N_HEAD).scale(0.4f)
                .window(KvWindow.of(SwaType.STANDARD, 4, true)).build();
        CpuAttentionCore.Params full = CpuAttentionCore.Params.builder(N_HEAD).scale(0.4f).build();
        float[] a = new float[N_HEAD * D_V];
        float[] b = new float[N_HEAD * D_V];
        CpuAttentionCore.attend(swa, f.q, 0, 6, f.dense(), 0, ROWS, a, 0, new CpuAttentionCore.Scratch());
        CpuAttentionCore.attend(full, f.q, 0, 6, f.dense(), 3, 7, b, 0, new CpuAttentionCore.Scratch());
        assertArrayEquals(b, a, TOL);
    }

    @Test
    public void fullyMaskedRowOutputsZero() {
        Fixture f = new Fixture(7);
        float[] zeros = new float[N_HEAD * D_V];
        float[] out = new float[N_HEAD * D_V];
        Arrays.fill(out, 7.0f);
        // every stored position is in the future of query position 0 or empty
        int[] pos = {3, 4, -1, 5, 6, 7, 8, 9, -1, 10};
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(f.k, 0, N_HEAD_KV * D_K, D_K, f.v, 0,
                N_HEAD_KV * D_V, D_V, N_HEAD_KV, D_K, D_V, pos, 0);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).build();
        CpuAttentionCore.attend(p, f.q, 0, 1, kv, 0, ROWS, out, 0, new CpuAttentionCore.Scratch());
        assertArrayEquals(zeros, out, 0.0f);
        // a sink alone: m' = sink, l' = 0 * 0 + e^0 = 1, output 0 / 1 = 0
        CpuAttentionCore.Params ps = CpuAttentionCore.Params.builder(N_HEAD).sinks(new float[]{1, 2, 3, 4}).build();
        Arrays.fill(out, 7.0f);
        CpuAttentionCore.attend(ps, f.q, 0, 1, kv, 0, ROWS, out, 0, new CpuAttentionCore.Scratch());
        assertArrayEquals(zeros, out, 0.0f);
        // an empty key range behaves the same
        CpuAttentionCore.attend(p, f.q, 0, 5, f.dense(), 4, 4, out, 0, new CpuAttentionCore.Scratch());
        assertArrayEquals(zeros, out, 0.0f);
    }

    @Test
    public void ringCacheThroughAccessorMatchesOrderedRows() {
        Fixture f = new Fixture(8);
        // ring of 8 rows holding positions 8, 9, 2..7 (row = pos % 8); query at 9 with a window of 6 sees 4..9
        int[] ringPos = {8, 9, 2, 3, 4, 5, 6, 7};
        HeapKv ring = new HeapKv(8, N_HEAD_KV, D_K, D_V);
        float[] kRow = new float[N_HEAD_KV * D_K];
        float[] vRow = new float[N_HEAD_KV * D_V];
        for (int row = 0; row < 8; row++) {
            System.arraycopy(f.k, ringPos[row] * kRow.length, kRow, 0, kRow.length);
            System.arraycopy(f.v, ringPos[row] * vRow.length, vRow, 0, vRow.length);
            ring.writeK(row, kRow, 0);
            ring.writeV(row, vRow, 0);
        }
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.6f)
                .window(KvWindow.of(SwaType.STANDARD, 6, true)).build();
        float[] viaRing = new float[N_HEAD * D_V];
        CpuAttentionCore.attend(p, f.q, 0, 9, ring, r -> ringPos[r], 0, 8, viaRing, 0, new CpuAttentionCore.Scratch());
        CpuAttentionCore.Params full = CpuAttentionCore.Params.builder(N_HEAD).scale(0.6f).build();
        float[] ordered = new float[N_HEAD * D_V];
        CpuAttentionCore.attend(full, f.q, 0, 9, f.dense(), 4, 10, ordered, 0, new CpuAttentionCore.Scratch());
        assertArrayEquals(ordered, viaRing, TOL);
    }

    @Test
    public void mlaLatentReadsVFromTheKRow() {
        int dK = 12;
        int dV = 8;
        int rows = 6;
        Random rnd = new Random(9);
        float[] latent = randomArray(rnd, rows * dK);
        float[] q = randomArray(rnd, 3 * dK);
        int[] pos = {0, 1, 2, 3, 4, 5};
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(latent, 0, dK, 0, latent, 0, dK, 0, 1, dK, dV, pos, 0);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(3).scale(0.2f).build();
        float[] out = new float[3 * dV];
        CpuAttentionCore.attend(p, q, 0, 5, kv, 0, rows, out, 0, new CpuAttentionCore.Scratch());
        for (int h = 0; h < 3; h++) {
            double[] s = new double[rows];
            double m = Double.NEGATIVE_INFINITY;
            for (int r = 0; r < rows; r++) {
                double dot = 0;
                for (int d = 0; d < dK; d++) {
                    dot += (double) q[h * dK + d] * latent[r * dK + d];
                }
                s[r] = dot * 0.2;
                m = Math.max(m, s[r]);
            }
            double l = 0;
            double[] o = new double[dV];
            for (int r = 0; r < rows; r++) {
                double e = Math.exp(s[r] - m);
                l += e;
                for (int d = 0; d < dV; d++) {
                    o[d] += e * latent[r * dK + d];
                }
            }
            for (int d = 0; d < dV; d++) {
                assertEquals(o[d] / l, out[h * dV + d], TOL);
            }
        }
    }

    @Test
    public void multiRowFormMatchesSingleRows() {
        Fixture f = new Fixture(10);
        int nTokens = 5;
        Random rnd = new Random(11);
        float[] q = randomArray(rnd, nTokens * N_HEAD * D_K);
        int[] qPos = {5, 6, 7, 8, 9};
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(0.3f)
                .sinks(new float[]{0.2f, 0.4f, -0.3f, 1.0f}).build();
        int outStride = N_HEAD * D_V + 3;
        float[] many = new float[nTokens * outStride];
        CpuAttentionCore.attendRows(p, nTokens, q, 0, N_HEAD * D_K, qPos, 0, f.dense(), 0, ROWS, many, 0, outStride);
        for (int t = 0; t < nTokens; t++) {
            float[] one = new float[N_HEAD * D_V];
            CpuAttentionCore.attend(p, q, t * N_HEAD * D_K, qPos[t], f.dense(), 0, ROWS, one, 0,
                    new CpuAttentionCore.Scratch());
            for (int i = 0; i < one.length; i++) {
                assertEquals(one[i], many[t * outStride + i], 0.0f);
            }
        }
    }

    @Test
    public void longRangesAndWideHeadGroupsMatchReference() {
        // 20 query heads per KV head (two head chunks) and 700 keys (three key blocks with online rescaling),
        // a STANDARD window that masks the first block for late queries, and sinks.
        int nHead = 40;
        int nHeadKv = 2;
        int dK = 16;
        int dV = 8;
        int rows = 700;
        Random rnd = new Random(21);
        float[] k = randomArray(rnd, rows * nHeadKv * dK);
        float[] v = randomArray(rnd, rows * nHeadKv * dV);
        for (int i = 0; i < k.length; i++) {
            k[i] *= 3.0f;
        }
        int[] pos = new int[rows];
        for (int i = 0; i < rows; i++) {
            pos[i] = (i % 97 == 5) ? -1 : i;
        }
        float[] sinks = randomArray(rnd, nHead);
        int[] qPos = {699, 420, 3};
        float[] q = randomArray(rnd, qPos.length * nHead * dK);
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(k, 0, nHeadKv * dK, dK, v, 0, nHeadKv * dV, dV,
                nHeadKv, dK, dV, pos, 0);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(nHead).scale(0.25f).sinks(sinks)
                .window(KvWindow.of(SwaType.STANDARD, 450, true)).build();
        float[] many = new float[qPos.length * nHead * dV];
        CpuAttentionCore.attendRows(p, qPos.length, q, 0, nHead * dK, qPos, 0, kv, 0, rows, many, 0, nHead * dV);
        for (int t = 0; t < qPos.length; t++) {
            float[] one = new float[nHead * dV];
            CpuAttentionCore.attend(p, q, t * nHead * dK, qPos[t], kv, 0, rows, one, 0, new CpuAttentionCore.Scratch());
            for (int h = 0; h < nHead; h++) {
                int kvh = h / (nHead / nHeadKv);
                double m = sinks[h];
                double[] s = new double[rows];
                for (int r = 0; r < rows; r++) {
                    boolean visible = pos[r] >= 0 && pos[r] <= qPos[t] && qPos[t] - pos[r] < 450;
                    if (!visible) {
                        s[r] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    double dot = 0;
                    for (int d = 0; d < dK; d++) {
                        dot += (double) q[t * nHead * dK + h * dK + d] * k[r * nHeadKv * dK + kvh * dK + d];
                    }
                    s[r] = dot * 0.25;
                    m = Math.max(m, s[r]);
                }
                double l = Math.exp(sinks[h] - m);
                double[] o = new double[dV];
                for (int r = 0; r < rows; r++) {
                    if (s[r] == Double.NEGATIVE_INFINITY) {
                        continue;
                    }
                    double e = Math.exp(s[r] - m);
                    l += e;
                    for (int d = 0; d < dV; d++) {
                        o[d] += e * v[r * nHeadKv * dV + kvh * dV + d];
                    }
                }
                for (int d = 0; d < dV; d++) {
                    assertEquals("token " + t + " head " + h + " d " + d, o[d] / l, one[h * dV + d], 1e-4);
                    assertEquals(one[h * dV + d], many[t * nHead * dV + h * dV + d], SPLIT_TOL);
                }
            }
        }
    }

    @Test
    public void splitCountOccupiesIdleWorkersOnLongRanges() {
        assertEquals(1, CpuAttentionCore.splitCount(2, CpuAttentionCore.SPLIT_MIN_KEYS - 1, 16));
        assertEquals(1, CpuAttentionCore.splitCount(16, 4096, 16));
        assertEquals(1, CpuAttentionCore.splitCount(32, 4096, 16));
        assertEquals(8, CpuAttentionCore.splitCount(2, 4096, 16));
        assertEquals(6, CpuAttentionCore.splitCount(3, 4096, 16));
        // at most one split per key block
        assertEquals(3, CpuAttentionCore.splitCount(1, 600, 64));
        assertEquals(2, CpuAttentionCore.splitCount(1, CpuAttentionCore.SPLIT_MIN_KEYS, 64));
    }

    @Test
    public void keySplitsMatchTheSinglePass() {
        // one decode row, 20 query heads per KV head (two head chunks), 1500 keys with empty rows, a window that
        // hides whole leading splits, and sinks: every split count must give the single-pass result.
        int nHead = 40;
        int nHeadKv = 2;
        int dK = 16;
        int dV = 8;
        int rows = 1500;
        Random rnd = new Random(31);
        float[] k = randomArray(rnd, rows * nHeadKv * dK);
        float[] v = randomArray(rnd, rows * nHeadKv * dV);
        for (int i = 0; i < k.length; i++) {
            k[i] *= 4.0f;
        }
        int[] pos = new int[rows];
        for (int i = 0; i < rows; i++) {
            pos[i] = (i % 89 == 7) ? -1 : i;
        }
        float[] sinks = randomArray(rnd, nHead);
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(k, 0, nHeadKv * dK, dK, v, 0, nHeadKv * dV, dV,
                nHeadKv, dK, dV, pos, 0);
        for (KvWindow window : new KvWindow[]{KvWindow.FULL_CAUSAL, KvWindow.of(SwaType.STANDARD, 300, true)}) {
            CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(nHead).scale(0.25f).sinks(sinks)
                    .window(window).build();
            for (int qPos : new int[]{rows - 1, 700, 3}) {
                float[] q = randomArray(rnd, nHead * dK);
                float[] one = new float[nHead * dV];
                CpuAttentionCore.attend(p, q, 0, qPos, kv, 0, rows, one, 0, new CpuAttentionCore.Scratch());
                for (int splits : new int[]{1, 2, 3, 5, 6, 100}) {
                    float[] split = new float[nHead * dV];
                    CpuAttentionCore.attendRowsSplit(p, 1, q, 0, nHead * dK, new int[]{qPos}, 0, kv, 0, rows, split,
                            0, nHead * dV, splits);
                    for (int i = 0; i < one.length; i++) {
                        assertEquals(window + " q " + qPos + " splits " + splits + " i " + i, one[i], split[i],
                                SPLIT_TOL);
                    }
                }
            }
        }
    }

    @Test
    public void keySplitsOfAFullyMaskedRowOutputZeros() {
        int rows = 1024;
        Fixture f = new Fixture(41);
        int[] pos = new int[rows];
        Arrays.fill(pos, -1);
        float[] k = new float[rows * N_HEAD_KV * D_K];
        float[] v = new float[rows * N_HEAD_KV * D_V];
        Arrays.fill(v, 1.0f);
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(k, 0, N_HEAD_KV * D_K, D_K, v, 0, N_HEAD_KV * D_V,
                D_V, N_HEAD_KV, D_K, D_V, pos, 0);
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).build();
        float[] out = new float[N_HEAD * D_V];
        Arrays.fill(out, 7.0f);
        CpuAttentionCore.attendRowsSplit(p, 1, f.q, 0, N_HEAD * D_K, new int[]{5}, 0, kv, 0, rows, out, 0,
                N_HEAD * D_V, 4);
        assertArrayEquals(new float[N_HEAD * D_V], out, 0.0f);
    }

    @Test
    public void paramsFromSpecCarryScaleWindowAndSinks() {
        AttentionSpec spec = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(D_K).headDimV(D_V)
                .nRot(D_K).causal(true).swaType(SwaType.STANDARD).nSwa(4).swaLayer(true).sinks(true)
                .softcap(30.0f).kqScale(0.125f).kvOwnerLayer(0).build();
        CpuAttentionCore.Params p = CpuAttentionCore.Params.of(spec, new float[]{1, 2, 3, 4}, null);
        assertEquals(0.125f, p.scale(), 0.0f);
        assertEquals(30.0f, p.softcap(), 0.0f);
        assertEquals(SwaType.STANDARD, p.window().type());
        assertEquals(4, p.window().nSwa());
        assertTrue(p.hasSinks());
        assertEquals(3.0f, p.sink(2), 0.0f);
        assertThrows(NullPointerException.class, () -> CpuAttentionCore.Params.of(spec, null, null));
    }

    // ------------------------------------------------------------------ fixtures and reference

    private static void check(Fixture f, CpuAttentionCore.Params p, Ref r, int qPos) {
        float[] out = new float[N_HEAD * D_V];
        CpuAttentionCore.attend(p, f.q, 0, qPos, f.dense(), 0, ROWS, out, 0, new CpuAttentionCore.Scratch());
        double[] ref = reference(f, r, qPos);
        for (int i = 0; i < out.length; i++) {
            assertEquals("element " + i + " at position " + qPos, ref[i], out[i], TOL);
        }
        float[] viaAccessor = new float[N_HEAD * D_V];
        CpuAttentionCore.attend(p, f.q, 0, qPos, f.heap(), row -> f.pos[row], 0, ROWS, viaAccessor, 0,
                new CpuAttentionCore.Scratch());
        assertArrayEquals(out, viaAccessor, 0.0f);
    }

    private static double[] reference(Fixture f, Ref r, int qPos) {
        double[] out = new double[N_HEAD * D_V];
        int g0 = N_HEAD / N_HEAD_KV;
        for (int h = 0; h < N_HEAD; h++) {
            int kvh = h / g0;
            double[] s = new double[ROWS];
            double m = Double.NEGATIVE_INFINITY;
            for (int row = 0; row < ROWS; row++) {
                int pk = f.pos[row];
                if (!refVisible(qPos, pk, r)) {
                    s[row] = Double.NEGATIVE_INFINITY;
                    continue;
                }
                double dot = 0;
                for (int d = 0; d < D_K; d++) {
                    dot += (double) f.q[h * D_K + d] * f.k[row * N_HEAD_KV * D_K + kvh * D_K + d];
                }
                double x = r.softcap > 0 ? r.softcap * Math.tanh(dot * r.scale / r.softcap) : dot * r.scale;
                if (r.slopes != null) {
                    x += r.slopes[h] * -Math.abs(qPos - pk);
                }
                if (r.relBias != null) {
                    x += r.relBias[CpuAttentionCore.t5Bucket(pk, qPos, r.nBuckets, 128, r.bidir) * N_HEAD + h];
                }
                s[row] = x;
                m = Math.max(m, x);
            }
            if (r.sinks != null) {
                m = Math.max(m, r.sinks[h]);
            }
            if (m == Double.NEGATIVE_INFINITY) {
                continue;
            }
            double l = r.sinks != null ? Math.exp(r.sinks[h] - m) : 0;
            double[] o = new double[D_V];
            for (int row = 0; row < ROWS; row++) {
                if (s[row] == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                double e = Math.exp(s[row] - m);
                l += e;
                for (int d = 0; d < D_V; d++) {
                    o[d] += e * f.v[row * N_HEAD_KV * D_V + kvh * D_V + d];
                }
            }
            for (int d = 0; d < D_V; d++) {
                out[h * D_V + d] = l > 0 ? o[d] / l : 0;
            }
        }
        return out;
    }

    /** llama.cpp causal mask and is_masked_swa (llama-hparams.h:506-540), written out independently. */
    private static boolean refVisible(int pq, int pk, Ref r) {
        if (pk < 0 || (r.causal && pk > pq)) {
            return false;
        }
        switch (r.swa) {
            case STANDARD:
                return pq - pk < r.nSwa;
            case CHUNKED:
                return pk >= (pq / r.nSwa) * r.nSwa;
            case SYMMETRIC:
                return pq - pk >= -(r.nSwa / 2) && pq - pk <= r.nSwa / 2;
            default:
                return true;
        }
    }

    private static float[] randomArray(Random rnd, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = rnd.nextFloat() * 2 - 1;
        }
        return a;
    }

    /** Random q (one row), K/V rows with packed heads, positions 0..9 in row order. */
    private static final class Fixture {
        final float[] q;
        final float[] k;
        final float[] v;
        final int[] pos = new int[ROWS];

        Fixture(long seed) {
            Random rnd = new Random(seed);
            q = randomArray(rnd, N_HEAD * D_K);
            k = randomArray(rnd, ROWS * N_HEAD_KV * D_K);
            v = randomArray(rnd, ROWS * N_HEAD_KV * D_V);
            for (int i = 0; i < ROWS; i++) {
                pos[i] = i;
            }
        }

        CpuAttentionCore.DenseKv dense() {
            return CpuAttentionCore.DenseKv.of(k, 0, N_HEAD_KV * D_K, D_K, v, 0, N_HEAD_KV * D_V, D_V,
                    N_HEAD_KV, D_K, D_V, pos, 0);
        }

        HeapKv heap() {
            HeapKv kv = new HeapKv(ROWS, N_HEAD_KV, D_K, D_V);
            for (int r = 0; r < ROWS; r++) {
                kv.writeK(r, k, r * N_HEAD_KV * D_K);
                kv.writeV(r, v, r * N_HEAD_KV * D_V);
            }
            return kv;
        }
    }

    /** F32 heap cache implementing the memory API. */
    private static final class HeapKv implements KvAccessor {
        private final KvLayout layout;
        private final int nHeadKv;
        private final int dK;
        private final int dV;
        private final float[] kData;
        private final float[] vData;

        HeapKv(int rows, int nHeadKv, int dK, int dV) {
            this.layout = KvLayout.of(rows, false, nHeadKv * dK, nHeadKv * dV, KvDType.F32, KvDType.F32,
                    KvWindow.FULL_CAUSAL);
            this.nHeadKv = nHeadKv;
            this.dK = dK;
            this.dV = dV;
            this.kData = new float[rows * nHeadKv * dK];
            this.vData = new float[rows * nHeadKv * dV];
        }

        @Override public int layer() { return 0; }
        @Override public KvLayout layout() { return layout; }
        @Override public int group() { return 0; }
        @Override public int nHeadKv() { return nHeadKv; }
        @Override public int headDimK() { return dK; }
        @Override public int headDimV() { return dV; }

        @Override
        public void readK(int row, int head, float[] dst, int off) {
            System.arraycopy(kData, row * nHeadKv * dK + head * dK, dst, off, dK);
        }

        @Override
        public void readV(int row, int head, float[] dst, int off) {
            System.arraycopy(vData, row * nHeadKv * dV + head * dV, dst, off, dV);
        }

        @Override
        public void writeK(int row, float[] src, int off) {
            System.arraycopy(src, off, kData, row * nHeadKv * dK, nHeadKv * dK);
        }

        @Override
        public void writeKSlice(int row, int from, int n, float[] src, int off) {
            System.arraycopy(src, off, kData, row * nHeadKv * dK + from, n);
        }

        @Override
        public void writeV(int row, float[] src, int off) {
            System.arraycopy(src, off, vData, row * nHeadKv * dV, nHeadKv * dV);
        }
    }
}
