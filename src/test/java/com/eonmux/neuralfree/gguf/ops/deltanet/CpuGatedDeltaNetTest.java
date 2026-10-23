package com.eonmux.neuralfree.gguf.ops.deltanet;

import com.eonmux.neuralfree.gguf.arch.GdnHeadMap;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.deltanet.CpuGatedDeltaNet.GateShape;
import com.eonmux.neuralfree.gguf.ops.deltanet.CpuGatedDeltaNet.Rows;
import com.eonmux.neuralfree.gguf.ops.deltanet.CpuGatedDeltaNet.ScanShape;
import com.eonmux.neuralfree.gguf.spec.RecurrentSpec.DecayGate;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Tests for {@link CpuGatedDeltaNet} (gated delta rule and its gate prep), {@link CpuLightning} and the dispatch
 * helpers of {@link DeltaNetKernels}, against hand-computed values and direct transcriptions of the llama.cpp
 * formulas (ggml-cpu/ops.cpp:11214-11360, src/models/minimax-01.cpp:288-398).
 */
public class CpuGatedDeltaNetTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random rnd, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = (rnd.nextFloat() * 2f - 1f) * scale;
        }
        return a;
    }

    @Test
    public void singleStepMatchesHandComputation() {
        ScanShape shape = new ScanShape(2, 2, 1, 1, GdnHeadMap.TILED, false, 1.0f);
        float[] q = {1f, 0f};
        float[] k = {1f, 0f};
        float[] v = {2f, 4f};
        float[] g = {0f};
        float[] beta = {0.5f};
        float[] state = new float[4];
        float[] out = new float[2];
        CpuGatedDeltaNet.scan(shape, 1, new Rows(q, 0, 2), new Rows(k, 0, 2), new Rows(v, 0, 2), g, 0, beta, 0,
                state, 0, new Rows(out, 0, 2));
        // delta = (v - S^T k) * beta = [1, 2]; S[0][col] = delta[col]; stored s[col*S_k + i].
        assertArrayEquals(new float[]{1f, 0f, 2f, 0f}, state, 0f);
        assertArrayEquals(new float[]{1f, 2f}, out, 0f);
    }

    @Test
    public void decayIsAppliedBeforeTheUpdate() {
        ScanShape shape = new ScanShape(1, 1, 1, 1, GdnHeadMap.TILED, false, 1.0f);
        float[] state = {3f};
        float[] out = new float[1];
        float logDecay = (float) Math.log(0.5);
        CpuGatedDeltaNet.scan(shape, 1, new Rows(new float[]{1f}, 0, 1), new Rows(new float[]{1f}, 0, 1),
                new Rows(new float[]{5f}, 0, 1), new float[]{logDecay}, 0, new float[]{1f}, 0, state, 0,
                new Rows(out, 0, 1));
        // S = 3 * 0.5 = 1.5; delta = 5 - 1.5 = 3.5; S = 1.5 + 3.5 = 5; o = 5.
        assertEquals(5f, state[0], EPS);
        assertEquals(5f, out[0], EPS);
    }

    /** Direct transcription of ggml_compute_forward_gated_delta_net_one_chunk for one head (KDA and GDN). */
    private static void ggmlReference(int sV, float[] m, float[] q, float[] k, float[] v, float[] g, boolean kda,
                                      float beta, float scale, float[] attn) {
        float[] delta = new float[sV];
        if (kda) {
            for (int i = 0; i < sV; i++) {
                delta[i] = (float) Math.exp(g[i]);
            }
            for (int j = 0; j < sV; j++) {
                for (int i = 0; i < sV; i++) {
                    m[j * sV + i] *= delta[i];
                }
            }
        } else {
            float e = (float) Math.exp(g[0]);
            for (int i = 0; i < sV * sV; i++) {
                m[i] *= e;
            }
        }
        for (int j = 0; j < sV; j++) {
            float sum = 0f;
            for (int i = 0; i < sV; i++) {
                sum += m[j * sV + i] * k[i];
            }
            delta[j] = (v[j] - sum) * beta;
        }
        for (int j = 0; j < sV; j++) {
            for (int i = 0; i < sV; i++) {
                m[j * sV + i] += delta[j] * k[i];
            }
        }
        for (int j = 0; j < sV; j++) {
            float sum = 0f;
            for (int i = 0; i < sV; i++) {
                sum += m[j * sV + i] * q[i];
            }
            attn[j] = sum * scale;
        }
    }

    @Test
    public void perKeyGateMatchesGgmlKdaLoop() {
        Random rnd = new Random(7);
        int s = 8;
        ScanShape shape = new ScanShape(s, s, 1, 1, GdnHeadMap.TILED, true, CpuGatedDeltaNet.defaultScale(s));
        float[] state = random(rnd, s * s, 0.3f);
        float[] ref = state.clone();
        float[] q = random(rnd, s, 1f);
        float[] k = random(rnd, s, 1f);
        float[] v = random(rnd, s, 1f);
        float[] g = random(rnd, s, 0.5f);
        float[] out = new float[s];
        float[] expected = new float[s];
        CpuGatedDeltaNet.scan(shape, 1, new Rows(q, 0, s), new Rows(k, 0, s), new Rows(v, 0, s), g, 0,
                new float[]{0.7f}, 0, state, 0, new Rows(out, 0, s));
        ggmlReference(s, ref, q, k, v, g, true, 0.7f, shape.scale(), expected);
        assertArrayEquals(expected, out, EPS);
        assertArrayEquals(ref, state, EPS);
    }

    @Test
    public void uniformPerKeyGateEqualsScalarGate() {
        Random rnd = new Random(11);
        int sK = 4;
        int sV = 3;
        float[] q = random(rnd, sK, 1f);
        float[] k = random(rnd, sK, 1f);
        float[] v = random(rnd, sV, 1f);
        float[] init = random(rnd, sK * sV, 0.5f);
        float gate = -0.3f;
        float[] gKey = {gate, gate, gate, gate};

        float[] s1 = init.clone();
        float[] o1 = new float[sV];
        CpuGatedDeltaNet.scan(new ScanShape(sK, sV, 1, 1, GdnHeadMap.TILED, false, 0.5f), 1, new Rows(q, 0, sK),
                new Rows(k, 0, sK), new Rows(v, 0, sV), new float[]{gate}, 0, new float[]{0.4f}, 0, s1, 0,
                new Rows(o1, 0, sV));
        float[] s2 = init.clone();
        float[] o2 = new float[sV];
        CpuGatedDeltaNet.scan(new ScanShape(sK, sV, 1, 1, GdnHeadMap.TILED, true, 0.5f), 1, new Rows(q, 0, sK),
                new Rows(k, 0, sK), new Rows(v, 0, sV), gKey, 0, new float[]{0.4f}, 0, s2, 0, new Rows(o2, 0, sV));
        assertArrayEquals(o1, o2, 0f);
        assertArrayEquals(s1, s2, 0f);
    }

    @Test
    public void prefillEqualsSequentialDecode() {
        Random rnd = new Random(3);
        int sK = 8;
        int sV = 6;
        int nV = 4;
        int nK = 2;
        int nT = 5;
        int qStride = nK * sK + 3;
        int vStride = nV * sV + 1;
        ScanShape shape = new ScanShape(sK, sV, nV, nK, GdnHeadMap.GROUPED, false, CpuGatedDeltaNet.defaultScale(sK));
        float[] q = random(rnd, nT * qStride + 2, 1f);
        float[] k = random(rnd, nT * qStride + 2, 1f);
        float[] v = random(rnd, nT * vStride, 1f);
        float[] g = random(rnd, nT * nV, 0.4f);
        float[] beta = random(rnd, nT * nV, 1f);
        int stateOff = 10;
        float[] init = random(rnd, stateOff + (int) shape.stateFloats(), 0.2f);

        float[] sAll = init.clone();
        float[] oAll = new float[nT * vStride];
        CpuGatedDeltaNet.scan(shape, nT, new Rows(q, 2, qStride), new Rows(k, 2, qStride), new Rows(v, 0, vStride),
                g, 0, beta, 0, sAll, stateOff, new Rows(oAll, 0, vStride));

        float[] sSeq = init.clone();
        float[] oSeq = new float[nT * vStride];
        for (int t = 0; t < nT; t++) {
            CpuGatedDeltaNet.scan(shape, 1, new Rows(q, 2 + t * qStride, qStride), new Rows(k, 2 + t * qStride, qStride),
                    new Rows(v, t * vStride, vStride), g, t * nV, beta, t * nV, sSeq, stateOff,
                    new Rows(oSeq, t * vStride, vStride));
        }
        assertArrayEquals(oAll, oSeq, 0f);
        assertArrayEquals(sAll, sSeq, 0f);
    }

    @Test
    public void headMapSelectsTheKeyHead() {
        Random rnd = new Random(5);
        int s = 4;
        int nV = 4;
        int nK = 2;
        float[] q = random(rnd, nK * s, 1f);
        float[] k = random(rnd, nK * s, 1f);
        float[] v = random(rnd, nV * s, 1f);
        float[] g = random(rnd, nV, 0.3f);
        float[] beta = random(rnd, nV, 1f);
        for (GdnHeadMap map : GdnHeadMap.values()) {
            ScanShape shape = new ScanShape(s, s, nV, nK, map, false, 1f);
            float[] state = new float[(int) shape.stateFloats()];
            float[] out = new float[nV * s];
            CpuGatedDeltaNet.scan(shape, 1, new Rows(q, 0, nK * s), new Rows(k, 0, nK * s), new Rows(v, 0, nV * s), g, 0,
                    beta, 0, state, 0, new Rows(out, 0, nV * s));
            for (int h = 0; h < nV; h++) {
                int kh = map == GdnHeadMap.TILED ? h % nK : h / (nV / nK);
                assertEquals(kh, shape.keyHead(h));
                ScanShape one = new ScanShape(s, s, 1, 1, GdnHeadMap.TILED, false, 1f);
                float[] s1 = new float[s * s];
                float[] o1 = new float[s];
                CpuGatedDeltaNet.scan(one, 1, new Rows(q, kh * s, s), new Rows(k, kh * s, s), new Rows(v, h * s, s),
                        g, h, beta, h, s1, 0, new Rows(o1, 0, s));
                for (int c = 0; c < s; c++) {
                    assertEquals(map + " head " + h, o1[c], out[h * s + c], 0f);
                }
            }
        }
    }

    @Test
    public void outputMayAliasValues() {
        Random rnd = new Random(9);
        int s = 6;
        int nV = 2;
        int nT = 3;
        ScanShape shape = new ScanShape(s, s, nV, nV, GdnHeadMap.TILED, false, CpuGatedDeltaNet.defaultScale(s));
        float[] q = random(rnd, nT * nV * s, 1f);
        float[] k = random(rnd, nT * nV * s, 1f);
        float[] v = random(rnd, nT * nV * s, 1f);
        float[] g = random(rnd, nT * nV, 0.2f);
        float[] beta = random(rnd, nT * nV, 1f);
        float[] stateA = new float[(int) shape.stateFloats()];
        float[] outA = new float[v.length];
        CpuGatedDeltaNet.scan(shape, nT, new Rows(q, 0, nV * s), new Rows(k, 0, nV * s), new Rows(v, 0, nV * s), g, 0,
                beta, 0, stateA, 0, new Rows(outA, 0, nV * s));
        float[] inPlace = v.clone();
        float[] stateB = new float[(int) shape.stateFloats()];
        Rows vo = new Rows(inPlace, 0, nV * s);
        CpuGatedDeltaNet.scan(shape, nT, new Rows(q, 0, nV * s), new Rows(k, 0, nV * s), vo, g, 0, beta, 0, stateB, 0, vo);
        assertArrayEquals(outA, inPlace, 0f);
        assertArrayEquals(stateA, stateB, 0f);
    }

    @Test
    public void gatePrepSeparateGdn() {
        GateShape shape = new GateShape(2, 1, DeltaNetKernels.LAYOUT_SEPARATE, DecayGate.GDN, 0, 0f);
        float[] a = {0.5f, 30f};
        float[] b = {0f, 2f};
        float[] dt = {0.1f, -5f};
        float[] ssmA = {-2f, -0.5f};
        float[] gOut = new float[2];
        float[] betaOut = new float[2];
        CpuGatedDeltaNet.gatePrep(shape, 1, new Rows(a, 0, 2), new Rows(b, 0, 2), dt, ssmA, gOut, 0, betaOut, 0);
        assertEquals((float) Math.log(1.0 + Math.exp(0.6f)) * -2f, gOut[0], EPS);
        assertEquals(25f * -0.5f, gOut[1], 0f); // softplus is the identity above 20
        assertEquals(0.5f, betaOut[0], 0f);
        assertEquals((float) (1.0 / (1.0 + Math.exp(-2.0))), betaOut[1], EPS);
    }

    @Test
    public void gatePrepFusedBetaAlphaLayout() {
        // n_v = 4, n_k = 2, R = 2: per key head [b(2) | a(2)]; value head h = kh*2 + r.
        GateShape shape = new GateShape(4, 2, DeltaNetKernels.LAYOUT_FUSED_BA, DecayGate.GDN, 0, 0f);
        float[] ba = {10f, 11f, 0.1f, 0.2f, 12f, 13f, 0.3f, 0.4f};
        float[] dt = new float[4];
        float[] ssmA = {1f, 1f, 1f, 1f};
        float[] gOut = new float[4];
        float[] betaOut = new float[4];
        Rows rows = new Rows(ba, 0, ba.length);
        CpuGatedDeltaNet.gatePrep(shape, 1, rows, rows, dt, ssmA, gOut, 0, betaOut, 0);
        float[] alphas = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] betas = {10f, 11f, 12f, 13f};
        for (int h = 0; h < 4; h++) {
            assertEquals(ActMath.softplus(alphas[h]), gOut[h], 0f);
            assertEquals(ActMath.sigmoid(betas[h]), betaOut[h], 0f);
        }
    }

    @Test
    public void gatePrepKdaModes() {
        int d = 2;
        float[] f = {0.3f, -0.7f, 1.1f, 0.2f};
        float[] dt = {0.1f, 0.2f, -0.3f, 0.4f};
        float[] ssmA = {-1.5f, -0.25f};
        float[] braw = {0.2f, -0.4f};
        float lb = -5f;
        for (DecayGate gate : new DecayGate[]{DecayGate.SOFTPLUS, DecayGate.LB_NEG, DecayGate.LB_POS}) {
            GateShape shape = new GateShape(2, 2, DeltaNetKernels.LAYOUT_SEPARATE, gate, d, lb);
            float[] gOut = new float[4];
            float[] betaOut = new float[2];
            CpuGatedDeltaNet.gatePrep(shape, 1, new Rows(f, 0, 4), new Rows(braw, 0, 2), dt, ssmA, gOut, 0, betaOut, 0);
            for (int h = 0; h < 2; h++) {
                for (int j = 0; j < d; j++) {
                    float x = f[h * d + j] + dt[h * d + j];
                    float aH = ssmA[h];
                    float expected;
                    if (gate == DecayGate.SOFTPLUS) {
                        expected = ActMath.softplus(x) * aH;
                    } else if (gate == DecayGate.LB_NEG) {
                        expected = lb * ActMath.sigmoid(-(aH * x));
                    } else {
                        expected = lb * ActMath.sigmoid(aH * x);
                    }
                    assertEquals(gate + " h" + h + " d" + j, expected, gOut[h * d + j], 0f);
                }
                assertEquals(ActMath.sigmoid(braw[h]), betaOut[h], 0f);
            }
        }
    }

    @Test
    public void fusedLayoutIsRejectedForKda() {
        try {
            new GateShape(2, 2, DeltaNetKernels.LAYOUT_FUSED_BA, DecayGate.SOFTPLUS, 4, 0f);
            fail("fused ssm_ba with KDA must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void lowerBoundGatesNeedAFiniteBound() {
        for (DecayGate gate : new DecayGate[]{DecayGate.LB_NEG, DecayGate.LB_POS}) {
            try {
                new GateShape(2, 2, DeltaNetKernels.LAYOUT_SEPARATE, gate, 4, Float.NEGATIVE_INFINITY);
                fail(gate + " without a lower bound must be rejected");
            } catch (IllegalArgumentException expected) {
                // expected
            }
        }
    }

    @Test
    public void scanShapeLimits() {
        try {
            new ScanShape(DeltaNetKernels.GDN_MAX_KEY_DIM + 1, 128, 1, 1, GdnHeadMap.TILED, false, 1f);
            fail("S_k above the kernel limit must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        try {
            new ScanShape(128, 128, 3, 2, GdnHeadMap.GROUPED, false, 1f);
            fail("n_v not a multiple of n_k must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void lightningMatchesLlamaBlockForm() {
        Random rnd = new Random(13);
        int nHead = 2;
        int d = 4;
        int nT = 3;
        float[] slopes = CpuLightning.baseSlopes(8);
        float layerScale = CpuLightning.layerScale(3, 10);
        float[] qkv = random(rnd, nT * nHead * 3 * d, 1f);
        float[] kvOld = random(rnd, nHead * d * d, 0.5f);
        float[] state = kvOld.clone();
        float[] out = new float[nT * nHead * d];
        CpuLightning.forward(nT, nHead, d, layerScale, slopes, qkv, 0, nHead * 3 * d, state, 0, out, 0, nHead * d);

        for (int h = 0; h < nHead; h++) {
            double s = slopes[h] * layerScale;
            for (int t = 0; t < nT; t++) {
                int qb = t * nHead * 3 * d + h * 3 * d;
                for (int col = 0; col < d; col++) {
                    double o = 0;
                    for (int kk = 0; kk < d; kk++) {
                        o += Math.exp(-s * (t + 1)) * qkv[qb + kk] * kvOld[h * d * d + col * d + kk];
                    }
                    for (int j = 0; j <= t; j++) {
                        int jb = j * nHead * 3 * d + h * 3 * d;
                        double qk = 0;
                        for (int kk = 0; kk < d; kk++) {
                            qk += qkv[qb + kk] * qkv[jb + d + kk];
                        }
                        o += qk * Math.exp(-s * (t - j)) * qkv[jb + 2 * d + col];
                    }
                    assertEquals("h" + h + " t" + t + " v" + col, o, out[t * nHead * d + h * d + col], 1e-4);
                }
            }
            for (int col = 0; col < d; col++) {
                for (int kk = 0; kk < d; kk++) {
                    double kv = Math.exp(-s * nT) * kvOld[h * d * d + col * d + kk];
                    for (int j = 0; j < nT; j++) {
                        int jb = j * nHead * 3 * d + h * 3 * d;
                        kv += Math.exp(-s * (nT - 1 - j)) * qkv[jb + d + kk] * qkv[jb + 2 * d + col];
                    }
                    assertEquals("state h" + h + " v" + col + " k" + kk, kv, state[h * d * d + col * d + kk], 1e-4);
                }
            }
        }
    }

    @Test
    public void lightningSlopeSchedule() {
        float[] eight = CpuLightning.baseSlopes(8);
        for (int h = 0; h < 8; h++) {
            assertEquals((float) Math.pow(0.5, h + 1), eight[h], 1e-7f);
        }
        float[] sixtyFour = CpuLightning.baseSlopes(64);
        assertEquals((float) Math.pow(2.0, -0.125), sixtyFour[0], 1e-6f);
        assertEquals(1f + 1e-5f, CpuLightning.layerScale(0, 80), 1e-7f);
        assertEquals(1e-5f, CpuLightning.layerScale(79, 80), 1e-7f);
    }

    @Test
    public void dispatchGeometry() {
        assertEquals(32, DeltaNetKernels.recurrenceLocal(1));
        assertEquals(64, DeltaNetKernels.recurrenceLocal(64));
        assertEquals(128, DeltaNetKernels.recurrenceLocal(200));
        assertEquals(256, DeltaNetKernels.recurrenceLocal(1024));
        // 32 value heads, S_v = 128, local 128 (4 columns per group): 32 * 32 groups.
        assertEquals(32L * 32 * 128, DeltaNetKernels.recurrenceGlobal(32, 128, 128));
        // ceil(6 / 4) = 2 groups per head.
        assertEquals(3L * 2 * 128, DeltaNetKernels.recurrenceGlobal(3, 6, 128));
        assertEquals(256L, DeltaNetKernels.gatePrepGlobal(1, 32, -1, 128, 256));
        assertEquals(2L * 32 * 128, DeltaNetKernels.gatePrepGlobal(2, 32, 0, 128, 256));
        try {
            DeltaNetKernels.recurrenceGlobal(4, 128, 512);
            fail("a local size above LLM_DIMX overflows the kernels' local arrays and must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }
}
