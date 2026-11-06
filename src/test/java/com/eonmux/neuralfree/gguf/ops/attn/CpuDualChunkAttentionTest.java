package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.ops.rope.DualChunkRope;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link CpuDualChunkAttention} and {@link DualChunkRope} against a direct transcription of vLLM's dual-chunk
 * formulas (vllm dual_chunk_rope.py and the dual-chunk flash-attention backend): the five query rotations, the key
 * rotation modulo the chunk length, the intra/succ/inter grouping of a prefill and of a decode, one softmax over the
 * concatenated logits, and the original-context logit scaling.
 *
 * <p>The reference below scores every query independently: it gathers the query's own chunk (causal), the preceding
 * chunk and every earlier chunk with the matching query rotation, concatenates their logits and normalizes once — the
 * mathematical statement of vLLM's three separately scored stages that share one softmax.
 */
public class CpuDualChunkAttentionTest {

    private static final int N_HEAD = 4;
    private static final int N_HEAD_KV = 2;
    private static final int D_K = 8;
    private static final int D_V = 8;
    private static final int CHUNK_SIZE = 7;
    private static final int LOCAL_SIZE = 2;
    private static final int CHUNK_LEN = CHUNK_SIZE - LOCAL_SIZE;
    private static final float SCALE = 0.375f;
    private static final float TOL = 2e-5f;

    private static DualChunkSpec spec(int orig) {
        return DualChunkSpec.builder().chunkSize(CHUNK_SIZE).localSize(LOCAL_SIZE)
                .originalMaxPositionEmbeddings(orig).build();
    }

    private static AttentionSpec attention() {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV).headDimK(D_K).headDimV(D_V)
                .nRot(D_K).kqScale(SCALE).kvOwnerLayer(0).build();
    }

    // ================================================================== scaling factor

    @Test
    public void sequenceScalingFollowsTheLogFormula() {
        DualChunkSpec dc = spec(1024);
        assertEquals(1.0f, dc.scalingFactor(512), 0f);
        assertEquals(1.0f, dc.scalingFactor(1024), 0f);
        assertEquals(1.0f + 0.1f * (float) Math.log(2.0), dc.scalingFactor(2048), 1e-7f);
        assertEquals(1.0f + 0.1f * (float) Math.log(8.0), dc.scalingFactor(8192), 1e-7f);
        // the factor never drops below 1, and a zero original length disables it
        assertEquals(1.0f, dc.scalingFactor(0), 0f);
        assertEquals(1.0f, spec(0).scalingFactor(1 << 20), 0f);
        assertTrue(spec(1024).scales(4096));
    }

    // ================================================================== rotations

    @Test
    public void queryAndKeyPositionsFollowTheFiveVariants() {
        DualChunkSpec dc = spec(0);
        int[] pos = {0, 1, 4, 5, 6, 9, 13};
        int[] out = new int[DualChunkRope.VARIANTS * pos.length];
        DualChunkRope.queryPositions(dc, pos, 0, pos.length, out);
        int n = pos.length;
        for (int t = 0; t < n; t++) {
            int p = pos[t];
            int base = p % CHUNK_LEN;
            assertEquals("query " + p, base, out[t]);
            assertEquals("query_succ " + p, Math.min(base + CHUNK_LEN, CHUNK_SIZE), out[n + t]);
            assertEquals("query_inter " + p, Math.min(2 * CHUNK_LEN - 1, CHUNK_SIZE), out[2 * n + t]);
            assertEquals("query_succ_critical " + p, base + CHUNK_LEN, out[3 * n + t]);
            assertEquals("query_inter_critical " + p, base + CHUNK_SIZE, out[4 * n + t]);
        }
        int[] keyPos = new int[n];
        DualChunkRope.keyPositions(dc, pos, 0, n, keyPos);
        for (int t = 0; t < n; t++) {
            assertEquals(pos[t] % CHUNK_LEN, keyPos[t]);
        }
    }

    @Test
    public void queryPlanesRotateAtTheDerivedPositions() {
        RopeSpec rope = RopeSpec.builder(RopeType.NORM).nDims(D_K).freqBase(10000f).build();
        RopeFreqTable table = RopeFreqTable.forDualChunk(rope);
        DualChunkSpec dc = spec(0);
        int n = 5;
        int qDim = N_HEAD * D_K;
        int pitch = n * qDim;
        float[] q = new float[DualChunkRope.VARIANTS * pitch];
        SplittableRandom rng = new SplittableRandom(3);
        float[] original = new float[n * qDim];
        for (int i = 0; i < original.length; i++) {
            original[i] = (float) (-1 + 2 * rng.nextDouble());
        }
        System.arraycopy(original, 0, q, 0, original.length);
        int[] pos = {0, 3, 4, 5, 11};
        int[] scratch = new int[DualChunkRope.VARIANTS * n];
        DualChunkRope.applyQuery(table, dc, q, pitch, n, N_HEAD, D_K, qDim, 0, pos, 0, scratch);
        int[] derived = new int[DualChunkRope.VARIANTS * n];
        DualChunkRope.queryPositions(dc, pos, 0, n, derived);
        for (int plane = 0; plane < DualChunkRope.VARIANTS; plane++) {
            for (int t = 0; t < n; t++) {
                float[] expected = new float[qDim];
                referenceRope(original, t * qDim, N_HEAD, D_K, derived[plane * n + t], expected);
                for (int i = 0; i < qDim; i++) {
                    assertEquals("plane " + plane + " token " + t + " element " + i,
                            expected[i], q[plane * pitch + t * qDim + i], 2e-5f);
                }
            }
        }

        float[] k = new float[n * N_HEAD_KV * D_K];
        System.arraycopy(original, 0, k, 0, k.length);
        float[] kBefore = k.clone();
        DualChunkRope.applyKey(table, dc, k, n, N_HEAD_KV, D_K, N_HEAD_KV * D_K, 0, pos, 0, scratch);
        for (int t = 0; t < n; t++) {
            float[] expected = new float[N_HEAD_KV * D_K];
            referenceRope(kBefore, t * N_HEAD_KV * D_K, N_HEAD_KV, D_K, pos[t] % CHUNK_LEN, expected);
            for (int i = 0; i < expected.length; i++) {
                assertEquals("key token " + t + " element " + i, expected[i],
                        k[t * N_HEAD_KV * D_K + i], 2e-5f);
            }
        }
    }

    /**
     * NORM rotation of one token's heads (common/rope_common.h): pair {@code i} is {@code (2i, 2i + 1)}, the angle is
     * {@code p * base^(-2i/d)}, and {@code (x0, x1)} becomes {@code (x0 c - x1 s, x0 s + x1 c)}.
     */
    private static void referenceRope(float[] src, int off, int nHeads, int d, int p, float[] out) {
        for (int h = 0; h < nHeads; h++) {
            int base = off + h * d;
            for (int i = 0; i < d / 2; i++) {
                double theta = p * Math.pow(10000.0, -2.0 * i / d);
                float c = (float) Math.cos(theta);
                float s = (float) Math.sin(theta);
                float x0 = src[base + 2 * i];
                float x1 = src[base + 2 * i + 1];
                out[h * d + 2 * i] = x0 * c - x1 * s;
                out[h * d + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }

    // ================================================================== attention grouping

    @Test
    public void prefillMergesTheThreeStagesIntoOneSoftmax() {
        for (int pos0 : new int[]{0, 3, 6, 9, 12, 20}) {
            Fixture f = new Fixture(pos0 + 6, 11 + pos0);
            check(f, pos0, 6);
        }
        // a single chunk, a second chunk and several chunks after it
        check(new Fixture(4, 12), 0, 4);
        check(new Fixture(CHUNK_LEN + 3, 22), 0, CHUNK_LEN + 3);
        check(new Fixture(3 * CHUNK_LEN + 2, 33), 0, 3 * CHUNK_LEN + 2);
    }

    @Test
    public void decodeUsesTheCurrentAndPrecedingChunks() {
        for (int keys : new int[]{1, CHUNK_LEN, CHUNK_LEN + 1, 2 * CHUNK_LEN, 2 * CHUNK_LEN + 3,
                3 * CHUNK_LEN + 1}) {
            Fixture f = new Fixture(keys, 7 + keys);
            check(f, keys - 1, 1);
        }
    }

    /** A synthetic sequence of {@code keys} rows and the five query planes of its last {@code n} tokens. */
    private static final class Fixture {
        final int keys;
        final float[] k;
        final float[] v;
        final int[] pos;

        Fixture(int keys, long seed) {
            this.keys = keys;
            SplittableRandom rng = new SplittableRandom(seed);
            this.k = new float[keys * N_HEAD_KV * D_K];
            this.v = new float[keys * N_HEAD_KV * D_V];
            fill(k, rng);
            fill(v, rng);
            this.pos = new int[keys];
            for (int t = 0; t < keys; t++) {
                pos[t] = t;
            }
        }

        private static void fill(float[] a, SplittableRandom rng) {
            for (int i = 0; i < a.length; i++) {
                a[i] = (float) (-1 + 2 * rng.nextDouble());
            }
        }
    }

    private static void check(Fixture f, int pos0, int n) {
        int keys = pos0 + n;
        assertTrue("fixture holds " + f.keys + " rows, the call needs " + keys, f.keys >= keys);
        SplittableRandom rng = new SplittableRandom(17 + keys + n);
        int qDim = N_HEAD * D_K;
        int pitch = n * qDim;
        float[] q = new float[DualChunkRope.VARIANTS * pitch];
        for (int i = 0; i < q.length; i++) {
            q[i] = (float) (-1 + 2 * rng.nextDouble());
        }
        int[] pos = new int[n];
        for (int t = 0; t < n; t++) {
            pos[t] = pos0 + t;
        }
        CpuAttentionCore.Params p = CpuAttentionCore.Params.builder(N_HEAD).scale(SCALE).build();
        CpuAttentionCore.DenseKv kv = CpuAttentionCore.DenseKv.of(f.k, 0, N_HEAD_KV * D_K, D_K, f.v, 0,
                N_HEAD_KV * D_V, D_V, N_HEAD_KV, D_K, D_V, f.pos, 0);
        CpuDualChunkAttention.KeySpan span = new CpuDualChunkAttention.KeySpan() {
            @Override public int keys() {
                return keys;
            }

            @Override
            public void attend(CpuAttentionCore.Params params, int rows, float[] qa, int qOff, int qRowStride,
                               int[] qPos, int qPosOff, int keyStart, int keyEnd, float[] out, int outOff,
                               float[] lse, int lseOff) {
                CpuAttentionCore.attendRowsLse(params, rows, qa, qOff, qRowStride, qPos, qPosOff, kv, keyStart,
                        keyEnd, out, outOff, N_HEAD * D_V, lse, lseOff, N_HEAD);
            }
        };
        int scratchRows = Math.min(n, CHUNK_LEN);
        float[] out = new float[n * N_HEAD * D_V];
        CpuDualChunkAttention.attend(spec(0), attention(), p, p.nonCausal(), span, n, q, pitch, qDim, 0, pos, 0,
                out, 0, N_HEAD * D_V, new float[scratchRows * N_HEAD * D_V],
                new float[scratchRows * N_HEAD], new float[scratchRows * N_HEAD]);
        float[] expected = reference(q, n, pitch, qDim, f.k, f.v, pos, keys, pos0);
        for (int i = 0; i < out.length; i++) {
            assertEquals("pos0=" + pos0 + " n=" + n + " keys=" + keys + " value " + i,
                    expected[i], out[i], TOL);
        }
    }

    /** vLLM's one softmax: per row its own chunk (plane 0), the preceding chunk (1) and every earlier chunk (2). */
    private static float[] reference(float[] q, int n, int pitch, int qDim, float[] k, float[] v, int[] pos,
                                     int keys, int pos0) {
        int dV = N_HEAD * D_V;
        float[] out = new float[n * dV];
        int group = N_HEAD / N_HEAD_KV;
        for (int t = 0; t < n; t++) {
            int p = pos[t];
            int chunk = p / CHUNK_LEN;
            List<int[]> stages = new ArrayList<>();
            stages.add(new int[]{0, chunk * CHUNK_LEN, p + 1});
            if (chunk >= 1) {
                stages.add(new int[]{1, (chunk - 1) * CHUNK_LEN, chunk * CHUNK_LEN});
            }
            if (chunk >= 2) {
                stages.add(new int[]{2, 0, (chunk - 1) * CHUNK_LEN});
            }
            for (int h = 0; h < N_HEAD; h++) {
                int kvh = h / group;
                double max = Double.NEGATIVE_INFINITY;
                List<Double> logits = new ArrayList<>();
                for (int[] stage : stages) {
                    int plane = stage[0];
                    for (int j = stage[1]; j < Math.min(stage[2], keys); j++) {
                        double dot = 0;
                        for (int d = 0; d < D_K; d++) {
                            dot += q[plane * pitch + t * qDim + h * D_K + d] * k[j * N_HEAD_KV * D_K + kvh * D_K + d];
                        }
                        double s = dot * SCALE;
                        logits.add(s);
                        max = Math.max(max, s);
                    }
                }
                double sum = 0;
                double[] weights = new double[logits.size()];
                for (int i = 0; i < weights.length; i++) {
                    weights[i] = Math.exp(logits.get(i) - max);
                    sum += weights[i];
                }
                double[] acc = new double[D_V];
                int i = 0;
                for (int[] stage : stages) {
                    for (int j = stage[1]; j < Math.min(stage[2], keys); j++) {
                        double w = weights[i++] / sum;
                        for (int d = 0; d < D_V; d++) {
                            acc[d] += w * v[j * N_HEAD_KV * D_V + kvh * D_V + d];
                        }
                    }
                }
                for (int d = 0; d < D_V; d++) {
                    out[t * dV + h * D_V + d] = (float) acc[d];
                }
            }
        }
        return out;
    }
}
