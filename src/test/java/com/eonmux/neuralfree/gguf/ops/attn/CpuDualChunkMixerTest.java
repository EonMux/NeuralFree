package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.CpuMixerRig;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;

import static org.junit.Assert.assertArrayEquals;

/**
 * The CPU attention mixer's dual-chunk path (Qwen2.5-1M) against a direct transcription of vLLM's formulas: a
 * Qwen2-shaped layer (GQA, Q/K/V biases, NORM RoPE) whose context runs past several chunk lengths, its prefill steps
 * and decodes matching the reference step by step, over a contiguous cache and over a paged one.
 *
 * <p>The reference projects, rotates the five query variants and the keys at {@code position % (chunk_size -
 * local_size)}, applies the original-context scaling to the new K rows and to the queries, and scores every query
 * with one softmax over its own chunk (query plane 0, causal), the preceding chunk (plane 1) and every earlier chunk
 * (plane 2).
 */
public class CpuDualChunkMixerTest {

    private static final int N_EMBD = 16;
    private static final int N_HEAD = 2;
    private static final int N_HEAD_KV = 1;
    private static final int HEAD_DIM = 8;
    private static final int Q_DIM = N_HEAD * HEAD_DIM;
    private static final int KV_DIM = N_HEAD_KV * HEAD_DIM;
    private static final int CHUNK_SIZE = 6;
    private static final int LOCAL_SIZE = 2;
    private static final int CHUNK_LEN = CHUNK_SIZE - LOCAL_SIZE;
    private static final int ORIGINAL_MAX = 4;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;
    private static final float EPS = 1e-6f;
    private static final float BASE = 10000f;
    private static final float SCALE = (float) (1.0 / Math.sqrt(HEAD_DIM));
    private static final float TOL = 3e-4f;

    private static DualChunkSpec dualChunk() {
        return DualChunkSpec.builder().chunkSize(CHUNK_SIZE).localSize(LOCAL_SIZE)
                .originalMaxPositionEmbeddings(ORIGINAL_MAX).build();
    }

    private static final class Fixture implements AutoCloseable {
        final Path path;
        final CpuWeights weights;
        final ModelSpec spec;
        final LayerSpec layer;
        final float[] wq;
        final float[] bq;
        final float[] wk;
        final float[] bk;
        final float[] wv;
        final float[] bv;
        final float[] wo;

        Fixture() throws IOException {
            SplittableRandom rng = new SplittableRandom(5);
            GgufTestWriter w = new GgufTestWriter().architecture(Arch.QWEN2.ggufName());
            wq = matrix(w, rng, TensorNames.ATTN_Q, N_EMBD, Q_DIM);
            wk = matrix(w, rng, TensorNames.ATTN_K, N_EMBD, KV_DIM);
            wv = matrix(w, rng, TensorNames.ATTN_V, N_EMBD, KV_DIM);
            wo = matrix(w, rng, TensorNames.ATTN_OUT, Q_DIM, N_EMBD);
            bq = bias(w, rng, TensorNames.bias(TensorNames.ATTN_Q), Q_DIM);
            bk = bias(w, rng, TensorNames.bias(TensorNames.ATTN_K), KV_DIM);
            bv = bias(w, rng, TensorNames.bias(TensorNames.ATTN_V), KV_DIM);
            this.path = w.writeTempFile();
            com.eonmux.neuralfree.gguf.GGUFFileParser parser =
                    new com.eonmux.neuralfree.gguf.GGUFFileParser(path.toString());
            this.weights = CpuWeights.open(parser);

            ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.QWEN2)).layerCounts(1, 0)
                    .nCtxTrain(128).nEmbd(N_EMBD).normRmsEps(EPS).headKFull(HEAD_DIM).headVFull(HEAD_DIM)
                    .nRotFull(HEAD_DIM);
            hb.nHead(0, N_HEAD).nHeadKv(0, N_HEAD_KV);
            AttentionSpec attention = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD_KV)
                    .headDimK(HEAD_DIM).headDimV(HEAD_DIM).nRot(HEAD_DIM).kqScale(SCALE).kvOwnerLayer(0)
                    .qkNorm(QkNormKind.NONE).dualChunk(dualChunk()).build();
            this.layer = LayerSpec.builder(0, MixerKind.ATTENTION, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .rope(RopeSpec.builder(RopeType.NORM).nDims(HEAD_DIM).freqBase(BASE).build())
                    .attention(attention)
                    .build();
            this.spec = ModelSpec.builder(hb.build(), TensorCatalog.of(parser))
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                    .head(HeadSpec.builder().build())
                    .layer(layer)
                    .build();
        }

        private static float[] matrix(GgufTestWriter w, SplittableRandom rng, String base, int in, int out) {
            float[] v = new float[in * out];
            for (int i = 0; i < v.length; i++) {
                v[i] = (float) (-0.5 + rng.nextDouble());
            }
            w.tensorFromFloats(TensorNames.layerName(0, TensorNames.weight(base)), GGMLType.F32,
                    new long[]{in, out}, v);
            return v;
        }

        private static float[] bias(GgufTestWriter w, SplittableRandom rng, String name, int n) {
            float[] v = new float[n];
            for (int i = 0; i < n; i++) {
                v[i] = (float) (-0.25 + 0.5 * rng.nextDouble());
            }
            w.tensorFromFloats(TensorNames.layerName(0, name), GGMLType.F32, new long[]{n}, v);
            return v;
        }

        CpuMixerRig rig(boolean paged) {
            MemoryConfig.Builder cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32)
                    .typeV(KvDType.F32);
            if (paged) {
                cfg = cfg.parallel(2);
            }
            return CpuMixerRig.open(spec, weights, cfg.build(), N_CTX, N_EMBD, l -> new CpuAttentionMixer());
        }

        /** The layer's reference output of one step, over {@link Ref#k} / {@link Ref#v} as of the step's start. */
        float[] reference(Ref ref, int pos0, int n, float[] in) {
            int seqLen = n == 1 ? pos0 + 1 : pos0 + n;
            float f = dualChunk().scalingFactor(seqLen);
            float[][] qp = new float[3][n * Q_DIM];
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
                float[] q = matvec(wq, N_EMBD, Q_DIM, in, t * N_EMBD, bq);
                float[] k = matvec(wk, N_EMBD, KV_DIM, in, t * N_EMBD, bk);
                float[] v = matvec(wv, N_EMBD, KV_DIM, in, t * N_EMBD, bv);
                rope(k, HEAD_DIM, KV_DIM, p % CHUNK_LEN);
                for (int plane = 0; plane < 3; plane++) {
                    float[] qr = q.clone();
                    rope(qr, HEAD_DIM, Q_DIM, Derived.planePositions(dualChunk(), p)[plane]);
                    for (int i = 0; i < Q_DIM; i++) {
                        qp[plane][t * Q_DIM + i] = qr[i] * f;
                    }
                }
                for (int i = 0; i < KV_DIM; i++) {
                    ref.k[p * KV_DIM + i] = k[i] * f;
                    ref.v[p * KV_DIM + i] = v[i];
                }
            }
            float[] out = new float[n * N_EMBD];
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
                int chunk = p / CHUNK_LEN;
                int[] planes = new int[3];
                int[] starts = new int[3];
                int[] ends = new int[3];
                int stages = 1;
                planes[0] = 0;
                starts[0] = chunk * CHUNK_LEN;
                ends[0] = p + 1;
                if (chunk >= 1) {
                    planes[stages] = 1;
                    starts[stages] = (chunk - 1) * CHUNK_LEN;
                    ends[stages] = chunk * CHUNK_LEN;
                    stages++;
                }
                if (chunk >= 2) {
                    planes[stages] = 2;
                    starts[stages] = 0;
                    ends[stages] = (chunk - 1) * CHUNK_LEN;
                    stages++;
                }
                float[] attn = ref.attnRow;
                for (int h = 0; h < N_HEAD; h++) {
                    int kvh = h / (N_HEAD / N_HEAD_KV);
                    double max = Double.NEGATIVE_INFINITY;
                    double[] logits = new double[count(starts, ends, stages)];
                    int idx = 0;
                    for (int s = 0; s < stages; s++) {
                        for (int j = starts[s]; j < ends[s]; j++) {
                            double dot = 0;
                            for (int d = 0; d < HEAD_DIM; d++) {
                                dot += qp[s][t * Q_DIM + h * HEAD_DIM + d] * ref.k[j * KV_DIM + kvh * HEAD_DIM + d];
                            }
                            logits[idx++] = dot * SCALE;
                            max = Math.max(max, logits[idx - 1]);
                        }
                    }
                    double sum = 0;
                    for (int i = 0; i < logits.length; i++) {
                        logits[i] = Math.exp(logits[i] - max);
                        sum += logits[i];
                    }
                    double[] acc = new double[HEAD_DIM];
                    idx = 0;
                    for (int s = 0; s < stages; s++) {
                        for (int j = starts[s]; j < ends[s]; j++) {
                            double weight = logits[idx++] / sum;
                            for (int d = 0; d < HEAD_DIM; d++) {
                                acc[d] += weight * ref.v[j * KV_DIM + kvh * HEAD_DIM + d];
                            }
                        }
                    }
                    for (int d = 0; d < HEAD_DIM; d++) {
                        attn[h * HEAD_DIM + d] = (float) acc[d];
                    }
                }
                float[] row = matvec(wo, Q_DIM, N_EMBD, attn, 0, null);
                System.arraycopy(row, 0, out, t * N_EMBD, N_EMBD);
            }
            return out;
        }

        private static int count(int[] starts, int[] ends, int stages) {
            int n = 0;
            for (int s = 0; s < stages; s++) {
                n += Math.max(0, ends[s] - starts[s]);
            }
            return n;
        }

        private static float[] matvec(float[] w, int in, int out, float[] x, int xOff, float[] bias) {
            float[] y = new float[out];
            for (int o = 0; o < out; o++) {
                float acc = bias != null ? bias[o] : 0f;
                for (int i = 0; i < in; i++) {
                    acc += w[o * in + i] * x[xOff + i];
                }
                y[o] = acc;
            }
            return y;
        }

        /** NORM rotation of every head at position {@code p}. */
        private static void rope(float[] x, int headDim, int width, int p) {
            for (int h = 0; h < width / headDim; h++) {
                int base = h * headDim;
                for (int i = 0; i < headDim / 2; i++) {
                    double theta = p * Math.pow(BASE, -2.0 * i / headDim);
                    float c = (float) Math.cos(theta);
                    float s = (float) Math.sin(theta);
                    float x0 = x[base + 2 * i];
                    float x1 = x[base + 2 * i + 1];
                    x[base + 2 * i] = x0 * c - x1 * s;
                    x[base + 2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }

        @Override
        public void close() throws IOException {
            weights.close();
            Files.deleteIfExists(path);
        }
    }

    /** The reference's own KV cache and per-row attention scratch. */
    private static final class Ref {
        final float[] k = new float[N_CTX * KV_DIM];
        final float[] v = new float[N_CTX * KV_DIM];
        final float[] attnRow = new float[Q_DIM];
    }

    /** vLLM's five query positions of one token. */
    private static final class Derived {
        static int[] planePositions(DualChunkSpec dc, int p) {
            int chunkLen = dc.chunkLength();
            int base = p % chunkLen;
            return new int[]{base, Math.min(base + chunkLen, dc.chunkSize()),
                    Math.min(2 * chunkLen - 1, dc.chunkSize())};
        }
    }

    private static float[] inputs(long seed, int rows) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] v = new float[rows * N_EMBD];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (-1 + 2 * rng.nextDouble());
        }
        return v;
    }

    /**
     * The cache-write-only step stores exactly the keys the full step would: each projected row rotated at
     * {@code position % chunkLength} and scaled by the sequence's factor, the values untouched.
     */
    @Test
    public void theCacheWriteOnlyStepStoresTheDualChunkKeys() throws IOException {
        try (Fixture f = new Fixture(); CpuMixerRig rig = f.rig(false)) {
            int n = 5;
            int pos0 = 0;
            float[] in = inputs(13, n);
            rig.memory().beginStep(pos0, n);
            int[] pos = rig.context().scratch().ints(CpuScratch.POS);
            for (int t = 0; t < n; t++) {
                pos[t] = pos0 + t;
            }
            rig.mixer(0).prefillKV(pos0, n, in);

            float factor = dualChunk().scalingFactor(pos0 + n);
            float[] row = new float[KV_DIM];
            for (int t = 0; t < n; t++) {
                int p = pos0 + t;
                float[] key = Fixture.matvec(f.wk, N_EMBD, KV_DIM, in, t * N_EMBD, f.bk);
                Fixture.rope(key, HEAD_DIM, KV_DIM, p % CHUNK_LEN);
                for (int i = 0; i < KV_DIM; i++) {
                    key[i] *= factor;
                }
                rig.memory().kv(0).readK(p, 0, row, 0);
                assertArrayEquals("key row " + p, key, row, TOL);
            }
        }
    }

    @Test
    public void contiguousStepsMatchTheReference() throws IOException {
        try (Fixture f = new Fixture(); CpuMixerRig rig = f.rig(false)) {
            Ref ref = new Ref();
            int[] pos0s = {0, 3, 6, 7, 8, 10, 11, 12, 13, 15};
            int[] counts = {3, 3, 1, 1, 2, 1, 1, 1, 2, 1};
            float[] all = inputs(9, 16);
            int at = 0;
            for (int s = 0; s < pos0s.length; s++) {
                float[] in = new float[counts[s] * N_EMBD];
                System.arraycopy(all, at * N_EMBD, in, 0, in.length);
                at += counts[s];
                float[] expected = f.reference(ref, pos0s[s], counts[s], in);
                float[] out = rig.contiguous(pos0s[s], counts[s], in);
                assertArrayEquals("step " + s + " at " + pos0s[s] + " + " + counts[s], expected,
                        slice(out, 0, counts[s]), TOL);
            }
        }
    }

    @Test
    public void pagedStepsMatchTheReference() throws IOException {
        try (Fixture f = new Fixture(); CpuMixerRig rig = f.rig(true)) {
            Ref ref = new Ref();
            int[] pos0s = {0, 3, 6, 7, 8, 10, 11, 12, 13, 15};
            int[] counts = {3, 3, 1, 1, 2, 1, 1, 1, 2, 1};
            float[] all = inputs(13, 16);
            int at = 0;
            for (int s = 0; s < pos0s.length; s++) {
                float[] in = new float[counts[s] * N_EMBD];
                System.arraycopy(all, at * N_EMBD, in, 0, in.length);
                at += counts[s];
                float[] expected = f.reference(ref, pos0s[s], counts[s], in);
                float[] out = rig.paged(new int[]{0}, new int[]{counts[s]}, new int[]{pos0s[s]}, in, false);
                assertArrayEquals("paged step " + s + " at " + pos0s[s] + " + " + counts[s], expected,
                        slice(out, 0, counts[s]), TOL);
            }
        }
    }

    @Test
    public void twoPagedSequencesKeepTheirChunksApart() throws IOException {
        try (Fixture f = new Fixture(); CpuMixerRig one = f.rig(true); CpuMixerRig both = f.rig(true)) {
            float[] inA = inputs(21, 4);
            float[] inB = inputs(22, 4);
            float[] aloneA = one.paged(new int[]{0}, new int[]{4}, new int[]{0}, inA, false);
            float[] aloneB = one.paged(new int[]{1}, new int[]{4}, new int[]{0}, inB, false);
            float[] in = new float[8 * N_EMBD];
            System.arraycopy(inA, 0, in, 0, inA.length);
            System.arraycopy(inB, 0, in, inA.length, inB.length);
            float[] out = both.paged(new int[]{0, 1}, new int[]{4, 4}, new int[]{0, 0}, in, false);
            assertArrayEquals("sequence A", slice(aloneA, 0, 4), slice(out, 0, 4), 1e-5f);
            assertArrayEquals("sequence B", slice(aloneB, 0, 4), slice(out, 4, 4), 1e-5f);
        }
    }

    private static float[] slice(float[] out, int row, int n) {
        float[] s = new float[n * N_EMBD];
        System.arraycopy(out, row * N_EMBD, s, 0, s.length);
        return s;
    }

}
