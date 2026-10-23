package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuForwardContext;
import com.eonmux.neuralfree.gguf.cpu.forward.CpuScratch;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.ModelHparamsBuilder;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.HeadSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.MlaSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.NormSpec;
import com.eonmux.neuralfree.gguf.spec.ResidualSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * Numeric checks of the CPU MLA mixer ({@link CpuMlaMixer}) with an F32 latent cache against a direct transcription
 * of the llama.cpp absorbed-MLA graph (models/deepseek2.cpp:493-620, models/dots3note.cpp:330-344,
 * models/hy-v4.cpp:286-327): the {@code attn_k_norm} RMS norm of the shared rotary key, the per-head {@code attn_k_b}
 * absorption and {@code attn_v_b} expansion, and the per-head {@code attn_sinks} logits in the softmax denominator. The
 * layer applies no RoPE ({@code ropeApplied} false), so the reference needs no rotary tables.
 */
public class CpuMlaMixerTest {

    private static final int N_EMBD = 32;
    private static final int N_HEAD = 2;
    private static final int NOPE = 8;
    private static final int ROPE = 8;
    private static final int KV_LORA = 16;
    private static final int HEAD_V = 8;
    private static final int HEAD_K = NOPE + ROPE;
    private static final int WIDTH = KV_LORA + ROPE;
    private static final int N_CTX = 32;
    private static final int UBATCH = 8;
    private static final float KQ_SCALE = 0.25f;
    private static final float EPS = 1e-6f;

    /** A one-layer MLA file, its weights and memory, and the bound mixer. */
    private static final class Fixture implements AutoCloseable {
        final float[] wq = random(1, N_HEAD * HEAD_K * N_EMBD, -0.5f, 0.5f);
        final float[] wkva = random(2, WIDTH * N_EMBD, -0.5f, 0.5f);
        final float[] kvaNorm = random(3, KV_LORA, 0.5f, 1.5f);
        final float[] kNorm = random(4, ROPE, 0.5f, 1.5f);
        final float[] wkB = random(5, N_HEAD * KV_LORA * NOPE, -0.5f, 0.5f);
        final float[] wvB = random(6, N_HEAD * HEAD_V * KV_LORA, -0.5f, 0.5f);
        final float[] wo = random(7, N_EMBD * N_HEAD * HEAD_V, -0.5f, 0.5f);
        final float[] sinkLogits = random(8, N_HEAD, -1.0f, 1.0f);
        final boolean kPeNorm;
        final boolean sinks;
        final Path path;
        final GGUFFileParser parser;
        final CpuWeights weights;
        final LayerSpec layer;
        final HeapModelMemory memory;
        final CpuMlaMixer mixer;

        Fixture(boolean kPeNorm, boolean sinks) throws IOException {
            this.kPeNorm = kPeNorm;
            this.sinks = sinks;
            GgufTestWriter w = new GgufTestWriter().architecture(Arch.DEEPSEEK2.ggufName());
            put(w, TensorNames.ATTN_Q, wq, N_EMBD, N_HEAD * HEAD_K);
            put(w, TensorNames.ATTN_KV_A_MQA, wkva, N_EMBD, WIDTH);
            put(w, TensorNames.ATTN_KV_A_NORM, kvaNorm, KV_LORA);
            put(w, TensorNames.ATTN_K_NORM, kNorm, ROPE);
            put(w, TensorNames.ATTN_K_B, wkB, NOPE, KV_LORA, N_HEAD);
            put(w, TensorNames.ATTN_V_B, wvB, KV_LORA, HEAD_V, N_HEAD);
            put(w, TensorNames.ATTN_OUT, wo, N_HEAD * HEAD_V, N_EMBD);
            put(w, TensorNames.ATTN_SINKS, sinkLogits, N_HEAD);
            this.path = w.writeTempFile();
            this.parser = new GGUFFileParser(path.toString());
            this.weights = CpuWeights.open(parser);

            ModelHparamsBuilder hb = new ModelHparamsBuilder(ArchRegistry.traits(Arch.DEEPSEEK2)).layerCounts(1, 0)
                    .nCtxTrain(4096).nEmbd(N_EMBD).normRmsEps(EPS).headKFull(WIDTH).headVFull(KV_LORA);
            hb.nHead(0, N_HEAD).nHeadKv(0, 1);
            MlaSpec mla = MlaSpec.builder().nHead(N_HEAD).qLoraRank(0).kvLoraRank(KV_LORA).headKNope(NOPE)
                    .headKRope(ROPE).headV(HEAD_V).absorbed(true).latentCache(true).ropeApplied(false)
                    .kqScale(KQ_SCALE).build();
            AttentionSpec a = AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_K).headDimV(HEAD_V)
                    .nRot(0).noRope(true).qkNorm(kPeNorm ? QkNormKind.RMS_HEAD : QkNormKind.NONE).sinks(sinks)
                    .kqScale(KQ_SCALE).kvOwnerLayer(0).build();
            this.layer = LayerSpec.builder(0, MixerKind.MLA, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).qkNorm(a.qkNorm()).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .mla(mla)
                    .attention(a)
                    .build();
            ModelSpec spec = ModelSpec.builder(hb.build(), TensorCatalog.of(parser))
                    .schedule(LayerSchedule.linear(1))
                    .embedding(EmbeddingSpec.builder(null, N_EMBD).build())
                    .head(HeadSpec.builder().build())
                    .layer(layer)
                    .build();
            MemoryConfig cfg = MemoryConfig.builder().ubatch(UBATCH).typeK(KvDType.F32).typeV(KvDType.F32).build();
            MemoryPlan plan = MemoryPlanner.planForContext(spec, cfg, N_CTX);
            this.memory = HeapModelMemory.allocate(plan, spec, null);
            CpuForwardContext ctx = CpuForwardContext.builder(spec, weights, CpuScratch.forModel(spec, UBATCH))
                    .memory(memory).build();
            this.mixer = (CpuMlaMixer) MlaBlocks.cpuMixers().get(0).create(layer);
            mixer.bind(ctx, layer);
        }

        float[] step(int pos0, int n, float[] in) {
            memory.beginStep(pos0, n);
            float[] out = new float[n * N_EMBD];
            if (n == 1) {
                mixer.decode(pos0, in, out);
            } else {
                mixer.prefill(pos0, n, in, out);
            }
            return out;
        }

        @Override
        public void close() throws IOException {
            try {
                mixer.close();
                memory.close();
                weights.close();
            } finally {
                try {
                    parser.close();
                } finally {
                    Files.deleteIfExists(path);
                }
            }
        }
    }

    private static void put(GgufTestWriter w, String base, float[] values, long... ne) {
        w.tensorFromFloats(TensorNames.layerName(0, TensorNames.weight(base)), GGMLType.F32, ne, values);
    }

    private static float[] random(long seed, int n, float lo, float hi) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (lo + (hi - lo) * rng.nextDouble());
        }
        return v;
    }

    // ------------------------------------------------------------------ reference

    /** {@code y[r] = sum_c w[r*cols + c] * x[xOff + c]} for {@code r < rows}, w row-major. */
    private static double[] matvec(float[] w, int rows, int cols, double[] x, int xOff) {
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double acc = 0;
            for (int c = 0; c < cols; c++) {
                acc += w[r * cols + c] * x[xOff + c];
            }
            y[r] = acc;
        }
        return y;
    }

    /** {@code ggml_rms_norm} of {@code v[off .. off + n)} in place, then the weight. */
    private static void rmsNorm(double[] v, int off, int n, float[] weight) {
        double ss = 0;
        for (int i = 0; i < n; i++) {
            ss += v[off + i] * v[off + i];
        }
        double scale = 1.0 / Math.sqrt(ss / n + EPS);
        for (int i = 0; i < n; i++) {
            v[off + i] = v[off + i] * scale * weight[i];
        }
    }

    /**
     * The layer outputs of tokens {@code [0, total)} with causal attention over every earlier latent row: the absorbed
     * MLA graph, the optional k_pe norm and the optional sinks.
     */
    private static double[][] reference(Fixture f, float[] in, int total) {
        double[][] latentRows = new double[total][];
        double[][] queries = new double[total][];
        for (int t = 0; t < total; t++) {
            double[] x = new double[N_EMBD];
            for (int i = 0; i < N_EMBD; i++) {
                x[i] = in[t * N_EMBD + i];
            }
            double[] q = matvec(f.wq, N_HEAD * HEAD_K, N_EMBD, x, 0);
            double[] kva = matvec(f.wkva, WIDTH, N_EMBD, x, 0);
            if (f.kPeNorm) {
                rmsNorm(kva, KV_LORA, ROPE, f.kNorm);
            }
            rmsNorm(kva, 0, KV_LORA, f.kvaNorm);
            latentRows[t] = kva;
            double[] qlat = new double[N_HEAD * WIDTH];
            for (int h = 0; h < N_HEAD; h++) {
                for (int l = 0; l < KV_LORA; l++) {
                    double acc = 0;
                    for (int d = 0; d < NOPE; d++) {
                        acc += f.wkB[(h * KV_LORA + l) * NOPE + d] * q[h * HEAD_K + d];
                    }
                    qlat[h * WIDTH + l] = acc;
                }
                for (int d = 0; d < ROPE; d++) {
                    qlat[h * WIDTH + KV_LORA + d] = q[h * HEAD_K + NOPE + d];
                }
            }
            queries[t] = qlat;
        }
        double[][] out = new double[total][];
        for (int t = 0; t < total; t++) {
            double[] attn = new double[N_HEAD * HEAD_V];
            for (int h = 0; h < N_HEAD; h++) {
                double[] s = new double[t + 1];
                double m = f.sinks ? f.sinkLogits[h] : Double.NEGATIVE_INFINITY;
                for (int j = 0; j <= t; j++) {
                    double dot = 0;
                    for (int i = 0; i < WIDTH; i++) {
                        dot += queries[t][h * WIDTH + i] * latentRows[j][i];
                    }
                    s[j] = dot * KQ_SCALE;
                    m = Math.max(m, s[j]);
                }
                double denom = f.sinks ? Math.exp(f.sinkLogits[h] - m) : 0;
                double[] lat = new double[KV_LORA];
                for (int j = 0; j <= t; j++) {
                    double p = Math.exp(s[j] - m);
                    denom += p;
                    for (int l = 0; l < KV_LORA; l++) {
                        lat[l] += p * latentRows[j][l];
                    }
                }
                for (int l = 0; l < KV_LORA; l++) {
                    lat[l] /= denom;
                }
                for (int v = 0; v < HEAD_V; v++) {
                    double acc = 0;
                    for (int l = 0; l < KV_LORA; l++) {
                        acc += f.wvB[(h * HEAD_V + v) * KV_LORA + l] * lat[l];
                    }
                    attn[h * HEAD_V + v] = acc;
                }
            }
            out[t] = matvec(f.wo, N_EMBD, N_HEAD * HEAD_V, attn, 0);
        }
        return out;
    }

    private static void assertRows(double[][] expected, int first, float[] actual, int n) {
        for (int t = 0; t < n; t++) {
            for (int i = 0; i < N_EMBD; i++) {
                double e = expected[first + t][i];
                assertEquals("token " + (first + t) + " element " + i, e, actual[t * N_EMBD + i],
                        1e-4 * Math.max(1.0, Math.abs(e)));
            }
        }
    }

    private void check(boolean kPeNorm, boolean sinks) throws IOException {
        int prefill = 5;
        float[] in = random(42, (prefill + 1) * N_EMBD, -1.0f, 1.0f);
        try (Fixture f = new Fixture(kPeNorm, sinks)) {
            double[][] expected = reference(f, in, prefill + 1);
            float[] first = new float[prefill * N_EMBD];
            System.arraycopy(in, 0, first, 0, first.length);
            assertRows(expected, 0, f.step(0, prefill, first), prefill);
            float[] last = new float[N_EMBD];
            System.arraycopy(in, prefill * N_EMBD, last, 0, N_EMBD);
            assertRows(expected, prefill, f.step(prefill, 1, last), 1);
        }
    }

    // ------------------------------------------------------------------ tests

    @Test
    public void absorbedLatentStepsMatchTheGraph() throws IOException {
        check(false, false);
    }

    @Test
    public void sharedRotaryKeyIsRmsNormalizedBeforeItIsCached() throws IOException {
        check(true, false);
    }

    @Test
    public void sinksJoinTheSoftmaxDenominatorPerHead() throws IOException {
        check(false, true);
    }

    @Test
    public void kPeNormAndSinksTogether() throws IOException {
        check(true, true);
    }

    @Test
    public void unsupportedQkNormIsRefused() {
        MlaSpec mla = MlaSpec.builder().nHead(N_HEAD).qLoraRank(0).kvLoraRank(KV_LORA).headKNope(NOPE)
                .headKRope(ROPE).headV(HEAD_V).absorbed(true).latentCache(true).ropeApplied(false)
                .kqScale(KQ_SCALE).build();
        for (AttentionSpec a : new AttentionSpec[]{
                facet().noRope(true).nRot(0).qkNorm(QkNormKind.LN_HEAD).build(),
                facet().qkNorm(QkNormKind.RMS_HEAD).qkNormAfterRope(true).build()}) {
            LayerSpec layer = LayerSpec.builder(0, MixerKind.MLA, FfnKind.NONE)
                    .norm(NormSpec.builder(NormKind.RMS, EPS).qkNorm(a.qkNorm()).build())
                    .residual(ResidualSpec.builder(ResidualKind.STANDARD, N_EMBD).build())
                    .mla(mla)
                    .attention(a)
                    .build();
            assertThrows(UnsupportedOperationException.class, () -> MlaBlocks.cpuMixers().get(0).create(layer));
            assertThrows(UnsupportedOperationException.class, () -> MlaBlocks.gpuMixers().get(0).create(layer));
            assertThrows(UnsupportedOperationException.class, () -> MlaBlocks.kPeNorm(layer));
        }
    }

    /**
     * The MLA facet of the refusal layers: roped over the {@code ROPE} dims, so a Q/K norm may be placed after RoPE.
     * The MLA spec applies no RoPE; the refusals read only the facet.
     */
    private static AttentionSpec.Builder facet() {
        return AttentionSpec.builder().nHead(N_HEAD).nHeadKv(N_HEAD).headDimK(HEAD_K).headDimV(HEAD_V).nRot(ROPE)
                .kqScale(KQ_SCALE).kvOwnerLayer(0);
    }
}
