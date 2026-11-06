package com.eonmux.neuralfree.gguf.ops.sparse;

import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.gpu.block.GpuFactoryCatalog;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Full-model logits parity of the Step5 graph: the CPU engine's prefill logits against an independent scalar
 * transcription of vLLM's Step5 dense forward (GQA with per-head zero-centered RMS norms, partial NeoX RoPE, the
 * head gate, the region-summary indexer with weighted-ReLU scoring and stable top-k, SwiGLU FFNs and the final norm).
 *
 * <p>The transcription reproduces the representation points the vLLM kernels materialize: the per-head norm outputs
 * and the RoPE outputs of the main and indexer Q/K paths, the indexer Z rows, the BF16 keys/values in the cache, the
 * indexer queries and the region summaries in FP8 E4M3, the BF16 softmax weights and head gate, and the BF16
 * attention output. Everything else stays in F32, as the CPU forward runs it.</p>
 */
public class CpuStep5ParityTest {

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 32;
    private static final int V = 7;
    private static final int HEADS = 8;
    private static final int KV_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int ROTARY = 4;
    private static final int FF = 6;
    private static final int PROXY = 256;
    private static final int IDX_HEADS = 8;
    private static final int IDX_ROPE = 32;
    private static final int REGION = 8;
    private static final int TOP_K = 2;
    private static final int LAYERS = 2;
    private static final float EPS = 1e-5f;
    private static final float THETA = 10000f;
    private static final int TOKENS = 32;
    private static final int REGIONS = (TOKENS + REGION - 1) / REGION;
    private static final int SEL_WIDTH = Math.min(TOP_K, REGIONS) + 1;
    private static final float LOG2E = 1.4426950408889634f;

    private static final String SPARSE = "'sparse_config':{'enabled':true,'proxy_dim':" + PROXY
            + ",'sparse_indexer_rope_dim':" + IDX_ROPE + ",'sparse_indexer_use_rope':true"
            + ",'sparse_indexer_num_heads':" + IDX_HEADS + ",'sparse_indexer_num_k_heads':1"
            + ",'sparse_indexer_q_norm_type':'rmsnorm','sparse_indexer_k_norm_type':'layernorm'"
            + ",'sparse_indexer_csa_z_norm_type':'none','index_tp_size':" + KV_HEADS
            + ",'topk':" + TOP_K + ",'region_block_size':8,'compression_method':'csa_block_compress'"
            + ",'attention_impl':'sparse_gqa','sparse_indexer_softmax_variant':'softmax'"
            + ",'sparse_indexer_ssmax_s_granularity':'q_head','apply_to_layer_types':['full_attention']}";

    private static final String CONFIG = SyntheticHf.json("{'architectures':['Step5ForCausalLM'],"
            + "'model_type':'step5','hidden_size':" + D + ",'intermediate_size':" + FF
            + ",'num_hidden_layers':" + LAYERS + ",'num_attention_heads':" + HEADS
            + ",'num_attention_groups':" + KV_HEADS + ",'head_dim':" + HEAD_DIM
            + ",'partial_rotary_factor':" + ((float) ROTARY / HEAD_DIM)
            + ",'max_position_embeddings':64,'rms_norm_eps':" + EPS + ",'vocab_size':" + V
            + ",'rope_theta':" + THETA + ",'hidden_act':'silu','tie_word_embeddings':false"
            + ",'zero_centered':true,'fp32_residual_connection':true,'att_impl_type':'GQA'"
            + ",'use_head_wise_attn_gate':true," + SPARSE + "}");

    @Test
    public void engineLogitsMatchTheScalarStep5Forward() throws Exception {
        Path gguf = export("parity");
        int[] prompt = new int[TOKENS];
        for (int i = 0; i < TOKENS; i++) {
            prompt[i] = (i * 3 + i / 5) % V;
        }
        float[] expected;
        try (GGUFFileParser parser = new GGUFFileParser(gguf.toString())) {
            expected = new Reference(parser).logits(prompt);
        }

        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(TOKENS).ubatch(8).typeK(KvDType.BF16).typeV(KvDType.BF16)
                        .build()).build();
        try (EngineModel model = LLMModelFactory.open(gguf.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(prompt, 0);
            assertEquals(V, actual.length);
            double peak = 0.0;
            for (float v : expected) {
                peak = Math.max(peak, Math.abs(v));
            }
            assertTrue("the reference logits are degenerate", peak > 0.05);
            for (int i = 0; i < V; i++) {
                assertEquals("logit " + i, expected[i], actual[i], 1e-6f);
            }
            assertEquals(argmax(expected), argmax(actual));
        }
    }

    @Test
    public void gpuRegistryResolvesTheStep5Mixer() throws Exception {
        Path gguf = export("gpu-mixer");
        try (GGUFFileParser parser = new GGUFFileParser(gguf.toString())) {
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(parser), TensorCatalog.of(parser),
                    MemoryConfig.defaults());
            LayerSpec layer = spec.layers().get(0);
            assertEquals(MixerKind.STEP5, layer.mixerKind());
            assertTrue(GpuFactoryCatalog.defaults().mixer(layer) instanceof GpuStep5Mixer);
        }
    }

    /** Converts the synthetic checkpoint under {@code name} and exports it as a GGUF. */
    private Path export(String name) throws Exception {
        Path dir = builder().write(tmp.newFolder(name + "-hf").toPath());
        Path gguf = tmp.getRoot().toPath().resolve(name + ".gguf");
        try (HfModelSource src = HfModelSource.open(dir, ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null)) {
            GgufWriter.write(src, gguf, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        return gguf;
    }

    private static int argmax(float[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) {
                best = i;
            }
        }
        return best;
    }

    // ================================================================== the scalar reference

    /** The Step5 dense forward transcribed from vLLM over the converted GGUF weights. */
    private static final class Reference {

        private final GGUFFileParser parser;
        private final float[] embed;

        Reference(GGUFFileParser parser) {
            this.parser = parser;
            this.embed = tensor("token_embd.weight", V * D);
        }

        float[] logits(int[] tokens) {
            int n = tokens.length;
            float[][] x = new float[n][D];
            for (int t = 0; t < n; t++) {
                System.arraycopy(embed, tokens[t] * D, x[t], 0, D);
            }
            for (int il = 0; il < LAYERS; il++) {
                layer(il, x);
            }
            float[] last = rmsZeroCentered(x[n - 1], vector("output_norm.weight", D));
            return matvec(tensor("output.weight", V * D), V, D, last);
        }

        private void layer(int il, float[][] x) {
            LayerWeights w = new LayerWeights(il);
            int n = x.length;
            float[][] q = new float[n][HEADS * HEAD_DIM];
            float[][] kCache = new float[n][KV_HEADS * HEAD_DIM];
            float[][] vCache = new float[n][KV_HEADS * HEAD_DIM];
            float[][] gate = new float[n][HEADS];
            float[][] iq = new float[n][IDX_HEADS * PROXY];
            float[][] ik = new float[n][PROXY];
            float[][] iz = new float[n][PROXY];
            float[][] iw = new float[n][IDX_HEADS];
            for (int t = 0; t < n; t++) {
                float[] normed = rmsZeroCentered(x[t], w.attnNorm);
                projectMain(w, normed, t, q[t], kCache[t], vCache[t], gate[t]);
                projectIndexer(w, normed, t, iq[t], ik[t], iz[t], iw[t]);
            }
            float[][] summary = new float[REGIONS][PROXY];
            float[] att = new float[HEADS * HEAD_DIM];
            for (int t = 0; t < n; t++) {
                if ((t + 1) % REGION == 0) {
                    summary[t / REGION] = summarize(ik, iz, t / REGION * REGION, t);
                }
                int[] sel = select(iq[t], iw[t], summary, t);
                sparseAttention(q[t], kCache, vCache, gate[t], sel, t, att);
                add(x[t], matvec(w.wOut, D, HEADS * HEAD_DIM, att));
                float[] ffnIn = rmsZeroCentered(x[t], w.ffnNorm);
                float[] fg = matvec(w.wFg, FF, D, ffnIn);
                float[] fu = matvec(w.wFu, FF, D, ffnIn);
                for (int i = 0; i < FF; i++) {
                    fg[i] = silu(fg[i]) * fu[i];
                }
                add(x[t], matvec(w.wFd, D, FF, fg));
            }
        }

        /** Q/K/V and the gate, with the per-head zero-centered RMS norms, BF16 casts and partial NeoX rope. */
        private void projectMain(LayerWeights w, float[] in, int pos, float[] q, float[] k, float[] v,
                                 float[] gate) {
            copyInto(matvec(w.wQ, HEADS * HEAD_DIM, D, in), q);
            copyInto(matvec(w.wK, KV_HEADS * HEAD_DIM, D, in), k);
            copyInto(matvec(w.wV, KV_HEADS * HEAD_DIM, D, in), v);
            copyInto(matvec(w.wGate, HEADS, D, in), gate);
            float theta = (float) Math.pow(THETA, -2.0 / ROTARY);
            for (int h = 0; h < HEADS; h++) {
                float[] head = roundBf16(rmsZeroCentered(slice(q, h * HEAD_DIM, HEAD_DIM), w.qNorm));
                neoxRope(head, ROTARY, pos, theta);
                copyInto(roundBf16(head), 0, q, h * HEAD_DIM, HEAD_DIM);
            }
            for (int g = 0; g < KV_HEADS; g++) {
                float[] head = roundBf16(rmsZeroCentered(slice(k, g * HEAD_DIM, HEAD_DIM), w.kNorm));
                neoxRope(head, ROTARY, pos, theta);
                copyInto(roundBf16(head), 0, k, g * HEAD_DIM, HEAD_DIM);
                for (int d = 0; d < HEAD_DIM; d++) {
                    v[g * HEAD_DIM + d] = bf16(v[g * HEAD_DIM + d]);
                }
            }
        }

        /** The indexer projections, norms, BF16/FP8 casts, rope and the prescaled head weights. */
        private void projectIndexer(LayerWeights w, float[] in, int pos, float[] iq, float[] ik, float[] iz,
                                    float[] iw) {
            copyInto(matvec(w.wIq, IDX_HEADS * PROXY, D, in), iq);
            copyInto(layerNorm(matvec(w.wIk, PROXY, D, in), w.ikNormW, w.ikNormB), ik);
            copyInto(roundBf16(matvec(w.wIz, PROXY, D, in)), iz);
            copyInto(matvec(w.wIw, IDX_HEADS, D, in), iw);
            float theta = (float) Math.pow(THETA, -2.0 / IDX_ROPE);
            for (int h = 0; h < IDX_HEADS; h++) {
                float[] head = roundBf16(rmsZeroCentered(slice(iq, h * PROXY, PROXY), w.iqNorm));
                neoxRope(head, IDX_ROPE, pos, theta);
                for (int d = 0; d < PROXY; d++) {
                    iq[h * PROXY + d] = e4m3(bf16(head[d]));
                }
            }
            copyInto(roundBf16(ik), ik);
            neoxRope(ik, IDX_ROPE, pos, theta);
            copyInto(roundBf16(ik), ik);
            for (int h = 0; h < IDX_HEADS; h++) {
                iw[h] *= w.weightScale;
            }
        }

        /** The per-layer weights, read once from the converted GGUF. */
        private final class LayerWeights {
            final float[] attnNorm;
            final float[] qNorm;
            final float[] kNorm;
            final float[] iqNorm;
            final float[] ikNormW;
            final float[] ikNormB;
            final float[] wQ;
            final float[] wK;
            final float[] wV;
            final float[] wOut;
            final float[] wGate;
            final float[] wIq;
            final float[] wIk;
            final float[] wIz;
            final float[] wIw;
            final float[] wFg;
            final float[] wFu;
            final float[] wFd;
            final float[] ffnNorm;
            final float weightScale = (float) (1.0 / Math.sqrt((double) IDX_HEADS / KV_HEADS));

            LayerWeights(int il) {
                String p = "blk." + il + ".";
                attnNorm = vector(p + "attn_norm.weight", D);
                qNorm = vector(p + "attn_q_norm.weight", HEAD_DIM);
                kNorm = vector(p + "attn_k_norm.weight", HEAD_DIM);
                iqNorm = vector(p + "indexer.q_norm.weight", PROXY);
                ikNormW = vector(p + "indexer.k_norm.weight", PROXY);
                ikNormB = vector(p + "indexer.k_norm.bias", PROXY);
                wQ = tensor(p + "attn_q.weight", HEADS * HEAD_DIM * D);
                wK = tensor(p + "attn_k.weight", KV_HEADS * HEAD_DIM * D);
                wV = tensor(p + "attn_v.weight", KV_HEADS * HEAD_DIM * D);
                wOut = tensor(p + "attn_output.weight", D * HEADS * HEAD_DIM);
                wGate = tensor(p + "attn_gate.weight", HEADS * D);
                wIq = tensor(p + "indexer.attn_q_b.weight", IDX_HEADS * PROXY * D);
                wIk = tensor(p + "indexer.attn_k.weight", PROXY * D);
                wIz = tensor(p + "indexer.z.weight", PROXY * D);
                wIw = tensor(p + "indexer.proj.weight", IDX_HEADS * D);
                wFg = tensor(p + "ffn_gate.weight", FF * D);
                wFu = tensor(p + "ffn_up.weight", FF * D);
                wFd = tensor(p + "ffn_down.weight", D * FF);
                ffnNorm = vector(p + "ffn_norm.weight", D);
            }
        }

        /** The closed-region summary of positions {@code [first, last]} from the BF16 ring rows. */
        private float[] summarize(float[][] ik, float[][] iz, int first, int last) {
            float[] num = new float[PROXY];
            float[] den = new float[PROXY];
            float max = Float.NEGATIVE_INFINITY;
            for (int p = first; p <= last; p++) {
                for (int d = 0; d < PROXY; d++) {
                    max = Math.max(max, bf16(iz[p][d]));
                }
            }
            for (int p = first; p <= last; p++) {
                for (int d = 0; d < PROXY; d++) {
                    float weight = (float) Math.exp(bf16(iz[p][d]) - max);
                    num[d] += weight * bf16(ik[p][d]);
                    den[d] += weight;
                }
            }
            float[] out = new float[PROXY];
            for (int d = 0; d < PROXY; d++) {
                out[d] = e4m3(bf16(num[d] / Math.max(den[d], 1e-20f)));
            }
            return out;
        }

        /** The closed-region scores and the vLLM stable top-k plus open-region selection of every KV group. */
        private int[] select(float[] iq, float[] iw, float[][] summary, int pos) {
            int closed = Math.min(pos / REGION, summary.length);
            int k = Math.min(TOP_K, summary.length);
            int[] sel = new int[KV_HEADS * SEL_WIDTH];
            Arrays.fill(sel, -1);
            if (closed == 0) {
                for (int g = 0; g < KV_HEADS; g++) {
                    sel[g * SEL_WIDTH + SEL_WIDTH - 1] = 0;
                }
                return sel;
            }
            int headsPerGroup = IDX_HEADS / KV_HEADS;
            for (int g = 0; g < KV_HEADS; g++) {
                float[] scores = new float[closed];
                for (int r = 0; r < closed; r++) {
                    float[] sums = new float[4];
                    for (int h = 0; h < headsPerGroup; h++) {
                        int head = g * headsPerGroup + h;
                        float dot = 0f;
                        for (int d = 0; d < PROXY; d++) {
                            dot += iq[head * PROXY + d] * summary[r][d];
                        }
                        if (dot > 0f) {
                            sums[h & 3] += dot * iw[head];
                        }
                    }
                    scores[r] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
                }
                int out = g * SEL_WIDTH;
                if (closed <= k) {
                    for (int r = 0; r < closed; r++) {
                        sel[out + k - closed + r] = r;
                    }
                    sel[out + k] = closed;
                    continue;
                }
                boolean[] taken = new boolean[closed];
                for (int i = 0; i < k; i++) {
                    int best = -1;
                    for (int r = 0; r < closed; r++) {
                        if (!taken[r] && (best < 0 || scores[r] > scores[best])) {
                            best = r;
                        }
                    }
                    taken[best] = true;
                }
                int at = 0;
                for (int r = 0; r < closed; r++) {
                    if (taken[r]) {
                        sel[out + at++] = r;
                    }
                }
                sel[out + k] = closed;
            }
            return sel;
        }

        private void sparseAttention(float[] q, float[][] kCache, float[][] vCache, float[] gate, int[] sel,
                                     int pos, float[] out) {
            float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
            int headsPerGroup = HEADS / KV_HEADS;
            for (int h = 0; h < HEADS; h++) {
                int group = h / headsPerGroup;
                float[] logits = new float[SEL_WIDTH * REGION];
                int[] cells = new int[logits.length];
                int count = 0;
                for (int s = 0; s < SEL_WIDTH; s++) {
                    int region = sel[group * SEL_WIDTH + s];
                    if (region < 0) {
                        continue;
                    }
                    int first = region * REGION;
                    int last = Math.min(first + REGION - 1, pos);
                    for (int p = first; p <= last; p++) {
                        float dot = 0f;
                        for (int d = 0; d < HEAD_DIM; d++) {
                            dot += q[h * HEAD_DIM + d] * kCache[p][group * HEAD_DIM + d];
                        }
                        cells[count] = p;
                        logits[count++] = dot * scale * LOG2E;
                    }
                }
                float[] acc = new float[HEAD_DIM];
                if (count == 0) {
                    System.arraycopy(acc, 0, out, h * HEAD_DIM, HEAD_DIM);
                    continue;
                }
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < count; i++) {
                    max = Math.max(max, logits[i]);
                }
                double total = 0.0;
                for (int i = 0; i < count; i++) {
                    logits[i] = (float) Math.pow(2.0, logits[i] - max);
                    total += logits[i];
                }
                for (int i = 0; i < count; i++) {
                    float weight = bf16(logits[i]);
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[d] += weight * vCache[cells[i]][group * HEAD_DIM + d];
                    }
                }
                float inverse = total > 0.0 ? (float) (1.0 / Math.max(total, 1e-30)) : 0f;
                float sigmoid = bf16((float) (1.0 / (1.0 + Math.exp(-(double) gate[h]))));
                for (int d = 0; d < HEAD_DIM; d++) {
                    out[h * HEAD_DIM + d] = bf16(bf16(acc[d] * inverse) * sigmoid);
                }
            }
        }

        private float[] tensor(String name, int n) {
            float[] v;
            try {
                v = parser.readTensorDataAsFloat(parser.getTensor(name));
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
            assertEquals("tensor " + name, n, v.length);
            return v;
        }

        private float[] vector(String name, int n) {
            return tensor(name, n);
        }
    }

    // ================================================================== scalar primitives

    private static float[] matvec(float[] w, int rows, int cols, float[] x) {        float[] y = new float[rows];
        for (int r = 0; r < rows; r++) {
            float sum = 0f;
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                sum += w[base + c] * x[c];
            }
            y[r] = sum;
        }
        return y;
    }

    private static float[] rmsZeroCentered(float[] x, float[] weight) {
        double sum = 0.0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = x[i] * inv * weight[i];
        }
        return out;
    }

    private static float[] layerNorm(float[] x, float[] weight, float[] bias) {
        double sum = 0.0;
        for (float v : x) {
            sum += v;
        }
        double mean = sum / x.length;
        double variance = 0.0;
        for (float v : x) {
            variance += (v - mean) * (v - mean);
        }
        float inv = (float) (1.0 / Math.sqrt(variance / x.length + EPS));
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = (float) ((x[i] - mean) * inv) * weight[i] + bias[i];
        }
        return out;
    }

    private static void neoxRope(float[] v, int ropeDim, int pos, float thetaScale) {
        int half = ropeDim / 2;
        for (int i = 0; i < half; i++) {
            float angle = (float) pos * (float) Math.pow(thetaScale, i);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            float a = v[i];
            float b = v[i + half];
            v[i] = a * cos - b * sin;
            v[i + half] = a * sin + b * cos;
        }
    }

    private static float[] roundBf16(float[] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = bf16(x[i]);
        }
        return out;
    }

    private static float[] slice(float[] x, int off, int len) {
        return Arrays.copyOfRange(x, off, off + len);
    }

    private static void copyInto(float[] src, float[] dst) {
        System.arraycopy(src, 0, dst, 0, dst.length);
    }

    private static void copyInto(float[] src, int srcOff, float[] dst, int dstOff, int len) {
        System.arraycopy(src, srcOff, dst, dstOff, len);
    }

    private static void add(float[] dst, float[] delta) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] += delta[i];
        }
    }

    /** Round to bfloat16, nearest even. */
    private static float bf16(float x) {
        int bits = Float.floatToRawIntBits(x);
        if ((bits & 0x7F800000) == 0x7F800000) {
            return x;
        }
        int high = bits & 0xFFFF0000;
        int low = bits & 0xFFFF;
        if (low > 0x8000 || (low == 0x8000 && (bits & 0x10000) != 0)) {
            high += 0x10000;
        }
        return Float.intBitsToFloat(high);
    }

    /** Round to float8_e4m3fn (finite range, subnormals below 2^-6, ties to even, saturating at 448). */
    private static float e4m3(float x) {
        if (Float.isNaN(x)) {
            return Float.NaN;
        }
        boolean negative = x < 0f || (x == 0f && Float.floatToRawIntBits(x) < 0);
        double a = Math.abs((double) x);
        if (a > 448.0) {
            a = 448.0;
        }
        double value;
        if (a == 0.0) {
            value = 0.0;
        } else if (a < 0.015625) {
            value = Math.rint(a / 0.001953125) * 0.001953125;
        } else {
            int exponent = Math.getExponent(a);
            double step = Math.pow(2.0, exponent - 3);
            value = Math.rint(a / step) * step;
            if (value > 448.0) {
                value = 448.0;
            }
        }
        return (float) (negative ? -value : value);
    }

    private static float silu(float x) {
        return x / (1.0f + (float) Math.exp(-x));
    }

    // ================================================================== the HF fixture

    static SyntheticHf.Builder builder() {
        return builder(CONFIG);
    }

    /** The same checkpoint with layer 0 a sliding block of window 4 and layer 1 full attention. */
    static SyntheticHf.Builder slidingBuilder() {
        return builder(CONFIG.replace("'max_position_embeddings':64,",
                "'max_position_embeddings':64,'layer_types':['sliding_attention','full_attention'],"
                        + "'sliding_window':4,"));
    }

    static SyntheticHf.Builder builder(String config) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int il = 0; il < LAYERS; il++) {
            layer(b, il, 2 + 40 * il);
        }
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 90));
        return b;
    }

    private static void layer(SyntheticHf.Builder b, int il, int seed) {
        String p = "model.layers." + il + ".";
        tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "self_attn.q_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, seed));
        tensor(b, p + "self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, D, matrix(KV_HEADS * HEAD_DIM, D, seed + 1));
        tensor(b, p + "self_attn.v_proj.weight", KV_HEADS * HEAD_DIM, D, matrix(KV_HEADS * HEAD_DIM, D, seed + 2));
        tensor(b, p + "self_attn.o_proj.weight", D, HEADS * HEAD_DIM, matrix(D, HEADS * HEAD_DIM, seed + 3));
        tensor(b, p + "self_attn.q_norm.weight", HEAD_DIM, 1, ramp(HEAD_DIM, seed + 4));
        tensor(b, p + "self_attn.k_norm.weight", HEAD_DIM, 1, ramp(HEAD_DIM, seed + 5));
        tensor(b, p + "self_attn.g_proj.weight", HEADS, D, matrix(HEADS, D, seed + 6));
        tensor(b, p + "self_attn.sparse_indexer_q.weight", IDX_HEADS * PROXY, D,
                matrix(IDX_HEADS * PROXY, D, seed + 7));
        tensor(b, p + "self_attn.sparse_indexer_k.weight", PROXY, D, matrix(PROXY, D, seed + 8));
        tensor(b, p + "self_attn.sparse_indexer_z.weight", PROXY, D, matrix(PROXY, D, seed + 9));
        tensor(b, p + "self_attn.sparse_indexer_w.weight", IDX_HEADS, D, matrix(IDX_HEADS, D, seed + 10));
        tensor(b, p + "self_attn.sparse_indexer_q_norm.weight", PROXY, 1, ramp(PROXY, seed + 11));
        tensor(b, p + "self_attn.sparse_indexer_k_norm.weight", PROXY, 1, ramp(PROXY, seed + 12));
        tensor(b, p + "self_attn.sparse_indexer_k_norm.bias", PROXY, 1, ramp(PROXY, seed + 13));
        tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, seed + 14));
        tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 15));
        tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 16));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 5) % 19 - 9) * 0.025f;
        }
        return x;
    }

    /** A small nonzero ramp, so the norm weights are not all the same. */
    private static float[] ramp(int n, int seed) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = ((i * 3 + seed) % 5 - 2) * 0.05f;
        }
        return x;
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        Arrays.fill(x, 1.0f);
        return x;
    }
}
