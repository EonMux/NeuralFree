package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Step3.5 (Step3p5ForCausalLM) logits parity and export round trip: per-layer head counts (a sliding layer of one
 * head and a full layer of two), a sliding-window layer, the half-width rotary of the full layers, and the SwiGLU
 * limits on the routed and shared experts.
 */
public class HfStep35Test {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 8;
    private static final int V = 8;
    private static final int HEAD_DIM = 4;
    private static final int SLIDING_HEAD = 1;
    private static final int FULL_HEADS = 2;
    private static final int FF = 8;
    private static final int EXPERTS = 2;
    private static final int MOE_FF = 4;
    private static final int SHARE_FF = 8;
    private static final int WINDOW = 4;
    private static final float EPS = 1e-5f;
    private static final float THETA = 10000f;
    private static final float CLAMP = 0.5f;
    /** Step3.5 stores its RMS weights minus one; the converter adds it back (step3.py:283). */
    private static final float NORM_WEIGHT = 2f;

    private static final String CONFIG = SyntheticHf.json("{'architectures':['Step3p5ForCausalLM'],"
            + "'model_type':'step3p5','hidden_size':" + D + ",'head_dim':" + HEAD_DIM
            + ",'layer_types':['sliding_attention','full_attention'],"
            + "'attention_other_setting':{'num_attention_heads':" + SLIDING_HEAD + ",'num_attention_groups':1},"
            + "'sliding_window':" + WINDOW + ",'moe_num_experts':" + EXPERTS + ",'moe_top_k':1,"
            + "'moe_intermediate_size':" + MOE_FF + ",'share_expert_dim':" + SHARE_FF
            + ",'swiglu_limits':[0.0,0.5],'swiglu_limits_shared':[0.0,0.5],'moe_layers_enum':'1'"
            + ",'moe_every_n_layer':2,'num_hidden_layers':2,'num_attention_heads':" + FULL_HEADS
            + ",'num_attention_groups':1,'intermediate_size':" + FF + ",'max_position_embeddings':64"
            + ",'rms_norm_eps':" + EPS + ",'rope_theta':" + THETA + ",'vocab_size':" + V
            + ",'tie_word_embeddings':false}");

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        assertForward(checkpoint(), reference(new int[]{2, 3, 4}));
    }

    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(checkpoint().toString(), cfg)) {
            assertArrayEquals(reference(new int[]{2, 3, 4}),
                    model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone(), 3e-4f);
        }
    }

    @Test
    public void conversionWritesThePerLayerGeometry() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals("step35", md.get("general.architecture"));
            @SuppressWarnings("unchecked")
            java.util.List<Object> heads = (java.util.List<Object>) md.get("step35.attention.head_count");
            assertEquals(SLIDING_HEAD, ((Number) heads.get(0)).intValue());
            assertEquals(FULL_HEADS, ((Number) heads.get(1)).intValue());
            @SuppressWarnings("unchecked")
            java.util.List<Object> swa = (java.util.List<Object>) md.get("step35.attention.sliding_window_pattern");
            assertEquals(Boolean.TRUE, swa.get(0));
            assertEquals(Boolean.FALSE, swa.get(1));
            assertEquals(WINDOW, ((Number) md.get("step35.attention.sliding_window")).intValue());
            @SuppressWarnings("unchecked")
            java.util.List<Object> kvs = (java.util.List<Object>) md.get("step35.attention.head_count_kv");
            assertEquals(1, ((Number) kvs.get(0)).intValue());
            assertEquals(1, ((Number) kvs.get(1)).intValue());
            @SuppressWarnings("unchecked")
            java.util.List<Object> clampExp = (java.util.List<Object>) md.get("step35.swiglu_clamp_exp");
            assertEquals(0.0, ((Number) clampExp.get(0)).doubleValue(), 0.0);
            assertEquals(CLAMP, ((Number) clampExp.get(1)).doubleValue(), 1e-6);
            @SuppressWarnings("unchecked")
            java.util.List<Object> clampShexp = (java.util.List<Object>) md.get("step35.swiglu_clamp_shexp");
            assertEquals(0.0, ((Number) clampShexp.get(0)).doubleValue(), 0.0);
            assertEquals(CLAMP, ((Number) clampShexp.get(1)).doubleValue(), 1e-6);
            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String expected : new String[]{"blk.0.attn_q.weight", "blk.0.ffn_up.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_gate_shexp.weight"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
        }
    }

    @Test
    public void exportedGgufLoadsAsStep35() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step35.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        assertForward(out, reference(new int[]{2, 3, 4}));
    }

    private void assertForward(Path path, float[] expected) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            assertArrayEquals(expected, model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone(), 3e-4f);
        }
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("step35").toPath());
    }

    private static SyntheticHf.Builder builder() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(CONFIG);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            int seed = 2 + 40 * l;
            int heads = l == 0 ? SLIDING_HEAD : FULL_HEADS;
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_proj.weight", heads * HEAD_DIM, D,
                    matrix(heads * HEAD_DIM, D, seed));
            tensor(b, p + "self_attn.k_proj.weight", HEAD_DIM, D, matrix(HEAD_DIM, D, seed + 1));
            tensor(b, p + "self_attn.v_proj.weight", HEAD_DIM, D, matrix(HEAD_DIM, D, seed + 2));
            tensor(b, p + "self_attn.o_proj.weight", D, heads * HEAD_DIM,
                    matrix(D, heads * HEAD_DIM, seed + 3));
            if (l == 0) {
                tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, seed + 4));
                tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 5));
                tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 6));
            } else {
                tensor(b, p + "moe.gate.weight", EXPERTS, D, matrix(EXPERTS, D, seed + 7));
                tensor(b, p + "moe.gate_proj.weight", EXPERTS, MOE_FF, D, matrix(EXPERTS * MOE_FF, D, seed + 8));
                tensor(b, p + "moe.up_proj.weight", EXPERTS, MOE_FF, D, matrix(EXPERTS * MOE_FF, D, seed + 9));
                tensor(b, p + "moe.down_proj.weight", EXPERTS, D, MOE_FF, matrix(EXPERTS * D, MOE_FF, seed + 10));
                tensor(b, p + "mlp.shared_expert.gate_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 11));
                tensor(b, p + "mlp.shared_expert.up_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 12));
                tensor(b, p + "mlp.shared_expert.down_proj.weight", D, SHARE_FF, matrix(D, SHARE_FF, seed + 13));
            }
        }
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 95));
        return b;
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int experts, int rows, int cols, float[] data) {
        b.tensor(name, StDType.F32, new long[]{experts, rows, cols}, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 5) % 19 - 9) * 0.025f;
        }
        return x;
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1.0f);
        return x;
    }

    // ------------------------------------------------------------------ scalar reference

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
        }
        return y;
    }

    private static float[] rms(float[] x) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + EPS)) * NORM_WEIGHT;
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale;
        }
        return y;
    }

    /** GPT-NeoX rotary over the first {@code nRot} dims. */
    private static void rope(float[] v, int nRot, int pos) {
        int half = nRot / 2;
        for (int i = 0; i < half; i++) {
            double theta = pos / Math.pow(THETA, 2.0 * i / nRot);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = v[i];
            float b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    /** {@code clamp(silu(g), -inf, limit) * clamp(u, -limit, limit)} (llama-graph.cpp:2008-2018). */
    private static float[] limitedGlu(float[] gate, float[] up, float limit) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            float a = Math.min(gate[i] / (1.0f + (float) Math.exp(-gate[i])), limit);
            float u = Math.max(-limit, Math.min(limit, up[i]));
            y[i] = a * u;
        }
        return y;
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            y[i] = gate[i] / (1.0f + (float) Math.exp(-gate[i])) * up[i];
        }
        return y;
    }

    /** nRot of layer 0 (sliding) is the full head; the full layer rotates half of it. */
    private static int nRot(int layer) {
        return layer == 0 ? HEAD_DIM : HEAD_DIM / 2;
    }

    private static float[][] attention(float[][] x, int layer) {
        int t = x.length;
        int seed = 2 + 40 * layer;
        int heads = layer == 0 ? SLIDING_HEAD : FULL_HEADS;
        float[] wq = matrix(heads * HEAD_DIM, D, seed);
        float[] wk = matrix(HEAD_DIM, D, seed + 1);
        float[] wv = matrix(HEAD_DIM, D, seed + 2);
        float[] wo = matrix(D, heads * HEAD_DIM, seed + 3);
        float[][][] q = new float[t][heads][HEAD_DIM];
        float[][] k = new float[t][HEAD_DIM];
        float[][] v = new float[t][HEAD_DIM];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p]);
            float[] qf = linear(n, wq, heads * HEAD_DIM);
            float[] kf = linear(n, wk, HEAD_DIM);
            float[] vf = linear(n, wv, HEAD_DIM);
            for (int h = 0; h < heads; h++) {
                System.arraycopy(qf, h * HEAD_DIM, q[p][h], 0, HEAD_DIM);
                rope(q[p][h], nRot(layer), p);
            }
            System.arraycopy(kf, 0, k[p], 0, HEAD_DIM);
            System.arraycopy(vf, 0, v[p], 0, HEAD_DIM);
            rope(k[p], nRot(layer), p);
        }
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[][] out = new float[t][D];
        for (int p = 0; p < t; p++) {
            float[] acc = new float[heads * HEAD_DIM];
            int first = layer == 0 ? Math.max(0, p - WINDOW + 1) : 0;
            for (int h = 0; h < heads; h++) {
                float[] scores = new float[p - first + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = first; j <= p; j++) {
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += q[p][h][d] * k[j][d];
                    }
                    scores[j - first] = dot * scale;
                    max = Math.max(max, scores[j - first]);
                }
                float sum = 0;
                for (int j = first; j <= p; j++) {
                    scores[j - first] = (float) Math.exp(scores[j - first] - max);
                    sum += scores[j - first];
                }
                for (int j = first; j <= p; j++) {
                    for (int d = 0; d < HEAD_DIM; d++) {
                        acc[h * HEAD_DIM + d] += scores[j - first] / sum * v[j][d];
                    }
                }
            }
            out[p] = linear(acc, wo, D);
        }
        return out;
    }

    private static float[] denseMlp(float[] x, int seed) {
        float[] gate = linear(x, matrix(FF, D, seed + 4), FF);
        float[] up = linear(x, matrix(FF, D, seed + 5), FF);
        return linear(siluMul(gate, up), matrix(D, FF, seed + 6), D);
    }

    private static float[] expertLinear(float[] x, float[] stacked, int expert, int rows, int cols) {
        float[] w = new float[rows * cols];
        System.arraycopy(stacked, expert * rows * cols, w, 0, w.length);
        return linear(x, w, rows);
    }

    /** Sigmoid scoring, top-1 without renormalization, the clamped expert plus the clamped shared expert. */
    private static float[] moe(float[] x, int seed) {
        float[] logits = linear(x, matrix(EXPERTS, D, seed + 7), EXPERTS);
        float[] probs = new float[EXPERTS];
        for (int e = 0; e < EXPERTS; e++) {
            probs[e] = (float) (1.0 / (1.0 + Math.exp(-logits[e])));
        }
        int best = probs[0] >= probs[1] ? 0 : 1;
        float[] gate = expertLinear(x, matrix(EXPERTS * MOE_FF, D, seed + 8), best, MOE_FF, D);
        float[] up = expertLinear(x, matrix(EXPERTS * MOE_FF, D, seed + 9), best, MOE_FF, D);
        float[] down = expertLinear(limitedGlu(gate, up, CLAMP), matrix(EXPERTS * D, MOE_FF, seed + 10), best, D,
                MOE_FF);
        float[] sharedGate = linear(x, matrix(SHARE_FF, D, seed + 11), SHARE_FF);
        float[] sharedUp = linear(x, matrix(SHARE_FF, D, seed + 12), SHARE_FF);
        float[] shared = linear(limitedGlu(sharedGate, sharedUp, CLAMP), matrix(D, SHARE_FF, seed + 13), D);
        for (int i = 0; i < D; i++) {
            down[i] = probs[best] * down[i] + shared[i];
        }
        return down;
    }

    private static float[] reference(int[] tokens) {
        int t = tokens.length;
        float[][] x = new float[t][D];
        float[] emb = matrix(V, D, 1);
        for (int p = 0; p < t; p++) {
            System.arraycopy(emb, tokens[p] * D, x[p], 0, D);
        }
        for (int l = 0; l < 2; l++) {
            int seed = 2 + 40 * l;
            float[][] attn = attention(x, l);
            for (int p = 0; p < t; p++) {
                for (int i = 0; i < D; i++) {
                    x[p][i] += attn[p][i];
                }
                float[] n = rms(x[p]);
                float[] branch = l == 0 ? denseMlp(n, seed) : moe(n, seed);
                for (int i = 0; i < D; i++) {
                    x[p][i] += branch[i];
                }
            }
        }
        return linear(rms(x[t - 1]), matrix(V, D, 95), V);
    }
}
