package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Step3 text checkpoint loading: a dense lead and MoE layers whose attention RMS-normalizes a shared query
 * projection and widens it per head before RoPE, and whose MoE layers add a plain shared expert to the routed
 * experts. Forward parity against an independent scalar implementation of {@code step3_text.py} on CPU and OpenCL,
 * the converted layout, the export round trip and the named refusals.
 */
public class HfStep3TextTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 8;
    private static final int V = 7;
    private static final int HEADS = 2;
    private static final int HEAD_DIM = 4;
    private static final int SHARE_Q = 4;
    private static final int FF = 6;
    private static final int EXPERTS = 2;
    private static final int MOE_FF = 4;
    private static final int SHARE_FF = 3;
    private static final float EPS = 1e-5f;
    private static final float THETA = 10000f;

    private static final String CONFIG = SyntheticHf.json("{'architectures':['Step3TextForCausalLM'],"
            + "'model_type':'step3_text','hidden_size':" + D + ",'intermediate_size':" + FF
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'num_attention_groups':1"
            + ",'max_position_embedding':16,'rms_norm_eps':" + EPS + ",'vocab_size':" + V
            + ",'moe_intermediate_size':" + MOE_FF + ",'moe_num_experts':" + EXPERTS + ",'moe_top_k':1"
            + ",'share_expert_dim':" + SHARE_FF + ",'share_q_dim':" + SHARE_Q + ",'head_dim':" + HEAD_DIM
            + ",'rope_theta':" + THETA + ",'hidden_act':'silu','tie_word_embeddings':false"
            + ",'norm_expert_weight':false,'moe_layers_enum':[1]}");

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
    public void conversionWritesTheStep3TextLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals("step3text", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("step3text.block_count")).intValue());
            assertEquals(1, ((Number) md.get("step3text.attention.head_count_kv")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("step3text.attention.key_length")).intValue());
            assertEquals(SHARE_Q, ((Number) md.get("step3text.attention.q_lora_rank")).intValue());
            assertEquals(2, ((Number) md.get("step3text.expert_count")).intValue());
            assertEquals(SHARE_FF, ((Number) md.get("step3text.expert_shared_feed_forward_length")).intValue());
            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String expected : new String[]{"token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight",
                    "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                    "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight",
                    "blk.1.ffn_down_exps.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_up_shexp.weight",
                    "blk.1.ffn_down_shexp.weight"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
        }
    }

    @Test
    public void exportedGgufLoadsAsStep3Text() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("step3text.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        assertForward(out, reference(new int[]{2, 3, 4}));
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("Step3TextForCausalLM"));
        assertEquals(RopeType.NEOX, ArchRegistry.traits(Arch.STEP3_TEXT).ropeType());
    }

    @Test
    public void refusesANonSiluActivation() throws Exception {
        String config = CONFIG.replace("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"");
        Path dir = builder(config).write(tmp.newFolder("rejected").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), names(e, "hidden_act"));
    }

    @Test
    public void refusesMoreThanOneKeyValueHead() throws Exception {
        String config = CONFIG.replace("\"num_attention_groups\":1", "\"num_attention_groups\":2");
        Path dir = builder(config).write(tmp.newFolder("groups").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("groups-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), names(e, "num_attention_groups"));
    }

    private static boolean names(Throwable e, String field) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains(field)) {
                return true;
            }
        }
        return false;
    }

    private void assertForward(Path path, float[] expected) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            assertArrayEquals(expected, model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone(), 3e-4f);
        }
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("step3text").toPath());
    }

    private static SyntheticHf.Builder builder() {
        return builder(CONFIG);
    }

    private static SyntheticHf.Builder builder(String config) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            int seed = 2 + 40 * l;
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_proj.weight", SHARE_Q, D, matrix(SHARE_Q, D, seed));
            tensor(b, p + "self_attn.k_proj.weight", HEAD_DIM, D, matrix(HEAD_DIM, D, seed + 1));
            tensor(b, p + "self_attn.v_proj.weight", HEAD_DIM, D, matrix(HEAD_DIM, D, seed + 2));
            tensor(b, p + "self_attn.inter_norm.weight", SHARE_Q, 1, ones(SHARE_Q));
            tensor(b, p + "self_attn.wq.weight", HEADS * HEAD_DIM, SHARE_Q,
                    matrix(HEADS * HEAD_DIM, SHARE_Q, seed + 3));
            tensor(b, p + "self_attn.o_proj.weight", D, HEADS * HEAD_DIM,
                    matrix(D, HEADS * HEAD_DIM, seed + 4));
            if (l == 0) {
                tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, seed + 5));
                tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 6));
                tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 7));
            } else {
                tensor(b, p + "moe.gate.weight", EXPERTS, D, matrix(EXPERTS, D, seed + 8));
                tensor(b, p + "moe.gate_proj.weight", EXPERTS, MOE_FF, D, matrix3(EXPERTS, MOE_FF, D, seed + 9));
                tensor(b, p + "moe.up_proj.weight", EXPERTS, MOE_FF, D, matrix3(EXPERTS, MOE_FF, D, seed + 10));
                tensor(b, p + "moe.down_proj.weight", EXPERTS, D, MOE_FF, matrix3(EXPERTS, D, MOE_FF, seed + 11));
                tensor(b, p + "share_expert.gate_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 12));
                tensor(b, p + "share_expert.up_proj.weight", SHARE_FF, D, matrix(SHARE_FF, D, seed + 13));
                tensor(b, p + "share_expert.down_proj.weight", D, SHARE_FF, matrix(D, SHARE_FF, seed + 14));
            }
        }
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 90));
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

    private static float[] matrix3(int first, int rows, int cols, int seed) {
        return matrix(first * rows, cols, seed);
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

    private static float[] rms(float[] x, float[] w) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale * w[i];
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

    /** GPT-NeoX rotary: pair {@code i} with {@code i + headDim / 2}. */
    private static void rope(float[] v, int headDim, int pos) {
        int half = headDim / 2;
        for (int i = 0; i < half; i++) {
            double theta = pos / Math.pow(THETA, 2.0 * i / headDim);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = v[i];
            float b = v[i + half];
            v[i] = a * c - b * s;
            v[i + half] = a * s + b * c;
        }
    }

    private static float[] attention(float[][] x, int layer) {
        int t = x.length;
        int seed = 2 + 40 * layer;
        float[] wqProj = matrix(SHARE_Q, D, seed);
        float[] wkProj = matrix(HEAD_DIM, D, seed + 1);
        float[] wvProj = matrix(HEAD_DIM, D, seed + 2);
        float[] wq = matrix(HEADS * HEAD_DIM, SHARE_Q, seed + 3);
        float[] wo = matrix(D, HEADS * HEAD_DIM, seed + 4);
        float[][][] q = new float[t][HEADS][HEAD_DIM];
        float[][] k = new float[t][HEAD_DIM];
        float[][] v = new float[t][HEAD_DIM];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qs = linear(n, wqProj, SHARE_Q);
            float[] qn = rms(qs, ones(SHARE_Q));
            float[] qf = linear(qn, wq, HEADS * HEAD_DIM);
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(qf, h * HEAD_DIM, q[p][h], 0, HEAD_DIM);
                rope(q[p][h], HEAD_DIM, p);
            }
            k[p] = linear(n, wkProj, HEAD_DIM);
            v[p] = linear(n, wvProj, HEAD_DIM);
            rope(k[p], HEAD_DIM, p);
        }
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[][] out = new float[t][D];
        for (int p = 0; p < t; p++) {
            float[] heads = new float[HEADS * HEAD_DIM];
            for (int h = 0; h < HEADS; h++) {
                float[] scores = new float[p + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j <= p; j++) {
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += q[p][h][d] * k[j][d];
                    }
                    scores[j] = dot * scale;
                    max = Math.max(max, scores[j]);
                }
                float sum = 0;
                for (int j = 0; j <= p; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int j = 0; j <= p; j++) {
                    for (int d = 0; d < HEAD_DIM; d++) {
                        heads[h * HEAD_DIM + d] += scores[j] / sum * v[j][d];
                    }
                }
            }
            out[p] = linear(heads, wo, D);
        }
        return flatten(out);
    }

    private static float[] flatten(float[][] rows) {
        float[] out = new float[rows.length * rows[0].length];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, out, i * rows[i].length, rows[i].length);
        }
        return out;
    }

    private static float[] denseMlp(float[] x, int seed) {
        float[] gate = linear(x, matrix(FF, D, seed + 5), FF);
        float[] up = linear(x, matrix(FF, D, seed + 6), FF);
        return linear(siluMul(gate, up), matrix(D, FF, seed + 7), D);
    }

    private static float[] shareExpert(float[] x, int seed) {
        float[] gate = linear(x, matrix(SHARE_FF, D, seed + 12), SHARE_FF);
        float[] up = linear(x, matrix(SHARE_FF, D, seed + 13), SHARE_FF);
        return linear(siluMul(gate, up), matrix(D, SHARE_FF, seed + 14), D);
    }

    /** Softmax over all experts, top-k without renormalization, the selected experts summed; plus the shared MLP. */
    private static float[] moe(int layer, float[] x) {
        int seed = 2 + 40 * layer;
        float[] router = linear(x, matrix(EXPERTS, D, seed + 8), EXPERTS);
        float max = Math.max(router[0], router[1]);
        float e0 = (float) Math.exp(router[0] - max);
        float e1 = (float) Math.exp(router[1] - max);
        float[] probs = new float[]{e0 / (e0 + e1), e1 / (e0 + e1)};
        int best = probs[0] >= probs[1] ? 0 : 1;
        float[] gate = expertLinear(x, matrix(EXPERTS * MOE_FF, D, seed + 9), best, MOE_FF, D);
        float[] up = expertLinear(x, matrix(EXPERTS * MOE_FF, D, seed + 10), best, MOE_FF, D);
        float[] down = expertLinear(siluMul(gate, up), matrix(EXPERTS * D, MOE_FF, seed + 11), best, D, MOE_FF);
        float[] out = new float[D];
        for (int i = 0; i < D; i++) {
            out[i] = probs[best] * down[i];
        }
        float[] shared = shareExpert(x, seed);
        for (int i = 0; i < D; i++) {
            out[i] += shared[i];
        }
        return out;
    }

    /** {@code w @ x} with {@code w} the {@code expert}-th of {@code stacked} ({@code [E, rows, cols]}). */
    private static float[] expertLinear(float[] x, float[] stacked, int expert, int rows, int cols) {
        float[] w = new float[rows * cols];
        System.arraycopy(stacked, expert * rows * cols, w, 0, w.length);
        return linear(x, w, rows);
    }

    private static float[] reference(int[] tokens) {
        int t = tokens.length;
        float[][] x = new float[t][D];
        float[] emb = matrix(V, D, 1);
        for (int p = 0; p < t; p++) {
            System.arraycopy(emb, tokens[p] * D, x[p], 0, D);
        }
        for (int l = 0; l < 2; l++) {
            float[] attn = attention(x, l);
            for (int p = 0; p < t; p++) {
                for (int i = 0; i < D; i++) {
                    x[p][i] += attn[p * D + i];
                }
                float[] n = rms(x[p], ones(D));
                float[] branch = l == 0 ? denseMlp(n, 2 + 40 * l) : moe(l, n);
                for (int i = 0; i < D; i++) {
                    x[p][i] += branch[i];
                }
            }
        }
        return linear(rms(x[t - 1], ones(D)), matrix(V, D, 90), V);
    }
}
