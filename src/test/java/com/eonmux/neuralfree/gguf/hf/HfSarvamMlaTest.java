package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Sarvam MLA checkpoint loading: a DeepSeek-V2-style MLA trunk with one dense lead layer and one MoE layer, a
 * zero-mean router bias, sigmoid routing and a shared expert. The converted layout, the forward (against an
 * independent scalar implementation of vLLM's {@code SarvamMLAForCausalLM}) on CPU and OpenCL, the export round trip
 * and the named refusals.
 */
public class HfSarvamMlaTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 4;
    private static final int V = 7;
    private static final int HEADS = 2;
    private static final int NOPE = 2;
    private static final int ROPE_DIM = 2;
    private static final int QK = NOPE + ROPE_DIM;
    private static final int VHEAD = 1;
    private static final int KV_LORA = 4;
    private static final int Q_LORA = 2;
    private static final int FFN = 4;
    private static final int MOE_FFN = 2;
    private static final int SHEXP_FFN = 3;
    private static final int EXPERTS = 2;
    private static final float EPS = 1e-6f;

    private static final String CONFIG = SyntheticHf.json("{'architectures':['SarvamMLAForCausalLM'],"
            + "'model_type':'sarvam_moe','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'num_key_value_heads':" + HEADS
            + ",'q_lora_rank':" + Q_LORA + ",'kv_lora_rank':" + KV_LORA + ",'qk_nope_head_dim':" + NOPE
            + ",'qk_rope_head_dim':" + ROPE_DIM + ",'v_head_dim':" + VHEAD + ",'max_position_embeddings':16"
            + ",'rms_norm_eps':" + EPS + ",'rope_theta':10000.0,'vocab_size':" + V
            + ",'first_k_dense_replace':1,'moe_intermediate_size':" + MOE_FFN + ",'moe_shared_expert_intermediate_size':"
            + SHEXP_FFN + ",'num_experts':" + EXPERTS + ",'num_experts_per_tok':1,'norm_topk_prob':true"
            + ",'routed_scaling_factor':1.0,'score_function':'sigmoid','moe_router_enable_expert_bias':true"
            + ",'hidden_act':'silu','tie_word_embeddings':false}");

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 3e-4f);
        }
    }

    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 3e-4f);
        }
    }

    @Test
    public void conversionWritesTheDeepseekMlaLayoutAndCentersTheRouterBias() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("deepseek2", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("deepseek2.block_count")).intValue());
            assertEquals(1, ((Number) md.get("deepseek2.attention.head_count_kv")).intValue());
            assertEquals(Q_LORA, ((Number) md.get("deepseek2.attention.q_lora_rank")).intValue());
            assertEquals(KV_LORA, ((Number) md.get("deepseek2.attention.kv_lora_rank")).intValue());
            assertEquals(ROPE_DIM, ((Number) md.get("deepseek2.rope.dimension_count")).intValue());
            assertEquals(1, ((Number) md.get("deepseek2.leading_dense_block_count")).intValue());
            assertEquals(EXPERTS, ((Number) md.get("deepseek2.expert_count")).intValue());
            assertEquals(1, ((Number) md.get("deepseek2.expert_shared_count")).intValue());
            // the checkpoint omits n_group / topk_group; the class defaults them to one group.
            assertEquals(1, ((Number) md.get("deepseek2.expert_group_count")).intValue());
            assertEquals(1, ((Number) md.get("deepseek2.expert_group_used_count")).intValue());
            // sigmoid routing from the score_function key, top-k renormalization on.
            assertEquals(2, ((Number) md.get("deepseek2.expert_gating_func")).intValue());
            assertEquals(Boolean.TRUE, md.get("deepseek2.expert_weights_norm"));

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String expected : new String[]{"token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_norm.weight", "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight",
                    "blk.0.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight",
                    "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.0.attn_output.weight",
                    "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias", "blk.1.ffn_gate_exps.weight",
                    "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight", "blk.1.ffn_gate_shexp.weight",
                    "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
        }
    }

    @Test
    public void exportedGgufLoadsAndRuns() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("sarvam.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(out.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 3e-4f);
        }
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("SarvamMLAForCausalLM"));
    }

    @Test
    public void refusesMoeLayerFrequency() throws Exception {
        assertRejected("\"moe_layer_freq\":2", "moe_layer_freq");
    }

    @Test
    public void refusesNoSharedExpert() throws Exception {
        assertRejected("\"num_shared_experts\":0", "num_shared_experts");
    }

    private void assertRejected(String extra, String field) throws Exception {
        String config = CONFIG.substring(0, CONFIG.length() - 1) + "," + extra + "}";
        SyntheticHf.Builder b = builder(config);
        Path dir = b.write(tmp.newFolder("rejected-" + field).toPath());
        Exception e = assertThrows(Exception.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.toString(), names(e, field));
    }

    /** True when {@code e} or a cause names {@code field}. */
    private static boolean names(Throwable e, String field) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t.getMessage() != null && t.getMessage().contains(field)) {
                return true;
            }
        }
        return false;
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("sarvam").toPath());
    }

    private static SyntheticHf.Builder builder() {
        return builder(CONFIG);
    }

    private static SyntheticHf.Builder builder(String config) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_a_proj.weight", Q_LORA, D, matrix(Q_LORA, D, 2 + 10 * l));
            tensor(b, p + "self_attn.q_a_layernorm.weight", Q_LORA, 1, ones(Q_LORA));
            tensor(b, p + "self_attn.q_b_proj.weight", HEADS * QK, Q_LORA, matrix(HEADS * QK, Q_LORA, 3 + 10 * l));
            tensor(b, p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + ROPE_DIM, D,
                    matrix(KV_LORA + ROPE_DIM, D, 4 + 10 * l));
            tensor(b, p + "self_attn.kv_a_layernorm.weight", KV_LORA, 1, ones(KV_LORA));
            tensor(b, p + "self_attn.kv_b_proj.weight", HEADS * (NOPE + VHEAD), KV_LORA,
                    matrix(HEADS * (NOPE + VHEAD), KV_LORA, 5 + 10 * l));
            tensor(b, p + "self_attn.o_proj.weight", D, HEADS * VHEAD, matrix(D, HEADS * VHEAD, 6 + 10 * l));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        }
        // dense lead layer
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        // MoE layer
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 20));
        b.tensor("model.layers.1.mlp.gate.e_score_correction_bias", StDType.F32, new long[]{EXPERTS},
                SyntheticHf.f32(new float[]{0.5f, 0.5f}));
        for (int e = 0; e < EXPERTS; e++) {
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 21 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 22 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FFN,
                    matrix(D, MOE_FFN, 23 + 3 * e));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", SHEXP_FFN, D, matrix(SHEXP_FFN, D, 30));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", SHEXP_FFN, D, matrix(SHEXP_FFN, D, 31));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, SHEXP_FFN, matrix(D, SHEXP_FFN, 32));
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 40));
        return b;
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

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1.0f);
        return x;
    }

    // ------------------------------------------------------------------ scalar reference (vLLM forward)

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            float s = 0;
            for (int j = 0; j < x.length; j++) {
                s += x[j] * w[i * x.length + j];
            }
            y[i] = s;
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

    private static void rotate(float[] x, int pos, int nRot) {
        for (int i = 0; i < nRot / 2; i++) {
            double theta = pos / Math.pow(10000.0, 2.0 * i / nRot);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = x[2 * i];
            float b = x[2 * i + 1];
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = b * c + a * s;
        }
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            y[i] = gate[i] / (1.0f + (float) Math.exp(-gate[i])) * up[i];
        }
        return y;
    }

    private static float[] ffn(float[] x, int seedBase, int inter) {
        float[] gate = linear(x, matrix(inter, D, seedBase), inter);
        float[] up = linear(x, matrix(inter, D, seedBase + 1), inter);
        return linear(siluMul(gate, up), matrix(D, inter, seedBase + 2), D);
    }

    /** One MLA block: q_nope/q_rope, the shared rotary key, per-head k_nope/v, causal attention and o_proj. */
    private static float[][] attention(float[][] x, int layer) {
        int t = x.length;
        float[][][] q = new float[t][HEADS][QK];
        float[][][] k = new float[t][HEADS][QK];
        float[][][] v = new float[t][HEADS][VHEAD];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qa = linear(n, matrix(Q_LORA, D, 2 + 10 * layer), Q_LORA);
            float[] qb = linear(rms(qa, ones(Q_LORA)), matrix(HEADS * QK, Q_LORA, 3 + 10 * layer), HEADS * QK);
            float[] kva = linear(n, matrix(KV_LORA + ROPE_DIM, D, 4 + 10 * layer), KV_LORA + ROPE_DIM);
            float[] latent = new float[KV_LORA];
            System.arraycopy(kva, 0, latent, 0, KV_LORA);
            float[] pe = new float[ROPE_DIM];
            System.arraycopy(kva, KV_LORA, pe, 0, ROPE_DIM);
            float[] kvb = linear(rms(latent, ones(KV_LORA)),
                    matrix(HEADS * (NOPE + VHEAD), KV_LORA, 5 + 10 * layer), HEADS * (NOPE + VHEAD));
            rotate(pe, p, ROPE_DIM);
            for (int h = 0; h < HEADS; h++) {
                float[] qh = new float[QK];
                System.arraycopy(qb, h * QK, qh, 0, QK);
                float[] qn = new float[NOPE];
                System.arraycopy(qh, 0, qn, 0, NOPE);
                float[] qp = new float[ROPE_DIM];
                System.arraycopy(qh, NOPE, qp, 0, ROPE_DIM);
                rotate(qp, p, ROPE_DIM);
                System.arraycopy(qn, 0, q[p][h], 0, NOPE);
                System.arraycopy(qp, 0, q[p][h], NOPE, ROPE_DIM);
                System.arraycopy(kvb, h * (NOPE + VHEAD), k[p][h], 0, NOPE);
                System.arraycopy(pe, 0, k[p][h], NOPE, ROPE_DIM);
                v[p][h][0] = kvb[h * (NOPE + VHEAD) + NOPE];
            }
        }
        float[][] out = new float[t][D];
        float scale = (float) (1.0 / Math.sqrt(QK));
        for (int p = 0; p < t; p++) {
            float[] heads = new float[HEADS * VHEAD];
            for (int h = 0; h < HEADS; h++) {
                float[] scores = new float[p + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j <= p; j++) {
                    scores[j] = dot(q[p][h], k[j][h]) * scale;
                    max = Math.max(max, scores[j]);
                }
                float sum = 0;
                for (int j = 0; j <= p; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int j = 0; j <= p; j++) {
                    heads[h * VHEAD] += scores[j] / sum * v[j][h][0];
                }
            }
            out[p] = linear(heads, matrix(D, HEADS * VHEAD, 6 + 10 * layer), D);
        }
        return out;
    }

    private static float[] reference(int[] tokens) {
        float[][] x = new float[tokens.length][D];
        float[] emb = matrix(V, D, 1);
        for (int t = 0; t < tokens.length; t++) {
            System.arraycopy(emb, tokens[t] * D, x[t], 0, D);
        }
        // layer 0: dense
        float[][] attn = attention(x, 0);
        for (int t = 0; t < x.length; t++) {
            for (int i = 0; i < D; i++) {
                x[t][i] += attn[t][i];
            }
            float[] n = rms(x[t], ones(D));
            float[] ff = ffn(n, 7, FFN);
            for (int i = 0; i < D; i++) {
                x[t][i] += ff[i];
            }
        }
        // layer 1: MoE with sigmoid routing, top-1 and a shared expert
        attn = attention(x, 1);
        float[] router = matrix(EXPERTS, D, 20);
        for (int t = 0; t < x.length; t++) {
            for (int i = 0; i < D; i++) {
                x[t][i] += attn[t][i];
            }
            float[] n = rms(x[t], ones(D));
            float[] logits = linear(n, router, EXPERTS);
            int best = logits[0] > logits[1] ? 0 : 1;
            float[] expert = ffn(n, 21 + 3 * best, MOE_FFN);
            float[] shared = ffn(n, 30, SHEXP_FFN);
            for (int i = 0; i < D; i++) {
                x[t][i] += expert[i] + shared[i];
            }
        }
        float[] last = rms(x[x.length - 1], ones(D));
        return linear(last, matrix(V, D, 40), V);
    }
}
