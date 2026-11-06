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
import com.eonmux.neuralfree.gguf.memory.KvDType;
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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * AXK-1 checkpoint loading: a DeepSeek-V2-style MLA trunk with a dense lead layer and one sparse layer, the
 * post-MLP norm applied on the sparse layer only, sigmoid grouped routing and the {@code llama_4_scaling} query
 * temperature. The converted layout, the forward (against an independent scalar implementation of vLLM's
 * {@code AXK1ForCausalLM}) on CPU and OpenCL, the export round trip and the named refusals.
 */
public class HfAxk1Test {
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
    private static final int EXPERTS = 2;
    private static final float EPS = 1e-6f;
    private static final float BETA = 0.1f;
    private static final int ORIGINAL_MAX_POSITIONS = 2;

    /** The sparse layer's post-MLP norm weights; deliberately not ones so a misapplied norm shows. */
    private static final float[] POST_MLP_W = {1.2f, 0.8f, 1.1f, 0.9f};
    /** The dense layer's copy, which vLLM never applies; large values would show if it were loaded. */
    private static final float[] DROPPED_POST_MLP_W = {3.0f, 3.0f, 3.0f, 3.0f};

    private static final String CONFIG = SyntheticHf.json("{'architectures':['AXK1ForCausalLM'],"
            + "'model_type':'axk1','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'num_key_value_heads':" + HEADS
            + ",'q_lora_rank':" + Q_LORA + ",'kv_lora_rank':" + KV_LORA + ",'qk_nope_head_dim':" + NOPE
            + ",'qk_rope_head_dim':" + ROPE_DIM + ",'v_head_dim':" + VHEAD + ",'max_position_embeddings':16"
            + ",'rms_norm_eps':" + EPS + ",'rope_theta':10000.0,'vocab_size':" + V
            + ",'first_k_dense_replace':1,'moe_layer_freq':1,'moe_intermediate_size':" + MOE_FFN
            + ",'n_routed_experts':" + EXPERTS + ",'num_experts_per_tok':2"
            + ",'norm_topk_prob':true,'routed_scaling_factor':1.0,'scoring_func':'sigmoid'"
            + ",'topk_method':'noaux_tc','n_group':1,'topk_group':1,'hidden_act':'silu'"
            + ",'tie_word_embeddings':false"
            + ",'llama_4_scaling':{'beta':" + BETA + ",'original_max_position_embeddings':"
            + ORIGINAL_MAX_POSITIONS + "}}");

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(memory()).build();
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
                .memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 3e-4f);
        }
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(4)
                .typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    @Test
    public void conversionWritesTheAxk1Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("axk1", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("axk1.block_count")).intValue());
            assertEquals(1, ((Number) md.get("axk1.leading_dense_block_count")).intValue());
            assertEquals(Q_LORA, ((Number) md.get("axk1.attention.q_lora_rank")).intValue());
            assertEquals(KV_LORA, ((Number) md.get("axk1.attention.kv_lora_rank")).intValue());
            assertEquals(ROPE_DIM, ((Number) md.get("axk1.rope.dimension_count")).intValue());
            assertEquals(EXPERTS, ((Number) md.get("axk1.expert_count")).intValue());
            assertEquals(2, ((Number) md.get("axk1.expert_used_count")).intValue());
            assertEquals(0, ((Number) md.get("axk1.expert_shared_count")).intValue());
            assertEquals(1, ((Number) md.get("axk1.expert_group_count")).intValue());
            assertEquals(1, ((Number) md.get("axk1.expert_group_used_count")).intValue());
            assertEquals(2, ((Number) md.get("axk1.expert_gating_func")).intValue());
            assertEquals(Boolean.TRUE, md.get("axk1.expert_weights_norm"));
            assertEquals((double) BETA, ((Number) md.get("axk1.attention.temperature_scale")).doubleValue(), 0.0);
            assertEquals(ORIGINAL_MAX_POSITIONS,
                    ((Number) md.get("axk1.attention.temperature_length")).intValue());

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String expected : new String[]{"token_embd.weight", "output_norm.weight", "output.weight",
                    "blk.0.attn_norm.weight", "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight",
                    "blk.0.attn_q_b.weight", "blk.0.attn_kv_a_mqa.weight", "blk.0.attn_kv_a_norm.weight",
                    "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight", "blk.0.attn_output.weight",
                    "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                    "blk.1.attn_norm.weight", "blk.1.attn_q_a.weight", "blk.1.attn_q_b.weight",
                    "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_k_b.weight", "blk.1.attn_v_b.weight",
                    "blk.1.attn_output.weight", "blk.1.ffn_norm.weight",
                    "blk.1.post_ffw_norm.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.exp_probs_b.bias",
                    "blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight"}) {
                assertTrue("missing " + expected, tensors.contains(expected));
            }
            // vLLM applies post_mlp_layernorm only on the sparse layer, so the dense layer's copy is dropped.
            assertFalse("the dense layer must have no post-FFN norm",
                    tensors.contains("blk.0.post_ffw_norm.weight"));
        }
    }

    @Test
    public void exportedGgufLoadsAndRuns() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("axk1.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(memory()).build();
        try (GGUFModel model = new GGUFModel(out.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 3e-4f);
        }
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("AXK1ForCausalLM"));
        assertEquals(Arch.AXK1, Arch.fromGgufName("axk1"));
    }

    @Test
    public void refusesInterleavedMoeLayers() throws Exception {
        assertRejected("'mlp_layer_types':['dense','sparse','dense']", "mlp_layer_types");
    }

    @Test
    public void refusesMoeLayerFrequency() throws Exception {
        assertRejected("'moe_layer_freq':2", "moe_layer_freq");
    }

    @Test
    public void refusesThePlainMultiHeadVariant() throws Exception {
        assertRejected("'qk_nope_head_dim':0", "qk_nope_head_dim");
    }

    @Test
    public void refusesAnotherActivation() throws Exception {
        assertRejected("'hidden_act':'gelu'", "hidden_act");
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
        return builder().write(tmp.newFolder("axk1").toPath());
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
            // AXK-1 constructs post_mlp_layernorm on every layer; only the sparse layer applies it.
            tensor(b, p + "post_mlp_layernorm.weight", D, 1, l == 0 ? DROPPED_POST_MLP_W : POST_MLP_W);
        }
        // dense lead layer
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        // sparse layer: routed experts, no shared expert
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 20));
        b.tensor("model.layers.1.mlp.gate.e_score_correction_bias", StDType.F32, new long[]{EXPERTS},
                SyntheticHf.f32(new float[]{0.0f, 0.0f}));
        for (int e = 0; e < EXPERTS; e++) {
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 21 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 22 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FFN,
                    matrix(D, MOE_FFN, 23 + 3 * e));
        }
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

    /** vLLM's {@code _get_llama_4_scaling}: {@code 1 + beta * log(1 + floor(pos / original_max_positions))}. */
    private static float temperature(int pos) {
        return 1.0f + BETA * (float) Math.log(1.0 + Math.floor((double) pos / ORIGINAL_MAX_POSITIONS));
    }

    private static float silu(float v) {
        return v / (1.0f + (float) Math.exp(-v));
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            y[i] = silu(gate[i]) * up[i];
        }
        return y;
    }

    private static float[] ffn(float[] x, int seedBase, int inter) {
        float[] gate = linear(x, matrix(inter, D, seedBase), inter);
        float[] up = linear(x, matrix(inter, D, seedBase + 1), inter);
        return linear(siluMul(gate, up), matrix(D, inter, seedBase + 2), D);
    }

    /** One MLA block: q_nope/q_rope, the shared rotary key, per-head k_nope/v, the query temperature and o_proj. */
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
            float temp = temperature(p);
            for (int h = 0; h < HEADS; h++) {
                float[] qh = new float[QK];
                System.arraycopy(qb, h * QK, qh, 0, QK);
                float[] qn = new float[NOPE];
                System.arraycopy(qh, 0, qn, 0, NOPE);
                float[] qp = new float[ROPE_DIM];
                System.arraycopy(qh, NOPE, qp, 0, ROPE_DIM);
                rotate(qp, p, ROPE_DIM);
                for (int i = 0; i < NOPE; i++) {
                    q[p][h][i] = qn[i] * temp;
                }
                for (int i = 0; i < ROPE_DIM; i++) {
                    q[p][h][NOPE + i] = qp[i] * temp;
                }
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
        // layer 0: dense MLP, no post-MLP norm
        float[][] attn = attention(x, 0);
        for (int t = 0; t < x.length; t++) {
            add(x[t], attn[t]);
            float[] n = rms(x[t], ones(D));
            add(x[t], ffn(n, 7, FFN));
        }
        // layer 1: grouped sigmoid routing (one group, top-2 of two experts, renormalized)
        attn = attention(x, 1);
        float[] router = matrix(EXPERTS, D, 20);
        for (int t = 0; t < x.length; t++) {
            add(x[t], attn[t]);
            float[] n = rms(x[t], ones(D));
            float[] logits = linear(n, router, EXPERTS);
            float s0 = 1.0f / (1.0f + (float) Math.exp(-logits[0]));
            float s1 = 1.0f / (1.0f + (float) Math.exp(-logits[1]));
            float z = s0 + s1;
            float[] e0 = ffn(n, 21, MOE_FFN);
            float[] e1 = ffn(n, 24, MOE_FFN);
            float[] moe = new float[D];
            for (int i = 0; i < D; i++) {
                moe[i] = (s0 / z) * e0[i] + (s1 / z) * e1[i];
            }
            add(x[t], rms(moe, POST_MLP_W));
        }
        float[] last = rms(x[x.length - 1], ones(D));
        return linear(last, matrix(V, D, 40), V);
    }

    private static void add(float[] dst, float[] src) {
        for (int i = 0; i < dst.length; i++) {
            dst[i] += src[i];
        }
    }
}
