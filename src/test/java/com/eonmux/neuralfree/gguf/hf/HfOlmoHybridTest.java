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
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** OLMo Hybrid checkpoint loading and causal inference against a scalar reference of vLLM's forward. */
public class HfOlmoHybridTest {

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 4, V = 7, FF = 4;
    private static final int HEADS = 2, KV_HEADS = 1, HEAD_DIM = 2;
    private static final int N_K = 1, N_V = 2, HEAD_K = 2, HEAD_V = 2, CONV_K = 2;
    private static final float EPS = 1e-6f;
    /** HF's FLA gated output norm uses 1e-5 (OlmoHybridGatedDeltaNet). */
    private static final float DELTA_EPS = 1e-5f;
    /** FLA's q/k L2 norm epsilon. */
    private static final float L2_EPS = 1e-6f;
    private static final int[] TOKENS = {2, 3, 4};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, true), actual, 2e-4f);
        }
    }

    /** The same forward through OpenCL: the GDN, the GQA-expanded delta rule and the OLMo2 post norms. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint(Collections.<String, String>emptyMap());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, true), actual, 2e-4f);
        }
    }

    /** The recurrent state carries across steps: chunked prefill plus decode matches one prefill. */
    @Test
    public void incrementalDecodeMatchesPrefill() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(new int[]{TOKENS[0], TOKENS[1]}, 0);
            float[] actual = model.forwardBackend().decode(TOKENS[2], 2).clone();
            assertArrayEquals(reference(TOKENS, true), actual, 2e-4f);
        }
    }

    /** A checkpoint without {@code rope_parameters} runs NoPE attention, as the released files do. */
    @Test
    public void noRopeCheckpointRunsWithoutRotation() throws Exception {
        Map<String, String> overrides = new LinkedHashMap<>();
        overrides.put("rope_parameters", "null");
        overrides.put("rope_theta", "null");
        Path path = checkpoint(overrides);
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, false), actual, 2e-4f);
        }
    }

    /** OLMo Hybrid is a supported generative hybrid architecture. */
    @Test
    public void registeredAsASupportedHybrid() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("OlmoHybridForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.OLMO_HYBRID);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(MemoryKind.HYBRID, traits.memoryKind());
        assertEquals(PostNormRole.NONE, traits.postNormRole());
    }

    /** The exported GGUF file (NeuralFree's {@code olmo-hybrid} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsOlmoHybrid() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("olmo-hybrid.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, true), actual, 2e-4f);
        }
    }

    /** The converter writes the OLMo Hybrid layout, its Gated DeltaNet metadata and the per-layer mixer kinds. */
    @Test
    public void conversionWritesTheOlmoHybridLayout() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("olmo-hybrid", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("olmo-hybrid.block_count")).intValue());
            assertEquals(HEAD_K, ((Number) md.get("olmo-hybrid.ssm.state_size")).intValue());
            assertEquals(N_K, ((Number) md.get("olmo-hybrid.ssm.group_count")).intValue());
            assertEquals(N_V, ((Number) md.get("olmo-hybrid.ssm.time_step_rank")).intValue());
            assertEquals(HEAD_V * N_V, ((Number) md.get("olmo-hybrid.ssm.inner_size")).intValue());
            assertEquals(CONV_K, ((Number) md.get("olmo-hybrid.ssm.conv_kernel")).intValue());
            assertEquals(2.0, ((Number) md.get("olmo-hybrid.ssm.beta_scale")).doubleValue(), 1e-6);
            assertEquals((double) DELTA_EPS, ((Number) md.get("olmo-hybrid.ssm.norm_eps")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_qkv.weight"));
            assertTrue(tensors.contains("blk.0.ssm_conv1d.weight"));
            assertTrue(tensors.contains("blk.0.ssm_beta.weight"));
            assertTrue(tensors.contains("blk.0.ssm_alpha.weight"));
            assertTrue(tensors.contains("blk.0.ssm_a"));
            assertTrue(tensors.contains("blk.0.post_attention_norm.weight"));
            assertTrue(tensors.contains("blk.1.attn_q.weight"));
            assertTrue(tensors.contains("blk.1.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.1.post_ffw_norm.weight"));
            assertFalse(tensors.contains("blk.1.attn_norm.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.OLMO_HYBRID, spec.arch());
            LayerSpec linear = spec.layer(0);
            assertEquals(MixerKind.GDN, linear.mixerKind());
            assertEquals(PostNormRole.PRE_FFN, linear.norm().postRole());
            assertEquals(PostNormRole.SANDWICH, spec.layer(1).norm().postRole());
            assertEquals(MixerKind.ATTENTION, spec.layer(1).mixerKind());
            assertEquals(2.0f, linear.recurrent().betaScale(), 0f);
            assertEquals(DELTA_EPS, spec.hparams().ssmNormEps(), 0f);
            assertNotNull(linear.recurrent());
            assertNull(spec.layer(0).attention());
        }
    }

    /** The no-RoPE layout records a zero rotary width and an all-false rope pattern. */
    @Test
    public void noRopeLayoutRotatesNothing() throws Exception {
        Map<String, String> overrides = new LinkedHashMap<>();
        overrides.put("rope_parameters", "null");
        overrides.put("rope_theta", "null");
        Path dir = builder(overrides).write(tmp.newFolder("nope").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("nope-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals(0, ((Number) md.get("olmo-hybrid.rope.dimension_count")).intValue());
            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            assertNull(spec.layer(1).rope());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("hidden_act", "'gelu'"), "hidden_act");
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        assertRejected(Collections.singletonMap("linear_use_gate", "false"), "linear_use_gate");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'rope_type':'yarn','rope_theta':10000.0,'factor':2.0}"), "rope_parameters.rope_type");
        assertRejected(Collections.singletonMap("partial_rotary_factor", "0.5"), "partial_rotary_factor");
        Map<String, String> oddHeads = new LinkedHashMap<>();
        oddHeads.put("linear_num_key_heads", "2");
        oddHeads.put("linear_num_value_heads", "3");
        assertRejected(oddHeads, "linear_num_value_heads");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        String name = "rejected-" + field.replace('.', '-');
        Path dir = builder(overrides).write(tmp.newFolder(name).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint(Map<String, String> overrides) throws Exception {
        return builder(overrides).write(tmp.newFolder("olmo-hybrid").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(4).build();
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "model.embed_tokens.weight", V, D, embedding());
        // layer 0: Gated DeltaNet
        tensor(b, "model.layers.0.input_layernorm.weight", 1, D, normWeight(1));
        tensor(b, "model.layers.0.post_attention_layernorm.weight", 1, D, normWeight(2));
        tensor(b, "model.layers.0.linear_attn.q_proj.weight", N_K * HEAD_K, D, matrix(N_K * HEAD_K, D, 5));
        tensor(b, "model.layers.0.linear_attn.k_proj.weight", N_K * HEAD_K, D, matrix(N_K * HEAD_K, D, 6));
        tensor(b, "model.layers.0.linear_attn.v_proj.weight", N_V * HEAD_V, D, matrix(N_V * HEAD_V, D, 7));
        tensor(b, "model.layers.0.linear_attn.g_proj.weight", N_V * HEAD_V, D, matrix(N_V * HEAD_V, D, 8));
        tensor(b, "model.layers.0.linear_attn.b_proj.weight", N_V, D, matrix(N_V, D, 9));
        tensor(b, "model.layers.0.linear_attn.a_proj.weight", N_V, D, matrix(N_V, D, 10));
        conv(b, "model.layers.0.linear_attn.q_conv1d.weight", N_K * HEAD_K, 11);
        conv(b, "model.layers.0.linear_attn.k_conv1d.weight", N_K * HEAD_K, 12);
        conv(b, "model.layers.0.linear_attn.v_conv1d.weight", N_V * HEAD_V, 13);
        tensor(b, "model.layers.0.linear_attn.A_log", 1, N_V, vector(N_V, 14));
        tensor(b, "model.layers.0.linear_attn.dt_bias", 1, N_V, vector(N_V, 15));
        tensor(b, "model.layers.0.linear_attn.o_norm.weight", 1, HEAD_V, normWeightAt(16, HEAD_V));
        tensor(b, "model.layers.0.linear_attn.o_proj.weight", D, N_V * HEAD_V, matrix(D, N_V * HEAD_V, 17));
        ffn(b, 0, 18);
        // layer 1: OLMo2 post-norm attention
        tensor(b, "model.layers.1.post_attention_layernorm.weight", 1, D, normWeight(21));
        tensor(b, "model.layers.1.post_feedforward_layernorm.weight", 1, D, normWeight(22));
        tensor(b, "model.layers.1.self_attn.q_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, 23));
        tensor(b, "model.layers.1.self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, D,
                matrix(KV_HEADS * HEAD_DIM, D, 24));
        tensor(b, "model.layers.1.self_attn.v_proj.weight", KV_HEADS * HEAD_DIM, D,
                matrix(KV_HEADS * HEAD_DIM, D, 25));
        tensor(b, "model.layers.1.self_attn.o_proj.weight", D, HEADS * HEAD_DIM, matrix(D, HEADS * HEAD_DIM, 26));
        tensor(b, "model.layers.1.self_attn.q_norm.weight", 1, HEADS * HEAD_DIM, normWeightAt(27, HEADS * HEAD_DIM));
        tensor(b, "model.layers.1.self_attn.k_norm.weight", 1, KV_HEADS * HEAD_DIM,
                normWeightAt(28, KV_HEADS * HEAD_DIM));
        ffn(b, 1, 29);
        tensor(b, "model.norm.weight", 1, D, normWeight(32));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 33));
        return b;
    }

    private static void ffn(SyntheticHf.Builder b, int layer, int seed) {
        tensor(b, "model.layers." + layer + ".mlp.gate_proj.weight", FF, D, matrix(FF, D, seed));
        tensor(b, "model.layers." + layer + ".mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 1));
        tensor(b, "model.layers." + layer + ".mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 2));
    }

    private static void conv(SyntheticHf.Builder b, String name, int channels, int seed) {
        float[] w = new float[channels * CONV_K];
        for (int i = 0; i < w.length; i++) {
            w[i] = ((i * 5 + seed * 3) % 13 - 6) * 0.05f;
        }
        b.tensor(name, StDType.F32, new long[]{channels, 1, CONV_K}, SyntheticHf.f32(w));
    }

    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['OlmoHybridForCausalLM']");
        fields.put("model_type", "'olmo_hybrid'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("num_hidden_layers", "2");
        fields.put("num_attention_heads", Integer.toString(HEADS));
        fields.put("num_key_value_heads", Integer.toString(KV_HEADS));
        fields.put("head_dim", Integer.toString(HEAD_DIM));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("hidden_act", "'silu'");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("layer_types", "['linear_attention','full_attention']");
        fields.put("linear_num_key_heads", Integer.toString(N_K));
        fields.put("linear_num_value_heads", Integer.toString(N_V));
        fields.put("linear_key_head_dim", Integer.toString(HEAD_K));
        fields.put("linear_value_head_dim", Integer.toString(HEAD_V));
        fields.put("linear_conv_kernel_dim", Integer.toString(CONV_K));
        fields.put("linear_allow_neg_eigval", "true");
        fields.put("rope_parameters", "{'rope_type':'default','rope_theta':10000.0}");
        fields.putAll(overrides);
        StringBuilder json = new StringBuilder("{");
        for (Map.Entry<String, String> e : fields.entrySet()) {
            if (json.length() > 1) {
                json.append(',');
            }
            json.append('\'').append(e.getKey()).append("':").append(e.getValue());
        }
        return SyntheticHf.json(json.append('}').toString());
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = rows == 1 ? new long[]{cols} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static float[] embedding() {
        float[] x = new float[V * D];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 3) % 11 - 5) * 0.05f;
        }
        return x;
    }

    private static float[] normWeight(int seed) {
        return normWeightAt(seed, D);
    }

    private static float[] normWeightAt(int seed, int width) {
        float[] x = new float[width];
        for (int i = 0; i < width; i++) {
            x[i] = 0.7f + 0.05f * i + 0.01f * seed;
        }
        return x;
    }

    private static float[] vector(int width, int seed) {
        float[] x = new float[width];
        for (int i = 0; i < width; i++) {
            x[i] = ((i * 3 + seed) % 7 - 3) * 0.1f;
        }
        return x;
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.03f;
        }
        return x;
    }

    // ------------------------------------------------------------------ scalar reference

    private static float[] reference(int[] tokens, boolean rope) {
        float[] emb = embedding();
        float[][] x = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            x[t] = new float[D];
            for (int k = 0; k < D; k++) {
                x[t][k] = emb[tokens[t] * D + k];
            }
        }
        gdnLayer(x, rope);
        attentionLayer(x, rope);
        float[] logits = new float[V];
        for (int t = 0; t < tokens.length; t++) {
            float[] h = rmsNorm(x[t], normWeight(32), EPS);
            float[] y = linear(h, matrix(V, D, 33), V);
            if (t == tokens.length - 1) {
                logits = y;
            }
        }
        return logits;
    }

    /** Layer 0: pre-norm Gated DeltaNet, pre-norm FFN, both residual adds. */
    private static void gdnLayer(float[][] x, boolean rope) {
        int n = x.length;
        float[] n0 = new float[D];
        float[][] qIn = new float[n][], kIn = new float[n][], vIn = new float[n][], gate = new float[n][];
        float[][] beta = new float[n][], g = new float[n][];
        float[] n0w = normWeight(1);
        for (int t = 0; t < n; t++) {
            n0 = rmsNorm(x[t], n0w, EPS);
            qIn[t] = linear(n0, matrix(N_K * HEAD_K, D, 5), N_K * HEAD_K);
            kIn[t] = linear(n0, matrix(N_K * HEAD_K, D, 6), N_K * HEAD_K);
            vIn[t] = linear(n0, matrix(N_V * HEAD_V, D, 7), N_V * HEAD_V);
            gate[t] = linear(n0, matrix(N_V * HEAD_V, D, 8), N_V * HEAD_V);
            float[] b = linear(n0, matrix(N_V, D, 9), N_V);
            float[] a = linear(n0, matrix(N_V, D, 10), N_V);
            float[] dt = vector(N_V, 15);
            float[] aLog = vector(N_V, 14);
            beta[t] = new float[N_V];
            g[t] = new float[N_V];
            for (int h = 0; h < N_V; h++) {
                beta[t][h] = 2f * sigmoid(b[h]);
                g[t][h] = -(float) Math.exp(aLog[h]) * softplus(a[h] + dt[h]);
            }
        }
        float[][] q = conv(qIn, N_K * HEAD_K, 11);
        float[][] k = conv(kIn, N_K * HEAD_K, 12);
        float[][] v = conv(vIn, N_V * HEAD_V, 13);
        float[][][] state = new float[N_V][HEAD_V][HEAD_K];
        float[][] out = new float[n][N_V * HEAD_V];
        float scale = (float) (1.0 / Math.sqrt(HEAD_K));
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < N_V; h++) {
                int kh = h / (N_V / N_K);
                float[] qh = l2(normSlice(q[t], kh, HEAD_K));
                float[] khv = l2(normSlice(k[t], kh, HEAD_K));
                float[] vh = slice(v[t], h, HEAD_V);
                float decay = (float) Math.exp(g[t][h]);
                for (int col = 0; col < HEAD_V; col++) {
                    for (int i = 0; i < HEAD_K; i++) {
                        state[h][col][i] *= decay;
                    }
                }
                for (int col = 0; col < HEAD_V; col++) {
                    float kv = 0;
                    for (int i = 0; i < HEAD_K; i++) {
                        kv += state[h][col][i] * khv[i];
                    }
                    float delta = (vh[col] - kv) * beta[t][h];
                    float o = 0;
                    for (int i = 0; i < HEAD_K; i++) {
                        state[h][col][i] += khv[i] * delta;
                        o += state[h][col][i] * qh[i];
                    }
                    out[t][h * HEAD_V + col] = o * scale;
                }
            }
        }
        float[] oNorm = normWeightAt(16, HEAD_V);
        for (int t = 0; t < n; t++) {
            float[] normed = new float[N_V * HEAD_V];
            for (int h = 0; h < N_V; h++) {
                float[] head = new float[HEAD_V];
                System.arraycopy(out[t], h * HEAD_V, head, 0, HEAD_V);
                float[] nh = rmsNorm(head, oNorm, DELTA_EPS);
                for (int j = 0; j < HEAD_V; j++) {
                    normed[h * HEAD_V + j] = nh[j] * silu(gate[t][h * HEAD_V + j]);
                }
            }
            float[] attn = linear(normed, matrix(D, N_V * HEAD_V, 17), D);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += attn[kk];
            }
            float[] n2 = rmsNorm(x[t], normWeight(2), EPS);
            float[] mlp = mlp(n2, 18, D, FF);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += mlp[kk];
            }
        }
    }

    /** Layer 1: OLMo2 post-norm attention and post-norm FFN (no pre-norms). */
    private static void attentionLayer(float[][] x, boolean rope) {
        int n = x.length;
        float[] qw = matrix(HEADS * HEAD_DIM, D, 23);
        float[] kw = matrix(KV_HEADS * HEAD_DIM, D, 24);
        float[] vw = matrix(KV_HEADS * HEAD_DIM, D, 25);
        float[] ow = matrix(D, HEADS * HEAD_DIM, 26);
        float[] qNorm = normWeightAt(27, HEADS * HEAD_DIM);
        float[] kNorm = normWeightAt(28, KV_HEADS * HEAD_DIM);
        float[][] q = new float[n][], k = new float[n][], v = new float[n][];
        for (int t = 0; t < n; t++) {
            q[t] = rmsNorm(linear(x[t], qw, HEADS * HEAD_DIM), qNorm, EPS);
            k[t] = rmsNorm(linear(x[t], kw, KV_HEADS * HEAD_DIM), kNorm, EPS);
            v[t] = linear(x[t], vw, KV_HEADS * HEAD_DIM);
        }
        if (rope) {
            for (int t = 0; t < n; t++) {
                for (int h = 0; h < HEADS; h++) {
                    rotateHalf(q[t], h * HEAD_DIM, HEAD_DIM, t);
                }
                for (int h = 0; h < KV_HEADS; h++) {
                    rotateHalf(k[t], h * HEAD_DIM, HEAD_DIM, t);
                }
            }
        }
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[][] out = new float[n][HEADS * HEAD_DIM];
        for (int t = 0; t < n; t++) {
            float[][] weights = new float[HEADS][t + 1];
            for (int h = 0; h < HEADS; h++) {
                int kvh = h / (HEADS / KV_HEADS);
                float total = 0;
                for (int j = 0; j <= t; j++) {
                    float dot = 0;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        dot += q[t][h * HEAD_DIM + d] * k[j][kvh * HEAD_DIM + d];
                    }
                    weights[h][j] = (float) Math.exp(dot * scale);
                    total += weights[h][j];
                }
                for (int j = 0; j <= t; j++) {
                    float p = weights[h][j] / total;
                    for (int d = 0; d < HEAD_DIM; d++) {
                        out[t][h * HEAD_DIM + d] += p * v[j][kvh * HEAD_DIM + d];
                    }
                }
            }
        }
        for (int t = 0; t < n; t++) {
            float[] attn = linear(out[t], ow, D);
            float[] post = rmsNorm(attn, normWeight(21), EPS);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += post[kk];
            }
            float[] mlp = mlp(x[t], 29, D, FF);
            float[] postFfn = rmsNorm(mlp, normWeight(22), EPS);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += postFfn[kk];
            }
        }
    }

    private static float[][] conv(float[][] in, int channels, int seed) {
        float[] w = new float[channels * CONV_K];
        for (int i = 0; i < w.length; i++) {
            w[i] = ((i * 5 + seed * 3) % 13 - 6) * 0.05f;
        }
        float[][] out = new float[in.length][channels];
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < in.length; t++) {
                float sum = 0;
                for (int k = 0; k < CONV_K; k++) {
                    int src = t + k - (CONV_K - 1);
                    if (src >= 0) {
                        sum += in[src][c] * w[c * CONV_K + k];
                    }
                }
                out[t][c] = silu(sum);
            }
        }
        return out;
    }

    private static float[] normSlice(float[] row, int head, int dim) {
        float[] out = new float[dim];
        System.arraycopy(row, head * dim, out, 0, dim);
        return out;
    }

    private static float[] slice(float[] row, int head, int dim) {
        return normSlice(row, head, dim);
    }

    private static float[] l2(float[] v) {
        double sum = 0;
        for (float e : v) {
            sum += (double) e * e;
        }
        float inv = (float) (1.0 / Math.sqrt(sum + L2_EPS));
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * inv;
        }
        return out;
    }

    private static void rotateHalf(float[] x, int off, int dim, int position) {
        int half = dim / 2;
        float[] in = new float[dim];
        System.arraycopy(x, off, in, 0, dim);
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(10000.0, -2.0 * i / dim);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[off + i] = in[i] * cos - in[i + half] * sin;
            x[off + i + half] = in[i + half] * cos + in[i] * sin;
        }
    }

    private static float[] mlp(float[] in, int seed, int width, int ff) {
        float[] gate = linear(in, matrix(ff, width, seed), ff);
        float[] up = linear(in, matrix(ff, width, seed + 1), ff);
        float[] act = new float[ff];
        for (int i = 0; i < ff; i++) {
            act[i] = silu(gate[i]) * up[i];
        }
        return linear(act, matrix(width, ff, seed + 2), width);
    }

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
        }
        return y;
    }

    private static float[] rmsNorm(float[] x, float[] w, float eps) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum / x.length + eps));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * inv * w[i];
        }
        return y;
    }

    private static float silu(float v) {
        return v / (1f + (float) Math.exp(-v));
    }

    private static float sigmoid(float v) {
        return 1f / (1f + (float) Math.exp(-v));
    }

    private static float softplus(float v) {
        return (float) Math.log1p(Math.exp(v));
    }
}
