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
import com.eonmux.neuralfree.gguf.arch.AttnGateKind;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
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
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** AliceAI checkpoint loading and causal inference against a scalar reference of vLLM's {@code alice_ai.py} forward. */
public class HfAliceAiTest {

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 8, V = 7;
    private static final int A_HEADS = 2, A_KV = 1, A_HEAD_DIM = 4, ROT = 2;
    private static final int N_K = 4, N_V = 4, HEAD_K = 2, HEAD_V = 2, CONV_K = 2;
    private static final int KEY_DIM = N_K * HEAD_K, VAL_DIM = N_V * HEAD_V;
    private static final int E = 3, TOP_K = 2, FF = 3, SHEXP = 2;
    private static final int BLOCK = 2, N_LAYER = 4;
    private static final float EPS = 1e-6f;
    private static final int[] TOKENS = {2, 3, 4, 5};
    private static final String[] LAYER_TYPES =
            {"linear_attention", "full_attention", "linear_attention", "full_attention"};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The same forward through OpenCL: KDA, the gated attention, the MoE and the attention-residual mixes. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint(Collections.<String, String>emptyMap());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The KDA conv / recurrent state and the attention KV cache carry across steps. */
    @Test
    public void incrementalDecodeMatchesPrefill() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(new int[]{TOKENS[0], TOKENS[1], TOKENS[2]}, 0);
            float[] actual = model.forwardBackend().decode(TOKENS[3], 3).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** AliceAI is a supported generative hybrid with the Qwen NeoX rope. */
    @Test
    public void registeredAsASupportedHybrid() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("AliceAIForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.ALICE_AI);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(MemoryKind.HYBRID, traits.memoryKind());
    }

    /** The exported GGUF file loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsAliceAi() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("alice-ai.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The converter writes the AliceAI layout, its metadata and the per-layer mixer kinds. */
    @Test
    public void conversionWritesTheAliceAiLayout() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("alice-ai", md.get("general.architecture"));
            assertEquals(N_LAYER, ((Number) md.get("alice-ai.block_count")).intValue());
            assertEquals(ROT, ((Number) md.get("alice-ai.rope.dimension_count")).intValue());
            assertEquals(CONV_K, ((Number) md.get("alice-ai.ssm.conv_kernel")).intValue());
            assertEquals(HEAD_K, ((Number) md.get("alice-ai.kda.head_dim")).intValue());
            assertEquals(BLOCK, ((Number) md.get("alice-ai.attn_res.block_size")).intValue());
            assertEquals(FF, ((Number) md.get("alice-ai.expert_feed_forward_length")).intValue());
            assertEquals(SHEXP, ((Number) md.get("alice-ai.expert_shared_feed_forward_length")).intValue());
            assertEquals(1, ((Number) md.get("alice-ai.expert_shared_count")).intValue());
            assertEquals(0, ((Number) md.get("alice-ai.leading_dense_block_count")).intValue());
            assertEquals(1.0, ((Number) md.get("alice-ai.expert_weights_scale")).doubleValue(), 1e-6);
            assertEquals(2, ((Number) md.get("alice-ai.expert_gating_func")).intValue());
            assertTrue((Boolean) md.get("alice-ai.expert_weights_norm"));
            assertEquals(java.util.Arrays.asList(N_K, A_HEADS, N_K, A_HEADS),
                    md.get("alice-ai.attention.head_count"));
            assertEquals(java.util.Arrays.asList(0, A_KV, 0, A_KV),
                    md.get("alice-ai.attention.head_count_kv"));
            assertEquals((double) EPS,
                    ((Number) md.get("alice-ai.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
            assertTrue(tensors.contains("blk.0.ssm_f_a.weight"));
            assertTrue(tensors.contains("blk.0.ssm_f_b.weight"));
            assertTrue(tensors.contains("blk.0.ssm_g_a.weight"));
            assertTrue(tensors.contains("blk.0.ssm_g_b.weight"));
            assertTrue(tensors.contains("blk.0.ssm_beta.weight"));
            assertTrue(tensors.contains("blk.0.ssm_a"));
            assertTrue(tensors.contains("blk.0.ssm_dt.bias"));
            assertTrue(tensors.contains("blk.0.ssm_norm.weight"));
            assertTrue(tensors.contains("blk.0.ssm_conv1d_q.weight"));
            assertTrue(tensors.contains("blk.0.ffn_res_score.weight"));
            assertFalse(tensors.contains("blk.0.attn_res_score.weight"));
            assertTrue(tensors.contains("blk.1.attn_res_score.weight"));
            assertTrue(tensors.contains("blk.1.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.1.attn_k_norm.weight"));
            assertTrue(tensors.contains("blk.0.ffn_gate_exps.weight"));
            assertTrue(tensors.contains("blk.0.ffn_up_exps.weight"));
            assertTrue(tensors.contains("blk.0.ffn_down_exps.weight"));
            assertTrue(tensors.contains("blk.0.ffn_gate_inp_shexp.weight"));
            assertTrue(tensors.contains("blk.0.ffn_gate_shexp.weight"));
            assertTrue(tensors.contains("blk.0.exp_probs_b.bias"));
            assertTrue(tensors.contains("output_res_score.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.ALICE_AI, spec.arch());
            LayerSpec linear = spec.layer(0);
            assertEquals(MixerKind.KDA, linear.mixerKind());
            assertEquals(HEAD_K, linear.recurrent().headDim());
            assertEquals(N_K, linear.recurrent().nHead());
            assertEquals(CONV_K, linear.recurrent().convSteps() + 1);
            assertNull(linear.attention());
            LayerSpec full = spec.layer(1);
            assertEquals(MixerKind.ATTENTION, full.mixerKind());
            assertEquals(AttnGateKind.Q_INTERLEAVED, full.attention().gate());
            assertEquals(ROT, full.rope().nDims());
            assertEquals(2, spec.hparams().attnResBlockSize());
            assertTrue(linear.hasExtra(LayerSpec.Extras.ATTN_RES));
            assertTrue(full.hasExtra(LayerSpec.Extras.ATTN_RES));
        }
    }

    /** The unsupported graph variants are refused by name. */
    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("hidden_act", "'gelu'"), "hidden_act");
        assertRejected(Collections.singletonMap("kda_allow_negative_eigenvalues", "true"),
                "kda_allow_negative_eigenvalues");
        assertRejected(Collections.singletonMap("router_score_function", "'softmax'"), "router_score_function");
        assertRejected(Collections.singletonMap("router_bias_correction", "false"), "router_bias_correction");
        assertRejected(Collections.singletonMap("linear_num_value_heads", "6"), "linear_num_value_heads");
        assertRejected(Collections.singletonMap("linear_value_head_dim", "3"), "linear_value_head_dim");
        assertRejected(Collections.singletonMap("linear_conv_kernel_dim", "1"), "linear_conv_kernel_dim");
        assertRejected(Collections.singletonMap("block_attn_res_block_size", "0"), "block_attn_res_block_size");
        assertRejected(Collections.singletonMap("layer_types", "['linear_attention','full_attention']"),
                "layer_types");
        assertRejected(Collections.singletonMap("shared_expert_intermediate_size", "0"),
                "shared_expert_intermediate_size");
    }

    /** The checkpoint's own unsupported tensor names are refused by name. */
    @Test
    public void refusesUnsupportedTensorsByName() throws Exception {
        SyntheticHf.Builder b = builder(Collections.<String, String>emptyMap());
        b.tensor("model.layers.0.linear_attn.mystery.weight", StDType.F32, new long[]{2, D},
                SyntheticHf.f32(matrix(2, D, 91)));
        Path dir = b.write(tmp.newFolder("unknown-tensor").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("unknown-tensor-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains("mystery"));
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
        return builder(overrides).write(tmp.newFolder("alice-ai").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(8).build();
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "model.embed_tokens.weight", V, D, embedding());
        for (int il = 0; il < N_LAYER; il++) {
            layer(b, il);
        }
        tensor(b, "model.norm.weight", 1, D, zeroCentered(50, D));
        tensor(b, "model.attnres_final.res_norm_weight", 1, D, positive(51, D));
        tensor(b, "model.attnres_final.res_proj.weight", 1, D, matrix(1, D, 52));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 53));
        return b;
    }

    private static void layer(SyntheticHf.Builder b, int il) {
        int s = 100 + il * 30;
        tensor(b, "model.layers." + il + ".input_layernorm.weight", 1, D, zeroCentered(s, D));
        tensor(b, "model.layers." + il + ".post_attention_layernorm.weight", 1, D, zeroCentered(s + 1, D));
        tensor(b, "model.layers." + il + ".mlp_res_norm_weight", 1, D, positive(s + 2, D));
        tensor(b, "model.layers." + il + ".mlp_res_proj.weight", 1, D, matrix(1, D, s + 3));
        if (il > 0) {
            tensor(b, "model.layers." + il + ".attn_res_norm_weight", 1, D, positive(s + 4, D));
            tensor(b, "model.layers." + il + ".attn_res_proj.weight", 1, D, matrix(1, D, s + 5));
        }
        tensor(b, "model.layers." + il + ".mlp.gate.weight", E, D, matrix(E, D, s + 6));
        tensor(b, "model.layers." + il + ".mlp.gate.e_score_correction_bias", 1, E, vector(E, s + 7));
        tensor(b, "model.layers." + il + ".mlp.experts.gate_up_proj", E, 2 * FF, D, matrix(E * 2 * FF, D, s + 8));
        tensor(b, "model.layers." + il + ".mlp.experts.down_proj", E, D, FF, matrix(E * D, FF, s + 9));
        tensor(b, "model.layers." + il + ".mlp.shared_expert_gate.weight", 1, D, matrix(1, D, s + 10));
        tensor(b, "model.layers." + il + ".mlp.shared_expert.gate_proj.weight", SHEXP, D,
                matrix(SHEXP, D, s + 11));
        tensor(b, "model.layers." + il + ".mlp.shared_expert.up_proj.weight", SHEXP, D, matrix(SHEXP, D, s + 12));
        tensor(b, "model.layers." + il + ".mlp.shared_expert.down_proj.weight", D, SHEXP, matrix(D, SHEXP, s + 13));
        if (LAYER_TYPES[il].equals("linear_attention")) {
            kda(b, il, s + 14);
        } else {
            attention(b, il, s + 14);
        }
    }

    private static void kda(SyntheticHf.Builder b, int il, int s) {
        String p = "model.layers." + il + ".linear_attn.";
        tensor(b, p + "q_proj.weight", KEY_DIM, D, matrix(KEY_DIM, D, s));
        tensor(b, p + "k_proj.weight", KEY_DIM, D, matrix(KEY_DIM, D, s + 1));
        tensor(b, p + "v_proj.weight", VAL_DIM, D, matrix(VAL_DIM, D, s + 2));
        tensor(b, p + "o_proj.weight", D, VAL_DIM, matrix(D, VAL_DIM, s + 3));
        tensor(b, p + "f_a_proj.weight", HEAD_V, D, matrix(HEAD_V, D, s + 4));
        tensor(b, p + "f_b_proj.weight", KEY_DIM, HEAD_V, matrix(KEY_DIM, HEAD_V, s + 5));
        tensor(b, p + "b_proj.weight", N_K, D, matrix(N_K, D, s + 6));
        tensor(b, p + "g_a_proj.weight", HEAD_V, D, matrix(HEAD_V, D, s + 7));
        tensor(b, p + "g_b_proj.weight", VAL_DIM, HEAD_V, matrix(VAL_DIM, HEAD_V, s + 8));
        tensor(b, p + "o_norm.weight", 1, HEAD_V, positive(s + 9, HEAD_V));
        tensor(b, p + "a_log_bias", 1, N_K, vector(N_K, s + 10));
        tensor(b, p + "dt_bias", 1, KEY_DIM, vector(KEY_DIM, s + 11));
        conv(b, p + "q_conv1d.weight", KEY_DIM, s + 12);
        conv(b, p + "k_conv1d.weight", KEY_DIM, s + 13);
        conv(b, p + "v_conv1d.weight", VAL_DIM, s + 14);
    }

    private static void attention(SyntheticHf.Builder b, int il, int s) {
        String p = "model.layers." + il + ".self_attn.";
        tensor(b, p + "q_proj.weight", 2 * A_HEADS * A_HEAD_DIM, D, matrix(2 * A_HEADS * A_HEAD_DIM, D, s));
        tensor(b, p + "k_proj.weight", A_KV * A_HEAD_DIM, D, matrix(A_KV * A_HEAD_DIM, D, s + 1));
        tensor(b, p + "v_proj.weight", A_KV * A_HEAD_DIM, D, matrix(A_KV * A_HEAD_DIM, D, s + 2));
        tensor(b, p + "o_proj.weight", D, A_HEADS * A_HEAD_DIM, matrix(D, A_HEADS * A_HEAD_DIM, s + 3));
        tensor(b, p + "q_norm.weight", 1, A_HEAD_DIM, zeroCentered(s + 4, A_HEAD_DIM));
        tensor(b, p + "k_norm.weight", 1, A_HEAD_DIM, zeroCentered(s + 5, A_HEAD_DIM));
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
        fields.put("architectures", "['AliceAIForCausalLM']");
        fields.put("model_type", "'alice_ai'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("num_hidden_layers", Integer.toString(N_LAYER));
        fields.put("num_attention_heads", Integer.toString(A_HEADS));
        fields.put("num_key_value_heads", Integer.toString(A_KV));
        fields.put("head_dim", Integer.toString(A_HEAD_DIM));
        fields.put("linear_num_key_heads", Integer.toString(N_K));
        fields.put("linear_num_value_heads", Integer.toString(N_V));
        fields.put("linear_key_head_dim", Integer.toString(HEAD_K));
        fields.put("linear_value_head_dim", Integer.toString(HEAD_V));
        fields.put("linear_conv_kernel_dim", Integer.toString(CONV_K));
        fields.put("num_experts", Integer.toString(E));
        fields.put("num_experts_per_tok", Integer.toString(TOP_K));
        fields.put("moe_intermediate_size", Integer.toString(FF));
        fields.put("shared_expert_intermediate_size", Integer.toString(SHEXP));
        fields.put("block_attn_res_block_size", Integer.toString(BLOCK));
        fields.put("router_score_function", "'sigmoid'");
        fields.put("router_bias_correction", "true");
        fields.put("kda_allow_negative_eigenvalues", "false");
        fields.put("max_position_embeddings", "16");
        fields.put("rope_theta", "1000000.0");
        fields.put("partial_rotary_factor", "0.5");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("hidden_act", "'silu'");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("layer_types", layerTypes());
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

    private static String layerTypes() {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < LAYER_TYPES.length; i++) {
            if (i > 0) {
                out.append(',');
            }
            out.append('\'').append(LAYER_TYPES[i]).append('\'');
        }
        return out.append(']').toString();
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = rows == 1 ? new long[]{cols} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int outer, int rows, int cols, float[] data) {
        b.tensor(name, StDType.F32, new long[]{outer, rows, cols}, SyntheticHf.f32(data));
    }

    private static float[] embedding() {
        float[] x = new float[V * D];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 3) % 11 - 5) * 0.05f;
        }
        return x;
    }

    /** A checkpoint zero-centered norm: the converter adds one before the runtime multiplies. */
    private static float[] zeroCentered(int seed, int width) {
        float[] x = new float[width];
        for (int i = 0; i < width; i++) {
            x[i] = 0.05f * i + 0.01f * seed;
        }
        return x;
    }

    private static float[] positive(int seed, int width) {
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

    private static float[] reference(int[] tokens) {
        float[] emb = embedding();
        float[][] x = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            x[t] = new float[D];
            System.arraycopy(emb, tokens[t] * D, x[t], 0, D);
        }
        List<float[][]> completed = new java.util.ArrayList<>();
        completed.add(copy(x));
        float[][] partial = null;
        for (int il = 0; il < N_LAYER; il++) {
            if (il > 0 && il % BLOCK == 0) {
                completed.add(partial);
                partial = null;
            }
            float[][] mixed = il == 0 ? copy(x) : mix(completed, partial, positive(100 + il * 30 + 4, D),
                    matrix(1, D, 100 + il * 30 + 5));
            float[][] n = gemma(mixed, zeroCentered(100 + il * 30, D));
            float[][] attn = LAYER_TYPES[il].equals("linear_attention") ? kdaLayer(n, il) : attentionLayer(n, il);
            partial = partial == null ? attn : add(partial, attn);
            float[][] mixed2 = mix(completed, partial, positive(100 + il * 30 + 2, D),
                    matrix(1, D, 100 + il * 30 + 3));
            float[][] moe = moe(gemma(mixed2, zeroCentered(100 + il * 30 + 1, D)), il);
            partial = add(partial, moe);
            x = partial;
        }
        float[][] finalMixed = mix(completed, partial, positive(51, D), matrix(1, D, 52));
        float[] logits = new float[V];
        float[][] h = gemma(finalMixed, zeroCentered(50, D));
        for (int t = 0; t < tokens.length; t++) {
            float[] y = linear(h[t], matrix(V, D, 53), V);
            if (t == tokens.length - 1) {
                logits = y;
            }
        }
        return logits;
    }

    /** {@code _depth_softmax_mix}: rms-normed sources scored by the fused norm*proj weight, softmax, weighted sum. */
    private static float[][] mix(List<float[][]> completed, float[][] partial, float[] normW, float[] projW) {
        List<float[][]> sources = new java.util.ArrayList<>(completed);
        if (partial != null) {
            sources.add(partial);
        }
        int n = sources.get(0).length;
        float[][] out = new float[n][D];
        for (int t = 0; t < n; t++) {
            float[] scores = new float[sources.size()];
            for (int c = 0; c < sources.size(); c++) {
                float[] keys = rms(sources.get(c)[t], normW, EPS);
                double dot = 0;
                for (int i = 0; i < D; i++) {
                    dot += (double) keys[i] * projW[i];
                }
                scores[c] = (float) dot;
            }
            softmax(scores);
            for (int c = 0; c < sources.size(); c++) {
                for (int i = 0; i < D; i++) {
                    out[t][i] += scores[c] * sources.get(c)[t][i];
                }
            }
        }
        return out;
    }

    /** Layer KDA: depthwise Q/K/V convolutions, low-rank decay/output gates, the delta rule, the gated output norm. */
    private static float[][] kdaLayer(float[][] n, int il) {
        int s = 100 + il * 30 + 14;
        float[][] qIn = project(n, KEY_DIM, D, s);
        float[][] kIn = project(n, KEY_DIM, D, s + 1);
        float[][] vIn = project(n, VAL_DIM, D, s + 2);
        float[][] gate = project(n, HEAD_V, D, s + 7, VAL_DIM, s + 8);
        float[][] alpha = project(n, HEAD_V, D, s + 4, KEY_DIM, s + 5);
        float[][] betaLogits = project(n, N_K, D, s + 6);
        float[][] q = conv(qIn, KEY_DIM, s + 12);
        float[][] k = conv(kIn, KEY_DIM, s + 13);
        float[][] v = conv(vIn, VAL_DIM, s + 14);
        float[] aLog = vector(N_K, s + 10);
        float[] dt = vector(KEY_DIM, s + 11);
        float[][][] state = new float[N_V][HEAD_V][HEAD_K];
        float[][] out = new float[n.length][VAL_DIM];
        float scale = (float) (1.0 / Math.sqrt(HEAD_K));
        for (int t = 0; t < n.length; t++) {
            for (int h = 0; h < N_V; h++) {
                float[] qh = l2(slice(q[t], h, HEAD_K));
                float[] kh = l2(slice(k[t], h, HEAD_K));
                float[] vh = slice(v[t], h, HEAD_V);
                float[] decay = new float[HEAD_K];
                for (int i = 0; i < HEAD_K; i++) {
                    decay[i] = (float) -Math.exp(aLog[h]) * softplus(alpha[t][h * HEAD_K + i] + dt[h * HEAD_K + i]);
                }
                float beta = sigmoid(betaLogits[t][h]);
                for (int col = 0; col < HEAD_V; col++) {
                    for (int i = 0; i < HEAD_K; i++) {
                        state[h][col][i] *= (float) Math.exp(decay[i]);
                    }
                    double kv = 0;
                    for (int i = 0; i < HEAD_K; i++) {
                        kv += (double) state[h][col][i] * kh[i];
                    }
                    float delta = (float) ((vh[col] - kv) * beta);
                    double o = 0;
                    for (int i = 0; i < HEAD_K; i++) {
                        state[h][col][i] += kh[i] * delta;
                        o += (double) state[h][col][i] * qh[i];
                    }
                    out[t][h * HEAD_V + col] = (float) o * scale;
                }
            }
        }
        float[] oNorm = positive(s + 9, HEAD_V);
        float[][] normed = new float[n.length][VAL_DIM];
        for (int t = 0; t < n.length; t++) {
            for (int h = 0; h < N_V; h++) {
                float[] head = rms(slice(out[t], h, HEAD_V), oNorm, EPS);
                for (int j = 0; j < HEAD_V; j++) {
                    normed[t][h * HEAD_V + j] = head[j] * sigmoid(gate[t][h * HEAD_V + j]);
                }
            }
        }
        return project(normed, D, VAL_DIM, s + 3);
    }

    /** Layer Qwen3-Next gated attention: per-head Gemma Q/K norms, partial NeoX RoPE and the Q gate. */
    private static float[][] attentionLayer(float[][] n, int il) {
        int s = 100 + il * 30 + 14;
        float[][] qg = project(n, 2 * A_HEADS * A_HEAD_DIM, D, s);
        float[][] k = project(n, A_KV * A_HEAD_DIM, D, s + 1);
        float[][] v = project(n, A_KV * A_HEAD_DIM, D, s + 2);
        float[] qNorm = zeroCentered(s + 4, A_HEAD_DIM);
        float[] kNorm = zeroCentered(s + 5, A_HEAD_DIM);
        float[][] q = new float[n.length][A_HEADS * A_HEAD_DIM];
        float[][] gate = new float[n.length][A_HEADS * A_HEAD_DIM];
        for (int t = 0; t < n.length; t++) {
            for (int h = 0; h < A_HEADS; h++) {
                for (int d = 0; d < A_HEAD_DIM; d++) {
                    q[t][h * A_HEAD_DIM + d] = qg[t][h * 2 * A_HEAD_DIM + d];
                    gate[t][h * A_HEAD_DIM + d] = qg[t][h * 2 * A_HEAD_DIM + A_HEAD_DIM + d];
                }
            }
            q[t] = gemmaHeads(q[t], qNorm, A_HEADS);
            rotate(q[t], t, A_HEADS);
            float[] kt = gemmaHeads(k[t], kNorm, A_KV);
            rotate(kt, t, A_KV);
            System.arraycopy(kt, 0, k[t], 0, kt.length);
        }
        float scale = (float) (1.0 / Math.sqrt(A_HEAD_DIM));
        float[][] out = new float[n.length][A_HEADS * A_HEAD_DIM];
        for (int t = 0; t < n.length; t++) {
            for (int h = 0; h < A_HEADS; h++) {
                int kvh = h / (A_HEADS / A_KV);
                float[] w = new float[t + 1];
                double total = 0;
                for (int j = 0; j <= t; j++) {
                    double dot = 0;
                    for (int d = 0; d < A_HEAD_DIM; d++) {
                        dot += (double) q[t][h * A_HEAD_DIM + d] * k[j][kvh * A_HEAD_DIM + d];
                    }
                    w[j] = (float) Math.exp(dot * scale);
                    total += w[j];
                }
                for (int j = 0; j <= t; j++) {
                    float p = (float) (w[j] / total);
                    for (int d = 0; d < A_HEAD_DIM; d++) {
                        out[t][h * A_HEAD_DIM + d] += p * v[j][kvh * A_HEAD_DIM + d];
                    }
                }
            }
            for (int i = 0; i < A_HEADS * A_HEAD_DIM; i++) {
                out[t][i] *= sigmoid(gate[t][i]);
            }
        }
        return project(out, D, A_HEADS * A_HEAD_DIM, s + 3);
    }

    /** Sigmoid top-k MoE with the selection bias, the fused expert stack and the gated shared expert. */
    private static float[][] moe(float[][] n, int il) {
        int s = 100 + il * 30 + 6;
        float[] gateW = matrix(E, D, s);
        float[] bias = vector(E, s + 1);
        float[] gateUp = matrix(E * 2 * FF, D, s + 2);
        float[] down = matrix(E * D, FF, s + 3);
        float[] sharedGate = matrix(1, D, s + 4);
        float[] sharedIn = matrix(SHEXP, D, s + 5);
        float[] sharedUp = matrix(SHEXP, D, s + 6);
        float[] sharedDown = matrix(D, SHEXP, s + 7);
        float[][] out = new float[n.length][D];
        for (int t = 0; t < n.length; t++) {
            float[] logits = linear(n[t], gateW, E);
            float[] probs = new float[E];
            float[] sel = new float[E];
            for (int e = 0; e < E; e++) {
                probs[e] = sigmoid(logits[e]);
                sel[e] = probs[e] + bias[e];
            }
            int[] ids = topK(sel, TOP_K);
            double sum = 0;
            for (int id : ids) {
                sum += probs[id];
            }
            float[] routed = new float[D];
            for (int id : ids) {
                float weight = (float) (probs[id] / sum);
                float[] gu = linear(n[t], rows(gateUp, id * 2 * FF, 2 * FF, D), 2 * FF);
                float[] act = new float[FF];
                for (int f = 0; f < FF; f++) {
                    act[f] = silu(gu[f]) * gu[FF + f];
                }
                float[] eOut = linear(act, rows(down, id * D, D, FF), D);
                for (int d = 0; d < D; d++) {
                    routed[d] += weight * eOut[d];
                }
            }
            float[] sGate = linear(n[t], sharedGate, 1);
            float[] sIn = linear(n[t], sharedIn, SHEXP);
            float[] sUp = linear(n[t], sharedUp, SHEXP);
            float[] sAct = new float[SHEXP];
            for (int f = 0; f < SHEXP; f++) {
                sAct[f] = silu(sIn[f]) * sUp[f];
            }
            float[] shared = linear(sAct, sharedDown, D);
            float g = sigmoid(sGate[0]);
            for (int d = 0; d < D; d++) {
                out[t][d] = routed[d] + g * shared[d];
            }
        }
        return out;
    }

    /** Two chained linear projections with a width in between. */
    private static float[][] project(float[][] in, int rows, int cols, int seed) {
        float[] w = matrix(rows, cols, seed);
        float[][] out = new float[in.length][];
        for (int t = 0; t < in.length; t++) {
            out[t] = linear(in[t], w, rows);
        }
        return out;
    }

    private static float[][] project(float[][] in, int mid, int inCols, int seedA, int rows, int seedB) {
        float[] a = matrix(mid, inCols, seedA);
        float[] b = matrix(rows, mid, seedB);
        float[][] out = new float[in.length][];
        for (int t = 0; t < in.length; t++) {
            out[t] = linear(linear(in[t], a, mid), b, rows);
        }
        return out;
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

    /** Gemma RMSNorm over whole rows with the checkpoint's zero-centered weight. */
    private static float[][] gemma(float[][] x, float[] w) {
        float[][] out = new float[x.length][];
        for (int t = 0; t < x.length; t++) {
            out[t] = rms(x[t], plusOne(w), EPS);
        }
        return out;
    }

    private static float[] gemmaHeads(float[] row, float[] w, int heads) {
        float[] out = new float[row.length];
        float[] w1 = plusOne(w);
        for (int h = 0; h < heads; h++) {
            float[] head = rms(slice(row, h, w.length), w1, EPS);
            System.arraycopy(head, 0, out, h * w.length, w.length);
        }
        return out;
    }

    private static float[] plusOne(float[] w) {
        float[] out = new float[w.length];
        for (int i = 0; i < w.length; i++) {
            out[i] = 1.0f + w[i];
        }
        return out;
    }

    /** Partial NeoX RoPE over the first {@link #ROT} values of every head. */
    private static void rotate(float[] row, int position, int heads) {
        int half = ROT / 2;
        for (int h = 0; h < heads; h++) {
            int off = h * (row.length / heads);
            float[] in = new float[ROT];
            System.arraycopy(row, off, in, 0, ROT);
            for (int i = 0; i < half; i++) {
                double angle = position * Math.pow(1000000.0, -2.0 * i / ROT);
                float cos = (float) Math.cos(angle);
                float sin = (float) Math.sin(angle);
                row[off + i] = in[i] * cos - in[i + half] * sin;
                row[off + i + half] = in[i + half] * cos + in[i] * sin;
            }
        }
    }

    private static int[] topK(float[] scores, int k) {
        boolean[] taken = new boolean[scores.length];
        int[] out = new int[k];
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < scores.length; e++) {
                if (!taken[e] && (best < 0 || scores[e] > scores[best])) {
                    best = e;
                }
            }
            out[j] = best;
            taken[best] = true;
        }
        return out;
    }

    private static float[] slice(float[] row, int head, int dim) {
        float[] out = new float[dim];
        System.arraycopy(row, head * dim, out, 0, dim);
        return out;
    }

    private static float[] rows(float[] w, int first, int count, int rowWidth) {
        float[] out = new float[count * rowWidth];
        System.arraycopy(w, first * rowWidth, out, 0, out.length);
        return out;
    }

    private static float[] l2(float[] v) {
        double sum = 0;
        for (float e : v) {
            sum += (double) e * e;
        }
        float inv = (float) (1.0 / Math.sqrt(sum + 1e-6));
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * inv;
        }
        return out;
    }

    private static void softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) Math.exp(x[i] - max);
            sum += x[i];
        }
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) (x[i] / sum);
        }
    }

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < x.length; j++) {
                sum += (double) x[j] * w[i * x.length + j];
            }
            y[i] = (float) sum;
        }
        return y;
    }

    private static float[] rms(float[] x, float[] w, float eps) {
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

    private static float[][] add(float[][] a, float[][] b) {
        float[][] out = new float[a.length][];
        for (int t = 0; t < a.length; t++) {
            out[t] = new float[a[t].length];
            for (int i = 0; i < a[t].length; i++) {
                out[t][i] = a[t][i] + b[t][i];
            }
        }
        return out;
    }

    private static float[][] copy(float[][] x) {
        float[][] out = new float[x.length][];
        for (int t = 0; t < x.length; t++) {
            out[t] = x[t].clone();
        }
        return out;
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
