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
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** Param2MoE conversion onto the dots1 layout and causal inference against a scalar reference of vLLM's forward. */
public class HfParam2MoeTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 4, V = 7, FF = 4, MOE_FF = 2;
    private static final int EXPERTS = 4, TOP_K = 2, SHARED_FF = 2;
    private static final float EPS = 1e-5f, SCALE = 2.5f, ROTARY = 1.0f;
    private static final int[] TOKENS = {2, 3, 4};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The same forward through the OpenCL backend: dense lead, sigmoid routing and the shared expert included. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** Param2MoE resolves to the supported dots1 architecture. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("Param2MoEForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.DOTS1);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
    }

    /** The exported GGUF file (NeuralFree's {@code dots1} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsDots1() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("param2moe.gguf");
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

    /** The converter splits the fused QKV, stacks the experts and writes the dots1 routing metadata. */
    @Test
    public void conversionWritesTheDots1Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("dots1", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("dots1.block_count")).intValue());
            assertEquals(1, ((Number) md.get("dots1.leading_dense_block_count")).intValue());
            assertEquals(EXPERTS, ((Number) md.get("dots1.expert_count")).intValue());
            assertEquals(SHARED_FF, ((Number) md.get("dots1.expert_feed_forward_length")).intValue());
            assertEquals(D, ((Number) md.get("dots1.rope.dimension_count")).intValue());
            assertEquals(SCALE, ((Number) md.get("dots1.expert_weights_scale")).doubleValue(), 1e-6);
            assertTrue((Boolean) md.get("dots1.expert_weights_norm"));
            assertEquals(2, ((Number) md.get("dots1.expert_gating_func")).intValue());

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
            assertTrue(tensors.contains("blk.0.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.0.attn_k_norm.weight"));
            assertTrue(tensors.contains("blk.0.ffn_gate.weight"));
            assertTrue(tensors.contains("blk.1.ffn_gate_inp.weight"));
            assertTrue(tensors.contains("blk.1.exp_probs_b.bias"));
            assertTrue(tensors.contains("blk.1.ffn_gate_exps.weight"));
            assertTrue(tensors.contains("blk.1.ffn_gate_shexp.weight"));
            assertTrue("the tied token table fills the dots1 head", tensors.contains("output.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.DOTS1, spec.arch());
            LayerSpec dense = spec.layer(0);
            assertTrue(dense.moe() == null);
            assertEquals(FF, dense.ffn().nFf());
            LayerSpec moe = spec.layer(1);
            assertNotNull(moe.moe());
            assertEquals(EXPERTS, moe.moe().nExpert());
            assertEquals(TOP_K, moe.moe().nUsed());
            assertEquals(MOE_FF, moe.moe().nFfExp());
            assertEquals(GatingFunc.SIGMOID, moe.moe().gating());
            assertTrue(moe.moe().normWeights());
            assertEquals(SCALE, moe.moe().weightsScale(), 1e-6);
            assertNotNull(moe.moe().sharedExpert());
            assertEquals(SHARED_FF, moe.moe().sharedExpert().nFf());
            assertEquals(D, moe.attention().nRot());
            assertEquals(1f / 2f, moe.attention().kqScale(), 1e-6);
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("use_qk_norm", "false"), "use_qk_norm");
        assertRejected(Collections.singletonMap("n_group", "2"), "n_group");
        assertRejected(Collections.singletonMap("num_nextn_predict_layers", "1"), "num_nextn_predict_layers");
        assertRejected(Collections.singletonMap("use_qkv_bias", "true"), "use_qkv_bias");
        assertRejected(Collections.singletonMap("hidden_act", "'gelu'"), "hidden_act");
        assertRejected(Collections.singletonMap("partial_rotary_factor", "0.5"), "partial_rotary_factor");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("param2moe").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(4).build();
    }

    private SyntheticHf.Builder builder() {
        return builder(Collections.<String, String>emptyMap());
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "model.word_embeddings.weight", V, D, embedding());
        tensor(b, "model.norm.weight", 1, D, normWeight(1));
        for (int il = 0; il < 2; il++) {
            String p = "model.layers." + il + ".";
            tensor(b, p + "input_layernorm.weight", 1, D, normWeight(2 + il));
            tensor(b, p + "post_attention_layernorm.weight", 1, D, normWeight(4 + il));
            tensor(b, p + "attention.query_key_value.weight", 3 * D, D, matrix(3 * D, D, 6 + il));
            tensor(b, p + "attention.query_layernorm.weight", 1, D, normWeight(8 + il));
            tensor(b, p + "attention.key_layernorm.weight", 1, D, normWeight(10 + il));
            tensor(b, p + "attention.dense.weight", D, D, matrix(D, D, 12 + il));
        }
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FF, D, matrix(FF, D, 14));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FF, D, matrix(FF, D, 15));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FF, matrix(D, FF, 16));
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 17));
        tensor(b, "model.layers.1.mlp.gate.expert_bias", 1, EXPERTS, expertBias());
        for (int e = 0; e < EXPERTS; e++) {
            String p = "model.layers.1.mlp.experts." + e + ".";
            tensor(b, p + "gate_proj.weight", MOE_FF, D, matrix(MOE_FF, D, 20 + e));
            tensor(b, p + "up_proj.weight", MOE_FF, D, matrix(MOE_FF, D, 30 + e));
            tensor(b, p + "down_proj.weight", D, MOE_FF, matrix(D, MOE_FF, 40 + e));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", SHARED_FF, D, matrix(SHARED_FF, D, 50));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", SHARED_FF, D, matrix(SHARED_FF, D, 51));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, SHARED_FF, matrix(D, SHARED_FF, 52));
        return b;
    }

    /** The Param2MoE {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['Param2MoEForCausalLM']");
        fields.put("model_type", "'param2moe'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("moe_intermediate_size", Integer.toString(MOE_FF));
        fields.put("num_hidden_layers", "2");
        fields.put("num_attention_heads", "1");
        fields.put("num_key_value_heads", "1");
        fields.put("head_dim", Integer.toString(D));
        fields.put("num_experts", Integer.toString(EXPERTS));
        fields.put("num_experts_per_tok", Integer.toString(TOP_K));
        fields.put("num_shared_experts", "1");
        fields.put("moe_router_enable_expert_bias", "true");
        fields.put("n_group", "1");
        fields.put("topk_group", "1");
        fields.put("norm_topk_prob", "true");
        fields.put("routed_scaling_factor", Float.toString(SCALE));
        fields.put("score_function", "'sigmoid'");
        fields.put("first_k_dense_replace", "1");
        fields.put("use_qk_norm", "true");
        fields.put("use_qkv_bias", "false");
        fields.put("use_bias", "false");
        fields.put("partial_rotary_factor", Float.toString(ROTARY));
        fields.put("hidden_act", "'silu'");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("rope_theta", "10000.0");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "true");
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

    private static float[] expertBias() {
        float[] x = new float[EXPERTS];
        for (int i = 0; i < EXPERTS; i++) {
            x[i] = 0.1f + 0.1f * i;
        }
        return x;
    }

    private static float[] normWeight(int seed) {
        float[] x = new float[D];
        for (int i = 0; i < D; i++) {
            x[i] = 0.7f + 0.1f * i + 0.01f * seed;
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

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
        }
        return y;
    }

    private static float[] rmsNorm(float[] x, float[] w) {
        double sum = 0;
        for (float v : x) {
            sum += v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * inv * w[i];
        }
        return y;
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = (float) (gate[i] / (1 + Math.exp(-gate[i]))) * up[i];
        }
        return y;
    }

    /** HF's {@code rotate_half} over the first {@code rotary} dims of the head, the rest passed through. */
    private static void rotateHalf(float[] x, int rotary, int position) {
        int half = rotary / 2;
        float[] in = x.clone();
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(10000.0, -2.0 * i / rotary);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[i] = in[i] * cos - in[i + half] * sin;
            x[i + half] = in[i + half] * cos + in[i] * sin;
        }
    }

    private static float[] reference(int[] tokens) {
        float[] embeddings = embedding();
        int rotary = (int) (D * ROTARY);
        float[] bias = expertBias();
        float mean = 0;
        for (float v : bias) {
            mean += v / bias.length;
        }
        float[][] hidden = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            hidden[t] = new float[D];
            System.arraycopy(embeddings, tokens[t] * D, hidden[t], 0, D);
        }
        for (int il = 0; il < 2; il++) {
            float[] qNorm = normWeight(8 + il), kNorm = normWeight(10 + il);
            float[][] qkv = new float[tokens.length][], keys = new float[tokens.length][],
                    values = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                float[] n = rmsNorm(hidden[t], normWeight(2 + il));
                qkv[t] = linear(n, matrix(3 * D, D, 6 + il), 3 * D);
                float[] q = new float[D], k = new float[D], v = new float[D];
                System.arraycopy(qkv[t], 0, q, 0, D);
                System.arraycopy(qkv[t], D, k, 0, D);
                System.arraycopy(qkv[t], 2 * D, v, 0, D);
                q = rmsNorm(q, qNorm);
                k = rmsNorm(k, kNorm);
                rotateHalf(q, rotary, t);
                rotateHalf(k, rotary, t);
                keys[t] = k;
                values[t] = v;
            }
            for (int t = 0; t < tokens.length; t++) {
                float[] q = new float[D];
                System.arraycopy(qkv[t], 0, q, 0, D);
                q = rmsNorm(q, qNorm);
                rotateHalf(q, rotary, t);
                float[] weights = new float[t + 1];
                float total = 0;
                for (int j = 0; j <= t; j++) {
                    float dot = 0;
                    for (int d = 0; d < D; d++) {
                        dot += q[d] * keys[j][d];
                    }
                    weights[j] = (float) Math.exp(dot / Math.sqrt(D));
                    total += weights[j];
                }
                float[] attention = new float[D];
                for (int j = 0; j <= t; j++) {
                    for (int d = 0; d < D; d++) {
                        attention[d] += weights[j] / total * values[j][d];
                    }
                }
                float[] stream = new float[D];
                float[] attnOut = linear(attention, matrix(D, D, 12 + il), D);
                for (int d = 0; d < D; d++) {
                    stream[d] = hidden[t][d] + attnOut[d];
                }
                float[] n2 = rmsNorm(stream, normWeight(4 + il));
                float[] mlpOut = il == 0 ? dense(n2) : moe(n2, bias, mean);
                for (int d = 0; d < D; d++) {
                    hidden[t][d] = stream[d] + mlpOut[d];
                }
            }
        }
        float[] x = rmsNorm(hidden[tokens.length - 1], normWeight(1));
        float[] logits = new float[V];
        for (int v = 0; v < V; v++) {
            float dot = 0;
            for (int d = 0; d < D; d++) {
                dot += embeddings[v * D + d] * x[d];
            }
            logits[v] = dot;
        }
        return logits;
    }

    private static float[] dense(float[] x) {
        float[] f = siluMul(linear(x, matrix(FF, D, 14), FF), linear(x, matrix(FF, D, 15), FF));
        return linear(f, matrix(D, FF, 16), D);
    }

    /** vLLM's FusedMoE with sigmoid scores, the correction bias, normalized top-k and the shared expert added. */
    private static float[] moe(float[] x, float[] bias, float mean) {
        float[] logits = linear(x, matrix(EXPERTS, D, 17), EXPERTS);
        float[] scores = new float[EXPERTS], choice = new float[EXPERTS];
        for (int e = 0; e < EXPERTS; e++) {
            scores[e] = (float) (1.0 / (1.0 + Math.exp(-logits[e])));
            choice[e] = scores[e] + (bias[e] - mean);
        }
        int[] top = topK(choice, TOP_K);
        float total = 0;
        for (int e : top) {
            total += scores[e];
        }
        float[] out = new float[D];
        for (int e : top) {
            float w = scores[e] / total * SCALE;
            float[] f = siluMul(linear(x, matrix(MOE_FF, D, 20 + e), MOE_FF),
                    linear(x, matrix(MOE_FF, D, 30 + e), MOE_FF));
            float[] y = linear(f, matrix(D, MOE_FF, 40 + e), D);
            for (int d = 0; d < D; d++) {
                out[d] += w * y[d];
            }
        }
        float[] f = siluMul(linear(x, matrix(SHARED_FF, D, 50), SHARED_FF),
                linear(x, matrix(SHARED_FF, D, 51), SHARED_FF));
        float[] shared = linear(f, matrix(D, SHARED_FF, 52), D);
        for (int d = 0; d < D; d++) {
            out[d] += shared[d];
        }
        return out;
    }

    private static int[] topK(float[] values, int k) {
        int[] top = new int[k];
        boolean[] used = new boolean[values.length];
        for (int i = 0; i < k; i++) {
            int best = -1;
            for (int e = 0; e < values.length; e++) {
                if (!used[e] && (best < 0 || values[e] > values[best])) {
                    best = e;
                }
            }
            used[best] = true;
            top[i] = best;
        }
        return top;
    }
}
