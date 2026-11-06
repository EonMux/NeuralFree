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
import com.eonmux.neuralfree.gguf.arch.PostNormRole;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** HyperCLOVAX checkpoint loading, the MuP scales of the layout and causal inference against a scalar reference. */
public class HfHyperclovaxTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 4, V = 7, LAYERS = 1, FF = 4;
    private static final float EPS = 1e-5f, ATTN = 0.7f, EMB = 2.0f, RES = 1.5f, LOGIT = 0.5f;
    private static final int[] TOKENS = {2, 3, 4};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The same forward through the OpenCL backend: the post norms, residual and logit multipliers included. */
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

    /** HyperCLOVAX is a supported text architecture with the sandwich norms and the scaled residual. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("HyperCLOVAXForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.HYPERCLOVAX);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(PostNormRole.SANDWICH, traits.postNormRole());
        assertEquals(ResidualKind.SCALED, traits.residualKind());
    }

    /** The exported GGUF file (NeuralFree's {@code hyperclovax} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsHyperclovax() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("hyperclovax.gguf");
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

    /** The converter writes the HyperCLOVAX layout and the MuP metadata. */
    @Test
    public void conversionWritesTheHyperclovaxLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("hyperclovax", md.get("general.architecture"));
            assertEquals(1, ((Number) md.get("hyperclovax.block_count")).intValue());
            assertEquals(D, ((Number) md.get("hyperclovax.embedding_length")).intValue());
            assertEquals(FF, ((Number) md.get("hyperclovax.feed_forward_length")).intValue());
            assertEquals(D, ((Number) md.get("hyperclovax.rope.dimension_count")).intValue());
            assertEquals(ATTN, ((Number) md.get("hyperclovax.attention.scale")).doubleValue(), 1e-6);
            assertEquals(EMB, ((Number) md.get("hyperclovax.embedding_scale")).doubleValue(), 1e-6);
            assertEquals(RES, ((Number) md.get("hyperclovax.residual_scale")).doubleValue(), 1e-6);
            assertEquals(LOGIT, ((Number) md.get("hyperclovax.logit_scale")).doubleValue(), 1e-6);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("token_embd.weight"));
            assertTrue(tensors.contains("blk.0.attn_norm.weight"));
            assertTrue(tensors.contains("blk.0.post_attention_norm.weight"));
            assertTrue(tensors.contains("blk.0.ffn_norm.weight"));
            assertTrue(tensors.contains("blk.0.post_ffw_norm.weight"));
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
            assertTrue(tensors.contains("blk.0.ffn_down.weight"));
            assertTrue(tensors.contains("output.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.HYPERCLOVAX, spec.arch());
            LayerSpec layer = spec.layer(0);
            assertEquals(RopeType.NEOX, layer.rope().type());
            assertEquals(ResidualKind.SCALED, layer.residual().kind());
            assertEquals(PostNormRole.SANDWICH, layer.norm().postRole());
            assertTrue(layer.norm().postAttnNorm());
            assertTrue(layer.norm().postFfnNorm());
            assertEquals(ATTN, layer.attention().kqScale(), 1e-6);
            assertFalse(spec.head().tied());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("use_post_norm", "false"), "use_post_norm");
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        assertRejected(Collections.singletonMap("hidden_act", "'gelu'"), "hidden_act");
        assertRejected(Collections.singletonMap("embedding_multiplier", "0"), "embedding_multiplier");
        assertRejected(Collections.singletonMap("rope_scaling", "{'rope_type':'yarn','factor':2.0}"),
                "rope_parameters.rope_type");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field.replace('.', '-')).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field.replace('.', '-') + "-cache").toPath()).build(),
                        null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("hyperclovax").toPath());
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
        tensor(b, "model.embed_tokens.weight", V, D, embedding());
        tensor(b, "model.layers.0.input_layernorm.weight", 1, D, normWeight(1));
        tensor(b, "model.layers.0.post_attention_layernorm.weight", 1, D, normWeight(2));
        tensor(b, "model.layers.0.post_norm1.weight", 1, D, normWeight(3));
        tensor(b, "model.layers.0.post_norm2.weight", 1, D, normWeight(4));
        tensor(b, "model.layers.0.self_attn.q_proj.weight", D, D, matrix(D, D, 5));
        tensor(b, "model.layers.0.self_attn.k_proj.weight", D, D, matrix(D, D, 6));
        tensor(b, "model.layers.0.self_attn.v_proj.weight", D, D, matrix(D, D, 7));
        tensor(b, "model.layers.0.self_attn.o_proj.weight", D, D, matrix(D, D, 8));
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FF, D, matrix(FF, D, 9));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FF, D, matrix(FF, D, 10));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FF, matrix(D, FF, 11));
        tensor(b, "model.norm.weight", 1, D, normWeight(12));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 13));
        return b;
    }

    /** The HyperCLOVAX {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['HyperCLOVAXForCausalLM']");
        fields.put("model_type", "'hyperclovax'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("num_hidden_layers", Integer.toString(LAYERS));
        fields.put("num_attention_heads", "1");
        fields.put("num_key_value_heads", "1");
        fields.put("head_dim", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("hidden_act", "'silu'");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("rope_theta", "10000.0");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("use_post_norm", "true");
        fields.put("attention_multiplier", Float.toString(ATTN));
        fields.put("embedding_multiplier", Float.toString(EMB));
        fields.put("residual_multiplier", Float.toString(RES));
        fields.put("logits_scaling", Float.toString(LOGIT));
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

    /** HF's {@code rotate_half} over the whole head (vLLM's {@code is_neox_style=True}). */
    private static void rotateHalf(float[] x, int position) {
        int half = x.length / 2;
        float[] in = x.clone();
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(10000.0, -2.0 * i / x.length);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[i] = in[i] * cos - in[i + half] * sin;
            x[i + half] = in[i + half] * cos + in[i] * sin;
        }
    }

    private static float[] reference(int[] tokens) {
        float[][] keys = new float[tokens.length][], values = new float[tokens.length][];
        float[] embeddings = embedding();
        float[][] qw = {matrix(D, D, 5), matrix(D, D, 6), matrix(D, D, 7), matrix(D, D, 8)};
        float[][] fw = {matrix(FF, D, 9), matrix(FF, D, 10), matrix(D, FF, 11)};
        float[] x = null;
        for (int t = 0; t < tokens.length; t++) {
            x = new float[D];
            for (int k = 0; k < D; k++) {
                x[k] = embeddings[tokens[t] * D + k] * EMB;
            }
            float[] n = rmsNorm(x, normWeight(1));
            float[] q = linear(n, qw[0], D);
            keys[t] = linear(n, qw[1], D);
            values[t] = linear(n, qw[2], D);
            rotateHalf(q, t);
            rotateHalf(keys[t], t);
            float[] weights = new float[t + 1];
            float total = 0;
            for (int j = 0; j <= t; j++) {
                float dot = 0;
                for (int k = 0; k < D; k++) {
                    dot += q[k] * keys[j][k];
                }
                weights[j] = (float) Math.exp(dot * ATTN);
                total += weights[j];
            }
            float[] attention = new float[D];
            for (int j = 0; j <= t; j++) {
                for (int k = 0; k < D; k++) {
                    attention[k] += weights[j] / total * values[j][k];
                }
            }
            float[] attnOut = rmsNorm(linear(attention, qw[3], D), normWeight(3));
            for (int k = 0; k < D; k++) {
                x[k] += attnOut[k] * RES;
            }
            float[] n2 = rmsNorm(x, normWeight(2));
            float[] gate = linear(n2, fw[0], FF);
            float[] up = linear(n2, fw[1], FF);
            for (int k = 0; k < FF; k++) {
                gate[k] = (float) (gate[k] / (1 + Math.exp(-gate[k]))) * up[k];
            }
            float[] ffnOut = rmsNorm(linear(gate, fw[2], D), normWeight(4));
            for (int k = 0; k < D; k++) {
                x[k] += ffnOut[k] * RES;
            }
        }
        float[] hidden = rmsNorm(x, normWeight(12));
        float[] logits = linear(hidden, matrix(V, D, 13), V);
        for (int i = 0; i < V; i++) {
            logits[i] *= LOGIT;
        }
        return logits;
    }
}
