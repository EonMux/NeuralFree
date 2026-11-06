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
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
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

/** Solar checkpoints on the Llama graph: loading, the plain layout and causal inference against a scalar reference. */
public class HfSolarTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 4, V = 7, FF = 4;
    private static final float EPS = 1e-5f;
    private static final int[] TOKENS = {2, 3, 4};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The same forward through the OpenCL backend. */
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

    /** Solar resolves to the supported Llama architecture. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("SolarForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.LLAMA);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NORM, traits.ropeType());
    }

    /** The exported GGUF file is a plain {@code llama} model that runs the same forward. */
    @Test
    public void exportedGgufLoadsAsLlama() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("solar.gguf");
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

    /** The converter writes the Llama layout, with the Q/K projections permuted for the NORM rotary. */
    @Test
    public void conversionWritesTheLlamaLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("llama", md.get("general.architecture"));
            assertEquals(1, ((Number) md.get("llama.block_count")).intValue());
            assertEquals(D, ((Number) md.get("llama.rope.dimension_count")).intValue());
            assertEquals(V, ((Number) md.get("llama.vocab_size")).intValue());
            assertEquals(EPS, ((Number) md.get("llama.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
            assertTrue(tensors.contains("blk.0.ffn_gate.weight"));
            assertTrue(tensors.contains("output.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.LLAMA, spec.arch());
            assertEquals(RopeType.NORM, spec.layer(0).rope().type());
            assertFalse(spec.head().tied());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("bskcn_1", "[1]"), "bskcn_1");
        assertRejected(Collections.singletonMap("bskcn_4", "[1]"), "bskcn_4");
        assertRejected(Collections.singletonMap("logit_scale", "0.5"), "logit_scale");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("solar").toPath());
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
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        tensor(b, "model.layers.0.input_layernorm.weight", 1, D, normWeight(2));
        tensor(b, "model.layers.0.self_attn.q_proj.weight", D, D, matrix(D, D, 3));
        tensor(b, "model.layers.0.self_attn.k_proj.weight", D, D, matrix(D, D, 4));
        tensor(b, "model.layers.0.self_attn.v_proj.weight", D, D, matrix(D, D, 5));
        tensor(b, "model.layers.0.self_attn.o_proj.weight", D, D, matrix(D, D, 6));
        tensor(b, "model.layers.0.post_attention_layernorm.weight", 1, D, normWeight(7));
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FF, D, matrix(FF, D, 8));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FF, D, matrix(FF, D, 9));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FF, matrix(D, FF, 10));
        tensor(b, "model.norm.weight", 1, D, normWeight(11));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 12));
        return b;
    }

    /** The Solar {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['SolarForCausalLM']");
        fields.put("model_type", "'llama'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("num_hidden_layers", "1");
        fields.put("num_attention_heads", "1");
        fields.put("num_key_value_heads", "1");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("rope_theta", "10000.0");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("attention_bias", "false");
        fields.put("mlp_bias", "false");
        fields.put("bskcn_1", "[]");
        fields.put("bskcn_2", "[]");
        fields.put("bskcn_3", "[]");
        fields.put("bskcn_4", "[]");
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

    /** HF's {@code rotate_half} over the whole head; the conversion permutes Q/K for the NORM rotary. */
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
        float[] embeddings = matrix(V, D, 1);
        float[] x = null;
        for (int t = 0; t < tokens.length; t++) {
            x = new float[D];
            System.arraycopy(embeddings, tokens[t] * D, x, 0, D);
            float[] n = rmsNorm(x, normWeight(2));
            float[] q = linear(n, matrix(D, D, 3), D);
            keys[t] = linear(n, matrix(D, D, 4), D);
            values[t] = linear(n, matrix(D, D, 5), D);
            rotateHalf(q, t);
            rotateHalf(keys[t], t);
            float[] weights = new float[t + 1];
            float total = 0;
            for (int j = 0; j <= t; j++) {
                float dot = 0;
                for (int k = 0; k < D; k++) {
                    dot += q[k] * keys[j][k];
                }
                weights[j] = (float) Math.exp(dot / Math.sqrt(D));
                total += weights[j];
            }
            float[] attention = new float[D];
            for (int j = 0; j <= t; j++) {
                for (int k = 0; k < D; k++) {
                    attention[k] += weights[j] / total * values[j][k];
                }
            }
            float[] attnOut = linear(attention, matrix(D, D, 6), D);
            for (int k = 0; k < D; k++) {
                x[k] += attnOut[k];
            }
            float[] n2 = rmsNorm(x, normWeight(7));
            float[] gate = linear(n2, matrix(FF, D, 8), FF);
            float[] up = linear(n2, matrix(FF, D, 9), FF);
            for (int k = 0; k < FF; k++) {
                gate[k] = (float) (gate[k] / (1 + Math.exp(-gate[k]))) * up[k];
            }
            float[] ffnOut = linear(gate, matrix(D, FF, 10), D);
            for (int k = 0; k < D; k++) {
                x[k] += ffnOut[k];
            }
        }
        return linear(rmsNorm(x, normWeight(11)), matrix(V, D, 12), V);
    }
}
