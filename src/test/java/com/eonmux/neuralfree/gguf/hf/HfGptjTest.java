package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.NormKind;
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** GPT-J checkpoint loading, converted layout and causal inference against a scalar implementation. */
public class HfGptjTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 4, V = 7;
    private static final float EPS = 1e-5f;

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 2e-4f);
        }
    }

    /** The same forward through the OpenCL backend: partial adjacent-pair RoPE, parallel residual and biases. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 2e-4f);
        }
    }

    /** GPT-J is a supported text architecture, listed for {@code nftool export --print-supported-models}. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("GPTJForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.GPTJ);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NORM, traits.ropeType());
        assertEquals(NormKind.LAYER, traits.normKind());
        assertEquals(ActOp.GELU_TANH, traits.ffnAct());
        assertEquals(ResidualKind.PARALLEL, traits.residualKind());
    }

    /** The exported GGUF file (NeuralFree's GPT-J layout) loads through the GGUF source and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsGptj() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("gptj.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(out.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(reference(new int[]{2, 3, 4}), actual, 2e-4f);
        }
    }

    /** The converter writes the GPT-J layout: partial adjacent-pair rotary, biased MLP and output head. */
    @Test
    public void conversionWritesTheGptjLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("gptj", md.get("general.architecture"));
            assertEquals(1, ((Number) md.get("gptj.block_count")).intValue());
            assertEquals(16, ((Number) md.get("gptj.context_length")).intValue());
            assertEquals(D, ((Number) md.get("gptj.embedding_length")).intValue());
            assertEquals(D, ((Number) md.get("gptj.feed_forward_length")).intValue());
            assertEquals(1, ((Number) md.get("gptj.attention.head_count")).intValue());
            assertEquals(2, ((Number) md.get("gptj.rope.dimension_count")).intValue());
            assertEquals(EPS, ((Number) md.get("gptj.attention.layer_norm_epsilon")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("token_embd.weight"));
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
            assertTrue(tensors.contains("blk.0.attn_k.weight"));
            assertTrue(tensors.contains("blk.0.attn_v.weight"));
            assertTrue(tensors.contains("blk.0.attn_output.weight"));
            assertTrue(tensors.contains("blk.0.attn_norm.bias"));
            assertTrue(tensors.contains("blk.0.ffn_up.bias"));
            assertTrue(tensors.contains("blk.0.ffn_down.bias"));
            assertTrue(tensors.contains("output_norm.bias"));
            assertTrue(tensors.contains("output.weight"));
            assertTrue(tensors.contains("output.bias"));
            assertFalse("GPT-J attention has no projection biases", tensors.contains("blk.0.attn_q.bias"));
            assertFalse("the causal-mask buffers are dropped", tensors.contains("blk.0.attn.bias"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.GPTJ, spec.arch());
            LayerSpec layer = spec.layer(0);
            assertEquals(RopeType.NORM, layer.rope().type());
            assertEquals(2, layer.rope().nDims());
            assertEquals(NormKind.LAYER, layer.norm().kind());
            assertEquals(ResidualKind.PARALLEL, layer.residual().kind());
            assertTrue(layer.attention().inBias());
            assertTrue(layer.attention().outBias());
            assertFalse(layer.ffn().gated());
            assertEquals(ActOp.GELU_TANH, layer.ffn().act());
            assertTrue(layer.ffn().biases());
            assertNotNull(spec.head().outputBias());
            assertFalse(spec.head().tied());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("tie_word_embeddings", "true"), "tie_word_embeddings");
        assertRejected(Collections.singletonMap("activation_function", "'gelu'"), "activation_function");
        assertRejected(Collections.singletonMap("rotary", "false"), "rotary");
        assertRejected(Collections.singletonMap("rotary_dim", "3"), "rotary_dim");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        SyntheticHf.Builder b = builder(overrides);
        Path dir = b.write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private SyntheticHf.Builder builder() {
        return builder(Collections.<String, String>emptyMap());
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "transformer.wte.weight", V, D, matrix(V, D, 1));
        tensor(b, "transformer.h.0.ln_1.weight", D, 1, ones());
        tensor(b, "transformer.h.0.ln_1.bias", D, 1, bias(1));
        for (int i = 0; i < 4; i++) {
            String[] names = {"q_proj", "k_proj", "v_proj", "out_proj"};
            tensor(b, "transformer.h.0.attn." + names[i] + ".weight", D, D, matrix(D, D, i + 2));
        }
        // Older checkpoints carry the two boolean causal-mask buffers the converter drops.
        b.tensor("transformer.h.0.attn.bias", StDType.BOOL, new long[]{1, 1, 16}, new byte[16]);
        b.tensor("transformer.h.0.attn.masked_bias", StDType.BOOL, new long[]{1}, new byte[]{1});
        tensor(b, "transformer.h.0.mlp.fc_in.weight", D, D, matrix(D, D, 6));
        tensor(b, "transformer.h.0.mlp.fc_in.bias", D, 1, bias(2));
        tensor(b, "transformer.h.0.mlp.fc_out.weight", D, D, matrix(D, D, 7));
        tensor(b, "transformer.h.0.mlp.fc_out.bias", D, 1, bias(3));
        tensor(b, "transformer.ln_f.weight", D, 1, ones());
        tensor(b, "transformer.ln_f.bias", D, 1, bias(4));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 8));
        tensor(b, "lm_head.bias", V, 1, bias(5, V));
        return b;
    }

    /** The GPT-J {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['GPTJForCausalLM']");
        fields.put("model_type", "'gptj'");
        fields.put("n_embd", Integer.toString(D));
        fields.put("n_layer", "1");
        fields.put("n_head", "1");
        fields.put("n_positions", "16");
        fields.put("rotary_dim", "2");
        fields.put("n_inner", Integer.toString(D));
        fields.put("activation_function", "'gelu_new'");
        fields.put("layer_norm_epsilon", "1e-5");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("bos_token_id", "0");
        fields.put("eos_token_id", "1");
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

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("gptj").toPath());
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.035f;
        return x;
    }
    private static float[] bias(int seed) { return bias(seed, D); }
    private static float[] bias(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (i - seed) * 0.025f;
        return x;
    }
    private static float[] ones() { return new float[]{1, 1, 1, 1}; }
    private static float[] linear(float[] x, float[] w, float[] bias, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = bias == null ? 0 : bias[i];
            for (int j = 0; j < x.length; j++) y[i] += x[j] * w[i * x.length + j];
        }
        return y;
    }
    private static float[] norm(float[] x, float[] bias) {
        float mean = 0, variance = 0;
        for (float v : x) mean += v / D;
        for (float v : x) variance += (v - mean) * (v - mean) / D;
        float[] y = new float[D];
        for (int i = 0; i < D; i++) y[i] = (float) ((x[i] - mean) / Math.sqrt(variance + EPS)) + bias[i];
        return y;
    }
    private static void rotate(float[] x, int position) {
        float a = x[0], b = x[1];
        x[0] = (float) (a * Math.cos(position) - b * Math.sin(position));
        x[1] = (float) (b * Math.cos(position) + a * Math.sin(position));
    }
    private static float[] reference(int[] tokens) {
        float[][] keys = new float[tokens.length][], values = new float[tokens.length][];
        float[] hidden = null;
        float[] embeddings = matrix(V, D, 1);
        for (int t = 0; t < tokens.length; t++) {
            float[] x = new float[D];
            System.arraycopy(embeddings, tokens[t] * D, x, 0, D);
            float[] n = norm(x, bias(1));
            float[] q = linear(n, matrix(D, D, 2), null, D);
            keys[t] = linear(n, matrix(D, D, 3), null, D);
            values[t] = linear(n, matrix(D, D, 4), null, D);
            rotate(q, t); rotate(keys[t], t);
            float[] weights = new float[t + 1];
            float total = 0;
            for (int j = 0; j <= t; j++) {
                float dot = 0;
                for (int k = 0; k < D; k++) dot += q[k] * keys[j][k];
                weights[j] = (float) Math.exp(dot / Math.sqrt(D)); total += weights[j];
            }
            float[] attention = new float[D];
            for (int j = 0; j <= t; j++) for (int k = 0; k < D; k++) attention[k] += weights[j] / total * values[j][k];
            attention = linear(attention, matrix(D, D, 5), null, D);
            float[] ff = linear(n, matrix(D, D, 6), bias(2), D);
            for (int k = 0; k < D; k++) ff[k] = (float) (0.5 * ff[k] * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (ff[k] + 0.044715 * ff[k] * ff[k] * ff[k]))));
            ff = linear(ff, matrix(D, D, 7), bias(3), D);
            for (int k = 0; k < D; k++) x[k] += attention[k] + ff[k];
            hidden = norm(x, bias(4));
        }
        return linear(hidden, matrix(V, D, 8), bias(5, V), V);
    }
}
