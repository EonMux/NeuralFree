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
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.AttentionSpec;
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

/**
 * RNJ-1 checkpoint loading, the Gemma 3 block with chunked local attention, and causal inference against a scalar
 * reference of vLLM's forward.
 */
public class HfRnj1Test {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 8, V = 7, FF = 8, HEADS = 2, HEAD_DIM = 4;
    /** Two layers: a full-attention layer 0 and a chunked layer 1 whose chunk is two positions wide. */
    private static final int LAYERS = 2;
    private static final int SLIDING_WINDOW = 4;
    private static final int CHUNK = SLIDING_WINDOW / 2;
    private static final float EPS = 1e-5f;
    private static final float ROPE_BASE = 10_000.0f;
    private static final float ROPE_BASE_SWA = 500_000.0f;
    private static final float QUERY_SCALAR = 16.0f;
    private static final float SOFT_CAP = 8.0f;
    private static final int[] TOKENS = {2, 3, 4, 5, 6, 0};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 2e-4f);
        }
    }

    /** The same forward through the OpenCL backend: the aligned chunk mask and the per-layer RoPE bases included. */
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

    /** The aligned chunks with one chunk of look-back, including the positions that push out the first chunk. */
    @Test
    public void chunkedLookbackWindowMasksWholeChunks() {
        KvWindow w = KvWindow.of(SwaType.CHUNKED_LOOKBACK, CHUNK, true);
        for (int q = 0; q < 8; q++) {
            for (int k = 0; k <= q; k++) {
                boolean expected = k >= Math.max(0, (q / CHUNK - 1) * CHUNK);
                assertEquals("q=" + q + " k=" + k, expected, w.visible(q, k));
            }
            assertFalse(w.visible(q, q + 1));
        }
        assertEquals(2 * CHUNK + 5 - 1, w.retainedPositions(5));
        assertEquals(2 * CHUNK, w.retainedPositions(1));
        assertEquals(2, w.firstVisible(4));
        assertEquals(0, w.firstVisible(1));
    }

    /** RNJ-1 is a supported text architecture with the Gemma 3 block and its own GGUF name. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("Rnj1ForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.RNJ1);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(PostNormRole.SANDWICH, traits.postNormRole());
        assertEquals(Arch.GEMMA3, traits.graphFamily());
    }

    /** The exported GGUF file (NeuralFree's {@code rnj1} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsRnj1() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("rnj1.gguf");
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

    /** The converter writes the Gemma 3 tensors plus the chunk window, its switches and the query scale. */
    @Test
    public void conversionWritesTheRnj1Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("rnj1", md.get("general.architecture"));
            assertEquals(LAYERS, ((Number) md.get("rnj1.block_count")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("rnj1.attention.key_length")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("rnj1.attention.value_length")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("rnj1.rope.dimension_count")).intValue());
            assertEquals(ROPE_BASE, ((Number) md.get("rnj1.rope.freq_base")).floatValue(), 1e-3f);
            assertEquals(ROPE_BASE_SWA, ((Number) md.get("rnj1.rope.freq_base_swa")).floatValue(), 1e-3f);
            assertEquals(1f / (float) Math.sqrt(QUERY_SCALAR),
                    ((Number) md.get("rnj1.attention.scale")).floatValue(), 1e-6f);
            assertEquals(CHUNK, ((Number) md.get("rnj1.attention.sliding_window")).intValue());
            assertEquals(java.util.Arrays.asList(false, true),
                    md.get("rnj1.attention.sliding_window_pattern"));
            assertEquals(SOFT_CAP, ((Number) md.get("rnj1.final_logit_softcapping")).floatValue(), 1e-6f);
            assertEquals(EPS, ((Number) md.get("rnj1.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);
            assertEquals(HEADS, ((Number) md.get("rnj1.attention.head_count")).intValue());
            assertEquals(HEADS, ((Number) md.get("rnj1.attention.head_count_kv")).intValue());

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.1.attn_k_norm.weight"));
            assertTrue(tensors.contains("blk.0.post_attention_norm.weight"));
            assertTrue(tensors.contains("blk.1.post_ffw_norm.weight"));
            assertTrue(tensors.contains("blk.1.ffn_norm.weight"));
            assertFalse("the head is tied to the token table", tensors.contains("output.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.RNJ1, spec.arch());
            assertEquals((float) Math.sqrt(D), spec.embedding().embdScale(), 1e-6f);
            assertEquals(SOFT_CAP, spec.head().finalSoftcap(), 1e-6f);
            assertTrue(spec.head().tied());
            LayerSpec full = spec.layer(0);
            assertEquals(SwaType.NONE, full.attention().effectiveSwaType());
            assertFalse(full.attention().isSwaLayer());
            assertEquals(ROPE_BASE, full.rope().freqBase(), 1e-3f);
            assertEquals(HEAD_DIM, full.rope().nDims());
            assertEquals(QkNormKind.RMS_HEAD, full.norm().qkNorm());
            assertTrue(full.norm().postAttnNorm());
            assertTrue(full.norm().postFfnNorm());
            LayerSpec chunked = spec.layer(1);
            assertEquals(SwaType.CHUNKED_LOOKBACK, chunked.attention().effectiveSwaType());
            assertTrue(chunked.attention().isSwaLayer());
            assertEquals(CHUNK, chunked.attention().nSwa());
            assertEquals(ROPE_BASE_SWA, chunked.rope().freqBase(), 1e-3f);
            assertEquals(1f / (float) Math.sqrt(QUERY_SCALAR), chunked.attention().kqScale(), 1e-6f);
            assertEquals(AttentionSpec.Temperature.NONE, chunked.attention().temperature());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        assertRejected(Collections.singletonMap("hidden_act", "'relu'"), "hidden_act");
        assertRejected(Collections.singletonMap("layer_types", "['full_attention','sliding_attention']"),
                "layer_types");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'full_attention':{'rope_type':'linear','factor':2.0,'rope_theta':10000.0},"
                        + "'chunked_attention':{'rope_theta':500000.0}}"), "rope_parameters");
        assertRejected(Collections.singletonMap("attn_logit_softcapping", "10.0"), "attn_logit_softcapping");
    }

    @Test
    public void refusesChunkedLayersWithoutAWindow() throws Exception {
        assertRejected(Collections.singletonMap("sliding_window", "null"), "sliding_window");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("rnj1").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(32).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    private SyntheticHf.Builder builder() {
        return builder(Collections.<String, String>emptyMap());
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int il = 0; il < LAYERS; il++) {
            String p = "model.layers." + il + ".";
            tensor(b, p + "input_layernorm.weight", 1, D, normWeight(2 + il, D));
            tensor(b, p + "post_attention_layernorm.weight", 1, D, normWeight(4 + il, D));
            tensor(b, p + "pre_feedforward_layernorm.weight", 1, D, normWeight(6 + il, D));
            tensor(b, p + "post_feedforward_layernorm.weight", 1, D, normWeight(8 + il, D));
            tensor(b, p + "self_attn.q_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, 10 + il));
            tensor(b, p + "self_attn.k_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, 12 + il));
            tensor(b, p + "self_attn.v_proj.weight", HEADS * HEAD_DIM, D, matrix(HEADS * HEAD_DIM, D, 14 + il));
            tensor(b, p + "self_attn.o_proj.weight", D, HEADS * HEAD_DIM, matrix(D, HEADS * HEAD_DIM, 16 + il));
            tensor(b, p + "self_attn.q_norm.weight", 1, HEAD_DIM, normWeight(18 + il, HEAD_DIM));
            tensor(b, p + "self_attn.k_norm.weight", 1, HEAD_DIM, normWeight(19 + il, HEAD_DIM));
            tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, 20 + il));
            tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, 22 + il));
            tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, 24 + il));
        }
        tensor(b, "model.norm.weight", 1, D, normWeight(26, D));
        return b;
    }

    /** The RNJ-1 {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['Rnj1ForCausalLM']");
        fields.put("model_type", "'rnj1'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("num_hidden_layers", Integer.toString(LAYERS));
        fields.put("num_attention_heads", Integer.toString(HEADS));
        fields.put("num_key_value_heads", Integer.toString(HEADS));
        fields.put("head_dim", Integer.toString(HEAD_DIM));
        fields.put("query_pre_attn_scalar", Float.toString(QUERY_SCALAR));
        fields.put("layer_types", "['full_attention', 'chunked_attention']");
        fields.put("sliding_window", Integer.toString(SLIDING_WINDOW));
        fields.put("rope_parameters", "{'full_attention':{'rope_theta':" + ROPE_BASE + "},"
                + "'chunked_attention':{'rope_theta':" + ROPE_BASE_SWA + "}}");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "128");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "true");
        fields.put("attention_bias", "false");
        fields.put("hidden_act", "'gelu_pytorch_tanh'");
        fields.put("hidden_activation", "'gelu_pytorch_tanh'");
        fields.put("final_logit_softcapping", Float.toString(SOFT_CAP));
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

    /** Effective Gemma norm weight; the checkpoint stores it offset by one, as the converter expects. */
    private static float[] normWeight(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            // stored raw = effective - 1, so the Gemma conversion's +1 restores the effective weight
            x[i] = 0.7f + 0.1f * i + 0.01f * seed - 1.0f;
        }
        return x;
    }

    /** The effective norm weight the reference uses (raw + 1). */
    private static float[] effectiveNorm(int seed, int n) {
        float[] x = normWeight(seed, n);
        for (int i = 0; i < n; i++) {
            x[i] += 1.0f;
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
        float inv = (float) (1.0 / Math.sqrt(meanSquare(x) + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = w == null ? x[i] * inv : x[i] * inv * w[i];
        }
        return y;
    }

    private static double meanSquare(float[] x) {
        double sum = 0;
        for (float v : x) {
            sum += v * v;
        }
        return sum / x.length;
    }

    private static float[] scaled(float[] x, float s) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * s;
        }
        return y;
    }

    /** HF's {@code rotate_half} over one head (vLLM's {@code is_neox_style=True}) at {@code position}. */
    private static void rotateHalf(float[] x, int off, int dim, int position, float base) {
        int half = dim / 2;
        float[] in = new float[dim];
        System.arraycopy(x, off, in, 0, dim);
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(base, -2.0 * i / dim);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[off + i] = in[i] * cos - in[i + half] * sin;
            x[off + i + half] = in[i + half] * cos + in[i] * sin;
        }
    }

    /** Whether the causal window of layer {@code il} shows key {@code j} to query {@code pos}. */
    private static boolean visible(int il, int pos, int j) {
        if (j > pos) {
            return false;
        }
        if (il == 0) {
            return true;
        }
        return j >= Math.max(0, (pos / CHUNK - 1) * CHUNK);
    }

    private static float[] reference(int[] tokens) {
        float[] embeddings = matrix(V, D, 1);
        float[][] hidden = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            hidden[t] = scaled(embeddingOf(embeddings, tokens[t]), (float) Math.sqrt(D));
        }
        float scale = 1f / (float) Math.sqrt(QUERY_SCALAR);
        for (int il = 0; il < LAYERS; il++) {
            float base = il == 0 ? ROPE_BASE : ROPE_BASE_SWA;
            float[] qNorm = effectiveNorm(18 + il, HEAD_DIM);
            float[] kNorm = effectiveNorm(19 + il, HEAD_DIM);
            float[][] keys = new float[tokens.length][], values = new float[tokens.length][];
            float[][] queries = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                float[] n = rmsNorm(hidden[t], effectiveNorm(2 + il, D));
                float[] q = linear(n, matrix(HEADS * HEAD_DIM, D, 10 + il), HEADS * HEAD_DIM);
                keys[t] = linear(n, matrix(HEADS * HEAD_DIM, D, 12 + il), HEADS * HEAD_DIM);
                values[t] = linear(n, matrix(HEADS * HEAD_DIM, D, 14 + il), HEADS * HEAD_DIM);
                for (int h = 0; h < HEADS; h++) {
                    normHead(q, h, qNorm);
                    normHead(keys[t], h, kNorm);
                    rotateHalf(q, h * HEAD_DIM, HEAD_DIM, t, base);
                    rotateHalf(keys[t], h * HEAD_DIM, HEAD_DIM, t, base);
                }
                queries[t] = q;
            }
            for (int t = 0; t < tokens.length; t++) {
                float[] attention = new float[HEADS * HEAD_DIM];
                for (int h = 0; h < HEADS; h++) {
                    int off = h * HEAD_DIM;
                    float[] weights = new float[t + 1];
                    float total = 0;
                    for (int j = 0; j <= t; j++) {
                        if (!visible(il, t, j)) {
                            continue;
                        }
                        float dot = 0;
                        for (int k = 0; k < HEAD_DIM; k++) {
                            dot += queries[t][off + k] * keys[j][off + k];
                        }
                        weights[j] = (float) Math.exp(dot * scale);
                        total += weights[j];
                    }
                    for (int j = 0; j <= t; j++) {
                        for (int k = 0; k < HEAD_DIM; k++) {
                            attention[off + k] += weights[j] / total * values[j][off + k];
                        }
                    }
                }
                float[] attnOut = rmsNorm(linear(attention, matrix(D, HEADS * HEAD_DIM, 16 + il), D),
                        effectiveNorm(4 + il, D));
                float[] stream = new float[D];
                for (int k = 0; k < D; k++) {
                    stream[k] = hidden[t][k] + attnOut[k];
                }
                float[] n2 = rmsNorm(stream, effectiveNorm(6 + il, D));
                float[] gate = linear(n2, matrix(FF, D, 20 + il), FF);
                float[] up = linear(n2, matrix(FF, D, 22 + il), FF);
                for (int k = 0; k < FF; k++) {
                    gate[k] = geluTanh(gate[k]) * up[k];
                }
                float[] ffnOut = rmsNorm(linear(gate, matrix(D, FF, 24 + il), D),
                        effectiveNorm(8 + il, D));
                for (int k = 0; k < D; k++) {
                    hidden[t][k] = stream[k] + ffnOut[k];
                }
            }
        }
        float[] out = rmsNorm(hidden[tokens.length - 1], effectiveNorm(26, D));
        float[] logits = linear(out, embeddings, V);
        if (SOFT_CAP > 0f) {
            for (int i = 0; i < V; i++) {
                logits[i] = SOFT_CAP * (float) Math.tanh(logits[i] / SOFT_CAP);
            }
        }
        return logits;
    }

    private static void normHead(float[] x, int head, float[] w) {
        int off = head * HEAD_DIM;
        float[] slice = new float[HEAD_DIM];
        System.arraycopy(x, off, slice, 0, HEAD_DIM);
        float[] normed = rmsNorm(slice, w);
        System.arraycopy(normed, 0, x, off, HEAD_DIM);
    }

    private static float geluTanh(float x) {
        return 0.5f * x * (1.0f + (float) Math.tanh(0.7978845608028654
                * x * (1.0f + 0.044715f * x * x)));
    }

    /** The embedding row of {@code token}, the stream every layer starts from. */
    private static float[] embeddingOf(float[] embeddings, int token) {
        float[] v = new float[D];
        System.arraycopy(embeddings, token * D, v, 0, D);
        return v;
    }
}
