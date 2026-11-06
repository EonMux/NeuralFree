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

/** DomynEdge checkpoint loading, the per-layer geometry and causal inference against a scalar reference. */
public class HfDomynEdgeTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private static final int D = 4, V = 7, FF = 4;
    /** Layer 0 is a full-attention NoPE layer of four-wide heads; layer 1 a RoPE layer of two-wide heads, window 1. */
    private static final int[] HEADS = {4, 2};
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

    /** The same forward through the OpenCL backend: the per-layer head widths and the window included. */
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

    /** DomynEdge is a supported text architecture with the sandwich norms. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("DomynEdgeForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.DOMYN_EDGE);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(PostNormRole.SANDWICH, traits.postNormRole());
    }

    /** The exported GGUF file (NeuralFree's {@code domyn-edge} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsDomynEdge() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("domyn-edge.gguf");
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

    /** The converter writes both head widths, the shared window, the per-layer switches and the tie. */
    @Test
    public void conversionWritesTheDomynEdgeLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("domyn-edge", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("domyn-edge.block_count")).intValue());
            assertEquals(4, ((Number) md.get("domyn-edge.attention.key_length")).intValue());
            assertEquals(2, ((Number) md.get("domyn-edge.attention.key_length_swa")).intValue());
            assertEquals(2, ((Number) md.get("domyn-edge.rope.dimension_count")).intValue());
            assertEquals(2, ((Number) md.get("domyn-edge.attention.sliding_window")).intValue());
            assertEquals(java.util.Arrays.asList(false, true),
                    md.get("domyn-edge.attention.sliding_window_pattern"));
            assertEquals(java.util.Arrays.asList(false, true), md.get("domyn-edge.attention.rope_pattern"));
            assertEquals(EPS, ((Number) md.get("domyn-edge.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_k_norm.weight"));
            assertTrue(tensors.contains("blk.1.attn_k_norm.weight"));
            assertFalse("Q's per-head norm has no weight in the checkpoint",
                    tensors.contains("blk.0.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.0.post_attention_norm.weight"));
            assertTrue(tensors.contains("blk.1.post_ffw_norm.weight"));
            assertTrue(tensors.contains("blk.1.ffn_down.weight"));
            assertFalse("the head is tied to the token table", tensors.contains("output.weight"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.DOMYN_EDGE, spec.arch());
            LayerSpec full = spec.layer(0);
            assertFalse(full.rope() != null && full.rope().nDims() > 0);
            assertEquals(QkNormKind.RMS_HEAD, full.norm().qkNorm());
            assertFalse(full.attention().isSwaLayer());
            LayerSpec swa = spec.layer(1);
            assertNotNull(swa.rope());
            assertEquals(2, swa.rope().nDims());
            assertTrue(swa.attention().isSwaLayer());
            assertEquals(2, swa.attention().nSwa());
            assertEquals((float) (1.0 / Math.sqrt(2.0)), swa.attention().kqScale(), 1e-6);
            assertTrue(spec.head().tied());
            assertTrue(swa.norm().postAttnNorm());
            assertTrue(swa.norm().postFfnNorm());
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        // Both layers sliding: their head sizes (4 and 2) then clash in one class.
        assertRejected(Collections.singletonMap("window_sizes", "[[1, 0], [1, 0]]"), "head_sizes");
        assertRejected(Collections.singletonMap("position_layers", "['nope', 'absolute']"), "position_layers");
        assertRejected(Collections.singletonMap("window_sizes", "[[-1, 0], [1, 2]]"), "window_sizes");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("domyn-edge").toPath());
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
        for (int il = 0; il < 2; il++) {
            int h = HEADS[il];
            String p = "model.layers." + il + ".";
            tensor(b, p + "pre_attention_norm.weight", 1, D, normWeight(2 + il));
            tensor(b, p + "post_attention_norm.weight", 1, D, normWeight(4 + il));
            tensor(b, p + "pre_feedforward_norm.weight", 1, D, normWeight(6 + il));
            tensor(b, p + "post_feedforward_norm.weight", 1, D, normWeight(8 + il));
            tensor(b, p + "attention.q_proj.weight", h, D, matrix(h, D, 10 + il));
            tensor(b, p + "attention.k_proj.weight", h, D, matrix(h, D, 12 + il));
            tensor(b, p + "attention.v_proj.weight", h, D, matrix(h, D, 14 + il));
            tensor(b, p + "attention.o_proj.weight", D, h, matrix(D, h, 16 + il));
            tensor(b, p + "attention.k_norm.weight", 1, h, normWeight(18 + il, h));
            tensor(b, p + "feedforward.gate_proj.weight", FF, D, matrix(FF, D, 20 + il));
            tensor(b, p + "feedforward.up_proj.weight", FF, D, matrix(FF, D, 22 + il));
            tensor(b, p + "feedforward.down_proj.weight", D, FF, matrix(D, FF, 24 + il));
        }
        tensor(b, "model.norm.weight", 1, D, normWeight(26));
        return b;
    }

    /** The DomynEdge {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['DomynEdgeForCausalLM']");
        fields.put("model_type", "'domynedge'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("num_hidden_layers", "2");
        fields.put("num_attention_heads", "1");
        fields.put("num_key_value_heads", "1");
        fields.put("head_sizes", "[4, 2]");
        fields.put("window_sizes", "[[-1, 0], [1, 0]]");
        fields.put("position_layers", "['nope', 'rope']");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("rope_theta", "10000.0");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "true");
        fields.put("attention_bias", "false");
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
        return normWeight(seed, D);
    }

    private static float[] normWeight(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
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

    /** Whether the causal window of layer {@code il} shows key {@code j} to query {@code pos}. */
    private static boolean visible(int il, int pos, int j) {
        return il == 0 || pos - j < 2;
    }

    private static float[] reference(int[] tokens) {
        float[] embeddings = matrix(V, D, 1);
        float[][] hidden = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            hidden[t] = embeddingOf(embeddings, tokens[t]);
        }
        for (int il = 0; il < 2; il++) {
            int h = HEADS[il];
            float sheet = (float) (1.0 / Math.sqrt(h));
            float[] kNorm = normWeight(18 + il, h);
            float[][] keys = new float[tokens.length][], values = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                float[] n = rmsNorm(hidden[t], normWeight(2 + il));
                float[] q = rmsNorm(linear(n, matrix(h, D, 10 + il), h), null);
                keys[t] = rmsNorm(linear(n, matrix(h, D, 12 + il), h), kNorm);
                values[t] = linear(n, matrix(h, D, 14 + il), h);
                if (il == 1) {
                    rotateHalf(q, t);
                    rotateHalf(keys[t], t);
                }
                float[] weights = new float[t + 1];
                float total = 0;
                for (int j = 0; j <= t; j++) {
                    if (!visible(il, t, j)) {
                        continue;
                    }
                    float dot = 0;
                    for (int k = 0; k < h; k++) {
                        dot += q[k] * keys[j][k];
                    }
                    weights[j] = (float) Math.exp(dot * sheet);
                    total += weights[j];
                }
                float[] attention = new float[h];
                for (int j = 0; j <= t; j++) {
                    for (int k = 0; k < h; k++) {
                        attention[k] += weights[j] / total * values[j][k];
                    }
                }
                float[] attnOut = rmsNorm(linear(attention, matrix(D, h, 16 + il), D), normWeight(4 + il));
                float[] stream = new float[D];
                for (int k = 0; k < D; k++) {
                    stream[k] = hidden[t][k] + attnOut[k];
                }
                float[] n2 = rmsNorm(stream, normWeight(6 + il));
                float[] gate = linear(n2, matrix(FF, D, 20 + il), FF);
                float[] up = linear(n2, matrix(FF, D, 22 + il), FF);
                for (int k = 0; k < FF; k++) {
                    gate[k] = (float) (gate[k] / (1 + Math.exp(-gate[k]))) * up[k];
                }
                float[] ffnOut = rmsNorm(linear(gate, matrix(D, FF, 24 + il), D), normWeight(8 + il));
                for (int k = 0; k < D; k++) {
                    hidden[t][k] = stream[k] + ffnOut[k];
                }
            }
        }
        return linear(rmsNorm(hidden[tokens.length - 1], normWeight(26)), embeddings, V);
    }

    /** The embedding row of {@code token}, the stream every layer starts from. */
    private static float[] embeddingOf(float[] embeddings, int token) {
        float[] v = new float[D];
        System.arraycopy(embeddings, token * D, v, 0, D);
        return v;
    }
}
