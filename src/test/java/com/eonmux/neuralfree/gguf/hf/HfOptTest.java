package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.NormKind;
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

/**
 * OPT checkpoint loading, converted layout and causal inference against a scalar implementation of vLLM's forward
 * ({@code opt.py}): the position offset, the input and output projections and the pre- and post-LN variants.
 */
public class HfOptTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    /** Hidden width. */
    private static final int D = 4;
    /** Word width {@code word_embed_proj_dim}. */
    private static final int W = 3;
    private static final int V = 7;
    private static final int F = 5;
    private static final int N_POS = 16;
    private static final float EPS = 1e-5f;
    private static final int[] TOKENS = {2, 3, 4};

    @Test
    public void preLayerNormCheckpointMatchesVllmForwardOnTheCpu() throws Exception {
        assertForward(preCheckpoint(), false);
    }

    /** The 350M post-LN variant: both block norms apply after their residual adds, with the projections. */
    @Test
    public void postLayerNormCheckpointMatchesVllmForwardOnTheCpu() throws Exception {
        assertForward(postCheckpoint(), true);
    }

    @Test
    public void preLayerNormCheckpointMatchesVllmForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertForwardOnGpu(preCheckpoint(), false);
    }

    @Test
    public void postLayerNormCheckpointMatchesVllmForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertForwardOnGpu(postCheckpoint(), true);
    }

    /** OPT is a supported text architecture, listed for {@code nftool export --print-supported-models}. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("OPTForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.OPT);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NONE, traits.ropeType());
        assertEquals(NormKind.LAYER, traits.normKind());
        assertEquals(ActOp.RELU, traits.ffnAct());
    }

    /** The exported pre-LN GGUF file loads through the GGUF source and runs the same forward. */
    @Test
    public void exportedPreLayerNormGgufLoadsAsOpt() throws Exception {
        assertExportRoundTrip(preCheckpoint(), false);
    }

    /** The exported 350M-style file keeps the post-LN placement and the projections through the round trip. */
    @Test
    public void exportedPostLayerNormGgufLoadsAsOpt() throws Exception {
        assertExportRoundTrip(postCheckpoint(), true);
    }

    /** The converter writes the pre-LN OPT layout, including the projections and the dropped mask buffers. */
    @Test
    public void conversionWritesThePreLayerNormLayout() throws Exception {
        Path dir = preCheckpoint();
        try (ModelSource src = ModelSources.open(dir.toString(), options("layout-pre"), null)) {
            ModelSpec spec = spec(src);
            assertEquals(Arch.OPT, spec.arch());
            LayerSpec layer = spec.layer(0);
            assertEquals(2, spec.embedding().posOffset());
            assertNotNull("word_embed_proj_dim needs project_in", spec.embedding().inputProjection());
            assertNotNull("word_embed_proj_dim needs project_out", spec.head().outputProjection());
            assertEquals(W, spec.embedding().tokenWidth());
            assertTrue(layer.norm().attnNorm());
            assertTrue(layer.norm().ffnNorm());
            assertEquals(PostNormRole.NONE, layer.norm().postRole());
            Set<String> names = tensorNames(src);
            assertTrue(names.contains("position_embd.weight"));
            assertTrue(names.contains("project_in.weight"));
            assertTrue(names.contains("project_out.weight"));
            assertTrue(names.contains("blk.0.attn_norm.weight"));
            assertTrue(names.contains("blk.0.ffn_norm.weight"));
            assertTrue(names.contains("blk.0.ffn_up.weight"));
            assertTrue(names.contains("blk.0.ffn_down.weight"));
            assertFalse("the causal-mask buffers are dropped", names.contains("blk.0.self_attn.bias"));
        }
    }

    /** The 350M layout names both block norms after the residual adds. */
    @Test
    public void conversionWritesThePostLayerNormLayout() throws Exception {
        Path dir = postCheckpoint();
        try (ModelSource src = ModelSources.open(dir.toString(), options("layout-post"), null)) {
            ModelSpec spec = spec(src);
            LayerSpec layer = spec.layer(0);
            assertFalse(layer.norm().attnNorm());
            assertFalse(layer.norm().ffnNorm());
            assertTrue(layer.norm().attnOutputNorm());
            assertTrue(layer.norm().layerOutputNorm());
            assertNull("the post-LN decoder has no final norm", spec.head().outputNorm());
            Set<String> names = tensorNames(src);
            assertTrue(names.contains("blk.0.attn_output_norm.weight"));
            assertTrue(names.contains("blk.0.layer_output_norm.weight"));
            assertFalse(names.contains("blk.0.attn_norm.weight"));
            assertFalse(names.contains("output_norm.weight"));
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("activation_function", "'gelu'"), "activation_function");
        assertRejected(Collections.singletonMap("layer_norm_elementwise_affine", "false"),
                "layer_norm_elementwise_affine");
        assertRejected(Collections.singletonMap("num_attention_heads", "3"), "num_attention_heads");
        // A stored decoder final norm the graph would not have, and a missing or surplus projection pair.
        assertRejected(Collections.singletonMap("_remove_final_layer_norm", "true"), "final_layer_norm");
        assertRejected(Collections.singletonMap("do_layer_norm_before", "false"), "final_layer_norm");
        assertRejected(Collections.singletonMap("word_embed_proj_dim", Integer.toString(D)), "word_embed_proj_dim");
    }

    /** A checkpoint that ties the head to the token table and drops the decoder's final norm. */
    @Test
    public void tiedHeadWithoutFinalNormMatchesVllmForward() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap(), true, true, false)
                .write(tmp.newFolder("opt-tied").toPath());
        try (GGUFModel model = new GGUFModel(dir.toString(), cpuConfig())) {
            assertTrue("the missing lm_head is tied to the token table", model.spec().head().tied());
            assertNull(model.spec().head().outputNorm());
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(false, true, false), actual, 2e-4f);
        }
    }

    /** Most OPT sizes keep {@code word_embed_proj_dim == hidden_size}, so neither projection exists. */
    @Test
    public void equalWidthCheckpointMatchesVllmForward() throws Exception {
        Path dir = plainCheckpoint();
        try (GGUFModel model = new GGUFModel(dir.toString(), cpuConfig())) {
            assertNull(model.spec().embedding().inputProjection());
            assertNull(model.spec().head().outputProjection());
            assertEquals(D, model.spec().embedding().tokenWidth());
            assertArrayEquals(referencePlain(false), model.forwardBackend().prefill(TOKENS, 0).clone(), 2e-4f);
        }
    }

    /** A position past the learned table (the two padding rows included) is refused by the embedding stage. */
    @Test
    public void refusesPositionsPastTheLearnedTable() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap(), true)
                .write(tmp.newFolder("positions").toPath());
        try (GGUFModel model = new GGUFModel(dir.toString(), cpuConfig())) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> model.forwardBackend().prefill(new int[]{2, 3, 4}, N_POS - 2));
            assertTrue(e.getMessage(), e.getMessage().contains("position"));
        }
    }

    private void assertForward(Path path, boolean postLn) throws Exception {
        try (GGUFModel model = new GGUFModel(path.toString(), cpuConfig())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(postLn), actual, 2e-4f);
        }
    }

    private void assertForwardOnGpu(Path path, boolean postLn) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(32).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(postLn), actual, 2e-4f);
        }
    }

    private void assertExportRoundTrip(Path dir, boolean postLn) throws Exception {
        Path out = tmp.getRoot().toPath().resolve("opt-" + postLn + ".gguf");
        try (HfModelSource src = HfModelSource.open(dir, options("export-" + postLn), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpuConfig())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(postLn), actual, 2e-4f);
        }
    }

    private static EngineConfig cpuConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(4).build()).build();
    }

    private static ModelSourceOptions options(String cache) throws Exception {
        return ModelSourceOptions.builder()
                .cacheDir(java.nio.file.Files.createTempDirectory(cache)).build();
    }

    private static ModelSpec spec(ModelSource src) {
        ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
        TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
        return spec;
    }

    private static Set<String> tensorNames(ModelSource src) {
        Set<String> names = new HashSet<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            names.add(t.name());
        }
        return names;
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        Path dir = builder(overrides, true).write(tmp.newFolder().toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), options("rejected-" + field), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    // ------------------------------------------------------------------ checkpoints

    private Path preCheckpoint() throws Exception {
        return builder(Collections.<String, String>emptyMap(), true).write(tmp.newFolder("opt-pre").toPath());
    }

    private Path postCheckpoint() throws Exception {
        return builder(Collections.<String, String>emptyMap(), false).write(tmp.newFolder("opt-post").toPath());
    }

    private Path plainCheckpoint() throws Exception {
        return plainBuilder(Collections.<String, String>emptyMap(), true)
                .write(tmp.newFolder("opt-plain").toPath());
    }

    /** A one-layer OPT checkpoint; {@code preLayerNorm} selects the 350M variant when false. */
    private static SyntheticHf.Builder builder(Map<String, String> overrides, boolean preLayerNorm) {
        return builder(overrides, preLayerNorm, false, true);
    }

    /**
     * A one-layer OPT checkpoint; {@code tied} drops the LM head (the token table serves it) and {@code finalNorm}
     * false drops the decoder's final LayerNorm.
     */
    private static SyntheticHf.Builder builder(Map<String, String> overrides, boolean preLayerNorm, boolean tied,
                                               boolean finalNorm) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides, preLayerNorm, tied, finalNorm));
        tensor(b, "model.decoder.embed_tokens.weight", V, W, matrix(V, W, 1));
        tensor(b, "model.decoder.embed_positions.weight", N_POS + 2, D, matrix(N_POS + 2, D, 2));
        tensor(b, "model.decoder.project_in.weight", D, W, matrix(D, W, 3));
        tensor(b, "model.decoder.project_out.weight", W, D, matrix(W, D, 4));
        if (preLayerNorm && finalNorm) {
            tensor(b, "model.decoder.final_layer_norm.weight", D, 1, ones());
            tensor(b, "model.decoder.final_layer_norm.bias", D, 1, bias(4));
        }
        optLayerTensors(b);
        if (!tied) {
            tensor(b, "lm_head.weight", V, W, matrix(V, W, 8));
        }
        return b;
    }

    /** A one-layer OPT checkpoint whose word width equals the hidden width (most OPT sizes). */
    private static SyntheticHf.Builder plainBuilder(Map<String, String> overrides, boolean preLayerNorm) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(plainConfig(overrides, preLayerNorm));
        tensor(b, "model.decoder.embed_tokens.weight", V, D, matrix(V, D, 1));
        tensor(b, "model.decoder.embed_positions.weight", N_POS + 2, D, matrix(N_POS + 2, D, 2));
        tensor(b, "model.decoder.final_layer_norm.weight", D, 1, ones());
        tensor(b, "model.decoder.final_layer_norm.bias", D, 1, bias(4));
        optLayerTensors(b);
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 8));
        return b;
    }

    /** The block tensors both OPT variants share: norms, ReLU MLP, biased projections and a mask buffer. */
    private static void optLayerTensors(SyntheticHf.Builder b) {
        String prefix = "model.decoder.layers.0.";
        tensor(b, prefix + "self_attn_layer_norm.weight", D, 1, ones());
        tensor(b, prefix + "self_attn_layer_norm.bias", D, 1, bias(1));
        tensor(b, prefix + "fc1.weight", F, D, matrix(F, D, 6));
        tensor(b, prefix + "fc1.bias", F, 1, bias(2, F));
        tensor(b, prefix + "fc2.weight", D, F, matrix(D, F, 7));
        tensor(b, prefix + "fc2.bias", D, 1, bias(3));
        tensor(b, prefix + "final_layer_norm.weight", D, 1, ones());
        tensor(b, prefix + "final_layer_norm.bias", D, 1, bias(2));
        String[] names = {"q_proj", "k_proj", "v_proj", "out_proj"};
        for (int i = 0; i < names.length; i++) {
            tensor(b, prefix + "self_attn." + names[i] + ".weight", D, D, matrix(D, D, i + 2));
            tensor(b, prefix + "self_attn." + names[i] + ".bias", D, 1, bias(i + 5));
        }
        b.tensor(prefix + "self_attn.bias", StDType.BOOL, new long[]{1, 1, N_POS}, new byte[N_POS]);
    }

    /** The OPT {@code config.json}; {@code overrides} replace values (raw JSON). */
    private static String config(Map<String, String> overrides, boolean preLayerNorm) {
        return config(overrides, preLayerNorm, false, true);
    }

    private static String plainConfig(Map<String, String> overrides, boolean preLayerNorm) {
        return config(overrides, preLayerNorm, false, true, D);
    }

    private static String config(Map<String, String> overrides, boolean preLayerNorm, boolean tied,
                                 boolean finalNorm) {
        return config(overrides, preLayerNorm, tied, finalNorm, W);
    }

    private static String config(Map<String, String> overrides, boolean preLayerNorm, boolean tied,
                                 boolean finalNorm, int wordWidth) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['OPTForCausalLM']");
        fields.put("model_type", "'opt'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("word_embed_proj_dim", Integer.toString(wordWidth));
        fields.put("ffn_dim", Integer.toString(F));
        fields.put("num_hidden_layers", "1");
        fields.put("num_attention_heads", "1");
        fields.put("max_position_embeddings", Integer.toString(N_POS));
        fields.put("layer_norm_eps", "1e-5");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("activation_function", "'relu'");
        fields.put("do_layer_norm_before", Boolean.toString(preLayerNorm));
        fields.put("layer_norm_elementwise_affine", "true");
        fields.put("enable_bias", "true");
        fields.put("tie_word_embeddings", Boolean.toString(tied));
        fields.put("_remove_final_layer_norm", Boolean.toString(!finalNorm));
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

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    // ------------------------------------------------------------------ scalar reference

    private static float[] reference(boolean postLn) {
        return reference(postLn, false, true);
    }

    private static float[] reference(boolean postLn, boolean tied, boolean finalNorm) {
        return reference(postLn, finalNorm, matrix(V, W, 1), W, matrix(D, W, 3), matrix(W, D, 4),
                tied ? null : matrix(V, W, 8));
    }

    /** The {@code word_embed_proj_dim == hidden_size} variant: the token table is already {@code n_embd} wide. */
    private static float[] referencePlain(boolean postLn) {
        return reference(postLn, true, matrix(V, D, 1), D, null, null, matrix(V, D, 8));
    }

    private static float[] reference(boolean postLn, boolean finalNorm, float[] tok, int tokenWidth, float[] pin,
                                     float[] pout, float[] lm) {
        float[] pos = matrix(N_POS + 2, D, 2);
        float[] q = matrix(D, D, 2), k = matrix(D, D, 3), v = matrix(D, D, 4), o = matrix(D, D, 5);
        float[] fc1 = matrix(F, D, 6), fc2 = matrix(D, F, 7);
        float[][] keys = new float[TOKENS.length][];
        float[][] values = new float[TOKENS.length][];
        float[] hidden = null;
        for (int t = 0; t < TOKENS.length; t++) {
            float[] e = row(tok, TOKENS[t], tokenWidth);
            float[] projected = pin == null ? e : linear(e, pin, null, D);
            for (int i = 0; i < D; i++) {
                projected[i] += pos[(t + 2) * D + i];              // OPT's two padding rows
            }
            float[] x = projected;
            if (postLn) {
                keys[t] = linear(x, k, bias(6), D);
                values[t] = linear(x, v, bias(7), D);
                float[] attn = attention(linear(x, q, bias(5), D), keys, values, t);
                x = layerNorm(add(x, linear(attn, o, bias(8), D)), bias(1));
                float[] ff = relu(linear(x, fc1, bias(2, F), F));
                x = layerNorm(add(x, linear(ff, fc2, bias(3), D)), bias(2));
            } else {
                float[] n = layerNorm(x, bias(1));
                keys[t] = linear(n, k, bias(6), D);
                values[t] = linear(n, v, bias(7), D);
                float[] attn = attention(linear(n, q, bias(5), D), keys, values, t);
                x = add(x, linear(attn, o, bias(8), D));
                float[] n2 = layerNorm(x, bias(2));
                float[] ff = relu(linear(n2, fc1, bias(2, F), F));
                x = add(x, linear(ff, fc2, bias(3), D));
            }
            hidden = x;
        }
        if (!postLn && finalNorm) {
            hidden = layerNorm(hidden, bias(4));
        }
        float[] out = pout == null ? hidden : linear(hidden, pout, null, W);
        return lm == null ? linear(out, tok, null, V) : linear(out, lm, null, V);
    }

    private static float[] attention(float[] q, float[][] keys, float[][] values, int t) {
        float[] weights = new float[t + 1];
        float total = 0;
        for (int j = 0; j <= t; j++) {
            float dot = 0;
            for (int i = 0; i < D; i++) {
                dot += q[i] * keys[j][i];
            }
            weights[j] = (float) Math.exp(dot / Math.sqrt(D));
            total += weights[j];
        }
        float[] out = new float[D];
        for (int j = 0; j <= t; j++) {
            for (int i = 0; i < D; i++) {
                out[i] += weights[j] / total * values[j][i];
            }
        }
        return out;
    }

    private static float[] add(float[] a, float[] b) {
        float[] y = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    private static float[] relu(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = Math.max(0f, x[i]);
        }
        return y;
    }

    private static float[] layerNorm(float[] x, float[] bias) {
        float mean = 0;
        for (float value : x) {
            mean += value / x.length;
        }
        float variance = 0;
        for (float value : x) {
            variance += (value - mean) * (value - mean) / x.length;
        }
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) ((x[i] - mean) / Math.sqrt(variance + EPS)) + bias[i];
        }
        return y;
    }

    private static float[] row(float[] m, int r, int cols) {
        float[] y = new float[cols];
        System.arraycopy(m, r * cols, y, 0, cols);
        return y;
    }

    private static float[] linear(float[] x, float[] w, float[] bias, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            y[i] = bias == null ? 0 : bias[i];
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
        }
        return y;
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.035f;
        }
        return x;
    }

    private static float[] bias(int seed) {
        return bias(seed, D);
    }

    private static float[] bias(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (i - seed) * 0.025f;
        }
        return x;
    }

    private static float[] ones() {
        return new float[]{1, 1, 1, 1};
    }
}
