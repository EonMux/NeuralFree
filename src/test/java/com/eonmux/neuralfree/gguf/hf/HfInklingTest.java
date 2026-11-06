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
import com.eonmux.neuralfree.gguf.arch.SwaType;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
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
 * Inkling checkpoint loading, the NoPE relative-bias attention with its K/V short convolutions and residual-stream
 * short convolutions, and causal inference against a scalar reference of vLLM's forward.
 */
public class HfInklingTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 8;
    private static final int V = 7;
    /** The checkpoint pads the vocabulary to nine rows; vLLM slices the extra logits off. */
    private static final int PADDED_V = 9;
    private static final int FF = 8;
    private static final int HEADS = 2;
    private static final int KV_HEADS = 1;
    private static final int HEAD_DIM = 4;
    private static final int SWA_HEADS = 2;
    private static final int SWA_KV_HEADS = 1;
    private static final int SWA_HEAD_DIM = 2;
    private static final int D_REL = 2;
    private static final int REL_EXTENT = 3;
    private static final int LAYERS = 2;
    private static final int LOCAL_LAYER = 1;
    /** {@code sliding_window_size}: vLLM's local window is {@code (local_extent - 1, 0)}. */
    private static final int WINDOW = 2;
    private static final int CONV_WIDTH = 2;
    private static final int N_FLOOR = 2;
    private static final float ALPHA = 0.3f;
    private static final float EPS = 1e-5f;
    private static final float GLOBAL_SCALE = 0.5f;
    private static final float MUP = 1.5f;
    private static final int[] TOKENS = {2, 3, 4, 5, 6, 0};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertMaskedPadding(actual);
            assertArrayEquals(reference(TOKENS), head(actual), 2e-4f);
        }
    }

    /** The conv states carry across steps: chunked prefill plus decode matches one prefill. */
    @Test
    public void incrementalDecodeMatchesPrefill() throws Exception {
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(new int[]{TOKENS[0], TOKENS[1]}, 0);
            for (int t = 2; t < TOKENS.length; t++) {
                float[] actual = model.forwardBackend().decode(TOKENS[t], t).clone();
                assertMaskedPadding(actual);
                assertArrayEquals("position " + t, reference(prefix(t + 1)), head(actual), 2e-4f);
            }
        }
    }

    /** The same forward through OpenCL: the relative bias, the four conv streams and the muP divisor included. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), gpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertMaskedPadding(actual);
            assertArrayEquals(reference(TOKENS), head(actual), 2e-4f);
        }
    }

    /** The GPU conv state follows the paged sequence rows across a decode step. */
    @Test
    public void incrementalDecodeMatchesPrefillOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint(Collections.<String, String>emptyMap());
        try (GGUFModel model = new GGUFModel(path.toString(), gpu())) {
            model.forwardBackend().prefill(new int[]{TOKENS[0], TOKENS[1]}, 0);
            for (int t = 2; t < TOKENS.length; t++) {
                float[] actual = model.forwardBackend().decode(TOKENS[t], t).clone();
                assertMaskedPadding(actual);
                assertArrayEquals("position " + t, reference(prefix(t + 1)), head(actual), 2e-4f);
            }
        }
    }

    /** Inkling is a supported text architecture with NoPE attention and its own GGUF name. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("InklingForCausalLM"));
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT)
                .contains("InklingForConditionalGeneration"));
        ArchTraits traits = ArchRegistry.traits(Arch.INKLING);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NONE, traits.ropeType());
        assertEquals("inkling", Arch.INKLING.ggufName());
    }

    /** The exported GGUF file (NeuralFree's {@code inkling} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsInkling() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("inkling.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertMaskedPadding(actual);
            assertArrayEquals(reference(TOKENS), head(actual), 2e-4f);
        }
    }

    /** The converter writes the split Inkling tensors, the per-layer geometry and the scaling keys. */
    @Test
    public void conversionWritesTheInklingLayout() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap()).write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("inkling", md.get("general.architecture"));
            assertEquals(LAYERS, ((Number) md.get("inkling.block_count")).intValue());
            assertEquals(REL_EXTENT, ((Number) md.get("inkling.inkling.relative_extent")).intValue());
            assertEquals(D_REL, ((Number) md.get("inkling.inkling.relative_dim")).intValue());
            assertEquals(CONV_WIDTH, ((Number) md.get("inkling.inkling.sconv_kernel_size")).intValue());
            assertEquals(N_FLOOR, ((Number) md.get("inkling.inkling.log_scaling_n_floor")).intValue());
            assertEquals(ALPHA, ((Number) md.get("inkling.inkling.log_scaling_alpha")).floatValue(), 1e-6f);
            assertEquals(MUP, ((Number) md.get("inkling.logit_scale")).floatValue(), 1e-6f);
            assertEquals(WINDOW, ((Number) md.get("inkling.attention.sliding_window")).intValue());
            assertEquals(java.util.Arrays.asList(false, true),
                    md.get("inkling.attention.sliding_window_pattern"));
            assertEquals(HEAD_DIM, ((Number) md.get("inkling.attention.key_length")).intValue());
            assertEquals(SWA_HEAD_DIM, ((Number) md.get("inkling.attention.key_length_swa")).intValue());
            assertEquals(V, ((Number) md.get("inkling.vocab_size")).intValue());
            assertEquals(FF, ((Number) md.get("inkling.feed_forward_length")).intValue());
            assertEquals(EPS, ((Number) md.get("inkling.attention.layer_norm_rms_epsilon")).doubleValue(), 1e-9);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("token_embd_norm.weight"));
            assertTrue(tensors.contains("output.weight"));
            for (String suffix : new String[]{"attn_q", "attn_k", "attn_v", "attn_rel", "attn_output",
                    "attn_rel_proj", "attn_q_norm", "attn_k_norm", "attn_sconv_k", "attn_sconv_v", "attn_sconv",
                    "ffn_sconv", "ffn_gate", "ffn_up", "ffn_down", "ffn_global_scale"}) {
                assertTrue("blk.1." + suffix, tensors.contains("blk.1." + suffix + ".weight"));
            }

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.INKLING, spec.arch());
            assertTrue(spec.embedding().inputRmsNorm());
            assertNotNull(spec.embedding().inputRmsNormWeight());

            LayerSpec full = spec.layer(0);
            AttentionSpec a0 = full.attention();
            assertEquals(SwaType.NONE, a0.effectiveSwaType());
            assertTrue(a0.noRope());
            assertEquals(0, a0.nRot());
            assertEquals(AttentionSpec.BiasMode.INKLING, a0.biasMode());
            assertEquals(REL_EXTENT, a0.relBiasExtent());
            assertEquals(1f / HEAD_DIM, a0.kqScale(), 1e-6f);
            assertTrue(a0.inBias());
            assertTrue(a0.outBias());

            LayerSpec local = spec.layer(LOCAL_LAYER);
            assertTrue(local.attention().isSwaLayer());
            assertEquals(WINDOW, local.attention().nSwa());
            assertEquals(WINDOW, local.attention().relBiasExtent());
            assertEquals(SWA_HEAD_DIM, local.attention().headDimK());
            assertEquals(CONV_WIDTH, local.recurrent().dConv());
        }
    }

    /** The multimodal towers of {@code InklingForConditionalGeneration} are dropped, leaving the text backbone. */
    @Test
    public void multimodalCheckpointConvertsItsTextBackbone() throws Exception {
        SyntheticHf.Builder b = builder(Collections.singletonMap("architectures", "['InklingForConditionalGeneration']"));
        tensor(b, "model.visual.patch_embed.proj.weight", 2, D, matrix(2, D, 50));
        tensor(b, "model.audio.frontend.weight", 2, D, matrix(2, D, 52));
        Path dir = b.write(tmp.newFolder("multimodal").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("multimodal-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertFalse(tensors.contains("model.visual.patch_embed.proj.weight"));
            assertFalse(tensors.contains("model.audio.frontend.weight"));
            assertTrue(tensors.contains("blk.0.attn_q.weight"));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), cpu())) {
            assertArrayEquals(reference(TOKENS), head(model.forwardBackend().prefill(TOKENS, 0).clone()), 2e-4f);
        }
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        for (String field : new String[]{"dense_mlp_idx", "num_nextn_predict_layers", "tie_word_embeddings",
                "hidden_act", "chain_hidden_post_norm", "sliding_window_size", "local_layer_ids", "use_global_scale",
                "q_bias", "o_bias", "logits_mup_width_multiplier", "final_logit_softcapping", "sconv_kernel_size",
                "d_rel", "use_embed_norm"}) {
            assertRejected(field);
        }
    }

    private void assertRejected(String field) throws Exception {
        Map<String, String> overrides = rejection(field);
        Path dir = builder(overrides).write(tmp.newFolder("rejected-" + field).toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), options, null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    /** The configuration change that makes the converter refuse {@code field}. */
    private static Map<String, String> rejection(String field) {
        Map<String, String> m = new LinkedHashMap<>();
        switch (field) {
            case "dense_mlp_idx":
                m.put("n_routed_experts", "4");
                m.put("dense_mlp_idx", "1");
                break;
            case "num_nextn_predict_layers":
                m.put("num_nextn_predict_layers", "1");
                break;
            case "tie_word_embeddings":
                m.put("tie_word_embeddings", "true");
                break;
            case "sliding_window_size":
                m.put("sliding_window_size", "0");
                break;
            case "local_layer_ids":
                m.put("local_layer_ids", "[5]");
                break;
            case "use_global_scale":
                m.put("use_global_scale", "false");
                break;
            case "q_bias":
                m.put("q_bias", "false");
                break;
            case "o_bias":
                m.put("o_bias", "false");
                break;
            case "hidden_act":
                m.put("hidden_act", "'gelu'");
                break;
            case "chain_hidden_post_norm":
                m.put("chain_hidden_post_norm", "true");
                break;
            case "logits_mup_width_multiplier":
                m.put("final_logit_softcapping", "8.0");
                break;
            case "sconv_kernel_size":
                m.put("sconv_kernel_size", "1");
                break;
            case "d_rel":
                m.put("d_rel", "0");
                break;
            case "final_logit_softcapping":
                m.put("final_logit_softcapping", "-1.0");
                break;
            case "use_embed_norm":
                m.put("use_embed_norm", "false");
                break;
            default:
                throw new IllegalArgumentException(field);
        }
        return m;
    }

    /** The last {@link #V} logits of a padded row. */
    private static float[] head(float[] logits) {
        assertEquals(PADDED_V, logits.length);
        float[] out = new float[V];
        System.arraycopy(logits, 0, out, 0, V);
        return out;
    }

    /** vLLM's logits processor slices the padded rows off; the graph masks them instead. */
    private static void assertMaskedPadding(float[] logits) {
        for (int i = V; i < logits.length; i++) {
            assertEquals("row " + i, -Float.MAX_VALUE, logits[i], 0f);
        }
    }

    private static int[] prefix(int n) {
        int[] out = new int[n];
        System.arraycopy(TOKENS, 0, out, 0, n);
        return out;
    }

    private Path checkpoint(Map<String, String> overrides) throws Exception {
        return builder(overrides).write(tmp.newFolder("inkling").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static EngineConfig gpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(32).ubatch(8).typeK(KvDType.F32).typeV(KvDType.F32).build();
    }

    // ------------------------------------------------------------------ the synthetic checkpoint

    /** The per-layer attention geometry: local layers may use their own heads and head width. */
    private static final class Geometry {
        final int heads;
        final int kvHeads;
        final int headDim;
        final int qDim;
        final int kDim;
        final int vDim;
        final int rDim;
        final int relExtent;

        Geometry(int heads, int kvHeads, int headDim, int relExtent) {
            this.heads = heads;
            this.kvHeads = kvHeads;
            this.headDim = headDim;
            this.qDim = heads * headDim;
            this.kDim = kvHeads * headDim;
            this.vDim = kvHeads * headDim;
            this.rDim = heads * D_REL;
            this.relExtent = relExtent;
        }
    }

    private static Geometry geometry(int il) {
        boolean local = il == LOCAL_LAYER;
        return local
                ? new Geometry(SWA_HEADS, SWA_KV_HEADS, SWA_HEAD_DIM, WINDOW)
                : new Geometry(HEADS, KV_HEADS, HEAD_DIM, REL_EXTENT);
    }

    private SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        tensor(b, "model.embed_tokens.weight", PADDED_V, D, matrix(PADDED_V, D, 1));
        tensor(b, "model.embed_norm.weight", 1, D, norm(2, D));
        for (int il = 0; il < LAYERS; il++) {
            Geometry g = geometry(il);
            String p = "model.layers." + il + ".";
            int qkvr = g.qDim + 2 * g.kDim + g.rDim;
            tensor(b, p + "attn_norm.weight", 1, D, norm(10 + il, D));
            tensor(b, p + "mlp_norm.weight", 1, D, norm(12 + il, D));
            tensor(b, p + "attn.qkvr.weight", qkvr, D, qkvrMatrix(il));
            tensor(b, p + "attn.qkvr.bias", 1, qkvr, qkvrBias(il));
            tensor(b, p + "attn.wo_ud.weight", D, g.qDim, matrix(D, g.qDim, 20 + il));
            tensor(b, p + "attn.wo_ud.bias", 1, D, bias(22 + il, D));
            tensor(b, p + "attn.rel_logits_proj.proj", D_REL, g.relExtent, relProj(il));
            tensor(b, p + "attn.q_norm.weight", 1, g.headDim, norm(24 + il, g.headDim));
            tensor(b, p + "attn.k_norm.weight", 1, g.headDim, norm(26 + il, g.headDim));
            conv(b, p + "attn.k_sconv.weight", g.kDim, 28 + il);
            conv(b, p + "attn.v_sconv.weight", g.vDim, 30 + il);
            conv(b, p + "attn_sconv.weight", D, 32 + il);
            conv(b, p + "mlp_sconv.weight", D, 34 + il);
            tensor(b, p + "mlp.gate_up_proj.weight", 2 * FF, D, gateUp(il));
            tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, 38 + il));
            tensor(b, p + "mlp.global_scale", 1, 1, new float[]{GLOBAL_SCALE});
        }
        tensor(b, "model.norm.weight", 1, D, norm(40, D));
        tensor(b, "lm_head.weight", PADDED_V, D, matrix(PADDED_V, D, 42));
        return b;
    }

    /** The {@code [Q | K | V | R]} rows of layer {@code il}'s fused projection. */
    private static float[] qkvrMatrix(int il) {
        Geometry g = geometry(il);
        return matrix(g.qDim + 2 * g.kDim + g.rDim, D, 16 + il);
    }

    private static float[] qkvrBias(int il) {
        Geometry g = geometry(il);
        return bias(18 + il, g.qDim + 2 * g.kDim + g.rDim);
    }

    /** The per-layer {@code [d_rel][rel_extent]} relative table. */
    private static float[] relProj(int il) {
        return matrix(D_REL, geometry(il).relExtent, 36 + il);
    }



    /** The interleaved {@code [gate0, up0, gate1, up1, ...]} rows of layer {@code il}'s MLP input projection. */
    private static float[] gateUp(int il) {
        return matrix(2 * FF, D, 44 + il);
    }

    private static void conv(SyntheticHf.Builder b, String name, int channels, int seed) {
        float[] w = new float[channels * CONV_WIDTH];
        for (int i = 0; i < channels; i++) {
            for (int k = 0; k < CONV_WIDTH; k++) {
                w[i * CONV_WIDTH + k] = ((i * 5 + k * 3 + seed) % 11 - 5) * 0.05f;
            }
        }
        b.tensor(name, StDType.F32, new long[]{channels, 1, CONV_WIDTH}, SyntheticHf.f32(w));
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = rows == 1 ? new long[]{cols} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    /** The Inkling {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['InklingForCausalLM']");
        fields.put("model_type", "'inkling'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("dense_intermediate_size", Integer.toString(FF));
        fields.put("num_hidden_layers", Integer.toString(LAYERS));
        fields.put("dense_mlp_idx", Integer.toString(LAYERS));
        fields.put("n_routed_experts", "0");
        fields.put("num_attention_heads", Integer.toString(HEADS));
        fields.put("num_key_value_heads", Integer.toString(KV_HEADS));
        fields.put("head_dim", Integer.toString(HEAD_DIM));
        fields.put("swa_num_attention_heads", Integer.toString(SWA_HEADS));
        fields.put("swa_num_key_value_heads", Integer.toString(SWA_KV_HEADS));
        fields.put("swa_head_dim", Integer.toString(SWA_HEAD_DIM));
        fields.put("swa_v_head_dim", Integer.toString(SWA_HEAD_DIM));
        fields.put("local_layer_ids", "[" + LOCAL_LAYER + "]");
        fields.put("sliding_window_size", Integer.toString(WINDOW));
        fields.put("d_rel", Integer.toString(D_REL));
        fields.put("rel_extent", Integer.toString(REL_EXTENT));
        fields.put("sconv_kernel_size", Integer.toString(CONV_WIDTH));
        fields.put("log_scaling_n_floor", Integer.toString(N_FLOOR));
        fields.put("log_scaling_alpha", Float.toString(ALPHA));
        fields.put("use_embed_norm", "true");
        fields.put("use_global_scale", "true");
        fields.put("q_bias", "true");
        fields.put("o_bias", "true");
        fields.put("hidden_act", "'silu'");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "64");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("padded_vocab_size", Integer.toString(PADDED_V));
        fields.put("tie_word_embeddings", "false");
        fields.put("logits_mup_width_multiplier", Float.toString(MUP));
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

    // ------------------------------------------------------------------ scalar reference

    /** One layer's running state: the four short-convolution histories of the layer. */
    private static final class LayerState {
        final float[][] k;
        final float[][] v;
        final float[][] attn;
        final float[][] mlp;

        LayerState(Geometry g) {
            this.k = new float[CONV_WIDTH - 1][g.kDim];
            this.v = new float[CONV_WIDTH - 1][g.vDim];
            this.attn = new float[CONV_WIDTH - 1][D];
            this.mlp = new float[CONV_WIDTH - 1][D];
        }
    }




    private static float[] reference(int[] tokens) {
        float[] embd = matrix(PADDED_V, D, 1);
        float[] embedNorm = norm(2, D);
        float[][] hidden = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            hidden[t] = rmsNorm(row(embd, tokens[t], D), embedNorm);
        }
        for (int il = 0; il < LAYERS; il++) {
            Geometry g = geometry(il);
            LayerState state = new LayerState(g);
            boolean local = il == LOCAL_LAYER;
            float[] attnNorm = norm(10 + il, D);
            float[] mlpNorm = norm(12 + il, D);
            float[] qkvr = qkvrMatrix(il);
            float[] qkvrB = qkvrBias(il);
            float[] wo = matrix(D, g.qDim, 20 + il);
            float[] woB = bias(22 + il, D);
            float[] proj = relProj(il);
            float[] qNorm = norm(24 + il, g.headDim);
            float[] kNorm = norm(26 + il, g.headDim);
            float[] kW = convWeights(g.kDim, 28 + il);
            float[] vW = convWeights(g.vDim, 30 + il);
            float[] attnW = convWeights(D, 32 + il);
            float[] mlpW = convWeights(D, 34 + il);
            float[] gateUp = gateUp(il);
            float[] down = matrix(D, FF, 38 + il);
            float[][] qHeads = new float[tokens.length][], kHeads = new float[tokens.length][];
            float[][] values = new float[tokens.length][];
            float[][][] rel = new float[tokens.length][][];
            for (int t = 0; t < tokens.length; t++) {
                float[] x = rmsNorm(hidden[t], attnNorm);
                float[] fused = linear(x, qkvr, g.qDim + 2 * g.kDim + g.rDim);
                for (int i = 0; i < fused.length; i++) {
                    fused[i] += qkvrB[i];
                }
                float[] q = slice(fused, 0, g.qDim);
                float[] k = convAdd(state.k, slice(fused, g.qDim, g.kDim), kW, g.kDim);
                float[] v = convAdd(state.v, slice(fused, g.qDim + g.kDim, g.vDim), vW, g.vDim);
                float[] r = slice(fused, g.qDim + 2 * g.kDim, g.rDim);
                float tau = local ? 1.0f : tau(t);
                for (int h = 0; h < g.heads; h++) {
                    normHead(q, h, g.headDim, qNorm);
                    for (int d = 0; d < g.headDim; d++) {
                        q[h * g.headDim + d] *= tau;
                    }
                }
                for (int h = 0; h < g.kvHeads; h++) {
                    normHead(k, h, g.headDim, kNorm);
                }
                qHeads[t] = q;
                kHeads[t] = k;
                values[t] = v;
                rel[t] = new float[g.heads][g.relExtent];
                for (int h = 0; h < g.heads; h++) {
                    for (int e = 0; e < g.relExtent; e++) {
                        float acc = 0;
                        for (int j = 0; j < D_REL; j++) {
                            acc += r[h * D_REL + j] * proj[j * g.relExtent + e];
                        }
                        rel[t][h][e] = acc * tau;
                    }
                }
                float[] attention = new float[g.qDim];
                for (int h = 0; h < g.heads; h++) {
                    attention(h, g, t, local, qHeads, kHeads, values, rel, attention);
                }
                float[] out = linear(attention, wo, D);
                for (int i = 0; i < D; i++) {
                    out[i] += woB[i];
                }
                float[] attnOut = convAdd(state.attn, out, attnW, D);
                float[] stream = new float[D];
                for (int i = 0; i < D; i++) {
                    stream[i] = hidden[t][i] + attnOut[i];
                }
                float[] n2 = rmsNorm(stream, mlpNorm);
                float[] gateUpOut = linear(n2, gateUp, 2 * FF);
                float[] swiglu = new float[FF];
                for (int i = 0; i < FF; i++) {
                    float gate = gateUpOut[2 * i];
                    swiglu[i] = gate * sigmoid(gate) * gateUpOut[2 * i + 1];
                }
                float[] mlp = linear(swiglu, down, D);
                for (int i = 0; i < D; i++) {
                    mlp[i] *= GLOBAL_SCALE;
                }
                float[] mlpOut = convAdd(state.mlp, mlp, mlpW, D);
                hidden[t] = new float[D];
                for (int i = 0; i < D; i++) {
                    hidden[t][i] = stream[i] + mlpOut[i];
                }
            }
        }
        float[] last = rmsNorm(hidden[tokens.length - 1], norm(40, D));
        float[] logits = linear(last, matrix(PADDED_V, D, 42), V);
        for (int i = 0; i < V; i++) {
            logits[i] /= MUP;
        }
        return logits;
    }

    /** One head's softmax over the visible keys of token {@code t}, into {@code attention}. */
    private static void attention(int h, Geometry g, int t, boolean local, float[][] qHeads, float[][] kHeads,
                                  float[][] values, float[][][] rel, float[] attention) {
        int kvHead = h / (g.heads / g.kvHeads);
        float[] weights = new float[t + 1];
        double max = Double.NEGATIVE_INFINITY;
        for (int j = 0; j <= t; j++) {
            if (!visible(local, t, j)) {
                continue;
            }
            float dot = 0;
            for (int d = 0; d < g.headDim; d++) {
                dot += qHeads[t][h * g.headDim + d] * kHeads[j][kvHead * g.headDim + d];
            }
            // The FA4 score mod clamps the distance to [0, rel_extent): farther keys take no bias.
            int dist = t - j;
            weights[j] = dot / g.headDim + (dist < g.relExtent ? rel[t][h][dist] : 0f);
            max = Math.max(max, weights[j]);
        }
        double total = 0;
        for (int j = 0; j <= t; j++) {
            if (!visible(local, t, j)) {
                continue;
            }
            weights[j] = (float) Math.exp(weights[j] - max);
            total += weights[j];
        }
        for (int j = 0; j <= t; j++) {
            if (!visible(local, t, j)) {
                continue;
            }
            for (int d = 0; d < g.headDim; d++) {
                attention[h * g.headDim + d] += weights[j] / (float) total * values[j][kvHead * g.headDim + d];
            }
        }
    }

    /** Whether the attention mask of {@code local} shows key {@code j} to query {@code t}. */
    private static boolean visible(boolean local, int t, int j) {
        if (j > t) {
            return false;
        }
        return !local || j >= Math.max(0, t - (WINDOW - 1));
    }

    /** {@code tau = 1 + alpha * log(max(1, (pos + 1) / n_floor))} on every full-attention layer. */
    private static float tau(int pos) {
        return 1.0f + ALPHA * (float) Math.log(Math.max(1.0, (pos + 1) / (double) N_FLOOR));
    }

    private static void normHead(float[] x, int head, int headDim, float[] w) {
        float[] slice = new float[headDim];
        System.arraycopy(x, head * headDim, slice, 0, headDim);
        System.arraycopy(rmsNorm(slice, w), 0, x, head * headDim, headDim);
    }

    /** {@code conv(x) + x} over the stream's history, which the call advances. */
    private static float[] convAdd(float[][] history, float[] cur, float[] w, int dim) {
        float[] out = new float[dim];
        for (int tap = 0; tap < CONV_WIDTH; tap++) {
            int lag = tap - (CONV_WIDTH - 1);
            float[] src = lag < 0 ? history[CONV_WIDTH - 1 + lag] : cur;
            for (int c = 0; c < dim; c++) {
                out[c] += w[c * CONV_WIDTH + tap] * src[c];
            }
        }
        for (int i = 0; i < CONV_WIDTH - 2; i++) {
            System.arraycopy(history[i + 1], 0, history[i], 0, dim);
        }
        history[CONV_WIDTH - 2] = cur.clone();
        for (int c = 0; c < dim; c++) {
            out[c] += cur[c];
        }
        return out;
    }

    private static float[] convWeights(int channels, int seed) {
        float[] w = new float[channels * CONV_WIDTH];
        for (int i = 0; i < channels; i++) {
            for (int k = 0; k < CONV_WIDTH; k++) {
                w[i * CONV_WIDTH + k] = ((i * 5 + k * 3 + seed) % 11 - 5) * 0.05f;
            }
        }
        return w;
    }

    private static float[] row(float[] m, int r, int cols) {
        float[] out = new float[cols];
        System.arraycopy(m, r * cols, out, 0, cols);
        return out;
    }

    private static float[] slice(float[] x, int from, int n) {
        float[] out = new float[n];
        System.arraycopy(x, from, out, 0, n);
        return out;
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.03f;
        }
        return x;
    }

    private static float[] bias(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = ((i * 3 + seed) % 9 - 4) * 0.02f;
        }
        return x;
    }

    private static float[] norm(int seed, int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = 0.7f + 0.1f * i + 0.01f * seed;
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

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
}
