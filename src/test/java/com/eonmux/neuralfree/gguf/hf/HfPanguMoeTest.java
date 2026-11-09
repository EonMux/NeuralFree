package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * openPangu MoE checkpoint loading. The configuration picks the trunk: the DeepSeek-V2 MLA layout
 * ({@code PanguUltraMoEForCausalLM}), the static sink attention ({@code param_sink_number > 0}) or the pangu-embedded
 * attention with a dense lead and routed plus shared experts ({@code PanguProMoEV2ForCausalLM}). Forward parity
 * against an independent scalar implementation on CPU and OpenCL, the export round trip, and the refusals of the
 * attention configurations NeuralFree cannot run.
 */
public class HfPanguMoeTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 4;
    private static final int V = 7;
    private static final float EPS = 1e-6f;

    /**
     * The plain trunk's head dim. GPU-servable geometry: the flash-attention kernels serve head dims that are
     * positive multiples of 4, so the GPU trunk tests run their attention on the device instead of the load
     * refusing it.
     */
    private static final int HEAD_DIM = 4;

    /**
     * The plain trunk's head count. The openpangu converter writes no key or value length when the configuration
     * carries {@code head_dim}, so the engine's default key length {@code hidden_size / num_attention_heads} must
     * equal the head dim; with the hidden size of {@link #D}, that is one head of {@link #HEAD_DIM}.
     */
    private static final int PLAIN_HEADS = 1;

    // ---- MLA trunk geometry
    private static final int HEADS = 2;
    private static final int NOPE = 4;
    private static final int ROPE_DIM = 4;
    private static final int QK = NOPE + ROPE_DIM;
    private static final int VHEAD = 1;
    private static final int KV_LORA = 4;
    private static final int Q_LORA = 2;
    private static final int FFN = 4;
    private static final int MOE_FFN = 2;
    private static final int EXPERTS = 2;

    private static final String MLA_CONFIG = SyntheticHf.json("{'architectures':['PanguUltraMoEForCausalLM'],"
            + "'model_type':'openpangu','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'q_lora_rank':" + Q_LORA
            + ",'kv_lora_rank':" + KV_LORA + ",'qk_nope_head_dim':" + NOPE + ",'qk_rope_head_dim':" + ROPE_DIM
            + ",'v_head_dim':" + VHEAD + ",'max_position_embeddings':16,'rms_norm_eps':" + EPS
            + ",'rope_theta':10000.0,'vocab_size':" + V + ",'first_k_dense_replace':1,'moe_intermediate_size':"
            + MOE_FFN + ",'num_experts':" + EXPERTS + ",'num_experts_per_tok':1,'norm_topk_prob':true"
            + ",'routed_scaling_factor':1.0,'n_routed_experts':" + EXPERTS + ",'n_shared_experts':1"
            + ",'hidden_act':'silu','tie_word_embeddings':false}");

    private static final String PLAIN_CONFIG = SyntheticHf.json("{'architectures':['PanguProMoEV2ForCausalLM'],"
            + "'model_type':'openpangu','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + PLAIN_HEADS + ",'num_key_value_heads':" + PLAIN_HEADS
            + ",'head_dim':" + HEAD_DIM + ",'max_position_embeddings':16,'rms_norm_eps':" + EPS + ",'rope_theta':10000.0"
            + ",'vocab_size':" + V + ",'first_k_dense_replace':1,'moe_intermediate_size':" + MOE_FFN
            + ",'num_experts':" + EXPERTS + ",'n_routed_experts':" + EXPERTS + ",'n_shared_experts':1"
            + ",'num_experts_per_tok':1,'norm_topk_prob':true,'routed_scaling_factor':1.0,'hidden_act':'silu'"
            + ",'tie_word_embeddings':false}");

    private static final String PLAIN_SANDWICH_CONFIG = PLAIN_CONFIG.replace("\"tie_word_embeddings\"",
            "\"sandwich_norm\":true,\"tie_word_embeddings\"");
    private static final String PLAIN_SWA_CONFIG = PLAIN_CONFIG.replace("\"tie_word_embeddings\"",
            "\"interleaved_sliding_window\":[2,0],\"tie_word_embeddings\"");
    private static final String PLAIN_MIXED_SWA_CONFIG = PLAIN_CONFIG.replace("\"tie_word_embeddings\"",
            "\"interleaved_sliding_window\":[2,4],\"tie_word_embeddings\"");

    // ---- static sink attention geometry of the non-MLA trunk
    private static final int SINK_NOPE = 2;
    private static final int SINK_ROPE = 2;
    private static final int SINK_K = SINK_NOPE + SINK_ROPE;
    private static final int SINK_V = 4;
    private static final int SINKS = 2;

    private static final String SINK_CONFIG = SyntheticHf.json("{'architectures':['PanguProMoEV2ForCausalLM'],"
            + "'model_type':'openpangu','hidden_size':" + D + ",'intermediate_size':" + FFN
            + ",'num_hidden_layers':2,'num_attention_heads':" + HEADS + ",'param_sink_number':" + SINKS
            + ",'param_sink_with_value':true,'qk_nope_dim':" + SINK_NOPE + ",'qk_rope_dim':" + SINK_ROPE
            + ",'v_channels':" + SINK_V
            + ",'max_position_embeddings':16,'rms_norm_eps':" + EPS + ",'rope_theta':10000.0,'vocab_size':" + V
            + ",'first_k_dense_replace':1,'moe_intermediate_size':" + MOE_FFN + ",'num_experts':" + EXPERTS
            + ",'num_experts_per_tok':1,'norm_topk_prob':true,'routed_scaling_factor':1.0,'n_routed_experts':"
            + EXPERTS + ",'n_shared_experts':1,'hidden_act':'silu','tie_word_embeddings':false}");

    /** Static sinks in layers whose interleaved windows differ, so the per-layer window list is exported too. */
    private static final String SINK_MIXED_SWA_CONFIG = SINK_CONFIG.replace("\"tie_word_embeddings\"",
            "\"interleaved_sliding_window\":[3,4],\"tie_word_embeddings\"");

    /** Enough context for a prefill long enough to split its cached keys on the device. */
    private static final String SINK_LONG_CONFIG = SINK_CONFIG.replace("\"max_position_embeddings\":16",
            "\"max_position_embeddings\":512");
    private static final String SINK_LONG_SWA_CONFIG = SINK_LONG_CONFIG.replace("\"tie_word_embeddings\"",
            "\"interleaved_sliding_window\":[3,4],\"tie_word_embeddings\"");

    @Test
    public void mlaTrunkMatchesScalarForward() throws Exception {
        Path path = mlaCheckpoint();
        assertForward(path, mlaReference(new int[]{2, 3, 4}));
    }

    @Test
    public void mlaTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(mlaCheckpoint(), mlaReference(new int[]{2, 3, 4}));
    }

    @Test
    public void plainTrunkMatchesScalarForward() throws Exception {
        Path path = plainCheckpoint();
        assertForward(path, plainReference(new int[]{2, 3, 4}));
    }

    @Test
    public void plainTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(plainCheckpoint(), plainReference(new int[]{2, 3, 4}));
    }

    @Test
    public void sandwichNormTrunkMatchesScalarForward() throws Exception {
        Path path = plainCheckpoint(PLAIN_SANDWICH_CONFIG, "pangu-sandwich", true);
        assertForward(path, plainReference(new int[]{2, 3, 4}, true, new int[]{0, 0}));
    }

    @Test
    public void sandwichNormTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(plainCheckpoint(PLAIN_SANDWICH_CONFIG, "pangu-sandwich-gpu", true),
                plainReference(new int[]{2, 3, 4}, true, new int[]{0, 0}));
    }

    @Test
    public void interleavedSlidingWindowTrunkMatchesScalarForward() throws Exception {
        Path path = plainCheckpoint(PLAIN_SWA_CONFIG, "pangu-swa", false);
        assertForward(path, plainReference(new int[]{2, 3, 4}, false, new int[]{2, 0}));
    }

    @Test
    public void interleavedSlidingWindowTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(plainCheckpoint(PLAIN_SWA_CONFIG, "pangu-swa-gpu", false),
                plainReference(new int[]{2, 3, 4}, false, new int[]{2, 0}));
    }

    @Test
    public void mixedSlidingWindowTrunkMatchesScalarForward() throws Exception {
        Path path = plainCheckpoint(PLAIN_MIXED_SWA_CONFIG, "pangu-swa-mixed", false);
        assertForward(path, plainReference(new int[]{2, 3, 4}, false, new int[]{2, 4}));
    }

    @Test
    public void mixedSlidingWindowTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(plainCheckpoint(PLAIN_MIXED_SWA_CONFIG, "pangu-swa-mixed-gpu", false),
                plainReference(new int[]{2, 3, 4}, false, new int[]{2, 4}));
    }

    @Test
    public void sinkTrunkMatchesScalarForward() throws Exception {
        assertForward(sinkCheckpoint(), sinkReference(new int[]{2, 3, 4}, new int[]{0, 0}));
    }

    @Test
    public void sinkTrunkMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(sinkCheckpoint(), sinkReference(new int[]{2, 3, 4}, new int[]{0, 0}));
    }

    @Test
    public void sinkTrunkWithMixedWindowsMatchesScalarForward() throws Exception {
        Path path = sinkCheckpoint(SINK_MIXED_SWA_CONFIG, "pangu-sink-swa");
        assertForward(path, sinkReference(new int[]{2, 3, 4}, new int[]{3, 4}));
    }

    @Test
    public void sinkTrunkWithMixedWindowsMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuForward(sinkCheckpoint(SINK_MIXED_SWA_CONFIG, "pangu-sink-swa-gpu"),
                sinkReference(new int[]{2, 3, 4}, new int[]{3, 4}));
    }

    @Test
    public void sinkTrunkDecodeMatchesScalarForward() throws Exception {
        assertDecode(sinkCheckpoint(), sinkReference(new int[]{2, 3, 4, 5}, new int[]{0, 0}));
    }

    @Test
    public void sinkTrunkDecodeMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        assertGpuDecode(sinkCheckpoint(), sinkReference(new int[]{2, 3, 4, 5}, new int[]{0, 0}));
    }

    @Test
    public void sinkTrunkLongPrefillWithWindowsMatchesScalarForward() throws Exception {
        int[] tokens = tokens(300);
        Path path = sinkCheckpoint(SINK_LONG_SWA_CONFIG, "pangu-sink-long");
        assertForward(path, tokens, sinkReference(tokens, new int[]{3, 4}), 512, 512);
    }

    @Test
    public void sinkTrunkLongPrefillWithWindowsMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        int[] tokens = tokens(300);
        Path path = sinkCheckpoint(SINK_LONG_SWA_CONFIG, "pangu-sink-long-gpu");
        assertGpuForward(path, tokens, sinkReference(tokens, new int[]{3, 4}), 512, 512);
    }

    @Test
    public void sinkAndMixedWindowVariantsExportAndReload() throws Exception {
        Path sink = tmp.getRoot().toPath().resolve("pangu-sink.gguf");
        Path mixed = tmp.getRoot().toPath().resolve("pangu-swa-mixed.gguf");
        export(sinkCheckpoint(), sink);
        export(plainCheckpoint(PLAIN_MIXED_SWA_CONFIG, "pangu-swa-mixed-export", false), mixed);
        assertForward(sink, sinkReference(new int[]{2, 3, 4}, new int[]{0, 0}));
        assertForward(mixed, plainReference(new int[]{2, 3, 4}, false, new int[]{2, 4}));
    }

    @Test
    public void sandwichAndSlidingWindowVariantsExportAndReload() throws Exception {
        Path sandwich = tmp.getRoot().toPath().resolve("pangu-sandwich.gguf");
        Path swa = tmp.getRoot().toPath().resolve("pangu-swa.gguf");
        export(plainCheckpoint(PLAIN_SANDWICH_CONFIG, "pangu-sandwich-export", true), sandwich);
        export(plainCheckpoint(PLAIN_SWA_CONFIG, "pangu-swa-export", false), swa);
        assertForward(sandwich, plainReference(new int[]{2, 3, 4}, true, new int[]{0, 0}));
        assertForward(swa, plainReference(new int[]{2, 3, 4}, false, new int[]{2, 0}));
    }

    @Test
    public void bothTrunksExportAndReload() throws Exception {
        Path mla = tmp.getRoot().toPath().resolve("pangu-mla.gguf");
        Path plain = tmp.getRoot().toPath().resolve("pangu-plain.gguf");
        export(mlaCheckpoint(), mla);
        export(plainCheckpoint(), plain);
        assertForward(mla, mlaReference(new int[]{2, 3, 4}));
        assertForward(plain, plainReference(new int[]{2, 3, 4}));
    }

    @Test
    public void resolvesTheTrunkFromTheConfiguration() throws Exception {
        assertEquals(Arch.DEEPSEEK2, resolve("{\"architectures\":[\"PanguUltraMoEForCausalLM\"],"
                + "\"qk_nope_head_dim\":2,\"qk_rope_head_dim\":2,\"v_head_dim\":1,\"kv_lora_rank\":4}"));
        assertEquals(Arch.PANGU_MOE, resolve("{\"architectures\":[\"PanguProMoEV2ForCausalLM\"]}"));
    }

    @Test
    public void resolvesTheSinkTrunkFromTheConfiguration() {
        assertEquals(Arch.PANGU_MOE, resolve(SINK_CONFIG));
    }

    @Test
    public void refusesUnsupportedSinkConfigurationsByName() {
        assertRefused("\"param_sink_number\":4", "qk_nope_dim");
        assertRefused("\"param_sink_number\":4,\"qk_nope_dim\":2,\"qk_rope_dim\":2,\"v_channels\":2,"
                + "\"is_causal\":false", "is_causal");
    }

    @Test
    public void registeredAsSupportedTextArchitectures() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("PanguProMoEV2ForCausalLM"));
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("PanguUltraMoEForCausalLM"));
    }

    private static Arch resolve(String json) {
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = HfConfigLoader.fromJson(Path.of("."), json);
        return HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).arch()
                .resolve(cfg, HfTensorIndex.of(new java.util.LinkedHashMap<>()));
    }

    private void assertRefused(String extra, String field) {
        String json = "{\"architectures\":[\"PanguProMoEV2ForCausalLM\"],\"num_hidden_layers\":2," + extra + "}";
        com.eonmux.neuralfree.gguf.hf.config.HfConfig cfg = HfConfigLoader.fromJson(Path.of("."), json);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT).arch()
                        .resolve(cfg, HfTensorIndex.of(new java.util.LinkedHashMap<>())));
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private void assertForward(Path path, float[] expected) throws Exception {
        assertForward(path, new int[]{2, 3, 4}, expected, 16, 4);
    }

    private void assertForward(Path path, int[] tokens, float[] expected, int nCtx, int ubatch) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(ubatch).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            assertArrayEquals(expected, model.forwardBackend().prefill(tokens, 0).clone(), 3e-4f);
        }
    }

    private void assertGpuForward(Path path, float[] expected) throws Exception {
        assertGpuForward(path, new int[]{2, 3, 4}, expected, 16, 4);
    }

    private void assertGpuForward(Path path, int[] tokens, float[] expected, int nCtx, int ubatch) throws Exception {
        try (EngineModel model = openGpu(path, nCtx, ubatch)) {
            assertArrayEquals(expected, model.forwardBackend().prefill(tokens, 0).clone(), 3e-4f);
        }
    }

    /** Opens {@code path} on the OpenCL device, skipping the test when no device or kernel set is available. */
    private static EngineModel openGpu(Path path, int nCtx, int ubatch) throws Exception {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .gpuLayers(EngineConfig.GPU_LAYERS_ALL)
                .memory(MemoryConfig.builder().nCtx(nCtx).ubatch(ubatch).build()).build();
        EngineModel model = LLMModelFactory.open(path.toString(), cfg);
        if (!(model instanceof com.eonmux.neuralfree.gguf.GGUFModelGPU)) {
            model.close();
            Assume.assumeTrue("OpenCL kernels were not built: the engine fell back to the CPU", false);
        }
        return model;
    }

    /** Prefills {@code 2, 3, 4} then decodes {@code 5} at position 3; {@code expected} is the logits at that row. */
    private void assertDecode(Path path, float[] expected) throws Exception {
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            model.forwardBackend().prefill(new int[]{2, 3, 4}, 0);
            assertArrayEquals(expected, model.forwardBackend().decode(5, 3).clone(), 3e-4f);
        }
    }

    private void assertGpuDecode(Path path, float[] expected) throws Exception {
        try (EngineModel model = openGpu(path, 16, 4)) {
            model.forwardBackend().prefill(new int[]{2, 3, 4}, 0);
            assertArrayEquals(expected, model.forwardBackend().decode(5, 3).clone(), 3e-4f);
        }
    }

    /** {@code n} token ids of the vocabulary, in a repeating pattern. */
    private static int[] tokens(int n) {
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = (i * 3 + 2) % V;
        }
        return ids;
    }

    private void export(Path dir, Path out) throws Exception {
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder().toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
    }

    // ------------------------------------------------------------------ checkpoints

    private Path mlaCheckpoint() throws Exception {
        SyntheticHf.Builder b = SyntheticHf.builder().config(MLA_CONFIG);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_a_proj.weight", Q_LORA, D, matrix(Q_LORA, D, 2 + 10 * l));
            tensor(b, p + "self_attn.q_a_layernorm.weight", Q_LORA, 1, ones(Q_LORA));
            tensor(b, p + "self_attn.q_b_proj.weight", HEADS * QK, Q_LORA, matrix(HEADS * QK, Q_LORA, 3 + 10 * l));
            tensor(b, p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + ROPE_DIM, D,
                    matrix(KV_LORA + ROPE_DIM, D, 4 + 10 * l));
            tensor(b, p + "self_attn.kv_a_layernorm.weight", KV_LORA, 1, ones(KV_LORA));
            tensor(b, p + "self_attn.kv_b_proj.weight", HEADS * (NOPE + VHEAD), KV_LORA,
                    matrix(HEADS * (NOPE + VHEAD), KV_LORA, 5 + 10 * l));
            tensor(b, p + "self_attn.o_proj.weight", D, HEADS * VHEAD, matrix(D, HEADS * VHEAD, 6 + 10 * l));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        }
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 20));
        for (int e = 0; e < EXPERTS; e++) {
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 21 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 22 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FFN,
                    matrix(D, MOE_FFN, 23 + 3 * e));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 30));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 31));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, MOE_FFN, matrix(D, MOE_FFN, 32));
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 40));
        return b.write(tmp.newFolder("pangu-mla").toPath());
    }

    private Path plainCheckpoint() throws Exception {
        return plainCheckpoint(PLAIN_CONFIG, "pangu-plain", false);
    }

    private Path plainCheckpoint(String config, String folder, boolean sandwich) throws Exception {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        int headDim = HEAD_DIM;
        int qkv = PLAIN_HEADS * headDim;
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_proj.weight", qkv, D, matrix(qkv, D, 2 + 10 * l));
            tensor(b, p + "self_attn.k_proj.weight", qkv, D, matrix(qkv, D, 3 + 10 * l));
            tensor(b, p + "self_attn.v_proj.weight", qkv, D, matrix(qkv, D, 4 + 10 * l));
            tensor(b, p + "self_attn.o_proj.weight", D, qkv, matrix(D, qkv, 5 + 10 * l));
            if (sandwich) {
                tensor(b, p + "post_attention_layernorm.weight", D, 1, norm(50 + 10 * l));
                tensor(b, p + "pre_mlp_layernorm.weight", D, 1, norm(51 + 10 * l));
                tensor(b, p + "post_mlp_layernorm.weight", D, 1, norm(52 + 10 * l));
            } else {
                tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
            }
        }
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 20));
        for (int e = 0; e < EXPERTS; e++) {
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 21 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 22 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FFN,
                    matrix(D, MOE_FFN, 23 + 3 * e));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 30));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 31));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, MOE_FFN, matrix(D, MOE_FFN, 32));
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 40));
        return b.write(tmp.newFolder(folder).toPath());
    }

    private Path sinkCheckpoint() throws Exception {
        return sinkCheckpoint(SINK_CONFIG, "pangu-sink");
    }

    /**
     * The non-MLA trunk with the static sink attention: the split key/value geometry ({@code qk_nope_dim} +
     * {@code qk_rope_dim} key values, {@code v_channels} value values), the K norm and the learned sink tokens.
     */
    private Path sinkCheckpoint(String config, String folder) throws Exception {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config);
        tensor(b, "model.embed_tokens.weight", V, D, matrix(V, D, 1));
        int qkv = HEADS * SINK_K;
        int vw = HEADS * SINK_V;
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
            tensor(b, p + "self_attn.q_proj.weight", qkv, D, matrix(qkv, D, 2 + 10 * l));
            tensor(b, p + "self_attn.k_proj.weight", qkv, D, matrix(qkv, D, 3 + 10 * l));
            tensor(b, p + "self_attn.v_proj.weight", vw, D, matrix(vw, D, 4 + 10 * l));
            tensor(b, p + "self_attn.o_proj.weight", D, vw, matrix(D, vw, 5 + 10 * l));
            tensor(b, p + "self_attn.k_layernorm.weight", SINK_K, 1, sinkNorm(l));
            tensor(b, p + "self_attn.param_sink_key", SINKS * HEADS, SINK_K,
                    matrix(SINKS * HEADS, SINK_K, 33 + 10 * l));
            tensor(b, p + "self_attn.param_sink_value", SINKS * HEADS, SINK_V,
                    matrix(SINKS * HEADS, SINK_V, 35 + 10 * l));
            tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        }
        tensor(b, "model.layers.0.mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, "model.layers.0.mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, "model.layers.0.mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        tensor(b, "model.layers.1.mlp.gate.weight", EXPERTS, D, matrix(EXPERTS, D, 20));
        for (int e = 0; e < EXPERTS; e++) {
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 21 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FFN, D,
                    matrix(MOE_FFN, D, 22 + 3 * e));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FFN,
                    matrix(D, MOE_FFN, 23 + 3 * e));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 30));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", MOE_FFN, D, matrix(MOE_FFN, D, 31));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, MOE_FFN, matrix(D, MOE_FFN, 32));
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, 40));
        return b.write(tmp.newFolder(folder).toPath());
    }

    /** The K norm weight of the sink layer: one {@code qk_nope_dim + qk_rope_dim} row, non-unit. */
    private static float[] sinkNorm(int layer) {
        return norm(60 + 10 * layer);
    }

    /** A non-unit norm weight so a misplaced sandwich norm is visible. */
    private static float[] norm(int seed) {
        float[] x = new float[D];
        for (int i = 0; i < D; i++) {
            x[i] = 0.5f + 0.1f * ((i + seed) % 5);
        }
        return x;
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 5) % 19 - 9) * 0.025f;
        }
        return x;
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1.0f);
        return x;
    }

    // ------------------------------------------------------------------ scalar references

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            float s = 0;
            for (int j = 0; j < x.length; j++) {
                s += x[j] * w[i * x.length + j];
            }
            y[i] = s;
        }
        return y;
    }

    private static float[] rms(float[] x, float[] w) {
        double ss = 0;
        for (float v : x) {
            ss += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(ss / x.length + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale * w[i];
        }
        return y;
    }

    private static void rotate(float[] x, int pos, int nRot) {
        for (int i = 0; i < nRot / 2; i++) {
            double theta = pos / Math.pow(10000.0, 2.0 * i / nRot);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = x[2 * i];
            float b = x[2 * i + 1];
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = b * c + a * s;
        }
    }

    /**
     * NEOX-style rotation of a whole head (the openPangu graph's rope type): dim {@code i} and its opposite
     * {@code i + nRot / 2} share {@code theta_i}. For a 2-dim head it coincides with {@link #rotate}; for wider
     * heads the pairing is what distinguishes the styles.
     */
    private static void rotateNeox(float[] x, int pos, int nRot) {
        for (int i = 0; i < nRot / 2; i++) {
            double theta = pos / Math.pow(10000.0, 2.0 * i / nRot);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = x[i];
            float b = x[i + nRot / 2];
            x[i] = a * c - b * s;
            x[i + nRot / 2] = b * c + a * s;
        }
    }

    /** NORM-style rotation of {@code nRot} values starting at {@code off} (a partial rotary over a head). */
    private static void rotateRange(float[] x, int off, int pos, int nRot) {
        for (int i = 0; i < nRot / 2; i++) {
            double theta = pos / Math.pow(10000.0, 2.0 * i / nRot);
            float c = (float) Math.cos(theta);
            float s = (float) Math.sin(theta);
            float a = x[off + 2 * i];
            float b = x[off + 2 * i + 1];
            x[off + 2 * i] = a * c - b * s;
            x[off + 2 * i + 1] = b * c + a * s;
        }
    }

    private static float[] siluMul(float[] gate, float[] up) {
        float[] y = new float[gate.length];
        for (int i = 0; i < gate.length; i++) {
            y[i] = gate[i] / (1.0f + (float) Math.exp(-gate[i])) * up[i];
        }
        return y;
    }

    private static float[] ffn(float[] x, int seedBase, int inter) {
        float[] gate = linear(x, matrix(inter, D, seedBase), inter);
        float[] up = linear(x, matrix(inter, D, seedBase + 1), inter);
        return linear(siluMul(gate, up), matrix(D, inter, seedBase + 2), D);
    }

    private static float[] moe(float[] x) {
        float[] logits = linear(x, matrix(EXPERTS, D, 20), EXPERTS);
        int best = logits[0] > logits[1] ? 0 : 1;
        float[] expert = ffn(x, 21 + 3 * best, MOE_FFN);
        float[] shared = ffn(x, 30, MOE_FFN);
        for (int i = 0; i < D; i++) {
            expert[i] += shared[i];
        }
        return expert;
    }

    private static float[][] mlaAttention(float[][] x, int layer) {
        int t = x.length;
        float[][][] q = new float[t][HEADS][QK];
        float[][][] k = new float[t][HEADS][QK];
        float[][][] v = new float[t][HEADS][VHEAD];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qa = linear(n, matrix(Q_LORA, D, 2 + 10 * layer), Q_LORA);
            float[] qb = linear(rms(qa, ones(Q_LORA)), matrix(HEADS * QK, Q_LORA, 3 + 10 * layer), HEADS * QK);
            float[] kva = linear(n, matrix(KV_LORA + ROPE_DIM, D, 4 + 10 * layer), KV_LORA + ROPE_DIM);
            float[] latent = new float[KV_LORA];
            System.arraycopy(kva, 0, latent, 0, KV_LORA);
            float[] pe = new float[ROPE_DIM];
            System.arraycopy(kva, KV_LORA, pe, 0, ROPE_DIM);
            float[] kvb = linear(rms(latent, ones(KV_LORA)),
                    matrix(HEADS * (NOPE + VHEAD), KV_LORA, 5 + 10 * layer), HEADS * (NOPE + VHEAD));
            rotate(pe, p, ROPE_DIM);
            for (int h = 0; h < HEADS; h++) {
                float[] qh = new float[QK];
                System.arraycopy(qb, h * QK, qh, 0, QK);
                System.arraycopy(qh, 0, q[p][h], 0, NOPE);
                float[] qp = new float[ROPE_DIM];
                System.arraycopy(qh, NOPE, qp, 0, ROPE_DIM);
                rotate(qp, p, ROPE_DIM);
                System.arraycopy(qp, 0, q[p][h], NOPE, ROPE_DIM);
                System.arraycopy(kvb, h * (NOPE + VHEAD), k[p][h], 0, NOPE);
                System.arraycopy(pe, 0, k[p][h], NOPE, ROPE_DIM);
                for (int d = 0; d < VHEAD; d++) {
                    v[p][h][d] = kvb[h * (NOPE + VHEAD) + NOPE + d];
                }
            }
        }
        float[][] out = new float[t][D];
        float scale = (float) (1.0 / Math.sqrt(QK));
        for (int p = 0; p < t; p++) {
            float[] heads = new float[HEADS * VHEAD];
            for (int h = 0; h < HEADS; h++) {
                float[] scores = new float[p + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j <= p; j++) {
                    scores[j] = dot(q[p][h], k[j][h]) * scale;
                    max = Math.max(max, scores[j]);
                }
                float sum = 0;
                for (int j = 0; j <= p; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int j = 0; j <= p; j++) {
                    for (int d = 0; d < VHEAD; d++) {
                        heads[h * VHEAD + d] += scores[j] / sum * v[j][h][d];
                    }
                }
            }
            out[p] = linear(heads, matrix(D, HEADS * VHEAD, 6 + 10 * layer), D);
        }
        return out;
    }

    private static float[] mlaReference(int[] tokens) {
        float[][] x = embed(tokens);
        float[][] attn = mlaAttention(x, 0);
        for (int t = 0; t < x.length; t++) {
            add(x[t], attn[t]);
            add(x[t], ffn(rms(x[t], ones(D)), 7, FFN));
        }
        attn = mlaAttention(x, 1);
        for (int t = 0; t < x.length; t++) {
            add(x[t], attn[t]);
            add(x[t], moe(rms(x[t], ones(D))));
        }
        return linear(rms(x[x.length - 1], ones(D)), matrix(V, D, 40), V);
    }

    private static float[][] plainAttention(float[][] x, int layer, int window) {
        int t = x.length;
        int headDim = HEAD_DIM;
        int qkv = PLAIN_HEADS * headDim;
        float[][][] q = new float[t][PLAIN_HEADS][headDim];
        float[][][] k = new float[t][PLAIN_HEADS][headDim];
        float[][][] v = new float[t][PLAIN_HEADS][headDim];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qv = linear(n, matrix(qkv, D, 2 + 10 * layer), qkv);
            float[] kv = linear(n, matrix(qkv, D, 3 + 10 * layer), qkv);
            float[] vv = linear(n, matrix(qkv, D, 4 + 10 * layer), qkv);
            for (int h = 0; h < PLAIN_HEADS; h++) {
                System.arraycopy(qv, h * headDim, q[p][h], 0, headDim);
                System.arraycopy(kv, h * headDim, k[p][h], 0, headDim);
                System.arraycopy(vv, h * headDim, v[p][h], 0, headDim);
                rotateNeox(q[p][h], p, headDim);
                rotateNeox(k[p][h], p, headDim);
            }
        }
        float[][] out = new float[t][D];
        float scale = (float) (1.0 / Math.sqrt(headDim));
        for (int p = 0; p < t; p++) {
            float[] heads = new float[qkv];
            for (int h = 0; h < PLAIN_HEADS; h++) {
                int first = window > 0 ? Math.max(0, p - window + 1) : 0;
                float[] scores = new float[p - first + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = first; j <= p; j++) {
                    scores[j - first] = dot(q[p][h], k[j][h]) * scale;
                    max = Math.max(max, scores[j - first]);
                }
                float sum = 0;
                for (int j = first; j <= p; j++) {
                    scores[j - first] = (float) Math.exp(scores[j - first] - max);
                    sum += scores[j - first];
                }
                for (int j = first; j <= p; j++) {
                    for (int d = 0; d < headDim; d++) {
                        heads[h * headDim + d] += scores[j - first] / sum * v[j][h][d];
                    }
                }
            }
            out[p] = linear(heads, matrix(D, qkv, 5 + 10 * layer), D);
        }
        return out;
    }

    private static float[] plainReference(int[] tokens) {
        return plainReference(tokens, false, new int[]{0, 0});
    }

    private static float[] plainReference(int[] tokens, boolean sandwich, int[] windows) {
        float[][] x = embed(tokens);
        for (int l = 0; l < 2; l++) {
            float[][] attn = plainAttention(x, l, windows[l]);
            for (int t = 0; t < x.length; t++) {
                if (sandwich) {
                    float[] postAttn = rms(attn[t], norm(50 + 10 * l));
                    float[] residual = new float[D];
                    for (int i = 0; i < D; i++) {
                        residual[i] = x[t][i] + postAttn[i];
                    }
                    float[] n = rms(residual, norm(51 + 10 * l));
                    float[] branch = l == 0 ? ffn(n, 7, FFN) : moe(n);
                    branch = rms(branch, norm(52 + 10 * l));
                    for (int i = 0; i < D; i++) {
                        x[t][i] = residual[i] + branch[i];
                    }
                } else {
                    add(x[t], attn[t]);
                    float[] n = rms(x[t], ones(D));
                    add(x[t], l == 0 ? ffn(n, 7, FFN) : moe(n));
                }
            }
        }
        return linear(rms(x[x.length - 1], ones(D)), matrix(V, D, 40), V);
    }

    /**
     * The static sink attention of one layer: the sinks precede the cached keys (vLLM {@code StaticSinkAttention}
     * adds {@code param_sink_number} to the sequence length), their keys are normalized by the K norm and carry no
     * RoPE, and their index is the query's position plus the sink count under the causal and window masks.
     */
    private static float[][] sinkAttention(float[][] x, int layer, int window) {
        int t = x.length;
        int qkv = HEADS * SINK_K;
        int vw = HEADS * SINK_V;
        float[] kNorm = sinkNorm(layer);
        float[] sinkKeys = matrix(SINKS * HEADS, SINK_K, 33 + 10 * layer);
        float[] sinkValues = matrix(SINKS * HEADS, SINK_V, 35 + 10 * layer);
        float[][] normSinkKeys = new float[SINKS * HEADS][SINK_K];
        for (int r = 0; r < SINKS * HEADS; r++) {
            float[] raw = new float[SINK_K];
            System.arraycopy(sinkKeys, r * SINK_K, raw, 0, SINK_K);
            normSinkKeys[r] = rms(raw, kNorm);
        }
        float[][][] q = new float[t][HEADS][SINK_K];
        float[][][] k = new float[t][HEADS][SINK_K];
        float[][][] v = new float[t][HEADS][SINK_V];
        for (int p = 0; p < t; p++) {
            float[] n = rms(x[p], ones(D));
            float[] qv = linear(n, matrix(qkv, D, 2 + 10 * layer), qkv);
            float[] kv = linear(n, matrix(qkv, D, 3 + 10 * layer), qkv);
            float[] vv = linear(n, matrix(vw, D, 4 + 10 * layer), vw);
            for (int h = 0; h < HEADS; h++) {
                System.arraycopy(qv, h * SINK_K, q[p][h], 0, SINK_K);
                System.arraycopy(kv, h * SINK_K, k[p][h], 0, SINK_K);
                System.arraycopy(vv, h * SINK_V, v[p][h], 0, SINK_V);
                /* vLLM's partial rotary rotates the head prefix (qk_rope_dim of qk_nope_dim + qk_rope_dim). */
                rotateRange(q[p][h], 0, p, SINK_ROPE);
                System.arraycopy(rms(k[p][h], kNorm), 0, k[p][h], 0, SINK_K);
                rotateRange(k[p][h], 0, p, SINK_ROPE);
            }
        }
        float[][] out = new float[t][D];
        float scale = (float) (1.0 / Math.sqrt(SINK_K));
        for (int p = 0; p < t; p++) {
            float[] heads = new float[vw];
            for (int h = 0; h < HEADS; h++) {
                int first = window > 0 ? Math.max(0, p - window + 1) : 0;
                float[] sinkScores = new float[SINKS];
                float[] cacheScores = new float[p - first + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int r = 0; r < SINKS; r++) {
                    if (sinkVisible(window, p, r)) {
                        sinkScores[r] = dot(q[p][h], normSinkKeys[r * HEADS + h]) * scale;
                        max = Math.max(max, sinkScores[r]);
                    } else {
                        sinkScores[r] = Float.NEGATIVE_INFINITY;
                    }
                }
                for (int j = first; j <= p; j++) {
                    cacheScores[j - first] = dot(q[p][h], k[j][h]) * scale;
                    max = Math.max(max, cacheScores[j - first]);
                }
                float sum = 0;
                for (int r = 0; r < SINKS; r++) {
                    if (sinkScores[r] == Float.NEGATIVE_INFINITY) {
                        continue;
                    }
                    float e = (float) Math.exp(sinkScores[r] - max);
                    sum += e;
                    for (int d = 0; d < SINK_V; d++) {
                        heads[h * SINK_V + d] += e * sinkValues[(r * HEADS + h) * SINK_V + d];
                    }
                }
                for (int j = first; j <= p; j++) {
                    float e = (float) Math.exp(cacheScores[j - first] - max);
                    sum += e;
                    for (int d = 0; d < SINK_V; d++) {
                        heads[h * SINK_V + d] += e * v[j][h][d];
                    }
                }
                for (int d = 0; d < SINK_V; d++) {
                    heads[h * SINK_V + d] /= sum;
                }
            }
            out[p] = linear(heads, matrix(D, vw, 5 + 10 * layer), D);
        }
        return out;
    }

    /**
     * A sink token is visible when the causal window admits its index (the query's position plus the sink count) or
     * when the layer has no window.
     */
    private static boolean sinkVisible(int window, int queryPos, int sink) {
        return window <= 0 || sink > queryPos + SINKS - window;
    }

    private static float[] sinkReference(int[] tokens, int[] windows) {
        float[][] x = embed(tokens);
        for (int l = 0; l < 2; l++) {
            float[][] attn = sinkAttention(x, l, windows[l]);
            for (int t = 0; t < x.length; t++) {
                add(x[t], attn[t]);
                float[] n = rms(x[t], ones(D));
                add(x[t], l == 0 ? ffn(n, 7, FFN) : moe(n));
            }
        }
        return linear(rms(x[x.length - 1], ones(D)), matrix(V, D, 40), V);
    }

    private static float[][] embed(int[] tokens) {
        float[][] x = new float[tokens.length][D];
        float[] emb = matrix(V, D, 1);
        for (int t = 0; t < tokens.length; t++) {
            System.arraycopy(emb, tokens[t] * D, x[t], 0, D);
        }
        return x;
    }

    private static void add(float[] x, float[] delta) {
        for (int i = 0; i < x.length; i++) {
            x[i] += delta[i];
        }
    }
}
