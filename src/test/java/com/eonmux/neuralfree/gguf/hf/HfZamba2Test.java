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
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.cpu.HeapModelMemory;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.Zamba2Spec;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.spec.resolve.TensorClaims;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assume;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.nio.file.Files;
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
 * Zamba2 ({@code Zamba2ForCausalLM}) checkpoint loading, converted layout and inference against a scalar
 * implementation of vLLM's {@code Zamba2ForCausalLM.forward} (vllm/model_executor/models/zamba2.py): Mamba-2
 * layers interleaved with hybrid layers whose shared attention/MLP block is stored once per bare block, reused by
 * several depths through per-use adapters.
 */
public class HfZamba2Test {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int H = 4;
    private static final int V = 7;
    private static final int NL = 4;
    private static final int NH = 2;
    private static final int HD = 4;
    private static final int F = 6;
    private static final int DI = 8;
    private static final int DS = 2;
    private static final int NG = 1;
    private static final int MH = 2;
    private static final int MHD = DI / MH;
    private static final int DCONV = 3;
    private static final int RANK = 2;
    private static final int CONV_CH = DI + 2 * NG * DS;
    private static final int IN_PROJ = DI + CONV_CH + MH;
    private static final float EPS = 1e-5f;
    private static final int[] HYBRID = {1, 3};
    private static final int[] TOKENS = {2, 3, 4};

    private static final float[] EMB = matrix(V, H, 1);
    private static final float[] FINAL_NORM = vec(H, 2);
    private static final float[] HEAD = matrix(V, H, 3);
    private static final float[] SHARED_ATTN_NORM = vec(2 * H, 10);
    private static final float[] SHARED_PRE_FF_NORM = vec(H, 11);
    private static final float[] Q_W = matrix(2 * H, 2 * H, 12);
    private static final float[] K_W = matrix(2 * H, 2 * H, 13);
    private static final float[] V_W = matrix(2 * H, 2 * H, 14);
    private static final float[] O_W = matrix(H, 2 * H, 15);
    private static final float[] GATE_UP_W = matrix(2 * F, H, 16);
    private static final float[] DOWN_W = matrix(H, F, 17);

    // ------------------------------------------------------------------ conversion

    /** The converter writes the shared blocks once and the per-use adapters under their own layers. */
    @Test
    public void conversionWritesTheZamba2Layout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("zamba2", md.get("general.architecture"));
            assertEquals(NL, ((Number) md.get("zamba2.block_count")).intValue());
            assertEquals(16, ((Number) md.get("zamba2.context_length")).intValue());
            assertEquals(H, ((Number) md.get("zamba2.embedding_length")).intValue());
            assertEquals(F, ((Number) md.get("zamba2.feed_forward_length")).intValue());
            assertEquals(NH, ((Number) md.get("zamba2.attention.head_count")).intValue());
            assertEquals(NH, ((Number) md.get("zamba2.attention.head_count_kv")).intValue());
            assertEquals(HD, ((Number) md.get("zamba2.attention.key_length")).intValue());
            assertEquals(HD, ((Number) md.get("zamba2.attention.value_length")).intValue());
            assertEquals(DI, ((Number) md.get("zamba2.ssm.inner_size")).intValue());
            assertEquals(DS, ((Number) md.get("zamba2.ssm.state_size")).intValue());
            assertEquals(DCONV, ((Number) md.get("zamba2.ssm.conv_kernel")).intValue());
            assertEquals(MH, ((Number) md.get("zamba2.ssm.time_step_rank")).intValue());
            assertEquals(NG, ((Number) md.get("zamba2.ssm.group_count")).intValue());
            assertEquals(HD, ((Number) md.get("zamba2.rope.dimension_count")).intValue());
            assertEquals(10000.0, ((Number) md.get("zamba2.rope.freq_base")).doubleValue(), 1e-6);
            assertEquals(1, ((Number) md.get("zamba2.shared_blocks.count")).intValue());
            assertEquals(RANK, ((Number) md.get("zamba2.shared_blocks.adapter_rank")).intValue());
            assertEquals(Boolean.TRUE, md.get("zamba2.shared_blocks.attention_adapter"));
            assertEquals(Boolean.TRUE, md.get("zamba2.shared_blocks.memory_rope"));
            Object flags = md.get("zamba2.shared_blocks.hybrid_layers");
            assertNotNull(flags);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("token_embd.weight"));
            assertTrue(tensors.contains("output_norm.weight"));
            assertTrue(tensors.contains("output.weight"));
            assertTrue(tensors.contains("blk.0.ssm_in.weight"));
            assertTrue(tensors.contains("blk.0.ssm_conv1d.weight"));
            assertTrue(tensors.contains("blk.0.ssm_conv1d.bias"));
            assertTrue(tensors.contains("blk.0.ssm_dt.bias"));
            assertTrue(tensors.contains("blk.0.ssm_a"));
            assertTrue(tensors.contains("blk.0.ssm_d"));
            assertTrue(tensors.contains("blk.0.ssm_norm.weight"));
            assertTrue(tensors.contains("blk.0.ssm_out.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_norm.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_q.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_k.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_v.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_output.weight"));
            assertTrue(tensors.contains("shared_blk.0.attn_pre_ff_norm.weight"));
            assertTrue(tensors.contains("shared_blk.0.ffn_gate_up.weight"));
            assertTrue(tensors.contains("shared_blk.0.ffn_down.weight"));
            assertTrue(tensors.contains("blk.1.shared_linear.weight"));
            assertTrue(tensors.contains("blk.1.ffn_gate_up_adapter_a.weight"));
            assertTrue(tensors.contains("blk.3.ffn_gate_up_adapter_a.weight"));
            assertTrue(tensors.contains("blk.1.attn_q_adapter_a.weight"));
            assertTrue(tensors.contains("blk.3.attn_q_adapter_b.weight"));
            assertFalse("the hybrid layer holds no attention weights of its own",
                    tensors.contains("blk.1.attn_q.weight"));
            assertTrue(tensors.contains("blk.1.attn_norm.weight"));
            long sharedBlocks = tensors.stream().filter(n -> n.startsWith("shared_blk.")).count();
            assertEquals("every shared block tensor is stored once", 8, sharedBlocks);
            assertEquals("one adapter pair per use and projection", 16,
                    tensors.stream().filter(n -> n.contains("_adapter_")).count());

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.ZAMBA2, spec.arch());
            assertEquals(Zamba2Spec.NO_SHARED_BLOCK, spec.layer(0).zamba2().sharedBlock());
            assertEquals(0, spec.layer(1).zamba2().sharedBlock());
            assertEquals(0, spec.layer(3).zamba2().sharedBlock());
            assertTrue(spec.layer(1).usesKv());
            assertFalse(spec.layer(0).usesKv());
            assertTrue(spec.layer(0).isRecurrent());
            assertTrue(spec.layer(1).isRecurrent());
            assertFalse(spec.layer(1).parallelAttention());
            assertNotNull(spec.layer(1).attention());
            assertNull(spec.layer(0).attention());
            assertEquals(Zamba2Spec.NO_SHARED_BLOCK, spec.layer(0).zamba2().sharedBlock());
        }
    }

    /** {@code use_long_context} rescales the RoPE base and the context length the way Zamba2Config does. */
    @Test
    public void longContextRescalesTheRopeBase() throws Exception {
        Map<String, String> overrides = new LinkedHashMap<>();
        overrides.put("use_long_context", "true");
        Path dir = builder(overrides).write(tmp.newFolder("long-context").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("long-context-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals(16384, ((Number) md.get("zamba2.context_length")).intValue());
            double expected = 10000.0 * Math.pow(8.0, HD / (HD - 2.0));
            assertEquals(expected, ((Number) md.get("zamba2.rope.freq_base")).doubleValue(), 1e-3);
        }
    }

    // ------------------------------------------------------------------ inference

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), config(EngineConfig.BackendMode.CPU))) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** The configuration without memory RoPE or shared-attention adapters runs the plain shared block. */
    @Test
    public void checkpointWithoutMemoryRopeOrAttentionAdaptersMatchesTheReference() throws Exception {
        Map<String, String> overrides = new LinkedHashMap<>();
        overrides.put("use_mem_rope", "false");
        overrides.put("use_shared_attention_adapter", "false");
        Path dir = builder(overrides).write(tmp.newFolder("no-rope").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("no-rope-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            Map<String, Object> md = src.getMetadata();
            assertEquals(Boolean.FALSE, md.get("zamba2.shared_blocks.attention_adapter"));
            assertEquals(Boolean.FALSE, md.get("zamba2.shared_blocks.memory_rope"));
            assertFalse(md.containsKey("zamba2.rope.dimension_count"));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(), config(EngineConfig.BackendMode.CPU))) {
            assertArrayEquals(reference(TOKENS, false, false),
                    model.forwardBackend().prefill(TOKENS, 0).clone(), 3e-4f);
        }
    }

    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** Decoding one token after a prefill reuses the shared attention's cache and the Mamba-2 state exactly. */
    @Test
    public void decodeContinuesThePrefill() throws Exception {
        Path path = checkpoint();
        int[] prefix = {2, 3};
        int next = 4;
        try (GGUFModel model = new GGUFModel(path.toString(), config(EngineConfig.BackendMode.CPU))) {
            model.forwardBackend().prefill(prefix, 0);
            float[] stepwise = model.forwardBackend().decode(next, prefix.length).clone();
            model.forwardBackend().reset();
            float[] whole = model.forwardBackend().prefill(new int[]{2, 3, 4}, 0).clone();
            assertArrayEquals(whole, stepwise, 3e-5f);
        }
    }

    /** The exported GGUF file loads through the GGUF source and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsZamba2() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("zamba2.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), config(EngineConfig.BackendMode.CPU))) {
            assertArrayEquals(reference(TOKENS), model.forwardBackend().prefill(TOKENS, 0).clone(), 3e-4f);
        }
    }

    /** A snapshot of the recurrent state rewinds the shared attention's cache rows with it. */
    @Test
    public void snapshotRestoresTheRecurrentStateAndCaches() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), snapshotConfig())) {
            model.forwardBackend().prefill(new int[]{2, 3}, 0);
            model.forwardBackend().memory().snapshot();
            float[] first = model.forwardBackend().decode(4, 2).clone();
            model.forwardBackend().memory().restore();
            float[] second = model.forwardBackend().decode(4, 2).clone();
            assertArrayEquals(first, second, 1e-6f);
        }
    }

    /** A session file restores the recurrent state and the per-use KV caches into a fresh model. */
    @Test
    public void sessionRoundTripRestoresTheModelState() throws Exception {
        Path path = checkpoint();
        Path session = tmp.newFolder("session").toPath().resolve("zamba2.session");
        float[] expected;
        try (GGUFModel model = new GGUFModel(path.toString(), config(EngineConfig.BackendMode.CPU))) {
            model.forwardBackend().prefill(new int[]{2, 3}, 0);
            model.saveSession(session);
            assertTrue(Files.size(session) > 0);
            expected = model.forwardBackend().decode(4, 2).clone();
        }
        try (GGUFModel model = new GGUFModel(path.toString(), config(EngineConfig.BackendMode.CPU))) {
            model.loadSession(session);
            assertArrayEquals(expected, model.forwardBackend().decode(4, 2).clone(), 1e-6f);
        }
    }

    /** A Zamba2 model keeps recurrent state, so a context shift refuses by name. */
    @Test
    public void contextShiftRefusesByTheRecurrentState() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), config(EngineConfig.BackendMode.CPU))) {
            model.forwardBackend().prefill(new int[]{2, 3, 4}, 0);
            HeapModelMemory memory = (HeapModelMemory) model.forwardBackend().memory();
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> memory.shiftPositions(1, 1));
            assertTrue(e.getMessage(), e.getMessage().contains("recurrent state"));
        }
    }

    /** Zamba2 is a supported generative architecture with a hybrid memory kind. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("Zamba2ForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.ZAMBA2);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(NormKind.RMS, traits.normKind());
        assertEquals(ActOp.GELU_ERF, traits.ffnAct());
        assertEquals(MemoryKind.HYBRID, traits.memoryKind());
        assertTrue(Arch.ZAMBA2.isRecurrent());
        assertTrue(Arch.ZAMBA2.isHybrid());
    }

    // ------------------------------------------------------------------ refusals

    /** A shared-block tensor whose shape contradicts the attention geometry refuses by tensor name. */
    @Test
    public void refusesASharedBlockTensorOfTheWrongShape() throws Exception {
        SyntheticHf.Builder b = builder();
        float[] wrong = matrix(H, 2 * H, 99);
        for (int il : HYBRID) {
            b.tensor("model.layers." + il + ".shared_transformer.self_attn.q_proj.weight", StDType.F32,
                    new long[]{H, 2 * H}, SyntheticHf.f32(wrong));
        }
        Path dir = b.write(tmp.newFolder("bad-shape").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> {
            try (GGUFModel model = new GGUFModel(dir.toString(), config(EngineConfig.BackendMode.CPU))) {
                assertNotNull(model);
            }
        });
        assertTrue(e.getMessage(), e.getMessage().contains("shared_blk.0.attn_q.weight"));
        assertTrue(e.getMessage(), e.getMessage().contains("Q width"));
    }

    @Test
    public void refusesConfigurationsTheSharedBlockCannotExpress() throws Exception {
        assertRejected(Collections.singletonMap("add_bias_linear", "true"), "add_bias_linear");
        assertRejected(Collections.singletonMap("hidden_act", "'silu'"), "hidden_act");
        assertRejected(Collections.singletonMap("num_key_value_heads", "1"), "num_key_value_heads");
        assertRejected(Collections.singletonMap("num_mem_blocks", "3"), "num_mem_blocks");
        assertRejected(Collections.singletonMap("layers_block_type", "['mamba']"), "hybrid");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        SyntheticHf.Builder b = builder(overrides);
        Path dir = b.write(tmp.newFolder("rejected-" + field).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder("rejected-" + field + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    // ------------------------------------------------------------------ checkpoint

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("zamba2").toPath());
    }

    private static EngineConfig config(EngineConfig.BackendMode backend) {
        return EngineConfig.builder().backend(backend).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(4).build();
    }

    private static EngineConfig snapshotConfig() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(16).ubatch(4).stateSnapshots(true).build()).build();
    }

    private static SyntheticHf.Builder builder() {
        return builder(Collections.<String, String>emptyMap());
    }

    private static SyntheticHf.Builder builder(Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides));
        boolean attnAdapters = !"false".equals(overrides.get("use_shared_attention_adapter"));
        t(b, "model.embed_tokens.weight", V, H, EMB);
        t(b, "model.final_layernorm.weight", H, 1, FINAL_NORM);
        t(b, "lm_head.weight", V, H, HEAD);
        for (int il = 0; il < NL; il++) {
            boolean hybrid = isHybrid(il);
            String mamba = hybrid ? "mamba_decoder.mamba." : "mamba.";
            String norm = hybrid ? "mamba_decoder.input_layernorm.weight" : "input_layernorm.weight";
            String p = "model.layers." + il + ".";
            t(b, p + norm, H, 1, attnNorm(il));
            t(b, p + mamba + "in_proj.weight", IN_PROJ, H, inProj(il));
            b.tensor(p + mamba + "conv1d.weight", StDType.F32, new long[]{CONV_CH, 1, DCONV},
                    SyntheticHf.f32(convW(il)));
            t(b, p + mamba + "conv1d.bias", CONV_CH, 1, convB(il));
            t(b, p + mamba + "dt_bias", MH, 1, dtBias(il));
            t(b, p + mamba + "A_log", MH, 1, aLog(il));
            t(b, p + mamba + "D", MH, 1, dSkip(il));
            t(b, p + mamba + "norm.weight", DI, 1, ssmNorm(il));
            t(b, p + mamba + "out_proj.weight", H, DI, outProj(il));
            if (hybrid) {
                t(b, p + "linear.weight", H, H, sharedLinear(il));
                String s = p + "shared_transformer.";
                t(b, s + "input_layernorm.weight", 2 * H, 1, SHARED_ATTN_NORM);
                t(b, s + "self_attn.q_proj.weight", 2 * H, 2 * H, Q_W);
                t(b, s + "self_attn.k_proj.weight", 2 * H, 2 * H, K_W);
                t(b, s + "self_attn.v_proj.weight", 2 * H, 2 * H, V_W);
                t(b, s + "self_attn.o_proj.weight", H, 2 * H, O_W);
                t(b, s + "pre_ff_layernorm.weight", H, 1, SHARED_PRE_FF_NORM);
                t(b, s + "feed_forward.gate_up_proj.weight", 2 * F, H, GATE_UP_W);
                t(b, s + "feed_forward.down_proj.weight", H, F, DOWN_W);
                int ordinal = ordinalOf(il);
                if (attnAdapters) {
                    adapters(b, s + "self_attn.linear_q_adapter_list." + ordinal, qA(il), qB(il));
                    adapters(b, s + "self_attn.linear_k_adapter_list." + ordinal, kA(il), kB(il));
                    adapters(b, s + "self_attn.linear_v_adapter_list." + ordinal, vA(il), vB(il));
                }
                adapters(b, s + "feed_forward.gate_up_proj_adapter_list." + ordinal, ffnA(il), ffnB(il));
            }
        }
        if (attnAdapters) {
            // A checkpoint may repeat the tied adapter of another use under the first layer of the block; the
            // converter keeps one copy.
            String s = "model.layers.1.shared_transformer.";
            adapters(b, s + "self_attn.linear_q_adapter_list.1", qA(3), qB(3));
            adapters(b, s + "feed_forward.gate_up_proj_adapter_list.1", ffnA(3), ffnB(3));
        }
        return b;
    }

    private static void adapters(SyntheticHf.Builder b, String prefix, float[] a, float[] bWeights) {
        t(b, prefix + ".0.weight", RANK, a.length / RANK, a);
        t(b, prefix + ".1.weight", bWeights.length / RANK, RANK, bWeights);
    }

    private static boolean isHybrid(int il) {
        for (int h : HYBRID) {
            if (h == il) {
                return true;
            }
        }
        return false;
    }

    private static int ordinalOf(int il) {
        int ordinal = 0;
        for (int i = 0; i < il; i++) {
            if (isHybrid(i)) {
                ordinal++;
            }
        }
        return ordinal;
    }

    private static String config(Map<String, String> overrides) {
        Map<String, String> f = new LinkedHashMap<>();
        f.put("architectures", "['Zamba2ForCausalLM']");
        f.put("model_type", "'zamba2'");
        f.put("vocab_size", Integer.toString(V));
        f.put("hidden_size", Integer.toString(H));
        f.put("num_hidden_layers", Integer.toString(NL));
        f.put("layers_block_type", "['mamba','hybrid','mamba','hybrid']");
        f.put("intermediate_size", Integer.toString(F));
        f.put("hidden_act", "'gelu'");
        f.put("num_attention_heads", Integer.toString(NH));
        f.put("num_key_value_heads", Integer.toString(NH));
        f.put("max_position_embeddings", "16");
        f.put("mamba_d_state", Integer.toString(DS));
        f.put("mamba_d_conv", Integer.toString(DCONV));
        f.put("mamba_expand", "2");
        f.put("mamba_ngroups", Integer.toString(NG));
        f.put("n_mamba_heads", Integer.toString(MH));
        f.put("num_mem_blocks", "1");
        f.put("adapter_rank", Integer.toString(RANK));
        f.put("use_shared_attention_adapter", "true");
        f.put("use_mem_rope", "true");
        f.put("use_long_context", "false");
        f.put("rope_theta", "10000.0");
        f.put("rms_norm_eps", "1e-5");
        f.put("tie_word_embeddings", "false");
        f.put("use_conv_bias", "true");
        f.put("add_bias_linear", "false");
        f.put("bos_token_id", "0");
        f.put("eos_token_id", "1");
        f.putAll(overrides);
        StringBuilder json = new StringBuilder("{");
        for (Map.Entry<String, String> e : f.entrySet()) {
            if (json.length() > 1) {
                json.append(',');
            }
            json.append('\'').append(e.getKey()).append("':").append(e.getValue());
        }
        return SyntheticHf.json(json.append('}').toString());
    }

    private static void t(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        long[] shape = cols == 1 ? new long[]{rows} : new long[]{rows, cols};
        b.tensor(name, StDType.F32, shape, SyntheticHf.f32(data));
    }

    // ------------------------------------------------------------------ weights

    private static float[] attnNorm(int il) { return vec(H, 20 + il); }
    private static float[] inProj(int il) { return matrix(IN_PROJ, H, 30 + il); }
    private static float[] convW(int il) { return matrix(CONV_CH, DCONV, 40 + il); }
    private static float[] convB(int il) { return vec(CONV_CH, 50 + il); }
    private static float[] dtBias(int il) { return vec(MH, 60 + il); }
    private static float[] aLog(int il) { return vec(MH, 70 + il); }
    private static float[] dSkip(int il) { return vec(MH, 80 + il); }
    private static float[] ssmNorm(int il) { return vec(DI, 90 + il); }
    private static float[] outProj(int il) { return matrix(H, DI, 100 + il); }
    private static float[] sharedLinear(int il) { return matrix(H, H, 110 + il); }
    private static float[] qA(int il) { return matrix(RANK, 2 * H, 140 + il); }
    private static float[] qB(int il) { return matrix(2 * H, RANK, 150 + il); }
    private static float[] kA(int il) { return matrix(RANK, 2 * H, 160 + il); }
    private static float[] kB(int il) { return matrix(2 * H, RANK, 170 + il); }
    private static float[] vA(int il) { return matrix(RANK, 2 * H, 180 + il); }
    private static float[] vB(int il) { return matrix(2 * H, RANK, 190 + il); }
    private static float[] ffnA(int il) { return matrix(RANK, H, 200 + il); }
    private static float[] ffnB(int il) { return matrix(2 * F, RANK, 210 + il); }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.035f;
        }
        return x;
    }

    private static float[] vec(int n, int seed) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = 1.0f + ((i + seed) % 3 - 1) * 0.05f;
        }
        return x;
    }

    // ------------------------------------------------------------------ scalar reference

    /** vLLM's {@code Zamba2ForCausalLM.forward} for one sequence, in double-free float arithmetic. */
    private static float[] reference(int[] tokens) {
        return reference(tokens, true, true);
    }

    /** {@link #reference(int[])} for a checkpoint with or without memory RoPE and attention adapters. */
    private static float[] reference(int[] tokens, boolean memRope, boolean attnAdapters) {
        int n = tokens.length;
        float[][] x = new float[n][];
        float[][] orig = new float[n][];
        for (int t = 0; t < n; t++) {
            orig[t] = row(EMB, tokens[t], H);
            x[t] = orig[t].clone();
        }
        for (int il = 0; il < NL; il++) {
            float[][] mambaIn = isHybrid(il) ? hybridInput(x, orig, il, memRope, attnAdapters) : x;
            float[][] mambaOut = mamba2(mambaIn, il);
            float[][] next = new float[n][];
            for (int t = 0; t < n; t++) {
                next[t] = add(x[t], mambaOut[t]);
            }
            x = next;
        }
        return linear(rms(x[n - 1], FINAL_NORM), HEAD);
    }

    /** The shared transformer block: concat, attention with its per-use adapters, gated MLP, shared_linear. */
    private static float[][] hybridInput(float[][] x, float[][] orig, int il, boolean memRope, boolean attnAdapters) {
        int n = x.length;
        float[][] q = new float[n][];
        float[][] k = new float[n][];
        float[][] v = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] h = rms(cat(x[t], orig[t]), SHARED_ATTN_NORM);
            q[t] = linear(h, Q_W);
            k[t] = linear(h, K_W);
            v[t] = linear(h, V_W);
            if (attnAdapters) {
                q[t] = add(q[t], adapter(h, qA(il), qB(il), 2 * H));
                k[t] = add(k[t], adapter(h, kA(il), kB(il), 2 * H));
                v[t] = add(v[t], adapter(h, vA(il), vB(il), 2 * H));
            }
            if (memRope) {
                rope(q[t], t);
                rope(k[t], t);
            }
        }
        float[][] out = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] o = linear(attention(q, k, v, t), O_W);
            float[] pre = rms(o, SHARED_PRE_FF_NORM);
            float[] gateUp = add(linear(pre, GATE_UP_W), adapter(pre, ffnA(il), ffnB(il), 2 * F));
            float[] act = new float[F];
            for (int j = 0; j < F; j++) {
                act[j] = gelu(gateUp[j]) * gateUp[F + j];
            }
            out[t] = add(x[t], linear(linear(act, DOWN_W), sharedLinear(il)));
        }
        return out;
    }

    private static float[] attention(float[][] q, float[][] k, float[][] v, int t) {
        float[] out = new float[NH * HD];
        float scale = (float) Math.pow(HD / 2.0, -0.5);
        for (int h = 0; h < NH; h++) {
            float[] scores = new float[t + 1];
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j <= t; j++) {
                float dot = 0;
                for (int d = 0; d < HD; d++) {
                    dot += q[t][h * HD + d] * k[j][h * HD + d];
                }
                scores[j] = dot * scale;
                max = Math.max(max, scores[j]);
            }
            float sum = 0;
            for (int j = 0; j <= t; j++) {
                scores[j] = (float) Math.exp(scores[j] - max);
                sum += scores[j];
            }
            for (int j = 0; j <= t; j++) {
                for (int d = 0; d < HD; d++) {
                    out[h * HD + d] += scores[j] / sum * v[j][h * HD + d];
                }
            }
        }
        return out;
    }

    /** Whole-head NEOX RoPE, the pairing vLLM's {@code is_neox_style=True} rotation uses. */
    private static void rope(float[] x, int pos) {
        for (int h = 0; h < NH; h++) {
            int base = h * HD;
            for (int j = 0; j < HD / 2; j++) {
                double theta = pos * Math.pow(10000.0, -2.0 * j / HD);
                float cos = (float) Math.cos(theta);
                float sin = (float) Math.sin(theta);
                float a = x[base + j];
                float b = x[base + j + HD / 2];
                x[base + j] = a * cos - b * sin;
                x[base + j + HD / 2] = b * cos + a * sin;
            }
        }
    }

    /** vLLM's Mamba-2 mixer (MambaMixer2): conv, selective scan, D skip, z gate, grouped norm, out_proj. */
    private static float[][] mamba2(float[][] in, int il) {
        int n = in.length;
        float[][] proj = new float[n][];
        for (int t = 0; t < n; t++) {
            proj[t] = linear(rms(in[t], attnNorm(il)), inProj(il));
        }
        float[][] conv = new float[n][CONV_CH];
        float[] w = convW(il);
        float[] cb = convB(il);
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < CONV_CH; c++) {
                float sum = 0;
                for (int k = 0; k < DCONV; k++) {
                    int src = t + k - (DCONV - 1);
                    if (src >= 0) {
                        sum += proj[src][DI + c] * w[c * DCONV + k];
                    }
                }
                conv[t][c] = silu(sum + cb[c]);
            }
        }
        float[][] y = new float[n][DI];
        float[] scan = new float[DI * DS];
        float[] a = aLog(il);
        float[] d = dSkip(il);
        float[] nb = dtBias(il);
        int gs = NG * DS;
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < MH; h++) {
                float dt = softplus(proj[t][DI + CONV_CH + h] + nb[h]);
                float decay = (float) Math.exp(dt * -(float) Math.exp(a[h]));
                for (int p = 0; p < MHD; p++) {
                    int ch = h * MHD + p;
                    float xDt = conv[t][ch] * dt;
                    for (int s = 0; s < DS; s++) {
                        int idx = ch * DS + s;
                        scan[idx] = scan[idx] * decay + conv[t][DI + s] * xDt;
                    }
                }
            }
            for (int h = 0; h < MH; h++) {
                for (int p = 0; p < MHD; p++) {
                    int ch = h * MHD + p;
                    float sum = 0;
                    for (int s = 0; s < DS; s++) {
                        sum += scan[ch * DS + s] * conv[t][DI + gs + s];
                    }
                    y[t][ch] = sum + d[h] * conv[t][ch];
                }
            }
        }
        float[][] out = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] gated = new float[DI];
            for (int c = 0; c < DI; c++) {
                gated[c] = y[t][c] * silu(proj[t][c]);
            }
            out[t] = linear(groupRms(gated, ssmNorm(il)), outProj(il));
        }
        return out;
    }

    // ------------------------------------------------------------------ scalar primitives

    private static float[] row(float[] w, int r, int cols) {
        float[] out = new float[cols];
        System.arraycopy(w, r * cols, out, 0, cols);
        return out;
    }

    private static float[] linear(float[] x, float[] w) {
        int rows = w.length / x.length;
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            float sum = 0;
            for (int j = 0; j < x.length; j++) {
                sum += w[i * x.length + j] * x[j];
            }
            y[i] = sum;
        }
        return y;
    }

    private static float[] adapter(float[] x, float[] a, float[] b, int width) {
        return linear(linear(x, a), b);
    }

    private static float[] add(float[] a, float[] b) {
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = a[i] + b[i];
        }
        return out;
    }

    private static float[] cat(float[] a, float[] b) {
        float[] out = new float[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static float[] rms(float[] x, float[] w) {
        float mean = 0;
        for (float v : x) {
            mean += v * v / x.length;
        }
        float scale = (float) (1.0 / Math.sqrt(mean + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale * w[i];
        }
        return y;
    }

    private static float[] groupRms(float[] x, float[] w) {
        float[] y = new float[x.length];
        int gs = x.length / NG;
        for (int g = 0; g < NG; g++) {
            float mean = 0;
            for (int i = 0; i < gs; i++) {
                mean += x[g * gs + i] * x[g * gs + i] / gs;
            }
            float scale = (float) (1.0 / Math.sqrt(mean + EPS));
            for (int i = 0; i < gs; i++) {
                y[g * gs + i] = x[g * gs + i] * scale * w[g * gs + i];
            }
        }
        return y;
    }

    private static float silu(float x) {
        return x / (1.0f + (float) Math.exp(-x));
    }

    private static float softplus(float x) {
        return x > 20.0f ? x : (float) Math.log(1.0 + Math.exp(x));
    }

    private static float gelu(float x) {
        return 0.5f * x * (1.0f + erf(x / (float) Math.sqrt(2.0)));
    }

    /** {@code erf} to about 1e-7 (Abramowitz-Stegun 7.1.26). */
    private static float erf(float x) {
        float sign = x < 0 ? -1.0f : 1.0f;
        float ax = Math.abs(x);
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f
                + t * 1.061405429f))));
        return sign * (1.0f - poly * (float) Math.exp(-ax * ax));
    }
}
