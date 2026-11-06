package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFModelGPU;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.ArchTraits;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.GenerationRequest;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;
import com.eonmux.neuralfree.gguf.engine.batch.BatchEngine;
import com.eonmux.neuralfree.gguf.engine.batch.BatchSequence;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlanner;
import com.eonmux.neuralfree.gguf.spec.LayerSchedule;
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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * Dory checkpoint loading, the repeat-block schedule with one KV cache per loop pass, and causal inference against a
 * scalar transcription of vLLM's {@code dory.py} forward.
 */
public class HfDoryTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private int rejected;
    private static final int D = 8, V = 7, FF = 8, HEADS = 2, KV_HEADS = 1, HEAD_DIM = 4;
    private static final int LAYERS = 4;
    private static final int N_INPUT = 1, N_RECURRENT = 2, N_OUTPUT = 1, LOOPS = 2;
    private static final int SLOTS = N_INPUT + N_RECURRENT * LOOPS + N_OUTPUT;
    private static final int WINDOW = 3;
    private static final float EPS = 1e-6f;
    private static final float INIT = 0.02f;
    private static final float ALPHA_INIT = 0.05f;
    private static final float ROPE_BASE = 10_000.0f;
    private static final float ROPE_BASE_SWA = 5_000.0f;
    /** Layer 0 and 3 are full attention, layer 1 is sliding attention and layer 2 is an MLP-only block. */
    private static final String[] TYPES = {"full_attention", "sliding_attention", "mlp", "full_attention"};
    private static final int[] TOKENS = {2, 4, 6, 1, 3, 5};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** The same forward through the OpenCL backend: the per-pass caches and the per-layer RoPE bases included. */
    @Test
    public void checkpointMatchesScalarForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory()).build();
        try (GGUFModelGPU model = new GGUFModelGPU(path.toString(), null, cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** The exported GGUF file (NeuralFree's {@code dory} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsDory() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("dory.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** Dory is a supported text architecture with its own GGUF name and the nGPT slerp residual. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("DoryForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.DORY);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(ResidualKind.SLERP, traits.residualKind());
    }

    /** The converter writes the Dory metadata, merges the nGPT scales and produces the graph's tensors. */
    @Test
    public void conversionWritesTheDoryLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("dory", md.get("general.architecture"));
            assertEquals(LAYERS, ((Number) md.get("dory.block_count")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("dory.attention.key_length")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("dory.attention.value_length")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("dory.rope.dimension_count")).intValue());
            assertEquals(ROPE_BASE, ((Number) md.get("dory.rope.freq_base")).floatValue(), 1e-3f);
            assertEquals(ROPE_BASE_SWA, ((Number) md.get("dory.rope.freq_base_swa")).floatValue(), 1e-3f);
            assertEquals(1f / (float) Math.sqrt(HEAD_DIM),
                    ((Number) md.get("dory.attention.scale")).floatValue(), 1e-6f);
            assertEquals(WINDOW, ((Number) md.get("dory.attention.sliding_window")).intValue());
            assertEquals(java.util.Arrays.asList(false, true, false, false),
                    md.get("dory.attention.sliding_window_pattern"));
            assertEquals(java.util.Arrays.asList(false, false, true, false),
                    md.get("dory.mlp_layer_pattern"));
            assertEquals(N_INPUT, ((Number) md.get("dory.recurrent_block.n_input_layers")).intValue());
            assertEquals(N_RECURRENT, ((Number) md.get("dory.recurrent_block.n_recurrent_layers")).intValue());
            assertEquals(N_OUTPUT, ((Number) md.get("dory.recurrent_block.n_output_layers")).intValue());
            assertEquals(LOOPS, ((Number) md.get("dory.recurrent_block.num_loops")).intValue());
            assertEquals(Boolean.TRUE, md.get("dory.recurrent_block.justnorm"));
            assertEquals(1e-6f, ((Number) md.get("dory.recurrent_block.justnorm_epsilon")).floatValue(), 1e-9f);
            assertEquals(ALPHA_INIT / INIT, ((Number) md.get("dory.slerp_alpha_scale")).floatValue(), 1e-6f);

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_q_norm.weight"));
            assertTrue(tensors.contains("blk.0.attn_gate.weight"));
            assertTrue(tensors.contains("blk.0.attn_slerp_alpha.weight"));
            assertFalse("K is justnormed without a weight", tensors.contains("blk.0.attn_k_norm.weight"));
            assertTrue(tensors.contains("blk.2.mlp_slerp_alpha.weight") || tensors.contains("blk.2.ffn_slerp_alpha.weight"));
            assertTrue(tensors.contains("output.weight"));
            assertTrue(tensors.contains("output_logit_scale.weight"));
            assertFalse("the nGPT scales are folded into the weights", tensors.contains("blk.0.sqk"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.DORY, spec.arch());
            assertEquals(LayerSchedule.Kind.REPEAT, spec.schedule().kind());
            assertEquals(SLOTS, spec.schedule().slots());
            assertFalse("Dory has no final output norm", spec.head().outputNorm() != null);
            assertFalse("Dory has no block pre-norms", spec.layer(0).norm().attnNorm());
            assertEquals(QkNormKind.L2_HEAD, spec.layer(0).norm().qkNorm());
            assertEquals(QkNormKind.L2_HEAD, spec.layer(0).attention().qkNorm());
            assertTrue(spec.layer(0).attention().qkNormAfterRope());
            assertEquals(ResidualKind.SLERP, spec.layer(0).residual().kind());
            assertEquals(MixerKind.ATTENTION, spec.layer(0).mixerKind());
            assertEquals(MixerKind.NONE, spec.layer(2).mixerKind());
            assertTrue(spec.layer(2).ffn().justNorm());
            assertNotNull(spec.head().logitScaleVector());
            assertEquals(V, spec.head().logitScaleVector().ne(0));
            assertEquals(ROPE_BASE, spec.layer(0).rope().freqBase(), 1e-3f);
            assertEquals(ROPE_BASE_SWA, spec.layer(1).rope().freqBase(), 1e-3f);
            assertEquals(HEAD_DIM, spec.layer(1).rope().nDims());
        }
    }

    /** Every pass of the repeated block is a slot with its own cache, and the blocks keep their physical weights. */
    @Test
    public void repeatScheduleGivesEveryPassItsOwnCache() throws Exception {
        ModelSpec spec = spec();
        LayerSchedule s = spec.schedule();
        assertArrayEquals(new int[]{0, 1, 2, 1, 2, 3}, weightLayers(s));
        assertArrayEquals(new int[]{-1, 0, 0, 1, 1, -1}, loopIndexes(s));
        assertEquals(1, s.recurrentStart());
        assertEquals(5, s.recurrentEnd());
        assertTrue(s.loopStart(1));
        assertTrue(s.loopEnd(2));
        assertTrue(s.loopStart(3));
        assertTrue(s.loopEnd(4));

        MemoryPlan plan = MemoryPlanner.planForContext(spec, memory(), 32);
        assertEquals(SLOTS, plan.layers().size());
        for (int slot = 0; slot < SLOTS; slot++) {
            LayerSpec layer = spec.layer(slot);
            assertEquals(slot, layer.il());
            assertEquals(s.weightLayer(slot), layer.weightLayer());
            if (layer.mixerKind() == MixerKind.NONE) {
                assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(slot).kind());
                continue;
            }
            assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(slot).kind());
            assertEquals("slot " + slot + " owns its cache", slot, layer.attention().kvOwnerLayer());
        }
        assertNotEquals("the two passes of block 1 keep separate caches", plan.layer(1).kv(), plan.layer(3).kv());
    }

    /** A session file moves every pass's cache and restores the same logits. */
    @Test
    public void sessionRestoresEveryLoopPass() throws Exception {
        Path path = checkpoint();
        Path file = tmp.getRoot().toPath().resolve("dory.session");
        float[] before;
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(TOKENS, 0);
            model.saveSession(file);
            before = model.forwardBackend().decode(3, TOKENS.length).clone();
        }
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.loadSession(file);
            float[] after = model.forwardBackend().decode(3, TOKENS.length).clone();
            assertArrayEquals(before, after, 1e-5f);
        }
    }

    /** Paged memory refuses the session by name (its rows live in a page pool, not in the session file's scope). */
    @Test
    public void pagedMemoryRefusesTheSessionByName() throws Exception {
        Path path = checkpoint();
        MemoryConfig paged = memory().toBuilder().kvPageSize(4).parallel(2).prefixReuse(true).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(paged).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> model.saveSession(tmp.getRoot().toPath().resolve("paged.session")));
            assertTrue(e.getMessage(), e.getMessage().contains("paged"));
        }
    }

    /** The context shift either moves every pass's cache or refuses by name. */
    @Test
    public void contextShiftCoversEveryLoopPass() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(TOKENS, 0);
            String refusal = model.forwardBackend().shiftSequenceRefusal();
            if (refusal != null) {
                assertFalse(refusal.trim().isEmpty());
                return;
            }
            model.forwardBackend().shiftSequence(0, 2, 1);
            assertEquals(V, model.forwardBackend().decode(TOKENS[0], TOKENS.length - 1).length);
        }
    }

    /** Paged memory plans one pool per window geometry and keeps every pass's rows distinct. */
    @Test
    public void pagedPlanKeepsOneCachePerPass() throws Exception {
        ModelSpec spec = spec();
        MemoryConfig paged = memory().toBuilder().kvPageSize(4).parallel(2).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, paged, 32);
        assertEquals(SLOTS, plan.layers().size());
        for (int slot = 0; slot < SLOTS; slot++) {
            if (spec.layer(slot).mixerKind() == MixerKind.NONE) {
                assertEquals(LayerMemoryPlan.Kind.NONE, plan.layer(slot).kind());
                continue;
            }
            assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(slot).kind());
            assertTrue(plan.layer(slot).kv().rows() > 0);
        }
        assertNotEquals(plan.layer(1).kv(), plan.layer(3).kv());
        assertTrue(plan.group(0).paged());

    }

    /** The batch engine over paged memory keeps every pass's cache and reuses the prefix on a second prompt. */
    @Test
    public void pagedBatchEngineReusesThePrefixAcrossEveryPass() throws Exception {
        Path path = checkpoint();
        MemoryConfig paged = memory().toBuilder().kvPageSize(4).parallel(2).prefixReuse(true).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(paged).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(1)
                .sampler(SamplerConfig.greedyDecoding()).build();
        int expected = argmax(reference(TOKENS));
        try (GGUFModel model = new GGUFModel(path.toString(), cfg);
             BatchEngine engine = BatchEngine.forModel(model, cfg)) {
            BatchSequence first = engine.submit(TOKENS, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(BatchSequence.State.DONE, first.state());
            assertEquals(expected, first.result().tokens()[0]);
            BatchSequence again = engine.submit(TOKENS, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(expected, again.result().tokens()[0]);
            assertTrue("the shared prompt rows come from the block pool", again.reusedPrefix() > 0);
        }
    }

    /** The same paged batch run on the device: the per-pass caches and the block pool. */
    @Test
    public void pagedBatchEngineReusesThePrefixAcrossEveryPassOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint();
        MemoryConfig paged = memory().toBuilder().kvPageSize(4).parallel(2).prefixReuse(true).build();
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(paged).build();
        GenerationRequest request = GenerationRequest.builder().maxNewTokens(1)
                .sampler(SamplerConfig.greedyDecoding()).build();
        int expected = argmax(reference(TOKENS));
        try (GGUFModelGPU model = new GGUFModelGPU(path.toString(), null, cfg);
             BatchEngine engine = BatchEngine.forModel(model, cfg)) {
            BatchSequence first = engine.submit(TOKENS, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(expected, first.result().tokens()[0]);
            BatchSequence again = engine.submit(TOKENS, request,
                    (com.eonmux.neuralfree.gguf.TokenCallback) null);
            engine.drain();
            assertEquals(expected, again.result().tokens()[0]);
            assertTrue(again.reusedPrefix() > 0);
        }
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("hidden_act", "'relu'"), "hidden_act");
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        assertRejected(Collections.singletonMap("mlp_bias", "true"), "mlp_bias");
        assertRejected(Collections.singletonMap("tie_word_embeddings", "true"), "tie_word_embeddings");
        assertRejected(Collections.singletonMap("recurrent_kv_cache_mode", "'last_loop'"),
                "recurrent_kv_cache_mode");
        assertRejected(Collections.singletonMap("layer_types",
                "['full_attention','mlp','linear_attention','full_attention']"), "layer_types");
        assertRejected(Collections.singletonMap("n_recurrent_layers", "1"), "n_recurrent_layers");
        assertRejected(Collections.singletonMap("n_recurrent_loops", "0"), "n_recurrent_layers");
        assertRejected(Collections.singletonMap("sliding_window", "0"), "sliding_window");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'full_attention':{'rope_type':'linear','factor':2.0,'rope_theta':10000.0},"
                        + "'sliding_attention':{'rope_theta':5000.0}}"), "rope_parameters");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'full_attention':{'rope_theta':10000.0}}"), "rope_parameters");
    }

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        String name = "rejected-" + (rejected++);
        Path dir = builder(overrides).write(tmp.newFolder(name).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint() throws Exception {
        return builder().write(tmp.newFolder("dory").toPath());
    }

    private ModelSpec spec() throws Exception {
        Path dir = builder().write(tmp.newFolder("spec").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("spec-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            return ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
        }
    }

    private static int[] weightLayers(LayerSchedule s) {
        int[] out = new int[s.slots()];
        for (int i = 0; i < out.length; i++) {
            out[i] = s.weightLayer(i);
        }
        return out;
    }

    private static int[] loopIndexes(LayerSchedule s) {
        int[] out = new int[s.slots()];
        for (int i = 0; i < out.length; i++) {
            out[i] = s.loopIndex(i);
        }
        return out;
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
        b.tensor("backbone.embeddings.weight", StDType.F32, new long[]{V, D}, SyntheticHf.f32(matrix(V, D, 1)));
        b.tensor("lm_head.weight", StDType.F32, new long[]{V, D}, SyntheticHf.f32(matrix(V, D, 40)));
        b.tensor("lm_head.logit_scale", StDType.F32, new long[]{V}, SyntheticHf.f32(logitScale()));
        for (int il = 0; il < LAYERS; il++) {
            String p = "backbone.layers." + il + ".";
            if ("mlp".equals(TYPES[il])) {
                tensor(b, p + "mixer.gate_proj.weight", FF, D, matrix(FF, D, 20 + il));
                tensor(b, p + "mixer.up_proj.weight", FF, D, matrix(FF, D, 22 + il));
                tensor(b, p + "mixer.down_proj.weight", D, FF, matrix(D, FF, 24 + il));
                b.tensor(p + "mixer.suv_gate", StDType.F32, new long[]{FF}, SyntheticHf.f32(suv(30 + il)));
                b.tensor(p + "mixer.suv_up", StDType.F32, new long[]{FF}, SyntheticHf.f32(suv(32 + il)));
                b.tensor(p + "mlp_alpha", StDType.F32, new long[]{D}, SyntheticHf.f32(alpha(34 + il)));
            } else {
                int q = HEADS * HEAD_DIM, kv = KV_HEADS * HEAD_DIM;
                tensor(b, p + "mixer.q_proj.weight", q, D, matrix(q, D, 10 + il));
                tensor(b, p + "mixer.k_proj.weight", kv, D, matrix(kv, D, 12 + il));
                tensor(b, p + "mixer.v_proj.weight", kv, D, matrix(kv, D, 14 + il));
                tensor(b, p + "mixer.o_proj.weight", D, q, matrix(D, q, 16 + il));
                tensor(b, p + "mixer.g_proj.weight", q, D, matrix(q, D, 18 + il));
                b.tensor(p + "mixer.sqk", StDType.F32, new long[]{q}, SyntheticHf.f32(sqk(36 + il)));
                b.tensor(p + "mixer.s_gate", StDType.F32, new long[]{q}, SyntheticHf.f32(sGate(38 + il)));
                b.tensor(p + "attn_alpha", StDType.F32, new long[]{D}, SyntheticHf.f32(alpha(40 + il)));
            }
        }
        return b;
    }

    /** The Dory {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['DoryForCausalLM']");
        fields.put("model_type", "'dory'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("num_hidden_layers", Integer.toString(LAYERS));
        fields.put("num_attention_heads", Integer.toString(HEADS));
        fields.put("num_key_value_heads", Integer.toString(KV_HEADS));
        fields.put("head_dim", Integer.toString(HEAD_DIM));
        fields.put("max_position_embeddings", "128");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("hidden_act", "'silu'");
        fields.put("initializer_range", Float.toString(INIT));
        fields.put("sliding_window", Integer.toString(WINDOW));
        fields.put("layer_types", typesJson());
        fields.put("ngpt_alpha_init", Float.toString(ALPHA_INIT));
        fields.put("mlp_bias", "false");
        fields.put("attention_bias", "false");
        fields.put("n_input_layers", Integer.toString(N_INPUT));
        fields.put("n_recurrent_layers", Integer.toString(N_RECURRENT));
        fields.put("n_output_layers", Integer.toString(N_OUTPUT));
        fields.put("n_recurrent_loops", Integer.toString(LOOPS));
        fields.put("recurrent_kv_cache_mode", "'per_loop'");
        fields.put("rope_parameters", "{'full_attention':{'rope_theta':" + ROPE_BASE + "},"
                + "'sliding_attention':{'rope_theta':" + ROPE_BASE_SWA + "}}");
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

    private static String typesJson() {
        StringBuilder s = new StringBuilder("[");
        for (int i = 0; i < TYPES.length; i++) {
            if (i > 0) {
                s.append(',');
            }
            s.append('\'').append(TYPES[i]).append('\'');
        }
        return s.append(']').toString();
    }

    private static void tensor(SyntheticHf.Builder b, String name, int rows, int cols, float[] data) {
        b.tensor(name, StDType.F32, new long[]{rows, cols}, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.03f;
        }
        return x;
    }

    private static float[] sqk(int seed) {
        float[] x = new float[HEADS * HEAD_DIM];
        for (int i = 0; i < x.length; i++) {
            x[i] = 0.018f + 0.0005f * (i % 5) + 0.00001f * seed;
        }
        return x;
    }

    private static float[] sGate(int seed) {
        float[] x = new float[HEADS * HEAD_DIM];
        for (int i = 0; i < x.length; i++) {
            x[i] = 0.02f + 0.0004f * (i % 4) + 0.00001f * seed;
        }
        return x;
    }

    private static float[] suv(int seed) {
        float[] x = new float[FF];
        float neutral = (float) (1.0 / Math.sqrt(D));
        for (int i = 0; i < x.length; i++) {
            x[i] = neutral + 0.001f * (i % 3) + 0.00001f * seed;
        }
        return x;
    }

    private static float[] alpha(int seed) {
        float[] x = new float[D];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 5 + seed) % 11 - 5) * 0.04f;
        }
        return x;
    }

    private static float[] logitScale() {
        float[] x = new float[V];
        for (int i = 0; i < x.length; i++) {
            x[i] = 1.0f + 0.1f * i;
        }
        return x;
    }

    // ---------------------------------------------------------------- scalar reference

    /** A scalar transcription of vLLM's {@code DoryBackbone.forward}: the repeat schedule, both block kinds. */
    private static float[] reference(int[] tokens) {
        float[] embeddings = matrix(V, D, 1);
        float[] head = matrix(V, D, 40);
        float[] logitScale = logitScale();
        float[][] hidden = new float[tokens.length][];
        for (int t = 0; t < tokens.length; t++) {
            hidden[t] = embeddingOf(embeddings, tokens[t]);
        }
        for (int il = 0; il < N_INPUT; il++) {
            hidden = applyLayer(il, hidden, tokens.length);
        }
        float[][] inputH = copy(hidden);
        float[][] state = copy(inputH);
        for (int loop = 0; loop < LOOPS; loop++) {
            float[][] combined = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                combined[t] = justNorm(add(state[t], inputH[t]));
            }
            hidden = combined;
            for (int j = 0; j < N_RECURRENT; j++) {
                hidden = applyLayer(N_INPUT + j, hidden, tokens.length);
            }
            float[][] next = new float[tokens.length][];
            for (int t = 0; t < tokens.length; t++) {
                next[t] = justNorm(hidden[t]);
            }
            state = next;
        }
        hidden = state;
        for (int il = N_INPUT + N_RECURRENT; il < LAYERS; il++) {
            hidden = applyLayer(il, hidden, tokens.length);
        }
        float[] last = hidden[tokens.length - 1];
        float[] logits = new float[V];
        for (int v = 0; v < V; v++) {
            float dot = 0;
            for (int k = 0; k < D; k++) {
                dot += last[k] * head[v * D + k];
            }
            logits[v] = dot * (logitScale[v] / INIT);
        }
        return logits;
    }

    /** One physical block: a SwiGLU MLP with a justnorm, or the gated GQA attention, then the slerp residual. */
    private static float[][] applyLayer(int il, float[][] hidden, int n) {
        float[] alpha = alpha(("mlp".equals(TYPES[il]) ? 34 : 40) + il);
        if ("mlp".equals(TYPES[il])) {
            float[][] out = new float[n][];
            float[][] self = copy(hidden);
            for (int t = 0; t < n; t++) {
                float[] g = linear(hidden[t], matrix(FF, D, 20 + il), FF);
                float[] u = linear(hidden[t], matrix(FF, D, 22 + il), FF);
                float[] suvG = suv(30 + il), suvU = suv(32 + il);
                for (int k = 0; k < FF; k++) {
                    g[k] = silu(g[k] * (suvG[k] * (float) Math.sqrt(D))) * (u[k] * (suvU[k] * (float) Math.sqrt(D)));
                }
                g = justNorm(g);
                float[] down = linear(g, matrix(D, FF, 24 + il), D);
                out[t] = slerp(self[t], down, alpha);
            }
            return out;
        }
        float base = "sliding_attention".equals(TYPES[il]) ? ROPE_BASE_SWA : ROPE_BASE;
        int qSize = HEADS * HEAD_DIM, kvSize = KV_HEADS * HEAD_DIM;
        float[] qW = matrix(qSize, D, 10 + il), kW = matrix(kvSize, D, 12 + il), vW = matrix(kvSize, D, 14 + il);
        float[] oW = matrix(D, qSize, 16 + il), gW = matrix(qSize, D, 18 + il);
        float[] sqk = sqk(36 + il), sGate = sGate(38 + il);
        float qkFactor = (float) (Math.sqrt(HEAD_DIM) / INIT);
        float gateFactor = (float) (Math.sqrt(Math.sqrt(D)) / INIT);
        float attnScale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        float[][] q = new float[n][], k = new float[n][], v = new float[n][], gate = new float[n][];
        for (int t = 0; t < n; t++) {
            q[t] = linear(hidden[t], qW, qSize);
            k[t] = linear(hidden[t], kW, kvSize);
            v[t] = linear(hidden[t], vW, kvSize);
            gate[t] = linear(hidden[t], gW, qSize);
            for (int h = 0; h < HEADS; h++) {
                rotateHalf(q[t], h * HEAD_DIM, HEAD_DIM, t, base);
                normHeadInPlace(q[t], h, HEAD_DIM, sqk, qkFactor);
            }
            for (int h = 0; h < KV_HEADS; h++) {
                rotateHalf(k[t], h * HEAD_DIM, HEAD_DIM, t, base);
                normHeadPlain(k[t], h, HEAD_DIM);
            }
        }
        float[][] out = new float[n][];
        for (int t = 0; t < n; t++) {
            float[] attn = new float[qSize];
            for (int h = 0; h < HEADS; h++) {
                int off = h * HEAD_DIM;
                int kvHead = h / (HEADS / KV_HEADS);
                float[] weights = new float[t + 1];
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j <= t; j++) {
                    if (!visible(il, t, j)) {
                        continue;
                    }
                    float dot = 0;
                    for (int kk = 0; kk < HEAD_DIM; kk++) {
                        dot += q[t][off + kk] * k[j][kvHead * HEAD_DIM + kk];
                    }
                    weights[j] = dot * attnScale;
                    max = Math.max(max, weights[j]);
                }
                float total = 0;
                for (int j = 0; j <= t; j++) {
                    if (visible(il, t, j)) {
                        weights[j] = (float) Math.exp(weights[j] - max);
                        total += weights[j];
                    }
                }
                for (int j = 0; j <= t; j++) {
                    for (int kk = 0; kk < HEAD_DIM; kk++) {
                        attn[off + kk] += weights[j] / total * v[j][kvHead * HEAD_DIM + kk];
                    }
                }
            }
            for (int i = 0; i < qSize; i++) {
                float sg = sGate[i] * gateFactor;
                attn[i] *= sigmoid(gate[t][i] * sg * sg);
            }
            attn = justNorm(attn);
            float[] o = linear(attn, oW, D);
            out[t] = slerp(hidden[t], o, alpha);
        }
        return out;
    }

    private static boolean visible(int il, int pos, int j) {
        if (j > pos) {
            return false;
        }
        return !"sliding_attention".equals(TYPES[il]) || pos - j < WINDOW;
    }

    private static void normHeadInPlace(float[] x, int head, int headDim, float[] sqk, float factor) {
        int off = head * headDim;
        float[] slice = new float[headDim];
        System.arraycopy(x, off, slice, 0, headDim);
        float[] normed = justNorm(slice);
        for (int i = 0; i < headDim; i++) {
            float scale = sqk[off + i] * factor;
            x[off + i] = normed[i] * scale * scale;
        }
    }

    private static void normHeadPlain(float[] x, int head, int headDim) {
        int off = head * headDim;
        float[] slice = new float[headDim];
        System.arraycopy(x, off, slice, 0, headDim);
        float[] normed = justNorm(slice);
        System.arraycopy(normed, 0, x, off, headDim);
    }

    private static float[] justNorm(float[] x) {
        double sum = 0;
        for (float v : x) {
            sum += v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * inv;
        }
        return y;
    }

    private static float[] slerp(float[] a, float[] b, float[] alpha) {
        float[] ya = justNorm(a), yb = justNorm(b);
        float[] mixed = new float[a.length];
        for (int i = 0; i < mixed.length; i++) {
            float lr = Math.abs(alpha[i] * (ALPHA_INIT / INIT));
            mixed[i] = ya[i] + lr * (yb[i] - ya[i]);
        }
        return justNorm(mixed);
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

    private static float[] add(float[] a, float[] b) {
        float[] y = new float[a.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    private static float[][] copy(float[][] x) {
        float[][] y = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i].clone();
        }
        return y;
    }

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

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }

    private static float[] embeddingOf(float[] embeddings, int token) {
        float[] v = new float[D];
        System.arraycopy(embeddings, token * D, v, 0, D);
        return v;
    }
}
