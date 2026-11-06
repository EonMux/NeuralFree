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
import com.eonmux.neuralfree.gguf.arch.ResidualKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
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
 * IQuestLoopCoder checkpoint loading, the loop schedule with one KV cache per pass and the gated global/local
 * attention of the later passes, checked against a scalar transcription of vLLM's {@code iquest_loopcoder.py}.
 */
public class HfIQuestTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();
    private int rejected;
    private static final int D = 8, V = 7, FF = 8, HEADS = 2, KV_HEADS = 1, HEAD_DIM = 4;
    private static final int LAYERS = 2;
    private static final int LOOPS = 2;
    private static final int SLOTS = LAYERS * LOOPS;
    private static final int WINDOW = 2;
    private static final float EPS = 1e-6f;
    private static final float ROPE_BASE = 10_000.0f;
    private static final int[] TOKENS = {2, 5, 1, 4};

    @Test
    public void checkpointMatchesScalarForwardAcrossPositions() throws Exception {
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS), actual, 3e-4f);
        }
    }

    /** The same forward through the OpenCL backend: both caches per pass and the per-head gate. */
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

    /** The exported GGUF file (NeuralFree's {@code iquest-loopcoder} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsIQuest() throws Exception {
        Path dir = builder().write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("iquest.gguf");
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

    /** IQuestLoopCoder is a supported text architecture with its own GGUF name and plain residual adds. */
    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("IQuestLoopCoderForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.IQUEST_LOOPCODER);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(RopeType.NEOX, traits.ropeType());
        assertEquals(ResidualKind.STANDARD, traits.residualKind());
    }

    /** The converter writes the loop metadata and the Llama block tensors plus the per-layer attention gate. */
    @Test
    public void conversionWritesTheIQuestLayout() throws Exception {
        Path dir = builder().write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("iquest-loopcoder", md.get("general.architecture"));
            assertEquals(LAYERS, ((Number) md.get("iquest-loopcoder.block_count")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("iquest-loopcoder.attention.key_length")).intValue());
            assertEquals(HEAD_DIM, ((Number) md.get("iquest-loopcoder.rope.dimension_count")).intValue());
            assertEquals(ROPE_BASE, ((Number) md.get("iquest-loopcoder.rope.freq_base")).floatValue(), 1e-3f);
            assertEquals(1f / (float) Math.sqrt(HEAD_DIM),
                    ((Number) md.get("iquest-loopcoder.attention.scale")).floatValue(), 1e-6f);
            assertEquals(LOOPS, ((Number) md.get("iquest-loopcoder.recurrent_block.num_loops")).intValue());
            assertEquals(0, ((Number) md.get("iquest-loopcoder.recurrent_block.n_input_layers")).intValue());
            assertEquals(LAYERS, ((Number) md.get("iquest-loopcoder.recurrent_block.n_recurrent_layers")).intValue());
            assertEquals(0, ((Number) md.get("iquest-loopcoder.recurrent_block.n_output_layers")).intValue());
            assertEquals(Boolean.FALSE, md.get("iquest-loopcoder.recurrent_block.justnorm"));
            assertEquals(Boolean.TRUE, md.get("iquest-loopcoder.recurrent_block.dual_attention"));
            assertEquals(WINDOW, ((Number) md.get("iquest-loopcoder.recurrent_block.local_window")).intValue());

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            assertTrue(tensors.contains("blk.0.attn_norm.weight"));
            assertTrue(tensors.contains("blk.0.ffn_norm.weight"));
            assertTrue(tensors.contains("blk.0.attn_loop_gate.weight"));
            assertTrue(tensors.contains("blk.0.attn_loop_gate.bias"));
            assertTrue(tensors.contains("output_norm.weight"));
            assertTrue(tensors.contains("output.weight"));
            assertFalse("the gate is the only attention bias", tensors.contains("blk.0.attn_q.bias"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.IQUEST_LOOPCODER, spec.arch());
            assertEquals(LayerSchedule.Kind.REPEAT, spec.schedule().kind());
            assertEquals(SLOTS, spec.schedule().slots());
            assertEquals(LayerSchedule.LoopCombine.NONE, spec.schedule().loopCombine());
            assertNotNull(spec.head().outputNorm());
            assertTrue(spec.layer(0).norm().attnNorm());
            assertEquals(MixerKind.ATTENTION, spec.layer(0).mixerKind());
            assertFalse("the first pass attends its own cache alone", spec.layer(0).attention().dual());
            assertFalse(spec.layer(1).attention().dual());
        }
    }

    /** Every pass of the trunk is a slot with its own cache; later passes add the read-only first-pass cache. */
    @Test
    public void repeatScheduleGivesEveryPassItsOwnCache() throws Exception {
        ModelSpec spec = spec();
        LayerSchedule s = spec.schedule();
        assertArrayEquals(new int[]{0, 1, 0, 1}, weightLayers(s));
        assertArrayEquals(new int[]{0, 0, 1, 1}, loopIndexes(s));
        assertEquals(0, s.recurrentStart());
        assertEquals(4, s.recurrentEnd());
        assertTrue(s.loopStart(0));
        assertTrue(s.loopEnd(1));
        assertTrue(s.loopStart(2));
        assertTrue(s.loopEnd(3));

        MemoryPlan plan = MemoryPlanner.planForContext(spec, memory(), 32);
        assertEquals(SLOTS, plan.layers().size());
        for (int slot = 0; slot < SLOTS; slot++) {
            LayerSpec layer = spec.layer(slot);
            assertEquals(slot, layer.il());
            assertEquals(s.weightLayer(slot), layer.weightLayer());
            assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(slot).kind());
            assertEquals("slot " + slot + " owns its cache", slot, layer.attention().kvOwnerLayer());
        }
        assertNotEquals("the two passes of block 0 keep separate caches", plan.layer(0).kv(), plan.layer(2).kv());

        assertEquals("pass 1 of block 0 reads pass 0's cache", 0, spec.layer(2).attention().dualReadLayer());
        assertEquals(1, spec.layer(3).attention().dualReadLayer());
        assertEquals(WINDOW, spec.layer(2).attention().dualWindow());
        assertEquals(WINDOW, spec.layer(3).attention().dualWindow());
    }

    /** A session file moves every pass's cache and restores the same logits. */
    @Test
    public void sessionRestoresEveryLoopPass() throws Exception {
        Path path = checkpoint();
        Path file = tmp.getRoot().toPath().resolve("iquest.session");
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

    /** Paged memory plans one cache per pass; the dual attention then refuses the paged step by name. */
    @Test
    public void pagedMemoryRefusesTheDualAttentionByName() throws Exception {
        ModelSpec spec = spec();
        MemoryConfig paged = memory().toBuilder().kvPageSize(4).parallel(2).build();
        MemoryPlan plan = MemoryPlanner.planForContext(spec, paged, 32);
        assertEquals(SLOTS, plan.layers().size());
        for (int slot = 0; slot < SLOTS; slot++) {
            assertEquals(LayerMemoryPlan.Kind.KV, plan.layer(slot).kind());
            assertTrue(plan.layer(slot).kv().rows() > 0);
        }
        assertNotEquals(plan.layer(0).kv(), plan.layer(2).kv());

        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(paged).build();
        Path path = checkpoint();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            UnsupportedOperationException e = assertThrows(UnsupportedOperationException.class,
                    () -> model.forwardBackend().prefill(TOKENS, 0));
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

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("hidden_act", "'relu'"), "hidden_act");
        assertRejected(Collections.singletonMap("is_causal", "false"), "is_causal");
        assertRejected(Collections.singletonMap("attention_bias", "true"), "attention_bias");
        assertRejected(Collections.singletonMap("mlp_bias", "true"), "mlp_bias");
        assertRejected(Collections.singletonMap("loop_num", "0"), "loop_num");
        assertRejected(Collections.singletonMap("loop_window_size", "0"), "loop_window_size");
        assertRejected(Collections.singletonMap("rms_norm_eps", "0"), "rms_norm_eps");
        assertRejected(Collections.singletonMap("head_dim", "2"), "head_dim");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'rope_type':'linear','factor':2.0,'rope_theta':10000.0}"), "rope_parameters");
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
        return builder().write(tmp.newFolder("iquest").toPath());
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
        b.tensor("model.embed_tokens.weight", StDType.F32, new long[]{V, D}, SyntheticHf.f32(matrix(V, D, 1)));
        b.tensor("model.norm.weight", StDType.F32, new long[]{D}, SyntheticHf.f32(scaleVector(D, 3)));
        b.tensor("lm_head.weight", StDType.F32, new long[]{V, D}, SyntheticHf.f32(matrix(V, D, 40)));
        for (int il = 0; il < LAYERS; il++) {
            String p = "model.layers." + il + ".";
            int q = HEADS * HEAD_DIM, kv = KV_HEADS * HEAD_DIM;
            b.tensor(p + "input_layernorm.weight", StDType.F32, new long[]{D},
                    SyntheticHf.f32(scaleVector(D, 5 + il)));
            b.tensor(p + "post_attention_layernorm.weight", StDType.F32, new long[]{D},
                    SyntheticHf.f32(scaleVector(D, 7 + il)));
            tensor(b, p + "self_attn.q_proj.weight", q, D, matrix(q, D, 10 + il));
            tensor(b, p + "self_attn.k_proj.weight", kv, D, matrix(kv, D, 12 + il));
            tensor(b, p + "self_attn.v_proj.weight", kv, D, matrix(kv, D, 14 + il));
            tensor(b, p + "self_attn.o_proj.weight", D, q, matrix(D, q, 16 + il));
            tensor(b, p + "mlp.gate_proj.weight", FF, D, matrix(FF, D, 20 + il));
            tensor(b, p + "mlp.up_proj.weight", FF, D, matrix(FF, D, 22 + il));
            tensor(b, p + "mlp.down_proj.weight", D, FF, matrix(D, FF, 24 + il));
            tensor(b, "model.gate_projections." + il + ".gate_proj.weight", HEADS, HEAD_DIM,
                    matrix(HEADS, HEAD_DIM, 30 + il));
            b.tensor("model.gate_projections." + il + ".gate_proj.bias", StDType.F32, new long[]{HEADS},
                    SyntheticHf.f32(gateBias(HEADS, 32 + il)));
        }
        return b;
    }

    /** The IQuestLoopCoder {@code config.json}, with {@code overrides} (raw JSON values) replacing the defaults. */
    private static String config(Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['IQuestLoopCoderForCausalLM']");
        fields.put("model_type", "'iquest_loopcoder'");
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
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("attention_bias", "false");
        fields.put("mlp_bias", "false");
        fields.put("is_causal", "true");
        fields.put("loop_num", Integer.toString(LOOPS));
        fields.put("loop_window_size", Integer.toString(WINDOW));
        fields.put("rope_parameters", "{'rope_theta':" + ROPE_BASE + "}");
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
        b.tensor(name, StDType.F32, new long[]{rows, cols}, SyntheticHf.f32(data));
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.03f;
        }
        return x;
    }

    private static float[] scaleVector(int n, int seed) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = 1.0f + ((i + seed) % 5 - 2) * 0.02f;
        }
        return x;
    }

    private static float[] gateBias(int n, int seed) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = ((i * 3 + seed) % 7 - 3) * 0.02f;
        }
        return x;
    }

    // ---------------------------------------------------------------- scalar reference

    /** A scalar transcription of vLLM's {@code IQuestLoopCoderModel.forward}: the loop passes and both attentions. */
    private static float[] reference(int[] tokens) {
        int n = tokens.length;
        int qSize = HEADS * HEAD_DIM, kvSize = KV_HEADS * HEAD_DIM;
        float[] embeddings = matrix(V, D, 1);
        float[] head = matrix(V, D, 40);
        float[][] hidden = new float[n][];
        for (int t = 0; t < n; t++) {
            hidden[t] = embeddingOf(embeddings, tokens[t]);
        }
        float[][][] globalK = new float[LAYERS][n][kvSize];
        float[][][] globalV = new float[LAYERS][n][kvSize];
        float[][][][] localK = new float[LOOPS][LAYERS][n][kvSize];
        float[][][][] localV = new float[LOOPS][LAYERS][n][kvSize];
        for (int loop = 0; loop < LOOPS; loop++) {
            for (int il = 0; il < LAYERS; il++) {
                float[] normW = scaleVector(D, 5 + il);
                float[][] x = new float[n][];
                for (int t = 0; t < n; t++) {
                    x[t] = rmsNorm(hidden[t], normW);
                }
                float[][] q = new float[n][], k = new float[n][], v = new float[n][];
                for (int t = 0; t < n; t++) {
                    q[t] = linear(x[t], matrix(qSize, D, 10 + il), qSize);
                    k[t] = linear(x[t], matrix(kvSize, D, 12 + il), kvSize);
                    v[t] = linear(x[t], matrix(kvSize, D, 14 + il), kvSize);
                    for (int h = 0; h < HEADS; h++) {
                        rotateHalf(q[t], h * HEAD_DIM, HEAD_DIM, t, ROPE_BASE);
                    }
                    for (int h = 0; h < KV_HEADS; h++) {
                        rotateHalf(k[t], h * HEAD_DIM, HEAD_DIM, t, ROPE_BASE);
                    }
                }
                if (loop == 0) {
                    for (int t = 0; t < n; t++) {
                        System.arraycopy(k[t], 0, globalK[il][t], 0, kvSize);
                        System.arraycopy(v[t], 0, globalV[il][t], 0, kvSize);
                    }
                } else {
                    for (int t = 0; t < n; t++) {
                        System.arraycopy(k[t], 0, localK[loop][il][t], 0, kvSize);
                        System.arraycopy(v[t], 0, localV[loop][il][t], 0, kvSize);
                    }
                }
                float[][] attn = new float[n][qSize];
                for (int t = 0; t < n; t++) {
                    if (loop == 0) {
                        attend(attn[t], q[t], globalK[il], globalV[il], t, n, false);
                        continue;
                    }
                    float[] local = new float[qSize];
                    float[] global = new float[qSize];
                    attend(global, q[t], globalK[il], globalV[il], t, n, false);
                    attend(local, q[t], localK[loop][il], localV[loop][il], t, n, true);
                    float[] gw = matrix(HEADS, HEAD_DIM, 30 + il);
                    float[] gb = gateBias(HEADS, 32 + il);
                    for (int h = 0; h < HEADS; h++) {
                        float sum = gb[h];
                        for (int d = 0; d < HEAD_DIM; d++) {
                            sum += q[t][h * HEAD_DIM + d] * gw[h * HEAD_DIM + d];
                        }
                        float g = sigmoid(sum);
                        for (int d = 0; d < HEAD_DIM; d++) {
                            int i = h * HEAD_DIM + d;
                            attn[t][i] = local[i] * (1f - g) + global[i] * g;
                        }
                    }
                }
                float[] oW = matrix(D, qSize, 16 + il);
                float[][] mlp = new float[n][];
                for (int t = 0; t < n; t++) {
                    float[] o = linear(attn[t], oW, D);
                    float[] sum = new float[D];
                    for (int i = 0; i < D; i++) {
                        sum[i] = hidden[t][i] + o[i];
                    }
                    float[] y = rmsNorm(sum, scaleVector(D, 7 + il));
                    float[] g = linear(y, matrix(FF, D, 20 + il), FF);
                    float[] u = linear(y, matrix(FF, D, 22 + il), FF);
                    for (int i = 0; i < FF; i++) {
                        g[i] = silu(g[i]) * u[i];
                    }
                    float[] down = linear(g, matrix(D, FF, 24 + il), D);
                    mlp[t] = new float[D];
                    for (int i = 0; i < D; i++) {
                        mlp[t][i] = sum[i] + down[i];
                    }
                }
                hidden = mlp;
            }
        }
        float[] last = rmsNorm(hidden[n - 1], scaleVector(D, 3));
        float[] logits = new float[V];
        for (int vIdx = 0; vIdx < V; vIdx++) {
            for (int i = 0; i < D; i++) {
                logits[vIdx] += last[i] * head[vIdx * D + i];
            }
        }
        return logits;
    }

    /** Causal GQA attention of one query row; {@code local} applies the loop window to the key positions. */
    private static void attend(float[] out, float[] q, float[][] keys, float[][] values, int pos, int n,
                               boolean local) {
        float scale = (float) (1.0 / Math.sqrt(HEAD_DIM));
        for (int h = 0; h < HEADS; h++) {
            int kvHead = h / (HEADS / KV_HEADS);
            float[] weights = new float[pos + 1];
            float max = Float.NEGATIVE_INFINITY;
            boolean any = false;
            for (int j = 0; j <= pos; j++) {
                if (local && pos - j >= WINDOW) {
                    continue;
                }
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q[h * HEAD_DIM + d] * keys[j][kvHead * HEAD_DIM + d];
                }
                weights[j] = dot * scale;
                max = Math.max(max, weights[j]);
                any = true;
            }
            if (!any) {
                continue;
            }
            float total = 0;
            for (int j = 0; j <= pos; j++) {
                if (local && pos - j >= WINDOW) {
                    continue;
                }
                weights[j] = (float) Math.exp(weights[j] - max);
                total += weights[j];
            }
            for (int j = 0; j <= pos; j++) {
                if (local && pos - j >= WINDOW) {
                    continue;
                }
                for (int d = 0; d < HEAD_DIM; d++) {
                    out[h * HEAD_DIM + d] += weights[j] / total * values[j][kvHead * HEAD_DIM + d];
                }
            }
        }
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

    private static float[] linear(float[] x, float[] w, int rows) {
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < x.length; j++) {
                y[i] += x[j] * w[i * x.length + j];
            }
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
