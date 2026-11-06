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
import com.eonmux.neuralfree.gguf.arch.GatingFunc;
import com.eonmux.neuralfree.gguf.arch.MemoryKind;
import com.eonmux.neuralfree.gguf.arch.MixerKind;
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
 * Bailing MoE V2.5 checkpoint loading and causal inference against a scalar transcription of vLLM's
 * {@code bailing_moe_linear.py}: a lightning-attention layer with SiLU, per-head Q/K RMS norms, the partial NEOX
 * rotation, the {@code head_dim^-0.5} Q scale and the grouped output norm, then an MLA layer with the absorbed
 * latent and a routed-plus-shared MoE.
 */
public class HfBailingV25Test {

    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    private static final int D = 8, V = 7, FF = 4;
    private static final int HEADS = 2, HD = 4, NOPE = 2, ROPE = 2, VH = 4, KVL = 4, QL = 4;
    private static final int N_EXP = 4, TOPK = 2, MOE_FF = 4, SH_FF = 4;
    private static final int SHARED_GROUPS = 2;
    private static final float EPS = 1e-6f;
    private static final float THETA = 10000f;
    private static final float ROUTED_SCALE = 1.5f;
    private static final int[] TOKENS = {2, 3, 4, 5};

    // layer 0: lightning attention, dense FFN; layer 1: MLA attention, routed plus shared experts
    private static final int L0_IN = 1, L0_POST = 2, L0_QKV = 3, L0_G = 4, L0_DENSE = 5, L0_GNORM = 6;
    private static final int L0_QNORM = 7, L0_KNORM = 8, L0_FF = 9;
    private static final int L1_IN = 12, L1_POST = 13, L1_QKV_A = 14, L1_Q_A_NORM = 15, L1_Q_B = 16;
    private static final int L1_KV_A_NORM = 17, L1_KV_B = 18, L1_O = 19, L1_ROUTER = 20;
    private static final int L1_EXPERTS = 21, L1_SHARED = 33, NORM = 36, HEAD = 37;

    @Test
    public void checkpointMatchesScalarForward() throws Exception {
        Path path = checkpoint(softmax());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, softmax()), actual, 2e-4f);
        }
    }

    /** The same forward through OpenCL: the lightning recurrence, the group norm and the MLA absorption. */
    @Test
    public void checkpointMatchesScalarForwardOnOpenCl() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        Path path = checkpoint(softmax());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0").memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, softmax()), actual, 2e-4f);
        }
    }

    /** The same forward through CUDA. */
    @Test
    public void checkpointMatchesScalarForwardOnCuda() throws Exception {
        GpuTestUtil.assumeCudaAvailable();
        Path path = checkpoint(softmax());
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CUDA).addGpuDevice("0").memory(memory()).build();
        try (GGUFModel model = new GGUFModel(path.toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, softmax()), actual, 2e-4f);
        }
    }

    /** The lightning state and the MLA KV cache carry across steps: chunked prefill plus decode matches one prefill. */
    @Test
    public void incrementalDecodeMatchesPrefill() throws Exception {
        Path path = checkpoint(softmax());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            model.forwardBackend().prefill(new int[]{TOKENS[0], TOKENS[1]}, 0);
            float[] actual = model.forwardBackend().decode(TOKENS[2], 2).clone();
            assertArrayEquals(reference(new int[]{TOKENS[0], TOKENS[1], TOKENS[2]}, softmax()), actual, 2e-4f);
        }
    }

    /** An implicit softmax router with the correction bias selects by the biased score, as vLLM's kernel does. */
    @Test
    public void implicitSoftmaxBiasRoutingMatchesTheScalarForward() throws Exception {
        Path path = checkpoint(implicitSoftmaxBias());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, implicitSoftmaxBias()), actual, 2e-4f);
        }
    }

    /** Sigmoid routing with the correction bias and grouped top-k matches vLLM's grouped_topk. */
    @Test
    public void groupedSigmoidRoutingMatchesTheScalarForward() throws Exception {
        Path path = checkpoint(grouped());
        try (GGUFModel model = new GGUFModel(path.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, grouped()), actual, 2e-4f);
        }
    }

    /** The exported GGUF (NeuralFree's {@code bailingmoe-v25} layout) loads back and runs the same forward. */
    @Test
    public void exportedGgufLoadsAsBailingV25() throws Exception {
        Path dir = builder(softmax(), Collections.emptyMap()).write(tmp.newFolder("export-hf").toPath());
        Path out = tmp.getRoot().toPath().resolve("bailingmoe-v25.gguf");
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("export-cache").toPath()).build();
        try (HfModelSource src = HfModelSource.open(dir, options, null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpu())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(TOKENS, softmax()), actual, 2e-4f);
        }
    }

    /** The converter writes the layout, its metadata and the per-layer mixer kinds. */
    @Test
    public void conversionWritesTheBailingV25Layout() throws Exception {
        Path dir = builder(softmax(), Collections.emptyMap()).write(tmp.newFolder("layout").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("layout-cache").toPath()).build();
        try (ModelSource src = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(src instanceof HfModelSource);
            Map<String, Object> md = src.getMetadata();
            assertEquals("bailingmoe-v25", md.get("general.architecture"));
            assertEquals(2, ((Number) md.get("bailingmoe-v25.block_count")).intValue());
            assertEquals(HD, ((Number) md.get("bailingmoe-v25.attention.key_length")).intValue());
            assertEquals(NOPE + ROPE, ((Number) md.get("bailingmoe-v25.attention.key_length_mla")).intValue());
            assertEquals(VH, ((Number) md.get("bailingmoe-v25.attention.value_length_mla")).intValue());
            assertEquals(KVL, ((Number) md.get("bailingmoe-v25.attention.kv_lora_rank")).intValue());
            assertEquals(QL, ((Number) md.get("bailingmoe-v25.attention.q_lora_rank")).intValue());
            assertEquals(ROPE, ((Number) md.get("bailingmoe-v25.rope.dimension_count")).intValue());
            assertEquals(HD, ((Number) md.get("bailingmoe-v25.ssm.rope_dimension_count")).intValue());
            assertEquals(SHARED_GROUPS, ((Number) md.get("bailingmoe-v25.ssm.group_norm_size")).intValue());
            assertEquals(Boolean.TRUE, md.get("bailingmoe-v25.ssm.qk_norm"));
            assertEquals(Boolean.TRUE, md.get("bailingmoe-v25.ssm.linear_silu"));
            assertEquals(Boolean.TRUE, md.get("bailingmoe-v25.ssm.q_scale"));
            assertEquals(SH_FF, ((Number) md.get("bailingmoe-v25.expert_shared_feed_forward_length")).intValue());
            assertEquals(1, ((Number) md.get("bailingmoe-v25.expert_shared_count")).intValue());
            assertEquals(1, ((Number) md.get("bailingmoe-v25.leading_dense_block_count")).intValue());
            assertEquals(ROUTED_SCALE, ((Number) md.get("bailingmoe-v25.expert_weights_scale")).doubleValue(), 0);
            assertEquals(Boolean.TRUE, md.get("bailingmoe-v25.expert_weights_norm"));
            assertEquals(GatingFunc.SOFTMAX.ggufCode(),
                    ((Number) md.get("bailingmoe-v25.expert_gating_func")).intValue());
            assertEquals("[true, false]", String.valueOf(md.get("bailingmoe-v25.attention.recurrent_layers")));
            assertEquals("[0, 1]", String.valueOf(md.get("bailingmoe-v25.attention.head_count_kv")));

            Set<String> tensors = new HashSet<>();
            for (GGUFTensorInfo t : src.getTensors()) {
                tensors.add(t.name());
            }
            for (String name : new String[]{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.attn_norm_2.weight",
                    "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight", "blk.0.ffn_gate.weight",
                    "blk.1.attn_q_a.weight", "blk.1.attn_kv_a_mqa.weight", "blk.1.attn_k_b.weight",
                    "blk.1.attn_v_b.weight", "blk.1.attn_q_a_norm.weight", "blk.1.attn_kv_a_norm.weight",
                    "blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight",
                    "blk.1.ffn_up_exps.weight", "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_down_shexp.weight",
                    "blk.1.ffn_up_shexp.weight"}) {
                assertTrue(name, tensors.contains(name));
            }
            assertFalse(tensors.contains("blk.1.exp_probs_b.bias"));

            ModelSpec spec = ModelSpecBuilder.build(GgufKv.of(src), TensorCatalog.of(src), MemoryConfig.defaults());
            TensorClaims.requireAllClaimed(spec.hparams(), spec.tensors(), ArchRegistry.get(spec.arch()).tensors());
            assertEquals(Arch.BAILINGMOE_V25, spec.arch());
            LayerSpec linear = spec.layer(0);
            assertEquals(MixerKind.LIGHTNING, linear.mixerKind());
            assertNull(linear.mla());
            assertEquals(HEADS, linear.recurrent().nHead());
            assertEquals(HD, linear.recurrent().headDim());
            assertEquals(SHARED_GROUPS, linear.recurrent().nNormGroups());
            assertTrue(linear.recurrent().qkNorm());
            assertTrue(linear.recurrent().linearSilu());
            assertEquals(0.5f, linear.recurrent().qScale(), 1e-6f);
            assertNotNull(linear.recurrent().rope());
            assertEquals(RopeType.NEOX, linear.recurrent().rope().type());
            assertEquals(HD, linear.recurrent().rope().nDims());
            float slope = (float) (0.0625 * (1.0 + 1e-5));
            assertEquals(slope, linear.recurrent().slopes()[0], 1e-7f);
            assertEquals((float) (0.00390625 * (1.0 + 1e-5)), linear.recurrent().slopes()[1], 1e-8f);

            LayerSpec mla = spec.layer(1);
            assertEquals(MixerKind.MLA, mla.mixerKind());
            assertNotNull(mla.mla());
            assertEquals(QL, mla.mla().qLoraRank());
            assertEquals(KVL, mla.mla().kvLoraRank());
            assertEquals(NOPE, mla.mla().headKNope());
            assertEquals(ROPE, mla.mla().headKRope());
            assertEquals(VH, mla.mla().headV());
            assertTrue(mla.mla().absorbed());
            assertEquals(N_EXP, mla.moe().nExpert());
            assertEquals(TOPK, mla.moe().nUsed());
            assertEquals(GatingFunc.SOFTMAX, mla.moe().gating());
            assertTrue(mla.moe().normWeights());
            assertEquals(ROUTED_SCALE, mla.moe().weightsScale(), 0f);
        }
    }

    /** Bailing MoE V2.5 is registered as a supported generative hybrid under vLLM's class name. */
    @Test
    public void registeredAsASupportedHybrid() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("BailingMoeV2_5ForCausalLM"));
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("BailingMoeV25ForCausalLM"));
        ArchTraits traits = ArchRegistry.traits(Arch.BAILINGMOE_V25);
        assertEquals(SupportClass.GEN, traits.support());
        assertEquals(MemoryKind.HYBRID, traits.memoryKind());
        assertEquals(RopeType.NORM, traits.ropeType());
    }

    @Test
    public void refusesGraphVariantsItCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("use_bias", "true"), "use_bias");
        assertRejected(Collections.singletonMap("use_qkv_bias", "true"), "use_bias");
        assertRejected(Collections.singletonMap("use_mla_nope", "true"), "use_mla_nope");
        assertRejected(Collections.singletonMap("score_function", "'gelu'"), "score_function");
        assertRejected(Collections.singletonMap("score_function", "'sigmoid'"), "moe_router_enable_expert_bias");
        Map<String, String> softmaxBias = new LinkedHashMap<>();
        softmaxBias.put("score_function", "'softmax'");
        softmaxBias.put("moe_router_enable_expert_bias", "true");
        assertRejected(softmaxBias, "moe_router_enable_expert_bias");
        assertRejected(Collections.singletonMap("n_group", "2"), "n_group");
        Map<String, String> softmaxGroups = new LinkedHashMap<>();
        softmaxGroups.put("n_group", "2");
        softmaxGroups.put("topk_group", "1");
        assertRejected(softmaxGroups, "n_group");
        assertRejected(Collections.singletonMap("group_norm_size", "3"), "group_norm_size");
        assertRejected(Collections.singletonMap("partial_rotary_factor", "0.3"), "partial_rotary_factor");
        assertRejected(Collections.singletonMap("rope_parameters",
                "{'rope_type':'longrope','rope_theta':10000.0,'factor':2.0}"), "rope_parameters.rope_type");
    }

    private int rejected;

    private void assertRejected(Map<String, String> overrides, String field) throws Exception {
        String name = "rejected-" + (rejected++);
        Path dir = builder(softmax(), overrides).write(tmp.newFolder(name).toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), ModelSourceOptions.builder()
                        .cacheDir(tmp.newFolder(name + "-cache").toPath()).build(), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    private Path checkpoint(Variant variant) throws Exception {
        return builder(variant, Collections.emptyMap()).write(tmp.newFolder("bailingmoe-v25").toPath());
    }

    private static EngineConfig cpu() {
        return EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).memory(memory()).build();
    }

    private static MemoryConfig memory() {
        return MemoryConfig.builder().nCtx(16).ubatch(8).build();
    }

    /** The routing variant of one checkpoint. */
    private static final class Variant {
        final boolean sigmoid;
        final boolean bias;
        final int nGroup;
        final int topkGroup;
        /** False leaves {@code score_function} out, which vLLM reads as a softmax router. */
        final boolean explicitScore;

        private Variant(boolean sigmoid, boolean bias, int nGroup, int topkGroup, boolean explicitScore) {
            this.sigmoid = sigmoid;
            this.bias = bias;
            this.nGroup = nGroup;
            this.topkGroup = topkGroup;
            this.explicitScore = explicitScore;
        }
    }

    private static Variant softmax() {
        return new Variant(false, false, 0, 0, true);
    }

    /** Softmax with the correction bias and no explicit {@code score_function}, vLLM's implicit default. */
    private static Variant implicitSoftmaxBias() {
        return new Variant(false, true, 0, 0, false);
    }

    private static Variant grouped() {
        return new Variant(true, true, 2, 1, true);
    }

    private SyntheticHf.Builder builder(Variant variant, Map<String, String> overrides) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(variant, overrides));
        tensor(b, "model.embed_tokens.weight", V, D, embedding());
        // layer 0: lightning attention and dense FFN
        tensor(b, "model.layers.0.input_layernorm.weight", 1, D, normWeight(L0_IN));
        tensor(b, "model.layers.0.post_attention_layernorm.weight", 1, D, normWeight(L0_POST));
        tensor(b, "model.layers.0.self_attn.query_key_value.weight", 3 * HD * HEADS, D,
                matrix(3 * HD * HEADS, D, L0_QKV));
        tensor(b, "model.layers.0.self_attn.g_proj.weight", HD * HEADS, D, matrix(HD * HEADS, D, L0_G));
        tensor(b, "model.layers.0.self_attn.dense.weight", D, HD * HEADS, matrix(D, HD * HEADS, L0_DENSE));
        tensor(b, "model.layers.0.self_attn.g_norm.weight", 1, HD * HEADS, normWeightAt(L0_GNORM, HD * HEADS));
        tensor(b, "model.layers.0.self_attn.query_layernorm.weight", 1, HD, normWeightAt(L0_QNORM, HD));
        tensor(b, "model.layers.0.self_attn.key_layernorm.weight", 1, HD, normWeightAt(L0_KNORM, HD));
        denseFfn(b, 0, L0_FF);
        // layer 1: MLA attention and MoE
        tensor(b, "model.layers.1.input_layernorm.weight", 1, D, normWeight(L1_IN));
        tensor(b, "model.layers.1.post_attention_layernorm.weight", 1, D, normWeight(L1_POST));
        tensor(b, "model.layers.1.attention.fused_qkv_a_proj.weight", QL + KVL + ROPE, D,
                matrix(QL + KVL + ROPE, D, L1_QKV_A));
        tensor(b, "model.layers.1.attention.q_a_layernorm.weight", 1, QL, normWeightAt(L1_Q_A_NORM, QL));
        tensor(b, "model.layers.1.attention.q_b_proj.weight", HEADS * (NOPE + ROPE), QL,
                matrix(HEADS * (NOPE + ROPE), QL, L1_Q_B));
        tensor(b, "model.layers.1.attention.kv_a_layernorm.weight", 1, KVL, normWeightAt(L1_KV_A_NORM, KVL));
        tensor(b, "model.layers.1.attention.kv_b_proj.weight", HEADS * (NOPE + VH), KVL,
                matrix(HEADS * (NOPE + VH), KVL, L1_KV_B));
        tensor(b, "model.layers.1.attention.o_proj.weight", D, HEADS * VH, matrix(D, HEADS * VH, L1_O));
        tensor(b, "model.layers.1.mlp.gate.weight", N_EXP, D, matrix(N_EXP, D, L1_ROUTER));
        if (variant.bias) {
            tensor(b, "model.layers.1.mlp.gate.expert_bias", 1, N_EXP, vector(N_EXP, L1_ROUTER + 1));
        }
        for (int e = 0; e < N_EXP; e++) {
            int seed = L1_EXPERTS + 3 * e;
            tensor(b, "model.layers.1.mlp.experts." + e + ".gate_proj.weight", MOE_FF, D,
                    matrix(MOE_FF, D, seed));
            tensor(b, "model.layers.1.mlp.experts." + e + ".up_proj.weight", MOE_FF, D,
                    matrix(MOE_FF, D, seed + 1));
            tensor(b, "model.layers.1.mlp.experts." + e + ".down_proj.weight", D, MOE_FF,
                    matrix(D, MOE_FF, seed + 2));
        }
        tensor(b, "model.layers.1.mlp.shared_experts.gate_proj.weight", SH_FF, D, matrix(SH_FF, D, L1_SHARED));
        tensor(b, "model.layers.1.mlp.shared_experts.up_proj.weight", SH_FF, D, matrix(SH_FF, D, L1_SHARED + 1));
        tensor(b, "model.layers.1.mlp.shared_experts.down_proj.weight", D, SH_FF, matrix(D, SH_FF, L1_SHARED + 2));
        tensor(b, "model.norm.weight", 1, D, normWeight(NORM));
        tensor(b, "lm_head.weight", V, D, matrix(V, D, HEAD));
        return b;
    }

    private static void denseFfn(SyntheticHf.Builder b, int layer, int seed) {
        tensor(b, "model.layers." + layer + ".mlp.gate_proj.weight", FF, D, matrix(FF, D, seed));
        tensor(b, "model.layers." + layer + ".mlp.up_proj.weight", FF, D, matrix(FF, D, seed + 1));
        tensor(b, "model.layers." + layer + ".mlp.down_proj.weight", D, FF, matrix(D, FF, seed + 2));
    }

    private static String config(Variant variant, Map<String, String> overrides) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", "['BailingMoeV2_5ForCausalLM']");
        fields.put("model_type", "'bailing_moe'");
        fields.put("hidden_size", Integer.toString(D));
        fields.put("num_hidden_layers", "2");
        fields.put("num_attention_heads", Integer.toString(HEADS));
        fields.put("head_dim", Integer.toString(HD));
        fields.put("intermediate_size", Integer.toString(FF));
        fields.put("hidden_act", "'silu'");
        fields.put("rms_norm_eps", Float.toString(EPS));
        fields.put("max_position_embeddings", "16");
        fields.put("vocab_size", Integer.toString(V));
        fields.put("tie_word_embeddings", "false");
        fields.put("layer_group_size", "2");
        fields.put("first_k_dense_replace", "1");
        fields.put("num_experts", Integer.toString(N_EXP));
        fields.put("num_experts_per_tok", Integer.toString(TOPK));
        fields.put("num_shared_experts", "1");
        fields.put("moe_intermediate_size", Integer.toString(MOE_FF));
        fields.put("moe_shared_expert_intermediate_size", Integer.toString(SH_FF));
        fields.put("routed_scaling_factor", Float.toString(ROUTED_SCALE));
        fields.put("norm_topk_prob", "true");
        if (variant.explicitScore) {
            fields.put("score_function", variant.sigmoid ? "'sigmoid'" : "'softmax'");
        }
        fields.put("moe_router_enable_expert_bias", Boolean.toString(variant.bias));
        if (variant.nGroup > 0) {
            fields.put("n_group", Integer.toString(variant.nGroup));
            fields.put("topk_group", Integer.toString(variant.topkGroup));
        }
        fields.put("qk_nope_head_dim", Integer.toString(NOPE));
        fields.put("qk_rope_head_dim", Integer.toString(ROPE));
        fields.put("v_head_dim", Integer.toString(VH));
        fields.put("kv_lora_rank", Integer.toString(KVL));
        fields.put("q_lora_rank", Integer.toString(QL));
        fields.put("use_qk_norm", "true");
        fields.put("use_linear_silu", "true");
        fields.put("linear_rope", "true");
        fields.put("group_norm_size", Integer.toString(SHARED_GROUPS));
        fields.put("partial_rotary_factor", "1.0");
        fields.put("rope_theta", Float.toString(THETA));
        fields.put("rope_parameters", "{'rope_type':'default','rope_theta':" + THETA + "}");
        fields.put("use_bias", "false");
        fields.put("use_qkv_bias", "false");
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

    private static float[] embedding() {
        float[] x = new float[V * D];
        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 3) % 11 - 5) * 0.05f;
        }
        return x;
    }

    private static float[] normWeight(int seed) {
        return normWeightAt(seed, D);
    }

    private static float[] normWeightAt(int seed, int width) {
        float[] x = new float[width];
        for (int i = 0; i < width; i++) {
            x[i] = 0.7f + 0.05f * i + 0.01f * seed;
        }
        return x;
    }

    private static float[] vector(int width, int seed) {
        float[] x = new float[width];
        for (int i = 0; i < width; i++) {
            x[i] = ((i * 3 + seed) % 7 - 3) * 0.1f;
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

    // ------------------------------------------------------------------ scalar reference

    /** vLLM's forward over {@code tokens}, returning the last token's logits. */
    private static float[] reference(int[] tokens, Variant variant) {
        float[] emb = embedding();
        float[][] x = new float[tokens.length][D];
        for (int t = 0; t < tokens.length; t++) {
            System.arraycopy(emb, tokens[t] * D, x[t], 0, D);
        }
        lightningLayer(x, tokens);
        mlaLayer(x, tokens, variant);
        float[] logits = new float[V];
        for (int t = 0; t < tokens.length; t++) {
            float[] h = rmsNorm(x[t], normWeight(NORM), EPS);
            float[] y = linear(h, matrix(V, D, HEAD), V);
            if (t == tokens.length - 1) {
                logits = y;
            }
        }
        return logits;
    }

    /** Layer 0: SiLU on the fused Q/K/V, per-head norms, the NEOX rotation, the Q scale and the gated group norm. */
    private static void lightningLayer(float[][] x, int[] tokens) {
        int n = x.length;
        int width = HD * HEADS;
        float[] w = matrix(3 * width, D, L0_QKV);
        float[] qNorm = normWeightAt(L0_QNORM, HD);
        float[] kNorm = normWeightAt(L0_KNORM, HD);
        float[] gNorm = normWeightAt(L0_GNORM, width);
        float[] wg = matrix(width, D, L0_G);
        float[] wo = matrix(D, width, L0_DENSE);
        float[][][] state = new float[HEADS][HD][HD];
        float[] slopes = {0.0625f * (1f + 1e-5f), (float) (0.00390625 * (1.0 + 1e-5))};
        float[][] out = new float[n][width];
        float[][] gate = new float[n][width];
        for (int t = 0; t < n; t++) {
            float[] in = rmsNorm(x[t], normWeight(L0_IN), EPS);
            float[] qkv = linear(in, w, 3 * width);
            for (int i = 0; i < qkv.length; i++) {
                qkv[i] = silu(qkv[i]);
            }
            for (int h = 0; h < HEADS; h++) {
                int qOff = h * HD;
                int kOff = width + h * HD;
                int vOff = 2 * width + h * HD;
                float[] q = rmsNorm(java.util.Arrays.copyOfRange(qkv, qOff, qOff + HD), qNorm, EPS);
                float[] k = rmsNorm(java.util.Arrays.copyOfRange(qkv, kOff, kOff + HD), kNorm, EPS);
                ropeNeox(q, 0, HD, tokens[t]);
                ropeNeox(k, 0, HD, tokens[t]);
                float scale = (float) (1.0 / Math.sqrt(HD));
                for (int col = 0; col < HD; col++) {
                    float o = 0.0f;
                    for (int i = 0; i < HD; i++) {
                        float s = state[h][col][i] * (float) Math.exp(-slopes[h]) + k[i] * qkv[vOff + col];
                        state[h][col][i] = s;
                        o += s * q[i];
                    }
                    out[t][h * HD + col] = o * scale;
                }
            }
            gate[t] = linear(in, wg, width);
        }
        for (int t = 0; t < n; t++) {
            int groupWidth = width / SHARED_GROUPS;
            float[] normed = new float[width];
            for (int g = 0; g < SHARED_GROUPS; g++) {
                float[] part = java.util.Arrays.copyOfRange(out[t], g * groupWidth, (g + 1) * groupWidth);
                float[] wPart = java.util.Arrays.copyOfRange(gNorm, g * groupWidth, (g + 1) * groupWidth);
                float[] nh = rmsNorm(part, wPart, EPS);
                System.arraycopy(nh, 0, normed, g * groupWidth, groupWidth);
            }
            for (int i = 0; i < width; i++) {
                normed[i] *= sigmoid(gate[t][i]);
            }
            float[] attn = linear(normed, wo, D);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += attn[kk];
            }
            float[] n2 = rmsNorm(x[t], normWeight(L0_POST), EPS);
            float[] mlp = mlp(n2, L0_FF, D, FF);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += mlp[kk];
            }
        }
    }

    /** Layer 1: MLA attention (absorbed latent, interleaved rope), then routed plus shared experts. */
    private static void mlaLayer(float[][] x, int[] tokens, Variant variant) {
        int n = x.length;
        float[] wQkvA = matrix(QL + KVL + ROPE, D, L1_QKV_A);
        float[] wQB = matrix(HEADS * (NOPE + ROPE), QL, L1_Q_B);
        float[] wKvB = matrix(HEADS * (NOPE + VH), KVL, L1_KV_B);
        float[] wO = matrix(D, HEADS * VH, L1_O);
        float[][] q = new float[n][HEADS * (NOPE + ROPE)];
        float[][] k = new float[n][HEADS * (NOPE + ROPE)];
        float[][] v = new float[n][HEADS * VH];
        for (int t = 0; t < n; t++) {
            float[] in = rmsNorm(x[t], normWeight(L1_IN), EPS);
            float[] qa = linear(in, wQkvA, QL + KVL + ROPE);
            float[] qaNormed = rmsNorm(java.util.Arrays.copyOfRange(qa, 0, QL), normWeightAt(L1_Q_A_NORM, QL), EPS);
            float[] qFull = linear(qaNormed, wQB, HEADS * (NOPE + ROPE));
            float[] kvC = rmsNorm(java.util.Arrays.copyOfRange(qa, QL, QL + KVL),
                    normWeightAt(L1_KV_A_NORM, KVL), EPS);
            float[] kPe = java.util.Arrays.copyOfRange(qa, QL + KVL, QL + KVL + ROPE);
            float[] kvB = linear(kvC, wKvB, HEADS * (NOPE + VH));
            ropeInterleaved(kPe, 0, ROPE, tokens[t]);
            for (int h = 0; h < HEADS; h++) {
                int qOff = h * (NOPE + ROPE);
                float scale = (float) (1.0 / Math.sqrt(NOPE + ROPE));
                for (int i = 0; i < NOPE + ROPE; i++) {
                    q[t][qOff + i] = qFull[qOff + i] * scale;
                }
                ropeInterleaved(q[t], qOff + NOPE, ROPE, tokens[t]);
                System.arraycopy(kvB, h * (NOPE + VH), k[t], qOff, NOPE);
                System.arraycopy(kPe, 0, k[t], qOff + NOPE, ROPE);
                System.arraycopy(kvB, h * (NOPE + VH) + NOPE, v[t], h * VH, VH);
            }
        }
        float[][] out = new float[n][HEADS * VH];
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < HEADS; h++) {
                double max = Double.NEGATIVE_INFINITY;
                float[] scores = new float[t + 1];
                for (int j = 0; j <= t; j++) {
                    float dot = 0;
                    int qOff = h * (NOPE + ROPE);
                    for (int i = 0; i < NOPE + ROPE; i++) {
                        dot += q[t][qOff + i] * k[j][qOff + i];
                    }
                    scores[j] = dot;
                    max = Math.max(max, dot);
                }
                double sum = 0;
                for (int j = 0; j <= t; j++) {
                    scores[j] = (float) Math.exp(scores[j] - max);
                    sum += scores[j];
                }
                for (int j = 0; j <= t; j++) {
                    float p = (float) (scores[j] / sum);
                    for (int d = 0; d < VH; d++) {
                        out[t][h * VH + d] += p * v[j][h * VH + d];
                    }
                }
            }
        }
        for (int t = 0; t < n; t++) {
            float[] attn = linear(out[t], wO, D);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += attn[kk];
            }
            float[] n2 = rmsNorm(x[t], normWeight(L1_POST), EPS);
            float[] moe = moe(n2, variant);
            for (int kk = 0; kk < D; kk++) {
                x[t][kk] += moe[kk];
            }
        }
    }

    /** The router, the routed experts, their scaled weights and the shared expert. */
    private static float[] moe(float[] in, Variant variant) {
        float[] logits = linear(in, matrix(N_EXP, D, L1_ROUTER), N_EXP);
        float[] probs = new float[N_EXP];
        for (int e = 0; e < N_EXP; e++) {
            probs[e] = variant.sigmoid ? sigmoid(logits[e]) : (float) Math.exp(logits[e]);
        }
        if (!variant.sigmoid) {
            float sum = 0;
            for (float p : probs) {
                sum += p;
            }
            for (int e = 0; e < N_EXP; e++) {
                probs[e] /= sum;
            }
        }
        float[] sel = probs.clone();
        if (variant.bias) {
            float[] bias = vector(N_EXP, L1_ROUTER + 1);
            for (int e = 0; e < N_EXP; e++) {
                sel[e] += bias[e];
            }
        }
        boolean[] allowed = new boolean[N_EXP];
        java.util.Arrays.fill(allowed, true);
        if (variant.nGroup > 0) {
            int per = N_EXP / variant.nGroup;
            float[] groupScores = new float[variant.nGroup];
            for (int g = 0; g < variant.nGroup; g++) {
                float best = Float.NEGATIVE_INFINITY;
                float second = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < per; i++) {
                    float s = sel[g * per + i];
                    if (s > best) {
                        second = best;
                        best = s;
                    } else if (s > second) {
                        second = s;
                    }
                }
                groupScores[g] = best + second;
            }
            boolean[] keep = new boolean[variant.nGroup];
            for (int r = 0; r < variant.topkGroup; r++) {
                int bi = -1;
                float bv = Float.NEGATIVE_INFINITY;
                for (int g = 0; g < variant.nGroup; g++) {
                    if (!keep[g] && groupScores[g] > bv) {
                        bv = groupScores[g];
                        bi = g;
                    }
                }
                keep[bi] = true;
            }
            for (int e = 0; e < N_EXP; e++) {
                allowed[e] = keep[e / per];
            }
        }
        int[] pick = new int[TOPK];
        boolean[] taken = new boolean[N_EXP];
        for (int j = 0; j < TOPK; j++) {
            int bi = -1;
            float bv = Float.NEGATIVE_INFINITY;
            for (int e = 0; e < N_EXP; e++) {
                if (allowed[e] && !taken[e] && sel[e] > bv) {
                    bv = sel[e];
                    bi = e;
                }
            }
            taken[bi] = true;
            pick[j] = bi;
        }
        float weightSum = 0;
        for (int e : pick) {
            weightSum += probs[e];
        }
        float[] y = new float[D];
        for (int e : pick) {
            float wgt = probs[e] / weightSum * ROUTED_SCALE;
            float[] expert = mlp(in, L1_EXPERTS + 3 * e, D, MOE_FF);
            for (int kk = 0; kk < D; kk++) {
                y[kk] += wgt * expert[kk];
            }
        }
        float[] shared = mlp(in, L1_SHARED, D, SH_FF);
        for (int kk = 0; kk < D; kk++) {
            y[kk] += shared[kk];
        }
        return y;
    }

    private static void ropeNeox(float[] x, int off, int dim, int position) {
        int half = dim / 2;
        float[] in = new float[dim];
        System.arraycopy(x, off, in, 0, dim);
        for (int i = 0; i < half; i++) {
            double angle = position * Math.pow(THETA, -2.0 * i / dim);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[off + i] = in[i] * cos - in[i + half] * sin;
            x[off + i + half] = in[i + half] * cos + in[i] * sin;
        }
    }

    private static void ropeInterleaved(float[] x, int off, int rot, int pos) {
        int half = rot / 2;
        float[] in = new float[rot];
        System.arraycopy(x, off, in, 0, rot);
        for (int i = 0; i < half; i++) {
            double angle = pos * Math.pow(THETA, -2.0 * i / rot);
            float cos = (float) Math.cos(angle);
            float sin = (float) Math.sin(angle);
            x[off + 2 * i] = in[2 * i] * cos - in[2 * i + 1] * sin;
            x[off + 2 * i + 1] = in[2 * i + 1] * cos + in[2 * i] * sin;
        }
    }

    private static float[] mlp(float[] in, int seed, int width, int ff) {
        float[] gate = linear(in, matrix(ff, width, seed), ff);
        float[] up = linear(in, matrix(ff, width, seed + 1), ff);
        float[] act = new float[ff];
        for (int i = 0; i < ff; i++) {
            act[i] = silu(gate[i]) * up[i];
        }
        return linear(act, matrix(width, ff, seed + 2), width);
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

    private static float[] rmsNorm(float[] x, float[] w, float eps) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float inv = (float) (1.0 / Math.sqrt(sum / x.length + eps));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * inv * w[i];
        }
        return y;
    }

    private static float silu(float x) {
        return x / (1f + (float) Math.exp(-x));
    }

    private static float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }
}
