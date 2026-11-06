package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.acceleration.GpuTestUtil;
import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GgufWriter;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ArchRegistry;
import com.eonmux.neuralfree.gguf.arch.FfnKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.arch.SupportClass;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.spec.EmbeddingSpec;
import com.eonmux.neuralfree.gguf.spec.LayerSpec;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.MoeSpec;
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
 * LongCat-Flash checkpoint loading, converted layout and causal inference against a scalar implementation of vLLM's
 * forward (longcat_flash.py {@code FlashDecoderLayer}): the dual attention and MLP blocks with the shortcut MoE, the
 * identity (zero-compute) experts, the absorbed MLA with the LoRA scales folded into the layernorms, and the n-gram
 * input embedding of the Lite variant.
 */
public class HfLongcatFlashTest {
    @Rule public final TemporaryFolder tmp = new TemporaryFolder();

    /** Hidden width. */
    private static final int D = 8;
    /** Attention heads. */
    private static final int H = 2;
    /** Non-rotary key width per head. */
    private static final int NOPE = 2;
    /** Rotary key width per head. */
    private static final int ROPE = 2;
    /** Value width per head. */
    private static final int VHEAD = 2;
    private static final int Q_LORA = 2;
    private static final int KV_LORA = 2;
    private static final int FFN = 4;
    private static final int EXPERT_FFN = 3;
    /** Routed experts with weights, and the identity slots appended after them. */
    private static final int N_EXPERT = 2;
    private static final int ZERO_EXPERTS = 1;
    private static final int TOP_K = 2;
    private static final float ROUTED_SCALE = 2.0f;
    private static final int VOCAB = 7;
    private static final float EPS = 1e-5f;
    private static final int[] TOKENS = {2, 3, 4};
    /** One logical layer: two physical blocks. */
    private static final int LAYERS = 1;

    // ------------------------------------------------------------------ tests

    @Test
    public void conversionWritesTheTwoBlockLayout() throws Exception {
        try (ModelSource src = ModelSources.open(checkpoint().toString(), options("layout"), null)) {
            ModelSpec spec = spec(src);
            assertEquals(Arch.LONGCAT_FLASH, spec.arch());
            assertEquals(2, spec.nLayer());
            LayerSpec first = spec.layer(0);
            LayerSpec second = spec.layer(1);
            assertEquals(FfnKind.MOE_SHORTCUT, first.ffnKind());
            assertEquals(FfnKind.SHORTCUT_ADD, second.ffnKind());
            assertNotNull("the first block holds the dense MLP", first.ffn());
            assertNotNull("the second block holds the dense MLP", second.ffn());
            assertNull(second.moe());
            MoeSpec moe = first.moe();
            assertNotNull(moe);
            assertEquals(N_EXPERT, moe.nExpert());
            assertEquals(ZERO_EXPERTS, moe.zeroExperts());
            assertEquals(N_EXPERT + ZERO_EXPERTS, moe.nRouterExpert());
            assertEquals(TOP_K, moe.nUsed());
            assertEquals(ROUTED_SCALE, moe.weightsScale(), 1e-6f);
            assertTrue("vLLM renormalize=False", !moe.normWeights());
            assertTrue(moe.probBias());
            assertTrue("absorbed MLA keeps the latent cache", first.mla().absorbed());
            assertTrue(first.mla().latentCache());
            assertEquals(ROPE, first.mla().headKRope());
            assertEquals(NOPE, first.mla().headKNope());
            assertEquals(VHEAD, first.mla().headV());
            Set<String> names = tensorNames(src);
            assertTrue(names.contains("blk.0.attn_k_b.weight"));
            assertTrue(names.contains("blk.0.attn_v_b.weight"));
            assertFalse("kv_b_proj is split", names.contains("blk.0.attn_kv_b.weight"));
            assertTrue(names.contains("blk.0.ffn_gate_exps.weight"));
            assertFalse("the second block has no router", names.contains("blk.1.ffn_gate_inp.weight"));
            assertTrue(names.contains("blk.1.ffn_up.weight"));
        }
    }

    @Test
    public void matchesVllmForwardOnTheCpu() throws Exception {
        try (GGUFModel model = new GGUFModel(checkpoint().toString(), cpuConfig())) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(false), actual, 2e-4f);
        }
    }

    @Test
    public void matchesVllmForwardOnTheGpu() throws Exception {
        Assume.assumeTrue("OpenCL GPU not available", GpuTestUtil.isOpenCLAvailable());
        EngineConfig cfg = EngineConfig.builder().backend(EngineConfig.BackendMode.OPENCL).addGpuDevice("0")
                .memory(MemoryConfig.builder().nCtx(32).ubatch(4).build()).build();
        try (GGUFModel model = new GGUFModel(checkpoint().toString(), cfg)) {
            float[] actual = model.forwardBackend().prefill(TOKENS, 0).clone();
            assertArrayEquals(reference(false), actual, 2e-4f);
        }
    }

    @Test
    public void exportedGgufLoadsAsLongcatFlash() throws Exception {
        Path out = tmp.getRoot().toPath().resolve("longcat.gguf");
        try (HfModelSource src = HfModelSource.open(checkpoint(), options("export"), null)) {
            GgufWriter.write(src, out, new GgufWriter.Options(GgufWriter.Mode.NATIVE));
        }
        try (GGUFModel model = new GGUFModel(out.toString(), cpuConfig())) {
            assertEquals(Arch.LONGCAT_FLASH, model.spec().arch());
            assertArrayEquals(reference(false), model.forwardBackend().prefill(TOKENS, 0).clone(), 2e-4f);
        }
    }

    @Test
    public void registeredAsASupportedTextArchitecture() {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("LongcatFlashForCausalLM"));
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains("LongcatFlashNgramForCausalLM"));
        assertEquals(SupportClass.GEN, ArchRegistry.traits(Arch.LONGCAT_FLASH).support());
        assertEquals(RopeType.NORM, ArchRegistry.traits(Arch.LONGCAT_FLASH).ropeType());
        assertEquals(SupportClass.GEN, ArchRegistry.traits(Arch.LONGCAT_FLASH_NGRAM).support());
    }

    @Test
    public void refusesConfigurationsTheLayoutCannotCarry() throws Exception {
        assertRejected(Collections.singletonMap("qk_head_dim", "5"), "qk_head_dim");
        assertRejected(Collections.singletonMap("num_hidden_layers", "3"), "num_hidden_layers");
    }

    /** The n-gram variant fuses the hashed tables of the preceding tokens with the token embedding. */
    @Test
    public void ngramInputEmbeddingMatchesVllmOnTheCpu() throws Exception {
        Path dir = ngramCheckpoint();
        try (GGUFModel model = new GGUFModel(dir.toString(), cpuConfig())) {
            EmbeddingSpec embd = model.spec().embedding();
            assertTrue(embd.ngram());
            assertEquals(1, embd.ngramEmbedders());
            assertEquals(D, embd.ngramWidth());
            assertArrayEquals(reference(true), model.forwardBackend().prefill(TOKENS, 0).clone(), 2e-4f);
        }
    }

    /** The MTP head of the checkpoint becomes the NextN block only when drafting is asked for. */
    @Test
    public void mtpHeadConvertsAsANextNBlockWhenDraftingIsEnabled() throws Exception {
        Path dir = builder(Collections.<String, String>emptyMap(), false, true).write(tmp.newFolder("mtp").toPath());
        try (ModelSource src = ModelSources.open(dir.toString(), options("mtp-on"), null)) {
            ModelSpec sp = spec(src);
            assertEquals(3, sp.hparams().nLayerAll());
            assertEquals(1, sp.hparams().nextnLayers());
            Set<String> names = tensorNames(src);
            assertTrue(names.contains("blk.2.nextn.eh_proj.weight"));
            assertTrue(names.contains("blk.2.nextn.enorm.weight"));
            assertTrue(names.contains("blk.2.nextn.hnorm.weight"));
            assertTrue(names.contains("blk.2.attn_q_a.weight"));
        }
        ModelSourceOptions off = ModelSourceOptions.builder().draftMode(ModelSourceOptions.DraftMode.NO_NEXTN)
                .cacheDir(java.nio.file.Files.createTempDirectory("mtp-off")).build();
        try (ModelSource src = ModelSources.open(dir.toString(), off, null)) {
            assertEquals(2, spec(src).hparams().nLayerAll());
            assertFalse(tensorNames(src).contains("blk.2.nextn.eh_proj.weight"));
        }
    }

    // ------------------------------------------------------------------ helpers

    private Path checkpoint() throws Exception {
        return builder(Collections.<String, String>emptyMap(), false).write(tmp.newFolder("longcat").toPath());
    }

    private Path ngramCheckpoint() throws Exception {
        return builder(Collections.<String, String>emptyMap(), true).write(tmp.newFolder("ngram").toPath());
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
        Path dir = builder(overrides, false).write(tmp.newFolder().toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.open(dir.toString(), options("rejected-" + field), null).close());
        assertTrue(e.getMessage(), e.getMessage().contains(field));
    }

    // ------------------------------------------------------------------ checkpoint

    private static SyntheticHf.Builder builder(Map<String, String> overrides, boolean ngram) {
        return builder(overrides, ngram, false);
    }

    private static SyntheticHf.Builder builder(Map<String, String> overrides, boolean ngram, boolean mtp) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(config(overrides, ngram));
        int qDim = H * (NOPE + ROPE);
        int kvB = H * (NOPE + VHEAD);
        tensor(b, "model.embed_tokens.weight", VOCAB, D, matrix(VOCAB, D, 1));
        tensor(b, "model.norm.weight", D, 1, ones(D));
        tensor(b, "lm_head.weight", VOCAB, D, matrix(VOCAB, D, 9));
        for (int half = 0; half < 2; half++) {
            String p = "model.layers.0.";
            tensor(b, p + "input_layernorm." + half + ".weight", D, 1, ones(D));
            tensor(b, p + "post_attention_layernorm." + half + ".weight", D, 1, ones(D));
            tensor(b, p + "self_attn." + half + ".q_a_proj.weight", Q_LORA, D, matrix(Q_LORA, D, 2 + half));
            tensor(b, p + "self_attn." + half + ".q_a_layernorm.weight", Q_LORA, 1, ones(Q_LORA));
            tensor(b, p + "self_attn." + half + ".q_b_proj.weight", qDim, Q_LORA, matrix(qDim, Q_LORA, 3 + half));
            tensor(b, p + "self_attn." + half + ".kv_a_proj_with_mqa.weight", KV_LORA + ROPE, D,
                    matrix(KV_LORA + ROPE, D, 4 + half));
            tensor(b, p + "self_attn." + half + ".kv_a_layernorm.weight", KV_LORA, 1, ones(KV_LORA));
            tensor(b, p + "self_attn." + half + ".kv_b_proj.weight", kvB, KV_LORA, matrix(kvB, KV_LORA, 5 + half));
            tensor(b, p + "self_attn." + half + ".o_proj.weight", D, H * VHEAD, matrix(D, H * VHEAD, 6 + half));
            tensor(b, p + "mlps." + half + ".gate_proj.weight", FFN, D, matrix(FFN, D, 7 + half));
            tensor(b, p + "mlps." + half + ".up_proj.weight", FFN, D, matrix(FFN, D, 8 + half));
            tensor(b, p + "mlps." + half + ".down_proj.weight", D, FFN, matrix(D, FFN, 9 + half));
        }
        tensor(b, "model.layers.0.mlp.router.classifier.weight", N_EXPERT + ZERO_EXPERTS, D,
                matrix(N_EXPERT + ZERO_EXPERTS, D, 10));
        b.tensor("model.layers.0.mlp.router.e_score_correction_bias", StDType.F32,
                new long[]{N_EXPERT + ZERO_EXPERTS}, SyntheticHf.f32(bias()));
        for (int e = 0; e < N_EXPERT; e++) {
            String p = "model.layers.0.mlp.experts." + e + ".";
            tensor(b, p + "gate_proj.weight", EXPERT_FFN, D, matrix(EXPERT_FFN, D, 11 + e));
            tensor(b, p + "up_proj.weight", EXPERT_FFN, D, matrix(EXPERT_FFN, D, 12 + e));
            tensor(b, p + "down_proj.weight", D, EXPERT_FFN, matrix(D, EXPERT_FFN, 13 + e));
        }
        if (ngram) {
            tensor(b, "model.ngram_embeddings.embedders.0.weight", NGRAM_ROWS, D, matrix(NGRAM_ROWS, D, 14));
            tensor(b, "model.ngram_embeddings.post_projs.0.weight", D, D, matrix(D, D, 15));
        }
        if (mtp) {
            mtpTensors(b);
        }
        return b;
    }


    /** The extra decoder layer the DeepSeek-family checkpoints append after the target's blocks. */
    private static void mtpTensors(SyntheticHf.Builder b) {
        String p = "model.layers." + (2 * LAYERS) + ".";
        tensor(b, p + "input_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "post_attention_layernorm.weight", D, 1, ones(D));
        tensor(b, p + "self_attn.q_a_proj.weight", Q_LORA, D, matrix(Q_LORA, D, 2));
        tensor(b, p + "self_attn.q_a_layernorm.weight", Q_LORA, 1, ones(Q_LORA));
        tensor(b, p + "self_attn.q_b_proj.weight", H * (NOPE + ROPE), Q_LORA, matrix(H * (NOPE + ROPE), Q_LORA, 3));
        tensor(b, p + "self_attn.kv_a_proj_with_mqa.weight", KV_LORA + ROPE, D,
                matrix(KV_LORA + ROPE, D, 4));
        tensor(b, p + "self_attn.kv_a_layernorm.weight", KV_LORA, 1, ones(KV_LORA));
        tensor(b, p + "self_attn.kv_b_proj.weight", H * (NOPE + VHEAD), KV_LORA,
                matrix(H * (NOPE + VHEAD), KV_LORA, 5));
        tensor(b, p + "self_attn.o_proj.weight", D, H * VHEAD, matrix(D, H * VHEAD, 6));
        tensor(b, p + "mlp.gate_proj.weight", FFN, D, matrix(FFN, D, 7));
        tensor(b, p + "mlp.up_proj.weight", FFN, D, matrix(FFN, D, 8));
        tensor(b, p + "mlp.down_proj.weight", D, FFN, matrix(D, FFN, 9));
        tensor(b, p + "eh_proj.weight", D, 2 * D, matrix(D, 2 * D, 16));
        tensor(b, p + "enorm.weight", D, 1, ones(D));
        tensor(b, p + "hnorm.weight", D, 1, ones(D));
        tensor(b, p + "norm.weight", D, 1, ones(D));
    }

    /** Base table size {@code m = ratio * vocab}, rounded. */
    private static final int NGRAM_BASE = 4;
    private static final int NGRAM_ROWS = NGRAM_BASE + 1;

    private static String config(Map<String, String> overrides, boolean ngram) {
        Map<String, String> fields = new LinkedHashMap<>();
        fields.put("architectures", ngram ? "['LongcatFlashNgramForCausalLM']" : "['LongcatFlashForCausalLM']");
        fields.put("model_type", "'longcat_flash'");
        fields.put("vocab_size", Integer.toString(VOCAB));
        fields.put("hidden_size", Integer.toString(D));
        fields.put("num_hidden_layers", Integer.toString(2 * LAYERS));
        fields.put("num_layers", Integer.toString(LAYERS));
        fields.put("num_attention_heads", Integer.toString(H));
        fields.put("num_key_value_heads", Integer.toString(H));
        fields.put("q_lora_rank", Integer.toString(Q_LORA));
        fields.put("kv_lora_rank", Integer.toString(KV_LORA));
        fields.put("qk_nope_head_dim", Integer.toString(NOPE));
        fields.put("qk_rope_head_dim", Integer.toString(ROPE));
        fields.put("v_head_dim", Integer.toString(VHEAD));
        fields.put("rms_norm_eps", "1e-5");
        fields.put("max_position_embeddings", "128");
        fields.put("ffn_hidden_size", Integer.toString(FFN));
        fields.put("expert_ffn_hidden_size", Integer.toString(EXPERT_FFN));
        fields.put("n_routed_experts", Integer.toString(N_EXPERT));
        fields.put("zero_expert_num", Integer.toString(ZERO_EXPERTS));
        fields.put("moe_topk", Integer.toString(TOP_K));
        fields.put("routed_scaling_factor", Float.toString(ROUTED_SCALE));
        fields.put("mla_scale_q_lora", "true");
        fields.put("mla_scale_kv_lora", "true");
        fields.put("tie_word_embeddings", "false");
        fields.put("bos_token_id", "0");
        fields.put("eos_token_id", "1");
        if (ngram) {
            fields.put("ngram_vocab_size_ratio", Double.toString((double) NGRAM_BASE / VOCAB));
            fields.put("emb_neighbor_num", "2");
            fields.put("emb_split_num", "1");
        }
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

    private static float[] reference(boolean ngram) {
        float[] tok = matrix(VOCAB, D, 1);
        float[] norm = ones(D);
        float[] lm = matrix(VOCAB, D, 9);
        float[] state = new float[TOKENS.length * D];
        float[][] kCache = new float[TOKENS.length][];
        float[][] vCache = new float[TOKENS.length][];
        for (int t = 0; t < TOKENS.length; t++) {
            float[] e = ngram ? ngramEmbedding(tok, t) : row(tok, TOKENS[t], D);
            System.arraycopy(e, 0, state, t * D, D);
        }
        float[][] moeOuts = new float[TOKENS.length][];
        for (int block = 0; block < 2; block++) {
            for (int t = 0; t < TOKENS.length; t++) {
                float[] x = row(state, t, D);
                float[] n = rmsNorm(x, norm);
                float[] attn = attention(n, block, t, kCache, vCache);
                float[] x1 = add(x, attn);
                float[] n1 = rmsNorm(x1, norm);
                if (block == 0) {
                    moeOuts[t] = moe(n1);
                }
                float[] x2 = add(x1, mlp(n1, block));
                if (block == 1) {
                    x2 = add(x2, moeOuts[t]);
                }
                System.arraycopy(x2, 0, state, t * D, D);
            }
        }
        float[] last = rmsNorm(row(state, TOKENS.length - 1, D), norm);
        return linear(last, lm, VOCAB);
    }

    /** {@code NgramEmbedding.embed_batched}: the mean of the word row and the projected hashed rows. */
    private static float[] ngramEmbedding(float[] tok, int t) {
        float[] word = row(tok, TOKENS[t], D);
        float[] table = matrix(NGRAM_ROWS, D, 14);
        float[] proj = matrix(D, D, 15);
        int id = 0;
        int vocabPowers = 1;
        for (int d = 0; d <= 1; d++) {
            if (t - d < 0) {
                break;
            }
            int token = TOKENS[t - d];
            if (token == 1) {
                break;
            }
            id += token * vocabPowers;
            id %= NGRAM_ROWS;
            vocabPowers = (vocabPowers * VOCAB) % NGRAM_ROWS;
        }
        float[] projected = linear(row(table, id, D), proj, D);
        float[] out = new float[D];
        for (int i = 0; i < D; i++) {
            out[i] = (word[i] + projected[i]) / 2f;
        }
        return out;
    }

    /** One physical block's MLA attention over the cache written so far. */
    private static float[] attention(float[] n, int block, int t, float[][] kCache, float[][] vCache) {
        float[] qa = linear(n, matrix(Q_LORA, D, 2 + block), Q_LORA);
        float scale = (float) Math.sqrt((double) D / Q_LORA);
        qa = rmsNorm(qa, scaled(ones(Q_LORA), scale));
        float[] q = linear(qa, matrix(H * (NOPE + ROPE), Q_LORA, 3 + block), H * (NOPE + ROPE));
        float[] kv = linear(n, matrix(KV_LORA + ROPE, D, 4 + block), KV_LORA + ROPE);
        float[] kvNorm = rmsNorm(slice(kv, 0, KV_LORA), scaled(ones(KV_LORA), (float) Math.sqrt((double) D / KV_LORA)));
        kCache[t] = kvNorm;
        vCache[t] = rotate(slice(kv, KV_LORA, ROPE), t);
        float[] kvB = matrix(H * (NOPE + VHEAD), KV_LORA, 5 + block);
        float[] head = new float[H * VHEAD];
        for (int h = 0; h < H; h++) {
            int qBase = h * (NOPE + ROPE);
            float[] qNope = slice(q, qBase, NOPE);
            float[] qRope = rotate(slice(q, qBase + NOPE, ROPE), t);
            float[] scores = new float[t + 1];
            float max = Float.NEGATIVE_INFINITY;
            float kq = 1.0f / (float) Math.sqrt(NOPE + ROPE);
            for (int j = 0; j <= t; j++) {
                float s = 0f;
                for (int i = 0; i < KV_LORA; i++) {
                    for (int c = 0; c < NOPE; c++) {
                        s += qNope[c] * kvB[(h * (NOPE + VHEAD) + c) * KV_LORA + i] * kCache[j][i];
                    }
                }
                s += dot(qRope, vCache[j]);
                scores[j] = s * kq;
                max = Math.max(max, scores[j]);
            }
            float total = 0;
            for (int j = 0; j <= t; j++) {
                scores[j] = (float) Math.exp(scores[j] - max);
                total += scores[j];
            }
            float[] latent = new float[KV_LORA];
            for (int j = 0; j <= t; j++) {
                for (int i = 0; i < KV_LORA; i++) {
                    latent[i] += scores[j] / total * kCache[j][i];
                }
            }
            int vBase = h * (NOPE + VHEAD) + NOPE;
            for (int r = 0; r < VHEAD; r++) {
                float s = 0;
                for (int i = 0; i < KV_LORA; i++) {
                    s += kvB[(vBase + r) * KV_LORA + i] * latent[i];
                }
                head[h * VHEAD + r] = s;
            }
        }
        return linear(head, matrix(D, H * VHEAD, 6 + block), D);
    }

    private static float[] moe(float[] n) {
        float[] router = matrix(N_EXPERT + ZERO_EXPERTS, D, 10);
        float[] logits = linear(n, router, N_EXPERT + ZERO_EXPERTS);
        float[] probs = softmax(logits);
        float[] sel = new float[probs.length];
        for (int i = 0; i < probs.length; i++) {
            sel[i] = probs[i] + bias()[i];
        }
        int[] pick = new int[TOP_K];
        boolean[] taken = new boolean[sel.length];
        for (int j = 0; j < TOP_K; j++) {
            int best = -1;
            for (int e = 0; e < sel.length; e++) {
                if (!taken[e] && (best < 0 || sel[e] > sel[best])) {
                    best = e;
                }
            }
            pick[j] = best;
            taken[best] = true;
        }
        float[] out = new float[D];
        for (int j = 0; j < TOP_K; j++) {
            float w = probs[pick[j]] * ROUTED_SCALE;
            if (pick[j] >= N_EXPERT) {
                for (int i = 0; i < D; i++) {
                    out[i] += w * n[i];
                }
                continue;
            }
            float[] expert = expert(n, pick[j]);
            for (int i = 0; i < D; i++) {
                out[i] += w * expert[i];
            }
        }
        return out;
    }

    private static float[] expert(float[] n, int e) {
        float[] gate = silu(linear(n, matrix(EXPERT_FFN, D, 11 + e), EXPERT_FFN));
        float[] up = linear(n, matrix(EXPERT_FFN, D, 12 + e), EXPERT_FFN);
        float[] act = new float[EXPERT_FFN];
        for (int i = 0; i < EXPERT_FFN; i++) {
            act[i] = gate[i] * up[i];
        }
        return linear(act, matrix(D, EXPERT_FFN, 13 + e), D);
    }

    private static float[] mlp(float[] n, int half) {
        float[] gate = silu(linear(n, matrix(FFN, D, 7 + half), FFN));
        float[] up = linear(n, matrix(FFN, D, 8 + half), FFN);
        float[] act = new float[FFN];
        for (int i = 0; i < FFN; i++) {
            act[i] = gate[i] * up[i];
        }
        return linear(act, matrix(D, FFN, 9 + half), D);
    }

    // ------------------------------------------------------------------ scalar primitives

    private static float[] rotate(float[] x, int pos) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i += 2) {
            double theta = Math.pow(10000.0, -2.0 * (i / 2) / x.length);
            double angle = pos * theta;
            float c = (float) Math.cos(angle);
            float s = (float) Math.sin(angle);
            y[i] = x[i] * c - x[i + 1] * s;
            y[i + 1] = x[i] * s + x[i + 1] * c;
        }
        return y;
    }

    private static float[] softmax(float[] x) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) {
            max = Math.max(max, v);
        }
        float[] y = new float[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            y[i] = (float) Math.exp(x[i] - max);
            sum += y[i];
        }
        for (int i = 0; i < x.length; i++) {
            y[i] /= (float) sum;
        }
        return y;
    }

    private static float[] rmsNorm(float[] x, float[] weight) {
        double sum = 0;
        for (float v : x) {
            sum += (double) v * v;
        }
        float scale = (float) (1.0 / Math.sqrt(sum / x.length + EPS));
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale * weight[i];
        }
        return y;
    }

    private static float[] silu(float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] / (1f + (float) Math.exp(-x[i]));
        }
        return y;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static float[] add(float[] a, float[] b) {
        float[] y = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = a[i] + b[i];
        }
        return y;
    }

    private static float[] slice(float[] x, int from, int n) {
        float[] y = new float[n];
        System.arraycopy(x, from, y, 0, n);
        return y;
    }

    private static float[] row(float[] m, int r, int cols) {
        return slice(m, r * cols, cols);
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

    private static float[] scaled(float[] x, float scale) {
        float[] y = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] * scale;
        }
        return y;
    }

    private static float[] ones(int n) {
        float[] x = new float[n];
        java.util.Arrays.fill(x, 1f);
        return x;
    }

    private static float[] bias() {
        float[] x = new float[N_EXPERT + ZERO_EXPERTS];
        for (int i = 0; i < x.length; i++) {
            x[i] = (i - 1) * 0.05f;
        }
        return x;
    }

    private static float[] matrix(int rows, int cols, int seed) {
        float[] x = new float[rows * cols];

        for (int i = 0; i < x.length; i++) {
            x[i] = ((i * 7 + seed * 3) % 17 - 8) * 0.035f;
        }
        return x;
    }
}
