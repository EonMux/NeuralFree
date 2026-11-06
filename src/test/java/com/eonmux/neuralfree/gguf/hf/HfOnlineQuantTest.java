package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.engine.EngineModel;
import com.eonmux.neuralfree.gguf.engine.LLMModelFactory;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * vLLM online (load-time) quantization of an unquantized Hugging Face checkpoint end to end: the methods
 * {@code fp8_per_tensor}, {@code fp8_per_block}, {@code fp8_per_channel}, {@code mxfp8}, {@code mxfp4},
 * {@code int8_per_channel_weight_only} / {@code experts_int8} and {@code nvfp4_per_token} plan their weight types on
 * exactly the modules vLLM quantizes (the linear layers and routed experts, never the embedding or the LM head; the
 * MoE-only methods on the routed experts alone), a quantization_config naming a shorthand quantizes by default while
 * a serialized {@code fp8} configuration does not, the bare {@code online} method quantizes the layers its
 * quantization config selects, {@code mxfp4} on an MXFP4 checkpoint names the checkpoint's own quantization, and the
 * quantized models run on the CPU backend with logits close to the BF16 model's. Bit parity with vLLM's codes is
 * {@link HfOnlineQuantParityTest}'s.
 */
public class HfOnlineQuantTest {

    private static final int DIM = 128;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int KV_DIM = DIM / HEADS * KV_HEADS;
    private static final int FFN = 256;
    private static final int EXPERTS = 2;
    private static final int EXPERT_FFN = 128;
    private static final int VOCAB = 16;
    private static final int[] PROMPT = {2, 5, 3, 6, 4};

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options;

    @Before
    public void setUp() throws IOException {
        options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    // ------------------------------------------------------------------ fixtures

    /** Gaussian BF16 values of standard deviation {@code std}, seeded per tensor. */
    private static byte[] gaussian(long seed, int n, double std) {
        Random r = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (r.nextGaussian() * std);
        }
        return SyntheticHf.bf16(v);
    }

    private static byte[] ones(int n) {
        float[] v = new float[n];
        java.util.Arrays.fill(v, 1f);
        return SyntheticHf.bf16(v);
    }

    private static void matrix(SyntheticHf.Builder b, String name, int rows, int cols) {
        b.tensor(name, StDType.BF16, new long[]{rows, cols},
                gaussian(name.hashCode(), rows * cols, 1.0 / Math.sqrt(cols)));
    }

    private static void norm(SyntheticHf.Builder b, String name) {
        b.tensor(name, StDType.BF16, new long[]{DIM}, ones(DIM));
    }

    private static String base(String arch, String extra) {
        return SyntheticHf.json("{'architectures':['" + arch + "'],'model_type':'"
                + (arch.startsWith("Mixtral") ? "mixtral" : "llama") + "','hidden_size':" + DIM
                + ",'num_hidden_layers':1,'num_attention_heads':" + HEADS + ",'num_key_value_heads':" + KV_HEADS
                + ",'max_position_embeddings':128,'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':" + VOCAB
                + ",'torch_dtype':'bfloat16','bos_token_id':0,'eos_token_id':1,'tie_word_embeddings':false"
                + (extra.isEmpty() ? "" : "," + extra) + "}");
    }

    private static void attention(SyntheticHf.Builder b, String p) {
        norm(b, p + "input_layernorm.weight");
        matrix(b, p + "self_attn.q_proj.weight", DIM, DIM);
        matrix(b, p + "self_attn.k_proj.weight", KV_DIM, DIM);
        matrix(b, p + "self_attn.v_proj.weight", KV_DIM, DIM);
        matrix(b, p + "self_attn.o_proj.weight", DIM, DIM);
        norm(b, p + "post_attention_layernorm.weight");
    }

    /** A one-layer dense Llama with Gaussian weights; {@code quantConfig} is a quantization_config member or empty. */
    private Path llama(String quantConfig) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(base("LlamaForCausalLM",
                "'intermediate_size':" + FFN + (quantConfig.isEmpty() ? "" : ",'quantization_config':" + quantConfig)));
        matrix(b, "model.embed_tokens.weight", VOCAB, DIM);
        attention(b, "model.layers.0.");
        matrix(b, "model.layers.0.mlp.gate_proj.weight", FFN, DIM);
        matrix(b, "model.layers.0.mlp.up_proj.weight", FFN, DIM);
        matrix(b, "model.layers.0.mlp.down_proj.weight", DIM, FFN);
        norm(b, "model.norm.weight");
        matrix(b, "lm_head.weight", VOCAB, DIM);
        return b.write(tmp.newFolder().toPath());
    }

    /** A one-layer Mixtral of {@value #EXPERTS} experts, both routed per token, with Gaussian weights. */
    private Path mixtral() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(base("MixtralForCausalLM",
                "'intermediate_size':" + EXPERT_FFN + ",'num_local_experts':" + EXPERTS
                        + ",'num_experts_per_tok':" + EXPERTS));
        matrix(b, "model.embed_tokens.weight", VOCAB, DIM);
        String p = "model.layers.0.";
        attention(b, p);
        matrix(b, p + "block_sparse_moe.gate.weight", EXPERTS, DIM);
        for (int e = 0; e < EXPERTS; e++) {
            String x = p + "block_sparse_moe.experts." + e + ".";
            matrix(b, x + "w1.weight", EXPERT_FFN, DIM);
            matrix(b, x + "w2.weight", DIM, EXPERT_FFN);
            matrix(b, x + "w3.weight", EXPERT_FFN, DIM);
        }
        norm(b, "model.norm.weight");
        matrix(b, "lm_head.weight", VOCAB, DIM);
        return b.write(tmp.newFolder().toPath());
    }

    private ModelSourceOptions online(OnlineQuant q) {
        return options.toBuilder().onlineQuant(q).build();
    }

    private static TensorPlan plan(Path dir, ModelSourceOptions o) throws IOException {
        try (HfModelSource s = HfModelSource.open(dir, o, null)) {
            return s.plan();
        }
    }

    // ------------------------------------------------------------------ planned types

    @Test
    public void linearAndMoeMethodsQuantizeTheLinearsButNeverTheEmbeddingOrLmHead() throws IOException {
        Path dir = llama("");
        OnlineQuant[] methods = {OnlineQuant.FP8_PER_TENSOR, OnlineQuant.FP8_PER_BLOCK, OnlineQuant.FP8_PER_CHANNEL,
            OnlineQuant.MXFP8};
        GGMLType[] types = {GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B128, GGMLType.F8_E4M3_B128, GGMLType.MXFP8};
        for (int i = 0; i < methods.length; i++) {
            try (HfModelSource s = HfModelSource.open(dir, online(methods[i]), null)) {
                TensorPlan p = s.plan();
                for (String w : new String[]{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up",
                    "ffn_down"}) {
                    assertEquals(methods[i] + " " + w, types[i], p.type("blk.0." + w + ".weight"));
                }
                assertEquals(methods[i] + " keeps the embedding", GGMLType.BF16, p.type("token_embd.weight"));
                assertEquals(methods[i] + " keeps the LM head", GGMLType.BF16, p.type("output.weight"));
                assertEquals(GGMLType.F32, p.type("blk.0.attn_norm.weight"));
                assertEquals("the scales live in the blocks", null, s.plan().tensor("blk.0.attn_q.weight")
                        .source().outputScale());
                assertEquals("file_type follows the quantized plan", HfFileType.of(types[i]),
                        ((Number) s.getMetadata().get("general.file_type")).intValue());
            }
        }
    }

    @Test
    public void moeOnlyMethodsQuantizeTheRoutedExpertsAlone() throws IOException {
        Path dir = mixtral();
        for (OnlineQuant q : new OnlineQuant[]{OnlineQuant.EXPERTS_INT8, OnlineQuant.INT8_PER_CHANNEL_WEIGHT_ONLY,
                OnlineQuant.NVFP4_PER_TOKEN}) {
            GGMLType type = q == OnlineQuant.NVFP4_PER_TOKEN ? GGMLType.NVFP4 : GGMLType.I8_B128;
            TensorPlan p = plan(dir, online(q));
            for (String w : new String[]{"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
                assertEquals(q + " " + w, type, p.type("blk.0." + w + ".weight"));
            }
            assertEquals(q + " leaves the linears", GGMLType.BF16, p.type("blk.0.attn_q.weight"));
            assertEquals(q + " leaves the router", GGMLType.F32, p.type("blk.0.ffn_gate_inp.weight"));
            assertEquals(GGMLType.BF16, p.type("output.weight"));
        }
    }

    @Test
    public void nvfp4PerTokenCarriesOneGlobalScalePerExpert() throws IOException {
        TensorPlan p = plan(mixtral(), online(OnlineQuant.NVFP4_PER_TOKEN));
        RowSource scale = p.tensor("blk.0.ffn_up_exps.scale").source();
        assertArrayEquals(new long[]{EXPERTS}, scale.shape());
        ByteBuffer b = ByteBuffer.allocate((int) scale.rowBytes()).order(ByteOrder.LITTLE_ENDIAN);
        scale.readRows(0, 1, b);
        b.flip();
        float[] ws = new float[EXPERTS];
        for (int e = 0; e < EXPERTS; e++) {
            ws[e] = b.getFloat();
            assertTrue("weight_scale_2 = amax / 2688 of a ~N(0, 1/128) expert: " + ws[e],
                    ws[e] > 1e-4f && ws[e] < 1e-3f);
        }
        assertTrue("each expert has its own global scale", ws[0] != ws[1]);
        assertNull("the I8 experts carry no output scale",
                plan(mixtral(), online(OnlineQuant.EXPERTS_INT8)).tensor("blk.0.ffn_up_exps.weight").source()
                        .outputScale());
    }

    @Test
    public void aShorthandQuantizationConfigQuantizesAtLoadAndAnExplicitOptionWins() throws IOException {
        Path dir = llama("{'quant_method':'fp8_per_block'}");
        assertEquals(GGMLType.F8_E4M3_B128, plan(dir, options).type("blk.0.attn_q.weight"));
        assertEquals(GGMLType.BF16, plan(dir, options).type("output.weight"));
        assertEquals(GGMLType.Q8_0, plan(dir, online(OnlineQuant.Q8_0)).type("blk.0.attn_q.weight"));
        Path experts = llama("{'quant_method':'experts_int8'}");
        assertEquals("experts_int8 leaves a dense model's linears", GGMLType.BF16,
                plan(experts, options).type("blk.0.attn_q.weight"));
    }

    @Test
    public void aSerializedFp8ConfigKeepsItsFloatModulesAsStored() throws IOException {
        Path dir = llama("{'quant_method':'fp8','activation_scheme':'dynamic','weight_block_size':[128,128]}");
        assertEquals(GGMLType.BF16, plan(dir, options).type("blk.0.attn_q.weight"));
        assertEquals("--quant-online composes over the serialized method's float modules", GGMLType.F8_E4M3_B128,
                plan(dir, online(OnlineQuant.FP8_PER_TENSOR)).type("blk.0.attn_q.weight"));
    }

    @Test
    public void mxfp4QuantizesTheLinearsAndRoutedExpertsButNeverTheEmbeddingOrLmHead() throws IOException {
        TensorPlan dense = plan(llama(""), online(OnlineQuant.MXFP4));
        for (String w : new String[]{"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}) {
            assertEquals(w, GGMLType.MXFP4, dense.type("blk.0." + w + ".weight"));
        }
        assertEquals(GGMLType.BF16, dense.type("token_embd.weight"));
        assertEquals(GGMLType.BF16, dense.type("output.weight"));
        TensorPlan moe = plan(mixtral(), online(OnlineQuant.MXFP4));
        for (String w : new String[]{"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "attn_q"}) {
            assertEquals(w, GGMLType.MXFP4, moe.type("blk.0." + w + ".weight"));
        }
        assertEquals("the router stays", GGMLType.F32, moe.type("blk.0.ffn_gate_inp.weight"));
    }

    @Test
    public void leaveOutputTensorKeepsTheLmHeadAsStored() throws IOException {
        Path dir = llama("");
        assertEquals("llama-quantize quantizes output.weight by default", GGMLType.Q8_0,
                plan(dir, online(OnlineQuant.Q8_0)).type("output.weight"));
        TensorPlan kept = plan(dir, online(OnlineQuant.Q8_0).toBuilder().quantizeOutputTensor(false).build());
        assertEquals("--leave-output-tensor keeps the LM head", GGMLType.BF16, kept.type("output.weight"));
        assertEquals("the other linears still quantize", GGMLType.Q8_0, kept.type("blk.0.attn_q.weight"));
    }

    @Test
    public void theMxfp4ShorthandNamesAnMxfp4CheckpointsOwnQuantization() throws IOException {
        Path dir = llama("{'quant_method':'mxfp4'}");
        assertEquals("no online quantization is added", GGMLType.BF16,
                plan(dir, online(OnlineQuant.MXFP4)).type("blk.0.attn_q.weight"));
        assertEquals("a quantization config composes over the checkpoint's float modules", GGMLType.MXFP4,
                plan(dir, online(OnlineQuant.MXFP4).toBuilder().quantConfigJson("{\"ignore\": [\"*.o_proj\"]}")
                        .build()).type("blk.0.attn_q.weight"));
        assertEquals("another method quantizes", GGMLType.MXFP8,
                plan(dir, online(OnlineQuant.MXFP8)).type("blk.0.attn_q.weight"));
    }

    @Test
    public void theOnlineMethodQuantizesTheLayersItsConfigSelects() throws IOException {
        Path dir = mixtral();
        TensorPlan p = plan(dir, options.toBuilder().quantConfigJson("{\"linear\": \"mxfp8\","
                + " \"moe\": \"int8_per_channel_weight_only\", \"ignore\": [\"re:.*\\\\.o_proj$\"]}").build());
        assertEquals(GGMLType.MXFP8, p.type("blk.0.attn_q.weight"));
        assertEquals("ignored", GGMLType.BF16, p.type("blk.0.attn_output.weight"));
        assertEquals(GGMLType.I8_B128, p.type("blk.0.ffn_down_exps.weight"));
        assertEquals(GGMLType.BF16, p.type("output.weight"));
        TensorPlan off = plan(dir, online(OnlineQuant.FP8_PER_BLOCK).toBuilder()
                .quantConfigJson("{\"moe\": {\"weight\": null}}").build());
        assertEquals(GGMLType.F8_E4M3_B128, off.type("blk.0.attn_q.weight"));
        assertEquals("a null weight leaves the experts", GGMLType.BF16, off.type("blk.0.ffn_up_exps.weight"));
    }

    @Test
    public void unusableOnlineConfigurationsAreRefusedByName() throws IOException {
        Path dense = llama("");
        String none = assertThrows(GgufFormatException.class, () -> plan(dense, online(OnlineQuant.ONLINE)))
                .getMessage();
        assertTrue(none, none.contains("neuralfree.quant.online=online") && none.contains("none is given"));
        String noSpec = assertThrows(GgufFormatException.class, () -> plan(dense, options.toBuilder()
                .quantConfigJson("{\"ignore\": [\"lm_head\"]}").build())).getMessage();
        assertTrue(noSpec, noSpec.contains("at least one of quantization_config.linear"));
        String shards = assertThrows(GgufFormatException.class, () -> plan(dense, options.toBuilder()
                .quantConfigJson("{\"targets\": {\"*.q_proj\": \"mxfp8\"}}").build())).getMessage();
        assertTrue(shards, shards.contains("online quantization of blk.0.attn_")
                && shards.contains("every shard of a fused layer must match a target"));
        String partialIgnore = assertThrows(GgufFormatException.class, () -> plan(dense, options.toBuilder()
                .quantConfigJson("{\"linear\": \"mxfp8\", \"ignore\": [\"*.v_proj\"]}").build())).getMessage();
        assertTrue(partialIgnore, partialIgnore.contains("ignore matches only some shards of the fused layer"
                + " model.layers.0.self_attn.qkv_proj"));
        Path ckpt = llama("{'quant_method':'online','linear':{'weight':'fp8_per_token'}}");
        String unsupported = assertThrows(GgufFormatException.class, () -> plan(ckpt, options)).getMessage();
        assertTrue(unsupported, unsupported.contains("weight fp8_per_token is not supported"));
        String malformed = assertThrows(GgufFormatException.class,
                () -> plan(llama("{'quant_method':'online','linaer':'mxfp8'}"), options)).getMessage();
        assertTrue(malformed, malformed.contains("quantization_config: unknown member 'linaer'"));
    }

    @Test
    public void aCheckpointOnlineConfigMergesUnderTheUserConfig() throws IOException {
        Path dir = mixtral();
        Path ckpt = llama("{'quant_method':'online','linear':'mxfp8','ignore':['*.o_proj']}");
        TensorPlan own = plan(ckpt, options);
        assertEquals(GGMLType.MXFP8, own.type("blk.0.attn_q.weight"));
        assertEquals(GGMLType.BF16, own.type("blk.0.attn_output.weight"));
        TensorPlan merged = plan(ckpt, options.toBuilder().quantConfigJson("{\"linear\": \"fp8_per_channel\"}")
                .build());
        assertEquals("the user's linear wins", GGMLType.F8_E4M3_B128, merged.type("blk.0.attn_q.weight"));
        assertEquals("the checkpoint's ignore stays", GGMLType.BF16, merged.type("blk.0.attn_output.weight"));
        assertEquals(GGMLType.BF16, plan(dir, options).type("blk.0.attn_q.weight"));
    }

    // ------------------------------------------------------------------ inference

    private float[] logits(Path dir, ModelSourceOptions o) throws IOException {
        EngineConfig cfg = EngineConfig.builder()
                .backend(EngineConfig.BackendMode.CPU)
                .memory(MemoryConfig.builder().nCtx(32).ubatch(8).build())
                .sourceOptions(o)
                .build();
        try (EngineModel m = LLMModelFactory.open(dir.toString(), cfg)) {
            return m.forwardBackend().prefill(PROMPT, 0).clone();
        }
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }

    private void assertCloseLogits(Path dir, OnlineQuant q, double minCosine) throws IOException {
        float[] ref = logits(dir, options);
        float[] got = logits(dir, online(q));
        assertEquals(ref.length, got.length);
        double c = cosine(ref, got);
        assertTrue(q + ": cosine " + c + " to the BF16 logits", c >= minCosine);
        assertTrue(q + " changed the weights the logits come from", !java.util.Arrays.equals(ref, got));
    }

    @Test
    public void denseQuantizedModelsRunWithLogitsCloseToBf16() throws IOException {
        Path dir = llama("");
        assertCloseLogits(dir, OnlineQuant.FP8_PER_TENSOR, 0.995);
        assertCloseLogits(dir, OnlineQuant.FP8_PER_BLOCK, 0.995);
        assertCloseLogits(dir, OnlineQuant.FP8_PER_CHANNEL, 0.995);
        assertCloseLogits(dir, OnlineQuant.MXFP8, 0.995);
    }

    @Test
    public void moeQuantizedModelsRunWithLogitsCloseToBf16() throws IOException {
        Path dir = mixtral();
        assertCloseLogits(dir, OnlineQuant.EXPERTS_INT8, 0.999);
        assertCloseLogits(dir, OnlineQuant.NVFP4_PER_TOKEN, 0.97);
        assertCloseLogits(dir, OnlineQuant.FP8_PER_TENSOR, 0.995);
        assertCloseLogits(dir, OnlineQuant.FP8_PER_BLOCK, 0.995);
    }

    @Test
    public void mxfp4ModelsRunWithLogitsCloseToBf16() throws IOException {
        assertCloseLogits(llama(""), OnlineQuant.MXFP4, 0.95);
        assertCloseLogits(mixtral(), OnlineQuant.MXFP4, 0.95);
    }
}
