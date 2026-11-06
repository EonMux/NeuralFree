package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OnlineQuant;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.StringJoiner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Bit parity of NeuralFree's online quantization with vLLM's on a small Qwen2-MoE checkpoint
 * ({@code vllm-online-parity.txt}, produced by vLLM's own quantizers): the fused projections' shared fp8 scales
 * (q/k/v, the shared expert's gate/up, the routed experts' w13 per expert, with vLLM's per-shard row padding for
 * the routed per-block tiles), mxfp4 on the linear layers and experts, and nvfp4_per_token's per-expert global
 * scales, reached through {@code --quant-online}, {@code --quantization-config} and a checkpoint's quant_method
 * {@code online} configuration alike.
 */
public class HfOnlineQuantParityTest {

    private static final int HIDDEN = 256;
    private static final int HEADS = 4;
    private static final int KV_ROWS = 64;
    private static final int MOE_FFN = 192;
    private static final int SHARED_FFN = 160;
    private static final int EXPERTS = 2;
    private static final int VOCAB = 16;
    private static final String P = "model.layers.0.";

    /** The golden digests: method, then "{@code <gguf tensor> <kind>}", then the digest. */
    private static final Map<String, Map<String, String>> GOLDEN = new LinkedHashMap<>();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void loadGolden() throws IOException {
        for (String[] f : VllmParityDigests.lines("vllm-online-parity.txt")) {
            String digest = String.join(" ", java.util.Arrays.copyOfRange(f, 3, f.length));
            GOLDEN.computeIfAbsent(f[0], k -> new LinkedHashMap<>()).put(f[1] + " " + f[2], digest);
        }
    }

    // ------------------------------------------------------------------ the checkpoint

    private static void weight(SyntheticHf.Builder b, String name, int rows, int cols, long seed, int base) {
        b.tensor(P + name + ".weight", StDType.BF16, new long[]{rows, cols},
                VllmParityDigests.lcgBf16(rows, cols, seed, base));
    }

    private static void filled(SyntheticHf.Builder b, String name, long[] shape, float value) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        float[] v = new float[n];
        java.util.Arrays.fill(v, value);
        b.tensor(name, StDType.BF16, shape, SyntheticHf.bf16(v));
    }

    /** The golden's checkpoint; {@code quantConfig} is a quantization_config member or empty. */
    static Path qwen2Moe(TemporaryFolder tmp, String quantConfig) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2MoeForCausalLM'],'model_type':'qwen2_moe','hidden_size':" + HIDDEN
                + ",'num_hidden_layers':1,'num_attention_heads':" + HEADS + ",'num_key_value_heads':1"
                + ",'intermediate_size':384,'moe_intermediate_size':" + MOE_FFN + ",'shared_expert_intermediate_size':"
                + SHARED_FFN + ",'num_experts':" + EXPERTS + ",'num_experts_per_tok':2,'decoder_sparse_step':1"
                + ",'mlp_only_layers':[],'norm_topk_prob':false,'max_position_embeddings':128,'rms_norm_eps':1e-06"
                + ",'rope_theta':10000.0,'vocab_size':" + VOCAB + ",'torch_dtype':'bfloat16','bos_token_id':0"
                + ",'eos_token_id':1,'tie_word_embeddings':false"
                + (quantConfig.isEmpty() ? "" : ",'quantization_config':" + quantConfig) + "}"));
        filled(b, "model.embed_tokens.weight", new long[]{VOCAB, HIDDEN}, 0.01f);
        filled(b, P + "input_layernorm.weight", new long[]{HIDDEN}, 1f);
        weight(b, "self_attn.q_proj", HIDDEN, HIDDEN, 11, 118);
        weight(b, "self_attn.k_proj", KV_ROWS, HIDDEN, 12, 121);
        weight(b, "self_attn.v_proj", KV_ROWS, HIDDEN, 13, 116);
        filled(b, P + "self_attn.q_proj.bias", new long[]{HIDDEN}, 0f);
        filled(b, P + "self_attn.k_proj.bias", new long[]{KV_ROWS}, 0f);
        filled(b, P + "self_attn.v_proj.bias", new long[]{KV_ROWS}, 0f);
        weight(b, "self_attn.o_proj", HIDDEN, HIDDEN, 14, 119);
        filled(b, P + "post_attention_layernorm.weight", new long[]{HIDDEN}, 1f);
        filled(b, P + "mlp.gate.weight", new long[]{EXPERTS, HIDDEN}, 0.01f);
        long[][] experts = {{21, 117, 22, 120, 23, 117}, {24, 121, 25, 118, 26, 119}};
        for (int e = 0; e < EXPERTS; e++) {
            long[] x = experts[e];
            weight(b, "mlp.experts." + e + ".gate_proj", MOE_FFN, HIDDEN, x[0], (int) x[1]);
            weight(b, "mlp.experts." + e + ".up_proj", MOE_FFN, HIDDEN, x[2], (int) x[3]);
            weight(b, "mlp.experts." + e + ".down_proj", HIDDEN, MOE_FFN, x[4], (int) x[5]);
        }
        weight(b, "mlp.shared_expert.gate_proj", SHARED_FFN, HIDDEN, 31, 117);
        weight(b, "mlp.shared_expert.up_proj", SHARED_FFN, HIDDEN, 32, 120);
        weight(b, "mlp.shared_expert.down_proj", HIDDEN, SHARED_FFN, 33, 118);
        filled(b, P + "mlp.shared_expert_gate.weight", new long[]{1, HIDDEN}, 0.01f);
        filled(b, "model.norm.weight", new long[]{HIDDEN}, 1f);
        filled(b, "lm_head.weight", new long[]{VOCAB, HIDDEN}, 0.01f);
        return b.write(tmp.newFolder().toPath());
    }

    private Path qwen2Moe(String quantConfig) throws IOException {
        return qwen2Moe(tmp, quantConfig);
    }

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder().toPath()).build();
    }

    /** A check of a plan, run while its source is open (the planned tensors read the checkpoint). */
    private interface PlanCheck {
        void check(TensorPlan p) throws IOException;
    }

    private static void withPlan(Path dir, ModelSourceOptions o, PlanCheck check) throws IOException {
        try (HfModelSource s = HfModelSource.open(dir, o, null)) {
            check.check(s.plan());
        }
    }

    /** Asserts every golden digest of {@code method} on {@code p}, each tensor planned as {@code type}. */
    private static void assertParity(String method, TensorPlan p, GGMLType type) throws IOException {
        Map<String, String> golden = GOLDEN.get(method);
        assertNotNull(method, golden);
        Map<String, Map<String, String>> computed = new LinkedHashMap<>();
        StringJoiner mismatches = new StringJoiner("\n");
        for (Map.Entry<String, String> g : golden.entrySet()) {
            String[] key = g.getKey().split(" ");
            if (!computed.containsKey(key[0])) {
                assertEquals(method + " " + key[0], type, p.type(key[0]));
                computed.put(key[0], VllmParityDigests.digests(p.tensor(key[0]).source()));
            }
            String got = computed.get(key[0]).get(key[1]);
            if (!g.getValue().equals(got)) {
                mismatches.add(method + " " + g.getKey() + ": vLLM " + g.getValue() + ", NeuralFree " + got);
            }
        }
        assertEquals("", mismatches.toString());
    }

    // ------------------------------------------------------------------ parity

    @Test
    public void fp8PerTensorScalesEachFusedProjectionByItsJointAmax() throws IOException {
        withPlan(qwen2Moe(""), options().toBuilder().onlineQuant(OnlineQuant.FP8_PER_TENSOR).build(), p -> {
            assertParity("fp8_per_tensor", p, GGMLType.F8_E4M3_B128);
            assertEquals("the 192-wide expert rows are 32-value runs", GGMLType.F8_E4M3_B32,
                    p.type("blk.0.ffn_down_exps.weight"));
            assertEquals(GGMLType.F8_E4M3_B32, p.type("blk.0.ffn_down_shexp.weight"));
            assertEquals("vLLM builds the shared-expert gate without a quantization config", GGMLType.F32,
                    p.type("blk.0.ffn_gate_inp_shexp.weight"));
            assertEquals(GGMLType.BF16, p.type("output.weight"));
        });
    }

    @Test
    public void fp8PerBlockTilesTheFusedRowsWithPaddedExpertShards() throws IOException {
        withPlan(qwen2Moe(""), options().toBuilder().onlineQuant(OnlineQuant.FP8_PER_BLOCK).build(),
                p -> assertParity("fp8_per_block", p, GGMLType.F8_E4M3_B128));
    }

    @Test
    public void mxfp4QuantizesTheLinearsAndExpertsAsTheTritonKernel() throws IOException {
        withPlan(qwen2Moe(""), options().toBuilder().onlineQuant(OnlineQuant.MXFP4).build(), p -> {
            assertParity("mxfp4", p, GGMLType.MXFP4);
            assertEquals(GGMLType.BF16, p.type("token_embd.weight"));
            assertEquals(GGMLType.BF16, p.type("output.weight"));
            assertEquals(GGMLType.F32, p.type("blk.0.ffn_gate_inp.weight"));
        });
    }

    @Test
    public void nvfp4PerTokenScalesEachExpertOverItsW13() throws IOException {
        withPlan(qwen2Moe(""), options().toBuilder().onlineQuant(OnlineQuant.NVFP4_PER_TOKEN).build(), p -> {
            assertParity("nvfp4_per_token", p, GGMLType.NVFP4);
            assertEquals("nvfp4_per_token leaves the linears", GGMLType.BF16, p.type("blk.0.attn_q.weight"));
            assertEquals(GGMLType.BF16, p.type("blk.0.ffn_up_shexp.weight"));
        });
    }

    @Test
    public void aQuantizationConfigReachesTheSameCodes() throws IOException {
        Path dir = qwen2Moe("");
        withPlan(dir, options().toBuilder().onlineQuant(OnlineQuant.ONLINE)
                .quantConfigJson("{\"linear\": \"fp8_per_block\", \"moe\": {\"weight\": \"fp8_per_block_static\"}}")
                .build(), p -> assertParity("fp8_per_block", p, GGMLType.F8_E4M3_B128));
        withPlan(dir, options().toBuilder().quantConfigJson("{\"targets\": {\"re:.*\\\\.experts$\":"
                + " \"nvfp4_per_token\", \"*.self_attn.qkv_proj\": \"fp8_per_tensor\"}}").build(), p -> {
                    assertParity("nvfp4_per_token", p, GGMLType.NVFP4);
                    for (String w : new String[]{"attn_q", "attn_k", "attn_v"}) {
                        String name = "blk.0." + w + ".weight";
                        assertEquals(name, GGMLType.F8_E4M3_B128, p.type(name));
                        assertEquals(name, GOLDEN.get("fp8_per_tensor").get(name + " codes"),
                                VllmParityDigests.digests(p.tensor(name).source()).get("codes"));
                    }
                    assertEquals("no target matches o_proj", GGMLType.BF16, p.type("blk.0.attn_output.weight"));
                    assertEquals(GGMLType.BF16, p.type("blk.0.ffn_gate_shexp.weight"));
                });
    }

    @Test
    public void aCheckpointOnlineConfigQuantizesWhatItSelects() throws IOException {
        Path dir = qwen2Moe("{'quant_method':'online','linear':'mxfp4','moe':'mxfp4','ignore':['re:.*o_proj']}");
        Map<String, String> golden = GOLDEN.get("mxfp4");
        withPlan(dir, options(), p -> {
            assertEquals(GGMLType.BF16, p.type("blk.0.attn_output.weight"));
            for (String name : new String[]{"blk.0.attn_q.weight", "blk.0.ffn_down_shexp.weight",
                "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"}) {
                assertEquals(name, GGMLType.MXFP4, p.type(name));
                Map<String, String> d = VllmParityDigests.digests(p.tensor(name).source());
                assertEquals(name, golden.get(name + " codes"), d.get("codes"));
                assertEquals(name, golden.get(name + " scales"), d.get("scales"));
            }
        });
        withPlan(dir, options().toBuilder().onlineQuant(OnlineQuant.FP8_PER_BLOCK).build(),
                p -> assertEquals("the flag names another method over the checkpoint's", GGMLType.F8_E4M3_B128,
                        p.type("blk.0.attn_output.weight")));
    }
}
