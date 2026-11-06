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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.StringJoiner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Bit parity of NeuralFree's online FP8 quantization with vLLM's where NeuralFree stores the weight in another
 * layout than vLLM quantizes it ({@code vllm-online-layouts.txt}, from {@code vllm_online_reference.py}): rows of no
 * whole 128-value blocks as F8_E4M3_B32 runs, LLaMA's rope-permuted q/k tiled in checkpoint row order, MLA's
 * {@code fused_qkv_a_proj} and {@code kv_b_proj} (k_b stored transposed), and Qwen3.5's {@code in_proj_qkvz} and
 * {@code in_proj_ba} across the regrouped value heads, matched by vLLM's layer names.
 */
public class HfOnlineQuantLayoutParityTest {

    /** The golden digests: scenario and method, then "{@code <gguf tensor> <kind>}", then the digest. */
    private static final Map<String, Map<String, String>> GOLDEN = new LinkedHashMap<>();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @BeforeClass
    public static void loadGolden() throws IOException {
        for (String[] f : VllmParityDigests.lines("vllm-online-layouts.txt")) {
            GOLDEN.computeIfAbsent(f[0] + " " + f[1], k -> new LinkedHashMap<>()).put(f[2] + " " + f[3], f[4]);
        }
    }

    // ------------------------------------------------------------------ checkpoints

    private static void weight(SyntheticHf.Builder b, String name, int rows, int cols, long seed, int base) {
        b.tensor(name + ".weight", StDType.BF16, new long[]{rows, cols},
                VllmParityDigests.lcgBf16(rows, cols, seed, base));
    }

    private static void filled(SyntheticHf.Builder b, String name, float value, long... shape) {
        int n = 1;
        for (long d : shape) {
            n *= (int) d;
        }
        float[] v = new float[n];
        Arrays.fill(v, value);
        b.tensor(name, StDType.BF16, shape, SyntheticHf.bf16(v));
    }

    private static String common(String arch, String modelType, int hidden) {
        return "'architectures':['" + arch + "'],'model_type':'" + modelType + "','hidden_size':" + hidden
                + ",'num_hidden_layers':1,'max_position_embeddings':128,'rms_norm_eps':1e-06,'rope_theta':10000.0"
                + ",'vocab_size':16,'torch_dtype':'bfloat16','bos_token_id':0,'eos_token_id':1"
                + ",'tie_word_embeddings':false";
    }

    private static void ends(SyntheticHf.Builder b, String model, int hidden) {
        filled(b, model + "embed_tokens.weight", 0.01f, 16, hidden);
        filled(b, model + "norm.weight", 1f, hidden);
        filled(b, "lm_head.weight", 0.01f, 16, hidden);
    }

    /** LLaMA with 4 heads of 96, so a head's rope permutation moves rows across 128-row tiles. */
    private Path llama() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{" + common("LlamaForCausalLM",
                "llama", 384) + ",'num_attention_heads':4,'num_key_value_heads':2,'intermediate_size':320}"));
        String p = "model.layers.0.";
        filled(b, p + "input_layernorm.weight", 1f, 384);
        weight(b, p + "self_attn.q_proj", 384, 384, 41, 118);
        weight(b, p + "self_attn.k_proj", 192, 384, 42, 121);
        weight(b, p + "self_attn.v_proj", 192, 384, 43, 116);
        weight(b, p + "self_attn.o_proj", 384, 384, 44, 119);
        filled(b, p + "post_attention_layernorm.weight", 1f, 384);
        weight(b, p + "mlp.gate_proj", 320, 384, 45, 117);
        weight(b, p + "mlp.up_proj", 320, 384, 46, 120);
        weight(b, p + "mlp.down_proj", 384, 320, 47, 118);
        ends(b, "model.", 384);
        return b.write(tmp.newFolder().toPath());
    }

    /** DeepSeek-V2 MLA with a q LoRA (fused_qkv_a_proj) and kv_b rows of 160 values. */
    private Path deepseekV2() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{" + common("DeepseekV2ForCausalLM",
                "deepseek_v2", 256) + ",'num_attention_heads':2,'num_key_value_heads':2,'intermediate_size':256"
                + ",'moe_intermediate_size':64,'n_routed_experts':2,'num_experts_per_tok':1,'first_k_dense_replace':1"
                + ",'q_lora_rank':192,'kv_lora_rank':160,'qk_nope_head_dim':64,'qk_rope_head_dim':32"
                + ",'v_head_dim':64}"));
        String p = "model.layers.0.";
        filled(b, p + "input_layernorm.weight", 1f, 256);
        weight(b, p + "self_attn.q_a_proj", 192, 256, 51, 118);
        filled(b, p + "self_attn.q_a_layernorm.weight", 1f, 192);
        weight(b, p + "self_attn.q_b_proj", 192, 192, 52, 120);
        weight(b, p + "self_attn.kv_a_proj_with_mqa", 192, 256, 53, 117);
        filled(b, p + "self_attn.kv_a_layernorm.weight", 1f, 160);
        weight(b, p + "self_attn.kv_b_proj", 256, 160, 54, 121);
        weight(b, p + "self_attn.o_proj", 256, 128, 55, 119);
        filled(b, p + "post_attention_layernorm.weight", 1f, 256);
        weight(b, p + "mlp.gate_proj", 256, 256, 56, 117);
        weight(b, p + "mlp.up_proj", 256, 256, 57, 120);
        weight(b, p + "mlp.down_proj", 256, 256, 58, 118);
        ends(b, "model.", 256);
        return b.write(tmp.newFolder().toPath());
    }

    /**
     * Qwen3.5 (the vision-language checkpoint layout) with one linear-attention layer: 2 key heads of 32 and 4
     * value heads of 48, so the regrouped value heads move rows across 128-row tiles.
     */
    private Path qwen35(String quantConfig) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen3_5ForConditionalGeneration'],'model_type':'qwen3_5','tie_word_embeddings':false"
                + ",'text_config':{'model_type':'qwen3_5_text','hidden_size':256,'num_hidden_layers':1"
                + ",'num_attention_heads':2,'num_key_value_heads':1,'head_dim':64,'intermediate_size':256"
                + ",'max_position_embeddings':128,'rms_norm_eps':1e-06,'vocab_size':16,'bos_token_id':0"
                + ",'eos_token_id':1,'torch_dtype':'bfloat16','linear_num_key_heads':2,'linear_num_value_heads':4"
                + ",'linear_key_head_dim':32,'linear_value_head_dim':48,'linear_conv_kernel_dim':4"
                + ",'layer_types':['linear_attention'],'full_attention_interval':4"
                + ",'rope_parameters':{'rope_type':'default','rope_theta':10000.0,'mrope_section':[11,11,10]"
                + ",'partial_rotary_factor':0.25}}"
                + (quantConfig.isEmpty() ? "" : ",'quantization_config':" + quantConfig) + "}"));
        String m = "model.language_model.";
        String p = m + "layers.0.";
        filled(b, p + "input_layernorm.weight", 0f, 256);
        weight(b, p + "linear_attn.in_proj_qkv", 320, 256, 61, 118);
        weight(b, p + "linear_attn.in_proj_z", 192, 256, 62, 120);
        weight(b, p + "linear_attn.in_proj_b", 4, 256, 63, 117);
        weight(b, p + "linear_attn.in_proj_a", 4, 256, 64, 121);
        filled(b, p + "linear_attn.conv1d.weight", 0.1f, 320, 1, 4);
        filled(b, p + "linear_attn.A_log", 0f, 4);
        filled(b, p + "linear_attn.dt_bias", 1f, 4);
        filled(b, p + "linear_attn.norm.weight", 1f, 48);
        weight(b, p + "linear_attn.out_proj", 256, 192, 65, 119);
        filled(b, p + "post_attention_layernorm.weight", 0f, 256);
        weight(b, p + "mlp.gate_proj", 256, 256, 66, 117);
        weight(b, p + "mlp.up_proj", 256, 256, 67, 120);
        weight(b, p + "mlp.down_proj", 256, 256, 68, 118);
        filled(b, m + "embed_tokens.weight", 0.01f, 16, 256);
        filled(b, m + "norm.weight", 0f, 256);
        filled(b, "lm_head.weight", 0.01f, 16, 256);
        return b.write(tmp.newFolder().toPath());
    }

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder().toPath()).build();
    }

    private ModelSourceOptions options(OnlineQuant q) throws IOException {
        return options().toBuilder().onlineQuant(q).build();
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

    // ------------------------------------------------------------------ parity

    /**
     * Asserts every golden digest of {@code scenario}'s {@code method} on {@code p}: an FP8 tensor of whole
     * 128-value rows as F8_E4M3_B128, of 32-value runs as F8_E4M3_B32, a {@code values} digest as BF16.
     */
    private static void assertParity(String scenario, String method, TensorPlan p) throws IOException {
        Map<String, String> golden = GOLDEN.get(scenario + " " + method);
        assertNotNull(scenario + " " + method, golden);
        Map<String, Map<String, String>> computed = new LinkedHashMap<>();
        StringJoiner mismatches = new StringJoiner("\n");
        for (Map.Entry<String, String> g : golden.entrySet()) {
            String[] key = g.getKey().split(" ");
            if (!computed.containsKey(key[0])) {
                GGMLType t = p.type(key[0]);
                assertTrue(scenario + " " + method + " " + key[0] + " is " + t, key[1].equals("values")
                        ? t == GGMLType.BF16 : t == GGMLType.F8_E4M3_B128 || t == GGMLType.F8_E4M3_B32);
                computed.put(key[0], VllmParityDigests.digests(p.tensor(key[0]).source()));
            }
            String got = computed.get(key[0]).get(key[1]);
            if (!g.getValue().equals(got)) {
                mismatches.add(scenario + " " + method + " " + g.getKey() + ": vLLM " + g.getValue() + ", NeuralFree "
                        + got);
            }
        }
        assertEquals("", mismatches.toString());
    }

    @Test
    public void rowsOfNoWhole128ValueBlocksQuantizeAs32ValueRunsOfTheirScales() throws IOException {
        Path dir = HfOnlineQuantParityTest.qwen2Moe(tmp, "");
        String[] methods = {"fp8_per_tensor", "fp8_per_block", "fp8_per_channel"};
        OnlineQuant[] quants = {OnlineQuant.FP8_PER_TENSOR, OnlineQuant.FP8_PER_BLOCK, OnlineQuant.FP8_PER_CHANNEL};
        for (int i = 0; i < methods.length; i++) {
            String method = methods[i];
            withPlan(dir, options(quants[i]), p -> {
                assertParity("qwen2_moe", method, p);
                assertEquals(GGMLType.F8_E4M3_B32, p.type("blk.0.ffn_down_exps.weight"));
                assertEquals(GGMLType.F8_E4M3_B32, p.type("blk.0.ffn_down_shexp.weight"));
                assertEquals(GGMLType.F8_E4M3_B128, p.type("blk.0.attn_q.weight"));
            });
        }
    }

    @Test
    public void ropePermutedQueryAndKeyRowsShareTheTilesOfTheirCheckpointOrder() throws IOException {
        Path dir = llama();
        withPlan(dir, options(OnlineQuant.FP8_PER_BLOCK), p -> assertParity("llama", "fp8_per_block", p));
        withPlan(dir, options(OnlineQuant.FP8_PER_TENSOR), p -> {
            assertParity("llama", "fp8_per_tensor", p);
            assertEquals("the 320-wide down rows are 32-value runs", GGMLType.F8_E4M3_B32,
                    p.type("blk.0.ffn_down.weight"));
        });
    }

    @Test
    public void mlaFusesQaWithKvaAndQuantizesKvbAcrossItsSplitParts() throws IOException {
        Path dir = deepseekV2();
        withPlan(dir, options(OnlineQuant.FP8_PER_TENSOR), p -> assertParity("deepseek_v2", "fp8_per_tensor", p));
        withPlan(dir, options(OnlineQuant.FP8_PER_BLOCK), p -> {
            assertParity("deepseek_v2", "fp8_per_block", p);
            assertEquals("k_b's 64-value rows run down kv_b's columns", GGMLType.F8_E4M3_B32,
                    p.type("blk.0.attn_k_b.weight"));
        });
        withPlan(dir, options(OnlineQuant.FP8_PER_CHANNEL), p -> {
            assertParity("deepseek_v2", "fp8_per_channel", p);
            assertEquals("per-row scales run across k_b's rows: k_b is vLLM's dequantized W_UK", GGMLType.BF16,
                    p.type("blk.0.attn_k_b.weight"));
        });
        withPlan(dir, options(OnlineQuant.Q8_0), p -> {
            // llama-quant.cpp only keeps the MLA projections float on GLM5-next (:377-379); DeepSeek's are
            // ordinary matmul weights, so Q8_0 gives them the default type (llama.cpp's deepseek-v3.1 snapshot).
            assertEquals("the llama-quantize targets quantize DeepSeek's MLA projections", GGMLType.Q8_0,
                    p.type("blk.0.attn_kv_a_mqa.weight"));
            assertEquals(GGMLType.Q8_0, p.type("blk.0.attn_v_b.weight"));
            assertEquals(GGMLType.Q8_0, p.type("blk.0.attn_q_a.weight"));
            assertEquals("the stored attn_k_b is vLLM's dequantized W_UK, so it stays in the float type",
                    GGMLType.BF16, p.type("blk.0.attn_k_b.weight"));
        });
    }

    @Test
    public void gatedDeltaNetProjectionsShareScalesAcrossTheirRegroupedValueHeads() throws IOException {
        withPlan(qwen35(""), options(OnlineQuant.FP8_PER_BLOCK), p -> assertParity("qwen35", "fp8_per_block", p));
        withPlan(qwen35(""), options(OnlineQuant.FP8_PER_TENSOR),
                p -> assertParity("qwen35", "fp8_per_tensor", p));
    }

    @Test
    public void quantizationConfigPatternsMatchVllmsMappedLayerNames() throws IOException {
        Path dir = qwen35("{'quant_method':'online','linear':'fp8_per_block',"
                + "'ignore':['re:^language_model\\\\.model\\\\.layers\\\\.0\\\\.mlp\\\\..*']}");
        withPlan(dir, options(), p -> {
            assertParity("qwen35", "fp8_per_block", p);
            for (String mlp : new String[]{"ffn_gate", "ffn_up", "ffn_down"}) {
                assertEquals("vLLM names the MLP language_model.model.layers.0.mlp", GGMLType.BF16,
                        p.type("blk.0." + mlp + ".weight"));
            }
            assertEquals("out_proj's 192-wide rows are 32-value runs", GGMLType.F8_E4M3_B32,
                    p.type("blk.0.ssm_out.weight"));
        });
        Path checkpointNames = qwen35("{'quant_method':'online','linear':'fp8_per_block',"
                + "'ignore':['re:^model\\\\.language_model\\\\.layers\\\\.0\\\\.mlp\\\\..*']}");
        withPlan(checkpointNames, options(), p -> assertEquals("vLLM never sees the checkpoint's name",
                GGMLType.F8_E4M3_B128, p.type("blk.0.ffn_gate.weight")));
    }
}
