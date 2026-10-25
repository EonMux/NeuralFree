package com.eonmux.neuralfree.gguf.hf.parity;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.io.IOException;
import java.nio.file.Path;

/**
 * PT-1 fixtures of the MoE families: DeepSeek-V3's MLA with the kv_b split and stacked routed experts, and the
 * Qwen3-MoE fused expert tensors, each in its F32 and BF16 form. Every expected GGUF value and tensor here is
 * hand-derived from the converter rules the ports cite — never read back from NeuralFree:
 *
 * <ul>
 *   <li>{@code DeepseekV3Model} metadata: deepseek.py:319-388 through {@code HfDeepseekHooks} (the KV head count
 *       forced to 1, the MLA keys, the leading dense count, the expert keys); the {@code kv_b_proj} split of
 *       deepseek.py:434-450; the per-expert stacking of parts down_proj/gate_proj/up_proj (deepseek.py:416); the
 *       default tensor mapping with no Q/K permutation.</li>
 *   <li>{@code Qwen3MoeModel} metadata: qwen.py:86-93 via {@code HfQwenMoeHooks} (the generic rows plus
 *       {@code expert_feed_forward_length}); the fused {@code mlp.experts.gate_up_proj} split in halves along
 *       dim {@code -2}, the gate first, and the fused {@code mlp.experts.down_proj} mapping (qwen.py:109-122).</li>
 *   <li>The shared GPT-2 BPE vocabulary of {@link ParityFixturesDense#tokenizerKv} and the base.py tensor storage
 *       rules of {@code ParityFixturesDense} (rank ≤ 1 and %{*_norm.weight} tensors F32, every other weight through
 *       the auto dtype).</li>
 * </ul>
 *
 * <p>The checkpoint element values are the {@link SyntheticHf#values} rule seeded by each tensor's position, stored
 * F32 or BF16 (both exact for every value in this rule), so a deeper tensor widens from BF16 to F32 without loss.</p>
 */
public final class ParityFixturesMoe {

    private ParityFixturesMoe() {
    }

    // ------------------------------------------------------------------ deepseek

    /** The HF tensors of the deepseek-v3 checkpoint, in the order the planner walks them. */
    public static Path deepSeekV2Checkpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['DeepseekV3ForCausalLM'],'model_type':'deepseek_v3',"
                        + "'hidden_size':16,'intermediate_size':12,'moe_intermediate_size':8,"
                        + "'num_attention_heads':2,'num_key_value_heads':2,'num_hidden_layers':2,"
                        + "'max_position_embeddings':128,'vocab_size':16,'bos_token_id':0,'eos_token_id':1,"
                        + "'q_lora_rank':4,'kv_lora_rank':4,'qk_nope_head_dim':2,'qk_rope_head_dim':2,"
                        + "'v_head_dim':1,'n_routed_experts':2,'first_k_dense_replace':1,'rms_norm_eps':1e-05,"
                        + "'rope_theta':10000.0,'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"));
        b.floats(dtype, "model.embed_tokens.weight", 16, 16);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 16);
            b.floats(dtype, p + "self_attn.q_a_proj.weight", 4, 16);
            b.floats(dtype, p + "self_attn.q_a_layernorm.weight", 4);
            b.floats(dtype, p + "self_attn.q_b_proj.weight", 2 * 4, 4);
            b.floats(dtype, p + "self_attn.kv_a_proj_with_mqa.weight", 6, 16);
            b.floats(dtype, p + "self_attn.kv_a_layernorm.weight", 6);
            b.floats(dtype, p + "self_attn.kv_b_proj.weight", 2 * (2 + 1), 4);
            b.floats(dtype, p + "self_attn.o_proj.weight", 16, 16);
            b.floats(dtype, p + "post_attention_layernorm.weight", 16);
            if (l == 0) { // first_k_dense_replace 1: a dense FFN block, then the routed experts
                b.floats(dtype, p + "mlp.gate_proj.weight", 12, 16);
                b.floats(dtype, p + "mlp.up_proj.weight", 12, 16);
                b.floats(dtype, p + "mlp.down_proj.weight", 16, 12);
            } else {
                for (int e = 0; e < 2; e++) {
                    String x = p + "mlp.experts." + Integer.toString(e) + ".";
                    b.floats(dtype, x + "down_proj.weight", 16, 8);
                    b.floats(dtype, x + "gate_proj.weight", 8, 16);
                    b.floats(dtype, x + "up_proj.weight", 8, 16);
                }
            }
        }
        b.floats(dtype, "model.norm.weight", 16);
        b.floats(dtype, "lm_head.weight", 16, 16);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #deepSeekV2Checkpoint}. */
    public static GgufTestWriter deepSeekV2Expected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F32;
        GgufTestWriter g = new GgufTestWriter().architecture("deepseek2");
        g.str("general.architecture", "deepseek2");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 /* MOSTLY_BF16 */ : 0 /* ALL_F32 */);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("deepseek2.block_count", 2);
        g.u32("deepseek2.context_length", 128);
        g.u32("deepseek2.embedding_length", 16);
        g.u32("deepseek2.feed_forward_length", 12);
        g.u32("deepseek2.attention.head_count", 2);
        g.u32("deepseek2.attention.head_count_kv", 1); // MLA stored as MQA
        g.u32("deepseek2.leading_dense_block_count", 1);
        g.u32("deepseek2.vocab_size", 16);
        g.u32("deepseek2.attention.q_lora_rank", 4);
        g.u32("deepseek2.attention.kv_lora_rank", 4);
        g.u32("deepseek2.attention.key_length", 6); // kv_lora_rank + qk_rope_head_dim
        g.u32("deepseek2.attention.value_length", 4); // kv_lora_rank
        g.u32("deepseek2.attention.key_length_mla", 4); // qk_nope_head_dim + qk_rope_head_dim
        g.u32("deepseek2.attention.value_length_mla", 1); // v_head_dim
        g.u32("deepseek2.expert_feed_forward_length", 8);
        g.u32("deepseek2.expert_count", 2);
        g.u32("deepseek2.expert_shared_count", 0);
        g.f32("deepseek2.attention.layer_norm_rms_epsilon", 1e-5f);
        g.u32("deepseek2.rope.dimension_count", 2); // qk_rope_head_dim
        g.f32("deepseek2.rope.freq_base", 10000.0f);
        ParityFixturesDense.tokenizerKv(g, ParityFixturesDense.gpt2Tokens(16),
                ParityFixturesDense.gpt2Types(16));

        int seed = 0;
        ParityFixturesDense.planMatrix(g, "token_embd.weight", w, new long[]{16, 16}, seed++);
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(norm, 16));
            ParityFixturesDense.planMatrix(g, b + "attn_q_a.weight", w, new long[]{16, 4}, seed++);
            int qANorm = seed++;
            g.tensorFromFloats(b + "attn_q_a_norm.weight", GGMLType.F32, new long[]{4},
                    ParityFixturesDense.valuesAt(qANorm, 4));
            ParityFixturesDense.planMatrix(g, b + "attn_q_b.weight", w, new long[]{4, 8}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_kv_a_mqa.weight", w, new long[]{16, 6}, seed++);
            int kvANorm = seed++;
            g.tensorFromFloats(b + "attn_kv_a_norm.weight", GGMLType.F32, new long[]{6},
                    ParityFixturesDense.valuesAt(kvANorm, 6));
            int kvB = seed++;
            mlaKvB(g, b, kvB, w);
            ParityFixturesDense.planMatrix(g, b + "attn_output.weight", w, new long[]{16, 16}, seed++);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "ffn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(ffnNorm, 16));
            if (l == 0) {
                ParityFixturesDense.planMatrix(g, b + "ffn_gate.weight", w, new long[]{16, 12}, seed++);
                ParityFixturesDense.planMatrix(g, b + "ffn_up.weight", w, new long[]{16, 12}, seed++);
                ParityFixturesDense.planMatrix(g, b + "ffn_down.weight", w, new long[]{12, 16}, seed++);
            } else {
                // Two routed experts, each gate/up [8, 16] and down [16, 8], stacked as [E, rows, cols].
                int down0 = seed++, gate0 = seed++, up0 = seed++;
                int down1 = seed++, gate1 = seed++, up1 = seed++;
                expertStack(g, b + "ffn_gate_exps.weight", w, gate0, gate1, 8, 16);
                expertStack(g, b + "ffn_up_exps.weight", w, up0, up1, 8, 16);
                expertStack(g, b + "ffn_down_exps.weight", w, down0, down1, 16, 8);
            }
        }
        int finalNorm = seed++;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{16},
                ParityFixturesDense.valuesAt(finalNorm, 16));
        ParityFixturesDense.planMatrix(g, "output.weight", w, new long[]{16, 16}, seed);
        return g;
    }

    /**
     * deepseek.py:434-450: {@code kv_b [n_head*(nope+v), kv_lora]} becomes {@code k_b [n_head, kv_lora, nope]}
     * (transposed per head) and {@code v_b [n_head, v, kv_lora]}. The checkpoint holds {@code n_head=2},
     * {@code qk_nope_head_dim=2}, {@code v_head_dim=1}, {@code kv_lora_rank=4}, so {@code k_b} is {@code [2,4,2]}
     * and {@code v_b} is {@code [2,1,4]}; both keep the checkpoint's dtype.
     */
    private static void mlaKvB(GgufTestWriter g, String blk, int seed, GGMLType type) {
        float[] kvB = ParityFixturesDense.valuesAt(seed, 2 * 3 * 4);
        float[] kb = new float[2 * 4 * 2];
        for (int h = 0; h < 2; h++) {
            for (int j = 0; j < 4; j++) {
                for (int n = 0; n < 2; n++) {
                    kb[n + 2 * (j + 4 * h)] = kvB[(h * 3 + n) * 4 + j];
                }
            }
        }
        float[] vb = new float[2 * 1 * 4];
        for (int h = 0; h < 2; h++) {
            for (int j = 0; j < 4; j++) {
                vb[j + 4 * h] = kvB[(h * 3 + 2) * 4 + j];
            }
        }
        g.tensorFromFloats(blk + "attn_k_b.weight", type, new long[]{2, 4, 2}, kb);
        g.tensorFromFloats(blk + "attn_v_b.weight", type, new long[]{4, 1, 2}, vb);
    }

    /** The expert stack {@code [e0|e1]} of one part: ne {@code [cols, rows, 2]} over the two experts' matrices. */
    private static void expertStack(GgufTestWriter g, String name, GGMLType type, int seed0, int seed1, int rows,
                                    int cols) {
        float[] s0 = ParityFixturesDense.valuesAt(seed0, rows * cols);
        float[] s1 = ParityFixturesDense.valuesAt(seed1, rows * cols);
        float[] out = new float[2 * rows * cols];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                long flat = c + cols * ((long) r + rows * 0L);
                out[(int) flat] = s0[r * cols + c];
                flat = c + cols * ((long) r + rows);
                out[(int) flat] = s1[r * cols + c];
            }
        }
        g.tensorFromFloats(name, type, new long[]{cols, rows, 2}, out);
    }

    // ------------------------------------------------------------------ qwen3 moe

    /** The HF tensors of the qwen3-moe checkpoint: the fused expert gate/up and down tensors per layer. */
    public static Path qwen3MoeCheckpoint(Path root, StDType dtype) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json("{'architectures':['Qwen3MoeForCausalLM'],'model_type':'qwen3_moe',"
                        + "'hidden_size':16,'intermediate_size':8,'moe_intermediate_size':8,"
                        + "'num_attention_heads':2,'num_key_value_heads':1,'num_hidden_layers':2,"
                        + "'max_position_embeddings':128,'vocab_size':16,'bos_token_id':0,'eos_token_id':1,"
                        + "'num_experts':2,'num_experts_per_tok':1,'rms_norm_eps':1e-05,"
                        + "'rope_theta':10000.0,'torch_dtype':'" + ParityFixturesDense.torchDtype(dtype) + "'}"));
        b.floats(dtype, "model.embed_tokens.weight", 16, 16);
        for (int l = 0; l < 2; l++) {
            String p = "model.layers." + l + ".";
            b.floats(dtype, p + "input_layernorm.weight", 16);
            b.floats(dtype, p + "self_attn.q_proj.weight", 16, 16);
            b.floats(dtype, p + "self_attn.k_proj.weight", 8, 16);
            b.floats(dtype, p + "self_attn.v_proj.weight", 8, 16);
            b.floats(dtype, p + "self_attn.o_proj.weight", 16, 16);
            b.floats(dtype, p + "mlp.experts.down_proj.weight", 2, 16, 8);
            b.floats(dtype, p + "mlp.experts.gate_up_proj.weight", 2, 16, 16);
            b.floats(dtype, p + "post_attention_layernorm.weight", 16);
        }
        b.floats(dtype, "model.norm.weight", 16);
        b.floats(dtype, "lm_head.weight", 16, 16);
        return b.write(root);
    }

    /** The expected GGUF view of {@link #qwen3MoeCheckpoint}: fused experts split, no Q/K permutation. */
    public static GgufTestWriter qwen3MoeExpected(StDType dtype) {
        GGMLType w = dtype == StDType.BF16 ? GGMLType.BF16 : GGMLType.F32;
        GgufTestWriter g = new GgufTestWriter().architecture("qwen3moe");
        g.str("general.architecture", "qwen3moe");
        g.u32("general.file_type", dtype == StDType.BF16 ? 32 : 0);
        g.str("general.type", "model");
        g.u32("general.quantization_version", 2);
        g.u32("qwen3moe.block_count", 2);
        g.u32("qwen3moe.context_length", 128);
        g.u32("qwen3moe.embedding_length", 16);
        g.u32("qwen3moe.feed_forward_length", 8);
        g.u32("qwen3moe.attention.head_count", 2);
        g.u32("qwen3moe.attention.head_count_kv", 1);
        g.u32("qwen3moe.expert_count", 2);
        g.u32("qwen3moe.expert_feed_forward_length", 8);
        g.f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-5f);
        g.f32("qwen3moe.rope.freq_base", 10000.0f);
        ParityFixturesDense.tokenizerKv(g, ParityFixturesDense.gpt2Tokens(16),
                ParityFixturesDense.gpt2Types(16));

        int seed = 0;
        ParityFixturesDense.planMatrix(g, "token_embd.weight", w, new long[]{16, 16}, seed++);
        for (int l = 0; l < 2; l++) {
            String b = "blk." + l + ".";
            int norm = seed++;
            g.tensorFromFloats(b + "attn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(norm, 16));
            ParityFixturesDense.planMatrix(g, b + "attn_q.weight", w, new long[]{16, 16}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_k.weight", w, new long[]{16, 8}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_v.weight", w, new long[]{16, 8}, seed++);
            ParityFixturesDense.planMatrix(g, b + "attn_output.weight", w, new long[]{16, 16}, seed++);
            int down = seed++;
            int gateUp = seed++;
            splitGateUp(g, down, gateUp, w, b);
            int ffnNorm = seed++;
            g.tensorFromFloats(b + "ffn_norm.weight", GGMLType.F32, new long[]{16},
                    ParityFixturesDense.valuesAt(ffnNorm, 16));
        }
        int finalNorm = seed++;
        g.tensorFromFloats("output_norm.weight", GGMLType.F32, new long[]{16},
                ParityFixturesDense.valuesAt(finalNorm, 16));
        ParityFixturesDense.planMatrix(g, "output.weight", w, new long[]{16, 16}, seed);
        return g;
    }

    /**
     * qwen.py:109-122: the fused {@code down_proj [E, D, F]} maps whole, and the fused {@code gate_up_proj
     * [E, 2F, D]} splits in halves along dim {@code -2}, the gate half first, into {@code ffn_gate_exps} and
     * {@code ffn_up_exps}. Seeds must be the two checkpoint tensors' positions.
     */
    private static void splitGateUp(GgufTestWriter g, int downSeed, int gateUpSeed, GGMLType type, String blk) {
        float[] down = ParityFixturesDense.valuesAt(downSeed, 2 * 16 * 8);
        // [E, D=16, F=8] row-major -> ne [8, 16, 2], the checkpoint's flat order unchanged.
        g.tensorFromFloats(blk + "ffn_down_exps.weight", type, new long[]{8, 16, 2}, down);
        float[] gateUp = ParityFixturesDense.valuesAt(gateUpSeed, 2 * 16 * 16);
        float[] gate = new float[2 * 8 * 16];
        float[] up = new float[2 * 8 * 16];
        for (int e = 0; e < 2; e++) {
            for (int d = 0; d < 16; d++) {
                for (int r = 0; r < 16; r++) {
                    long src = 16L * (r + 16 * e) + d; // fused [E, 2F, D] row-major: the row of dim -2
                    float v = gateUp[(int) src];
                    if (r < 8) {
                        gate[(int) (d + 16L * (r + 8 * e))] = v;   // gate half: ne [D, F, E]
                    } else {
                        up[(int) (d + 16L * (r - 8 + 8 * e))] = v; // up half: second half of dim -2
                    }
                }
            }
        }
        g.tensorFromFloats(blk + "ffn_gate_exps.weight", type, new long[]{16, 8, 2}, gate);
        g.tensorFromFloats(blk + "ffn_up_exps.weight", type, new long[]{16, 8, 2}, up);
    }
}
