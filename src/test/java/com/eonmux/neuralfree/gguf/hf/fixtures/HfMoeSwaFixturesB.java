package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the HK-MS part B converter classes: {@code MimoV2Model} (mimo.py:18-403, the FP8
 * qkv dequant and TP regroup, the hybrid attention pattern) and {@code BeamModel} (beam.py:14-141, the layer
 * types and the per-head gate bias). Two layers, a hidden size of 8, two query heads, one key/value head and
 * the checkpoint's real names and config keys.
 */
public final class HfMoeSwaFixturesB {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int EXPERTS = 2;
    private static final int FF = 4;

    private HfMoeSwaFixturesB() {
    }

    /** {@code MiMoV2FlashForCausalLM}: the FP8 qkv with TP-sharded block scales and the expert projections. */
    public static SyntheticHf.Builder mimo() {
        SyntheticHf.Builder b = builder("{'architectures': ['MiMoV2FlashForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'n_routed_experts': 2, 'num_experts_per_tok': 1,"
                + " 'hybrid_layer_pattern': [0, 1], 'swa_num_key_value_heads': 1,"
                + " 'v_head_dim': 4, 'head_dim': 4, 'sliding_window': 4, 'partial_rotary_factor': 0.5,"
                + " 'quantization_config': {'quant_method': 'fp8', 'activation_scheme': 'dynamic'},"
                + " 'norm_topk_prob': true,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0,"
                + " 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            // the TP-4 sharded fused [Q(2 heads x 4)|K(4)|V(4)] rows: 16 x 8 stored as e4m3 1.0 bytes,
            // one scale block per rank
            b.tensor(p + "self_attn.qkv_proj.weight", com.eonmux.neuralfree.io.safetensors.StDType.U8,
                    new long[]{16, 8}, e4m3Ones(16, 8));
            b.tensor(p + "self_attn.qkv_proj.weight_scale_inv",
                    com.eonmux.neuralfree.io.safetensors.StDType.F32, new long[]{4, 1},
                    SyntheticHf.f32(1, 1, 1, 1));
            b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
            for (int x = 0; x < EXPERTS; x++) {
                b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** {@code BeamForCausalLM}: the dense/sparse MLP split, layer types and the per-head gate bias. */
    public static SyntheticHf.Builder beam() {
        SyntheticHf.Builder b = builder("{'architectures': ['BeamForCausalLM'], 'hidden_size': 8,"
                + " 'moe_intermediate_size': 4, 'num_local_experts': 2, 'num_experts_per_tok': 1,"
                + " 'routed_scaling_factor': 1.0, 'norm_topk_prob': true, 'attention_bias': false,"
                + " 'layer_types': ['sliding_attention', 'full_attention'],"
                + " 'mlp_layer_types': ['dense', 'sparse'], 'head_dim': 4,"
                + " 'num_hidden_layers': 2, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
                + " 'sliding_window': 4, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05,"
                + " 'rope_theta': 10000.0, 'vocab_size': 8}");
        b.bf16("model.embed_tokens.weight", 8, 8);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", 4, DIM);
            b.bf16(p + "self_attn.v_proj.weight", 4, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.gate_bias", 2);
            // the beam attention gate and q/k norms (tensor_patterns.tsv:292; the BEAM set)
            b.bf16(p + "self_attn.attn_gate_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_norm.weight", 4);
            b.bf16(p + "self_attn.k_norm.weight", 4);
            // the dual FFN norms (FFN_PRE_NORM/FFN_POST_NORM)
            b.bf16(p + "pre_feedforward_layernorm.weight", DIM);
            b.bf16(p + "post_feedforward_layernorm.weight", DIM);
            if (l == 0) {
                b.bf16(p + "post_attention_layernorm.weight", DIM);
                b.bf16(p + "mlp.gate_proj.weight", 8, DIM);
                b.bf16(p + "mlp.up_proj.weight", 8, DIM);
                b.bf16(p + "mlp.down_proj.weight", DIM, 8);
            } else {
                b.bf16(p + "post_attention_layernorm.weight", DIM);
                b.bf16(p + "mlp.gate.weight", EXPERTS, DIM);
                b.bf16(p + "mlp.gate.e_score_correction.bias", EXPERTS);
                for (int x = 0; x < EXPERTS; x++) {
                    b.bf16(p + "mlp.experts." + x + ".gate_proj.weight", FF, DIM);
                    b.bf16(p + "mlp.experts." + x + ".up_proj.weight", FF, DIM);
                    b.bf16(p + "mlp.experts." + x + ".down_proj.weight", DIM, FF);
                }
            }
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", 8, 8);
        return b;
    }

    /** The e4m3-on-one bytes of an FP8 weight ({@code rows} by {@code k}). */
    private static byte[] e4m3Ones(int rows, int k) {
        byte[] bytes = new byte[rows * k];
        for (int i = 0; i < bytes.length; i++) {
            bytes[i] = (byte) 0x38;
        }
        return bytes;
    }

    /** One end-to-end fixture: its leading HF architecture name and the checkpoint builder. */
    public static final class Named {
        /** The HF architecture of the fixture. */
        public final String hfArch;
        /** The builder of the checkpoint. */
        public final Supplier<SyntheticHf.Builder> fixture;

        Named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
            this.hfArch = hfArch;
            this.fixture = fixture;
        }

        @Override
        public String toString() {
            return hfArch;
        }
    }

    private static Named named(String hfArch, java.util.function.Supplier<SyntheticHf.Builder> fixture) {
        return new Named(hfArch, fixture);
    }

    /** Every fixture of this file, in table order. */
    public static java.util.List<Named> all() {
        return java.util.Arrays.asList(
                named("MiMoV2FlashForCausalLM", HfMoeSwaFixturesB::mimo),
                named("BeamForCausalLM", HfMoeSwaFixturesB::beam));
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }
}
