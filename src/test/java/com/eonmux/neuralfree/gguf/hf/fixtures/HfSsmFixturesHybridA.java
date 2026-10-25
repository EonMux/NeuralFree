package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the first half of the SSM hybrid converters ({@code FalconH1Model}, {@code JambaModel},
 * {@code Plamo2Model}, {@code GraniteHybridModel}, {@code NemotronHModel} and {@code NemotronHPuzzleModel}):
 * two layers, hidden size 8, and the exact tensor names and configuration keys of the converters and the generated
 * pattern tables. Every backbone layer uses the {@code mixer.*} names the tables carry for the {@code backbone.*}
 * rows (the attention projections are {@code mixer.q/k/v/o_proj}, the Mamba core is {@code mixer.in_proj/conv1d/
 * A_log/D/norm/out_proj}), so each registered entry builds and claims a complete model of 16 padded vocabulary rows.
 */
public final class HfSsmFixturesHybridA {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int INNER = 16;
    private static final int STATE = 8;
    private static final int CONV = 4;
    private static final int GROUPS = 2;
    private static final int FFN = 16;
    private static final int EXPERTS = 2;
    private static final int EXPERT_FFN = 8;
    private static final int VOCAB = 16;

    private HfSsmFixturesHybridA() {
    }

    /**
     * {@code FalconH1ForCausalLM}: a Mamba-2 trunk with a parallel attention branch and a dense FFN on every block,
     * the plain {@code model.} checkpoint form (falcon_h1.py:16-119).
     */
    public static SyntheticHf.Builder falconH1() {
        SyntheticHf.Builder b = builder("{'architectures': ['FalconH1ForCausalLM'], 'hidden_size': " + DIM
                + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': " + VOCAB
                + ", 'mamba_d_ssm': " + INNER + ", 'mamba_d_head': 8, 'd_head': 8, 'n_groups': " + GROUPS
                + ", 'state_size': " + STATE + ", 'conv_kernel': " + CONV + ", 'hidden_act': 'silu'"
                + ", 'intermediate_size': " + FFN + ", 'num_attention_heads': 2, 'num_key_value_heads': 2"
                + ", 'max_position_embeddings': 16, 'rope_parameters': {'rope_theta': 10000.0}}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", DIM, VOCAB);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "input_layernorm.weight", DIM);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.v_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "post_attention_layernorm.weight", DIM);
            b.bf16(p + "mlp.gate_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.up_proj.weight", FFN, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FFN);
            b.bf16(p + "mamba.in_proj.weight", 2 * INNER + 2 * GROUPS * STATE, DIM);
            b.bf16(p + "mamba.conv1d.weight", INNER, 1, CONV);
            b.bf16(p + "mamba.conv1d.bias", INNER);
            b.bf16(p + "mamba.dt_proj.weight", INNER, 8);
            b.bf16(p + "mamba.dt_proj.bias", INNER);
            b.bf16(p + "mamba.A_log", INNER / GROUPS, STATE);
            b.bf16(p + "mamba.D", INNER);
            b.bf16(p + "mamba.norm.weight", GROUPS * (INNER / GROUPS));
            b.bf16(p + "mamba.out_proj.weight", DIM, INNER);
        }
        return b;
    }

    /**
     * {@code JambaForCausalLM}: alternating attention-MoE and Mamba layers over the plain {@code model.} form
     * (jamba.py:15-120); the dense attention layer's experts are the single-expert routed path.
     */
    public static SyntheticHf.Builder jamba() {
        SyntheticHf.Builder b = builder("{'architectures': ['JambaForCausalLM'], 'hidden_size': " + DIM
                + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': " + VOCAB
                + ", 'mamba_expand': 2, 'mamba_d_state': " + STATE + ", 'mamba_d_conv': " + CONV
                + ", 'attn_layer_offset': 0, 'attn_layer_period': 2, 'expert_layer_offset': 0"
                + ", 'expert_layer_period': 2, 'num_key_value_heads': 2, 'num_attention_heads': 2"
                + ", 'intermediate_size': " + FFN + ", 'max_position_embeddings': 16"
                + ", 'num_local_experts': " + EXPERTS + ", 'num_experts_per_tok': 1"
                + ", 'rms_norm_eps': 1e-06, 'rope_parameters': {'rope_theta': 10000.0}}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", DIM, VOCAB);
        // layer 0: attention + routed experts
        String a = "model.layers.0.";
        b.bf16(a + "input_layernorm.weight", DIM);
        b.bf16(a + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.k_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.v_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(a + "post_attention_layernorm.weight", DIM);
        b.bf16(a + "mlp.gate.weight", EXPERTS, DIM);
        for (int e = 0; e < EXPERTS; e++) {
            b.bf16(a + "mlp.experts." + e + ".gate_proj.weight", FFN, DIM);
            b.bf16(a + "mlp.experts." + e + ".up_proj.weight", FFN, DIM);
            b.bf16(a + "mlp.experts." + e + ".down_proj.weight", DIM, FFN);
        }
        // layer 1: Mamba-1
        String m = "model.layers.1.";
        b.bf16(m + "input_layernorm.weight", DIM);
        b.bf16(m + "mamba.in_proj.weight", 2 * INNER, DIM);
        b.bf16(m + "mamba.conv1d.weight", INNER, 1, CONV);
        b.bf16(m + "mamba.conv1d.bias", INNER);
        b.bf16(m + "mamba.x_proj.weight", 1 + 2 * STATE, INNER);
        b.bf16(m + "mamba.dt_proj.weight", INNER, 1);
        b.bf16(m + "mamba.dt_proj.bias", INNER);
        b.bf16(m + "mamba.A_log", INNER, STATE);
        b.bf16(m + "mamba.D", INNER);
        b.bf16(m + "mamba.out_proj.weight", DIM, INNER);
        return b;
    }

    /**
     * {@code Plamo2ForCausalLM}: a Mamba layer (the {@code linear_attn} core) and an attention layer over the plain
     * {@code model.} form (plamo.py:63-148), the PLaMo-2 per-head state sizes.
     */
    public static SyntheticHf.Builder plamo2() {
        SyntheticHf.Builder b = builder("{'architectures': ['Plamo2ForCausalLM'], 'vocab_size': " + VOCAB
                + ", 'hidden_size': " + DIM + ", 'num_hidden_layers': " + LAYERS
                + ", 'mamba_step': 2, 'mamba_enabled': true, 'num_key_value_heads': 2, 'num_attention_heads': 2"
                + ", 'hidden_size_per_head': 4, 'max_position_embeddings': 16, 'mamba_d_state': " + STATE
                + ", 'mamba_d_conv': " + CONV + ", 'mamba_num_heads': 2, 'intermediate_size': " + FFN
                + ", 'rms_norm_eps': 1e-05, 'rope_parameters': {'rope_theta': 10000.0}}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", DIM, VOCAB);
        // layer 0: Mamba (linear_attn)
        String m = "model.layers.0.";
        b.bf16(m + "input_layernorm.weight", DIM);
        b.bf16(m + "linear_attn.in_proj_qkvz.weight", 2 * INNER + 2 * GROUPS * STATE, DIM);
        b.bf16(m + "linear_attn.conv1d.weight", INNER, 1, CONV);
        b.bf16(m + "linear_attn.conv1d.bias", INNER);
        b.bf16(m + "linear_attn.dt_proj.weight", INNER, 8);
        b.bf16(m + "linear_attn.dt_proj.bias", INNER);
        b.bf16(m + "linear_attn.dt_norm.weight", 8);
        b.bf16(m + "linear_attn.A_log", INNER / GROUPS, STATE);
        b.bf16(m + "linear_attn.D", INNER);
        b.bf16(m + "linear_attn.norm.weight", GROUPS * (INNER / GROUPS));
        b.bf16(m + "linear_attn.q_norm.weight", GROUPS * (INNER / GROUPS));
        b.bf16(m + "linear_attn.k_norm.weight", GROUPS * (INNER / GROUPS));
        b.bf16(m + "linear_attn.out_proj.weight", DIM, INNER);
        // layer 1: attention
        String a = "model.layers.1.";
        b.bf16(a + "input_layernorm.weight", DIM);
        b.bf16(a + "self_attn.q_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.k_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.v_proj.weight", DIM, DIM);
        b.bf16(a + "self_attn.o_proj.weight", DIM, DIM);
        b.bf16(a + "post_attention_layernorm.weight", DIM);
        b.bf16(a + "mlp.gate_proj.weight", FFN, DIM);
        b.bf16(a + "mlp.up_proj.weight", FFN, DIM);
        b.bf16(a + "mlp.down_proj.weight", DIM, FFN);
        return b;
    }

    /**
     * {@code BambaForCausalLM}: the Granite hybrid over the {@code backbone.} form (granite.py:436-576) — one Mamba
     * layer and one attention layer, both with the fused MoE feed-forward (the {@code mixer.} gate, latent and
     * shared-expert projections).
     */
    public static SyntheticHf.Builder graniteHybrid() {
        SyntheticHf.Builder b = builder("{'architectures': ['BambaForCausalLM'], 'mamba_head_dim': " + DIM
                + ", 'num_hidden_layers': " + LAYERS + ", 'vocab_size': " + VOCAB
                + ", 'num_key_value_heads': 2, 'n_head_kv': 2, 'expand': 2, 'n_groups': " + GROUPS
                + ", 'head_dim': 8, 'conv_kernel': " + CONV + ", 'state_size': " + STATE + ", 'n_heads': 4"
                + ", 'num_attention_heads': 2, 'intermediate_size': " + FFN
                + ", 'num_local_experts': " + EXPERTS + ", 'num_experts_per_tok': 1, 'd_head': 8"
                + ", 'max_position_embeddings': 16, 'hidden_act': 'silu', 'rms_norm_eps': 1e-05"
                + ", 'rope_theta': 10000.0, 'layer_types': ['mamba', 'attention']}");
        b.bf16("backbone.embeddings.weight", VOCAB, DIM);
        b.bf16("backbone.final_layer_norm.weight", DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "backbone.layers." + l + ".";
            b.bf16(p + "norm.weight", DIM);
            if (l == 0) {
                mambaMixer(b, p);
            } else {
                b.bf16(p + "mixer.q_proj.weight", DIM, DIM);
                b.bf16(p + "mixer.k_proj.weight", DIM, DIM);
                b.bf16(p + "mixer.v_proj.weight", DIM, DIM);
                b.bf16(p + "mixer.o_proj.weight", DIM, DIM);
            }
            moeMixer(b, p);
        }
        return b;
    }

    /**
     * {@code NemotronHForCausalLM}: the dense Nemotron-H over the {@code backbone.} form (nemotron.py:405-731) — a
     * Mamba layer and an attention layer, the feed-forward only on the attention block.
     */
    public static SyntheticHf.Builder nemotronH() {
        SyntheticHf.Builder b = builder("{'architectures': ['NemotronHForCausalLM'], 'mamba_head_dim': " + DIM
                + ", 'vocab_size': " + VOCAB + ", 'num_key_value_heads': 2, 'n_head_kv': 2, 'expand': 2"
                + ", 'n_groups': " + GROUPS + ", 'conv_kernel': " + CONV + ", 'state_size': " + STATE
                + ", 'n_heads': 4, 'num_attention_heads': 2, 'intermediate_size': " + FFN + ", 'd_head': 8"
                + ", 'head_dim': 8, 'max_position_embeddings': 16, 'hidden_act': 'silu'"
                + ", 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0"
                + ", 'layers_block_type': ['mamba', 'attention']}");
        b.bf16("backbone.embeddings.weight", VOCAB, DIM);
        b.bf16("backbone.final_layer_norm.weight", DIM);
        // layer 0: Mamba
        String m = "backbone.layers.0.";
        b.bf16(m + "norm.weight", DIM);
        mambaMixer(b, m);
        // layer 1: attention + dense FFN
        String a = "backbone.layers.1.";
        b.bf16(a + "norm.weight", DIM);
        b.bf16(a + "mixer.q_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.k_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.v_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.o_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.up_proj.weight", FFN, DIM);
        b.bf16(a + "mixer.down_proj.weight", DIM, FFN);
        return b;
    }

    /**
     * {@code NemotronHForCausalLM} MoE build: the same {@code backbone.} layout with the routed expert feed-forward
     * (the {@code mixer.experts.*} down/up parts the Nemotron-H path stacks) on both blocks.
     */
    public static SyntheticHf.Builder nemotronHmoe() {
        SyntheticHf.Builder b = nemotronH();
        for (int l = 0; l < LAYERS; l++) {
            String p = "backbone.layers." + l + ".";
            b.bf16(p + "mixer.gate.weight", EXPERTS, DIM);
            b.bf16(p + "mixer.gate.e_score_correction.bias", EXPERTS);
            b.bf16(p + "mixer.fc1_latent_proj.weight", 4, DIM);
            b.bf16(p + "mixer.fc2_latent_proj.weight", 4, DIM);
            b.bf16(p + "mixer.shared_experts.up_proj.weight", EXPERT_FFN, 4);
            b.bf16(p + "mixer.shared_experts.down_proj.weight", 4, EXPERT_FFN);
            for (int e = 0; e < EXPERTS; e++) {
                b.bf16(p + "mixer.experts." + e + ".up_proj.weight", EXPERT_FFN, 4);
                b.bf16(p + "mixer.experts." + e + ".down_proj.weight", 4, EXPERT_FFN);
            }
        }
        return b;
    }

    /**
     * {@code NemotronHPuzzleForCausalLM}: the per-block configured MoE build (nemotron.py:733-928) over the
     * checkpoint's {@code model.} trunk names the converter renames to {@code backbone.}, one Mamba, one attention and
     * one feed-forward-only block.
     */
    public static SyntheticHf.Builder puzzle() {
        SyntheticHf.Builder b = builder("{'architectures': ['NemotronHPuzzleForCausalLM'], 'mamba_head_dim': " + DIM
                + ", 'vocab_size': " + VOCAB + ", 'num_key_value_heads': 2, 'n_head_kv': 2, 'expand': 2"
                + ", 'n_groups': " + GROUPS + ", 'conv_kernel': " + CONV + ", 'state_size': " + STATE
                + ", 'n_heads': 4, 'num_attention_heads': 2, 'd_head': 8, 'head_dim': 8"
                + ", 'moe_shared_expert_intermediate_size': " + EXPERT_FFN + ", 'n_routed_experts': " + EXPERTS
                + ", 'n_shared_experts': 1, 'norm_topk_prob': false, 'routed_scaling_factor': 1.0, 'n_group': 1"
                + ", 'moe_latent_size': 4, 'max_position_embeddings': 16, 'hidden_act': 'silu'"
                + ", 'rms_norm_eps': 1e-05, 'rope_theta': 10000.0"
                + ", 'block_configs': [{'block_type': 'mamba'}, {'block_type': 'attention'},"
                + " {'block_type': 'moe', 'moe_intermediate_size': " + FFN + ", 'num_experts_per_tok': 1}]}");
        b.bf16("model.embeddings.weight", VOCAB, DIM);
        b.bf16("model.final_layer_norm.weight", DIM);
        // block 0: Mamba
        String m = "model.layers.0.";
        b.bf16(m + "norm.weight", DIM);
        mambaMixer(b, m);
        moeMixer(b, m);
        // block 1: attention
        String a = "model.layers.1.";
        b.bf16(a + "norm.weight", DIM);
        b.bf16(a + "mixer.q_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.k_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.v_proj.weight", DIM, DIM);
        b.bf16(a + "mixer.o_proj.weight", DIM, DIM);
        moeMixer(b, a);
        // block 2: feed-forward only
        String f = "model.layers.2.";
        b.bf16(f + "norm.weight", DIM);
        moeMixer(b, f);
        return b;
    }

    /** The Mamba-2 core of a {@code backbone.} block under the {@code mixer.} names of the tables. */
    private static void mambaMixer(SyntheticHf.Builder b, String p) {
        b.bf16(p + "mixer.in_proj.weight", 2 * INNER + 2 * GROUPS * STATE, DIM);
        b.bf16(p + "mixer.conv1d.weight", INNER, 1, CONV);
        b.bf16(p + "mixer.conv1d.bias", INNER);
        b.bf16(p + "mixer.dt_proj.weight", INNER, 4);
        b.bf16(p + "mixer.dt_proj.bias", INNER);
        b.bf16(p + "mixer.A_log", INNER / GROUPS, STATE);
        b.bf16(p + "mixer.D", INNER);
        b.bf16(p + "mixer.norm.weight", GROUPS * (INNER / GROUPS));
        b.bf16(p + "mixer.out_proj.weight", DIM, INNER);
    }

    /** The fused MoE feed-forward of a {@code backbone.} block under the {@code mixer.} names of the tables. */
    private static void moeMixer(SyntheticHf.Builder b, String p) {
        b.bf16(p + "mixer.gate.weight", EXPERTS, DIM);
        b.bf16(p + "mixer.fc1_latent_proj.weight", 4, DIM);
        b.bf16(p + "mixer.fc2_latent_proj.weight", 4, DIM);
        b.bf16(p + "mixer.up_proj.weight", FFN, 4);
        b.bf16(p + "mixer.down_proj.weight", 4, FFN);
        b.bf16(p + "mixer.shared_experts.up_proj.weight", EXPERT_FFN, 4);
        b.bf16(p + "mixer.shared_experts.down_proj.weight", 4, EXPERT_FFN);
        for (int e = 0; e < EXPERTS; e++) {
            b.bf16(p + "mixer.experts." + e + ".up_proj.weight", EXPERT_FFN, 4);
            b.bf16(p + "mixer.experts." + e + ".down_proj.weight", 4, EXPERT_FFN);
        }
    }

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }

    /**
     * One end-to-end fixture: its leading HF architecture name and the checkpoint builder.
     */
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

    private static Named named(String hfArch, Supplier<SyntheticHf.Builder> fixture) {
        return new Named(hfArch, fixture);
    }

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(
                named("FalconH1ForCausalLM", HfSsmFixturesHybridA::falconH1),
                named("JambaForCausalLM", HfSsmFixturesHybridA::jamba),
                named("Plamo2ForCausalLM", HfSsmFixturesHybridA::plamo2),
                named("BambaForCausalLM", HfSsmFixturesHybridA::graniteHybrid),
                named("NemotronHForCausalLM", HfSsmFixturesHybridA::nemotronH),
                named("NemotronHForCausalLM-MoE", HfSsmFixturesHybridA::nemotronHmoe),
                named("NemotronHPuzzleForCausalLM", HfSsmFixturesHybridA::puzzle));
    }
}
