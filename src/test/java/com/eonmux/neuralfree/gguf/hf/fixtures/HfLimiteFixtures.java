package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the Limite converter ({@code LimiteModel}, limite.py:13-184): the pinned layout keys, two
 * attention layers with value embeddings, the learned residual scales, the padded attention/value gates and the
 * per-layer mixing scalars the converter folds into the projections. Weightless RMS norms carry no tensors; the
 * embedding table is tied to the output.
 */
public final class HfLimiteFixtures {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int HEAD_DIM = 4;
    private static final int KV = 4;
    private static final int HEADS = 2;
    private static final int FF = 8;
    private static final int VOCAB = 16;
    private static final int VE_DIM = 2;

    private HfLimiteFixtures() {
    }

    /** {@code LimiteForCausalLM}: weightless RMS norms, a tied head and no MUDD taps. */
    public static SyntheticHf.Builder limite() {
        SyntheticHf.Builder b = builder("{'architectures': ['LimiteForCausalLM'],"
                + " 'pos_mode': 'rope', 'qk_norm': 'rms_pre_rope',"
                + " 'rope_style': 'interleaved_pairs_odd_lane_sign_flip', 'rope_per_layer': false,"
                + " 'rms_norm_has_weight': false, 'rms_norm_eps_mode': 'torch_finfo_default',"
                + " 'mlp_type': 'swiglu', 'attn_gate_applied': 'per_head_before_o_proj',"
                + " 've_head_slice': 'first_num_key_value_heads', 've_applied_before_qk_norm': true,"
                + " 'global_nope': true, 'global_window': -1, 'mudd_r_site': 'resid',"
                + " 'tie_word_embeddings': true, 'final_softcap': 0.0,"
                + " 'softcap_logits': {'kind': 'sigmoid', 'a': 1.5, 'b': 0.25, 'c': 2.0},"
                + " 'hidden_size': " + DIM + ", 'intermediate_size': " + FF + ", 'num_hidden_layers': " + LAYERS
                + ", 'num_attention_heads': " + HEADS + ", 'num_key_value_heads': 1,"
                + " 'head_dim': " + HEAD_DIM + ", 'rope_n_pairs': 2, 'rope_base_local': 10000.0,"
                + " 'attention_softmax_scale': 0.5, 'sliding_window': 3, 'global_layers': [0],"
                + " 'vocab_size': " + VOCAB + ", 'max_position_embeddings': 16, 've_dim': " + VE_DIM
                + ", 've_gate_scale': 2.0, 'attn_gate_scale': 0.5}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        b.bf16("model.value_embeds.weight", VOCAB, 1, VE_DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "self_attn.qkv_scale", 1);
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.o_scale", 1);
            b.bf16(p + "post_lambda_attn", 1);
            b.bf16(p + "post_lambda_mlp", 1);
            b.bf16(p + "resid_lambda_attn", 1);
            b.bf16(p + "resid_lambda_mlp", 1);
            b.bf16(p + "self_attn.attn_gate", HEADS, 1);
            b.bf16(p + "self_attn.ve_gate", 1, VE_DIM);
            b.bf16(p + "mlp.gate_proj.weight", FF, DIM);
            b.bf16(p + "mlp.up_proj.weight", FF, DIM);
            b.bf16(p + "mlp.down_proj.weight", DIM, FF);
        }
        return b;
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

    private static SyntheticHf.Builder builder(String singleQuotedConfig) {
        return SyntheticHf.builder().config(SyntheticHf.json(singleQuotedConfig));
    }

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(new Named("LimiteForCausalLM", HfLimiteFixtures::limite));
    }
}
