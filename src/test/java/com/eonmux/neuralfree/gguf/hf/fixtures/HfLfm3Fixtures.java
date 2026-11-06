package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the LFM3 text converter ({@code LFM3Model}, lfm3.py:18-119): two sliding/full attention
 * layers around a convolution layer, a hidden size of 8, two query heads and one key/value head. The config carries
 * the per-layer {@code layer_types}, the learned partial rotary factors, the sliding window and the short-convolution
 * cache; the tensors use the checkpoint's own names ({@code operator_norm}, {@code feed_forward}, {@code conv}).
 */
public final class HfLfm3Fixtures {

    private static final int LAYERS = 3;
    private static final int DIM = 8;
    private static final int HEAD_DIM = 4;
    private static final int KV = 4;
    private static final int HEADS = 2;
    private static final int FF = 16;
    private static final int VOCAB = 16;
    private static final int CONV_CACHE = 4;

    private HfLfm3Fixtures() {
    }

    /** {@code Lfm3ForCausalLM}: full attention, a short convolution, sliding-window attention. */
    public static SyntheticHf.Builder lfm3() {
        SyntheticHf.Builder b = builder("{'architectures': ['Lfm3ForCausalLM'], 'model_type': 'lfm3',"
                + " 'hidden_size': " + DIM + ", 'intermediate_size': " + FF + ", 'num_hidden_layers': " + LAYERS
                + ", 'num_attention_heads': " + HEADS + ", 'num_key_value_heads': 1, 'head_dim': " + HEAD_DIM
                + ", 'vocab_size': " + VOCAB + ", 'max_position_embeddings': 16, 'norm_eps': 1e-05,"
                + " 'layer_types': ['full_attention', 'conv', 'sliding_attention'],"
                + " 'rope_parameters': {'full_attention': {'rope_theta': 10000.0, 'partial_rotary_factor': 0.5},"
                + " 'sliding_attention': {'rope_theta': 1000.0}},"
                + " 'sliding_window': 8, 'conv_L_cache': " + CONV_CACHE + "}");
        b.bf16("model.embed_tokens.weight", VOCAB, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "model.layers." + l + ".";
            b.bf16(p + "operator_norm.weight", DIM);
            b.bf16(p + "ffn_norm.weight", DIM);
            b.bf16(p + "feed_forward.w1.weight", FF, DIM);
            b.bf16(p + "feed_forward.w3.weight", FF, DIM);
            b.bf16(p + "feed_forward.w2.weight", DIM, FF);
            if (l == 1) {
                b.bf16(p + "conv.conv.weight", DIM, 1, CONV_CACHE);
                b.bf16(p + "conv.in_proj.weight", 3 * DIM, DIM);
                b.bf16(p + "conv.out_proj.weight", DIM, DIM);
                continue;
            }
            b.bf16(p + "self_attn.q_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.k_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.v_proj.weight", KV, DIM);
            b.bf16(p + "self_attn.o_proj.weight", DIM, DIM);
            b.bf16(p + "self_attn.q_norm.weight", HEAD_DIM);
            b.bf16(p + "self_attn.k_norm.weight", HEAD_DIM);
            b.bf16(p + "self_attn.gate_proj.weight", HEADS, DIM);
        }
        b.bf16("model.norm.weight", DIM);
        b.bf16("lm_head.weight", VOCAB, DIM);
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
        return Arrays.asList(new Named("Lfm3ForCausalLM", HfLfm3Fixtures::lfm3));
    }
}
