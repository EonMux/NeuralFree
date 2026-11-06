package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the Mistral Large 4 text converter ({@code MistralLarge4Model}, mistral4.py:15-129): two
 * layers of multi-head latent attention, a rope layer and a NoPE layer, the DSA indexer on the first layer, a routed
 * MoE of two experts with one shared expert on every layer, the learned LeRoPE factors and the SSMax scales. The
 * tokenizer is a tekken file, the format the converter's Mistral vocabulary writer reads.
 */
public final class HfMistralLarge4Fixtures {

    private static final int LAYERS = 2;
    private static final int DIM = 8;
    private static final int HEADS = 2;
    private static final int HEAD_DIM = 4;
    private static final int KV_LORA = 4;
    private static final int ROT = 2;
    private static final int NOPE = 2;
    private static final int V_HEAD = 2;
    private static final int EXPERTS = 2;
    private static final int MOE_FF = 4;
    private static final int INDEXER_HEADS = 2;
    private static final int INDEXER_HEAD_DIM = 2;
    /** The tekken vocabulary of {@link #tekkenJson()}: 20 specials, the 256 bytes and one extra piece. */
    private static final int VOCAB = 277;

    private HfMistralLarge4Fixtures() {
    }

    /** {@code MistralLarge4ForConditionalGeneration}: MLA, DSA, routed experts and shared experts. */
    public static SyntheticHf.Builder mistralLarge4() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['MistralLarge4ForConditionalGeneration'], 'dim': " + DIM + ", 'hidden_dim': 16,"
                + " 'n_layers': " + LAYERS + ", 'n_heads': " + HEADS + ", 'n_kv_heads': 1, 'norm_eps': 1e-05,"
                + " 'vocab_size': " + VOCAB + ", 'max_position_embeddings': 128, 'rope_theta': 10000.0,"
                + " 'qk_rope_head_dim': " + ROT + ", 'qk_nope_head_dim': " + NOPE + ", 'v_head_dim': " + V_HEAD
                + ", 'kv_lora_rank': " + KV_LORA + ", 'nope': 'rope,nope', 'softmax_tanh_gating': 50.0,"
                + " 'moe': {'num_experts': " + EXPERTS + ", 'expert_hidden_dim': " + MOE_FF
                + ", 'num_experts_per_tok': 1, 'num_shared_experts': 1, 'route_every_n': 1,"
                + " 'first_k_dense_replace': 0, 'routed_scale': 1.0},"
                + " 'dsa': {'n_heads': " + INDEXER_HEADS + ", 'head_dim': " + INDEXER_HEAD_DIM + ", 'topk': 1,"
                + " 'sharing': 'full,shared', 'norm_eps': 1e-06, 'rope_head_dim': " + ROT
                + ", 'match_nope': true, 'ssmax_scaling': true},"
                + " 'yarn': {'factor': 4.0, 'beta': 32.0, 'alpha': 1.0,"
                + " 'original_max_position_embeddings': 128, 'apply_scale': false}}"));
        b.file("tekken.json", tekkenJson());
        b.bf16("tok_embeddings.weight", VOCAB, DIM);
        b.bf16("norm.weight", DIM);
        b.bf16("output.weight", VOCAB, DIM);
        b.bf16("lerope.alpha", ROT / 2);
        for (int l = 0; l < LAYERS; l++) {
            String p = "layers." + l + ".attention.";
            b.bf16("layers." + l + ".attention_norm.weight", DIM);
            b.bf16("layers." + l + ".ffn_norm.weight", DIM);
            b.bf16(p + "wq.weight", HEADS * HEAD_DIM, DIM);
            int qkNope = l == 0 ? NOPE : NOPE + ROT;
            b.bf16(p + "wkv_a_with_mqa.weight", KV_LORA + (l == 0 ? ROT : 0), DIM);
            b.bf16(p + "kv_a_norm.weight", KV_LORA);
            b.bf16(p + "wkv_b.weight", HEADS * (qkNope + V_HEAD), KV_LORA);
            b.bf16(p + "wo.weight", HEADS * V_HEAD, DIM);
            b.bf16(p + "ssmax_scaling.s", HEADS);
            b.bf16(p + "ssmax_scaling.b", HEADS);
            if (l == 0) {
                String i = p + "indexer.";
                b.bf16(i + "q_input_norm.weight", DIM);
                b.bf16(i + "q_input_norm.bias", DIM);
                b.bf16(i + "wq_b.weight", INDEXER_HEADS * INDEXER_HEAD_DIM, DIM);
                b.bf16(i + "wk.weight", INDEXER_HEAD_DIM, DIM);
                b.bf16(i + "k_norm.weight", INDEXER_HEAD_DIM);
                b.bf16(i + "k_norm.bias", INDEXER_HEAD_DIM);
                b.bf16(i + "weights_proj.weight", INDEXER_HEADS, DIM);
                b.bf16(i + "ssmax_scaling.s", INDEXER_HEADS);
                b.bf16(i + "ssmax_scaling.b", INDEXER_HEADS);
            }
            String moe = "layers." + l + ".feed_forward.gate.weight";
            b.bf16(moe, EXPERTS, DIM);
            for (int e = 0; e < EXPERTS; e++) {
                b.bf16("layers." + l + ".experts." + e + ".w1.weight", MOE_FF, DIM);
                b.bf16("layers." + l + ".experts." + e + ".w2.weight", DIM, MOE_FF);
                b.bf16("layers." + l + ".experts." + e + ".w3.weight", MOE_FF, DIM);
            }
            b.bf16("layers." + l + ".shared_experts.w1.weight", MOE_FF, DIM);
            b.bf16("layers." + l + ".shared_experts.w2.weight", DIM, MOE_FF);
            b.bf16("layers." + l + ".shared_experts.w3.weight", MOE_FF, DIM);
        }
        return b;
    }

    /** A tekken v7 tokenizer: 20 special tokens, the 256 single bytes and the piece {@code ab}. */
    public static String tekkenJson() {
        StringBuilder vocab = new StringBuilder();
        for (int i = 0; i < 257; i++) {
            byte[] bytes = i < 256 ? new byte[]{(byte) i} : "ab".getBytes(StandardCharsets.UTF_8);
            if (i > 0) {
                vocab.append(',');
            }
            vocab.append("{\"rank\":").append(i).append(",\"token_bytes\":\"")
                    .append(Base64.getEncoder().encodeToString(bytes)).append("\",\"token_str\":null}");
        }
        return "{\"config\":{\"pattern\":\"\\\\p{L}+|\\\\s+\",\"num_vocab_tokens\":257,"
                + "\"default_vocab_size\":277,\"default_num_special_tokens\":20,\"version\":\"v7\"},"
                + "\"vocab\":[" + vocab + "]}";
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

    /** Every fixture of this file, in table order. */
    public static List<Named> all() {
        return Arrays.asList(new Named("MistralLarge4ForConditionalGeneration",
                HfMistralLarge4Fixtures::mistralLarge4));
    }
}
