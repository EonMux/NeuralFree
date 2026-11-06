package com.eonmux.neuralfree.gguf.hf.fixtures;

import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Synthetic checkpoints of the DeepSeek-V4.1 text converter ({@code DeepseekV41Model}, deepseek41.py:117-292): two
 * sparse-attention layers with hyper connections, the compressed KV cache on the second layer, the indexer and its
 * candidate selection, the routed and shared experts, and a variant with the two Engram n-gram embedding layers (the
 * E4M3 tables and their E8M0 scales, the key and query gate tables and the per-layer KV projections).
 */
public final class HfDeepseekV41Fixtures {

    private static final int LAYERS = 2;
    private static final int DIM = 64;
    private static final int FF = 128;
    private static final int MOE_FF = 32;
    private static final int HEADS = 4;
    private static final int HEAD_DIM = 32;
    private static final int Q_LORA = 24;
    private static final int O_LORA = 16;
    private static final int EXPERTS = 2;
    private static final int VOCAB = 100;

    /** The Engram rows of the two layers: the bucket moduli of {@code engram_primes(2, 3, 2, 100)}. */
    private static final int[] ENGRAM_ROWS = {420, 508};
    /** {@code engram_head_dim}: two E8M0 blocks per row. */
    private static final int ENGRAM_DIM = 64;

    private HfDeepseekV41Fixtures() {
    }

    /** {@code DeepseekV41ForCausalLM}: the lossless V4.1 text rows, without Engram layers. */
    public static SyntheticHf.Builder deepseekV41() {
        return deepseekV41("", VOCAB);
    }

    /**
     * {@code DeepseekV41ForCausalLM} with Engram layers 0 and 1: the vocabulary is the seven ids of
     * {@link SyntheticHf#GPT2_TOKENIZER} so that the compressed token map agrees with the configuration.
     */
    public static SyntheticHf.Builder deepseekV41Engram() {
        SyntheticHf.Builder b = deepseekV41(", 'engram_layer_ids': [0, 1], 'engram_max_ngram_size': 3,"
                + " 'engram_n_heads': 2, 'engram_vocab_size': 100, 'engram_num_embeddings': [420, 508],"
                + " 'engram_head_dim': " + ENGRAM_DIM + ", 'engram_pad_token_id': 7,"
                + " 'engram_compressed_vocab_size': 7", 7);
        for (int l = 0; l < LAYERS; l++) {
            int rows = ENGRAM_ROWS[l];
            byte[] codes = new byte[rows * ENGRAM_DIM];
            byte[] scales = new byte[rows * (ENGRAM_DIM / 32)];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < ENGRAM_DIM; c++) {
                    codes[r * ENGRAM_DIM + c] = (byte) ((r * 7 + c * 13) % 127);
                }
                scales[r * (ENGRAM_DIM / 32)] = (byte) (3 + r % 244);
                scales[r * (ENGRAM_DIM / 32) + 1] = (byte) (3 + (r + 71) % 244);
            }
            b.tensor("layers." + l + ".engram.embed.weight", StDType.F8_E4M3,
                    new long[]{rows, ENGRAM_DIM}, codes);
            b.tensor("layers." + l + ".engram.embed.scale", StDType.U8,
                    new long[]{rows, ENGRAM_DIM / 32}, scales);
        }
        b.bf16("layers.0.engram.wkv.weight", DIM, DIM);
        b.bf16("layers.0.engram.q_weight", DIM, DIM);
        b.bf16("layers.0.engram.k_weight", DIM, DIM);
        return b;
    }

    private static SyntheticHf.Builder deepseekV41(String extra, int vocab) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['DeepseekV41ForCausalLM'], 'model_type': 'deepseek_v41', 'hidden_size': " + DIM + ","
                + " 'intermediate_size': " + FF + ", 'moe_intermediate_size': " + MOE_FF + ","
                + " 'num_attention_heads': " + HEADS + ", 'num_key_value_heads': 1, 'head_dim': " + HEAD_DIM + ","
                + " 'num_hidden_layers': " + LAYERS + ", 'max_position_embeddings': 1024,"
                + " 'rms_norm_eps': 1e-06, 'vocab_size': " + vocab + ", 'rope_theta': 10000.0,"
                + " 'qk_rope_head_dim': 16, 'q_lora_rank': " + Q_LORA + ", 'sliding_window': 128,"
                + " 'n_routed_experts': " + EXPERTS + ", 'num_experts_per_tok': 1, 'n_shared_experts': 1,"
                + " 'routed_scaling_factor': 1.5, 'norm_topk_prob': true, 'swiglu_limit': 10.0,"
                + " 'index_n_heads': 2, 'index_head_dim': 16, 'index_topk': 8, 'o_groups': 2,"
                + " 'o_lora_rank': " + O_LORA + ", 'compress_ratios': [0, 4], 'compress_rope_theta': 160000.0,"
                + " 'hc_mult': 4, 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-06, 'scoring_func': 'sqrtsoftplus',"
                + " 'kv_source_layer_ids': [0, 1], 'index_source_layer_ids': [1, 1],"
                + " 'candidate_source_layer_id': 0, 'candidate_topk_blocks': 4, 'candidate_block_size': 64"
                + extra + "}"));
        b.bf16("embed.weight", vocab, DIM);
        b.bf16("norm.weight", DIM);
        b.bf16("head.weight", vocab, DIM);
        for (int l = 0; l < LAYERS; l++) {
            String p = "layers." + l + ".";
            b.bf16(p + "attn_norm.weight", DIM);
            b.bf16(p + "ffn_norm.weight", DIM);
            b.bf16(p + "attn.wq_a.weight", Q_LORA, DIM);
            b.bf16(p + "attn.wq_b.weight", HEADS * HEAD_DIM, Q_LORA);
            b.bf16(p + "attn.q_norm.weight", Q_LORA);
            b.bf16(p + "attn.wkv.weight", DIM, DIM);
            b.bf16(p + "attn.kv_norm.weight", DIM);
            b.bf16(p + "attn.wo_a.weight", O_LORA, DIM);
            b.bf16(p + "attn.wo_b.weight", DIM, O_LORA);
            b.bf16(p + "attn.attn_sink", HEADS);
            b.bf16(p + "hc_attn_fn", 4, 4);
            b.bf16(p + "hc_attn_base", 4);
            b.bf16(p + "hc_attn_scale", 1);
            b.bf16(p + "hc_ffn_fn", 4, 4);
            b.bf16(p + "hc_ffn_base", 4);
            b.bf16(p + "hc_ffn_scale", 1);
            b.bf16(p + "ffn.gate.weight", EXPERTS, DIM);
            b.bf16(p + "ffn.gate.bias", EXPERTS);
            b.bf16(p + "ffn.shared_experts.w1.weight", MOE_FF, DIM);
            b.bf16(p + "ffn.shared_experts.w2.weight", DIM, MOE_FF);
            b.bf16(p + "ffn.shared_experts.w3.weight", MOE_FF, DIM);
            for (int e = 0; e < EXPERTS; e++) {
                b.bf16(p + "ffn.experts." + e + ".w1.weight", MOE_FF, DIM);
                b.bf16(p + "ffn.experts." + e + ".w2.weight", DIM, MOE_FF);
                b.bf16(p + "ffn.experts." + e + ".w3.weight", MOE_FF, DIM);
            }
            if (l == 1) {
                b.bf16(p + "attn.compressor.wkv.weight", DIM, DIM);
                b.bf16(p + "attn.compressor.wgate.weight", DIM, DIM);
                b.bf16(p + "attn.compressor.norm.weight", DIM);
                b.bf16(p + "attn.indexer.wq_b.weight", 2 * 16, DIM);
                b.bf16(p + "attn.indexer.wk.weight", 16, DIM);
                b.bf16(p + "attn.indexer.weights_proj.weight", 2, DIM);
            }
        }
        return b;
    }

    /** Geometry of {@link #deepseekV41EngramRunnable()}. */
    public static final int RUNNABLE_DIM = 128;
    /** Engram table rows per layer of {@link #deepseekV41EngramRunnable()}. */
    public static final int[] RUNNABLE_ENGRAM_ROWS = {420, 508};
    /** {@code engram_head_dim}: one E8M0 block per row. */
    public static final int RUNNABLE_ENGRAM_DIM = 32;
    /** The hyper-connection count of {@link #deepseekV41EngramRunnable()} (the runtime needs more than one). */
    public static final int RUNNABLE_HC = 2;
    /** {@code hc_eps} of {@link #deepseekV41EngramRunnable()}. */
    public static final float RUNNABLE_HC_EPS = 1e-6f;
    /** {@code hc_sinkhorn_iters} of {@link #deepseekV41EngramRunnable()}. */
    public static final int RUNNABLE_HC_ITERS = 20;
    /** {@code hc_*_base} value that saturates {@code sigmoid} to 1 (selects a stream or a comb entry). */
    public static final float HC_HIGH = 100f;
    /** {@code hc_*_base} value that saturates {@code sigmoid} to 0. */
    public static final float HC_LOW = -100f;
    /** Vocabulary of {@link #deepseekV41EngramRunnable()} (the seven synthetic GPT-2 ids). */
    public static final int RUNNABLE_VOCAB = 7;

    /**
     * A DeepSeek-V4.1 checkpoint that the engine can load and run with its Engram layers: two hyper-connection
     * streams ({@code hc_mult = 2}), both layers selecting compressed rows (ratio 2, layer 0 the shared source),
     * every attention and FFN output projection zero and the hyper-connection coefficients pinned to the identity
     * ({@code hc_*_fn} and {@code hc_*_scale} zero, {@code hc_*_base} selecting stream 0 for the collapse and the
     * identity for the Sinkhorn comb). The residual streams then carry the token embedding plus each Engram layer's
     * contribution through the fixed comb, which is what the round-trip test computes independently.
     *
     * <p>The Engram layout is the {@code engram_primes(2, 3, 2, 100)} one ({@code [420, 508]} rows of 32 values,
     * offsets {@code [0, 101, 204, 311, 420, 533, 660, 791]}); the compressed token map of the synthetic tokenizer
     * is the identity over the seven ids.</p>
     */
    public static SyntheticHf.Builder deepseekV41EngramRunnable() {
        int dim = RUNNABLE_DIM;
        int layers = 2;
        int heads = 4;
        int headDim = 32;
        int qLora = 24;
        int oLora = 16;
        int groups = 2;
        int experts = 2;
        int moeFf = 32;
        int vocab = RUNNABLE_VOCAB;
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['DeepseekV41ForCausalLM'], 'model_type': 'deepseek_v41', 'hidden_size': " + dim + ","
                + " 'intermediate_size': 128, 'moe_intermediate_size': " + moeFf + ","
                + " 'num_attention_heads': " + heads + ", 'num_key_value_heads': 1, 'head_dim': " + headDim + ","
                + " 'num_hidden_layers': " + layers + ", 'max_position_embeddings': 1024,"
                + " 'rms_norm_eps': 1e-06, 'vocab_size': " + vocab + ", 'rope_theta': 10000.0,"
                + " 'qk_rope_head_dim': 16, 'q_lora_rank': " + qLora + ", 'sliding_window': 128,"
                + " 'n_routed_experts': " + experts + ", 'num_experts_per_tok': 1, 'n_shared_experts': 1,"
                + " 'routed_scaling_factor': 1.5, 'norm_topk_prob': true, 'swiglu_limit': 10.0,"
                + " 'index_n_heads': 2, 'index_head_dim': 16, 'index_topk': 8, 'o_groups': " + groups + ","
                + " 'o_lora_rank': " + oLora + ", 'compress_ratios': [2, 2], 'compress_rope_theta': 160000.0,"
                + " 'hc_mult': 2, 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-06, 'scoring_func': 'sqrtsoftplus',"
                + " 'kv_source_layer_ids': [0], 'index_source_layer_ids': [0],"
                + " 'engram_layer_ids': [0, 1], 'engram_max_ngram_size': 3, 'engram_n_heads': 2,"
                + " 'engram_vocab_size': 100, 'engram_num_embeddings': [420, 508],"
                + " 'engram_head_dim': " + RUNNABLE_ENGRAM_DIM + ", 'engram_pad_token_id': 0,"
                + " 'engram_compressed_vocab_size': 7}"));
        b.bf16("embed.weight", vocab, dim);
        b.bf16("norm.weight", dim);
        b.bf16("head.weight", vocab, dim);
        for (int l = 0; l < layers; l++) {
            String p = "layers." + l + ".";
            b.bf16(p + "attn_norm.weight", dim);
            b.bf16(p + "ffn_norm.weight", dim);
            b.bf16(p + "attn.wq_a.weight", qLora, dim);
            b.bf16(p + "attn.wq_b.weight", dim, qLora);
            b.bf16(p + "attn.q_norm.weight", qLora);
            b.bf16(p + "attn.wkv.weight", headDim, dim);
            b.bf16(p + "attn.kv_norm.weight", headDim);
            b.bf16(p + "attn.wo_a.weight", (long) oLora * groups, dim / groups);
            zeros(b, p + "attn.wo_b.weight", dim, (long) oLora * groups);
            b.bf16(p + "attn.attn_sink", heads);
            zeros(b, p + "hc_attn_fn", 8, 2L * dim);
            b.tensor(p + "hc_attn_base", StDType.BF16, new long[]{8},
                    SyntheticHf.bf16(HC_HIGH, HC_LOW, 0f, 0f, HC_HIGH, HC_LOW, HC_LOW, HC_HIGH));
            zeros(b, p + "hc_attn_scale", 3);
            zeros(b, p + "hc_ffn_fn", 8, 2L * dim);
            b.tensor(p + "hc_ffn_base", StDType.BF16, new long[]{8},
                    SyntheticHf.bf16(HC_HIGH, HC_LOW, 0f, 0f, HC_HIGH, HC_LOW, HC_LOW, HC_HIGH));
            zeros(b, p + "hc_ffn_scale", 3);
            b.bf16(p + "ffn.gate.weight", experts, dim);
            b.bf16(p + "ffn.gate.bias", experts);
            b.bf16(p + "ffn.shared_experts.w1.weight", moeFf, dim);
            zeros(b, p + "ffn.shared_experts.w2.weight", dim, moeFf);
            b.bf16(p + "ffn.shared_experts.w3.weight", moeFf, dim);
            for (int e = 0; e < experts; e++) {
                b.bf16(p + "ffn.experts." + e + ".w1.weight", moeFf, dim);
                zeros(b, p + "ffn.experts." + e + ".w2.weight", dim, moeFf);
                b.bf16(p + "ffn.experts." + e + ".w3.weight", moeFf, dim);
            }
            int rows = RUNNABLE_ENGRAM_ROWS[l];
            int width = RUNNABLE_ENGRAM_DIM;
            byte[] codes = new byte[rows * width];
            byte[] scales = new byte[rows * (width / 32)];
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < width; c++) {
                    codes[r * width + c] = (byte) ((r * 7 + c * 13) % 127);
                }
                for (int c = 0; c < width / 32; c++) {
                    scales[r * (width / 32) + c] = (byte) (120 + (r + 37 * c) % 9);
                }
            }
            b.tensor(p + "engram.embed.weight", StDType.F8_E4M3, new long[]{rows, width}, codes);
            b.tensor(p + "engram.embed.scale", StDType.U8, new long[]{rows, width / 32}, scales);
            b.bf16(p + "engram.wkv.weight", (long) 3 * dim, dim);
            b.bf16(p + "engram.q_weight", 2, dim);
            b.bf16(p + "engram.k_weight", 2, dim);
        }
        String p = "layers.0.";
        b.bf16(p + "attn.compressor.wkv.weight", headDim, dim);
        b.bf16(p + "attn.compressor.wgate.weight", headDim, dim);
        b.bf16(p + "attn.compressor.norm.weight", headDim);
        b.bf16(p + "attn.indexer.wq_b.weight", 32, qLora);
        b.bf16(p + "attn.indexer.wk.weight", 16, headDim);
        b.bf16(p + "attn.indexer.k_norm.weight", 16);
        b.bf16(p + "attn.indexer.weights_proj.weight", 2, dim);
        return b;
    }

    /** A BF16 tensor of zeros. */
    private static void zeros(SyntheticHf.Builder b, String name, long... shape) {
        long count = 1;
        for (long n : shape) {
            count *= n;
        }
        b.tensor(name, StDType.BF16, shape, new byte[Math.toIntExact(count * 2)]);
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
        return Arrays.asList(new Named("DeepseekV41ForCausalLM", HfDeepseekV41Fixtures::deepseekV41));
    }
}
