package com.eonmux.neuralfree.gguf.testutil;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.blk;
import static com.eonmux.neuralfree.gguf.testutil.SyntheticText.requirePositive;

/**
 * Synthetic GGUF writers of families whose attention runs a routed or grouped product of its own, sized like
 * {@link SyntheticModels}: k2-horizon, whose MoE layers route each token to value experts (MoVA,
 * {@code attn_v_gate} + {@code attn_v_exps}), and deepseek4, whose attention output runs one low-rank projection
 * per head group ({@code attn_output_a}). Byte-level BPE tokenizer.
 */
public final class SyntheticRoutedModels {

    /** Value experts of a k2-horizon MoVA layer and how many each token uses. */
    public static final int VALUE_EXPERTS = 4;
    public static final int VALUE_EXPERTS_USED = 2;

    /** Head groups of the deepseek4 grouped output projection and the rank of each group's projection. */
    public static final int DSV4_OUTPUT_GROUPS = 2;
    public static final int DSV4_OUTPUT_LORA = 16;
    /** deepseek4 query rank, hyper-connection streams and lightning-indexer heads and width. */
    private static final int DSV4_Q_LORA = 32;
    private static final int DSV4_HC = 4;
    private static final int DSV4_INDEXER_HEADS = 2;
    private static final int DSV4_INDEXER_DIM = 64;
    /** The compressed-attention ratio of the deepseek4 layer with the lightning indexer (CSA). */
    private static final int DSV4_CSA_RATIO = 4;
    /** llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, the scoring deepseek4 requires. */
    private static final int GATING_SQRT_SOFTPLUS = 4;
    /** DeepSeek-V4.1 Engram: 2-gram hashes of 2 heads per n-gram over 32-value table rows. */
    private static final int DSV41_NGRAM = 2;
    private static final int DSV41_HEADS_PER_NGRAM = 2;
    private static final int DSV41_HEAD_DIM = 32;
    private static final int DSV41_TABLE_ROWS = 64;
    private static final int DSV41_TOP_K = 4;
    /** Compress ratios of {@link #deepseek41()}: two selected layers followed by two dense layers. */
    private static final int[] DSV41_RATIOS = {2, 2, 1, 1};
    private static final int[] DSV41_SOURCES = {0, 2};
    private static final int[] DSV41_CANDIDATE_RATIOS = {2, 2, 2};

    private SyntheticRoutedModels() {
    }

    /** 2 layers of 4 query heads of width 32 (16 rotary) over one shared KV head, 4 experts of width 32, 2 used. */
    public static SyntheticModels.Config deepseek4Config() {
        return new SyntheticModels.Config.Builder().nLayer(2).nHead(4).nHeadKv(1).headDim(32).nRot(16).nFf(128)
                .nExpert(4).nExpertUsed(2).nFfExp(32).normEps(1e-6f).build();
    }

    public static GgufTestWriter deepseek4() {
        return deepseek4(deepseek4Config());
    }

    /**
     * LLM_ARCH_DEEPSEEK4 (models/deepseek4.cpp): sliding-window MQA attention with sinks over a window wider than any
     * test context, block 0 uncompressed and block 1 a CSA layer (compression ratio {@value #DSV4_CSA_RATIO} with
     * the lightning indexer selecting every compressed row), the output through {@link #DSV4_OUTPUT_GROUPS} grouped
     * low-rank projections, four hyper-connection streams, and sqrt-softplus routed experts with a selection bias
     * and one shared expert on every layer (no hash-routed layers).
     */
    public static GgufTestWriter deepseek4(SyntheticModels.Config c) {
        return deepseek4(c, 0, false);
    }

    /**
     * {@link #deepseek4(SyntheticModels.Config)} whose first {@code hashLayers} blocks route token rows through
     * {@code ffn_gate_tid2eid} (expert {@code (t + 2j) % n_expert} for token {@code t}, slot {@code j}) instead of
     * {@code exp_probs_b}, and whose blocks carry the vision routing bias {@code exp_probs_b_vl} when
     * {@code visionBias} (deepseek4.cpp:148-155).
     *
     * @throws IllegalArgumentException when {@code hashLayers} is outside {@code [0, n_layer]}
     */
    public static GgufTestWriter deepseek4(SyntheticModels.Config c, int hashLayers, boolean visionBias) {
        if (hashLayers < 0 || hashLayers > c.nLayer()) {
            throw new IllegalArgumentException("hash layer count " + hashLayers + " is outside [0, " + c.nLayer()
                    + "]");
        }
        requirePositive("deepseek4 expert count", c.nExpert());
        requirePositive("deepseek4 expert feed-forward length", c.nFfExp());
        if (c.nLayer() != 2 || ((long) c.nHead() * c.headDim()) % DSV4_OUTPUT_GROUPS != 0) {
            throw new IllegalArgumentException("the deepseek4 fixture writes 2 layers and needs the query width to"
                    + " split into " + DSV4_OUTPUT_GROUPS + " output groups");
        }
        GgufTestWriter w = SyntheticText.base("deepseek4", c);
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("attention.q_lora_rank"), DSV4_Q_LORA);
        w.u32(w.archKey("attention.sliding_window"), c.nCtxTrain());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.f32(w.archKey("expert_weights_scale"), 1.5f);
        w.bool(w.archKey("expert_weights_norm"), true);
        w.u32(w.archKey("expert_gating_func"), GATING_SQRT_SOFTPLUS);
        w.f32(w.archKey("swiglu_clamp_exp"), 10.0f);
        w.u32(w.archKey("attention.indexer.head_count"), DSV4_INDEXER_HEADS);
        w.u32(w.archKey("attention.indexer.key_length"), DSV4_INDEXER_DIM);
        w.u32(w.archKey("attention.indexer.top_k"), c.nCtxTrain());
        w.u32(w.archKey("attention.output_group_count"), DSV4_OUTPUT_GROUPS);
        w.u32(w.archKey("attention.output_lora_rank"), DSV4_OUTPUT_LORA);
        w.f32(w.archKey("attention.compress_rope_freq_base"), 160000.0f);
        w.u32Array(w.archKey("attention.compress_ratios"), 0, DSV4_CSA_RATIO);
        w.u32(w.archKey("hyper_connection.count"), DSV4_HC);
        w.u32(w.archKey("hyper_connection.sinkhorn_iterations"), 20);
        w.f32(w.archKey("hyper_connection.epsilon"), 1e-6f);
        w.u32(w.archKey("hash_layer_count"), hashLayers);
        bpeTokenizer(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long hcDim = (long) DSV4_HC * c.nEmbd();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            dsv4Attention(t, c, il);
            for (String site : new String[] {"attn", "ffn"}) {
                t.random(blk(il, "hc_" + site + "_fn.weight"), -0.05f, 0.05f, hcDim, (2L + DSV4_HC) * DSV4_HC);
                t.random(blk(il, "hc_" + site + "_base.weight"), -0.1f, 0.1f, (2L + DSV4_HC) * DSV4_HC);
                t.random(blk(il, "hc_" + site + "_scale.weight"), 0.5f, 1.0f, 3);
            }
            if (il == 1) {
                dsv4Compressor(t, c, il);
            }
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            if (il < hashLayers) {
                hashTable(w, c, il);
            } else {
                t.random(blk(il, "exp_probs_b.bias"), -0.02f, 0.02f, c.nExpert());
            }
            if (visionBias) {
                t.random(blk(il, "exp_probs_b_vl.bias"), -0.5f, 0.5f, c.nExpert());
            }
            t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
            t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
        }
        t.random("output_hc_fn.weight", -0.05f, 0.05f, hcDim, DSV4_HC);
        t.random("output_hc_base.weight", -0.1f, 0.1f, DSV4_HC);
        t.random("output_hc_scale.weight", 0.5f, 1.0f, 1);
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** {@code ffn_gate_tid2eid} of block {@code il}: I32 {@code [n_expert_used, n_vocab]}. */
    private static void hashTable(GgufTestWriter w, SyntheticModels.Config c, int il) {
        int k = c.nExpertUsed();
        ByteBuffer ids = ByteBuffer.allocate(k * c.nVocab() * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int tok = 0; tok < c.nVocab(); tok++) {
            for (int j = 0; j < k; j++) {
                ids.putInt((tok + 2 * j) % c.nExpert());
            }
        }
        w.tensor(blk(il, "ffn_gate_tid2eid.weight"), GGMLType.I32, new long[] {k, c.nVocab()}, ids.array());
    }

    /** The sliding-window MQA attention of deepseek4 block {@code il}, with its grouped output projection. */
    private static void dsv4Attention(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long qWidth = (long) c.nHead() * c.headDim();
        t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
        t.random(blk(il, "attn_sinks.weight"), -0.5f, 0.5f, c.nHead());
        t.matrix(blk(il, "attn_q_a.weight"), c.nEmbd(), DSV4_Q_LORA);
        t.ones(blk(il, "attn_q_a_norm.weight"), DSV4_Q_LORA);
        t.matrix(blk(il, "attn_q_b.weight"), DSV4_Q_LORA, qWidth);
        t.matrix(blk(il, "attn_kv.weight"), c.nEmbd(), c.headDim());
        t.ones(blk(il, "attn_kv_a_norm.weight"), c.headDim());
        t.matrix(blk(il, "attn_output_a.weight"), qWidth / DSV4_OUTPUT_GROUPS, DSV4_OUTPUT_LORA, DSV4_OUTPUT_GROUPS);
        t.matrix(blk(il, "attn_output_b.weight"), (long) DSV4_OUTPUT_GROUPS * DSV4_OUTPUT_LORA, c.nEmbd());
    }

    /** The KV compressor and the lightning indexer of a CSA block (deepseek4.cpp, ratio 4). */
    private static void dsv4Compressor(SyntheticModels.Tensors t, SyntheticModels.Config c, int il) {
        long kvWidth = 2L * c.headDim();
        t.matrix(blk(il, "attn_compressor_kv.weight"), c.nEmbd(), kvWidth);
        t.matrix(blk(il, "attn_compressor_gate.weight"), c.nEmbd(), kvWidth);
        t.random(blk(il, "attn_compressor_ape.weight"), -0.1f, 0.1f, kvWidth, DSV4_CSA_RATIO);
        t.ones(blk(il, "attn_compressor_norm.weight"), c.headDim());
        t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), DSV4_INDEXER_HEADS);
        t.matrix(blk(il, "indexer.attn_q_b.weight"), DSV4_Q_LORA, (long) DSV4_INDEXER_HEADS * DSV4_INDEXER_DIM);
        long idxWidth = 2L * DSV4_INDEXER_DIM;
        t.matrix(blk(il, "indexer_compressor_kv.weight"), c.nEmbd(), idxWidth);
        t.matrix(blk(il, "indexer_compressor_gate.weight"), c.nEmbd(), idxWidth);
        t.random(blk(il, "indexer_compressor_ape.weight"), -0.1f, 0.1f, idxWidth, DSV4_CSA_RATIO);
        t.ones(blk(il, "indexer_compressor_norm.weight"), DSV4_INDEXER_DIM);
    }

    /** 4 layers of 4 heads of width 32 (16 rotary), 4 experts of width 32, 2 used, context 64. */
    public static SyntheticModels.Config deepseek41Config() {
        return new SyntheticModels.Config.Builder().nLayer(4).nHead(4).nHeadKv(1).headDim(32).nRot(16).nFf(128)
                .nExpert(4).nExpertUsed(2).nFfExp(32).nCtxTrain(64).normEps(1e-6f).build();
    }

    /** The 3-layer all-selected fixture of {@link #deepseek41Candidate()}. */
    public static SyntheticModels.Config deepseek41CandidateConfig() {
        return deepseek41Config().toBuilder().nLayer(3).build();
    }

    /**
     * LLM_ARCH_DEEPSEEK41 (models/deepseek41.cpp): four layers of ratios {@code 2, 2, 1, 1}, layers 0 and 2 owning
     * the compressed rows and the indexer top-k, layers 1 and 3 reading them, an Engram layer on layer 1, grouped
     * low-rank output, four hyper-connection streams and no {@code output_hc}.
     */
    public static GgufTestWriter deepseek41() {
        return deepseek41(deepseek41Config());
    }

    /** {@link #deepseek41()} over {@code c}, which must hold 4 layers. */
    public static GgufTestWriter deepseek41(SyntheticModels.Config c) {
        return deepseek41(c, DSV41_RATIOS, sourceLayers(DSV41_RATIOS), -1, new int[]{1});
    }

    /**
     * The three-selected-layer variant with the two-level candidate selection: layer 0 owns the compressed rows, the
     * top-k and the candidate blocks (2 blocks of 2 rows), layer 1 reads layer 0's top-k, and layer 2 owns its own
     * rows and top-k but scores inside layer 0's candidate blocks.
     */
    public static GgufTestWriter deepseek41Candidate() {
        return deepseek41(deepseek41CandidateConfig(), DSV41_CANDIDATE_RATIOS, new int[]{0, 2}, 0,
                new int[]{2});
    }

    private static GgufTestWriter deepseek41(SyntheticModels.Config c, int[] ratios, int[] sourceLayers,
                                             int candidateSource, int[] engramLayers) {
        requirePositive("deepseek41 expert count", c.nExpert());
        requirePositive("deepseek41 expert feed-forward length", c.nFfExp());
        if (c.nLayer() != ratios.length || ((long) c.nHead() * c.headDim()) % DSV4_OUTPUT_GROUPS != 0) {
            throw new IllegalArgumentException("the deepseek41 fixture writes " + ratios.length + " layers and needs"
                    + " the query width to split into " + DSV4_OUTPUT_GROUPS + " output groups");
        }
        if ((long) DSV41_HEADS_PER_NGRAM * (DSV41_NGRAM - 1) * DSV41_HEAD_DIM != c.nEmbd()) {
            throw new IllegalArgumentException("the deepseek41 Engram heads must span n_embd " + c.nEmbd());
        }
        java.util.List<Integer> sources = asList(sourceLayers);
        GgufTestWriter w = SyntheticText.base("deepseek41", c);
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("attention.q_lora_rank"), DSV4_Q_LORA);
        w.u32(w.archKey("attention.sliding_window"), c.nCtxTrain());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("expert_shared_count"), 1);
        w.f32(w.archKey("expert_weights_scale"), 1.5f);
        w.bool(w.archKey("expert_weights_norm"), true);
        w.u32(w.archKey("expert_gating_func"), GATING_SQRT_SOFTPLUS);
        w.f32(w.archKey("swiglu_clamp_exp"), 10.0f);
        w.u32(w.archKey("attention.indexer.head_count"), DSV4_INDEXER_HEADS);
        w.u32(w.archKey("attention.indexer.key_length"), DSV4_INDEXER_DIM);
        w.u32(w.archKey("attention.indexer.top_k"), DSV41_TOP_K);
        w.u32(w.archKey("attention.output_group_count"), DSV4_OUTPUT_GROUPS);
        w.u32(w.archKey("attention.output_lora_rank"), DSV4_OUTPUT_LORA);
        w.f32(w.archKey("attention.compress_rope_freq_base"), 160000.0f);
        w.u32Array(w.archKey("attention.compress_ratios"), toLongs(ratios));
        w.u32Array(w.archKey("attention.kv_source_layers"), toLongs(sources));
        w.u32Array(w.archKey("attention.indexer.source_layers"), toLongs(sources));
        if (candidateSource >= 0) {
            w.u32(w.archKey("attention.indexer.candidate_source_layer"), candidateSource);
            w.u32(w.archKey("attention.indexer.candidate_top_k_blocks"), 2);
            w.u32(w.archKey("attention.indexer.candidate_block_size"), 2);
        }
        w.u32(w.archKey("hyper_connection.count"), DSV4_HC);
        w.u32(w.archKey("hyper_connection.sinkhorn_iterations"), 20);
        w.f32(w.archKey("hyper_connection.epsilon"), 1e-6f);
        w.u32(w.archKey("hash_layer_count"), 0);
        engramKeys(w, c, engramLayers);
        bpeTokenizer(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long hcDim = (long) DSV4_HC * c.nEmbd();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        t.matrix("per_layer_token_embd.weight", DSV41_HEAD_DIM, DSV41_TABLE_ROWS);
        int kvSource = -1;
        int topkSource = -1;
        for (int il = 0; il < c.nLayer(); il++) {
            int ratio = ratios[il];
            boolean source = contains(sourceLayers, il);
            if (source) {
                kvSource = il;
                topkSource = il;
            }
            dsv4Attention(t, c, il);
            for (String site : new String[]{"attn", "ffn"}) {
                t.random(blk(il, "hc_" + site + "_fn.weight"), -0.05f, 0.05f, hcDim, (2L + DSV4_HC) * DSV4_HC);
                t.random(blk(il, "hc_" + site + "_base.weight"), -0.1f, 0.1f, (2L + DSV4_HC) * DSV4_HC);
                t.random(blk(il, "hc_" + site + "_scale.weight"), 0.5f, 1.0f, 3);
            }
            if (ratio != 0) {
                dsv41Compressor(t, c, il, ratio, kvSource == il, topkSource == il);
            }
            if (contains(engramLayers, il)) {
                t.matrix(blk(il, "engram_kv.weight"), c.nEmbd(), (long) c.nEmbd() * (DSV4_HC + 1));
                t.matrix(blk(il, "engram_gate.weight"), c.nEmbd(), DSV4_HC);
            }
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            t.random(blk(il, "exp_probs_b.bias"), -0.02f, 0.02f, c.nExpert());
            t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
            t.matrix(blk(il, "ffn_gate_shexp.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(il, "ffn_up_shexp.weight"), c.nEmbd(), c.nFfExp());
            t.matrix(blk(il, "ffn_down_shexp.weight"), c.nFfExp(), c.nEmbd());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }

    /** The compressed-KV compressor and index-key tensors of a DeepSeek-V4.1 compressed layer. */
    private static void dsv41Compressor(SyntheticModels.Tensors t, SyntheticModels.Config c, int il, int ratio,
                                        boolean kvSource, boolean topkSource) {
        if (kvSource) {
            t.matrix(blk(il, "attn_compressor_kv.weight"), c.nEmbd(), c.headDim());
            if (ratio > 1) {
                t.matrix(blk(il, "attn_compressor_gate.weight"), c.nEmbd(), c.headDim());
            }
            t.ones(blk(il, "attn_compressor_norm.weight"), c.headDim());
            t.matrix(blk(il, "indexer.attn_k.weight"), c.headDim(), DSV4_INDEXER_DIM);
            t.ones(blk(il, "indexer.k_norm.weight"), DSV4_INDEXER_DIM);
        }
        if (topkSource) {
            t.matrix(blk(il, "indexer.proj.weight"), c.nEmbd(), DSV4_INDEXER_HEADS);
            t.matrix(blk(il, "indexer.attn_q_b.weight"), DSV4_Q_LORA, (long) DSV4_INDEXER_HEADS * DSV4_INDEXER_DIM);
        }
    }

    /** The Engram metadata and its shared table: {@code per_layer_token_embd} is written by the caller. */
    private static void engramKeys(GgufTestWriter w, SyntheticModels.Config c, int[] engramLayers) {
        int nCols = DSV41_HEADS_PER_NGRAM * (DSV41_NGRAM - 1);
        w.u32Array(w.archKey("ple.layers"), toLongs(engramLayers));
        w.u32(w.archKey("ple.ngram_size"), DSV41_NGRAM);
        w.u32(w.archKey("ple.heads_per_ngram"), DSV41_HEADS_PER_NGRAM);
        w.u32(w.archKey("embedding_length_per_layer_input"), DSV41_HEAD_DIM);
        long[] mult = new long[engramLayers.length * DSV41_NGRAM];
        long[] offsets = new long[engramLayers.length * nCols];
        long[] vocab = new long[engramLayers.length * nCols];
        for (int e = 0; e < engramLayers.length; e++) {
            for (int o = 0; o < DSV41_NGRAM; o++) {
                mult[e * DSV41_NGRAM + o] = 1000003L + 31L * o + e;
            }
            for (int col = 0; col < nCols; col++) {
                offsets[e * nCols + col] = (long) col * DSV41_TABLE_ROWS / 2;
                vocab[e * nCols + col] = DSV41_TABLE_ROWS / 2;
            }
        }
        w.u64Array(w.archKey("ple.layer_multipliers"), mult);
        w.u64Array(w.archKey("ple.head_offsets"), offsets);
        w.u64Array(w.archKey("ple.head_vocab_sizes"), vocab);
        w.u32(w.archKey("engram.pad_token_id"), 0);
        long[] map = new long[c.nVocab()];
        for (int tok = 0; tok < map.length; tok++) {
            map[tok] = tok % DSV41_TABLE_ROWS;
        }
        w.u32Array(w.archKey("engram.token_map"), map);
    }

    /** Layer indices at which a new KV/indexer source starts: every run of equal non-zero ratios. */
    private static java.util.List<Integer> asList(int[] values) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (int v : values) {
            out.add(v);
        }
        return out;
    }

    private static int[] sourceLayers(int[] ratios) {
        int[] tmp = new int[ratios.length];
        int n = 0;
        for (int il = 0; il < ratios.length; il++) {
            if (ratios[il] != 0 && (il == 0 || ratios[il - 1] != ratios[il])) {
                tmp[n++] = il;
            }
        }
        return java.util.Arrays.copyOf(tmp, n);
    }

    private static boolean contains(int[] values, int v) {
        for (int value : values) {
            if (value == v) {
                return true;
            }
        }
        return false;
    }

    private static long[] toLongs(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static long[] toLongs(java.util.List<Integer> values) {
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static void bpeTokenizer(GgufTestWriter w, SyntheticModels.Config c) {
        SyntheticText.bpeVocab(w, c.nVocab(), "gpt-2", new String[] {"<|endoftext|>"});
        w.u32("tokenizer.ggml.bos_token_id", 0).u32("tokenizer.ggml.eos_token_id", 0)
                .bool("tokenizer.ggml.add_bos_token", false);
    }

    public static GgufTestWriter k2Horizon() {
        return k2Horizon(SyntheticModels.Config.qwen3moe());
    }

    /**
     * LLM_ARCH_K2_HORIZON (models/k2-horizon.cpp): GQA attention with per-head Q/K norms; block 0 is a dense lead
     * with a plain V projection, the later blocks route {@code nExpert} sigmoid-gated FFN experts and take V from
     * {@link #VALUE_EXPERTS} routed value experts ({@link #VALUE_EXPERTS_USED} per token) of the attention input.
     */
    public static GgufTestWriter k2Horizon(SyntheticModels.Config c) {
        requirePositive("k2-horizon expert count", c.nExpert());
        requirePositive("k2-horizon expert feed-forward length", c.nFfExp());
        GgufTestWriter w = SyntheticText.base("k2-horizon", c);
        SyntheticText.attentionKeys(w, c, true);
        w.u32(w.archKey("rope.dimension_count"), c.nRot());
        w.f32(w.archKey("rope.freq_base"), c.ropeFreqBase());
        w.f32(w.archKey("attention.layer_norm_rms_epsilon"), c.normEps());
        w.u32(w.archKey("expert_count"), c.nExpert());
        w.u32(w.archKey("expert_used_count"), c.nExpertUsed());
        w.u32(w.archKey("expert_feed_forward_length"), c.nFfExp());
        w.u32(w.archKey("leading_dense_block_count"), 1);
        w.u32(w.archKey("expert_gating_func"), 2);
        w.bool(w.archKey("expert_weights_norm"), true);
        w.u32(w.archKey("attention.value_expert_count"), VALUE_EXPERTS);
        w.u32(w.archKey("attention.value_expert_used_count"), VALUE_EXPERTS_USED);
        bpeTokenizer(w, c);

        SyntheticModels.Tensors t = new SyntheticModels.Tensors(w, c);
        long kvRows = (long) c.nHeadKv() * c.headDim();
        t.matrix("token_embd.weight", c.nEmbd(), c.nVocab());
        for (int il = 0; il < c.nLayer(); il++) {
            boolean moe = il > 0;
            t.ones(blk(il, "attn_norm.weight"), c.nEmbd());
            t.matrix(blk(il, "attn_q.weight"), c.nEmbd(), (long) c.nHead() * c.headDim());
            t.matrix(blk(il, "attn_k.weight"), c.nEmbd(), kvRows);
            t.ones(blk(il, "attn_q_norm.weight"), (long) c.headDim() * c.nHead());
            t.ones(blk(il, "attn_k_norm.weight"), kvRows);
            if (moe) {
                t.random(blk(il, "attn_v_gate.weight"), -0.1f, 0.1f, c.nEmbd(), VALUE_EXPERTS);
                t.matrix(blk(il, "attn_v_exps.weight"), c.nEmbd(), kvRows, VALUE_EXPERTS);
            } else {
                t.matrix(blk(il, "attn_v.weight"), c.nEmbd(), kvRows);
            }
            t.matrix(blk(il, "attn_output.weight"), (long) c.nHead() * c.headDim(), c.nEmbd());
            t.ones(blk(il, "ffn_norm.weight"), c.nEmbd());
            if (!moe) {
                t.gatedFfn(il, c.nFf());
                continue;
            }
            t.random(blk(il, "ffn_gate_inp.weight"), -0.1f, 0.1f, c.nEmbd(), c.nExpert());
            t.matrix(blk(il, "ffn_gate_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_up_exps.weight"), c.nEmbd(), c.nFfExp(), c.nExpert());
            t.matrix(blk(il, "ffn_down_exps.weight"), c.nFfExp(), c.nEmbd(), c.nExpert());
        }
        t.ones("output_norm.weight", c.nEmbd());
        t.matrix("output.weight", c.nEmbd(), c.nVocab());
        return w;
    }
}
