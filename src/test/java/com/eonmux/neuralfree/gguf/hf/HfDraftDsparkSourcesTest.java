package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfMistralLarge4Fixtures;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The DSpark draft exports of Mistral Large 4 ({@code MistralLarge4DSparkModel}, mistral4.py:132-217) and
 * DeepSeek-V4.1 ({@code DeepseekV41DSparkModel}, deepseek41.py:331-442) open as DFLASH checkpoints: the stage tensors
 * carry their DFLASH names, the head tensors map to the Markov/confidence projections, the routed experts stack and
 * the block protocol keys are written.
 */
public class HfDraftDsparkSourcesTest {

    private static final int DIM = 8;
    private static final int RANK = 4;
    private static final int VOCAB = 8;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theMistralLarge4DsparkDraftExportsItsStageAndHeadTensors() throws IOException {
        Path dir = mistralLarge4().write(tmp.newFolder("ml4").toPath());
        try (ModelSource src = ModelSources.open(dir.toString(), options(), null)) {
            assertEquals("dflash", src.getString("general.architecture"));
            assertEquals(1, number(src, "dflash.block_count"));
            assertEquals(4, number(src, "dflash.block_size"));
            assertEquals(1, number(src, "dflash.attention.head_count_kv"));
            List<Integer> targets = ints(src, "dflash.target_layers");
            assertEquals(3, targets.get(0).intValue());
            assertEquals(Boolean.FALSE, src.getMetadata().get("dflash.has_confidence_head"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.sample_from_anchor"));
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "mask_embd.weight", "markov_w1.weight", "markov_w2.weight", "joint_proj.weight",
                    "joint_proj.bias", "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k_b.weight",
                    "blk.0.attn_v_b.weight"}) {
                assertTrue(name + " in " + names(src), names(src).contains(name));
            }
            assertEquals("the mask id is the backbone's mask embedding", 0,
                    number(src, "tokenizer.ggml.mask_token_id"));
        }
    }

    @Test
    public void theDeepseekV41DsparkDraftExportsItsStageAndHeadTensors() throws IOException {
        Path dir = deepseekV41().write(tmp.newFolder("v41").toPath());
        try (ModelSource src = ModelSources.open(dir.toString(), options(), null)) {
            assertEquals("dflash", src.getString("general.architecture"));
            assertEquals(1, number(src, "dflash.block_count"));
            assertEquals(4, number(src, "dflash.block_size"));
            assertEquals(1, number(src, "dflash.attention.head_count_kv"));
            assertEquals(4, number(src, "dflash.expert_count"));
            assertEquals(1, number(src, "dflash.expert_used_count"));
            assertEquals(1, number(src, "dflash.expert_shared_count"));
            assertEquals(2, number(src, "dflash.hyper_connection.count"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.has_confidence_head"));
            List<Integer> targets = ints(src, "dflash.target_layers");
            assertEquals(2, targets.size());
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight", "blk.0.attn_norm.weight",
                    "blk.0.attn_q_a.weight", "blk.0.attn_q_a_norm.weight", "blk.0.attn_q_b.weight",
                    "blk.0.attn_kv.weight", "blk.0.attn_kv_a_norm.weight", "blk.0.ffn_gate_inp.weight",
                    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_up_exps.weight",
                    "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight", "blk.0.ffn_up_shexp.weight"}) {
                assertTrue(name + " in " + names(src), names(src).contains(name));
            }
            assertEquals(7, number(src, "tokenizer.ggml.mask_token_id"));
        }
    }

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache-" + System.nanoTime()).toPath()).build();
    }

    // ------------------------------------------------------------------ fixtures

    private static SyntheticHf.Builder mistralLarge4() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['MistralLarge4DSparkModel'], 'dim': " + DIM + ", 'hidden_dim': 16, 'n_layers': 1,"
                + " 'n_heads': 2, 'n_kv_heads': 1, 'norm_eps': 1e-05, 'vocab_size': 277,"
                + " 'max_position_embeddings': 128, 'rope_theta': 10000.0, 'qk_rope_head_dim': 2,"
                + " 'qk_nope_head_dim': 2, 'v_head_dim': 2, 'kv_lora_rank': 4, 'nope': 'rope',"
                + " 'softmax_tanh_gating': 50.0,"
                + " 'moe': {'num_experts': 2, 'expert_hidden_dim': 4, 'num_experts_per_tok': 1,"
                + "   'num_shared_experts': 1, 'route_every_n': 1, 'first_k_dense_replace': 0,"
                + "   'routed_scale': 1.0},"
                + " 'dsa': {'n_heads': 2, 'head_dim': 2, 'topk': 1, 'sharing': 'full', 'norm_eps': 1e-06,"
                + "   'rope_head_dim': 2, 'match_nope': true, 'ssmax_scaling': true},"
                + " 'yarn': {'factor': 4.0, 'beta': 32.0, 'alpha': 1.0,"
                + "   'original_max_position_embeddings': 128, 'apply_scale': false},"
                + " 'mtp': {'n_layers': 1, 'depth': 4, 'ragged_attention': '8',"
                + "   'target_activation_layers': '2',"
                + "   'dspark': {'head_type': 'rnn', 'draft_input': 'position_specific'}}}"));
        b.file("tekken.json", HfMistralLarge4Fixtures.tekkenJson());
        b.bf16("mtp_dspark.backbone.target_projection.weight", DIM, DIM);
        b.bf16("mtp_dspark.backbone.target_norm.weight", DIM);
        b.bf16("mtp_dspark.backbone.final_norm.weight", DIM);
        b.bf16("mtp_dspark.backbone.mask_embedding", DIM, RANK);
        b.bf16("mtp_dspark.autoregressive_head.embedding.weight", 277, RANK);
        b.bf16("mtp_dspark.autoregressive_head.output.weight", 277, RANK);
        b.bf16("mtp_dspark.autoregressive_head.joint_proj.weight", 3 * RANK, DIM + 2 * RANK);
        b.bf16("mtp_dspark.autoregressive_head.joint_proj.bias", 3 * RANK);
        b.bf16("mtp_dspark.backbone.layers.0.attention_norm.weight", DIM);
        b.bf16("mtp_dspark.backbone.layers.0.ffn_norm.weight", DIM);
        b.bf16("mtp_dspark.backbone.layers.0.attention.wq.weight", 2 * 4, DIM);
        b.bf16("mtp_dspark.backbone.layers.0.attention.wkv_b.weight", 2 * (2 + 2), 4);
        return b;
    }

    private static SyntheticHf.Builder deepseekV41() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['DeepseekV41DSparkModel'], 'model_type': 'deepseek_v41', 'hidden_size': " + DIM + ","
                + " 'intermediate_size': 16, 'moe_intermediate_size': 4, 'num_attention_heads': 2,"
                + " 'num_key_value_heads': 1, 'head_dim': 4, 'num_hidden_layers': 2, 'num_nextn_predict_layers': 1,"
                + " 'max_position_embeddings': 128, 'rms_norm_eps': 1e-06, 'vocab_size': " + VOCAB + ","
                + " 'rope_theta': 10000.0, 'qk_rope_head_dim': 2, 'q_lora_rank': 4, 'sliding_window': 8,"
                + " 'n_routed_experts': 2, 'num_experts_per_tok': 1, 'n_shared_experts': 1,"
                + " 'routed_scaling_factor': 1.5, 'norm_topk_prob': true, 'swiglu_limit': 10.0,"
                + " 'o_groups': 2, 'o_lora_rank': 2, 'hc_mult': 2, 'hc_sinkhorn_iters': 4, 'hc_eps': 1e-06,"
                + " 'scoring_func': 'sqrtsoftplus', 'gate_temp': 1.0,"
                + " 'dspark_block_size': 4, 'dspark_target_layer_ids': [0, 1], 'dspark_noise_token_id': 7,"
                + " 'dspark_n_routed_experts': 4, 'dspark_num_experts_per_tok': 1}"));
        b.bf16("mtp.0.main_proj.weight", DIM, DIM);
        b.bf16("mtp.0.main_norm.weight", DIM);
        b.bf16("mtp.0.norm.weight", DIM);
        b.bf16("mtp.0.markov_head.embed.weight", VOCAB, RANK);
        b.bf16("mtp.0.markov_head.head.weight", VOCAB, RANK);
        b.bf16("mtp.0.confidence_head.proj.weight", 1, DIM + RANK);
        b.bf16("mtp.0.attn_norm.weight", DIM);
        b.bf16("mtp.0.ffn_norm.weight", DIM);
        b.bf16("mtp.0.attn.wq_a.weight", DIM, 4);
        b.bf16("mtp.0.attn.q_norm.weight", 4);
        b.bf16("mtp.0.attn.wq_b.weight", 4, 2 * 4);
        b.bf16("mtp.0.attn.wkv.weight", DIM, 4);
        b.bf16("mtp.0.attn.kv_norm.weight", 4);
        b.bf16("mtp.0.ffn.gate.weight", 4, DIM);
        for (int e = 0; e < 4; e++) {
            b.bf16("mtp.0.ffn.experts." + e + ".w1.weight", 4, DIM);
            b.bf16("mtp.0.ffn.experts." + e + ".w2.weight", DIM, 4);
            b.bf16("mtp.0.ffn.experts." + e + ".w3.weight", 4, DIM);
        }
        b.bf16("mtp.0.ffn.shared_experts.w1.weight", 4, DIM);
        b.bf16("mtp.0.ffn.shared_experts.w2.weight", DIM, 4);
        b.bf16("mtp.0.ffn.shared_experts.w3.weight", 4, DIM);
        return b;
    }

    private static List<String> names(ModelSource src) {
        List<String> out = new ArrayList<>();
        src.getTensors().forEach(t -> out.add(t.name()));
        return out;
    }

    private static long number(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertNotNull(key + " in " + src.getMetadata().keySet(), v);
        return ((Number) v).longValue();
    }

    @SuppressWarnings("unchecked")
    private static List<Integer> ints(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertNotNull(key + " in " + src.getMetadata().keySet(), v);
        return (List<Integer>) v;
    }
}
