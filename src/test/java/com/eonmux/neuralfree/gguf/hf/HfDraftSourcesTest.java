package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDraftFixtures;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The DSpark draft exports of {@code DeepseekV41DSparkModel} (deepseek41.py:331-442) and
 * {@code MistralLarge4DSparkModel} (mistral4.py:132-217) onto the {@code dflash} architecture: the DFLASH metadata
 * each converter writes, the re-keyed stage tensors, the Markov and confidence heads and the target layers.
 */
public class HfDraftSourcesTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    private static Set<String> names(ModelSource src) {
        Set<String> out = new HashSet<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static List<Integer> ints(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertTrue(key + " is missing", v instanceof List);
        List<Integer> out = new ArrayList<>();
        for (Object o : (List<?>) v) {
            out.add(((Number) o).intValue());
        }
        return out;
    }

    @Test
    public void theV41DsparkExportWritesItsStagesHeadsAndMetadata() throws IOException {
        Path dir = HfDraftFixtures.deepseekV41Dspark().write(tmp.newFolder("v41").toPath());
        try (ModelSource src = ModelSources.open(dir.toString(), options(), null)) {
            assertEquals("dflash", src.getString("general.architecture"));
            assertEquals(Arch.DFLASH, Arch.fromGgufName(src.getString("general.architecture")));
            assertEquals(2, ((Number) src.getMetadata().get("dflash.block_count")).intValue());
            assertEquals(8, ((Number) src.getMetadata().get("dflash.block_size")).intValue());
            assertEquals(Arrays.asList(0, 1), ints(src, "dflash.target_layers"));
            assertEquals(2, ((Number) src.getMetadata().get("dflash.expert_count")).intValue());
            assertEquals(2, ((Number) src.getMetadata().get("dflash.hyper_connection.count")).intValue());
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.hyper_connection.single_pass"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.has_confidence_head"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.sample_from_anchor"));

            Set<String> t = names(src);
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight",
                    "blk.0.attn_q_a.weight", "blk.0.attn_q_b.weight", "blk.0.attn_kv.weight",
                    "blk.0.attn_output_a.weight", "blk.0.attn_output_b.weight", "blk.0.hc_attn_fn.weight",
                    "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_shexp.weight",
                    "blk.1.attn_q_a.weight", "blk.1.ffn_down_exps.weight"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("an mtp tensor leaked: " + name, name.startsWith("mtp"));
            }
        }
    }

    @Test
    public void theMistralLarge4DsparkExportWritesItsMlaStageAndHeads() throws IOException {
        Path dir = HfDraftFixtures.mistralLarge4Dspark().write(tmp.newFolder("ml4").toPath());
        try (ModelSource src = ModelSources.open(dir.toString(), options(), null)) {
            assertEquals("dflash", src.getString("general.architecture"));
            assertEquals(1, ((Number) src.getMetadata().get("dflash.block_count")).intValue());
            assertEquals(8, ((Number) src.getMetadata().get("dflash.block_size")).intValue());
            assertEquals(Arrays.asList(2, 3), ints(src, "dflash.target_layers"));
            assertEquals(Boolean.FALSE, src.getMetadata().get("dflash.has_confidence_head"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("dflash.sample_from_anchor"));

            Set<String> t = names(src);
            for (String name : new String[]{"fc.weight", "enc.output_norm.weight", "output_norm.weight",
                    "mask_embd.weight", "markov_w1.weight", "markov_w2.weight", "joint_proj.weight",
                    "joint_proj.bias", "blk.0.attn_q.weight", "blk.0.attn_kv_a_mqa.weight",
                    "blk.0.attn_kv_a_norm.weight", "blk.0.attn_k_b.weight", "blk.0.attn_v_b.weight",
                    "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_shexp.weight"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("an mtp_dspark tensor leaked: " + name, name.startsWith("mtp_dspark"));
            }
        }
    }

    @Test
    public void aV41DsparkWithoutNextnStagesIsRefusedByName() {
        SyntheticHf.Builder b = HfDraftFixtures.deepseekV41Dspark()
                .config(SyntheticHf.json("{'architectures': ['DeepseekV41DSparkModel'], 'num_nextn_predict_layers': 0,"
                        + " 'dspark_block_size': 8, 'num_hidden_layers': 2, 'hidden_size': 32,"
                        + " 'num_attention_heads': 4, 'num_key_value_heads': 1, 'head_dim': 8,"
                        + " 'vocab_size': 16, 'rms_norm_eps': 1e-06, 'max_position_embeddings': 64}"));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(b, "v41-bad"));
        assertTrue(e.getMessage(), e.getMessage().contains("num_nextn_predict_layers"));
    }

    @Test
    public void aMistralLarge4DsparkWithAnotherHeadTypeIsRefusedByKey() {
        SyntheticHf.Builder b = HfDraftFixtures.mistralLarge4Dspark().file("config.json",
                SyntheticHf.json("{'architectures': ['MistralLarge4DSparkModel'], 'dim': 8, 'hidden_dim': 16,"
                        + " 'n_layers': 2, 'n_heads': 2, 'n_kv_heads': 1, 'norm_eps': 1e-05, 'vocab_size': 277,"
                        + " 'max_position_embeddings': 128, 'rope_theta': 10000.0, 'qk_rope_head_dim': 2,"
                        + " 'qk_nope_head_dim': 2, 'v_head_dim': 2, 'kv_lora_rank': 4,"
                        + " 'nope': 'rope,nope', 'softmax_tanh_gating': 50.0,"
                        + " 'moe': {'num_experts': 2, 'expert_hidden_dim': 4, 'num_experts_per_tok': 1,"
                        + " 'num_shared_experts': 1, 'route_every_n': 1, 'first_k_dense_replace': 0},"
                        + " 'dsa': {'n_heads': 2, 'head_dim': 2, 'topk': 1, 'sharing': 'full,shared',"
                        + " 'norm_eps': 1e-06, 'rope_head_dim': 2, 'match_nope': true, 'ssmax_scaling': true},"
                        + " 'yarn': {'factor': 4.0, 'beta': 32.0, 'alpha': 1.0,"
                        + " 'original_max_position_embeddings': 128, 'apply_scale': false},"
                        + " 'mtp': {'n_layers': 1, 'depth': 8, 'ragged_attention': '8',"
                        + " 'target_activation_layers': '1,2',"
                        + " 'dspark': {'head_type': 'mlp', 'draft_input': 'position_specific'}}}"));
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> open(b, "ml4-bad"));
        assertTrue(e.getMessage(), e.getMessage().contains("head_type"));
    }

    private ModelSource open(SyntheticHf.Builder b, String dirName) throws IOException {
        Path dir = b.write(tmp.newFolder(dirName).toPath());
        return ModelSources.open(dir.toString(), options(), null);
    }
}
