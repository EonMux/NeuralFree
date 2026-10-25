package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfSsmHybridHooksA} on synthetic GraniteHybrid, NemotronH and Puzzle checkpoints: the attention/SSM layer
 * selection rules, the GRANITE/GRANITE_MOE fallback of the no-SSM builds, the pad-8 fixup, the SSM/Granite routing
 * of the block tensors, the Nemotron pattern fixup with the folded MTP block, and the Puzzle per-block
 * configurations.
 */
public class HfSsmHybridHooksATest {

    /** A two-block granite-hybrid base with the mamba head-dim first. */
    private static final String GRANITE_BASE = "'mamba_head_dim': 8, 'num_hidden_layers': 2, 'vocab_size': 16,"
            + " 'num_key_value_heads': 2, 'n_head_kv': 2, 'expand': 2, 'n_groups': 2, 'head_dim': 2,"
            + " 'conv_kernel': 3, 'state_size': 4, 'n_heads': 4, 'num_attention_heads': 2,"
            + " 'intermediate_size': 6, 'max_position_embeddings': 16, 'rms_norm_eps': 1e-05,"
            + " 'rope_theta': 10000.0";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    @Test
    public void theLayerSelectionRulesRouteTheHybridBlockTensors() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        // layer 0: mamba (conv + A_log), layer 1: attention (q_proj)
        HfTensorIndex index = ckpt
                .f32("backbone.layers.0.mixer.conv1d.weight", new long[]{4, 1, 3}, seq(12))
                .f32("backbone.layers.0.mixer.A_log", new long[]{2}, new float[]{0f, 0f})
                .f32("backbone.layers.1.mixer.q_proj.weight", new long[]{2, 8}, seq(16))
                .f32("backbone.layers.1.block_sparse_moe.input_linear.weight", new long[]{2, 4, 8}, seq(64))
                .f32("backbone.final_layer_norm.weight", new long[]{8}, seq(8)).index();
        String json = "{'architectures': ['GraniteMoeHybridForCausalLM'], " + GRANITE_BASE
                + ", 'layer_types': ['mamba', 'attention'], 'intermediate_size': 4}";
        HfConfig cfg = HfLlamaHooksTest.config(json);
        HfLoadContext ctx = context(cfg, entry("GraniteMoeHybridForCausalLM"), index);
        assertEquals(Arch.GRANITE_HYBRID, ctx.arch());
        Sink out = plan(ctx);
        assertTrue(out.added.containsKey("blk.0.ssm_conv1d.weight"));
        assertTrue(out.added.containsKey("blk.1.attn_q.weight"));
        assertArrayEquals(new long[]{2, 2, 8}, out.get("blk.1.ffn_gate_exps.weight").shape());
        assertTrue(out.added.containsKey("output_norm.weight"));
    }

    @Test
    public void thePeriodOffsetIndicesAndLayerTypesSelectTheSameAttentionLayers() {
        String json = "{'architectures': ['GraniteMoeHybridForCausalLM'], " + GRANITE_BASE
                + ", 'layer_types': ['mamba', 'mamba', 'attention', 'mamba'], 'num_hidden_layers': 4}";
        HfConfig cfg = HfLlamaHooksTest.config(json);
        assertEquals(Arrays.asList(2), HfSsmHybridHooksA.graniteAttnLayers(cfg, 4));
        String indices = "{'architectures': ['GraniteMoeHybridForCausalLM'], " + GRANITE_BASE
                + ", 'attn_layer_indices': [2, 3], 'num_hidden_layers': 4}";
        HfConfig byIndex = HfLlamaHooksTest.config(indices);
        assertEquals(Arrays.asList(2, 3), HfSsmHybridHooksA.graniteAttnLayers(byIndex, 4));
        String period = "{'architectures': ['GraniteMoeHybridForCausalLM'], " + GRANITE_BASE
                + ", 'attn_layer_period': 3, 'attn_layer_offset': 2, 'num_hidden_layers': 4}";
        HfConfig byPeriod = HfLlamaHooksTest.config(period);
        assertEquals(Arrays.asList(2), HfSsmHybridHooksA.graniteAttnLayers(byPeriod, 4));
    }

    @Test
    public void thePeriodOffsetIndicesAndLayerTypesSelectTheSameNonHybridArch() {
        // all-attention build: the encoder falls back to granite / granitemoe with pad 1
        String json = "{'architectures': ['GraniteMoeHybridForCausalLM'], " + GRANITE_BASE
                + ", 'layer_types': ['attention', 'attention']}";
        HfConfig cfg = HfLlamaHooksTest.config(json);
        assertEquals(Arch.GRANITE, HfSsmHybridHooksA.graniteHybridArch(cfg, null));
        assertEquals(1, HfSsmHybridHooksA.graniteHybridFixup(cfg).findInt("pad_vocab_size_multiple"));
        HfConfig moe = HfLlamaHooksTest.config("{'architectures': ['GraniteMoeHybridForCausalLM'], "
                + GRANITE_BASE + ", 'layer_types': ['attention', 'attention'], 'num_experts_per_tok': 1}");
        assertEquals(Arch.GRANITE_MOE, HfSsmHybridHooksA.graniteHybridArch(moe, null));
        HfConfig hybrid = HfLlamaHooksTest.config("{'architectures': ['BambaForCausalLM'], " + GRANITE_BASE
                + ", 'layer_types': ['mamba', 'attention']}");
        assertEquals(Arch.GRANITE_HYBRID, HfSsmHybridHooksA.graniteHybridArch(hybrid, null));
        assertEquals(8, HfSsmHybridHooksA.graniteHybridFixup(hybrid).findInt("pad_vocab_size_multiple"));
    }

    @Test
    public void nemotronHFixesTheLayerCountFromThePatternAndFoldsTheMtpBlock() throws IOException {
        String json = "{'architectures': ['NemotronHForCausalLM'], " + GRANITE_BASE.replace("'vocab_size': 16",
                "'vocab_size': 16") + ", 'layers_block_type': ['mamba', 'attention'], 'num_hidden_layers': 7,"
                + " 'num_experts_per_tok': 1, 'num_nextn_predict_layers': 1, 'moe_intermediate_size': 6,"
                + " 'moe_shared_expert_intermediate_size': 6, 'n_routed_experts': 2, 'n_shared_experts': 1,"
                + " 'norm_topk_prob': false, 'routed_scaling_factor': 1.0, 'n_group': 1}";
        HfConfig cfg = HfLlamaHooksTest.config(json);
        HfConfig fixed = HfSsmHybridHooksA.nemotronHFixup(cfg);
        assertEquals(2, fixed.blockCount());
        assertEquals(Arch.NEMOTRON_H_MOE, HfSsmHybridHooksA.nemotronHArch(fixed, null));
        assertTrue(HfSsmHybridHooksA.nemotronMtpFolded(fixed));
        // MoE block experts of the Nemotron trunk stack down/up per block
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("backbone.layers.0.mixer.conv1d.weight", new long[]{4, 1, 3}, seq(12))
                .f32("backbone.layers.0.mixer.experts.0.down_proj.weight", new long[]{3, 4}, seq(12))
                .f32("backbone.layers.0.mixer.experts.0.up_proj.weight", new long[]{3, 4}, seq(12))
                .f32("backbone.layers.0.mixer.experts.1.down_proj.weight", new long[]{3, 4}, seq(12))
                .f32("backbone.layers.0.mixer.experts.1.up_proj.weight", new long[]{3, 4}, seq(12)).index();
        HfLoadContext ctx = context(fixed, entry("NemotronHForCausalLM"), index);
        assertEquals(3, HfSsmHybridHooksA.nemotronHBlockCount(ctx));
        Sink out = plan(context(fixed, entry("NemotronHForCausalLM"), index));
        assertArrayEquals(new long[]{4, 3}, out.get("blk.0.ssm_conv1d.weight").shape());
        assertArrayEquals(new long[]{2, 3, 4}, out.get("blk.0.ffn_down_exps.weight").shape());
    }

    @Test
    public void puzzleTakesItsPatternAndSizesFromThePerBlockConfigurations() throws IOException {
        String bc = "'block_configs': ["
                + "{'block_type': 'mamba'}, {'block_type': 'attention'}, {'block_type': 'moe',"
                + " 'moe_intermediate_size': 6, 'num_experts_per_tok': 1}],"
                + " 'moe_shared_expert_intermediate_size': 6, 'n_routed_experts': 2, 'n_shared_experts': 1,"
                + " 'norm_topk_prob': false, 'routed_scaling_factor': 1.0, 'n_group': 1, 'moe_latent_size': 8";
        String json = "{'architectures': ['NemotronHPuzzleForCausalLM'], " + GRANITE_BASE + ", " + bc
                + ", 'num_hidden_layers': 20}";
        HfConfig cfg = HfLlamaHooksTest.config(json);
        assertEquals(Arch.NEMOTRON_H_MOE, HfSsmHybridHooksA.nemotronHPuzzleArch(cfg, null));
        HfConfig fixed = HfSsmHybridHooksA.nemotronHPuzzleFixup(cfg);
        assertEquals(3, fixed.blockCount());
        HfMetadataBuilder kv = new HfMetadataBuilder("nemotron_h_moe");
        HfLoadContext ctx = context(fixed, entry("NemotronHPuzzleForCausalLM"), HfTensorIndex.of(Map.of()));
        entry("NemotronHPuzzleForCausalLM").meta().apply(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(Arrays.asList(6L, 0L, 6L), m.get("nemotron_h_moe.feed_forward_length"));
        Integer latent = (Integer) m.get("nemotron_h_moe.moe_latent_size");
        assertEquals(8, latent.intValue());
        // the official checkpoint's model.* trunk names rename to backbone.*
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("model.layers.0.mixer.conv1d.weight", new long[]{4, 1, 3}, seq(12))
                .f32("model.layers.0.mixer.A_log", new long[]{2}, new float[]{0f, 1f})
                .f32("model.layers.0.mixer.D", new long[]{2}, new float[]{1f, 2f})
                .f32("model.layers.0.mixer.norm.weight", new long[]{8}, seq(8)).index();
        Sink out = plan(context(fixed, entry("NemotronHPuzzleForCausalLM"), index));
        assertArrayEquals(new long[]{4, 3}, out.get("blk.0.ssm_conv1d.weight").shape());
        assertArrayEquals(new long[]{2, 1}, out.get("blk.0.ssm_a.weight").shape());
        assertArrayEquals(new long[]{2, 1}, out.get("blk.0.ssm_d.weight").shape());
        assertArrayEquals(new long[]{2, 4}, out.get("blk.0.ssm_norm.weight").shape());
        assertEquals(Arrays.asList(0, 2, 0), kv.build().get("nemotron_h_moe.attention.head_count_kv"));
    }
}
