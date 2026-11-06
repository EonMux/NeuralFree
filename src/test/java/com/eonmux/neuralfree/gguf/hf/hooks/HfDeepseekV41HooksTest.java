package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfDeepseekV41Hooks} on a synthetic DeepSeek-V4.1 checkpoint: the V4 fixup with the hash-layer default, the FP8
 * scale-format and gate-temperature checks, the Engram layout keys, the main-layer block count, the compressed-cache
 * metadata, the extended V4.1 name table, the no-MTP filter and the F32 first-layer gate.
 */
public class HfDeepseekV41HooksTest {

    private static final String BASE = "'architectures': ['DeepseekV41ForCausalLM'], 'model_type': 'deepseek_v41',"
            + " 'hidden_size': 64, 'intermediate_size': 128, 'moe_intermediate_size': 32, 'num_attention_heads': 4,"
            + " 'num_key_value_heads': 1, 'head_dim': 32, 'num_hidden_layers': 2, 'max_position_embeddings': 1024,"
            + " 'rms_norm_eps': 1e-06, 'vocab_size': 100, 'rope_theta': 10000.0, 'qk_rope_head_dim': 16,"
            + " 'q_lora_rank': 24, 'sliding_window': 128, 'n_routed_experts': 2, 'num_experts_per_tok': 1,"
            + " 'n_shared_experts': 1, 'routed_scaling_factor': 1.5, 'norm_topk_prob': true,"
            + " 'swiglu_limit': 10.0, 'index_n_heads': 2, 'index_head_dim': 16, 'index_topk': 8, 'o_groups': 2,"
            + " 'o_lora_rank': 16, 'compress_ratios': [0, 4], 'compress_rope_theta': 160000.0, 'hc_mult': 4,"
            + " 'hc_sinkhorn_iters': 20, 'hc_eps': 1e-06, 'scoring_func': 'sqrtsoftplus',"
            + " 'kv_source_layer_ids': [0, 1], 'index_source_layer_ids': [1, 1],"
            + " 'candidate_source_layer_id': 0, 'candidate_topk_blocks': 4, 'candidate_block_size': 64";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private static HfConfig v41(String extra) {
        return config("{" + BASE + extra + "}");
    }

    private static HfLoadContext load(HfConfig cfg, HfTensorIndex index) {
        return context(cfg, entry("DeepseekV41ForCausalLM"), index);
    }

    @Test
    public void theV4FixupDefaultsTheHashLayersAndKeepsTheMainBlockCount() {
        HfConfig fixed = HfDeepseekV41Hooks.configFixup(v41(""));
        assertEquals(0, fixed.requireInt("num_hash_layers"));
        assertEquals(2, HfDeepseekV41Hooks.blockCount(fixed));
        HfConfig withNextn = HfDeepseekV41Hooks.configFixup(v41(", 'num_nextn_predict_layers': 2"));
        assertEquals(2, HfDeepseekV41Hooks.blockCount(withNextn));
        assertEquals("INDEXER_ATTN_K", HfDeepseekV41Hooks.TABLE.map("layers.1.attn.indexer.wk.weight", 1).tensor());
    }

    @Test
    public void anotherFp8ScaleFormatOrGateTemperatureIsRefusedByName() {
        String format = ", 'quantization_config': {'quant_method': 'fp8', 'scale_fmt': 'e4m3'"
                + ", 'weight_block_size': [32, 32]}";
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfDeepseekV41Hooks.configFixup(v41(format)));
        assertTrue(e.getMessage(), e.getMessage().contains("scale_fmt"));
        GgufFormatException gate = assertThrows(GgufFormatException.class,
                () -> HfDeepseekV41Hooks.configFixup(v41(", 'gate_temp': 2.0")));
        assertTrue(gate.getMessage(), gate.getMessage().contains("gate_temp"));
        assertEquals(2, HfDeepseekV41Hooks.configFixup(v41(
                ", 'quantization_config': {'quant_method': 'fp8', 'scale_fmt': 'ue8m0'"
                        + ", 'weight_block_size': [32, 32]}")).requireInt("num_hidden_layers"));
    }

    @Test
    public void anEngramCheckpointKeepsItsLayersAndNeedsTheirLayoutKeys() {
        HfConfig fixed = HfDeepseekV41Hooks.configFixup(v41(", 'engram_layer_ids': [1]"));
        assertEquals(Collections.singletonList(1), fixed.findIntList("engram_layer_ids").get());
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfEngramLayout.of(fixed));
        assertTrue(e.getMessage(), e.getMessage().contains("engram_max_ngram_size"));
    }

    @Test
    public void theCompressedCacheSourcesAndTheCandidateSelectionAreWritten() throws IOException {
        HfLoadContext ctx = load(v41(""), HfTensorIndex.of(Map.of()));
        HfMetadataBuilder b = new HfMetadataBuilder(ctx.arch().ggufName());
        entry("DeepseekV41ForCausalLM").meta().apply(ctx, b, null);
        HfMetadata m = b.build();
        assertEquals(Arrays.asList(0, 1), m.get("deepseek41.attention.kv_source_layers"));
        assertEquals(Arrays.asList(1, 1), m.get("deepseek41.attention.indexer.source_layers"));
        assertEquals(0, m.get("deepseek41.attention.indexer.candidate_source_layer"));
        assertEquals(4, m.get("deepseek41.attention.indexer.candidate_top_k_blocks"));
        assertEquals(64, m.get("deepseek41.attention.indexer.candidate_block_size"));
        assertEquals(0, m.get("deepseek41.hash_layer_count"));
        assertNull(m.get("deepseek41.nextn_predict_layers"));
    }

    @Test
    public void aCheckpointWithoutCandidatesOmitsTheCandidateKeys() throws IOException {
        HfConfig cfg = v41("").withFlat(Collections.<String, Object>singletonMap("candidate_source_layer_id", -1));
        HfLoadContext ctx = load(cfg, HfTensorIndex.of(Map.of()));
        HfMetadataBuilder b = new HfMetadataBuilder(ctx.arch().ggufName());
        entry("DeepseekV41ForCausalLM").meta().apply(ctx, b, null);
        assertNull(b.build().get("deepseek41.attention.indexer.candidate_source_layer"));
    }

    @Test
    public void theV41NameTableAddsTheIndexerKeyAndKeyNorm() throws IOException {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        HfTensorIndex index = ckpt
                .f32("layers.0.attn.indexer.wk.weight", new long[]{4, 2}, seq(8))
                .f32("layers.0.attn.indexer.k_norm.weight", new long[]{4}, seq(4))
                .index();
        HfLoadContext ctx = load(v41(""), index);
        Sink out = new Sink();
        HfDeepseekV41Hooks.tensors(ctx, "layers.0.attn.indexer.wk.weight",
                index.get("layers.0.attn.indexer.wk.weight"), out);
        HfDeepseekV41Hooks.tensors(ctx, "layers.0.attn.indexer.k_norm.weight",
                index.get("layers.0.attn.indexer.k_norm.weight"), out);
        assertEquals(4, out.get("blk.0.indexer.attn_k.weight").shape()[0]);
        assertEquals(2, out.get("blk.0.indexer.attn_k.weight").shape()[1]);
        assertEquals(4, out.get("blk.0.indexer.k_norm.weight").shape()[0]);
    }

    @Test
    public void theFilterDropsVisionAndEveryMtpTensor() {
        HfArchEntry e = entry("DeepseekV41ForCausalLM");
        assertNull(HfDeepseekV41Hooks.filter().filter("mtp.0.e_proj.weight", n -> n));
        assertNull(HfDeepseekV41Hooks.filter().filter("aligner.linear.weight", n -> n));
        assertNull(HfDeepseekV41Hooks.filter().filter("image_newline", n -> n));
        assertEquals("layers.0.attn.wq_a.weight",
                HfDeepseekV41Hooks.filter().filter("layers.0.attn.wq_a.weight", n -> n));
        assertTrue(e.hfNames().contains("DeepseekV41ForCausalLM"));
    }

    @Test
    public void theFirstLayerGateStaysF32() {
        assertEquals(GGMLType.F32, HfDeepseekV41Hooks.forceType("blk.0.ffn_gate_inp.weight", 2, null,
                com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16));
        assertEquals(GGMLType.F32, HfDeepseekV41Hooks.forceType("blk.1.engram_gate.weight", 2, null,
                com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16));
        assertNull(HfDeepseekV41Hooks.forceType("blk.0.attn_q_a.weight", 2, ElementFormat.of(GGMLType.F16),
                com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType.F16));
    }

    @Test
    public void theEntryResolvesToDeepseekV41() {
        assertEquals(Arch.DEEPSEEK41, entry("DeepseekV41ForCausalLM").arch()
                .resolve(v41(""), HfTensorIndex.of(Map.of())));
    }
}
