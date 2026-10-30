package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.seq;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfQwenHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The long-tail hooks: the GroveMoe chunk experts (grovemoe.py:15-109), the K2Horizon value experts and router bias
 * (k2_horizon.py:17-105), the Henry uniform per-layer lists (hunyuan.py:75-110), the HYV3 NextN filter and block
 * count (hunyuan.py:405-430, 374-381), the InternLM3/Janus/Sarashina/Thinker name filters and the transposed
 * Qwen3-VL fused experts (qwen3vl.py:322-350).
 */
public class HfMoeClassicHooksCTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ GroveMoeModel

    @Test
    public void groveMoeDropsTheExpertBiasAndStacksTheChunkExperts() throws IOException {
        int nLocal = 4;
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < nLocal; x++) {
            index.put("model.layers.0.mlp.experts." + x + ".down_proj.weight", filled(x, 8, 2));
            index.put("model.layers.0.mlp.experts." + x + ".gate_proj.weight", filled(x, 2, 8));
            index.put("model.layers.0.mlp.experts." + x + ".up_proj.weight", filled(x, 2, 8));
        }
        // the chunked experts hold the other half of each expert's width
        for (int c = 0; c < nLocal / 2; c++) {
            index.put("model.layers.0.mlp.chunk_experts." + c + ".down_proj.weight", filled(c + 9, 8, 2));
            index.put("model.layers.0.mlp.chunk_experts." + c + ".gate_proj.weight", filled(c + 9, 2, 8));
            index.put("model.layers.0.mlp.chunk_experts." + c + ".up_proj.weight", filled(c + 9, 2, 8));
        }
        index.put("model.layers.0.mlp.expert_bias", seq(4));
        HfLoadContext ctx = context(dir(), Arch.GROVEMOE, "GroveMoeForCausalLM", "'num_local_experts': 4", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfMoeClassicHooksC.groveMoeTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        HfMoeClassicHooksC.groveMoeExtra(ctx, sink);
        Map<String, String> dropped = sink.dropped;
        assertTrue(dropped.containsKey("model.layers.0.mlp.expert_bias"));
        assertEquals(6, countKey(sink, "ffn_"));
    }
    @Test
    public void groveMoeLeftoversFailAndTheChunkCountNeedsAnEvenSplit() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        for (int x = 0; x < 4; x++) {
            for (String part : new String[]{"down_proj", "gate_proj", "up_proj"}) {
                index.put("model.layers.0.mlp.experts." + x + "." + part + ".weight", seq(2, 8));
            }
        }
        HfLoadContext ctx = context(dir(), Arch.GROVEMOE, "GroveMoeForCausalLM", "'num_local_experts': 4", index, false);
        RecordingSink sink = new RecordingSink();
        // one member offered of the twelve planned: the leftovers fail the extra check
        HfMoeClassicHooksC.groveMoeTensors(ctx, "model.layers.0.mlp.experts.0.down_proj.weight",
                ctx.index().get("model.layers.0.mlp.experts.0.down_proj.weight"), sink);
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksC.groveMoeExtra(ctx, sink));
        HfLoadContext odd = context(dir(), Arch.GROVEMOE, "GroveMoeForCausalLM", "'num_local_experts': 3",
                Collections.singletonMap("model.layers.0.mlp.chunk_experts.0.gate_proj.weight", seq(2, 8)), false);
        assertThrows(GgufFormatException.class,
                () -> HfMoeClassicHooksC.groveMoeTensors(odd, "model.layers.0.mlp.chunk_experts.0.gate_proj.weight",
                        seq(2, 8), new RecordingSink()));
    }

    // ------------------------------------------------------------------ K2HorizonModel

    @Test
    public void k2HorizonRouterBiasAndValueExpertsStack() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.mlp.gate.bias", filled(4, 2, 1));
        for (int x = 0; x < 2; x++) {
            index.put("model.layers.0.self_attn.v_experts." + x + ".weight", filled(x, 8, 8));
            index.put("model.layers.1.self_attn.v_experts." + x + ".weight", filled(x, 8, 8));
        }
        HfLoadContext ctx = context(dir(), Arch.K2_HORIZON, "K2HorizonForCausalLM", "'mova_num_experts': 2,"
                + " 'mova_num_experts_per_tok': 1", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfMoeClassicHooksC.k2HorizonTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        HfMoeClassicHooksC.k2HorizonExtra(ctx, sink);
        assertTrue(countKey(sink, "exp_probs_b") == 1 || countKey(sink, "exp_probs_b") == 2);
        // the GGUF name of ATTN_V_EXP is attn_v_exps (constants.py:2177)
        assertEquals(2, countKey(sink, "attn_v_exps"));
    }

    @Test
    public void k2HorizonLeftoverValueExpertsFail() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.1.mlp.gate.bias", filled(4, 2, 1));
        index.put("model.layers.1.self_attn.v_experts.0.weight", filled(0, 8, 8));
        HfLoadContext ctx = context(dir(), Arch.K2_HORIZON, "K2HorizonForCausalLM", "'mova_num_experts': 2,"
                + " 'mova_num_experts_per_tok': 1", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfMoeClassicHooksC.k2HorizonTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksC.k2HorizonExtra(ctx, sink));
    }

    // ------------------------------------------------------------------ HunYuanMoEModel

    @Test
    public void henryPerLayerListsMustBeUniform() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.HUNYUAN_MOE, "HunYuanMoEV1ForCausalLM", "'intermediate_size': 32,"
                + " 'moe_intermediate_size': [4, 4], 'moe_topk': [2, 2], 'num_shared_expert': [1, 1]");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.HUNYUAN_MOE.ggufName());
        HfMoeClassicHooksC.hunYuanMoeMeta(ctx, kv, null);
        HfLoadContext ragged = context(dir(), Arch.HUNYUAN_MOE, "HunYuanMoEV1ForCausalLM", "'intermediate_size': 32,"
                + " 'moe_intermediate_size': [4, 5], 'moe_topk': [2, 2], 'num_shared_expert': [1, 1]");
        assertThrows(GgufFormatException.class,
                () -> HfMoeClassicHooksC.hunYuanMoeMeta(ragged, new HfMetadataBuilder(Arch.HUNYUAN_MOE.ggufName()),
                        null));
    }

    @Test
    public void henryTiedOutputWeightIsDropped() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("lm_head.weight", seq(4, 8));
        index.put("model.embed_tokens.weight", seq(4, 8));
        for (String part : new String[]{"down_proj", "gate_proj", "up_proj"}) {
            index.put("model.layers.0.mlp.experts.0." + part + ".weight", seq(8, 2));
        }
        HfLoadContext ctx = context(dir(), Arch.HUNYUAN_MOE, "HunYuanMoEV1ForCausalLM", "'tie_word_embeddings': true,"
                + " 'num_local_experts': 1", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : ctx.index().names()) {
            if (!sink.isConsumed(name)) {
                HfMoeClassicHooksC.hunYuanMoeTensors(ctx, name, ctx.index().get(name), sink);
            }
        }
        assertTrue(sink.dropped.containsKey("lm_head.weight"));
    }

    // ------------------------------------------------------------------ HYV3Model

    private static final String HYV3_CONFIG = "'num_local_experts': 2, 'moe_intermediate_size': 4,"
            + " 'num_nextn_predict_layers': 1";

    @Test
    public void hyV3FilterDropsAndRenamesTheMtpLayers() throws IOException {
        HfLoadContext keep = context(dir(), Arch.HY_V3, "HYV3ForCausalLM", HYV3_CONFIG, Collections.emptyMap(), true);
        HfLoadContext drop = context(dir(), Arch.HY_V3, "HYV3ForCausalLM", HYV3_CONFIG, Collections.emptyMap(), false);
        assertNull(HfMoeClassicHooksC.hyV3Filter(drop).filter("model.layers.2.final_layernorm.weight", IDENTITY));
        // with MTP kept, the reference drops only under no_mtp — the MTP block's tensors stay
        assertEquals("model.layers.2.mlp.experts.0.up_proj.weight",
                HfMoeClassicHooksC.hyV3Filter(keep).filter("model.layers.2.mlp.experts.0.up_proj.weight", IDENTITY));
        assertEquals("model.layers.2.shared_head.norm.weight",
                HfMoeClassicHooksC.hyV3Filter(keep).filter("model.layers.2.final_layernorm.weight", IDENTITY));
        // a main-block norm stays untouched
        assertEquals("model.layers.1.final_layernorm.weight",
                HfMoeClassicHooksC.hyV3Filter(drop).filter("model.layers.1.final_layernorm.weight", IDENTITY));
        assertEquals(3, HfMoeClassicHooksC.hyV3BlockCount(keep));
        assertEquals(2, HfMoeClassicHooksC.hyV3BlockCount(drop));
    }

    // ------------------------------------------------------------------ the name filters

    @Test
    public void internLm3JanusSarashinaAndThinkerFilters() {
        assertNull(HfMoeClassicHooksD.INTERN_LM3_FILTER.filter("vision_model.blocks.0.attn.weight", IDENTITY));
        assertNull(HfMoeClassicHooksD.INTERN_LM3_FILTER.filter("mlp.fc1.weight", IDENTITY));
        assertEquals("model.embed_tokens.weight",
                HfMoeClassicHooksD.INTERN_LM3_FILTER.filter("model.embed_tokens.weight", IDENTITY));
        for (String drop : Arrays.asList("model.vision_model.", "model.aligner.", "model.vqmodel.",
                "model.generation_embeddings.", "model.generation_aligner.", "model.generation_head.")) {
            assertNull(HfMoeClassicHooksD.JANUS_FILTER.filter(drop + "fc.weight", IDENTITY));
        }
        assertNull(HfMoeClassicHooksD.SARASHINA_FILTER.filter("norm.weight", IDENTITY));
        // the filter returns the stripped checkpoint name; the OUTPUT_NORM global row maps
        // norm.weight to model.norm.weight at planning (tensor_patterns.tsv row 63)
        assertEquals("norm.weight", HfMoeClassicHooksD.SARASHINA_FILTER.filter("llm.norm.weight", IDENTITY));
        // the llm. strip feeds the rest to the Llama filter, which keeps model-space names
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfMoeClassicHooksD.SARASHINA_FILTER.filter("llm.model.layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("blk", HfMoeClassicHooksD.THINKER_FILTER.filter("thinker.blk", IDENTITY));
        assertEquals("plain", HfMoeClassicHooksD.THINKER_FILTER.filter("plain", IDENTITY));
    }

    // ------------------------------------------------------------------ Qwen3VLMoeTextModel

    @Test
    public void qwen3VlFusedExpertsAreTransposedBeforeTheQwen2MoeRoute() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.mlp.experts.gate_up_proj.weight", seq(2, 4, 4));
        index.put("model.layers.0.mlp.experts.down_proj.weight", seq(2, 2, 8));
        HfLoadContext ctx = context(dir(), Arch.QWEN3VLMOE, "Qwen3VLMoeForConditionalGeneration", index == null
                ? "" : "'num_local_experts': 2, 'moe_intermediate_size': 4", index, false);
        RecordingSink sink = new RecordingSink();
        for (String name : index.keySet()) {
            HfMoeClassicHooksD.qwen3VlMoeTensors(ctx, name, ctx.index().get(name), sink);
        }
        RowSource down = tensors(sink, "ffn_down_exps").get(0);
        assertArrayEquals(new long[]{2, 8, 2}, down.shape());
        RowSource gate = tensors(sink, "ffn_gate_exps").get(0);
        assertArrayEquals(new long[]{2, 2, 4}, gate.shape());
        assertArrayEquals(floats(0, 4, 8, 12, 1, 5, 9, 13, 16, 20, 24, 28, 17, 21, 25, 29), values(gate), 0f);
        assertArrayEquals(floats(2, 6, 10, 14, 3, 7, 11, 15, 18, 22, 26, 30, 19, 23, 27, 31),
                values(tensors(sink, "ffn_up_exps").get(0)), 0f);
    }

    @Test
    public void qwen3VlAnOddFusedWidthIsRejected() throws IOException {
        Map<String, RowSource> index = Collections.singletonMap("model.layers.0.mlp.experts.gate_up_proj.weight",
                seq(2, 4, 3));
        HfLoadContext ctx = context(dir(), Arch.QWEN3VLMOE, "Qwen3VLMoeForConditionalGeneration",
                "'num_local_experts': 2", index, false);
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksD.qwen3VlMoeTensors(ctx,
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                ctx.index().get("model.layers.0.mlp.experts.gate_up_proj.weight"), new RecordingSink()));
    }

    // ------------------------------------------------------------------ helpers

    private static long countKey(RecordingSink sink, String part) {
        return sink.tensors.keySet().stream().filter(k -> k.contains(part)).count();
    }

    private static java.util.List<RowSource> tensors(RecordingSink sink, String part) {
        return sink.tensors.entrySet().stream()
                .filter(e -> e.getKey().contains(part))
                .map(Map.Entry::getValue)
                .collect(java.util.stream.Collectors.toList());
    }

    private static RowSource filled(int value, long rows, long cols) {
        float[] v = new float[(int) (rows * cols)];
        java.util.Arrays.fill(v, value);
        return com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.fromFloats(new long[]{rows, cols}, v);
    }
}
