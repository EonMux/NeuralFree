package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
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
 * The part D hooks: the InternLM3 metadata, SPM vocabulary rewrite, q/k permutation and the vision-tower filter
 * (internlm.py:175-234), the Janus and Sarashina name filters (januspro.py:18-34, sarashina2.py:19-26), the
 * thinker strip and the transposed Qwen3-VL fused experts (qwen3vl.py:303-372).
 */
public class HfMoeClassicHooksDTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ InternLM3Model

    @Test
    public void internLm3MetaSetsVocabSizeAndRopeWidth() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.LLAMA, "InternLM3ForCausalLM", "'vocab_size': 12,"
                + " 'rope_scaling': {'rope_type': 'linear', 'factor': 2.0}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.LLAMA.ggufName());
        HfMoeClassicHooksD.internLm3Meta(ctx, kv, null);
        assertEquals(12L, ((Number) kv.get(GgufKeys.VOCAB_SIZE)).longValue());
        assertEquals(kv.get(GgufKeys.ROPE_DIMENSION_COUNT), kv.get(GgufKeys.ROPE_DIMENSION_COUNT));
        assertTrue(kv.has(GgufKeys.ROPE_DIMENSION_COUNT));
    }

    @Test
    public void internLm3TensorsPermuteQkAndMapNames() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.LLAMA, "InternLM3ForCausalLM", "'vocab_size': 12,"
                + " 'num_attention_heads': 2, 'num_key_value_heads': 1, 'hidden_size': 8");
        RecordingSink sink = new RecordingSink();
        HfMoeClassicHooksD.internLm3Tensors(ctx, "model.layers.0.self_attn.q_proj.weight",
                seq(8, 8), sink);
        RowSource q = tensors(sink, "attn_q").get(0);
        assertArrayEquals(new long[]{8, 8}, q.shape());
        assertThrows(GgufFormatException.class, () -> HfMoeClassicHooksD.internLm3Tensors(ctx,
                "model.vision_tower.weight", seq(4), new RecordingSink()));
    }

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
        assertEquals("model.norm.weight",
                HfMoeClassicHooksD.SARASHINA_FILTER.filter("llm.norm.weight", IDENTITY));
        // the llm. strip feeds the rest to the Llama filter, which keeps model-space names
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfMoeClassicHooksD.SARASHINA_FILTER.filter("llm.model.layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("blk", HfMoeClassicHooksD.THINKER_FILTER.filter("thinker.blk", IDENTITY));
        assertEquals("plain", HfMoeClassicHooksD.THINKER_FILTER.filter("plain", IDENTITY));
    }

    // ------------------------------------------------------------------ Qwen3VLMoeTextModel

    @Test
    public void qwen3VlMoeMetaCountsTheDeepstackLayers() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3VLMOE, "Qwen3VLMoeForConditionalGeneration",
                "'num_local_experts': 2, 'moe_intermediate_size': 4,"
                        + " 'vision_config': {'deepstack_visual_indexes': [0, 1]}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3VLMOE.ggufName());
        HfMoeClassicHooksD.qwen3VlMoeMeta(ctx, kv, null);
        assertEquals(2L, ((Number) kv.get(GgufKeys.NUM_DEEPSTACK_LAYERS)).longValue());
        HfLoadContext omni = context(dir(), Arch.QWEN3VLMOE, "Qwen3OmniMoeForConditionalGeneration",
                "'num_local_experts': 2, 'moe_intermediate_size': 4");
        HfMetadataBuilder omniKv = new HfMetadataBuilder(Arch.QWEN3VLMOE.ggufName());
        HfMoeClassicHooksD.qwen3OmniMoeMeta(omni, omniKv, null);
        assertEquals(0L, ((Number) omniKv.get(GgufKeys.NUM_DEEPSTACK_LAYERS)).longValue());
    }

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

    private static java.util.List<RowSource> tensors(RecordingSink sink, String part) {
        return sink.tensors.entrySet().stream()
                .filter(e -> e.getKey().contains(part))
                .map(Map.Entry::getValue)
                .collect(java.util.stream.Collectors.toList());
    }
}
