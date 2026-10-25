package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.Scaled;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.iota;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.json;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.planAll;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfGemma4Hooks} against {@code Gemma4Model} (LC/conversion/gemma.py:634-810): the per-layer metadata, the
 * proportional RoPE factors, the name filter, the router and per-expert scales on both the NVFP4 path (folded into
 * {@code weight_scale_2}, experts stacked) and the unquantized path ({@code ffn_down_exps.scale}), and the vocabulary.
 */
public class HfGemma4HooksTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String ROPE = "'rope_parameters': {'full_attention': {'rope_type': 'proportional',"
            + " 'partial_rotary_factor': 0.25, 'rope_theta': 1000000.0}, 'sliding_attention': {'rope_type': 'default',"
            + " 'rope_theta': 10000.0}}";

    private static final String GEMMA4 = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'head_dim': 4, 'num_hidden_layers': 4, 'num_kv_shared_layers': 2,"
            + " 'max_position_embeddings': 1024, 'rms_norm_eps': 1e-06, 'sliding_window': 512,"
            + " 'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'],"
            + " 'num_experts': 2, 'top_k_experts': 1, 'expert_intermediate_size': 4, " + ROPE;

    private static HfLoadContext gemma4(String members, Map<String, RowSource> tensors) {
        return context(Arch.GEMMA4, "Gemma4ForCausalLM", members, tensors);
    }

    private static HfMetadata meta(HfLoadContext ctx) {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        HfGemma4Hooks.meta(ctx, kv, null);
        return kv.build();
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metadataSplitsFullAndSlidingWindowLayers() {
        HfMetadata m = meta(gemma4(GEMMA4 + ", 'global_head_dim': 512, 'num_global_key_value_heads': 2,"
                + " 'use_double_wide_mlp': true", Map.of()));
        assertEquals(2, m.get("gemma4.attention.shared_kv_layers"));
        assertEquals(0, m.get("gemma4.embedding_length_per_layer_input"));
        assertEquals(Arrays.asList(true, true, true, false), m.get("gemma4.attention.sliding_window_pattern"));
        assertEquals(512, m.get("gemma4.attention.key_length"));
        assertEquals(512, m.get("gemma4.attention.value_length"));
        assertEquals(4, m.get("gemma4.attention.key_length_swa"));
        assertEquals(4, m.get("gemma4.attention.value_length_swa"));
        assertEquals(4, m.get("gemma4.expert_feed_forward_length"));
        assertEquals("KV-shared layers get the double-wide FFN", Arrays.asList(16, 16, 32, 32),
                m.get("gemma4.feed_forward_length"));
        assertEquals(Arrays.asList(1, 1, 1, 2), m.get("gemma4.attention.head_count_kv"));
        assertEquals(512, m.get("gemma4.rope.dimension_count"));
        assertEquals(4, m.get("gemma4.rope.dimension_count_swa"));
        assertEquals(1e6f, m.get("gemma4.rope.freq_base"));
        assertEquals(10000f, m.get("gemma4.rope.freq_base_swa"));
        assertEquals(512, m.get("gemma4.attention.sliding_window"));
    }

    @Test
    public void perLayerConfigSuppliesTheFullAttentionWidthAndKvHeads() {
        HfMetadata m = meta(gemma4(GEMMA4 + ", 'per_layer_config': {'0': {'head_dim': 4}, '3': {'head_dim': 256,"
                + " 'num_key_value_heads': 4}}", Map.of()));
        assertEquals(256, m.get("gemma4.attention.key_length"));
        assertEquals(Arrays.asList(1, 1, 1, 4), m.get("gemma4.attention.head_count_kv"));
        assertEquals("without use_double_wide_mlp the FFN width stays scalar", 16,
                m.get("gemma4.feed_forward_length"));
    }

    @Test
    public void metadataNeedsAFullAttentionWidth() {
        HfLoadContext ctx = gemma4(GEMMA4, Map.of());
        assertThrows(GgufFormatException.class, () -> meta(ctx));
    }

    // ------------------------------------------------------------------ extra tensors

    @Test
    public void proportionalRopeFreqsLeaveTheUnrotatedPairsFixed() throws IOException {
        RecordingSink out = new RecordingSink();
        HfGemma4Hooks.extra(gemma4(GEMMA4 + ", 'global_head_dim': 512", Map.of()), out);
        float[] f = values(out.tensors.get("rope_freqs.weight"));
        assertEquals(256, f.length);
        for (int i = 0; i < f.length; i++) {
            assertEquals("pair " + i, i < 64 ? 1f : 1e30f, f[i], 0f);
        }
    }

    @Test
    public void ropeFreqsNeedProportionalFullAttentionRope() {
        HfLoadContext ctx = gemma4(GEMMA4.replace("'proportional'", "'default'") + ", 'global_head_dim': 512",
                Map.of());
        assertThrows(GgufFormatException.class, () -> HfGemma4Hooks.extra(ctx, new RecordingSink()));
    }

    // ------------------------------------------------------------------ names

    @Test
    public void filterAddsTheWeightSuffix() {
        List<String> seen = new ArrayList<>();
        java.util.function.UnaryOperator<String> base = n -> {
            seen.add(n);
            return n;
        };
        assertEquals("model.layers.0.self_attn.per_dim_scale.weight",
                HfGemma4Hooks.filter("model.layers.0.self_attn.per_dim_scale", base));
        assertEquals("model.layers.0.layer_scalar.weight", HfGemma4Hooks.filter("model.layers.0.layer_scalar", base));
        assertEquals("model.layers.0.experts.gate_up_proj.weight",
                HfGemma4Hooks.filter("model.layers.0.experts.gate_up_proj", base));
        assertEquals("model.layers.0.experts.0.down_proj.weight_scale_2",
                HfGemma4Hooks.filter("model.layers.0.experts.0.down_proj.weight_scale_2", base));
        assertEquals("model.layers.0.router.scale", HfGemma4Hooks.filter("model.layers.0.router.scale", base));
        assertEquals(5, seen.size());
    }

    // ------------------------------------------------------------------ per-expert scale, NVFP4 path

    private static final String PES = "model.language_model.layers.0.router.per_expert_scale";
    private static final String W2 = "model.language_model.layers.0.experts.%d.down_proj.weight_scale_2";

    @Test
    public void nvfp4PerExpertScaleIsFoldedIntoWeightScale2() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.language_model.layers.0.experts.0.down_proj.weight", iota(2, 4));
        t.put(String.format(W2, 0), RopeTensors.fromFloats(new long[]{}, new float[]{0.5f}));
        t.put("model.language_model.layers.0.experts.1.down_proj.weight", iota(2, 4));
        t.put(String.format(W2, 1), RopeTensors.fromFloats(new long[]{}, new float[]{0.25f}));
        t.put(PES, RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        HfLoadContext ctx = gemma4(GEMMA4, t);
        HfTensorIndex index = HfGemma4Hooks.index(ctx, ctx.index());
        assertFalse("the per-expert scale is consumed", index.has(PES));
        assertArrayEquals(new float[]{1f}, values(index.get(String.format(W2, 0))), 0f);
        assertArrayEquals(new float[]{0.75f}, values(index.get(String.format(W2, 1))), 0f);
        assertEquals(0, index.get(String.format(W2, 0)).shape().length);
        assertEquals(4, index.size());
    }

    @Test
    public void partialNvfp4ExpertsAreRejected() {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put(String.format(W2, 0), RopeTensors.fromFloats(new long[]{}, new float[]{0.5f}));
        t.put(PES, RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        HfLoadContext ctx = gemma4(GEMMA4, t);
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfGemma4Hooks.index(ctx, ctx.index()));
        assertTrue(e.getMessage(), e.getMessage().contains("partial NVFP4"));
    }

    @Test
    public void withoutNvfp4ExpertsTheIndexIsUnchanged() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put(PES, RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        t.put("model.language_model.layers.0.experts.down_proj", iota(2, 8, 4));
        HfLoadContext ctx = gemma4(GEMMA4, t);
        assertSame(ctx.index(), HfGemma4Hooks.index(ctx, ctx.index()));
    }

    @Test
    public void perExpertWeightsAreStackedWithTheirScales() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.experts.0.down_proj.weight", new Scaled(iota(2, 4), 1f));
        t.put("model.layers.0.experts.1.down_proj.weight", new Scaled(iota(2, 4), 0.75f));
        HfLoadContext ctx = gemma4(GEMMA4 + ", 'global_head_dim': 512", t);
        RecordingSink out = planAll(ctx, HfGemma4Hooks::tensors);
        assertEquals(List.of("blk.0.ffn_down_exps.weight"), new ArrayList<>(out.tensors.keySet()));
        RowSource down = out.tensors.get("blk.0.ffn_down_exps.weight");
        assertArrayEquals(new long[]{2, 2, 4}, down.shape());
        assertArrayEquals("the per-expert weight_scale_2 becomes ffn_down_exps.scale", new float[]{1f, 0.75f},
                down.outputScale(), 0f);
        HfGemma4Hooks.extra(ctx, out);
        assertTrue(out.tensors.containsKey("rope_freqs.weight"));
    }

    @Test
    public void aMissingExpertIsRejected() {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.experts.0.down_proj.weight", new Scaled(iota(2, 4), 1f));
        HfLoadContext ctx = gemma4(GEMMA4, t);
        assertThrows(GgufFormatException.class, () -> planAll(ctx, HfGemma4Hooks::tensors));
    }

    // ------------------------------------------------------------------ per-expert scale, unquantized path

    @Test
    public void routerAndPerExpertScalesBecomeScaleTensors() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        RowSource routerScale = iota(8);
        RowSource pes = RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f});
        RowSource down = iota(2, 8, 4);
        t.put("model.layers.0.router.scale", routerScale);
        t.put("model.layers.0.router.per_expert_scale", pes);
        t.put("model.layers.0.experts.down_proj.weight", down);
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        assertSame(routerScale, out.tensors.get("blk.0.ffn_gate_inp.scale"));
        assertSame(pes, out.tensors.get("blk.0.ffn_down_exps.scale"));
        assertSame(down, out.tensors.get("blk.0.ffn_down_exps.weight"));
    }

    @Test
    public void aScaledDownTensorTakesThePerExpertScaleIntoItsScale() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.router.per_expert_scale", RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        RowSource scaled = new Scaled(iota(2, 2, 2), 2f, 4f);
        t.put("model.layers.0.experts.down_proj.weight", scaled);
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        RowSource down = out.tensors.get("blk.0.ffn_down_exps.weight");
        assertArrayEquals("weight_scale_2[e] * per_expert_scale[e]", new float[]{4f, 12f}, down.outputScale(), 0f);
        assertArrayEquals("the expert weights are untouched", values(scaled), values(down), 0f);
        assertEquals(scaled.format(), down.format());
        assertFalse("the one scale slot holds both factors", out.tensors.containsKey("blk.0.ffn_down_exps.scale"));
        assertTrue(out.consumed.contains("model.layers.0.router.per_expert_scale"));
    }

    @Test
    public void aPerTensorDownScaleBecomesPerExpertWithThePerExpertScale() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.experts.down_proj.weight", new Scaled(iota(2, 2, 2), 0.5f));
        t.put("model.layers.0.router.per_expert_scale", RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        HfLoadContext ctx = gemma4(GEMMA4 + ", 'global_head_dim': 512", t);
        RecordingSink out = planAll(ctx, HfGemma4Hooks::tensors);
        assertArrayEquals(new float[]{1f, 1.5f}, out.tensors.get("blk.0.ffn_down_exps.weight").outputScale(), 0f);
        assertFalse(out.tensors.containsKey("blk.0.ffn_down_exps.scale"));
        HfGemma4Hooks.extra(ctx, out);
    }

    @Test
    public void stackedScaledExpertsTakeThePerExpertScale() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.router.per_expert_scale", RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        t.put("model.layers.0.experts.0.down_proj.weight", new Scaled(iota(2, 4), 1f));
        t.put("model.layers.0.experts.1.down_proj.weight", new Scaled(iota(2, 4), 0.25f));
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        assertArrayEquals(new float[]{2f, 0.75f}, out.tensors.get("blk.0.ffn_down_exps.weight").outputScale(), 0f);
        assertFalse(out.tensors.containsKey("blk.0.ffn_down_exps.scale"));
    }

    @Test
    public void aPerExpertScaleWithoutExpertsIsItsOwnTensor() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        RowSource pes = RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f});
        t.put("model.layers.0.router.per_expert_scale", pes);
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        assertSame(pes, out.tensors.get("blk.0.ffn_down_exps.scale"));
    }

    @Test
    public void aDeferredPerExpertScaleMustBePlanned() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.layers.0.router.per_expert_scale", RopeTensors.fromFloats(new long[]{2}, new float[]{2f, 3f}));
        t.put("model.layers.0.experts.down_proj.weight", iota(2, 2, 2));
        HfLoadContext ctx = gemma4(GEMMA4 + ", 'global_head_dim': 512", t);
        RecordingSink out = new RecordingSink();
        String pes = "model.layers.0.router.per_expert_scale";
        HfGemma4Hooks.tensors(ctx, pes, ctx.index().get(pes), out);
        assertTrue(out.tensors.isEmpty());
        assertThrows(GgufFormatException.class, () -> HfGemma4Hooks.extra(ctx, out));
    }

    @Test
    public void aScaledDownTensorKeepsItsScaleWithoutAPerExpertScale() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        RowSource scaled = new Scaled(iota(2, 2, 2), 2f, 4f);
        t.put("model.layers.0.experts.down_proj.weight", scaled);
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        assertSame(scaled, out.tensors.get("blk.0.ffn_down_exps.weight"));
    }

    @Test
    public void normsAreNotShifted() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        RowSource norm = iota(4);
        t.put("model.layers.0.input_layernorm.weight", norm);
        RecordingSink out = planAll(gemma4(GEMMA4, t), HfGemma4Hooks::tensors);
        assertSame(norm, out.tensors.get("blk.0.attn_norm.weight"));
    }

    // ------------------------------------------------------------------ vocabulary

    @Test
    public void vocabularyIsGemma4WithVisibleChatTokens() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir, "tokenizer.json", "{'added_tokens':[{'id':1,'content':'<bos>','special':true,'normalized':false}],"
                + "'normalizer':null,'pre_tokenizer':null,'post_processor':null,"
                + "'decoder':{'type':'Sequence','decoders':[{'type':'Replace','pattern':{'String':'▁'},'content':' '},"
                + "{'type':'ByteFallback'},{'type':'Fuse'}]},"
                + "'model':{'type':'BPE','unk_token':'<unk>','fuse_unk':true,'byte_fallback':true,"
                + "'ignore_merges':false,"
                + "'vocab':{'<unk>':0,'<bos>':1,'<|channel>':2,'<channel|>':3,'a':4,'b':5,'ab':6},'merges':['a b']}}");
        write(dir, "tokenizer_config.json", "{'bos_token':'<bos>','add_bos_token':false}");
        HfArchEntry entry = HfArchEntry.of(List.of("Gemma4ForCausalLM"), HfArchResolver.of(Arch.GEMMA4),
                HfGemma4Hooks::meta, HfGemma4Hooks::tensors, HfGemma4Hooks::vocab);
        HfLoadContext ctx = HfLoadContext.builder(HfConfigLoader.fromJson(dir, json("{'architectures':"
                + " ['Gemma4ForCausalLM'], " + GEMMA4 + ", 'vocab_size': 7}")), entry, Arch.GEMMA4)
                .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();
        Map<String, Object> md = HfGemma4Hooks.vocab(ctx, (tj, encode) -> {
            throw new AssertionError("the gemma4 vocabulary has no pre-tokenizer name");
        });
        assertEquals("gemma4", md.get(GgufKeys.TOKENIZER_MODEL));
        assertFalse(md.containsKey(GgufKeys.TOKENIZER_PRE));
        assertEquals(Arrays.asList("<unk>", "<bos>", "<|channel>", "<channel|>", "a", "b", "ab"),
                md.get(GgufKeys.TOKENIZER_LIST));
        List<?> types = (List<?>) md.get(GgufKeys.TOKENIZER_TOKEN_TYPE);
        assertEquals("chat-parser tokens are USER_DEFINED", 4, types.get(2));
        assertEquals(4, types.get(3));
        assertEquals(1, types.get(4));
        assertEquals(List.of("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertEquals(false, md.get(GgufKeys.TOKENIZER_ADD_PREFIX));
        assertEquals("add_bos is forced on", true, md.get(GgufKeys.TOKENIZER_ADD_BOS));
        List<String> keys = new ArrayList<>(md.keySet());
        assertEquals("merges follow the token types", keys.indexOf(GgufKeys.TOKENIZER_TOKEN_TYPE) + 1,
                keys.indexOf(GgufKeys.TOKENIZER_MERGES));
    }

    private static void write(Path dir, String name, String singleQuoted) throws IOException {
        Files.write(dir.resolve(name), json(singleQuoted).getBytes(StandardCharsets.UTF_8));
    }
}
