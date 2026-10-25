package com.eonmux.neuralfree.gguf.hf.names;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import org.junit.Test;

import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Predicate;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The generated name tables load and validate, their views match the table rows (counts and sample rows derived from
 * the TSV files), the converter's tensor filters behave as {@code filter_tensors}, and the architecture registry
 * reports rejected, projector-only and unknown names with their distinct messages.
 */
public class HfNameTablesTest {

    private static HfConfig config(String architecture) {
        return HfConfigLoader.fromJson(Path.of("."), "{\"architectures\": [\"" + architecture + "\"]}");
    }

    @Test
    public void tableViewsMatchTheirRows() {
        assertEquals(618, HfNameTables.tensorNames().size());
        assertEquals("token_embd", HfNameTables.tensorNames().get("TOKEN_EMBD"));
        assertEquals("blk.{bid}.ffn_gate_inp", HfNameTables.tensorNames().get("FFN_GATE_INP"));
        assertEquals(42, HfNameTables.globalPatterns().size());
        assertEquals(443, HfNameTables.blockPatterns().size());
        assertEquals(Arrays.asList("gpt_neox.final_layer_norm", "transformer.ln_f", "model.norm"),
                HfNameTables.globalPatterns().get("OUTPUT_NORM").subList(0, 3));
        assertEquals(Arrays.asList("model.layers.{bid}.self_attn.q_proj", "model.layers.{bid}.attention.q_proj",
                "layers.{bid}.self_attn.q_proj"), HfNameTables.blockPatterns().get("ATTN_Q").subList(0, 3));
        assertTrue(HfNameTables.globalPatterns().get("ROPE_FACTORS_LONG").isEmpty());
        assertTrue(HfNameTables.blockPatterns().get("A_PRE_NORM").isEmpty());
        assertEquals(Arrays.asList("TOKEN_EMBD", "OUTPUT_NORM", "OUTPUT", "ROPE_FREQS", "ATTN_NORM", "ATTN_QKV",
                "ATTN_Q", "ATTN_K", "ATTN_V", "ATTN_OUT", "ATTN_ROT_EMBD", "FFN_GATE_INP", "FFN_NORM", "FFN_GATE",
                "FFN_DOWN", "FFN_UP", "FFN_GATE_EXP", "FFN_DOWN_EXP", "FFN_UP_EXP"),
                Arrays.asList(HfNameTables.archTensors("llama").toArray()));
        assertTrue(HfNameTables.archTensors("no-such-arch").isEmpty());
        assertEquals(Arrays.asList("FFN_NORM", "FFN_NORM_EXP"),
                Arrays.asList(HfNameTables.blockOverrides("arctic").keySet().toArray()));
        assertEquals(Collections.singletonList("model.layers.{bid}.residual_layernorm"),
                HfNameTables.blockOverrides("arctic").get("FFN_NORM"));
        assertTrue(HfNameTables.blockOverrides("llama").isEmpty());
    }

    @Test
    public void viewsAreUnmodifiable() {
        assertThrows(UnsupportedOperationException.class, () -> HfNameTables.tensorNames().put("X", "x"));
        assertThrows(UnsupportedOperationException.class,
                () -> HfNameTables.blockPatterns().get("ATTN_Q").add("x"));
        assertThrows(UnsupportedOperationException.class, () -> HfNameTables.archTensors("llama").add("X"));
        assertThrows(UnsupportedOperationException.class,
                () -> HfNameTables.filters(HfNameTables.FilterKind.DROP_PREFIX).clear());
    }

    @Test
    public void filterRowsLoadPerKind() {
        assertEquals(1, HfNameTables.filters(HfNameTables.FilterKind.RENAME_SUFFIX).size());
        assertEquals(1, HfNameTables.filters(HfNameTables.FilterKind.RENAME_CONTAINS).size());
        assertEquals(1, HfNameTables.filters(HfNameTables.FilterKind.STRIP_PREFIX).size());
        assertEquals(11, HfNameTables.filters(HfNameTables.FilterKind.DROP_PREFIX).size());
        assertEquals(16, HfNameTables.filters(HfNameTables.FilterKind.DROP_CONTAINS).size());
        assertEquals(1, HfNameTables.filters(HfNameTables.FilterKind.SKIP_CONTAINS).size());
        assertEquals(3, HfNameTables.filters(HfNameTables.FilterKind.SKIP_SUFFIX).size());
        assertEquals(22, HfNameTables.filters(HfNameTables.FilterKind.FORCED_F32).size());
        HfNameTables.FilterRule r = HfNameTables.filters(HfNameTables.FilterKind.RENAME_SUFFIX).get(0);
        assertEquals("e_score_correction_bias", r.value());
        assertEquals("e_score_correction.bias", r.replacement());
        assertEquals("base.py:588", r.source());
    }

    @Test
    public void forcedF32CoversEveryBlockAndOnlyWeights() {
        Predicate<String> f32 = HfNameTables.forcedF32();
        assertTrue(f32.test("blk.0.ffn_gate_inp.weight"));
        assertTrue(f32.test("blk.117.ffn_gate_inp.weight"));
        assertTrue(f32.test("position_embd.weight"));
        assertTrue(f32.test("token_types.weight"));
        assertTrue(f32.test("blk.3.ssm_conv1d.weight"));
        assertFalse(f32.test("blk.0.ffn_gate_inp.bias"));
        assertFalse(f32.test("blk.x.ffn_gate_inp.weight"));
        assertFalse(f32.test("blk.0.ffn_up.weight"));
        assertFalse(f32.test("xposition_embd.weight"));
    }

    @Test
    public void tensorNamesFormatWithBlocks() {
        assertEquals("blk.7.attn_q", HfNameTables.formatTensorName("ATTN_Q", 7));
        assertEquals("token_embd", HfNameTables.formatTensorName("TOKEN_EMBD", -1));
        assertThrows(IllegalArgumentException.class, () -> HfNameTables.formatTensorName("ATTN_Q", -1));
        assertThrows(IllegalArgumentException.class, () -> HfNameTables.formatTensorName("TOKEN_EMBD", 0));
        assertThrows(IllegalArgumentException.class, () -> HfNameTables.formatTensorName("NO_SUCH_TENSOR", 0));
    }

    @Test
    public void readerRejectsMissingFilesAndWrongColumns() {
        IllegalStateException missing = assertThrows(IllegalStateException.class,
                () -> HfNameTables.read("no_such_table.tsv", "a"));
        assertTrue(missing.getMessage(), missing.getMessage().startsWith("hf/no_such_table.tsv"));
        IllegalStateException columns = assertThrows(IllegalStateException.class,
                () -> HfNameTables.read("tensor_names.tsv", "tensor", "gguf_name"));
        assertTrue(columns.getMessage(), columns.getMessage().startsWith("hf/tensor_names.tsv:2: "));
        HfNameTables.Tsv t = HfNameTables.read("tensor_names.tsv", "tensor", "gguf_template");
        assertEquals("a37cc4610", t.commit());
        assertEquals(618, t.rows().size());
        assertEquals(3, t.rows().get(0).line());
        assertEquals("TOKEN_EMBD", t.rows().get(0).get(0));
        assertTrue(HfNameTables.isNone("-"));
        assertFalse(HfNameTables.isNone("x"));
    }

    @Test
    public void textFiltersDropRenameAndSkip() {
        HfArchEntry.Kind text = HfArchEntry.Kind.TEXT;
        assertEquals("model.layers.0.mlp.gate.e_score_correction.bias",
                HfTensorFilters.filterName("model.layers.0.mlp.gate.e_score_correction_bias", text));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfTensorFilters.filterName("model.language_model.layers.0.self_attn.q_proj.weight", text));
        assertNull(HfTensorFilters.filterName("model.vision_tower.blocks.0.attn.qkv.weight", text));
        assertNull(HfTensorFilters.filterName("model.visual.blocks.0.attn.qkv.weight", text));
        assertNull(HfTensorFilters.filterName("mlp1.0.weight", text));
        assertNull(HfTensorFilters.filterName("model.mlp1.0.weight", text));
        assertNull(HfTensorFilters.filterName("multi_modal_projector.linear_1.weight", text));
        assertNull(HfTensorFilters.filterName("model.layers.0.mlp.audio_tower.weight", text));
        assertEquals("model.layers.0.mlp.up_proj.weight",
                HfTensorFilters.filterName("model.layers.0.mlp.up_proj.weight", text));
        assertEquals("transformer.h.0.attn.attention.bias",
                HfTensorFilters.filterName("transformer.h.0.attn.attention.bias", text));
        assertTrue(HfTensorFilters.isSkipped("transformer.h.0.attn.attention.bias"));
        assertTrue(HfTensorFilters.isSkipped("model.layers.0.self_attn.rotary_emb.inv_freq"));
        assertFalse(HfTensorFilters.isSkipped("model.layers.0.self_attn.q_proj.bias"));
    }

    private static RowSource unread() {
        return (RowSource) Proxy.newProxyInstance(RowSource.class.getClassLoader(), new Class<?>[] {RowSource.class},
                (proxy, method, args) -> {
                    throw new UnsupportedOperationException("filters never read tensors");
                });
    }

    private static HfLoadContext context(HfArchEntry entry, int layers, String... names) {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        for (String n : names) {
            tensors.put(n, unread());
        }
        HfConfig cfg = HfConfigLoader.fromJson(Path.of("."),
                "{\"architectures\": [\"LlamaForCausalLM\"], \"num_hidden_layers\": " + layers + "}");
        return HfLoadContext.builder(cfg, entry, Arch.LLAMA).index(HfTensorIndex.of(tensors)).build();
    }

    @Test
    public void classNameFilterWrapsTheInheritedFilters() {
        HfArchEntry entry = HfArchEntry.of(List.of("LlamaForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, (ctx, pre) -> Collections.emptyMap());
        HfLoadContext plain = context(entry, 1);
        assertSame(HfNameFilter.INHERITED, entry.nameFilter(plain));
        assertEquals("model.text_model.layers.0.self_attn.q_proj.weight",
                HfTensorFilters.filterName(plain, "model.text_model.layers.0.self_attn.q_proj.weight"));
        HfArchEntry smolVlm = entry.withNameFilter((n, base) -> n.startsWith("lm_head_extra.") ? null
                : base.apply(n.replace("text_model.", "")));
        HfLoadContext ctx = context(smolVlm, 1, "model.text_model.embed_tokens.weight",
                "model.text_model.visual.patch.weight", "lm_head_extra.weight",
                "model.text_model.layers.0.self_attn.rotary_emb.inv_freq");
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfTensorFilters.filterName(ctx, "model.text_model.layers.0.self_attn.q_proj.weight"));
        assertNull(HfTensorFilters.filterName(ctx, "model.text_model.visual.blocks.0.attn.qkv.weight"));
        assertNull(HfTensorFilters.filterName(ctx, "lm_head_extra.weight"));

        HfTensorFilters.Result r = HfTensorFilters.apply(ctx, ctx.index(), HfArchEntry.Kind.TEXT);
        assertEquals(Collections.singletonList("model.embed_tokens.weight"), r.index().names());
        assertEquals(Collections.singletonList("model.visual"), r.droppedPrefixes());
    }

    @Test
    public void loadScopedNameFilterReadsTheLoadContext() {
        HfArchEntry entry = HfArchEntry.of(List.of("LlamaForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, (ctx, pre) -> Collections.emptyMap())
                .withLoadNameFilter(load -> {
                    int base = load.config().blockCount();
                    return (n, inherited) -> {
                        int bid = HfLoadContext.bidOf(n);
                        return bid >= base ? null : inherited.apply(n);
                    };
                });
        HfLoadContext two = context(entry, 2, "model.layers.0.mlp.up_proj.weight",
                "model.layers.1.mlp.up_proj.weight", "model.layers.2.mlp.up_proj.weight");
        HfLoadContext one = context(entry, 1);
        assertEquals("model.layers.1.mlp.up_proj.weight",
                HfTensorFilters.filterName(two, "model.layers.1.mlp.up_proj.weight"));
        assertNull(HfTensorFilters.filterName(one, "model.layers.1.mlp.up_proj.weight"));
        assertNull(HfTensorFilters.filterName(two, "model.layers.2.mlp.up_proj.weight"));
        assertEquals(Arrays.asList("model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.up_proj.weight"),
                HfTensorFilters.apply(two, two.index(), HfArchEntry.Kind.TEXT).index().names());

        HfArchEntry nullFactory = entry.withLoadNameFilter(load -> null);
        assertThrows(IllegalStateException.class, () -> nullFactory.nameFilter(one));
    }

    @Test
    public void indexedBlockCountOverridesTheConfigurationRule() {
        HfArchEntry entry = HfArchEntry.of(List.of("LlamaForCausalLM"), HfArchResolver.of(Arch.LLAMA),
                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, (ctx, pre) -> Collections.emptyMap());
        HfLoadContext ctx = context(entry, 2, "model.layers.0.mlp.up_proj.weight",
                "model.layers.2.mlp.up_proj.weight");
        assertEquals(2, entry.blockCount(ctx));
        HfArchEntry indexed = entry.withIndexedBlockCount(load -> {
            int max = -1;
            for (String n : load.index().names()) {
                max = Math.max(max, HfLoadContext.bidOf(n));
            }
            return max + 1;
        });
        assertEquals(2, indexed.blockCount(ctx.config()));
        assertEquals(3, indexed.blockCount(ctx));
        HfArchEntry tooMany = entry.withIndexedBlockCount(load -> (int) HfConfig.MAX_LAYERS + 1);
        assertThrows(GgufFormatException.class, () -> tooMany.blockCount(ctx));
    }

    @Test
    public void projectorFiltersDropTheLanguageModel() {
        HfArchEntry.Kind mmproj = HfArchEntry.Kind.MMPROJ;
        assertNull(HfTensorFilters.filterName("model.language_model.layers.0.self_attn.q_proj.weight", mmproj));
        assertEquals("model.visual.blocks.0.attn.qkv.weight",
                HfTensorFilters.filterName("model.visual.blocks.0.attn.qkv.weight", mmproj));
        assertEquals("vision_tower.x.e_score_correction.bias",
                HfTensorFilters.filterName("vision_tower.x.e_score_correction_bias", mmproj));
    }

    @Test
    public void archNamesCarryTheirConverterRows() {
        Optional<HfArchNames.Row> llama = HfArchNames.row("LlamaForCausalLM", HfArchEntry.Kind.TEXT);
        assertTrue(llama.isPresent());
        assertEquals("LlamaModel", llama.get().converterClass());
        assertEquals("llama.py:33", llama.get().source());
        assertEquals(Arch.LLAMA, llama.get().arch());
        // LlamaModel switches to eagle3 at runtime for single-layer drafts with a draft vocabulary (llama.py:52).
        assertEquals(1, llama.get().dynamicArchs().size());
        assertEquals(EnumSet.of(Arch.LLAMA, Arch.EAGLE3), llama.get().possibleArchs());
        Optional<HfArchNames.Row> nomic = HfArchNames.row("NomicBertModel", HfArchEntry.Kind.TEXT);
        assertTrue(nomic.isPresent());
        assertNull(nomic.get().arch());
        assertEquals(EnumSet.of(Arch.NOMIC_BERT, Arch.NOMIC_BERT_MOE), nomic.get().possibleArchs());
        List<HfArchNames.DynamicArch> mistral3 = HfArchNames.dynamicArchs("Mistral3Model");
        assertEquals(3, mistral3.size());
        assertEquals(Arch.MISTRAL4, mistral3.get(2).arch());
        assertEquals("self.hparams.get('model_type') == 'mistral4'", mistral3.get(2).condition());
        Optional<HfArchNames.Row> granite = HfArchNames.row("GraniteSpeechForConditionalGeneration",
                HfArchEntry.Kind.TEXT);
        assertTrue(granite.isPresent());
        assertFalse(granite.get().hasClass());
        assertTrue(HfArchNames.row("GlmasrModel", HfArchEntry.Kind.MMPROJ).isPresent());
        assertFalse(HfArchNames.row("GlmasrModel", HfArchEntry.Kind.TEXT).isPresent());
        assertEquals(375, HfArchNames.rows(HfArchEntry.Kind.TEXT).size()
                + HfArchNames.rows(HfArchEntry.Kind.MMPROJ).size());
    }

    @Test
    public void registryRejectsTextlessNamesWithTheirReason() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(config("GraniteSpeechForConditionalGeneration"), HfArchEntry.Kind.TEXT));
        assertEquals("'GraniteSpeechForConditionalGeneration' has no text model conversion in llama.cpp's converter;"
                + " only its projector converts, open it with ModelSources.openProjector", e.getMessage());
    }

    @Test
    public void registryPointsProjectorOnlyNamesAtTheProjectorApi() {
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(config("GlmasrModel"), HfArchEntry.Kind.TEXT));
        assertEquals("'GlmasrModel' is a vision/audio projector checkpoint; open it with ModelSources.openProjector",
                e.getMessage());
    }

    @Test
    public void registryNamesUnknownArchitectures() {
        GgufFormatException text = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(config("NoSuchModelForCausalLM"), HfArchEntry.Kind.TEXT));
        assertEquals("Hugging Face architecture 'NoSuchModelForCausalLM' is not supported: llama.cpp's converter"
                + " does not know it", text.getMessage());
        GgufFormatException mmproj = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(config("NoSuchModelForCausalLM"), HfArchEntry.Kind.MMPROJ));
        assertEquals("Hugging Face architecture 'NoSuchModelForCausalLM' has no projector conversion: llama.cpp's"
                + " converter does not know it as a vision/audio projector", mmproj.getMessage());
        GgufFormatException known = assertThrows(GgufFormatException.class,
                () -> HfArchRegistry.resolve(config("GlmasrModel"), HfArchEntry.Kind.MMPROJ));
        assertEquals("Hugging Face projector architecture 'GlmasrModel' (llama.cpp converter class"
                + " GlmASRWhisperEncoderModel, ultravox.py:24) has no projector conversion in NeuralFree",
                known.getMessage());
    }

    @Test
    public void testOverlayTakesPrecedenceUntilClosed() {
        HfArchEntry entry = HfArchEntry.of(Collections.singletonList("NoSuchModelForCausalLM"),
                HfArchResolver.of(Arch.LLAMA), HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        HfConfig cfg = config("NoSuchModelForCausalLM");
        try (HfArchRegistry.Overlay ignored = HfArchRegistry.withExtraForTest(Collections.singletonList(entry))) {
            assertSame(entry, HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT));
            assertTrue(HfArchRegistry.has("NoSuchModelForCausalLM", HfArchEntry.Kind.TEXT));
            assertThrows(IllegalStateException.class,
                    () -> HfArchRegistry.withExtraForTest(Collections.<HfArchEntry>emptyList()));
        }
        assertFalse(HfArchRegistry.has("NoSuchModelForCausalLM", HfArchEntry.Kind.TEXT));
        assertThrows(GgufFormatException.class, () -> HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT));
        assertThrows(IllegalArgumentException.class,
                () -> HfArchRegistry.withExtraForTest(Arrays.asList(entry, entry)));
    }
}
