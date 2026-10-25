package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.LLAMA_BASE;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.longs;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.rows;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfEagle3Hooks} through the Llama entry on synthetic EAGLE-3 drafts whose target is a GGUF file written with
 * {@link GgufTestWriter}: the {@code transformer_layer_config} flattening, the target-derived metadata, the
 * {@code midlayer.} renames and special tensors, the {@code d2t} conversion and validation, and the target tokenizer.
 */
public class HfEagle3HooksTest {

    private static final String DRAFT = "{'architectures': ['LlamaForCausalLMEagle3'], 'draft_vocab_size': 4, "
            + LLAMA_BASE + "%s}";
    private static final int TARGET_VOCAB = 10;
    /** A tokenizer key older converters wrote that the converter's key table no longer types. */
    private static final String UNTYPED_TOKENIZER_KEY = "tokenizer.ggml.prefix_token_id";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<AutoCloseable> closing = new ArrayList<>();

    @After
    public void close() throws Exception {
        for (AutoCloseable c : closing) {
            c.close();
        }
    }

    private Path target() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("target.gguf");
        if (!p.toFile().exists()) {
            GgufTestWriter w = new GgufTestWriter().architecture("llama");
            w.u32(w.archKey("embedding_length"), 4096).u32(w.archKey("block_count"), 32)
                    .u32(w.archKey("vocab_size"), TARGET_VOCAB)
                    .str(GgufKeys.TOKENIZER_MODEL, "gpt2").u32(UNTYPED_TOKENIZER_KEY, 3)
                    .strArray(GgufKeys.TOKENIZER_LIST, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                    .writeTo(p);
        }
        return p;
    }

    private HfLoadContext draft(String extra, HfTensorIndex index) throws IOException {
        return draftOf(String.format(DRAFT, extra), index);
    }

    private HfLoadContext draftOf(String json, HfTensorIndex index) throws IOException {
        return draftOf(json, index, target());
    }

    private HfLoadContext draftOf(String json, HfTensorIndex index, Path target) throws IOException {
        HfConfig cfg = config(json);
        HfArchEntry entry = entry(cfg.architecture());
        Arch arch = entry.arch().resolve(cfg, index);
        HfResources resources = new HfResources();
        closing.add(resources);
        return HfLoadContext.builder(entry.configFixup().apply(cfg), entry, arch).index(index)
                .resources(resources).targetPath(target).build();
    }

    private Checkpoint checkpoint() throws IOException {
        Checkpoint c = new Checkpoint(tmp.newFolder().toPath());
        closing.add(c);
        return c;
    }

    private static HfTensorIndex empty() {
        return HfTensorIndex.of(Collections.<String, RowSource>emptyMap());
    }

    @Test
    public void transformerLayerConfigIsFlattenedUnderTextConfig() {
        HfConfig cfg = config("{'architectures': ['Eagle3Speculator'], 'num_hidden_layers': 32, 'hidden_size': 16,"
                + " 'draft_vocab_size': 4, 'transformer_layer_config': {'num_hidden_layers': 1, 'hidden_size': 64,"
                + " 'intermediate_size': 128}, 'text_config': {'hidden_size': 32}}");
        HfConfig fixed = HfEagle3Hooks.fixup(cfg);
        assertEquals(1, fixed.requireInt("num_hidden_layers"));
        assertEquals(128, fixed.requireInt("intermediate_size"));
        assertEquals(32, fixed.requireInt("hidden_size"));
        assertTrue(HfEagle3Hooks.isEagle3(cfg));
        assertSame(cfg, HfEagle3Hooks.fixup(config("{'architectures': ['LlamaForCausalLM'],"
                + " 'num_hidden_layers': 1}")));
        assertThrows(GgufFormatException.class, () -> HfEagle3Hooks.fixup(config("{'architectures': ['A'],"
                + " 'num_hidden_layers': 1, 'transformer_layer_config': 3}")));
    }

    @Test
    public void metadataComesFromTheDraftConfigElseTheTarget() throws Exception {
        HfMetadata derived = meta(draft("", empty()));
        assertEquals(Arrays.asList(2, 16, 29), derived.get("eagle3.target_layers"));
        assertEquals(4096, derived.get("eagle3.target_hidden_size"));
        assertEquals(false, derived.get("eagle3.norm_before_residual"));
        assertEquals(false, derived.get("eagle3.norm_before_fc"));
        assertEquals(10, derived.get("eagle3.vocab_size"));

        HfMetadata own = meta(draft(", 'eagle_aux_hidden_state_layer_ids': [1, 5, 9], 'target_hidden_size': 2048,"
                + " 'norm_before_residual': true, 'norm_before_fc': true", empty()));
        assertEquals(Arrays.asList(1, 5, 9), own.get("eagle3.target_layers"));
        assertEquals(2048, own.get("eagle3.target_hidden_size"));
        assertEquals(true, own.get("eagle3.norm_before_residual"));
        assertEquals(true, own.get("eagle3.norm_before_fc"));

        HfMetadata nested = meta(draft(", 'eagle_config': {'eagle_aux_hidden_state_layer_ids': [3, 7, 11],"
                + " 'norm_before_fc': true}", empty()));
        assertEquals(Arrays.asList(3, 7, 11), nested.get("eagle3.target_layers"));
        assertEquals(true, nested.get("eagle3.norm_before_fc"));
        HfMetadata rootFirst = meta(draft(", 'eagle_aux_hidden_state_layer_ids': [1, 5, 9], 'norm_before_fc': false,"
                + " 'eagle_config': {'eagle_aux_hidden_state_layer_ids': [3, 7, 11], 'norm_before_fc': true}",
                empty()));
        assertEquals(Arrays.asList(1, 5, 9), rootFirst.get("eagle3.target_layers"));
        assertEquals(false, rootFirst.get("eagle3.norm_before_fc"));
    }

    private static HfMetadata meta(HfLoadContext ctx) throws IOException {
        assertEquals(Arch.EAGLE3, ctx.arch());
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }

    @Test
    public void checkpointTargetIsReadFromItsConfigWithoutBeingConverted() throws Exception {
        Path dir = tmp.newFolder("hf-target").toPath();
        Files.write(dir.resolve("config.json"), ("{\"architectures\": [\"LlamaForCausalLM\"], \"hidden_size\": 3072,"
                + " \"num_hidden_layers\": 28, \"vocab_size\": 10}").getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("model.safetensors"), "not a safetensors file".getBytes(StandardCharsets.UTF_8));
        HfMetadata md = meta(draftOf(String.format(DRAFT, ""), empty(), dir));
        assertEquals(3072, md.get("eagle3.target_hidden_size"));
        assertEquals(Arrays.asList(2, 14, 25), md.get("eagle3.target_layers"));
    }

    @Test
    public void draftTensorsAreRenamedAndTheSpecialOnesMapped() throws IOException {
        HfTensorIndex index = checkpoint().f32("midlayer.self_attn.q_proj.weight", new long[]{8, 2}, seq(16))
                .f32("midlayer.hidden_norm.weight", new long[]{8}, seq(8))
                .f32("layers.0.mlp.up_proj.weight", new long[]{16, 8}, seq(128))
                .f32("input_norm.weight", new long[]{8}, seq(8))
                .f32("fc.weight", new long[]{8, 24}, seq(192))
                .i64("d2t", new long[]{4}, 0, 1, 5, 2)
                .add("t2d", StDType.BOOL, new long[]{10}, new byte[]{1, 0, 1, 0, 0, 0, 0, 1, 0, 0})
                .f32("norm.weight", new long[]{8}, seq(8)).index();
        Sink out = plan(draft("", index));
        assertArrayEquals(rows(seq(16), 2, 0, 2, 1, 3, 4, 6, 5, 7), floats(out.get("blk.0.attn_q.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("blk.0.attn_norm_2.weight")), 0f);
        assertArrayEquals(seq(128), floats(out.get("blk.0.ffn_up.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("enc.output_norm.weight")), 0f);
        assertArrayEquals(seq(192), floats(out.get("fc.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("output_norm.weight")), 0f);
        RowSource d2t = out.get("d2t");
        assertEquals(GGMLType.I64, d2t.format().ggmlType());
        assertArrayEquals(new long[]{4}, d2t.shape());
        assertArrayEquals(new long[]{0, 2, 7, 5}, longs(d2t));
        assertEquals(Collections.singleton("t2d"), out.dropped.keySet());
    }

    @Test
    public void vllmAliasDraftsUseTheirVocabularyAsDraftVocabularyAndDropTheParallelMask() throws IOException {
        HfTensorIndex index = checkpoint().i64("d2t", new long[]{TARGET_VOCAB}, new long[TARGET_VOCAB])
                .f32("mask_hidden", new long[]{1, 24}, seq(24)).index();
        Sink out = plan(draftOf("{'architectures': ['PEagleDraftModel'], " + LLAMA_BASE + "}", index));
        long[] identity = new long[TARGET_VOCAB];
        for (int i = 0; i < identity.length; i++) {
            identity[i] = i;
        }
        assertArrayEquals(identity, longs(out.get("d2t")));
        assertEquals(Collections.singleton("mask_hidden"), out.dropped.keySet());
    }

    @Test
    public void d2tMustMapEveryDraftTokenToADistinctTargetToken() throws IOException {
        assertD2tRejected(new long[]{4}, 0, 1, 2, 7);
        assertD2tRejected(new long[]{4}, -1, 0, 0, 0);
        assertD2tRejected(new long[]{4}, 1, 0, 0, 0);
        assertD2tRejected(new long[]{3}, 0, 0, 0);
    }

    private void assertD2tRejected(long[] shape, long... values) throws IOException {
        HfTensorIndex index = checkpoint().i64("d2t", shape, values).index();
        HfLoadContext ctx = draft("", index);
        assertThrows(GgufFormatException.class, () -> plan(ctx));
    }

    @Test
    public void vocabularyIsTheTargetTokenizer() throws Exception {
        HfLoadContext ctx = draft("", empty());
        Map<String, Object> vocab = ctx.entry().vocab().build(ctx, new HfPreTokenizerResolver());
        assertEquals("gpt2", vocab.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(TARGET_VOCAB, ((List<?>) vocab.get(GgufKeys.TOKENIZER_LIST)).size());
        assertTrue(vocab.keySet().stream().allMatch(k -> k.startsWith("tokenizer.")));
        assertFalse(vocab.containsKey(UNTYPED_TOKENIZER_KEY));
    }
}
