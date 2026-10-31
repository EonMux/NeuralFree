package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
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

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.entry;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The EAGLE-1 HF draft route of WP SP-4a ({@link HfArchTableDrafts}, {@link HfEagle1Hooks}): the draft
 * {@code architectures[0]} names route to {@link Arch#EAGLE} (the layout {@code LLMModelFactory.draftRoute} sends to
 * {@code DraftRoute.HEAD} beside a target, the routing DR-1 delivered), the remaining vLLM draft classes stay
 * refused with the named reason, and the hooks derive the draft's GGUF keys and vocabulary from the target without
 * running a model. The targets are a GGUF file written with {@link GgufTestWriter} and a Hugging Face target
 * directory, as {@link HfEagle3HooksTest} reads them.
 */
public class HfEagle1HooksTest {

    /** The names SP-4a registers, resolved through the registry like a load resolves them. */
    private static final List<String> EAGLE1_NAMES = Collections.unmodifiableList(Arrays.asList(
            "EagleForCausalLM", "EagleModel", "EagleLlamaModel", "LlamaEagleForCausalLM", "EagleDeepSeekMTPModel"));

    /** The draft classes GAPS.md lists as still refused, at SP-4a's scope. */
    private static final List<String> STILL_REFUSED = Collections.unmodifiableList(Arrays.asList(
            "Eagle3Qwen3Model", "PeagleQwen3Model", "Eagle3DeepseekModel", "DFlash2BeamDraftModel",
            "DSparkV41DraftModel", "Qwen3OmniDSparkModel", "K3DSparkModel", "DFlashLagunaForCausalLM",
            "MiMoMTPLlamaForCausalLM"));

    private static final String DRAFT = "{'architectures': ['EagleModel'], 'hidden_size': 8,"
            + " 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 1,"
            + " 'num_hidden_layers': 1, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'vocab_size': 10%s}";

    private static final int TARGET_VOCAB = 10;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private final List<AutoCloseable> closing = new ArrayList<>();

    @After
    public void close() throws Exception {
        for (AutoCloseable c : closing) {
            c.close();
        }
    }

    // ------------------------------------------------------------------ the route

    @Test
    public void theEagle1NamesRouteToTheEagleArchitecture() {
        HfArchEntry eagle = entry("EagleModel");
        for (String name : EAGLE1_NAMES) {
            assertSame(name, eagle, entry(name));
        }
        assertEquals(Arrays.asList(EAGLE1_NAMES.toArray()), eagle.hfNames());
        assertEquals(Arch.EAGLE, eagle.arch().resolve(config("{'architectures': ['EagleModel'],"
                + " 'num_hidden_layers': 1}"), HfTensorIndex.of(Collections.emptyMap())));
    }

    @Test
    public void theRemainingVllmDraftClassesStayRefused() {
        for (String name : STILL_REFUSED) {
            HfConfig cfg = config("{'architectures': ['" + name + "'], 'num_hidden_layers': 1}");
            GgufFormatException e = assertThrows(name, GgufFormatException.class,
                    () -> HfArchRegistry.resolve(cfg, HfArchEntry.Kind.TEXT));
            assertTrue(name, e.getMessage().contains(name));
        }
        assertFalse(HfArchRegistry.has("Eagle3Qwen3Model", HfArchEntry.Kind.TEXT));
    }

    // ------------------------------------------------------------------ the hooks

    @Test
    public void metadataNamesTheTargetsLastLayerAndHiddenSize() throws Exception {
        HfMetadata md = meta(draft(""));
        assertEquals(Arrays.asList(3), md.get("eagle.target_layers"));
        assertEquals(8, md.get("eagle.target_hidden_size"));
        assertEquals(false, md.get("eagle.norm_before_residual"));
        assertEquals(false, md.get("eagle.norm_before_fc"));
    }

    @Test
    public void anAliasCheckpointNormalizesToOneDrafterLayer() {
        HfConfig fixed = HfEagle1Hooks.fixup(config("{'architectures': ['LlamaEagleForCausalLM'],"
                + " 'hidden_size': 8}"));
        assertEquals("EagleForCausalLM", fixed.architecture());
        assertEquals(1, fixed.requireInt("num_hidden_layers"));
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfEagle1Hooks.fixup(config("{'architectures': ['EagleForCausalLM'],"
                        + " 'num_hidden_layers': 2}")));
        assertTrue(e.getMessage().contains("one decoder layer"));
    }

    @Test
    public void vocabularyIsTheTargetTokenizer() throws Exception {
        HfLoadContext ctx = draft("");
        Map<String, Object> vocab = ctx.entry().vocab().build(ctx, new HfPreTokenizerResolver());
        assertEquals("gpt2", vocab.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(TARGET_VOCAB, ((List<?>) vocab.get(GgufKeys.TOKENIZER_LIST)).size());
        assertTrue(vocab.keySet().stream().allMatch(k -> k.startsWith("tokenizer.")));
    }

    @Test
    public void aCheckpointTargetIsReadFromItsConfigWithoutBeingConverted() throws Exception {
        Path dir = tmp.newFolder("hf-target").toPath();
        Files.write(dir.resolve("config.json"), ("{\"architectures\": [\"LlamaForCausalLM\"], \"hidden_size\": 3072,"
                + " \"num_hidden_layers\": 28, \"num_attention_heads\": 2, \"num_key_value_heads\": 1,"
                + " \"intermediate_size\": 16, \"rms_norm_eps\": 1e-06, \"vocab_size\": 10,"
                + " \"max_position_embeddings\": 64}").getBytes(StandardCharsets.UTF_8));
        writeEmptySafetensors(dir);
        // the target's tokenizer files feed the draft vocabulary when the target is a checkpoint
        write(dir, "tokenizer.json", json("{'model':{'type':'BPE','vocab':{'a':0,'b':1},'merges':[['a','b']]}}"));
        write(dir, "vocab.json", json("{'a':0,'b':1}"));
        write(dir, "merges.txt", "#version: 0.0\na b");
        HfMetadata md = meta(draftOf(String.format(DRAFT, ""), empty(), dir));
        assertEquals(Arrays.asList(27), md.get("eagle.target_layers"));
        assertEquals(3072, md.get("eagle.target_hidden_size"));
    }

    @Test
    public void aDraftWithoutATargetIsRefused() {
        HfConfig cfg = config(String.format(DRAFT, ""));
        HfArchEntry e = entry("EagleModel");
        HfLoadContext ctx = HfLoadContext.builder(e.configFixup().apply(cfg), e, Arch.EAGLE)
                .index(empty()).build();
        assertThrows(GgufFormatException.class, () -> HfEagle1Hooks.meta(ctx,
                new HfMetadataBuilder(Arch.EAGLE.ggufName()), null));
    }

    // ------------------------------------------------------------------ fixtures

    private Path target() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("target.gguf");
        if (!p.toFile().exists()) {
            GgufTestWriter w = new GgufTestWriter().architecture("llama");
            w.u32(w.archKey("embedding_length"), 8).u32(w.archKey("block_count"), 4)
                    .u32(w.archKey("vocab_size"), TARGET_VOCAB)
                    .str(GgufKeys.TOKENIZER_MODEL, "gpt2")
                    .strArray(GgufKeys.TOKENIZER_LIST, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                    .writeTo(p);
        }
        return p;
    }

    private HfLoadContext draft(String extra) throws IOException {
        return draftOf(String.format(DRAFT, extra), empty(), target());
    }

    private static HfTensorIndex empty() {
        return HfTensorIndex.of(Collections.emptyMap());
    }

    private HfLoadContext draftOf(String json, HfTensorIndex index, Path target) throws IOException {
        HfConfig cfg = config(json);
        HfArchEntry e = entry("EagleModel");
        HfResources resources = new HfResources();
        closing.add(resources);
        return HfLoadContext.builder(e.configFixup().apply(cfg), e, Arch.EAGLE).index(index)
                .resources(resources).targetPath(target).build();
    }

    private HfMetadata meta(HfLoadContext ctx) throws IOException {
        assertEquals(Arch.EAGLE, ctx.arch());
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }
    /** A structurally valid safetensors file with an empty header and no tensors. */
    private static void writeEmptySafetensors(Path dir) throws IOException {
        byte[] header = "{}".getBytes(StandardCharsets.UTF_8);
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(8 + header.length);
        b.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.putLong(header.length);
        b.put(header);
        Files.write(dir.resolve("model.safetensors"), b.array());
    }


    private static void write(Path dir, String name, String text) throws IOException {
        Files.write(dir.resolve(name), text.getBytes(StandardCharsets.UTF_8));
    }

    private static String json(String singleQuoted) {
        return singleQuoted.replace('\'', '"');
    }


}
