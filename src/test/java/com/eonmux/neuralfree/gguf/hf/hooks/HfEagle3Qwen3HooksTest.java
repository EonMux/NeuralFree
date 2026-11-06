package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Checkpoint;
import com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.Sink;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfPlanView;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.LLAMA_BASE;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.config;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.floats;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.longs;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.plan;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfLlamaHooksTest.seq;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The Qwen3 EAGLE-3 HF draft route ({@link HfArchTableLlama}, {@link HfEagle3Qwen3Hooks}): the
 * {@code Eagle3Qwen3ForCausalLM} / {@code PeagleQwen3ForCausalLM} names route to the EAGLE3 row, the configuration
 * fixup defaults the draft vocabulary and the one decoder layer, the metadata carries the target taps (declared or the
 * three-layer default) and the draft {@code logit_scale}, the Qwen3 projection norms and per-tap encoder norms map to
 * the GGUF names the drafter reads, and the variants the engine does not run are refused by name. Targets are the GGUF
 * file of {@link HfEagle3HooksTest} and a Hugging Face checkpoint directory.
 */
public class HfEagle3Qwen3HooksTest {

    private static final String ARCH = "Eagle3Qwen3ForCausalLM";
    /** The fusion variants appended to a draft configuration: {@code %s} sits before the closing brace. */
    private static final String DRAFT = "{'architectures': ['" + ARCH + "'], 'draft_vocab_size': 4, "
            + LLAMA_BASE + ", 'eagle_aux_hidden_state_layer_ids': [1, 4, 7], 'target_hidden_size': 8%s}";
    private static final String FC_NORM = ", 'fc_norm': true";
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
    public void bothNamesRouteToTheEagle3Row() {
        HfArchEntry qwen3 = entry(ARCH);
        assertSame(qwen3, entry("PeagleQwen3ForCausalLM"));
        assertEquals(Arrays.asList(ARCH, "PeagleQwen3ForCausalLM"), qwen3.hfNames());
        assertEquals(Arch.EAGLE3, qwen3.arch().resolve(
                config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE + "}"),
                HfTensorIndex.of(Collections.emptyMap())));
        assertTrue(HfArchRegistry.has(ARCH, HfArchEntry.Kind.TEXT));
    }

    @Test
    public void fixupDefaultsTheDraftVocabularyAndTheSingleDecoderLayer() {
        HfConfig fixed = HfEagle3Qwen3Hooks.fixup(config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE + "}"));
        assertEquals(10, fixed.requireInt("draft_vocab_size"));
        assertEquals(1, fixed.requireInt("num_hidden_layers"));
        // a checkpoint that declares both keeps them
        HfConfig kept = HfEagle3Qwen3Hooks.fixup(config(String.format(DRAFT, "")));
        assertEquals(4, kept.requireInt("draft_vocab_size"));
    }

    @Test
    public void unsupportedVariantsAreRefusedByName() {
        assertRefused("'norm_output': true", "norm_output");
        assertRefused("'use_aux_hidden_state': false", "use_aux_hidden_state");
        assertRefused("'num_aux_layers': 4", "num_aux_layers");
        assertRefused("'num_aux_hidden_states': 2", "num_aux_hidden_states");
        assertRefused("'layer_types': ['sliding_attention']", "sliding_attention");
        // the settings the drafter runs pass: the fusion projection and the pre-norm output state
        HfEagle3Qwen3Hooks.fixup(config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE
                + ", 'use_aux_hidden_state': true, 'norm_output': false, 'layer_types': ['full_attention']}"));
    }

    private void assertRefused(String extra, String named) {
        HfConfig cfg = config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE + ", " + extra + "}");
        GgufFormatException e = assertThrows(extra, GgufFormatException.class, () -> HfEagle3Qwen3Hooks.fixup(cfg));
        assertTrue(e.getMessage(), e.getMessage().contains(named));
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metadataCarriesTheDeclaredTapsAndTheLogitScale() throws Exception {
        HfMetadata md = meta(draft(", 'logit_scale': 1.5", empty()));
        assertEquals(Arrays.asList(1, 4, 7), md.get("eagle3.target_layers"));
        assertEquals(8, md.get("eagle3.target_hidden_size"));
        assertEquals(false, md.get("eagle3.norm_before_residual"));
        assertEquals(false, md.get("eagle3.norm_before_fc"));
        assertEquals(1.5f, ((Number) md.get("eagle3.logit_scale")).floatValue(), 1e-6f);
        // no logit_scale declared: the key is not written and the drafter scales by 1
        assertNull(meta(draft("", empty())).get("eagle3.logit_scale"));
    }

    @Test
    public void aDraftWithoutDeclaredTargetLayersFusesTheTargetsDefaultTaps() throws Exception {
        // vLLM's default layers of a 32-layer target: [2, 16, 29] (interfaces.py:1766-1787, llama.py:77-78)
        HfConfig cfg = config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE + "}");
        HfArchEntry e = entry(ARCH);
        HfLoadContext ctx = HfLoadContext.builder(e.configFixup().apply(cfg), e, Arch.EAGLE3)
                .index(empty()).targetPath(target()).build();
        HfMetadata md = meta(ctx);
        assertEquals(Arrays.asList(2, 16, 29), md.get("eagle3.target_layers"));
        assertEquals(8, md.get("eagle3.target_hidden_size"));
    }

    @Test
    public void theFcNormFlagAndThePlannedPerTapNormsMustAgree() throws Exception {
        HfLoadContext flagged = draft(", 'fc_norm': true", empty());
        GgufFormatException missing = assertThrows(GgufFormatException.class, () -> meta(flagged, planOf()));
        assertTrue(missing.getMessage(), missing.getMessage().contains("fc_norm.0.weight"));

        GgufFormatException unset = assertThrows(GgufFormatException.class,
                () -> meta(draft("", empty()), planOf("fc_norm.0.weight", "fc_norm.1.weight", "fc_norm.2.weight")));
        assertTrue(unset.getMessage(), unset.getMessage().contains("'fc_norm'"));

        GgufFormatException partial = assertThrows(GgufFormatException.class,
                () -> meta(flagged, planOf("fc_norm.0.weight", "fc_norm.1.weight")));
        assertTrue(partial.getMessage(), partial.getMessage().contains("fc_norm.2.weight"));

        GgufFormatException typed = assertThrows(GgufFormatException.class,
                () -> meta(draft(", 'fc_norm': 'yes'", empty()), planOf()));
        assertTrue(typed.getMessage(), typed.getMessage().contains("fc_norm"));
        assertTrue(typed.getMessage(), typed.getMessage().contains("boolean"));

        // the flag with all three norms is the layout vLLM builds
        HfMetadata md = meta(flagged, planOf("fc_norm.0.weight", "fc_norm.1.weight", "fc_norm.2.weight"));
        assertEquals(Arrays.asList(1, 4, 7), md.get("eagle3.target_layers"));
    }

    @Test
    public void nestedAuxLayersAreAccepted() throws Exception {
        HfMetadata md = meta(draft(", 'eagle_aux_hidden_state_layer_ids': null,"
                + " 'eagle_config': {'eagle_aux_hidden_state_layer_ids': [2, 5, 8], 'norm_before_fc': true}", empty()));
        assertEquals(Arrays.asList(2, 5, 8), md.get("eagle3.target_layers"));
        assertEquals(true, md.get("eagle3.norm_before_fc"));
    }

    private static HfMetadata meta(HfLoadContext ctx) throws IOException {
        return meta(ctx, planOf());
    }

    private static HfMetadata meta(HfLoadContext ctx, HfPlanView plan) throws IOException {
        assertEquals(Arch.EAGLE3, ctx.arch());
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, plan);
        return kv.build();
    }

    /** A plan view over the given GGUF names. */
    private static HfPlanView planOf(String... names) {
        Set<String> planned = new HashSet<>(Arrays.asList(names));
        return new HfPlanView() {
            @Override
            public List<String> names() {
                return new ArrayList<>(planned);
            }

            @Override
            public boolean has(String ggufName) {
                return planned.contains(ggufName);
            }

            @Override
            public long[] ne(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public GGMLType type(String ggufName) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public float[] readF32(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }

            @Override
            public long[] readI64(String ggufName, int maxElements) {
                throw new IllegalArgumentException(ggufName);
            }
        };
    }

    // ------------------------------------------------------------------ tensors

    @Test
    public void qwen3TensorsReachTheGgufNamesTheDrafterReads() throws IOException {
        HfTensorIndex index = checkpoint()
                .f32("model.layers.0.self_attn.q_norm.weight", new long[]{4}, seq(4))
                .f32("model.layers.0.self_attn.k_norm.weight", new long[]{4}, seq(4))
                .f32("model.layers.0.self_attn.q_proj.bias", new long[]{8}, seq(8))
                .f32("model.layers.0.hidden_norm.weight", new long[]{8}, seq(8))
                .f32("fc.weight", new long[]{8, 24}, seq(192))
                .f32("fc_norm.0.weight", new long[]{8}, seq(8))
                .f32("fc_norm.1.weight", new long[]{8}, seq(8))
                .f32("fc_norm.2.weight", new long[]{8}, seq(8))
                .i64("d2t", new long[]{4}, 0, 1, 5, 2)
                .index();
        Sink out = plan(draftOf(String.format(DRAFT, FC_NORM), index, target()));
        assertArrayEquals(seq(4), floats(out.get("blk.0.attn_q_norm.weight")), 0f);
        assertArrayEquals(seq(4), floats(out.get("blk.0.attn_k_norm.weight")), 0f);
        // the Q bias takes the same per-head permutation as the projection weight (LlamaModel.permute)
        assertArrayEquals(new float[]{0, 2, 1, 3, 4, 6, 5, 7}, floats(out.get("blk.0.attn_q.bias")), 0f);
        assertArrayEquals(seq(8), floats(out.get("blk.0.attn_norm_2.weight")), 0f);
        assertArrayEquals(seq(192), floats(out.get("fc.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("fc_norm.0.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("fc_norm.1.weight")), 0f);
        assertArrayEquals(seq(8), floats(out.get("fc_norm.2.weight")), 0f);
        assertArrayEquals(new long[]{0, 2, 7, 5}, longs(out.get("d2t")));
    }

    @Test
    public void anFcNormPartBeyondTheThreeTapsIsRefused() throws IOException {
        HfTensorIndex index = checkpoint().f32("fc_norm.3.weight", new long[]{8}, seq(8)).index();
        HfLoadContext ctx = draftOf(String.format(DRAFT, FC_NORM), index, target());
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> plan(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("fc_norm part 3"));
    }

    @Test
    public void theDraftVocabularyDefaultsToTheCheckpointVocabulary() throws IOException {
        // a checkpoint without draft_vocab_size has one d2t entry per target token
        HfTensorIndex index = checkpoint().i64("d2t", new long[]{TARGET_VOCAB}, new long[TARGET_VOCAB]).index();
        HfConfig cfg = config("{'architectures': ['PeagleQwen3ForCausalLM'], " + LLAMA_BASE
                + ", 'eagle_aux_hidden_state_layer_ids': [1, 4, 7]}");
        Sink out = plan(draftContext(cfg, index, target()));
        long[] identity = new long[TARGET_VOCAB];
        for (int i = 0; i < identity.length; i++) {
            identity[i] = i;
        }
        assertArrayEquals(identity, longs(out.get("d2t")));
    }

    @Test
    public void vocabularyIsTheTargetTokenizer() throws Exception {
        HfLoadContext ctx = draft("", empty());
        Map<String, Object> vocab = ctx.entry().vocab().build(ctx, new HfPreTokenizerResolver());
        assertEquals("gpt2", vocab.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(TARGET_VOCAB, ((List<?>) vocab.get(GgufKeys.TOKENIZER_LIST)).size());
    }

    @Test
    public void aCheckpointTargetIsReadFromItsConfig() throws Exception {
        Path dir = tmp.newFolder("hf-target").toPath();
        Files.write(dir.resolve("config.json"), ("{\"architectures\": [\"Qwen3ForCausalLM\"], \"hidden_size\": 32,"
                + " \"num_hidden_layers\": 28, \"num_attention_heads\": 2, \"num_key_value_heads\": 1,"
                + " \"intermediate_size\": 16, \"rms_norm_eps\": 1e-06, \"vocab_size\": 10,"
                + " \"max_position_embeddings\": 64}").getBytes(StandardCharsets.UTF_8));
        writeEmptySafetensors(dir);
        // the draft declares no target_hidden_size: it comes from the target checkpoint's config (32)
        HfConfig cfg = config("{'architectures': ['" + ARCH + "'], " + LLAMA_BASE
                + ", 'eagle_aux_hidden_state_layer_ids': [1, 4, 7]}");
        HfMetadata md = meta(draftContext(cfg, empty(), dir));
        assertEquals(32, md.get("eagle3.target_hidden_size"));
    }

    // ------------------------------------------------------------------ fixtures

    private Path target() throws IOException {
        Path p = tmp.getRoot().toPath().resolve("target.gguf");
        if (!p.toFile().exists()) {
            GgufTestWriter w = new GgufTestWriter().architecture("llama");
            w.u32(w.archKey("embedding_length"), 8).u32(w.archKey("block_count"), 32)
                    .u32(w.archKey("vocab_size"), TARGET_VOCAB)
                    .str(GgufKeys.TOKENIZER_MODEL, "gpt2")
                    .strArray(GgufKeys.TOKENIZER_LIST, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                    .writeTo(p);
        }
        return p;
    }

    private HfLoadContext draft(String extra, HfTensorIndex index) throws IOException {
        return draftOf(String.format(DRAFT, extra), index, target());
    }

    private HfLoadContext draftOf(String json, HfTensorIndex index, Path target) throws IOException {
        return draftContext(config(json), index, target);
    }

    private HfLoadContext draftContext(HfConfig cfg, HfTensorIndex index, Path target) throws IOException {
        HfArchEntry e = entry(cfg.architecture());
        Arch arch = e.arch().resolve(cfg, index);
        return HfLoadContext.builder(e.configFixup().apply(cfg), e, arch).index(index).targetPath(target).build();
    }

    private static HfArchEntry entry(String hfName) {
        return HfArchRegistry.resolve(HfLlamaHooksTest.config("{'architectures': ['" + hfName
                + "'], 'num_hidden_layers': 1}"), HfArchEntry.Kind.TEXT);
    }

    private static HfTensorIndex empty() {
        return HfTensorIndex.of(Collections.<String, RowSource>emptyMap());
    }

    private Checkpoint checkpoint() throws IOException {
        Checkpoint c = new Checkpoint(tmp.newFolder().toPath());
        closing.add(c);
        return c;
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
}
