package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.config.HfConfig;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.st.StRowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfPreTokenizerResolver;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.io.safetensors.SafetensorsFile;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import com.eonmux.neuralfree.io.safetensors.StTensor;
import org.junit.After;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfLlamaHooks} on synthetic checkpoints written with {@link SafetensorsWriter}: the table registration, the
 * EAGLE3 switch, the configuration fixup and name filter, the Q/K permutation (and its absence when the entry does not
 * undo it), Mixtral expert stacking, llama3 {@code rope_freqs}, the Llama metadata and the vocabulary dispatch's
 * post-steps. Also holds the fixtures the other hook tests share.
 */
public class HfLlamaHooksTest {

    static final String LLAMA_BASE = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'num_hidden_layers': 1, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-05, 'vocab_size': 10";

    private static final List<String> LLAMA_NAMES = Arrays.asList("IQuestCoderForCausalLM", "LLaMAForCausalLM",
            "LlamaForCausalLM", "LlamaModel", "Eagle3DraftModel", "Eagle3Speculator", "Eagle3LlamaForCausalLM",
            "LlamaForCausalLMEagle3", "LlavaForConditionalGeneration", "MistralForCausalLM", "MixtralForCausalLM",
            "VLlama3ForCausalLM", "VoxtralForConditionalGeneration", "Eagle3MiniMaxM2ForCausalLM",
            "Eagle3Qwen2_5vlForCausalLM", "Eagle3Qwen3vlForCausalLM", "PEagleDraftModel", "PeagleLlamaForCausalLM");

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private Checkpoint ckpt;

    @After
    public void close() throws IOException {
        if (ckpt != null) {
            ckpt.close();
        }
    }

    private Checkpoint checkpoint() {
        ckpt = new Checkpoint(tmp.getRoot().toPath());
        return ckpt;
    }

    // ------------------------------------------------------------------ registration and architecture

    @Test
    public void tableRegistersTheLlamaNamesAndVllmAliasesInOneEntry() {
        HfArchEntry llama = entry("LlamaForCausalLM");
        assertEquals(LLAMA_NAMES.size(), llama.hfNames().size());
        for (String name : LLAMA_NAMES) {
            assertSame(name, llama, entry(name));
        }
        assertTrue(llama.undoPermute());
        HfArchEntry mistral = HfArchRegistry.resolve(config("{'architectures': ['" + HfConfigLoader.MISTRAL_NATIVE
                + "'], " + LLAMA_BASE + "}"), HfArchEntry.Kind.TEXT);
        assertFalse(mistral.undoPermute());
        assertEquals(Arrays.asList(HfConfigLoader.MISTRAL_NATIVE), mistral.hfNames());
    }

    @Test
    public void eagle3NeedsDraftVocabAndOneLayerOrAnAliasName() {
        assertEquals(Arch.EAGLE3, HfLlamaHooks.arch(config("{'architectures': ['LlamaForCausalLMEagle3'],"
                + " 'draft_vocab_size': 4, 'num_hidden_layers': 1}"), null));
        assertEquals(Arch.LLAMA, HfLlamaHooks.arch(config("{'architectures': ['LlamaForCausalLMEagle3'],"
                + " 'draft_vocab_size': 4, 'num_hidden_layers': 2}"), null));
        assertEquals(Arch.LLAMA, HfLlamaHooks.arch(config("{'architectures': ['LlamaForCausalLM'],"
                + " 'num_hidden_layers': 1}"), null));
        assertEquals(Arch.EAGLE3, HfLlamaHooks.arch(config("{'architectures': ['Eagle3Speculator'],"
                + " 'draft_vocab_size': 4, 'num_hidden_layers': 32,"
                + " 'transformer_layer_config': {'num_hidden_layers': 1}}"), null));
        for (String alias : HfEagle3Hooks.VLLM_ALIASES) {
            assertEquals(alias, Arch.EAGLE3, HfLlamaHooks.arch(config("{'architectures': ['" + alias + "'],"
                    + " 'num_hidden_layers': 2}"), null));
        }
        assertThrows(GgufFormatException.class, () -> HfLlamaHooks.arch(config(
                "{'architectures': ['LlamaForCausalLM'], 'draft_vocab_size': 4}"), null));
    }

    @Test
    public void fixupDefaultsTheSmolVlm2HeadCount() {
        HfConfig fixed = HfLlamaHooks.fixup(config("{'architectures': ['VLlama3ForCausalLM'], 'hidden_size': 64,"
                + " 'num_hidden_layers': 1}"));
        assertEquals(32, fixed.requireInt("num_attention_heads"));
        HfConfig kept = HfLlamaHooks.fixup(config("{'architectures': ['VLlama3ForCausalLM'],"
                + " 'num_attention_heads': 4, 'num_hidden_layers': 1}"));
        assertEquals(4, kept.requireInt("num_attention_heads"));
        HfConfig other = HfLlamaHooks.fixup(config("{'architectures': ['LlamaForCausalLM'], 'num_hidden_layers': 1}"));
        assertFalse(other.flat().containsKey("num_attention_heads"));
    }

    @Test
    public void filterRemovesTheSmolVlmTextPrefixBeforeTheInheritedFilters() {
        List<String> seen = new ArrayList<>();
        UnaryOperator<String> base = n -> {
            seen.add(n);
            return n;
        };
        assertEquals("model.layers.0.mlp.up_proj.weight",
                HfLlamaHooks.FILTER.filter("model.text_model.layers.0.mlp.up_proj.weight", base));
        assertEquals("lm_head.weight", HfLlamaHooks.filter("lm_head.weight", base));
        assertEquals(Arrays.asList("model.layers.0.mlp.up_proj.weight", "lm_head.weight"), seen);
        assertNull(HfLlamaHooks.filter("x.weight", n -> null));
    }

    // ------------------------------------------------------------------ tensors

    @Test
    public void qAndKAreUnpermutedForLlama() throws IOException {
        HfTensorIndex index = checkpoint().f32("model.layers.0.self_attn.q_proj.weight", new long[]{8, 2}, seq(16))
                .f32("model.layers.0.self_attn.k_proj.weight", new long[]{4, 2}, seq(8))
                .f32("model.layers.0.self_attn.q_proj.bias", new long[]{8}, seq(8))
                .f32("model.layers.0.self_attn.v_proj.weight", new long[]{4, 2}, seq(8)).index();
        Sink out = plan(context(config("{'architectures': ['LlamaForCausalLM'], " + LLAMA_BASE + "}"),
                entry("LlamaForCausalLM"), index));
        assertArrayEquals(rows(seq(16), 2, 0, 2, 1, 3, 4, 6, 5, 7), floats(out.get("blk.0.attn_q.weight")), 0f);
        assertArrayEquals(rows(seq(8), 2, 0, 2, 1, 3), floats(out.get("blk.0.attn_k.weight")), 0f);
        assertArrayEquals(rows(seq(8), 1, 0, 2, 1, 3, 4, 6, 5, 7), floats(out.get("blk.0.attn_q.bias")), 0f);
        assertArrayEquals(seq(8), floats(out.get("blk.0.attn_v.weight")), 0f);
    }

    @Test
    public void headlessLlamaModelGetsTheModelPrefix() throws IOException {
        HfTensorIndex index = checkpoint().f32("layers.0.self_attn.q_proj.weight", new long[]{8, 2}, seq(16))
                .f32("embed_tokens.weight", new long[]{10, 8}, seq(80)).index();
        Sink out = plan(context(config("{'architectures': ['LlamaModel'], " + LLAMA_BASE + "}"),
                entry("LlamaModel"), index));
        assertArrayEquals(rows(seq(16), 2, 0, 2, 1, 3, 4, 6, 5, 7), floats(out.get("blk.0.attn_q.weight")), 0f);
        assertTrue(out.added.containsKey("token_embd.weight"));
    }

    @Test
    public void noPermutationWhenTheEntryKeepsQAndKAsStored() throws IOException {
        HfTensorIndex index = checkpoint().f32("model.layers.0.self_attn.q_proj.weight", new long[]{8, 2}, seq(16))
                .index();
        Sink out = plan(context(config("{'architectures': ['LlamaForCausalLM'], " + LLAMA_BASE + "}"),
                entry("LlamaForCausalLM").withUndoPermute(false), index));
        assertArrayEquals(seq(16), floats(out.get("blk.0.attn_q.weight")), 0f);
    }

    @Test
    public void mixtralExpertsStackPerLayerInExpertOrderAndConsumeTheParts() throws IOException {
        Checkpoint c = checkpoint();
        List<String> names = new ArrayList<>();
        for (int x = 0; x < 2; x++) {
            for (String w : Arrays.asList("w1", "w2", "w3")) {
                String name = "model.layers.0.block_sparse_moe.experts." + x + "." + w + ".weight";
                names.add(name);
                c.f32(name, new long[]{2, 2}, fill(4, 10 * x + part(w)));
            }
        }
        c.f32("model.layers.0.block_sparse_moe.gate.weight", new long[]{2, 8}, seq(16));
        HfTensorIndex index = c.index();
        Sink out = plan(context(config("{'architectures': ['MixtralForCausalLM'], 'num_local_experts': 2, "
                + LLAMA_BASE + "}"), entry("MixtralForCausalLM"), index));
        assertEquals(Arrays.asList("blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                "blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_inp.weight"), new ArrayList<>(out.added.keySet()));
        RowSource gate = out.get("blk.0.ffn_gate_exps.weight");
        assertArrayEquals(new long[]{2, 2, 2}, gate.shape());
        assertArrayEquals(concat(fill(4, 1), fill(4, 11)), floats(gate), 0f);
        assertArrayEquals(concat(fill(4, 2), fill(4, 12)), floats(out.get("blk.0.ffn_down_exps.weight")), 0f);
        assertArrayEquals(concat(fill(4, 3), fill(4, 13)), floats(out.get("blk.0.ffn_up_exps.weight")), 0f);
        assertEquals(new LinkedHashSet<>(names.subList(0, names.size() - 1)), out.consumed);
    }

    @Test
    public void mixtralLayerWithAMissingExpertFails() throws IOException {
        HfTensorIndex index = checkpoint()
                .f32("model.layers.0.block_sparse_moe.experts.0.w1.weight", new long[]{2, 2}, seq(4))
                .f32("model.layers.0.block_sparse_moe.experts.0.w2.weight", new long[]{2, 2}, seq(4))
                .f32("model.layers.0.block_sparse_moe.experts.0.w3.weight", new long[]{2, 2}, seq(4)).index();
        HfLoadContext ctx = context(config("{'architectures': ['MixtralForCausalLM'], 'num_local_experts': 2, "
                + LLAMA_BASE + "}"), entry("MixtralForCausalLM"), index);
        assertThrows(GgufFormatException.class, () -> plan(ctx));
    }

    @Test
    public void llama3RopeTypeAddsRopeFreqs() throws IOException {
        String rope = "'rope_theta': 500000.0, 'rope_scaling': {'rope_type': 'llama3', 'factor': 8.0,"
                + " 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192}";
        String base = "'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 2,"
                + " 'num_hidden_layers': 1, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-05, 'vocab_size': 10";
        HfTensorIndex index = HfTensorIndex.of(Collections.<String, RowSource>emptyMap());
        Sink out = plan(context(config("{'architectures': ['LlamaForCausalLM'], 'hidden_size': 256, " + base + ", "
                + rope + "}"), entry("LlamaForCausalLM"), index));
        float[] freqs = floats(out.get("rope_freqs.weight"));
        assertEquals(64, freqs.length);
        for (int i = 0; i < freqs.length; i++) {
            assertEquals("factor " + i, llama3Factor(500000.0, 128, i, 8.0, 1.0, 4.0, 8192), freqs[i], 1e-4);
        }
        assertEquals(1f, freqs[0], 0f);
        assertEquals(8f, freqs[freqs.length - 1], 0f);
        Sink plain = plan(context(config("{'architectures': ['LlamaForCausalLM'], 'hidden_size': 256, " + base
                + "}"), entry("LlamaForCausalLM"), index));
        assertTrue(plain.added.isEmpty());
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void metaAddsVocabSizeAndRopeDimension() throws IOException {
        HfMetadata m = meta(config("{'architectures': ['LlamaForCausalLM'], " + LLAMA_BASE + "}"));
        assertEquals(10, m.get("llama.vocab_size"));
        assertEquals(4, m.get("llama.rope.dimension_count"));
        assertEquals(GGUFValueType.UINT32, m.type("llama.rope.dimension_count"));
        HfMetadata withHeadDim = meta(config("{'architectures': ['LlamaForCausalLM'], 'head_dim': 6, "
                + LLAMA_BASE + "}"));
        assertEquals(6, withHeadDim.get("llama.rope.dimension_count"));
        assertEquals(6, withHeadDim.get("llama.attention.key_length"));
    }

    private HfMetadata meta(HfConfig cfg) throws IOException {
        HfLoadContext ctx = context(cfg, entry(cfg.architecture()),
                HfTensorIndex.of(Collections.<String, RowSource>emptyMap()));
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        ctx.entry().meta().apply(ctx, kv, null);
        return kv.build();
    }

    // ------------------------------------------------------------------ vocabulary

    @Test
    public void vocabularyPostStepsFollowTheConfiguredVocabularySize() throws Exception {
        Path dir = tmp.newFolder("model").toPath();
        Files.write(dir.resolve("tokenizer_config.json"),
                "{\"add_prefix_space\": false, \"eot_token\": \"<EOT>\"}".getBytes(StandardCharsets.UTF_8));
        Path gguf = tmp.getRoot().toPath().resolve("tokenizer.gguf");
        new GgufTestWriter().architecture("llama").str(GgufKeys.TOKENIZER_MODEL, "llama")
                .strArray(GgufKeys.TOKENIZER_LIST, "<unk>", "<s>", "</s>").writeTo(gguf);
        HfTokenizerFiles files = HfTokenizerFiles.locate(dir).withTokenizerDir(gguf);

        Map<String, Object> codeLlama = vocab(files, "{'architectures': ['LlamaForCausalLM'], 'eot_token_id': 7, "
                + LLAMA_BASE.replace("'vocab_size': 10", "'vocab_size': 32016") + "}");
        assertEquals("llama", codeLlama.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(Boolean.FALSE, codeLlama.get(GgufKeys.TOKENIZER_ADD_PREFIX));
        assertEquals(7, codeLlama.get(GgufKeys.TOKENIZER_EOT_ID));
        assertFalse(codeLlama.containsKey(GgufKeys.TOKENIZER_ADD_BOS));

        Map<String, Object> fallback = vocab(files, "{'architectures': ['LlamaForCausalLM'], "
                + LLAMA_BASE.replace("'vocab_size': 10", "'vocab_size': 32016") + "}");
        assertEquals(32010, fallback.get(GgufKeys.TOKENIZER_EOT_ID));

        Map<String, Object> granite = vocab(files, "{'architectures': ['LlamaForCausalLM'], "
                + LLAMA_BASE.replace("'vocab_size': 10", "'vocab_size': 49152") + "}");
        assertEquals(Boolean.FALSE, granite.get(GgufKeys.TOKENIZER_ADD_BOS));
        assertFalse(granite.containsKey(GgufKeys.TOKENIZER_EOT_ID));
    }

    private Map<String, Object> vocab(HfTokenizerFiles files, String json) throws Exception {
        HfConfig cfg = config(json);
        HfArchEntry e = entry(cfg.architecture());
        HfLoadContext ctx = HfLoadContext.builder(cfg, e, Arch.LLAMA).tokenizerFiles(files).build();
        return e.vocab().build(ctx, new HfPreTokenizerResolver());
    }

    // ------------------------------------------------------------------ shared fixtures

    /** A configuration from single-quoted JSON. */
    static HfConfig config(String singleQuotedJson) {
        return HfConfigLoader.fromJson(Path.of("."), singleQuotedJson.replace('\'', '"'));
    }

    /** The registered text entry of {@code hfName}. */
    static HfArchEntry entry(String hfName) {
        return HfArchRegistry.resolve(config("{'architectures': ['" + hfName + "'], 'num_hidden_layers': 1}"),
                HfArchEntry.Kind.TEXT);
    }

    /** The context the assembler derives: resolve on the raw configuration, then the entry's fixup. */
    static HfLoadContext context(HfConfig cfg, HfArchEntry entry, HfTensorIndex index) {
        Arch arch = entry.arch().resolve(cfg, index);
        return HfLoadContext.builder(entry.configFixup().apply(cfg), entry, arch).index(index).build();
    }

    /** Runs the entry's tensor hook over the index in order, skipping consumed names, then its extra tensors. */
    static Sink plan(HfLoadContext ctx) throws IOException {
        Sink out = new Sink();
        for (String name : ctx.index().names()) {
            if (!out.isConsumed(name)) {
                ctx.entry().tensors().apply(ctx, name, ctx.index().get(name), out);
            }
        }
        ctx.entry().extra().generate(ctx, out);
        return out;
    }

    static float[] seq(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = i;
        }
        return v;
    }

    static float[] fill(int n, float value) {
        float[] v = new float[n];
        Arrays.fill(v, value);
        return v;
    }

    static float[] concat(float[] a, float[] b) {
        float[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /** {@code src}'s rows of {@code width} elements in the order {@code order}. */
    static float[] rows(float[] src, int width, int... order) {
        float[] out = new float[order.length * width];
        for (int r = 0; r < order.length; r++) {
            System.arraycopy(src, order[r] * width, out, r * width, width);
        }
        return out;
    }

    private static int part(String w) {
        return w.charAt(1) - '0';
    }

    /** The llama3 factor of frequency {@code i} for a rotary width {@code dim}, in double precision. */
    private static double llama3Factor(double base, int dim, int i, double factor, double low, double high, int ctx) {
        double freq = 1.0 / Math.pow(base, (2.0 * i) / dim);
        double wavelen = 2 * Math.PI / freq;
        if (wavelen < ctx / high) {
            return 1.0;
        }
        if (wavelen > ctx / low) {
            return factor;
        }
        double smooth = (ctx / wavelen - low) / (high - low);
        return 1 / ((1 - smooth) / factor + smooth);
    }

    static byte[] bytes(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(Math.toIntExact(s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, Math.toIntExact(s.rows()), b);
        assertEquals(b.capacity(), b.position());
        return b.array();
    }

    /** The values of an F32 source. */
    static float[] floats(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(bytes(s)).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[b.capacity() / Float.BYTES];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat(i * Float.BYTES);
        }
        return out;
    }

    /** The values of an I64 source. */
    static long[] longs(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(bytes(s)).order(ByteOrder.LITTLE_ENDIAN);
        long[] out = new long[b.capacity() / Long.BYTES];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getLong(i * Long.BYTES);
        }
        return out;
    }

    /** A {@link TensorSink} that records what a hook plans. */
    static final class Sink implements TensorSink {
        final Map<String, RowSource> added = new LinkedHashMap<>();
        final Set<String> consumed = new LinkedHashSet<>();
        final Map<String, String> dropped = new LinkedHashMap<>();

        RowSource get(String ggufName) {
            RowSource s = added.get(ggufName);
            if (s == null) {
                throw new AssertionError("'" + ggufName + "' was not planned; planned " + added.keySet());
            }
            return s;
        }

        @Override
        public void add(String ggufName, RowSource src) {
            if (added.put(ggufName, src) != null) {
                throw new AssertionError("'" + ggufName + "' planned twice");
            }
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            throw new AssertionError("unexpected KV " + key);
        }

        @Override
        public void drop(String hfName, String reason) {
            dropped.put(hfName, reason);
        }

        @Override
        public boolean isConsumed(String hfName) {
            return consumed.contains(hfName);
        }

        @Override
        public void consume(String hfName) {
            consumed.add(hfName);
        }
    }

    /** A one-file checkpoint written with {@link SafetensorsWriter} and opened as an index. */
    static final class Checkpoint implements Closeable {
        private final Path dir;
        private final SafetensorsWriter.Builder builder = SafetensorsWriter.builder();
        private final List<SafetensorsFile> open = new ArrayList<>();
        private final List<String> order = new ArrayList<>();
        private int files;

        Checkpoint(Path dir) {
            this.dir = dir;
        }

        Checkpoint add(String name, StDType type, long[] shape, byte[] data) {
            builder.add(name, type, shape, data);
            order.add(name);
            return this;
        }

        Checkpoint f32(String name, long[] shape, float... v) {
            ByteBuffer b = ByteBuffer.allocate(Float.BYTES * v.length).order(ByteOrder.LITTLE_ENDIAN);
            for (float x : v) {
                b.putFloat(x);
            }
            return add(name, StDType.F32, shape, b.array());
        }

        Checkpoint i64(String name, long[] shape, long... v) {
            ByteBuffer b = ByteBuffer.allocate(Long.BYTES * v.length).order(ByteOrder.LITTLE_ENDIAN);
            for (long x : v) {
                b.putLong(x);
            }
            return add(name, StDType.I64, shape, b.array());
        }

        /** Writes the tensors added so far to a new file and indexes them in insertion order. */
        HfTensorIndex index() throws IOException {
            Path p = dir.resolve("model-" + files++ + ".safetensors");
            builder.build().write(p);
            SafetensorsFile file = SafetensorsFile.open(p);
            open.add(file);
            Map<String, StTensor> byName = new LinkedHashMap<>();
            for (StTensor t : file.tensors()) {
                byName.put(t.name(), t);
            }
            Map<String, RowSource> m = new LinkedHashMap<>();
            for (String name : order) {
                m.put(name, StRowSource.of(file, byName.get(name)));
            }
            return HfTensorIndex.of(m);
        }

        @Override
        public void close() throws IOException {
            for (SafetensorsFile f : open) {
                f.close();
            }
            open.clear();
        }
    }
}
