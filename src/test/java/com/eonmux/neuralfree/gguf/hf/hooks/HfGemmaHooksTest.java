package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.DtypePolicy;
import com.eonmux.neuralfree.gguf.hf.tensor.ElementTransforms;
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
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.RecordingSink;
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

/**
 * {@link HfGemmaHooks} against {@code Gemma3Model} (LC/conversion/gemma.py:128-183): the norm shift, the token
 * embedding cut to the vocabulary, the metadata defaults and the vocabulary choice.
 */
public class HfGemmaHooksTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private static final String GEMMA3 = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'head_dim': 4, 'num_hidden_layers': 1, 'max_position_embeddings': 1024,"
            + " 'rms_norm_eps': 1e-06, 'sliding_window': 512, 'rope_theta': 1000000.0,"
            + " 'rope_local_base_freq': 10000.0";

    private static HfLoadContext gemma3(String members, Map<String, RowSource> tensors) {
        return context(Arch.GEMMA3, "Gemma3ForCausalLM", members, tensors);
    }

    private static HfMetadata meta(HfLoadContext ctx) {
        HfMetadataBuilder kv = new HfMetadataBuilder(ctx.arch().ggufName());
        HfGemmaHooks.meta(ctx, kv, null);
        return kv.build();
    }

    @Test
    public void everyNormWeightIsShiftedByOne() throws IOException {
        List<String> norms = Arrays.asList("model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight", "model.layers.0.pre_feedforward_layernorm.weight",
                "model.layers.0.post_feedforward_layernorm.weight", "model.layers.0.self_attn.q_norm.weight",
                "model.layers.0.self_attn.k_norm.weight", "model.norm.weight");
        Map<String, RowSource> t = new LinkedHashMap<>();
        for (String n : norms) {
            t.put(n, iota(4));
        }
        t.put("model.layers.0.self_attn.q_proj.weight", iota(8, 8));
        HfLoadContext ctx = gemma3(GEMMA3, t);
        RecordingSink out = planAll(ctx, HfGemmaHooks::tensors);
        assertEquals(norms.size() + 1, out.tensors.size());
        for (String n : norms) {
            assertArrayEquals(n, new float[]{1, 2, 3, 4}, values(out.tensors.get(ctx.names().mapOrThrow(n))), 0f);
        }
        assertSame("other tensors are not shifted", t.get("model.layers.0.self_attn.q_proj.weight"),
                out.tensors.get("blk.0.attn_q.weight"));
    }

    @Test
    public void aBf16NormIsShiftedInFloat32() throws IOException {
        float w = 1f / 256;
        RowSource bf16 = ElementTransforms.convert(RopeTensors.fromFloats(new long[]{1}, new float[]{w}),
                GGMLType.BF16);
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.norm.weight", bf16);
        HfLoadContext ctx = gemma3(GEMMA3, t);
        RecordingSink out = planAll(ctx, HfGemmaHooks::tensors);
        String gguf = ctx.names().mapOrThrow("model.norm.weight");
        RowSource shifted = out.tensors.get(gguf);
        assertEquals("the view keeps the stored format until the plan types it", GGMLType.BF16,
                shifted.format().ggmlType());
        RowSource planned = new DtypePolicy(n -> false, OutType.SOURCE).apply(gguf, shifted, null);
        assertEquals(GGMLType.F32, planned.format().ggmlType());
        assertArrayEquals("prepare_tensors hands modify_tensors BF16 tensors as F32: 1 + 2^-8 is exact in F32 and"
                + " would round to 1 in BF16", new float[]{1f + w}, values(planned), 0f);
    }

    @Test
    public void theEmbeddingKeepsTheRowsOfTheVocabulary() throws IOException {
        int vocab = 262144 + 64;
        Map<String, RowSource> exact = new LinkedHashMap<>();
        exact.put("model.embed_tokens.weight", iota(vocab, 1));
        RecordingSink kept = planAll(gemma3(GEMMA3, exact).withVocabTokenCount(vocab), HfGemmaHooks::tensors);
        assertArrayEquals("262144 pieces + 64 added tokens keep all 262208 rows", new long[]{vocab, 1},
                kept.tensors.get("token_embd.weight").shape());

        Map<String, RowSource> padded = new LinkedHashMap<>();
        padded.put("model.embed_tokens.weight", iota(vocab + 92, 1));
        RecordingSink cut = planAll(gemma3(GEMMA3, padded).withVocabTokenCount(vocab), HfGemmaHooks::tensors);
        RowSource embd = cut.tensors.get("token_embd.weight");
        assertArrayEquals("the out-of-vocabulary rows are removed", new long[]{vocab, 1}, embd.shape());
        float[] v = values(embd);
        assertEquals(vocab - 1, v[v.length - 1], 0f);
    }

    @Test
    public void theEmbeddingNeedsTheVocabularyCount() {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("model.embed_tokens.weight", iota(16, 2));
        HfLoadContext ctx = gemma3(GEMMA3, t);
        assertThrows(IllegalStateException.class, () -> planAll(ctx, HfGemmaHooks::tensors));
    }

    @Test
    public void lmHeadIsKept() throws IOException {
        Map<String, RowSource> t = new LinkedHashMap<>();
        t.put("lm_head.weight", iota(16, 8));
        RecordingSink out = planAll(gemma3(GEMMA3, t), HfGemmaHooks::tensors);
        assertSame(t.get("lm_head.weight"), out.tensors.get("output.weight"));
    }

    @Test
    public void metadataFillsTheConverterDefaults() {
        HfMetadata m = meta(gemma3("'hidden_size': 8, 'num_hidden_layers': 1, 'sliding_window': 512", Map.of()));
        assertEquals(131072, m.get("gemma3.context_length"));
        assertEquals(8, m.get("gemma3.attention.head_count"));
        assertEquals(4, m.get("gemma3.attention.head_count_kv"));
        assertEquals(1e-6f, m.get("gemma3.attention.layer_norm_rms_epsilon"));
        assertEquals(256, m.get("gemma3.attention.key_length"));
        assertEquals(256, m.get("gemma3.attention.value_length"));
        assertEquals(1e6f, m.get("gemma3.rope.freq_base"));
        assertEquals(512, m.get("gemma3.attention.sliding_window"));
    }

    @Test
    public void metadataReadsTheConfiguredValuesAndTheLocalRopeBase() {
        HfMetadata m = meta(gemma3(GEMMA3 + ", 'final_logit_softcapping': 30.0", Map.of()));
        assertEquals(1024, m.get("gemma3.context_length"));
        assertEquals(2, m.get("gemma3.attention.head_count"));
        assertEquals(1, m.get("gemma3.attention.head_count_kv"));
        assertEquals(4, m.get("gemma3.attention.key_length"));
        assertEquals(1e6f, m.get("gemma3.rope.freq_base"));
        assertEquals(10000f, m.get("gemma3.rope.freq_base_swa"));
        assertEquals(30f, m.get("gemma3.final_logit_softcapping"));
    }

    @Test
    public void aPatternOfOneMeansNoSlidingWindow() {
        HfMetadata m = meta(gemma3(GEMMA3 + ", 'sliding_window_pattern': 1", Map.of()));
        assertFalse(m.keys().contains("gemma3.attention.sliding_window"));
        HfMetadata six = meta(gemma3(GEMMA3 + ", 'sliding_window_pattern': 6", Map.of()));
        assertEquals(512, six.get("gemma3.attention.sliding_window"));
    }

    @Test
    public void attentionLogitSoftcappingIsRejected() {
        HfLoadContext ctx = gemma3(GEMMA3 + ", 'attn_logit_softcapping': 50.0", Map.of());
        assertThrows(GgufFormatException.class, () -> meta(ctx));
    }

    @Test
    public void withoutTokenizerModelTheVocabularyIsGpt2() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir, "tokenizer.json", "{'added_tokens':[{'id':0,'content':'<bos>','special':true,'normalized':false}],"
                + "'normalizer':null,"
                + "'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,'trim_offsets':true,'use_regex':true},"
                + "'post_processor':null,"
                + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,'use_regex':true},"
                + "'model':{'type':'BPE','unk_token':null,'byte_fallback':false,'ignore_merges':false,"
                + "'vocab':{'<bos>':0,'a':1,'b':2,'ab':3},'merges':['a b']}}");
        write(dir, "tokenizer_config.json", "{'bos_token':'<bos>'}");
        HfArchEntry entry = HfArchEntry.of(List.of("Gemma3ForCausalLM"), HfArchResolver.of(Arch.GEMMA3),
                HfMetadataHook.GENERIC, HfGemmaHooks::tensors, HfGemmaHooks::vocab);
        HfLoadContext ctx = HfLoadContext.builder(HfConfigLoader.fromJson(dir, json("{'architectures':"
                + " ['Gemma3ForCausalLM'], " + GEMMA3 + ", 'vocab_size': 4}")), entry, Arch.GEMMA3)
                .tokenizerFiles(HfTokenizerFiles.locate(dir)).build();
        Map<String, Object> md = HfGemmaHooks.vocab(ctx, (tj, encode) -> "gpt-2");
        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        assertEquals(Arrays.asList("<bos>", "a", "b", "ab"), md.get(GgufKeys.TOKENIZER_LIST));
        assertEquals(List.of("a b"), md.get(GgufKeys.TOKENIZER_MERGES));
        assertFalse("only the SentencePiece branch sets the space prefix",
                md.containsKey(GgufKeys.TOKENIZER_ADD_PREFIX));
    }

    private static void write(Path dir, String name, String singleQuoted) throws IOException {
        Files.write(dir.resolve(name), json(singleQuoted).getBytes(StandardCharsets.UTF_8));
    }
}
