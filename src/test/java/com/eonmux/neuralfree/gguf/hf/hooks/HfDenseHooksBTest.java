package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.gguf.tokenizer.hf.PreNameResolver;
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
import java.util.function.UnaryOperator;

import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.RecordingSink;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.context;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.iota;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.json;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.planAll;
import static com.eonmux.neuralfree.gguf.hf.hooks.HfBaseDenseHooksTest.values;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfDenseHooksB} against the converter classes it ports: the Qwen (v1) tiktoken vocabulary
 * (LC/conversion/base.py:2240-2294), the {@code thinker.} strip of Qwen2-VL and Qwen3-VL (qwenvl.py:37-40,
 * qwen3vl.py:294-300), the deepstack count and its ASR template and bos overrides (qwen3vl.py:280-396) and the
 * SenseNova U1 language-model filter, q/k norm concatenation and spatial RoPE base (sensenova_u1.py:16-44).
 */
public class HfDenseHooksBTest {

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    private static final String QWEN2 = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-06, 'vocab_size': 8, 'rope_theta': 1000000.0";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ Qwen (v1)

    @Test
    public void qwen1VocabReadsTheTiktokenRanks() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir, "qwen.tiktoken", base64("ab") + " 0\n" + base64("a") + " 1\n" + base64("b") + " 2\n");
        write(dir, "tokenizer_config.json", json("{'name':'Qwen'}"));
        Files.writeString(dir.resolve("config.json"), json("{'architectures':['QWenLMHeadModel'],"
                        + "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
                        + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'rms_norm_eps': 1e-06,"
                        + " 'vocab_size': 4}"),
                StandardCharsets.UTF_8);
        HfLoadContext ctx = context(Arch.QWEN, "QWenLMHeadModel", dir, QWEN2.replace("'vocab_size': 8",
                "'vocab_size': 4"));
        Map<String, Object> md = HfDenseHooksB.qwenVocab(ctx, (tj, encode) -> "qwen2");
        assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));
        List<?> tokens = (List<?>) md.get(GgufKeys.TOKENIZER_LIST);
        assertEquals(4, tokens.size());
        assertEquals("ab", tokens.get(0));
        assertEquals("a", tokens.get(1));
    }

    /** The base64 of {@link java.util.Base64} (the encoder the tiktoken ranks use). */
    private static String base64(String s) {
        return java.util.Base64.getEncoder().encodeToString(s.getBytes(StandardCharsets.UTF_8));
    }

    private static void write(Path dir, String name, String text) throws IOException {
        Files.write(dir.resolve(name), text.getBytes(StandardCharsets.UTF_8));
    }

    /** A context over a model directory, with the checkpoint's own tokenizer files. */
    static HfLoadContext context(Arch arch, String hfArch, Path dir, String members) throws IOException {
        HfArchEntry entry = HfArchEntry.of(List.of(hfArch), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        String config = "{'architectures': ['" + hfArch + "'], " + members + "}";
        return HfLoadContext.builder(HfConfigLoader.fromJson(dir, json(config)), entry, arch)
                .options(ModelSourceOptions.builder().build())
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .index(HfTensorIndex.of(Map.of()))
                .build();
    }

    private static HfMetadata metaOf(HfLoadContext ctx, String hfArch) throws IOException {
        HfMetadataBuilder kv = new HfMetadataBuilder(hfArch.isEmpty() ? ctx.arch().ggufName() : ctx.arch().ggufName());
        return kv.build();
    }

    // ------------------------------------------------------------------ thinker strip

    @Test
    public void thinkerPrefixIsStrippedWhereverItOccurs() {
        assertNull(HfDenseHooksB.THINKER_FILTER.filter("language_model.thinker.model.layers.0.self_attn"
                + ".q_proj.weight", IDENTITY));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfDenseHooksB.THINKER_FILTER.filter("model.thinker.layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("model.layers.0.self_attn.q_proj.weight",
                HfDenseHooksB.THINKER_FILTER.filter("model.layers.0.self_attn.q_proj.weight", IDENTITY));
    }

    // ------------------------------------------------------------------ Qwen3-VL

    @Test
    public void qwen3VlMetaCountsTheDeepstackIndexes() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3VL, "Qwen3VLForConditionalGeneration", QWEN2
                + ", 'vision_config': {'deepstack_visual_indexes': [6, 12]}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3VL.ggufName());
        HfDenseHooksB.qwen3VlMeta(ctx, kv, null);
        assertEquals(2, kv.build().get(String.format(GgufKeys.NUM_DEEPSTACK_LAYERS, Arch.QWEN3VL.ggufName())));
    }

    private HfLoadContext context(Arch arch, String hfArch, String members) throws IOException {
        return context(arch, hfArch, tmp.newFolder().toPath(), members);
    }

    @Test
    public void deepstackCountAlsoReadsTheThinkerVisionConfig() throws IOException {
        Map<String, Object> flat = com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader.fromJson(
                tmp.newFolder().toPath(), json("{'architectures': ['Qwen3VLForConditionalGeneration'],"
                        + " 'thinker_config': {'vision_config': {'deepstack_visual_indexes': [1]}}}")).flat();
        HfLoadContext ctx = context(Arch.QWEN3VL, "Qwen3VLForConditionalGeneration", QWEN2);
        assertEquals(1, HfDenseHooksB.deepstackCount(com.eonmux.neuralfree.gguf.hf.hooks.HfDenseHooksBTest
                .configOf(flat)));
        assertEquals(0, HfDenseHooksB.deepstackCount(ctx.config()));
    }

    private static com.eonmux.neuralfree.gguf.hf.config.HfConfig configOf(Map<String, Object> flat)
            throws IOException {
        return com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader.fromJson(
                java.nio.file.Path.of("."), json("{'architectures': ['Qwen3VLForConditionalGeneration'],"
                        + " 'thinker_config': {'vision_config': {'deepstack_visual_indexes': [1]}}}"));
    }

    // ------------------------------------------------------------------ Qwen3ASRTextModel

    @Test
    public void asrMetaWritesNoDeepstack() throws IOException {
        HfLoadContext ctx = context(Arch.QWEN3VL, "Qwen3ASRForConditionalGeneration", QWEN2);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3VL.ggufName());
        HfDenseHooksB.asrMeta(ctx, kv, null);
        assertEquals(0, kv.build().get(String.format(GgufKeys.NUM_DEEPSTACK_LAYERS, Arch.QWEN3VL.ggufName())));
    }

    @Test
    public void asrVocabSetsTheChatmlTemplateAndTheImEndBosEos() throws IOException {
        Path dir = tmp.newFolder().toPath();
        write(dir, "tokenizer.json", json("{'added_tokens':[{'id':7,'content':'<|im_end|>','special':true,"
                + "'normalized':false}],'normalizer':null,'pre_tokenizer':null,'post_processor':null,"
                + "'decoder':null,'model':{'type':'BPE','unk_token':null,'byte_fallback':false,"
                + "'vocab':{'<|im_end|>':7,'<|endoftext|>':8,'a':2},'merges':[]}}"));
        write(dir, "tokenizer_config.json", json("{'eos_token':'<|im_end|>',"
                + "'added_tokens_decoder':{'151645':{'content':'<|im_end|>','special':true}}}"));
        HfLoadContext ctx = context(Arch.QWEN3VL, "Qwen3ASRForConditionalGeneration", dir, QWEN2);
        Map<String, Object> md = HfDenseHooksB.asrVocab(ctx, (tj, encode) -> "qwen2");
        assertEquals(HfDenseHooksB.CHATML_TEMPLATE, md.get(GgufKeys.TOKENIZER_CHAT_TEMPLATE));
        assertEquals(151645, md.get(GgufKeys.TOKENIZER_BOS_ID));
        assertEquals(151645, md.get(GgufKeys.TOKENIZER_EOS_ID));
    }

    // ------------------------------------------------------------------ SenseNovaU1Model

    @Test
    public void sensenovaKeepsOnlyTheLanguageModel() {
        assertNull(HfDenseHooksB.SENSENOVA_FILTER.filter("vision.branch.full_block_mot_gen.mlp.0.weight",
                IDENTITY));
        assertNull(HfDenseHooksB.SENSENOVA_FILTER.filter("model.layers.0.self_attn.q_proj.weight", IDENTITY));
        assertEquals("language_model.model.layers.0.self_attn.q_proj.weight",
                HfDenseHooksB.SENSENOVA_FILTER.filter("language_model.model.layers.0.self_attn.q_proj.weight",
                        IDENTITY));
    }

    @Test
    public void sensenovaConcatenatesEveryQkNormWithItsSpatialHalf() throws IOException {
        Map<String, RowSource> tensors = new LinkedHashMap<>();
        tensors.put("language_model.model.layers.0.self_attn.q_norm.weight", iota(4));
        tensors.put("language_model.model.layers.0.self_attn.q_norm_hw.weight", iota(4));
        tensors.put("language_model.model.layers.0.self_attn.k_norm.weight", iota(4));
        tensors.put("language_model.model.layers.0.self_attn.k_norm_hw.weight", iota(4));
        RecordingSink out = planAll(contextW(Arch.SENSENOVA_U1, "NEOChatModel", tensors),
                HfDenseHooksB::sensenovaTensors);
        assertEquals(2, out.tensors.size());
        assertEquals(2, out.dropped.size());
        assertTrue(out.dropped.containsKey("language_model.model.layers.0.self_attn.q_norm_hw.weight"));
        assertTrue(out.dropped.containsKey("language_model.model.layers.0.self_attn.k_norm_hw.weight"));
        for (RowSource s : out.tensors.values()) {
            assertArrayEquals(new float[] {0, 1, 2, 3, 0, 1, 2, 3}, values(s), 0f);
        }
    }

    /** A context over the tensors of an in-memory index, with real files on disk for the tokenizer layer. */
    private HfLoadContext contextW(Arch arch, String hfArch, Map<String, RowSource> tensors) throws IOException {
        HfArchEntry entry = HfArchEntry.of(List.of(hfArch), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        Path dir = tmp.newFolder().toPath();
        write(dir, "config.json", json("{'architectures': ['" + hfArch + "'], " + QWEN2 + "}"));
        return HfLoadContext.builder(HfConfigLoader.fromJson(dir,
                        json("{'architectures': ['" + hfArch + "'], " + QWEN2 + "}")), entry, arch)
                .options(ModelSourceOptions.builder().build())
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .index(com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex.of(tensors))
                .build();
    }

    @Test
    public void sensenovaMetaWritesTheSpatialRopeBase() throws IOException {
        HfLoadContext ctx = context(Arch.SENSENOVA_U1, "NEOChatModel", QWEN2 + ", 'rope_theta_hw': 100.0");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.SENSENOVA_U1.ggufName());
        HfDenseHooksB.sensenovaMeta(ctx, kv, null);
        assertEquals(100.0f, num(kv.build().get(String.format(GgufKeys.ROPE_FREQ_BASE_SPATIAL,
                Arch.SENSENOVA_U1.ggufName()))), 0f);
    }

    /** The float value of a boxed metadata number. */
    private static float num(Object v) {
        return ((Number) v).floatValue();
    }

    @Test
    public void sensenovaRefusesTheMoEAndScaledRopeCheckpoints() throws IOException {
        assertThrows(GgufFormatException.class, () -> {
            HfLoadContext ctx = context(Arch.SENSENOVA_U1, "NEOChatModel", QWEN2 + ", 'num_experts': 8");
            HfDenseHooksB.sensenovaMeta(ctx, new HfMetadataBuilder(Arch.SENSENOVA_U1.ggufName()), null);
        });
        assertThrows(GgufFormatException.class, () -> {
            HfLoadContext ctx = context(Arch.SENSENOVA_U1, "NEOChatModel", QWEN2 + ", 'rope_scaling':"
                    + " {'rope_type': 'yarn'}");
            HfDenseHooksB.sensenovaMeta(ctx, new HfMetadataBuilder(Arch.SENSENOVA_U1.ggufName()), null);
        });
        assertThrows(GgufFormatException.class, () -> {
            HfLoadContext ctx = context(Arch.SENSENOVA_U1, "NEOChatModel", QWEN2 + ", 'attention_bias': true");
            HfDenseHooksB.sensenovaMeta(ctx, new HfMetadataBuilder(Arch.SENSENOVA_U1.ggufName()), null);
        });
    }
}
