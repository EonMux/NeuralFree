package com.eonmux.neuralfree.gguf.hf.hooks;

import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfNameFilter;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.spec.TensorNames;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfChatTemplates;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.UnaryOperator;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfQwenHooks} against LC/conversion/qwen.py ({@code Qwen2Model}, {@code Qwen3Model}, {@code _QwenMtpMixin})
 * and {@code _try_set_pooling_type} (LC/conversion/base.py:2510-2562): the headless Qwen2 prefix, the Qwen3 reranker
 * head and metadata, the voyage-4-nano projection, the MTP renumbering filter, block count and metadata, and the
 * pooling side files. Also holds the fixtures the other Qwen hook tests share.
 */
public class HfQwenHooksTest {

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    /** The configuration keys the generic metadata of every Qwen class needs, as single-quoted JSON. */
    static final String BASE_CONFIG = "'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
            + " 'num_key_value_heads': 1, 'num_hidden_layers': 2, 'max_position_embeddings': 64,"
            + " 'rms_norm_eps': 1e-06, 'vocab_size': 4";

    private static final UnaryOperator<String> IDENTITY = UnaryOperator.identity();

    // ------------------------------------------------------------------ shared fixtures

    /** A tensor sink that records what hooks plan. */
    static final class RecordingSink implements TensorSink {
        final Map<String, RowSource> tensors = new LinkedHashMap<>();
        final Map<String, Object> kv = new LinkedHashMap<>();
        final Map<String, GGUFValueType> kvTypes = new LinkedHashMap<>();
        final Map<String, String> dropped = new LinkedHashMap<>();
        final Set<String> consumed = new LinkedHashSet<>();

        @Override
        public void add(String ggufName, RowSource src) {
            if (tensors.putIfAbsent(ggufName, src) != null) {
                throw new IllegalArgumentException("tensor '" + ggufName + "' is already planned");
            }
        }

        @Override
        public void addKv(String key, GGUFValueType type, GGUFValueType elementType, Object value) {
            if (kv.putIfAbsent(key, value) != null) {
                throw new IllegalArgumentException("metadata key '" + key + "' is already recorded");
            }
            kvTypes.put(key, type);
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

    /** A load context of {@code hfArch} resolved to {@code arch}, over the model directory {@code dir}. */
    static HfLoadContext context(Path dir, Arch arch, String hfArch, String extraConfig,
                                 Map<String, RowSource> index, boolean includeMtp) throws IOException {
        String config = "{'architectures': ['" + hfArch + "'], " + BASE_CONFIG
                + (extraConfig.isEmpty() ? "" : ", " + extraConfig) + "}";
        HfArchEntry entry = HfArchEntry.of(List.of(hfArch), HfArchResolver.of(arch), HfMetadataHook.GENERIC,
                HfTensorHook.DEFAULT, (ctx, pre) -> {
                    throw new UnsupportedOperationException("vocabulary is not part of this test");
                });
        return HfLoadContext.builder(HfConfigLoader.fromJson(dir, config.replace('\'', '"')), entry, arch)
                .options(ModelSourceOptions.builder().includeMtp(includeMtp).build())
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .index(HfTensorIndex.of(index))
                .build();
    }

    static HfLoadContext context(Path dir, Arch arch, String hfArch, String extraConfig) throws IOException {
        return context(dir, arch, hfArch, extraConfig, Collections.emptyMap(), false);
    }

    /** Writes {@code text} to {@code dir/name}, creating parent directories. */
    static void write(Path dir, String name, String text) throws IOException {
        Path p = dir.resolve(name);
        Files.createDirectories(p.getParent());
        Files.write(p, text.getBytes(StandardCharsets.UTF_8));
    }

    /** Writes single-quoted JSON to {@code dir/name}. */
    static void writeJson(Path dir, String name, String singleQuotedJson) throws IOException {
        write(dir, name, singleQuotedJson.replace('\'', '"'));
    }

    /** An F32 tensor holding {@code 0, 1, 2, ...}. */
    static RowSource seq(long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        return RopeTensors.fromFloats(shape, v);
    }

    /** The values of an F32 tensor, row-major. */
    static float[] values(RowSource s) throws IOException {
        long n = 1;
        for (long d : s.shape()) {
            n *= d;
        }
        int rows = (int) s.rows();
        ByteBuffer b = ByteBuffer.allocate((int) (rows * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, rows, b);
        b.flip();
        assertEquals("an F32 tensor", n * Float.BYTES, b.remaining());
        float[] out = new float[(int) n];
        b.asFloatBuffer().get(out);
        return out;
    }

    static float[] floats(float... v) {
        return v;
    }

    /** {@code template} for the architecture {@code arch}. */
    static String key(String template, Arch arch) {
        return String.format(Locale.ROOT, template, arch.ggufName());
    }

    static int intValue(HfMetadata m, String key) {
        Object v = m.get(key);
        assertTrue(key + " is set", v instanceof Number);
        return ((Number) v).intValue();
    }

    private Path dir() throws IOException {
        return tmp.newFolder().toPath();
    }

    // ------------------------------------------------------------------ Qwen2Model

    @Test
    public void headlessQwen2GetsTheModelPrefix() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN2, HfQwenHooks.QWEN2_HEADLESS, "");
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen2Tensors(ctx, "layers.0.self_attn.q_proj.weight", seq(2, 8), sink);
        HfQwenHooks.qwen2Tensors(ctx, "embed_tokens.weight", seq(4, 8), sink);
        assertEquals(Arrays.asList("blk.0.attn_q.weight", "token_embd.weight"), List.copyOf(sink.tensors.keySet()));
    }

    @Test
    public void causalQwen2KeepsItsNames() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN2, "Qwen2ForCausalLM", "");
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen2Tensors(ctx, "model.layers.1.mlp.down_proj.weight", seq(8, 16), sink);
        assertEquals(Collections.singleton("blk.1.ffn_down.weight"), sink.tensors.keySet());
    }

    @Test
    public void embeddingRowsComeFromTheTensorThatBecomesTheTokenEmbedding() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.self_attn.q_proj.weight", seq(2, 8));
        index.put("model.embed_tokens.weight", seq(5, 8));
        HfLoadContext ctx = context(dir(), Arch.QWEN2, "Qwen2ForCausalLM", "", index, false);
        assertEquals(5, HfHookSupport.embeddingRows(ctx).getAsInt());
        assertFalse(HfHookSupport.embeddingRows(context(dir(), Arch.QWEN2, "Qwen2ForCausalLM", "")).isPresent());
    }

    @Test
    public void originArchitectureIsTheRootArchitecture() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3, HfQwenHooks.INTERN_S1, "");
        assertEquals(HfQwenHooks.INTERN_S1, HfQwenHooks.originArchitecture(ctx.config()));
    }

    // ------------------------------------------------------------------ pooling

    @Test
    public void sentenceTransformersPoolingModuleSetsThePoolingType() throws IOException {
        Path dir = dir();
        writeJson(dir, "modules.json", "[{'idx': 0, 'path': '', 'type': 'sentence_transformers.models.Transformer'},"
                + " {'idx': 1, 'path': '1_Pooling', 'type': 'sentence_transformers.models.Pooling'}]");
        writeJson(dir, "1_Pooling/config.json", "{'pooling_mode_cls_token': false, 'pooling_mode_mean_tokens': 0,"
                + " 'pooling_mode_lasttoken': true}");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", "");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.qwen2Meta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(PoolingKind.LAST.ggufCode(), intValue(m, key(GgufKeys.POOLING_TYPE, Arch.QWEN3)));
        assertTrue(m.has(key(GgufKeys.BLOCK_COUNT, Arch.QWEN3)));
    }

    @Test
    public void poolingModeNamesTheModeWhenNoFlagIsSet() throws IOException {
        Path dir = dir();
        writeJson(dir, "modules.json", "[{'path': 'pool', 'type': 'Pooling'}]");
        writeJson(dir, "pool/config.json", "{'pooling_mode': 'mean'}");
        assertEquals(PoolingKind.MEAN.ggufCode(), pooling(dir, ""));
    }

    @Test
    public void poolingModuleAtTheModelRootReadsTheRootConfiguration() throws IOException {
        Path dir = dir();
        writeJson(dir, "modules.json", "[{'path': '', 'type': 'Pooling'}]");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", "'pooling_mode_cls_token': true");
        writeJson(dir, "config.json", "{'pooling_mode_cls_token': true}");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.pooling(ctx, kv);
        assertEquals(PoolingKind.CLS.ggufCode(), intValue(kv.build(), key(GgufKeys.POOLING_TYPE, Arch.QWEN3)));
    }

    @Test
    public void poolingModuleWithoutAKnownModeIsRejected() throws IOException {
        Path dir = dir();
        writeJson(dir, "modules.json", "[{'path': 'pool', 'type': 'Pooling'}]");
        writeJson(dir, "pool/config.json", "{'pooling_mode_max_tokens': true}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> pooling(dir, ""));
        assertTrue(e.getMessage(), e.getMessage().startsWith("Only MEAN, CLS, and LAST pooling types supported"));
    }

    @Test
    public void embeddingConfigLastTokenPoolingSetsLast() throws IOException {
        Path dir = dir();
        writeJson(dir, "embedding_config.json", "{'pooling': 'last_token'}");
        assertEquals(PoolingKind.LAST.ggufCode(), pooling(dir, ""));
    }

    @Test
    public void embeddingConfigWithAnotherPoolingIsRejected() throws IOException {
        Path dir = dir();
        writeJson(dir, "embedding_config.json", "{'pooling': 'mean'}");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> pooling(dir, ""));
        assertTrue(e.getMessage(), e.getMessage().endsWith("unsupported embedding_config.json pooling 'mean'"));
    }

    @Test
    public void classifierPoolingAndActivationAreWritten() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3, "Qwen3ForCausalLM",
                "'classifier_pooling': 'mean', 'classifier_activation': 'gelu'");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.pooling(ctx, kv);
        HfMetadata m = kv.build();
        assertFalse(m.has(key(GgufKeys.POOLING_TYPE, Arch.QWEN3)));
        assertEquals(PoolingKind.MEAN.ggufCode(), intValue(m, key(GgufKeys.CLASSIFIER_POOLING_TYPE, Arch.QWEN3)));
        assertEquals("gelu", m.get(key(GgufKeys.CLASSIFIER_ACTIVATION, Arch.QWEN3)));
    }

    @Test
    public void unsupportedClassifierPoolingIsRejected() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3, "Qwen3ForCausalLM", "'classifier_pooling': 'max'");
        assertThrows(GgufFormatException.class,
                () -> HfQwenHooks.pooling(ctx, new HfMetadataBuilder(Arch.QWEN3.ggufName())));
    }

    private static int pooling(Path dir, String extraConfig) throws IOException {
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", extraConfig);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.pooling(ctx, kv);
        return intValue(kv.build(), key(GgufKeys.POOLING_TYPE, Arch.QWEN3));
    }

    // ------------------------------------------------------------------ Qwen3 reranker

    /** A reranker directory: a README heading and a vocabulary where {@code no} = 1 and {@code yes} = 2. */
    private Path rerankerDir() throws IOException {
        Path dir = dir();
        write(dir, "README.md", "---\nlicense: apache-2.0\n---\n# Qwen3-Reranker-0.6B\n");
        writeJson(dir, "vocab.json", "{'a': 0, 'no': 1, 'yes': 2, 'b': 3}");
        return dir;
    }

    @Test
    public void rerankerKeepsTheYesAndNoRowsOfAnUntiedHead() throws IOException {
        HfLoadContext ctx = context(rerankerDir(), Arch.QWEN3, "Qwen3ForCausalLM", "'tie_word_embeddings': false");
        assertTrue(HfQwenHooks.isQwen3Reranker(ctx));
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen3Tensors(ctx, "lm_head.weight", seq(4, 3), sink);
        String cls = TensorNames.weight(TensorNames.CLS_OUT);
        assertEquals(Collections.singleton(cls), sink.tensors.keySet());
        assertArrayEquals(new long[]{2, 3}, sink.tensors.get(cls).shape());
        assertArrayEquals(floats(6, 7, 8, 3, 4, 5), values(sink.tensors.get(cls)), 0f);
        assertTrue(sink.dropped.containsKey("lm_head.weight"));

        HfQwenHooks.qwen3Tensors(ctx, "model.embed_tokens.weight", seq(4, 3), sink);
        assertEquals(Arrays.asList(cls, "token_embd.weight"), List.copyOf(sink.tensors.keySet()));
    }

    @Test
    public void rerankerOfATiedEmbeddingAlsoPlansTheEmbedding() throws IOException {
        Path dir = tmp.newFolder("Qwen3-Reranker-4B").toPath();
        writeJson(dir, "vocab.json", "{'yes': 0, 'x': 1, 'no': 3, 'y': 2}");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", "'tie_word_embeddings': true");
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen3Tensors(ctx, "model.embed_tokens.weight", seq(4, 2), sink);
        String cls = TensorNames.weight(TensorNames.CLS_OUT);
        assertEquals(Arrays.asList(cls, "token_embd.weight"), List.copyOf(sink.tensors.keySet()));
        assertArrayEquals(floats(0, 1, 6, 7), values(sink.tensors.get(cls)), 0f);
        assertArrayEquals(floats(0, 1, 2, 3, 4, 5, 6, 7), values(sink.tensors.get("token_embd.weight")), 0f);
        assertTrue(sink.dropped.isEmpty());
    }

    @Test
    public void rerankerMetadataWritesRankPoolingLabelsAndTheRerankTemplate() throws IOException {
        Path dir = dir();
        writeJson(dir, "vocab.json", "{'no': 0, 'yes': 1}");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForSequenceClassification", "");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.qwen3Meta(ctx, kv, null);
        HfMetadata m = kv.build();
        assertEquals(PoolingKind.RANK.ggufCode(), intValue(m, key(GgufKeys.POOLING_TYPE, Arch.QWEN3)));
        assertEquals(4, PoolingKind.RANK.ggufCode());
        assertEquals(Arrays.asList("yes", "no"), m.get(key(GgufKeys.CLASSIFIER_OUTPUT_LABELS, Arch.QWEN3)));
        Map<String, String> template = new LinkedHashMap<>();
        template.put("name", "rerank");
        template.put("template", HfQwenHooks.RERANK_TEMPLATE);
        Map<String, Object> expected = HfChatTemplates.toGguf(List.of(template));
        assertFalse(expected.isEmpty());
        for (Map.Entry<String, Object> e : expected.entrySet()) {
            assertEquals(e.getKey(), e.getValue(), m.get(e.getKey()));
        }
        assertTrue(expected.containsValue(HfQwenHooks.RERANK_TEMPLATE));
        assertTrue(HfQwenHooks.RERANK_TEMPLATE.startsWith("<|im_start|>system\nJudge whether the Document meets"));
        assertTrue(HfQwenHooks.RERANK_TEMPLATE.endsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    @Test
    public void rerankerHintsCoverReadmeNamesAndClassifierArchitectures() throws IOException {
        Path vl = dir();
        write(vl, "README.md", "# QWEN3-VL-Reranker-2B\n");
        assertTrue(HfQwenHooks.rerankerHints(context(vl, Arch.QWEN3, "Qwen3ForCausalLM", "").config()));
        assertTrue(HfQwenHooks.rerankerHints(context(dir(), Arch.QWEN3, "Qwen3ForCausalLM",
                "'_name_or_path': 'Qwen/Qwen3-Reranker-8B'").config()));
        assertTrue(HfQwenHooks.rerankerHints(
                context(dir(), Arch.QWEN3, "Qwen3ForSequenceClassification", "").config()));
        Path plain = dir();
        write(plain, "README.md", "# Qwen3-8B\nNot a reranker.\n");
        assertFalse(HfQwenHooks.rerankerHints(context(plain, Arch.QWEN3, "Qwen3ForCausalLM", "").config()));
    }

    @Test
    public void plainQwen3MapsItsHeadAsTheOutput() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3, "Qwen3ForCausalLM", "");
        assertFalse(HfQwenHooks.isQwen3Reranker(ctx));
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen3Tensors(ctx, "lm_head.weight", seq(4, 8), sink);
        assertEquals(Collections.singleton("output.weight"), sink.tensors.keySet());
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3.ggufName());
        HfQwenHooks.qwen3Meta(ctx, kv, null);
        assertFalse(kv.build().has(key(GgufKeys.POOLING_TYPE, Arch.QWEN3)));
    }

    @Test
    public void rerankerTokenOutsideTheHeadIsRejected() throws IOException {
        Path dir = dir();
        write(dir, "README.md", "# Qwen3-Reranker\n");
        writeJson(dir, "vocab.json", "{'no': 1, 'yes': 9}");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", "");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfQwenHooks.qwen3Tensors(ctx, "lm_head.weight", seq(4, 3), new RecordingSink()));
        assertTrue(e.getMessage(), e.getMessage().contains("token id 9"));
    }

    @Test
    public void rerankerWithoutTheLabelTokensIsRejected() throws IOException {
        Path dir = dir();
        write(dir, "README.md", "# Qwen3-Reranker\n");
        writeJson(dir, "vocab.json", "{'yes': 1}");
        HfLoadContext ctx = context(dir, Arch.QWEN3, "Qwen3ForCausalLM", "");
        GgufFormatException e = assertThrows(GgufFormatException.class, () -> HfQwenHooks.isQwen3Reranker(ctx));
        assertTrue(e.getMessage(), e.getMessage().contains("no token 'no'"));
    }

    @Test
    public void tokenIdPrefersAddedTokensOverTheVocabulary() throws IOException {
        Path dir = dir();
        writeJson(dir, "vocab.json", "{'yes': 2, 'no': 3}");
        writeJson(dir, "added_tokens.json", "{'yes': 7}");
        HfTokenizerFiles files = HfTokenizerFiles.locate(dir);
        assertEquals(7, HfQwenHooks.tokenId(files, "yes"));
        assertEquals(3, HfQwenHooks.tokenId(files, "no"));
    }

    // ------------------------------------------------------------------ voyage-4-nano

    @Test
    public void voyageProjectionRecordsTheOutputLengthAndBecomesDense2() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3, "Qwen3ForCausalLM", "");
        RecordingSink sink = new RecordingSink();
        HfQwenHooks.qwen3Tensors(ctx, HfQwenHooks.VOYAGE_LINEAR, seq(3, 8), sink);
        String lengthOut = key(GgufKeys.EMBEDDING_LENGTH_OUT, Arch.QWEN3);
        assertEquals(3L, sink.kv.get(lengthOut));
        assertEquals(GGUFValueType.UINT32, sink.kvTypes.get(lengthOut));
        assertEquals(Collections.singleton(TensorNames.weight(TensorNames.DENSE_2_OUT)), sink.tensors.keySet());
    }

    // ------------------------------------------------------------------ _QwenMtpMixin

    @Test
    public void mtpFilterRenumbersMtpLayersAfterTheBaseLayers() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", Collections.emptyMap(), true);
        assertTrue(HfQwenHooks.includeMtp(ctx));
        HfNameFilter f = HfQwenHooks.mtpFilter(ctx);
        assertEquals("model.layers.2.mlp.gate.weight", f.filter("mtp.layers.0.mlp.gate.weight", IDENTITY));
        assertEquals("model.layers.3.self_attn.q_proj.weight",
                f.filter("model.mtp.layers.1.self_attn.q_proj.weight", IDENTITY));
        assertEquals("model.layers.2.eh_proj.weight", f.filter("mtp.fc.weight", IDENTITY));
        assertEquals("model.layers.2.enorm.weight", f.filter("mtp.pre_fc_norm_embedding.weight", IDENTITY));
        assertEquals("model.layers.2.hnorm.weight", f.filter("model.mtp.pre_fc_norm_hidden.weight", IDENTITY));
        assertEquals("model.layers.2.shared_head.norm.weight", f.filter("mtp.norm.weight", IDENTITY));
        assertEquals("mtp.other.weight", f.filter("mtp.other.weight", IDENTITY));
        assertEquals("mtp.layers.x.weight", f.filter("mtp.layers.x.weight", IDENTITY));
        assertEquals("model.layers.0.input_layernorm.weight",
                f.filter("model.layers.0.input_layernorm.weight", IDENTITY));
    }

    @Test
    public void mtpFilterRunsTheInheritedFilterFirst() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", Collections.emptyMap(), true);
        HfNameFilter f = HfQwenHooks.mtpFilter(ctx);
        UnaryOperator<String> unwrap = n -> n.startsWith("wrapper.") ? n.substring("wrapper.".length()) : n;
        assertEquals("model.layers.2.eh_proj.weight", f.filter("wrapper.mtp.fc.weight", unwrap));
        assertNull(f.filter("mtp.fc.weight", n -> null));
    }

    @Test
    public void mtpFilterDropsMtpTensorsWhenTheyAreNotKept() throws IOException {
        HfLoadContext off = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", Collections.emptyMap(),
                false);
        assertFalse(HfQwenHooks.includeMtp(off));
        HfNameFilter f = HfQwenHooks.mtpFilter(off);
        assertNull(f.filter("mtp.fc.weight", IDENTITY));
        assertNull(f.filter("model.mtp.layers.0.mlp.gate.weight", IDENTITY));
        assertEquals("model.layers.1.mlp.gate.weight", f.filter("model.layers.1.mlp.gate.weight", IDENTITY));

        HfLoadContext text = context(dir(), Arch.QWEN35, HfQwenHooks.QWEN35_TEXT_MODEL, "",
                Collections.emptyMap(), true);
        assertFalse(HfQwenHooks.includeMtp(text));
        assertNull(HfQwenHooks.mtpFilter(text).filter("mtp.layers.0.mlp.gate.weight", IDENTITY));
    }

    @Test
    public void mtpBlockCountAddsTheConfiguredMtpLayers() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN35, "Qwen3_5ForConditionalGeneration",
                "'mtp_num_hidden_layers': 1", Collections.emptyMap(), true);
        assertEquals(3, HfQwenHooks.mtpBlockCount(ctx));
        HfLoadContext off = context(dir(), Arch.QWEN35, "Qwen3_5ForConditionalGeneration",
                "'mtp_num_hidden_layers': 1", Collections.emptyMap(), false);
        assertEquals(2, HfQwenHooks.mtpBlockCount(off));
    }

    @Test
    public void mtpBlockCountInfersTheMtpLayersFromTheIndex() throws IOException {
        Map<String, RowSource> index = new LinkedHashMap<>();
        index.put("model.layers.0.mlp.gate.weight", seq(1));
        index.put("model.mtp.layers.0.mlp.gate.weight", seq(1));
        index.put("mtp.layers.1.mlp.gate.weight", seq(1));
        index.put("mtp.fc.weight", seq(1));
        HfLoadContext ctx = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "'mtp_num_hidden_layers': 0",
                index, true);
        assertEquals(4, HfQwenHooks.mtpBlockCount(ctx));

        Map<String, RowSource> plain = Collections.singletonMap("model.layers.0.mlp.gate.weight", seq(1));
        assertEquals(2, HfQwenHooks.mtpBlockCount(
                context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", plain, true)));
    }

    @Test
    public void mtpMetaWritesTheNextnLayersOfTheBlockCount() throws IOException {
        HfLoadContext ctx = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", Collections.emptyMap(), true);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3NEXT.ggufName());
        HfQwenHooks.mtpMeta(ctx.withBlockCount(3), kv);
        assertEquals(1, intValue(kv.build(), key(GgufKeys.NEXTN_PREDICT_LAYERS, Arch.QWEN3NEXT)));

        HfMetadataBuilder none = new HfMetadataBuilder(Arch.QWEN3NEXT.ggufName());
        HfQwenHooks.mtpMeta(ctx.withBlockCount(2), none);
        HfLoadContext off = context(dir(), Arch.QWEN3NEXT, "Qwen3NextForCausalLM", "", Collections.emptyMap(),
                false);
        HfQwenHooks.mtpMeta(off.withBlockCount(3), none);
        assertFalse(none.build().has(key(GgufKeys.NEXTN_PREDICT_LAYERS, Arch.QWEN3NEXT)));
    }
}
