package com.eonmux.neuralfree.gguf.hf.torch;

import com.eonmux.neuralfree.gguf.GGUFModel;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.ClassifierScope;
import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.engine.EngineConfig;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchResolver;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfMetadataHook;
import com.eonmux.neuralfree.gguf.hf.arch.HfTensorHook;
import com.eonmux.neuralfree.gguf.hf.config.HfConfigLoader;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfDenseFixturesA;
import com.eonmux.neuralfree.gguf.hf.fixtures.HfSpecialFixturesT5;
import com.eonmux.neuralfree.gguf.hf.hooks.HfAsrForcedAlignerHooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfBgeM3Hooks;
import com.eonmux.neuralfree.gguf.hf.hooks.HfPoolingHooks;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadataBuilder;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorSink;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.memory.MemoryConfig;
import com.eonmux.neuralfree.gguf.ops.heads.EncodeGate;
import com.eonmux.neuralfree.gguf.spec.ModelSpec;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.spec.resolve.ModelSpecBuilder;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.tokenizer.hf.HfTokenizerFiles;
import com.eonmux.neuralfree.io.safetensors.StTensor;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static com.eonmux.neuralfree.gguf.hf.torch.RestrictedUnpicklerTest.Pb;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The BGE-M3 and forced-aligner converters against the vLLM sources: the {@code BgeM3EmbeddingModel} pooling,
 * normalization and BOS/EOS metadata and the {@code sparse_linear.pt} / {@code colbert_linear.pt} side tensors read
 * as the {@code cls.sparse} / {@code cls.colbert} projections (roberta.py:187-269); the forced aligner's
 * {@code thinker.lm_head} classifier, NONE pooling, softmax pooler activation and dropped audio branch
 * (qwen3_asr_forced_aligner.py:24-117, activations.py:19-60);
 * and the process-reward {@code classifier.step_tag_id} default (models/config.py:982-989).
 */
public class HfPoolingModelHeadsTest {

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ registry

    @Test
    public void theTwoConverterNamesResolveToRealEntries() throws IOException {
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(HfBgeM3Hooks.ARCH));
        assertTrue(HfArchRegistry.supportedNames(HfArchEntry.Kind.TEXT).contains(HfAsrForcedAlignerHooks.ARCH));
        assertDoesNotThrow(() -> HfArchRegistry.resolve(config("BgeM3EmbeddingModel"), HfArchEntry.Kind.TEXT));
        assertDoesNotThrow(() -> HfArchRegistry.resolve(config(HfAsrForcedAlignerHooks.ARCH),
                HfArchEntry.Kind.TEXT));
        // the remaining three names stay refused by name with their verified missing graph
        for (String name : new String[]{"GteNewForSequenceClassification", "OpenAIPrivacyFilterForTokenClassification",
                "LlamaNemotronVLForSequenceClassification", "ClefOmniForJointSchema"}) {
            GgufFormatException e = assertThrows(GgufFormatException.class,
                    () -> HfArchRegistry.resolve(config(name), HfArchEntry.Kind.TEXT));
            assertTrue(e.getMessage(), e.getMessage().contains(name));
        }
    }

    private static void assertDoesNotThrow(ThrowingRunnable r) throws IOException {
        r.run();
    }

    private interface ThrowingRunnable {
        void run() throws IOException;
    }

    // ------------------------------------------------------------------ BGE-M3 metadata

    @Test
    public void bgeM3WritesClsPoolingNormalizationAndBoundaries() throws IOException {
        Path dir = tmp.newFolder("bge-m3").toPath();
        HfLoadContext ctx = context(dir, Arch.BERT, HfBgeM3Hooks.ARCH);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.BERT.ggufName());
        HfBgeM3Hooks.meta(ctx, kv, null);
        assertEquals(PoolingKind.CLS.ggufCode(), kv.get(GgufKeys.POOLING_TYPE));
        assertEquals(Boolean.TRUE, kv.get(GgufKeys.POOLING_NORMALIZE));
        assertEquals(0L, ((Number) kv.get(GgufKeys.CLASSIFIER_BOS_TOKEN)).longValue());
        assertEquals(2L, ((Number) kv.get(GgufKeys.CLASSIFIER_EOS_TOKEN)).longValue());
    }

    // ------------------------------------------------------------------ BGE-M3 side tensors

    @Test
    public void theSideProjectionsBecomeTheTokenHeads() throws IOException {
        Path dir = tmp.newFolder("bge-m3-side").toPath();
        writeStateDict(dir.resolve("sparse_linear.pt"), shapes(weight(1), bias(1)));
        writeStateDict(dir.resolve("colbert_linear.pt"), shapes(weight(8), bias(8)));
        HfLoadContext ctx = context(dir, Arch.BERT, HfBgeM3Hooks.ARCH);
        RecordingSink sink = new RecordingSink();
        HfBgeM3Hooks.sideTensors(ctx, sink);
        assertEquals(Arrays.asList("cls.sparse.weight", "cls.sparse.bias", "cls.colbert.weight",
                "cls.colbert.bias"), new ArrayList<>(sink.tensors.keySet()));
        assertArrayEqualsLongs(new long[]{1, 8}, sink.tensors.get("cls.sparse.weight").shape());
        assertArrayEqualsLongs(new long[]{8, 8}, sink.tensors.get("cls.colbert.weight").shape());
    }

    @Test
    public void aMissingSideFileRefusesByName() throws IOException {
        Path dir = tmp.newFolder("bge-m3-missing").toPath();
        HfLoadContext ctx = context(dir, Arch.BERT, HfBgeM3Hooks.ARCH);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfBgeM3Hooks.sideTensors(ctx, new RecordingSink()));
        assertTrue(e.getMessage(), e.getMessage().contains("sparse_linear.pt"));
    }

    // ------------------------------------------------------------------ forced aligner

    @Test
    public void theForcedAlignerScoresEveryTokenFromTheThinkerHead() throws IOException {
        HfLoadContext ctx = context(tmp.newFolder("aligner").toPath(), Arch.QWEN3VL,
                HfAsrForcedAlignerHooks.ARCH);
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN3VL.ggufName());
        HfAsrForcedAlignerHooks.meta(ctx, kv, null);
        assertEquals(PoolingKind.NONE.ggufCode(), kv.get(GgufKeys.POOLING_TYPE));
        assertEquals(ClassifierScope.TOKEN.ggufName(), kv.get(GgufKeys.CLASSIFIER_SCOPE));
        assertEquals("softmax", kv.get(GgufKeys.CLASSIFIER_POOLER_ACTIVATION));

        RecordingSink sink = new RecordingSink();
        HfAsrForcedAlignerHooks.tensors(ctx, "lm_head.weight", seq(3, 8), sink);
        assertEquals(Collections.singleton("cls.output.weight"), sink.tensors.keySet());
        HfAsrForcedAlignerHooks.tensors(ctx, "audio_tower.conv1.weight", seq(4), sink);
        assertTrue("the audio tower belongs to the projector file",
                sink.dropped.containsKey("audio_tower.conv1.weight"));
    }

    // ------------------------------------------------------------------ process-reward default

    @Test
    public void theProcessRewardHeadWritesVllmsDefaultStepTag() throws IOException {        HfLoadContext ctx = context(tmp.newFolder("prm").toPath(), Arch.QWEN2, "Qwen2ForProcessRewardModel");
        HfMetadataBuilder kv = new HfMetadataBuilder(Arch.QWEN2.ggufName());
        HfPoolingHooks.qwen2ProcessMeta(ctx, kv, null);
        assertEquals(151651L, ((Number) kv.get(GgufKeys.CLASSIFIER_STEP_TAG_ID)).longValue());
        assertNotNull(kv.get(GgufKeys.POOLING_TYPE));
    }

    // ------------------------------------------------------------------ end to end

    @Test
    public void theBgeM3CheckpointConvertsAndAnswersAllThreeOutputs() throws IOException {
        Path dir = tmp.newFolder("bge-m3-e2e").toPath();
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json(
                "{'architectures': ['BgeM3EmbeddingModel'], 'model_type': 'xlm-roberta', 'hidden_size': 8,"
                        + " 'intermediate_size': 16, 'num_attention_heads': 2, 'num_key_value_heads': 2,"
                        + " 'num_hidden_layers': 2, 'max_position_embeddings': 64, 'layer_norm_eps': 1e-12,"
                        + " 'vocab_size': 8, 'type_vocab_size': 1, 'bos_token_id': 0, 'eos_token_id': 2}"))
                .binaryFile("sentencepiece.bpe.model", HfSpecialFixturesT5.unigramModel())
                .file("tokenizer_config.json", "{\"bos_token\": \"<s>\", \"eos_token\": \"</s>\","
                        + " \"cls_token\": \"<s>\", \"sep_token\": \"</s>\", \"unk_token\": \"<unk>\"}");
        String p = "roberta.";
        b.bf16(p + "embeddings.word_embeddings.weight", 8, 8);
        b.bf16(p + "embeddings.position_embeddings.weight", 64, 8);
        b.bf16(p + "embeddings.token_type_embeddings.weight", 1, 8);
        b.bf16(p + "embeddings.LayerNorm.weight", 8);
        b.bf16(p + "embeddings.LayerNorm.bias", 8);
        for (int l = 0; l < 2; l++) {
            String q = p + "encoder.layer." + l + ".";
            for (String proj : new String[]{"query", "key", "value"}) {
                b.bf16(q + "attention.self." + proj + ".weight", 8, 8);
                b.bf16(q + "attention.self." + proj + ".bias", 8);
            }
            b.bf16(q + "attention.output.dense.weight", 8, 8);
            b.bf16(q + "attention.output.dense.bias", 8);
            b.bf16(q + "attention.output.LayerNorm.weight", 8);
            b.bf16(q + "attention.output.LayerNorm.bias", 8);
            b.bf16(q + "intermediate.dense.weight", 16, 8);
            b.bf16(q + "intermediate.dense.bias", 16);
            b.bf16(q + "output.dense.weight", 8, 16);
            b.bf16(q + "output.dense.bias", 8);
            b.bf16(q + "output.LayerNorm.weight", 8);
            b.bf16(q + "output.LayerNorm.bias", 8);
        }
        b.write(dir);
        writeStateDict(dir.resolve("sparse_linear.pt"), shapes(weight(1), bias(1)));
        writeStateDict(dir.resolve("colbert_linear.pt"), shapes(weight(8), bias(8)));

        Path cache = tmp.newFolder("bge-m3-e2e-cache").toPath();
        try (ModelSource src = ModelSources.open(dir.toString(),
                ModelSourceOptions.builder().cacheDir(cache).build(), null)) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("bert", kv.archName());
            assertEquals(PoolingKind.CLS.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals(Boolean.TRUE, Boolean.valueOf(kv.bool(kv.key(GgufKeys.POOLING_NORMALIZE), false)));
            assertEquals(0, kv.u32(kv.key(GgufKeys.CLASSIFIER_BOS_TOKEN), -1));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLASSIFIER_EOS_TOKEN), -1));
            TensorCatalog tc = TensorCatalog.of(src);
            for (String name : new String[]{"cls.colbert.weight", "cls.colbert.bias", "cls.sparse.weight",
                    "cls.sparse.bias"}) {
                assertNotNull(name, tc.get(name));
            }
            ModelSpec spec = ModelSpecBuilder.build(kv, tc, MemoryConfig.defaults());
            assertTrue(EncodeGate.embedsPerToken(spec));
            assertTrue(EncodeGate.canTokenClassify(spec));
            assertTrue(EncodeGate.hasTokenEmbeddingProjection(spec));
            assertTrue(EncodeGate.hasTokenScoreProjection(spec));
        }
        try (GGUFModel model = new GGUFModel(dir.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build())) {
            int[] tokens = {0, 5, 6, 2};
            assertEquals("the dense CLS embedding", 8, model.encode(tokens).length);
            float[][] rows = model.encodeTokens(tokens);
            assertEquals("the leading BOS row drops; the token-embed filter leaves the trailing EOS row", 3,
                    rows.length);
            assertEquals("the colbert projection's width", 8, rows[0].length);
            float[][] scores = model.tokenClassify(tokens);
            assertEquals(2, scores.length);
            assertEquals("the sparse_linear's single score per token", 1, scores[0].length);
        }
    }

    @Test
    public void theForcedAlignerCheckpointConvertsAndSoftmaxesEveryToken() throws IOException {
        Path dir = HfDenseFixturesA.qwen3AsrForcedAligner().write(tmp.newFolder("aligner-e2e").toPath());
        Path cache = tmp.newFolder("aligner-e2e-cache").toPath();
        try (ModelSource src = ModelSources.open(dir.toString(),
                ModelSourceOptions.builder().cacheDir(cache).build(), null)) {
            GgufKv kv = GgufKv.of(src);
            assertEquals("qwen3vl", kv.archName());
            assertEquals(PoolingKind.NONE.ggufCode(), kv.u32Long(kv.key(GgufKeys.POOLING_TYPE), -1));
            assertEquals(ClassifierScope.TOKEN.ggufName(), kv.str(kv.key(GgufKeys.CLASSIFIER_SCOPE), null));
            assertEquals("softmax", kv.str(kv.key(GgufKeys.CLASSIFIER_POOLER_ACTIVATION), null));
            TensorCatalog tc = TensorCatalog.of(src);
            assertNotNull("the thinker's lm_head becomes the per-token classifier", tc.get("cls.output.weight"));
            for (String dropped : new String[]{"audio_tower.conv1.weight", "talker.model.embed_tokens.weight",
                    "code2wav.conv.weight"}) {
                assertEquals(dropped, null, tc.get(dropped));
            }
        }
        try (GGUFModel model = new GGUFModel(dir.toString(),
                EngineConfig.builder().backend(EngineConfig.BackendMode.CPU).build())) {
            float[][] rows = model.tokenClassify(new int[]{2, 3, 4});
            assertEquals("one row per prompt token", 3, rows.length);
            assertEquals("one score per time bin", 4, rows[0].length);
            for (float[] row : rows) {
                double sum = 0;
                for (float v : row) {
                    sum += v;
                }
                assertEquals("vLLM's default token-classify softmax", 1.0, sum, 1e-5);
            }
        }
    }

    // ------------------------------------------------------------------ helpers

    private static String weight(int out) {
        return "weight:" + out;
    }

    private static String bias(int size) {
        return "bias:" + size;
    }

    /** A {@code weight} of {@code [out, 8]} (as {@code weight:out}) and a {@code bias} of {@code [size]}. */
    private static LinkedHashMap<String, long[]> shapes(String... specs) {
        LinkedHashMap<String, long[]> out = new LinkedHashMap<>();
        for (String spec : specs) {
            String[] parts = spec.split(":");
            out.put(parts[0], "weight".equals(parts[0])
                    ? new long[]{Long.parseLong(parts[1]), 8} : new long[]{Long.parseLong(parts[1])});
        }
        return out;
    }

    private static void writeStateDict(Path file, LinkedHashMap<String, long[]> tensors) throws IOException {
        Pb p = new Pb().proto(2).emptyDict();
        TorchCheckpointTest.Bin bin = new TorchCheckpointTest.Bin();
        int i = 0;
        for (Map.Entry<String, long[]> e : tensors.entrySet()) {
            String key = String.valueOf(1000 + i);
            p.str(e.getKey());
            p.tensorOf("FloatStorage", key, 0L, e.getValue(), strides(e.getValue()));
            p.raw(0x73);
            long n = 1;
            for (long d : e.getValue()) {
                n *= d;
            }
            float[] values = new float[(int) n];
            for (int v = 0; v < values.length; v++) {
                values[v] = 0.1f * (v + 1);
            }
            bin.entry("archive/data/" + key, SyntheticHf.f32(values));
            i++;
        }
        bin.entry("archive/data.pkl", p.stop().build());
        Files.write(file, bin.build());
    }

    private static long[] strides(long[] shape) {
        long[] s = new long[shape.length];
        long next = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = next;
            next *= shape[i];
        }
        return s;
    }

    private HfLoadContext context(Path dir, Arch arch, String hfArch) throws IOException {
        String json = "{\"architectures\": [\"" + hfArch + "\"], \"hidden_size\": 8, \"intermediate_size\": 16,"
                + " \"num_attention_heads\": 2, \"num_key_value_heads\": 1, \"num_hidden_layers\": 2,"
                + " \"max_position_embeddings\": 64, \"layer_norm_eps\": 1e-12, \"vocab_size\": 8,"
                + " \"pad_token_id\": 0, \"bos_token_id\": 0, \"eos_token_id\": 2}";
        HfArchEntry entry = HfArchEntry.of(Collections.singletonList(hfArch), HfArchResolver.of(arch),
                HfMetadataHook.GENERIC, HfTensorHook.DEFAULT, (c, pre) -> {
                    throw new UnsupportedOperationException("the vocabulary is not part of this test");
                });
        return HfLoadContext.builder(HfConfigLoader.fromJson(dir, json), entry, arch)
                .tokenizerFiles(HfTokenizerFiles.locate(dir))
                .index(HfTensorIndex.of(Collections.emptyMap()))
                .build();
    }

    private static com.eonmux.neuralfree.gguf.hf.config.HfConfig config(String architecture) throws IOException {
        return HfConfigLoader.fromJson(Path.of("."), "{\"architectures\": [\"" + architecture + "\"],"
                + " \"model_type\": \"bert\"}");
    }

    private static RowSource seq(long... shape) {
        long n = 1;
        for (long d : shape) {
            n *= d;
        }
        float[] v = new float[(int) n];
        for (int i = 0; i < v.length; i++) {
            v[i] = i;
        }
        return com.eonmux.neuralfree.gguf.hf.tensor.RopeTensors.fromFloats(shape, v);
    }

    private static void assertArrayEqualsLongs(long[] want, long[] got) {
        assertEquals(Arrays.toString(want), Arrays.toString(got));
    }

    /** A sink recording the planned tensors and drops. */
    private static final class RecordingSink implements TensorSink {
        final Map<String, RowSource> tensors = new LinkedHashMap<>();
        final Map<String, String> dropped = new LinkedHashMap<>();

        @Override
        public void add(String ggufName, RowSource src) {
            tensors.put(ggufName, src);
        }

        @Override
        public void addKv(String key, com.eonmux.neuralfree.gguf.GGUFValueType type,
                          com.eonmux.neuralfree.gguf.GGUFValueType elementType, Object value) {
            throw new AssertionError("unexpected metadata " + key);
        }

        @Override
        public void drop(String hfName, String reason) {
            dropped.put(hfName, reason);
        }

        @Override
        public boolean isConsumed(String hfName) {
            return false;
        }

        @Override
        public void consume(String hfName) {
        }
    }
}
