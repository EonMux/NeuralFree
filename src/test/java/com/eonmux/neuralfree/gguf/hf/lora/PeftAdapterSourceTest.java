package com.eonmux.neuralfree.gguf.hf.lora;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFFileParser;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGUFValueType;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.SourceOrigin;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * A PEFT LoRA adapter directory of a synthetic Llama base, opened as the GGUF adapter view
 * ({@link PeftAdapterSource}, HF.md §LA-1): the PEFT branch of {@link ModelSources#openAdapter}; the metadata
 * ({@code general.type = adapter}, {@code adapter.type = lora}, the rslora-adjusted {@code adapter.lora.alpha} and the
 * aLoRA invocation tokens of both configuration forms); the factor plan ({@code blk.<n>.attn_q.weight.lora_a} with ne
 * {@code {in, r}} and {@code .lora_b} with ne {@code {r, out}}, the transposed {@code token_embd} A, the skipped
 * {@code .base_layer.weight}); the base-configuration resolution for a GGUF base (the named error without a config,
 * {@code base_model_name_or_path}, {@code loraBaseConfig}); and the trace: the q/k B factors come out permuted like the
 * base weights and the whole view is accepted by {@link LoraAdapterLoader#load}.
 */
public class PeftAdapterSourceTest {

    private static final int R = 4;
    private static final float ALPHA = 8.0f;
    /** The module prefix of the PEFT factor names (LC/convert_lora_to_gguf.py:272-279). */
    private static final String PREFIX = "base_model.model.";

    /** 2 layers, n_embd 64, 4 heads / 2 KV heads of dim 16 (kv 32), FFN 128, vocab 320. */
    private static final SyntheticModels.Config C = SyntheticModels.Config.llama();
    private static final int N_EMBD = C.nEmbd();
    private static final int KV_DIM = C.nHeadKv() * C.headDim();
    private static final int VOCAB = C.nVocab();

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options;
    private int index;

    @Before
    public void install() throws IOException {
        options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    // ------------------------------------------------------------------ fixtures

    /** A synthetic Llama checkpoint whose q/k weights carry one row tag per stored row (i + j/4). */
    private static SyntheticHf.Builder base() {
        SyntheticHf.Builder b = SyntheticHf.llama(C.nLayer(), N_EMBD, C.nHead(), C.nHeadKv(), VOCAB, StDType.F32);
        for (int il = 0; il < C.nLayer(); il++) {
            String l = "model.layers." + il + ".";
            tagRows(b, l + "self_attn.q_proj.weight", N_EMBD, N_EMBD);
            tagRows(b, l + "self_attn.k_proj.weight", KV_DIM, N_EMBD);
        }
        return b;
    }

    private static void tagRows(SyntheticHf.Builder b, String name, int rows, int cols) {
        b.tensor(name, StDType.F32, new long[]{rows, cols}, SyntheticHf.f32(tags(rows, cols)));
    }

    private static float[] tags(int rows, int cols) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                v[i * cols + j] = i + 0.25f * j;
            }
        }
        return v;
    }

    /** The values of the A factor stored as {@code {r, in}} (or {@code {r, vocab}} for the embedding). */
    private static float[] aValues(int in) {
        float[] v = new float[R * in];
        for (int k = 0; k < R; k++) {
            for (int j = 0; j < in; j++) {
                v[k * in + j] = j + 0.5f * k;
            }
        }
        return v;
    }

    /** The values of the B factor stored as {@code {out, r}}, one row tag per output row. */
    private static float[] bValues(int out) {
        return tags(out, R);
    }

    /** The two factor names of {@code module} written as F32 tensors of the PEFT factor shapes. */
    private static void addPair(Map<String, long[]> shapes, Map<String, float[]> values, String module, int in,
                                int out) {
        String a = PREFIX + module + ".lora_A.weight";
        shapes.put(a, new long[]{R, in});
        values.put(a, aValues(in));
        String b = PREFIX + module + ".lora_B.weight";
        shapes.put(b, new long[]{out, R});
        values.put(b, bValues(out));
    }

    private static void addTensor(Map<String, long[]> shapes, Map<String, float[]> values, String name, long[] shape,
                                  float[] v) {
        shapes.put(name, shape);
        values.put(name, v);
    }

    /** Writes the adapter directory: {@code adapter_config.json} plus the F32 tensors as the safetensors file. */
    private static void writeAdapter(Path dir, String configText, Map<String, long[]> shapes,
                                     Map<String, float[]> values) throws IOException {
        SafetensorsWriter.Builder w = SafetensorsWriter.builder();
        for (Map.Entry<String, long[]> e : shapes.entrySet()) {
            w.add(e.getKey(), StDType.F32, e.getValue(), SyntheticHf.f32(values.get(e.getKey())));
        }
        w.metadata("format", "pt");
        w.build().write(dir.resolve("adapter_model.safetensors"));
        Files.write(dir.resolve("adapter_config.json"),
                SyntheticHf.json(configText).getBytes(StandardCharsets.UTF_8));
    }

    private static String config(String extras) {
        return "{'peft_type':'lora','r':" + R + ",'lora_alpha':" + (int) ALPHA
                + (extras.isEmpty() ? "" : "," + extras) + "}";
    }

    private Path newDir() throws IOException {
        return Files.createDirectory(tmp.getRoot().toPath().resolve("peft" + ++index));
    }

    private Path hfBase() throws IOException {
        return base().write(Files.createDirectory(tmp.getRoot().toPath().resolve("hfbase" + ++index)));
    }

    private Path ggufBase() throws IOException {
        return SyntheticModels.llama().writeTempFile();
    }

    private SourceOrigin hfSource(Path dir) {
        return new SourceOrigin(dir, ModelFormat.SAFETENSORS);
    }

    // ------------------------------------------------------------------ read helpers

    private static List<String> names(List<GGUFTensorInfo> infos) {
        List<String> out = new ArrayList<>();
        for (GGUFTensorInfo t : infos) {
            out.add(t.name());
        }
        return out;
    }

    private static GGUFTensorInfo info(ModelSource s, String name) {
        for (GGUFTensorInfo t : s.getTensors()) {
            if (t.name().equals(name)) {
                return t;
            }
        }
        throw new AssertionError(name + " is not a tensor of " + s);
    }

    private static byte[] produce(com.eonmux.neuralfree.gguf.hf.tensor.RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        return b.array();
    }

    // ------------------------------------------------------------------ the PEFT branch

    @Test
    public void thePeftBranchPlansTheLlamaFactorsOfAnHfBase() throws Exception {
        Path base = hfBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        for (int il = 0; il < C.nLayer(); il++) {
            addPair(shapes, values, "model.layers." + il + ".self_attn.q_proj", N_EMBD, N_EMBD);
            addPair(shapes, values, "model.layers." + il + ".self_attn.k_proj", N_EMBD, KV_DIM);
        }
        addTensor(shapes, values, PREFIX + "model.embed_tokens.lora_embedding_A", new long[]{R, VOCAB},
                aValues(VOCAB));
        addTensor(shapes, values, PREFIX + "model.embed_tokens.lora_embedding_B", new long[]{N_EMBD, R},
                bValues(N_EMBD));
        addTensor(shapes, values, PREFIX + "model.layers.0.self_attn.q_proj.base_layer.weight",
                new long[]{N_EMBD, N_EMBD}, tags(N_EMBD, N_EMBD));
        writeAdapter(dir, config(""), shapes, values);

        ModelSource source = ModelSources.openAdapter(dir.toString(), hfSource(base), options);
        assertTrue("the PEFT branch of openAdapter builds the adapter view", source instanceof PeftAdapterSource);
        try (PeftAdapterSource s = (PeftAdapterSource) source) {
            assertEquals(ModelFormat.SAFETENSORS, s.format());
            assertEquals(dir.toString(), s.getFilePath());
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals("adapter", s.getString(GgufKeys.GENERAL_TYPE));
            assertEquals("lora", s.getString(GgufKeys.ADAPTER_TYPE));
            assertEquals(ALPHA, ((Number) s.getMetadata().get(GgufKeys.ADAPTER_LORA_ALPHA)).floatValue(), 0f);
            assertEquals(GGUFValueType.FLOAT32, s.getValueType(GgufKeys.ADAPTER_LORA_ALPHA));
            assertNull("no aLoRA keys in the config", s.getMetadata().get(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS));

            List<String> names = names(s.getTensors());
            assertEquals("two pairs per layer plus the embedding pair, the .base_layer.weight skipped", 10,
                    s.getTensorCount());
            for (int il = 0; il < C.nLayer(); il++) {
                assertTrue(names.contains("blk." + il + ".attn_q.weight.lora_a"));
                assertTrue(names.contains("blk." + il + ".attn_q.weight.lora_b"));
                assertTrue(names.contains("blk." + il + ".attn_k.weight.lora_a"));
                assertTrue(names.contains("blk." + il + ".attn_k.weight.lora_b"));
            }
            assertTrue(names.contains("token_embd.weight.lora_a"));
            assertTrue(names.contains("token_embd.weight.lora_b"));
            assertFalse("the base-layer copy is not a factor", names.stream().anyMatch(n -> n.contains("base_layer")));

            assertArrayEquals("a projection factor is ne {in, r}", new long[]{N_EMBD, R},
                    s.plan().ne("blk.0.attn_q.weight.lora_a"));
            assertArrayEquals("a projection factor is ne {r, out}", new long[]{R, N_EMBD},
                    s.plan().ne("blk.0.attn_q.weight.lora_b"));
            assertArrayEquals(new long[]{N_EMBD, KV_DIM}, s.plan().ne("blk.1.attn_k.weight.lora_a"));
            assertArrayEquals(new long[]{R, KV_DIM}, s.plan().ne("blk.1.attn_k.weight.lora_b"));
            assertArrayEquals("the token-embedding A is planned transposed ({r, vocab}, not {vocab, r})",
                    new long[]{R, VOCAB}, s.plan().ne("token_embd.weight.lora_a"));
            assertArrayEquals(new long[]{R, N_EMBD}, s.plan().ne("token_embd.weight.lora_b"));
            assertEquals(GGMLType.F32, info(s, "blk.0.attn_q.weight.lora_a").type());

            assertArrayEquals("the A factor of the q pair reads back as stored",
                    SyntheticHf.f32(aValues(N_EMBD)), produce(s.plan().tensor("blk.0.attn_q.weight.lora_a").source()));
            assertArrayEquals("the B factor of the embedding pair reads back as stored",
                    SyntheticHf.f32(bValues(N_EMBD)), produce(s.plan().tensor("token_embd.weight.lora_b").source()));
            float[] stored = aValues(VOCAB);
            float[] transposed = new float[VOCAB * R];
            for (int k = 0; k < R; k++) {
                for (int j = 0; j < VOCAB; j++) {
                    transposed[j * R + k] = stored[k * VOCAB + j];
                }
            }
            assertArrayEquals("the embedding A factor is the stored {r, vocab} matrix transposed to {vocab, r}",
                    SyntheticHf.f32(transposed), produce(s.plan().tensor("token_embd.weight.lora_a").source()));

            try (WeightStore store = s.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
                GGUFTensorInfo t = info(s, "token_embd.weight.lora_b");
                byte[] read = new byte[(int) t.sizeInBytes()];
                store.readTensor(t.offset(), read.length, read);
                assertArrayEquals(SyntheticHf.f32(bValues(N_EMBD)), read);
            }
        }
    }

    // ------------------------------------------------------------------ metadata

    @Test
    public void theMetadataCarriesTheRsloraAlphaAndTheALoRAInvocationTokens() throws Exception {
        Path base = hfBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "model.layers.0.self_attn.q_proj", N_EMBD, N_EMBD);
        writeAdapter(dir, "{'peft_type':'lora','r':4,'lora_alpha':16,'use_rslora':true,"
                + "'alora_invocation_tokens':[3,5,4096]}", shapes, values);

        try (PeftAdapterSource s = PeftAdapterSource.open(dir, hfSource(base), options)) {
            assertEquals("NF alpha = 16 * sqrt(4)", 32.0f,
                    ((Number) s.getMetadata().get(GgufKeys.ADAPTER_LORA_ALPHA)).floatValue(), 0f);
            assertEquals("adapter.alora.invocation_tokens is an array of uint32", GGUFValueType.ARRAY,
                    s.getValueType(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS));
            assertEquals(GGUFValueType.UINT32, s.getArrayElementType(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS));
            assertEquals(Arrays.asList(3, 5, 4096), s.getMetadata().get(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS));
            assertEquals(2, s.getTensorCount());
        }
    }

    @Test
    public void anInvocationStringIsTokenizedWithTheBaseVocabularyWithoutItsFirstToken() throws Exception {
        Path base = hfBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "model.layers.0.self_attn.q_proj", N_EMBD, N_EMBD);
        writeAdapter(dir, config("'invocation_string':'</s>ab'"), shapes, values);

        try (PeftAdapterSource s = PeftAdapterSource.open(dir, hfSource(base), options)) {
            List<?> tokens = (List<?>) s.getMetadata().get(GgufKeys.ADAPTER_ALORA_INVOCATION_TOKENS);
            assertEquals("'</s>' is token 1 and 'ab' is token 4 of the synthetic GPT-2 vocabulary; the first id is"
                            + " dropped", Arrays.asList(4), tokens);
        }
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    public void aModuleWithOnlyOneFactorHalfIsRefusedWithTheMissingHalfNamed() throws Exception {
        Path base = hfBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        String a = PREFIX + "model.layers.0.self_attn.q_proj.lora_A.weight";
        addTensor(shapes, values, a, new long[]{R, N_EMBD}, aValues(N_EMBD));
        writeAdapter(dir, config(""), shapes, values);

        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PeftAdapterSource.open(dir, hfSource(base), options));
        assertTrue(e.getMessage(), e.getMessage().contains("has no lora_B half"));
    }

    @Test
    public void aGgufBaseWithoutTheHfConfigGivesTheNamedError() throws Exception {
        Path gguf = ggufBase();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "model.layers.0.self_attn.q_proj", N_EMBD, N_EMBD);

        Path unnamed = newDir();
        writeAdapter(unnamed, config(""), shapes, values);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> PeftAdapterSource.open(unnamed, new SourceOrigin(gguf, ModelFormat.GGUF), options));
        assertTrue(e.getMessage(), e.getMessage().contains("a PEFT adapter on a GGUF base needs the base model's"
                + " Hugging Face config.json"));
        assertTrue(e.getMessage(), e.getMessage().contains("--lora-base-config <dir>"));

        Path named = newDir();
        writeAdapter(named, config("'base_model_name_or_path':'does/not/exist'"), shapes, values);
        GgufFormatException missing = assertThrows(GgufFormatException.class,
                () -> PeftAdapterSource.open(named, new SourceOrigin(gguf, ModelFormat.GGUF), options));
        assertTrue(missing.getMessage(), missing.getMessage().contains("--lora-base-config <dir>"));
    }

    // ------------------------------------------------------------------ the base configuration resolution

    @Test
    public void baseModelNameOrPathResolvesAnExistingLocalDirectory() throws Exception {
        Path base = hfBase();
        Path gguf = ggufBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "model.layers.0.self_attn.q_proj", N_EMBD, N_EMBD);
        writeAdapter(dir, config("'base_model_name_or_path':'../" + base.getFileName() + "'"), shapes, values);

        try (PeftAdapterSource s = PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), options)) {
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertTrue(names(s.getTensors()).contains("blk.0.attn_q.weight.lora_a"));
        }
    }

    @Test
    public void loraBaseConfigNamesADirectoryOrItsConfigJson() throws Exception {
        Path base = hfBase();
        Path gguf = ggufBase();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "model.layers.0.self_attn.q_proj", N_EMBD, N_EMBD);
        Path dir = newDir();
        writeAdapter(dir, config(""), shapes, values);

        ModelSourceOptions viaDir = options.toBuilder().loraBaseConfig(base).build();
        try (PeftAdapterSource s = PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), viaDir)) {
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertTrue(names(s.getTensors()).contains("blk.0.attn_q.weight.lora_a"));
        }
        ModelSourceOptions viaFile = options.toBuilder().loraBaseConfig(base.resolve("config.json")).build();
        try (PeftAdapterSource s = PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), viaFile)) {
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertTrue(names(s.getTensors()).contains("blk.0.attn_q.weight.lora_a"));
        }
    }

    // ------------------------------------------------------------------ the trace

    @Test
    public void theGgufBaseTraceIsAcceptedByLoraAdapterLoader() throws Exception {
        Path base = hfBase();
        Path gguf = ggufBase();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        for (int il = 0; il < C.nLayer(); il++) {
            addPair(shapes, values, "model.layers." + il + ".self_attn.q_proj", N_EMBD, N_EMBD);
            addPair(shapes, values, "model.layers." + il + ".self_attn.k_proj", N_EMBD, KV_DIM);
        }
        writeAdapter(dir, config(""), shapes, values);

        ModelSourceOptions withCfg = options.toBuilder().loraBaseConfig(base).build();
        try (CpuWeights weights = CpuWeights.open(new GGUFFileParser(gguf.toString()))) {
            LoraAdapter lora = LoraAdapterLoader.load(dir.toString(), weights, withCfg);
            assertEquals("llama", lora.arch());
            assertEquals(ALPHA, lora.alpha(), 0f);
            assertEquals("a plain LoRA is not activated", 0, lora.invocationTokens().length);
            assertEquals(new LinkedHashSet<>(Arrays.asList("blk.0.attn_q.weight", "blk.0.attn_k.weight",
                    "blk.1.attn_q.weight", "blk.1.attn_k.weight")), lora.targets());

            LoraAdapter.Pair q0 = lora.pair("blk.0.attn_q.weight");
            assertEquals(LoraAdapter.Target.PROJECTION, q0.kind());
            assertEquals(N_EMBD, q0.inDim());
            assertEquals(N_EMBD, q0.outDim());
            assertEquals(R, q0.rank());
            assertEquals(KV_DIM, lora.pair("blk.1.attn_k.weight").outDim());

            float[] a = lora.delta("blk.0.attn_q.weight", R / ALPHA).a();
            for (int k = 0; k < R; k++) {
                for (int j = 0; j < N_EMBD; j++) {
                    assertEquals("A[" + k + "][" + j + "]", j + 0.5f * k, a[k * N_EMBD + j], 0f);
                }
            }

            try (HfModelSource baseView = HfModelSource.open(base, options, null)) {
                assertPermutedRows(baseView, lora, "blk.0.attn_q.weight");
                assertPermutedRows(baseView, lora, "blk.0.attn_k.weight");
                assertPermutedRows(baseView, lora, "blk.1.attn_q.weight");
                assertPermutedRows(baseView, lora, "blk.1.attn_k.weight");
            }
        }
    }

    /**
     * The tags of the {@code target} rows of the base view (one tag per stored weight row) must be the same tags the
     * adapter's B factor carries per GGUF row: the q/k B factor is permuted exactly like the base weight.
     */
    private static void assertPermutedRows(HfModelSource baseView, LoraAdapter lora, String target) throws IOException {
        long[] ne = baseView.plan().ne(target);
        int rowLen = (int) ne[0];
        int rows = (int) ne[1];
        ByteBuffer le = ByteBuffer.wrap(produce(baseView.plan().tensor(target).source()))
                .order(ByteOrder.LITTLE_ENDIAN);
        float[] baseTags = new float[rows];
        for (int i = 0; i < rows; i++) {
            baseTags[i] = le.getFloat(i * rowLen * 4);
        }
        float[] bRows = lora.delta(target, R / ALPHA).scaledB();
        assertEquals(target, rows * R, bRows.length);
        for (int i = 0; i < rows; i++) {
            assertEquals("row " + i + " of " + target + " carries the tag of the permuted base row", baseTags[i],
                    bRows[i * R], 0f);
        }
    }
}
