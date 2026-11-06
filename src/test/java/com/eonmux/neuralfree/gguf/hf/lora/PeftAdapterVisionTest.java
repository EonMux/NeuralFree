package com.eonmux.neuralfree.gguf.hf.lora;

import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.SourceOrigin;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapter;
import com.eonmux.neuralfree.gguf.ops.heads.LoraAdapterLoader;
import com.eonmux.neuralfree.gguf.spec.TensorCatalog;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.gguf.testutil.SyntheticModels;
import com.eonmux.neuralfree.io.safetensors.SafetensorsWriter;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Assert;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * A PEFT adapter whose modules sit in the vision tower and the projector (vLLM {@code enable_tower_connector_lora}
 * accepts their Hugging Face names): the text route plans the language pairs and the projector conversion plans the
 * tower ({@code v.blk.<n>.attn_v.weight}) and connector ({@code mm.<n>.weight}) pairs of the base architecture's
 * MMPROJ entry, the factors carry the stored bytes, and the loaded adapter equals the same adapter converted to the
 * GGUF form. A vision module matching no projector pattern is refused by name, and so is a tower module on a base
 * whose architecture has no projector conversion.
 */
public class PeftAdapterVisionTest {

    private static final int R = 4;
    private static final float ALPHA = 8.0f;
    /** The module prefix of the PEFT factor names (LC/convert_lora_to_gguf.py:272-279). */
    private static final String PREFIX = "base_model.model.";
    private static final String TEXT_MODULE = "model.layers.0.self_attn.v_proj";
    private static final String TOWER_MODULE = "vision_tower.transformer.layers.0.attention.v_proj";
    private static final String PROJECTOR_MODULE = "multi_modal_projector.linear_1";
    private static final String TEXT_TARGET = "blk.0.attn_v.weight";
    private static final String TOWER_TARGET = "v.blk.0.attn_v.weight";
    private static final String PROJECTOR_TARGET = "mm.1.weight";
    /** The vision hidden size of the pixtral vision configuration. */
    private static final int V = 8;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options;
    private int index;

    @Before
    public void install() throws IOException {
        options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    // ------------------------------------------------------------------ fixtures

    /** 2 layers, n_embd 64, 4 heads / 2 KV heads of dim 16 (kv 32), FFN 128, vocab 320. */
    private static final SyntheticModels.Config C = SyntheticModels.Config.llama();
    private static final int N_EMBD = C.nEmbd();
    private static final int KV_DIM = C.nHeadKv() * C.headDim();

    private static float[] tags(int rows, int cols) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                v[i * cols + j] = i + 0.25f * j;
            }
        }
        return v;
    }

    /** The values of the A factor stored as {@code {r, in}}. */
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

    /**
     * The llava base of the GGUF checkpoint: the synthetic Llama checkpoint whose configuration names the llava
     * architecture and carries the pixtral vision configuration and the projector's spatial merge, plus the
     * preprocessor colors.
     */
    private Path llavaBase() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.llama(C.nLayer(), N_EMBD, C.nHead(), C.nHeadKv(), C.nVocab(), StDType.F32)
                .config(SyntheticHf.json("{'architectures':['LlavaForConditionalGeneration'],'model_type':'llava',"
                        + "'hidden_size':" + N_EMBD + ",'intermediate_size':" + (2 * N_EMBD)
                        + ",'num_hidden_layers':" + C.nLayer() + ",'num_attention_heads':" + C.nHead()
                        + ",'num_key_value_heads':" + C.nHeadKv() + ",'max_position_embeddings':128,"
                        + "'rms_norm_eps':1e-05,'rope_theta':10000.0,'vocab_size':" + C.nVocab()
                        + ",'torch_dtype':'float32','bos_token_id':0,'eos_token_id':1,'spatial_merge_size':2,"
                        + "'vision_config':{'model_type':'pixtral','hidden_size':" + V + ",'intermediate_size':16,"
                        + "'num_attention_heads':2,'image_size':64,'patch_size':4,'hidden_act':'gelu',"
                        + "'num_hidden_layers':4}}"));
        b.file("preprocessor_config.json",
                SyntheticHf.json("{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}"));
        return b.write(Files.createDirectory(tmp.getRoot().toPath().resolve("llava" + ++index)));
    }

    private Path llamaBase() throws IOException {
        return SyntheticHf.llama(C.nLayer(), N_EMBD, C.nHead(), C.nHeadKv(), C.nVocab(), StDType.F32)
                .write(Files.createDirectory(tmp.getRoot().toPath().resolve("llama" + ++index)));
    }

    /** An mmproj tensor table carrying the two vision targets, as an mmproj export of the base would. */
    private TensorCatalog visionCatalog() throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("clip");
        w.tensorF32(TOWER_TARGET, new long[]{V, V}, new float[V * V]);
        w.tensorF32(PROJECTOR_TARGET, new long[]{V, V}, new float[V * V]);
        Path file = tmp.newFolder("catalog").toPath().resolve("mmproj.gguf");
        w.writeTo(file);
        try (ModelSource source = ModelSources.parseGguf(file.toString())) {
            return TensorCatalog.of(source);
        }
    }

    /** The same factors as the PEFT adapter, as a converter would write them into the GGUF adapter. */
    private Path convertedAdapter(Map<String, long[]> textShapes) throws IOException {
        GgufTestWriter w = new GgufTestWriter().architecture("llama");
        w.str("general.type", "adapter").str("adapter.type", "lora").f32("adapter.lora.alpha", ALPHA);
        w.tensorF32(TEXT_TARGET + ".lora_a", new long[]{textShapes.get(TEXT_TARGET)[0], R}, aValues(N_EMBD));
        w.tensorF32(TEXT_TARGET + ".lora_b", new long[]{R, textShapes.get(TEXT_TARGET)[1]}, bValues(KV_DIM));
        w.tensorF32(TOWER_TARGET + ".lora_a", new long[]{V, R}, aValues(V));
        w.tensorF32(TOWER_TARGET + ".lora_b", new long[]{R, V}, bValues(V));
        w.tensorF32(PROJECTOR_TARGET + ".lora_a", new long[]{V, R}, aValues(V));
        w.tensorF32(PROJECTOR_TARGET + ".lora_b", new long[]{R, V}, bValues(V));
        Path file = tmp.newFolder("converted").toPath().resolve("adapter.gguf");
        w.writeTo(file);
        return file;
    }

    private long[] textTargetShape() {
        return SyntheticModels.llama().tensorShape(TEXT_TARGET);
    }

    // ------------------------------------------------------------------ read helpers

    private static List<String> names(List<com.eonmux.neuralfree.gguf.GGUFTensorInfo> infos) {
        List<String> out = new ArrayList<>();
        for (com.eonmux.neuralfree.gguf.GGUFTensorInfo t : infos) {
            out.add(t.name());
        }
        return out;
    }

    private static byte[] produce(com.eonmux.neuralfree.gguf.hf.tensor.RowSource source) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (source.rows() * source.rowBytes()))
                .order(ByteOrder.LITTLE_ENDIAN);
        source.readRows(0, (int) source.rows(), b);
        return b.array();
    }

    /** A PEFT adapter directory with the text, tower and connector pairs of the mixed adapter. */
    private Path mixedAdapter() throws IOException {
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, TEXT_MODULE, N_EMBD, KV_DIM);
        addPair(shapes, values, TOWER_MODULE, V, V);
        addPair(shapes, values, PROJECTOR_MODULE, V, V);
        writeAdapter(dir, config(""), shapes, values);
        return dir;
    }

    // ------------------------------------------------------------------ the vision route

    @Test
    public void aMixedPeftAdapterPlansTheTextTowerAndConnectorPairs() throws Exception {
        Path base = llavaBase();
        Path gguf = SyntheticModels.llama().writeTempFile();
        Path dir = mixedAdapter();
        ModelSourceOptions withCfg = options.toBuilder().loraBaseConfig(base).build();

        try (PeftAdapterSource s = PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), withCfg)) {
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            List<String> names = names(s.getTensors());
            assertEquals("one A and one B factor per module", 6, s.getTensorCount());
            for (String target : new String[]{TEXT_TARGET, TOWER_TARGET, PROJECTOR_TARGET}) {
                assertTrue(names.contains(target + ".lora_a"));
                assertTrue(names.contains(target + ".lora_b"));
            }
            assertArrayEquals("the text pair plans ne {in, r} / {r, out}", new long[]{N_EMBD, R},
                    s.plan().ne(TEXT_TARGET + ".lora_a"));
            assertArrayEquals(new long[]{R, KV_DIM}, s.plan().ne(TEXT_TARGET + ".lora_b"));
            assertArrayEquals("the tower pair plans the vision hidden size", new long[]{V, R},
                    s.plan().ne(TOWER_TARGET + ".lora_a"));
            assertArrayEquals(new long[]{R, V}, s.plan().ne(TOWER_TARGET + ".lora_b"));
            assertArrayEquals("the connector pair plans the projector's factors", new long[]{V, R},
                    s.plan().ne(PROJECTOR_TARGET + ".lora_a"));
            assertArrayEquals(new long[]{R, V}, s.plan().ne(PROJECTOR_TARGET + ".lora_b"));

            // v_proj factors carry no Q/K permute: the stored bytes pass through on both routes
            assertArrayEquals(SyntheticHf.f32(aValues(N_EMBD)),
                    produce(s.plan().tensor(TEXT_TARGET + ".lora_a").source()));
            assertArrayEquals(SyntheticHf.f32(bValues(KV_DIM)),
                    produce(s.plan().tensor(TEXT_TARGET + ".lora_b").source()));
            assertArrayEquals(SyntheticHf.f32(aValues(V)),
                    produce(s.plan().tensor(TOWER_TARGET + ".lora_a").source()));
            assertArrayEquals(SyntheticHf.f32(bValues(V)),
                    produce(s.plan().tensor(PROJECTOR_TARGET + ".lora_b").source()));
        }
    }

    @Test
    public void thePeftVisionAdapterLoadsLikeItsConvertedAdapter() throws Exception {
        Path base = llavaBase();
        Path gguf = SyntheticModels.llama().writeTempFile();
        Path dir = mixedAdapter();
        long[] textShape = textTargetShape();
        Path converted = convertedAdapter(Map.of(TEXT_TARGET, textShape));
        TensorCatalog catalog = visionCatalog();
        ModelSourceOptions withVision = options.toBuilder().loraBaseConfig(base)
                .loraVisionCatalog(catalog).build();

        LoraAdapter peft = load(dir.toString(), gguf.toString(), withVision);
        LoraAdapter fromGguf = load(converted.toString(), gguf.toString(), withVision);

        assertEquals(3, peft.targets().size());
        assertEquals("the PEFT and converted adapters serve the same targets",
                fromGguf.targets(), peft.targets());
        assertEquals(ALPHA, peft.alpha(), 0f);
        for (String target : peft.targets()) {
            LoraAdapter.Pair want = fromGguf.pair(target);
            LoraAdapter.Pair got = peft.pair(target);
            assertEquals("side of " + target, want.side(), got.side());
            assertEquals("in dim of " + target, want.inDim(), got.inDim());
            assertEquals("out dim of " + target, want.outDim(), got.outDim());
            assertEquals("rank of " + target, want.rank(), got.rank());
            assertEquals("kind of " + target, want.kind(), got.kind());
            // the deltas are the same matrices: s * B A over the base weight
            LoraAdapter.Delta wantDelta = fromGguf.delta(target, 1.0f);
            LoraAdapter.Delta gotDelta = peft.delta(target, 1.0f);
            assertArrayEquals("A of " + target, wantDelta.a(), gotDelta.a(), 0.0f);
            assertArrayEquals("scaled B of " + target, wantDelta.scaledB(), gotDelta.scaledB(), 0.0f);
        }
        assertEquals(LoraAdapter.Side.TEXT, peft.pair(TEXT_TARGET).side());
        assertEquals(LoraAdapter.Side.VISION, peft.pair(TOWER_TARGET).side());
        assertEquals(LoraAdapter.Side.VISION, peft.pair(PROJECTOR_TARGET).side());
        assertEquals(N_EMBD, peft.pair(TEXT_TARGET).inDim());
        assertEquals(KV_DIM, peft.pair(TEXT_TARGET).outDim());
        assertEquals(V, peft.pair(TOWER_TARGET).inDim());
        assertEquals(V, peft.pair(PROJECTOR_TARGET).inDim());
        assertEquals(ALPHA / R, peft.scale(PROJECTOR_TARGET, 1.0f), 0.0f);
    }

    private static LoraAdapter load(String adapter, String model, ModelSourceOptions options) throws IOException {
        try (ModelSource source = ModelSources.open(model, options);
             CpuWeights base = CpuWeights.open(source, TensorCatalog.of(source), null)) {
            return LoraAdapterLoader.load(adapter, base, options);
        }
    }

    // ------------------------------------------------------------------ the refusals

    @Test
    public void aVisionModuleMatchingNoProjectorPatternIsRefusedByName() throws Exception {
        Path base = llavaBase();
        Path gguf = SyntheticModels.llama().writeTempFile();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, "vision_tower.transformer.layers.0.attention.gate", V, V);
        writeAdapter(dir, config(""), shapes, values);

        ModelSourceOptions withCfg = options.toBuilder().loraBaseConfig(base).build();
        GgufFormatException e = Assert.assertThrows(GgufFormatException.class,
                () -> PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), withCfg));
        assertTrue(e.getMessage(), e.getMessage()
                .contains("vision_tower.transformer.layers.0.attention.gate"));
    }

    @Test
    public void aTowerModuleOnABaseWithoutAProjectorConversionIsRefused() throws Exception {
        Path base = llamaBase();
        Path gguf = SyntheticModels.llama().writeTempFile();
        Path dir = newDir();
        Map<String, long[]> shapes = new LinkedHashMap<>();
        Map<String, float[]> values = new LinkedHashMap<>();
        addPair(shapes, values, TOWER_MODULE, V, V);
        writeAdapter(dir, config(""), shapes, values);

        ModelSourceOptions withCfg = options.toBuilder().loraBaseConfig(base).build();
        GgufFormatException e = Assert.assertThrows(GgufFormatException.class,
                () -> PeftAdapterSource.open(dir, new SourceOrigin(gguf, ModelFormat.GGUF), withCfg));
        assertTrue(e.getMessage(), e.getMessage().contains("projector conversion"));
        assertTrue(e.getMessage(), e.getMessage().contains(TOWER_MODULE));
    }
}
