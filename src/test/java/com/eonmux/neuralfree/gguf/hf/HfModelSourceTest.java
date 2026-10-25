package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelFormat;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSourceOptions.OutType;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.StoreUse;
import com.eonmux.neuralfree.gguf.TensorReadMode;
import com.eonmux.neuralfree.gguf.TensorRegions;
import com.eonmux.neuralfree.gguf.WeightStore;
import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchEntry;
import com.eonmux.neuralfree.gguf.hf.arch.HfArchRegistry;
import com.eonmux.neuralfree.gguf.hf.arch.HfLoadContext;
import com.eonmux.neuralfree.gguf.hf.arch.HfResources;
import com.eonmux.neuralfree.gguf.hf.config.HfQuantMethod;
import com.eonmux.neuralfree.gguf.hf.meta.HfMetadata;
import com.eonmux.neuralfree.gguf.hf.quant.HfQuantRewriter;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainer;
import com.eonmux.neuralfree.gguf.hf.st.CheckpointContainers;
import com.eonmux.neuralfree.gguf.hf.store.VirtualLayout;
import com.eonmux.neuralfree.gguf.hf.tensor.HfTensorIndex;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hf.tensor.TensorPlan;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.memory.KvBlockCodec;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.quant.ActivationQuant;
import com.eonmux.neuralfree.gguf.quant.FpBlockDecoders;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.After;
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
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * {@link HfModelSource} end to end on synthetic checkpoints opened through {@link TestOnlyLlamaEntry}: a two-layer
 * Llama traced through every step of {@link HfSourceAssembler} to its tensor infos and read back through the weight
 * store and the CPU regions; sharded and cache-layout checkpoints; MTP layers through a load-scoped name filter and an
 * indexed block count; {@code --outtype auto}; KV-cache scales; the ModelOpt {@code prec_a4} flags; the online
 * Hadamard keys.
 */
public class HfModelSourceTest {

    private static final int LAYERS = 2;
    private static final int DIM = 64;
    private static final int HEADS = 4;
    private static final int KV_HEADS = 2;
    private static final int VOCAB = 16;
    private static final int KV_DIM = DIM / HEADS * KV_HEADS;
    private static final int FFN = 2 * DIM;
    private static final String L0 = "model.layers.0.";
    private static final String L1 = "model.layers.1.";
    private static final Pattern LAYER_TENSOR = Pattern.compile("model\\.layers\\.(\\d+)\\.(.+)\\.weight");

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private HfArchRegistry.Overlay overlay;
    private ModelSourceOptions options;

    @Before
    public void install() throws IOException {
        overlay = HfArchRegistry.withExtraForTest(Collections.singletonList(TestOnlyLlamaEntry.entry()));
        options = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    @After
    public void uninstall() {
        overlay.close();
    }

    private static SyntheticHf.Builder llama() {
        return SyntheticHf.llama(LAYERS, DIM, HEADS, KV_HEADS, VOCAB);
    }

    private Path write(SyntheticHf.Builder b) throws IOException {
        return b.write(tmp.newFolder().toPath());
    }

    private static String config(String extraMembers) {
        return SyntheticHf.json("{'architectures':['LlamaForCausalLM'],'model_type':'llama','hidden_size':" + DIM
                + ",'intermediate_size':" + FFN + ",'num_hidden_layers':" + LAYERS + ",'num_attention_heads':" + HEADS
                + ",'num_key_value_heads':" + KV_HEADS + ",'max_position_embeddings':128,'rms_norm_eps':1e-05,"
                + "'rope_theta':10000.0,'vocab_size':" + VOCAB + ",'bos_token_id':0,'eos_token_id':1"
                + (extraMembers.isEmpty() ? "" : "," + extraMembers) + "}");
    }

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

    private static byte[] produce(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        return b.array();
    }

    private static float[] floats(byte[] le) {
        ByteBuffer b = ByteBuffer.wrap(le).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[le.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    private static byte[] read(WeightStore store, GGUFTensorInfo t) throws IOException {
        byte[] out = new byte[(int) t.sizeInBytes()];
        store.readTensor(t.offset(), out.length, out);
        return out;
    }

    private static byte[] bytes(ByteBuffer b) {
        byte[] out = new byte[b.remaining()];
        b.duplicate().get(out);
        return out;
    }

    private static int intValue(Object v) {
        return ((Number) v).intValue();
    }

    /** The GGUF names of one Llama block. */
    private static List<String> block(int bid) {
        List<String> out = new ArrayList<>();
        for (String s : new String[]{"attn_norm", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_norm",
                "ffn_gate", "ffn_up", "ffn_down"}) {
            out.add("blk." + bid + "." + s + ".weight");
        }
        return out;
    }

    private static final Map<String, String> LAYER_NAMES = layerNames();

    private static Map<String, String> layerNames() {
        Map<String, String> m = new HashMap<>();
        m.put("input_layernorm", "attn_norm");
        m.put("self_attn.q_proj", "attn_q");
        m.put("self_attn.k_proj", "attn_k");
        m.put("self_attn.v_proj", "attn_v");
        m.put("self_attn.o_proj", "attn_output");
        m.put("post_attention_layernorm", "ffn_norm");
        m.put("mlp.gate_proj", "ffn_gate");
        m.put("mlp.up_proj", "ffn_up");
        m.put("mlp.down_proj", "ffn_down");
        return m;
    }

    /** The GGUF name the converter gives a Llama checkpoint tensor. */
    private static String ggufOf(String hf) {
        switch (hf) {
            case "lm_head.weight":
                return "output.weight";
            case "model.embed_tokens.weight":
                return "token_embd.weight";
            case "model.norm.weight":
                return "output_norm.weight";
            default:
                break;
        }
        Matcher m = LAYER_TENSOR.matcher(hf);
        if (!m.matches() || !LAYER_NAMES.containsKey(m.group(2))) {
            throw new AssertionError("not a Llama tensor: " + hf);
        }
        return "blk." + m.group(1) + "." + LAYER_NAMES.get(m.group(2)) + ".weight";
    }

    // ------------------------------------------------------------------ the eleven steps

    @Test
    public void aTwoLayerLlamaGoesThroughEveryStep() throws Exception {
        Path dir = write(llama());
        HfSourceAssembler.Config c = HfSourceAssembler.config(dir, options);
        assertEquals("LlamaForCausalLM", c.config().architecture());
        assertEquals(HfQuantMethod.NONE, c.quant().method());
        assertEquals(dir.toAbsolutePath().normalize(), c.dir());

        HfResources resources = new HfResources();
        try (CheckpointContainer ckpt = HfSourceAssembler.checkpoint(dir, c, options)) {
            assertEquals(1, ckpt.segments().size());

            HfLoadContext resolved = HfSourceAssembler.resolve(c, ckpt, HfArchEntry.Kind.TEXT, options, resources,
                    null);
            assertEquals(Arch.LLAMA, resolved.arch());
            assertTrue(resolved.entry().hfNames().contains("LlamaForCausalLM"));
            assertEquals(3 + 9 * LAYERS, resolved.index().names().size());

            HfLoadContext indexed = HfSourceAssembler.index(resolved);
            assertEquals(LAYERS, indexed.blockCount());

            HfQuantRewriter.Result q = HfSourceAssembler.quant(indexed, c.quant(), ActivationQuant.WEIGHT_ONLY);
            assertSame("an unquantized index is not rewritten", indexed.index(), q.index());
            assertTrue(q.kvScales().isEmpty());

            HfSourceAssembler.Filtered f = HfSourceAssembler.filter(indexed, q, HfArchEntry.Kind.TEXT);
            assertEquals(3 + 9 * LAYERS, f.ctx().index().names().size());
            assertEquals(Collections.emptyList(), f.droppedPrefixes());
            assertEquals(f.ctx().index().names(), f.guessIndex().names());

            HfSourceAssembler.Vocab v = HfSourceAssembler.vocab(f.ctx());
            List<?> tokens = (List<?>) v.metadata().get(GgufKeys.TOKENIZER_LIST);
            assertEquals("padded to the configured vocabulary", VOCAB, tokens.size());
            assertEquals("<s>", tokens.get(0));
            assertEquals(VOCAB, v.ctx().vocabTokenCount().getAsInt());

            HfSourceAssembler.Planned p = HfSourceAssembler.plan(v.ctx(), f, q.moduleArgs());
            TensorPlan plan = p.plan();
            List<String> expected = new ArrayList<>();
            for (String hf : f.ctx().index().names()) {
                expected.add(ggufOf(hf));
            }
            assertEquals("the plan follows the index order", expected, plan.names());
            List<String> all = new ArrayList<>(Arrays.asList("token_embd.weight", "output_norm.weight",
                    "output.weight"));
            all.addAll(block(0));
            all.addAll(block(1));
            assertEquals(new HashSet<>(all), new HashSet<>(plan.names()));
            assertEquals(GGMLType.BF16, plan.type("blk.0.attn_q.weight"));
            assertEquals("norms are F32", GGMLType.F32, plan.type("blk.0.attn_norm.weight"));
            assertArrayEquals(new long[]{DIM, KV_DIM}, plan.ne("blk.1.attn_k.weight"));
            assertArrayEquals(new long[]{FFN, DIM}, plan.ne("blk.1.ffn_down.weight"));
            assertTrue(p.ctx().precA4().isEmpty());

            HfMetadata md = HfSourceAssembler.metadata(p.ctx(), plan, q.hadamard().orElse(null), v.metadata(),
                    f.droppedPrefixes(), c.quant());
            assertEquals("llama", md.get(GgufKeys.GENERAL_ARCHITECTURE));
            assertEquals(LAYERS, intValue(md.get("llama.block_count")));
            assertEquals(DIM, intValue(md.get("llama.embedding_length")));
            assertEquals(HEADS, intValue(md.get("llama.attention.head_count")));
            assertEquals(KV_HEADS, intValue(md.get("llama.attention.head_count_kv")));
            assertEquals(dir.getFileName().toString(), md.get(GgufKeys.GENERAL_NAME));
            assertEquals(HfSourceAssembler.TYPE_MODEL, md.get(GgufKeys.GENERAL_TYPE));
            assertEquals("BF16 matrices", HfFileType.of(GGMLType.BF16), intValue(md.get(GgufKeys.GENERAL_FILE_TYPE)));
            assertEquals(HfSourceAssembler.QUANTIZATION_VERSION,
                    intValue(md.get(GgufKeys.GENERAL_QUANTIZATION_VERSION)));
            assertEquals(HfSourceAssembler.SOURCE_FORMAT, md.get(GgufKeys.NF_SOURCE_FORMAT));
            assertEquals("none", md.get(GgufKeys.NF_SOURCE_QUANT_METHOD));
            assertEquals(Collections.emptyList(), md.get(GgufKeys.NF_HF_DROPPED_PREFIXES));
            assertNull("no extension types", md.get(GgufKeys.NF_EXTENSION_TYPES));
            assertEquals("gpt2", md.get(GgufKeys.TOKENIZER_MODEL));

            HfSourceAssembler.validate(p.ctx(), plan, md);

            VirtualLayout layout = HfSourceAssembler.layout(ckpt, plan);
            assertEquals(expected, names(layout.tensors()));
            assertEquals("the five norms are converted", 5, layout.produced().size());
            assertEquals(0, layout.producedBase() % VirtualLayout.ALIGNMENT);
            for (GGUFTensorInfo t : layout.tensors()) {
                PlannedTensor pt = plan.tensor(t.name());
                assertEquals(t.name(), pt.byteSize(), t.sizeInBytes());
                if (pt.type() == GGMLType.BF16) {
                    assertEquals(t.name(), pt.source().directVirtualOffset(), t.offset());
                } else {
                    assertTrue(t.name(), t.offset() >= layout.producedBase());
                    assertEquals(t.name(), 0, t.offset() % VirtualLayout.ALIGNMENT);
                }
            }
        } finally {
            resources.close();
        }
    }

    @Test
    public void theSourceServesTheViewThroughStoresAndRegions() throws IOException {
        Path dir = write(llama());
        try (HfModelSource s = HfModelSource.open(dir, options, null)) {
            assertEquals(ModelFormat.SAFETENSORS, s.format());
            assertEquals(dir.toString(), s.getFilePath());
            assertEquals(3 + 9 * LAYERS, s.getTensorCount());
            assertEquals("llama", s.getString(GgufKeys.GENERAL_ARCHITECTURE));
            assertNull("a non-string value is not a string", s.getString("llama.block_count"));
            assertThrows(UnsupportedOperationException.class, () -> s.getMetadata().put("x", 1));
            assertTrue(s.getValueType(GgufKeys.TOKENIZER_LIST) != null);
            assertTrue(s.getArrayElementType(GgufKeys.TOKENIZER_LIST) != null);

            GGUFTensorInfo embd = info(s, "token_embd.weight");
            GGUFTensorInfo norm = info(s, "blk.0.attn_norm.weight");
            byte[] embdBytes = produce(s.plan().tensor("token_embd.weight").source());
            byte[] normBytes = produce(s.plan().tensor("blk.0.attn_norm.weight").source());
            assertArrayEquals("BF16 to F32 is exact", SyntheticHf.values(1, DIM), floats(normBytes), 0f);

            for (StoreUse use : StoreUse.values()) {
                try (WeightStore store = s.createMappedStore(use)) {
                    assertArrayEquals(use.toString(), embdBytes, read(store, embd));
                    assertArrayEquals(use.toString(), normBytes, read(store, norm));
                }
            }
            for (TensorReadMode mode : TensorReadMode.values()) {
                try (TensorRegions regions = s.openRegions(mode)) {
                    assertArrayEquals(mode.toString(), embdBytes, bytes(regions.region(embd, 0, embd.sizeInBytes())));
                    assertArrayEquals(mode.toString(), normBytes, bytes(regions.region(norm, 0, norm.sizeInBytes())));
                }
            }
            WeightStore kept = s.createMappedStore(StoreUse.STREAMING);
            s.close();
            assertThrows("closing the source closes its stores", IOException.class, () -> read(kept, norm));
            assertThrows(IOException.class, () -> s.createMappedStore(StoreUse.STREAMING));
        }
    }

    @Test
    public void modelSourcesOpensCheckpointDirectoriesAndFiles() throws IOException {
        Path dir = write(llama());
        try (ModelSource s = ModelSources.open(dir.toString(), options, null)) {
            assertTrue(s instanceof HfModelSource);
            assertEquals(3 + 9 * LAYERS, s.getTensorCount());
        }
        try (ModelSource s = ModelSources.open(dir.resolve("model.safetensors").toString(), options, null,
                ActivationQuant.CHECKPOINT)) {
            assertTrue(s instanceof HfModelSource);
            assertEquals(dir.toAbsolutePath().normalize(), ((HfModelSource) s).context().config().dir()
                    .toAbsolutePath().normalize());
        }
    }

    @Test
    public void aDetectorLayoutWithoutConfigJsonIsACheckpointNotAnAdapter() throws IOException {
        Path dir = tmp.newFolder().toPath();
        Files.write(dir.resolve("lev_release.json"), "{}".getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("adapter_config.json"), "{}".getBytes(StandardCharsets.UTF_8));
        Files.write(dir.resolve("adapter_model.safetensors"), "not read".getBytes(StandardCharsets.UTF_8));
        assertEquals(ModelFormat.SAFETENSORS, ModelSources.detect(dir));
    }

    @Test
    public void shardedAndCacheLayoutCheckpointsReadAcrossFiles() throws IOException {
        Path dir = write(llama().shards(3).symlinkSnapshot(true));
        assertEquals(SyntheticHf.REVISION, dir.getFileName().toString());
        try (HfModelSource s = HfModelSource.open(dir, options, null);
             WeightStore store = s.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            assertEquals(3, s.layout().segments().size());
            assertEquals(3 + 9 * LAYERS, s.getTensorCount());
            for (GGUFTensorInfo t : s.getTensors()) {
                assertArrayEquals(t.name(), produce(s.plan().tensor(t.name()).source()), read(store, t));
            }
        }
    }

    // ------------------------------------------------------------------ MTP layers

    private SyntheticHf.Builder llamaWithMtp() {
        SyntheticHf.Builder b = llama();
        SyntheticHf.llamaLayer(b, TestOnlyLlamaEntry.MTP_LAYERS + "0.", DIM, KV_DIM, FFN);
        return b;
    }

    @Test
    public void mtpLayersFollowTheLoadScopedFilterAndTheIndexedBlockCount() throws IOException {
        Path dir = write(llamaWithMtp());
        try (HfModelSource s = HfModelSource.open(dir, options, null)) {
            assertEquals("two configured layers and one MTP layer", 3, s.context().blockCount());
            assertEquals(3, intValue(s.getMetadata().get("llama.block_count")));
            List<String> names = names(s.getTensors());
            assertTrue(names.containsAll(block(2)));
            assertEquals(3 + 9 * 3, names.size());
            assertArrayEquals(produce(s.plan().tensor("blk.2.attn_q.weight").source()),
                    produce(s.context().index().get("model.layers.2.self_attn.q_proj.weight")));
        }
    }

    @Test
    public void excludingMtpDropsTheExtraLayers() throws IOException {
        Path dir = write(llamaWithMtp());
        ModelSourceOptions noMtp = options.toBuilder().includeMtp(false).build();
        try (HfModelSource s = HfModelSource.open(dir, noMtp, null)) {
            assertEquals(LAYERS, s.context().blockCount());
            assertEquals(LAYERS, intValue(s.getMetadata().get("llama.block_count")));
            List<String> names = names(s.getTensors());
            assertEquals(3 + 9 * LAYERS, names.size());
            for (String n : names) {
                assertFalse(n, n.startsWith("blk.2."));
            }
        }
    }

    // ------------------------------------------------------------------ output type

    @Test
    public void autoOutTypeIsGuessedFromTheStoredDtypesBeforeTheRewrite() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.llama(LAYERS, DIM, HEADS, KV_HEADS, VOCAB, StDType.F16)
                .config(config("'torch_dtype':'float16','quantization_config':{'quant_method':'fp8',"
                        + "'activation_scheme':'dynamic'}"));
        float[] channel = new float[DIM];
        Arrays.fill(channel, 0.5f);
        b.first(L0 + "self_attn.q_proj.weight_scale", StDType.BF16, new long[]{DIM}, SyntheticHf.bf16(channel));
        b.first(L0 + "self_attn.q_proj.weight", StDType.F8_E5M2, new long[]{DIM, DIM}, new byte[DIM * DIM]);
        Path dir = write(b.shards(11));

        ModelSourceOptions auto = options.toBuilder().outType(OutType.AUTO).build();
        try (CheckpointContainer ckpt = CheckpointContainers.open(dir, dir, false)) {
            assertEquals("the FP8 weight is the checkpoint's first tensor", L0 + "self_attn.q_proj.weight",
                    HfTensorIndex.of(ckpt).names().get(0));
        }
        try (HfModelSource s = HfModelSource.open(dir, auto, null)) {
            assertEquals("the stored F16 tensors decide, not the BF16 rewrite of the FP8 weight", GGMLType.F16,
                    s.plan().type("token_embd.weight"));
            assertEquals(GGMLType.F16, s.plan().type("blk.0.attn_q.weight"));
            assertEquals(GGMLType.F16, s.plan().type("blk.1.attn_q.weight"));
            assertEquals("fp8", s.getString(GgufKeys.NF_SOURCE_QUANT_METHOD));
        }
    }

    // ------------------------------------------------------------------ quantization metadata

    @Test
    public void kvCacheScalesBecomeF32Tensors() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'compressed-tensors'}"));
        b.tensor(L0 + "self_attn.k_scale", StDType.F32, new long[]{}, SyntheticHf.f32(0.5f));
        b.tensor(L1 + "self_attn.kv_scale", StDType.F32, new long[]{}, SyntheticHf.f32(0.25f));
        Path dir = write(b);
        try (HfModelSource s = HfModelSource.open(dir, options, null);
             WeightStore store = s.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            for (String[] e : new String[][]{{"blk.0.attn_k.kv_scale", "0.5"}, {"blk.1.attn_k.kv_scale", "0.25"},
                    {"blk.1.attn_v.kv_scale", "0.25"}}) {
                GGUFTensorInfo t = info(s, e[0]);
                assertEquals(GGMLType.F32, t.type());
                assertArrayEquals(new float[]{Float.parseFloat(e[1])}, floats(read(store, t)), 0f);
            }
            assertFalse(names(s.getTensors()).contains("blk.0.attn_v.kv_scale"));
        }
    }

    @Test
    public void kvCacheScalesOfMtpLayersFollowTheNameFilter() throws IOException {
        SyntheticHf.Builder b = llamaWithMtp().config(config("'quantization_config':{'quant_method':"
                + "'compressed-tensors'}"));
        b.tensor(L0 + "self_attn.k_scale", StDType.F32, new long[]{}, SyntheticHf.f32(0.5f));
        b.tensor(TestOnlyLlamaEntry.MTP_LAYERS + "0.self_attn.k_scale", StDType.F32, new long[]{},
                SyntheticHf.f32(0.75f));
        Path dir = write(b);
        try (HfModelSource s = HfModelSource.open(dir, options, null);
             WeightStore store = s.createMappedStore(StoreUse.RESIDENT_UPLOAD)) {
            assertArrayEquals(new float[]{0.5f}, floats(read(store, info(s, "blk.0.attn_k.kv_scale"))), 0f);
            assertArrayEquals("the MTP layer's scale is in the block its layer becomes", new float[]{0.75f},
                    floats(read(store, info(s, "blk.2.attn_k.kv_scale"))), 0f);
        }
        ModelSourceOptions noMtp = options.toBuilder().includeMtp(false).build();
        try (HfModelSource s = HfModelSource.open(dir, noMtp, null)) {
            List<String> names = names(s.getTensors());
            assertTrue(names.contains("blk.0.attn_k.kv_scale"));
            assertFalse(names.contains("blk.2.attn_k.kv_scale"));
        }
    }

    /** A ModelOpt NVFP4 module of {@code DIM x DIM}: packed codes, unit sub-scales and a global scale of 0.5. */
    private static void nvfp4(SyntheticHf.Builder b, String module, boolean withInputScale) {
        byte[] codes = new byte[DIM * DIM / 2];
        for (int i = 0; i < codes.length; i++) {
            codes[i] = (byte) (i * 37);
        }
        byte[] scales = new byte[DIM * DIM / 16];
        Arrays.fill(scales, (byte) 0x38);
        b.tensor(module + ".weight", StDType.U8, new long[]{DIM, DIM / 2}, codes);
        b.tensor(module + ".weight_scale", StDType.F8_E4M3, new long[]{DIM, DIM / 16}, scales);
        b.tensor(module + ".weight_scale_2", StDType.F32, new long[]{}, SyntheticHf.f32(0.5f));
        if (withInputScale) {
            b.tensor(module + ".input_scale", StDType.F32, new long[]{}, SyntheticHf.f32(2f));
        }
    }

    @Test
    public void modelOptMixedPrecisionTagsWeightOnlyNvfp4WeightsWithPrecA4() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'modelopt',"
                + "'quant_algo':'MIXED_PRECISION','quantized_layers':{"
                + "'model.layers.0.self_attn.q_proj':{'quant_algo':'W4A16_NVFP4','group_size':16},"
                + "'model.layers.1.self_attn.q_proj':{'quant_algo':'NVFP4','group_size':16}}}"));
        nvfp4(b, L0 + "self_attn.q_proj", false);
        nvfp4(b, L1 + "self_attn.q_proj", true);
        Path dir = write(b);
        try (HfModelSource s = HfModelSource.open(dir, options, null)) {
            assertEquals(GGMLType.NVFP4, s.plan().type("blk.0.attn_q.weight"));
            assertEquals(GGMLType.NVFP4, s.plan().type("blk.1.attn_q.weight"));
            Map<String, Object> md = s.getMetadata();
            assertEquals(Collections.singletonList("blk.0.attn_q.weight"),
                    md.get(GgufKeys.GENERAL_TENSOR_EXTRA_NAME));
            assertEquals(Collections.singletonList(false), md.get(GgufKeys.GENERAL_TENSOR_EXTRA_PREC_A4));
            assertEquals(Collections.singletonMap("blk.0.attn_q.weight", false), s.context().precA4());
        }
    }

    @Test
    public void anOnlineCompressedTensorsHadamardReachesThePrismKeys() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'compressed-tensors',"
                + "'transform_config':{'config_groups':{'u':{'type':'hadamard',"
                + "'apply':[{'targets':['Linear'],'location':'input'}]}}}}"));
        for (String layer : new String[]{L0, L1}) {
            b.tensor(layer + "mlp.down_proj.u_input.weight", StDType.BF16, new long[]{FFN, FFN},
                    new byte[2 * FFN * FFN]);
        }
        Path dir = write(b);
        try (HfModelSource s = HfModelSource.open(dir, options, null)) {
            Map<String, Object> md = s.getMetadata();
            assertEquals(1, intValue(md.get(GgufKeys.PRISM_HADAMARD_VERSION)));
            assertEquals(FFN, intValue(md.get(GgufKeys.PRISM_HADAMARD_BLOCK_SIZE)));
            assertEquals("identity", md.get(GgufKeys.PRISM_HADAMARD_SIGN_MODE));
            assertEquals(Arrays.asList("blk.0.ffn_down.weight", "blk.1.ffn_down.weight"),
                    md.get(GgufKeys.PRISM_HADAMARD_WEIGHT_NAMES));
            assertTrue(md.containsKey(GgufKeys.PRISM_HADAMARD_TRANSFORM));
            assertTrue(md.containsKey(GgufKeys.PRISM_HADAMARD_AXIS));
            for (String n : names(s.getTensors())) {
                assertFalse("the transform matrices are consumed: " + n, n.contains("u_input"));
            }
        }
    }

    // ------------------------------------------------------------------ online quantization and act-quant scales

    /** The FP8 per-channel weight and its F32 scale of an fbgemm_fp8 module, as the checkpoint stores them. */
    private static void fp8Channels(SyntheticHf.Builder b, String module, int out, int in) {
        byte[] codes = new byte[out * in];
        for (int i = 0; i < codes.length; i++) {
            codes[i] = (byte) (((i * 37) & 0x7E) + (i % 2 == 0 ? 0x00 : 0x81));
        }
        float[] scales = new float[out];
        Arrays.fill(scales, 0.125f);
        b.tensor(module + ".weight", StDType.F8_E4M3, new long[]{out, in}, codes);
        b.tensor(module + ".weight_scale", StDType.F32, new long[]{out, 1}, SyntheticHf.f32(scales));
    }

    @Test
    public void anFbgemmCheckpointWritesTheActivationScaleUpperBound() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'fbgemm_fp8',"
                + "'activation_scale_ub':0.5}"));
        fp8Channels(b, L0 + "mlp.down_proj", DIM, FFN);
        fp8Channels(b, L1 + "mlp.down_proj", DIM, FFN);
        Path dir = write(b);
        try (HfModelSource s = HfModelSource.open(dir, options, null)) {
            Map<String, Object> md = s.getMetadata();
            assertEquals("the configured activation_scale_ub", 0.5f,
                    ((Number) md.get(GgufKeys.NF_ACT_QUANT_SCALE_UB)).floatValue(), 0f);
            List<?> weights = (List<?>) md.get(GgufKeys.NF_ACT_QUANT_WEIGHT_NAMES);
            assertTrue("the FP8_TOKEN weight names are written",
                    !weights.isEmpty() && weights.contains("blk.0.ffn_down.weight"));
        }
    }

    @Test
    public void anUnboundedFbgemmActivationScaleRefusesAtTheMetadataStep() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'fbgemm_fp8',"
                + "'activation_scale_ub':0}"));
        fp8Channels(b, L0 + "mlp.down_proj", DIM, FFN);
        Path dir = write(b);
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.open(dir, options, null));
        assertTrue(e.getMessage(), e.getMessage().contains(GgufKeys.NF_ACT_QUANT_SCALE_UB));
    }

    /** The Q8_0 requantization of {@code rows} bf16 rows of {@code inner} values, row by row, as KvBlockCodec stores. */
    private static byte[] requantizedQ8_0(byte[] bf16Rows, int inner, int rows) {
        ByteBuffer in = ByteBuffer.wrap(bf16Rows).order(ByteOrder.LITTLE_ENDIAN);
        int rowBytes = inner / 32 * GGMLType.Q8_0.getTypeSize();
        ByteBuffer out = ByteBuffer.allocate(rows * rowBytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] x = new float[inner];
        byte[] block = new byte[GGMLType.Q8_0.getTypeSize()];
        for (int r = 0; r < rows; r++) {
            for (int i = 0; i < inner; i++) {
                x[i] = FpBlockDecoders.bf16ToFp32(in.getShort() & 0xFFFF);
            }
            for (int b = 0; b < inner / 32; b++) {
                KvBlockCodec.quantize(KvDType.Q8_0, x, b * 32, out, r * rowBytes + b * block.length);
            }
        }
        return out.array();
    }

    @Test
    public void quantOnlineRequantizesTheEligiblePlannedWeights() throws IOException {
        Path dir = write(llama());
        try (HfModelSource plain = HfModelSource.open(dir, options, null)) {
            assertEquals(GGMLType.BF16, info(plain, "blk.0.attn_q.weight").type());
            byte[] bf16Rows = produce(plain.plan().tensor("blk.0.attn_q.weight").source());

            ModelSourceOptions online = options.toBuilder()
                    .onlineQuant(ModelSourceOptions.OnlineQuant.Q8_0).build();
            try (HfModelSource s = HfModelSource.open(dir, online, null)) {
                GGUFTensorInfo q8 = info(s, "blk.0.attn_q.weight");
                assertEquals(GGMLType.Q8_0, q8.type());
                assertEquals(DIM * (DIM / 32 * GGMLType.Q8_0.getTypeSize()), q8.sizeInBytes());
                assertArrayEquals("row-by-row the same bytes the bf16 repack requantizes to",
                        requantizedQ8_0(bf16Rows, DIM, DIM),
                        produce(s.plan().tensor("blk.0.attn_q.weight").source()));
                assertEquals("the ranks stay F32", GGMLType.F32, s.plan().type("blk.0.attn_norm.weight"));
                assertEquals("file_type recomputes from the requantized plan", HfFileType.of(GGMLType.Q8_0),
                        intValue(s.getMetadata().get(GgufKeys.GENERAL_FILE_TYPE)));
            }
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void aProjectorViewWithoutAProjectorConversionIsRefused() throws IOException {
        Path dir = write(llama());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options));
        assertTrue(e.getMessage(), e.getMessage().contains("has no projector conversion"));
    }

    @Test
    public void anUnregisteredArchitectureIsRefused() throws IOException {
        Path dir = write(llama().config(SyntheticHf.json("{'architectures':['NoSuchModelForCausalLM'],"
                + "'num_hidden_layers':2}")));
        assertThrows(GgufFormatException.class, () -> HfModelSource.open(dir, options, null));
    }

    @Test
    public void checkpointModeRefusesAnOutputActivationScaleWhileOpening() throws IOException {
        SyntheticHf.Builder b = llama().config(config("'quantization_config':{'quant_method':'fp8',"
                + "'activation_scheme':'dynamic'}"));
        b.tensor(L0 + "self_attn.o_proj.output_scale", StDType.F32, new long[]{}, SyntheticHf.f32(2f));
        Path dir = write(b);
        assertThrows(GgufFormatException.class,
                () -> HfModelSource.open(dir, options, null, ActivationQuant.CHECKPOINT));
        try (HfModelSource s = HfModelSource.open(dir, options, null, ActivationQuant.WEIGHT_ONLY)) {
            assertEquals(3 + 9 * LAYERS, s.getTensorCount());
        }
    }
}
