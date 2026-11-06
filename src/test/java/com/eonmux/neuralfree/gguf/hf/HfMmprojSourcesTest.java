package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The projector views ({@code convert_hf_to_gguf --mmproj}) of the Hugging Face checkpoints this package delivers,
 * against llama.cpp's {@code MmprojModel} subclasses: the {@code clip.*} hyperparameters and projector type of every
 * family, the tower tensor names of the converter's MMPROJ map, the fused Q/K/V and two-frame patch splits, the
 * Pixtral Q/K permutation and its {@code [IMG_BREAK]} row, the Gemma 3 norm-plus-one, the Qwen3-VL deepstack block
 * indexes, and the Whisper conv biases and the language-model drop of the inherited filter.
 */
public class HfMmprojSourcesTest {

    private static final int TEXT = 8;
    private static final int V_EMBD = 8;
    private static final int V_FF = 16;
    private static final int HEADS = 2;
    private static final int BLOCKS = 2;
    private static final int MEL = 4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    private ModelSourceOptions options() throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache").toPath()).build();
    }

    /** Options whose output type is {@code outType}; forces are observed against it. */
    private ModelSourceOptions options(ModelSourceOptions.OutType outType) throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache-" + outType).toPath()).outType(outType)
                .build();
    }

    /** Adds the F32 tensor {@code name} of {@code shape} holding {@link SyntheticHf#values} of {@code seed}. */
    private static void f32(SyntheticHf.Builder b, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape[i];
        }
        b.tensor(name, StDType.F32, dims, f32Bytes(seed, shape));
    }

    /** The little-endian F32 bytes of {@link SyntheticHf#values} of {@code seed} over {@code shape}. */
    private static byte[] f32Bytes(int seed, int... shape) {
        return SyntheticHf.f32(SyntheticHf.values(seed, count(shape)));
    }

    private static int count(int... shape) {
        int n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    private static void file(SyntheticHf.Builder b, String name, String json) {
        b.file(name, SyntheticHf.json(json));
    }

    private static final String COLORS = "{'image_mean':[0.5,0.5,0.5],'image_std':[0.25,0.5,0.75]}";

    // ------------------------------------------------------------------ Qwen2-VL

    private Path qwen2vl() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2VLForConditionalGeneration'],'model_type':'qwen2_vl','hidden_size':" + TEXT
                + ",'rms_norm_eps':1e-06,'vision_config':{'depth':4,'num_heads':2,'embed_dim':8,'hidden_size':16,"
                + "'patch_size':2,'spatial_merge_size':2}}"));
        f32(b, 100, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 101, "visual.patch_embed.proj.bias", 8);
        int seed = 110;
        for (int n = 0; n < 2; n++) {
            f32(b, seed++, "visual.blocks." + n + ".norm1.weight", 8);
            f32(b, seed++, "visual.blocks." + n + ".norm2.weight", 8);
            f32(b, seed++, "visual.blocks." + n + ".attn.qkv.weight", 24, 8);
            f32(b, seed++, "visual.blocks." + n + ".attn.qkv.bias", 24);
            f32(b, seed++, "visual.blocks." + n + ".attn.proj.weight", 8, 8);
            f32(b, seed++, "visual.blocks." + n + ".attn.proj.bias", 8);
            f32(b, seed++, "visual.blocks." + n + ".mlp.fc1.weight", 16, 8);
            f32(b, seed++, "visual.blocks." + n + ".mlp.fc1.bias", 16);
            f32(b, seed++, "visual.blocks." + n + ".mlp.fc2.weight", 8, 16);
            f32(b, seed++, "visual.blocks." + n + ".mlp.fc2.bias", 8);
        }
        f32(b, 130, "visual.merger.ln_q.weight", 8);
        f32(b, 131, "visual.merger.ln_q.bias", 8);
        f32(b, 132, "visual.merger.mlp.0.weight", 16, 32);
        f32(b, 133, "visual.merger.mlp.0.bias", 16);
        f32(b, 134, "visual.merger.mlp.2.weight", 8, 16);
        f32(b, 135, "visual.merger.mlp.2.bias", 8);
        f32(b, 140, "model.language_model.layers.0.mlp.gate_proj.weight", 16, 8);
        file(b, "preprocessor_config.json", COLORS);
        return b.write(tmp.newFolder("qwen2vl").toPath());
    }

    @Test
    public void bothPatchFramesKeepThePatchKernelType() throws IOException {
        // ".patch_embd.weight" is a substring rule, so it also covers the second frame ".weight.1"
        // (base.py:3230). At Q8_0 the rule stores F32 where an unforced frame would fall back to F16.
        try (HfModelSource s = HfModelSource.openProjector(qwen2vl(), options(ModelSourceOptions.OutType.Q8_0))) {
            assertEquals(GGMLType.F32, s.plan().type("v.patch_embd.weight"));
            assertEquals(GGMLType.F32, s.plan().type("v.patch_embd.weight.1"));
        }
    }

    @Test
    public void noMtpDropLeavesAProjectorAlone() throws IOException {
        // --no-mtp concerns text-model MTP layers: a projector checkpoint has no text blocks, so the option must
        // neither drop its numbered tensors nor read a text block count it does not have.
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2VLForConditionalGeneration'],'model_type':'qwen2_vl','hidden_size':" + TEXT
                + ",'vision_config':{'depth':4,'num_heads':2,'embed_dim':8,'hidden_size':16,'patch_size':2,"
                + "'spatial_merge_size':2}}"));
        f32(b, 300, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 301, "visual.merger.ln_q.weight", 8);
        f32(b, 302, "visual.merger.mlp.0.weight", 16, 32);
        f32(b, 303, "visual.merger.mlp.2.weight", 8, 16);
        file(b, "preprocessor_config.json", COLORS);
        Path dir = b.write(tmp.newFolder("qwen2vl-nomtp").toPath());
        ModelSourceOptions noMtp = ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache-nomtp").toPath())
                .includeMtp(false).build();
        try (HfModelSource s = HfModelSource.openProjector(dir, noMtp)) {
            assertEquals(4, s.context().blockCount());
            assertNotNull("the merger's numbered layer survives --no-mtp", s.plan().tensor("mm.2.weight"));
            assertNotNull(s.plan().tensor("mm.0.weight"));
        }
    }

    @Test
    public void qwen2vlViewCarriesTheMergerMetadataAndSplitTensors() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(qwen2vl(), options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("clip", kv.archName());
            assertEquals("qwen2vl_merger", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_VISION_ENCODER), false));
            assertEquals(TEXT, kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTION_DIM), 0));
            assertEquals("the image size defaults to the converter's 560 (qwenvl.py:53)", 560,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_IMAGE_SIZE), 0));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_PATCH_SIZE), 0));
            assertEquals(V_EMBD, kv.u32(kv.key(GgufKeys.CLIP_VISION_EMBEDDING_LENGTH), 0));
            assertEquals("the merger width of the renamed config (qwenvl.py:58)", V_FF,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH), 0));
            assertEquals(4, kv.u32(kv.key(GgufKeys.CLIP_VISION_BLOCK_COUNT), 0));
            assertEquals(HEADS, kv.u32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT), 0));
            assertEquals(1e-6f, kv.f32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON), 0f), 1e-9f);
            assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f}, colorOf(s, GgufKeys.CLIP_VISION_IMAGE_MEAN), 1e-9f);
            assertArrayEquals(new float[]{0.25f, 0.5f, 0.75f}, colorOf(s, GgufKeys.CLIP_VISION_IMAGE_STD), 1e-9f);
            assertEquals("mmproj", s.getString(GgufKeys.GENERAL_TYPE));

            assertNotNull(s.plan().tensor("v.patch_embd.weight"));
            assertNotNull(s.plan().tensor("v.patch_embd.weight.1"));
            assertNotNull(s.plan().tensor("v.blk.0.attn_q.weight"));
            assertNotNull(s.plan().tensor("v.blk.1.attn_k.bias"));
            assertNotNull(s.plan().tensor("v.blk.1.attn_v.bias"));
            assertNotNull(s.plan().tensor("v.post_ln.weight"));
            assertNotNull(s.plan().tensor("mm.0.weight"));
            assertNotNull(s.plan().tensor("mm.2.bias"));

            assertFalse("the language-model tensor is dropped",
                    names(s).stream().anyMatch(n -> n.contains("language")));
            assertFusedSplit(s, "v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight",
                    f32Bytes(112, 24, 8), 8);
            assertPatchFrames(s);
        }
    }

    private static List<String> names(HfModelSource s) {
        java.util.List<String> out = new java.util.ArrayList<>();
        for (com.eonmux.neuralfree.gguf.GGUFTensorInfo t : s.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static float[] colorOf(HfModelSource s, String key) {
        Object v = s.getMetadata().get(key);
        assertTrue(key + " is an array", v instanceof List);
        List<?> list = (List<?>) v;
        float[] out = new float[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) list.get(i)).floatValue();
        }
        return out;
    }

    /** The three thirds of a fused Q/K/V row split, against the checkpoint's rows. */
    private static void assertFusedSplit(HfModelSource s, String q, String k, String v, byte[] fused, int inner)
            throws IOException {
        int rows = fused.length / 4 / inner / 3;
        int third = rows * inner * 4;
        assertArrayEquals(Arrays.copyOfRange(fused, 0, third), bytes(s, q));
        assertArrayEquals(Arrays.copyOfRange(fused, third, 2 * third), bytes(s, k));
        assertArrayEquals(Arrays.copyOfRange(fused, 2 * third, 3 * third), bytes(s, v));
    }

    private static byte[] bytes(HfModelSource s, String name) throws IOException {
        return produce(s.plan().tensor(name).source());
    }

    private static byte[] produce(RowSource s) throws IOException {
        ByteBuffer b = ByteBuffer.allocate((int) (s.rows() * s.rowBytes())).order(ByteOrder.LITTLE_ENDIAN);
        s.readRows(0, (int) s.rows(), b);
        return b.array();
    }

    private static void assertPatchFrames(HfModelSource s) throws IOException {
        long[] ne = s.plan().ne("v.patch_embd.weight");
        assertArrayEquals(new long[]{2, 2, 3, 8}, ne);
        assertArrayEquals(new long[]{2, 2, 3, 8}, s.plan().ne("v.patch_embd.weight.1"));
        // frame k of the [c1, c2, kt, kh, kw] kernel holds the elements with dim-2 index k
        float[] stored = floats(f32Bytes(100, 8, 3, 2, 2, 2));
        float[] f0 = floats(bytes(s, "v.patch_embd.weight"));
        float[] f1 = floats(bytes(s, "v.patch_embd.weight.1"));
        int perC1 = 3 * 2 * 2 * 2;
        int perC2 = 2 * 2 * 2;
        int after = 3 * 2 * 2;
        int perKt = 2 * 2;
        for (int c1 = 0; c1 < 8; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {
                for (int i = 0; i < 4; i++) {
                    assertEquals("frame 0 of [" + c1 + "," + c2 + "]", stored[c1 * perC1 + c2 * perC2 + i],
                            f0[c1 * after + c2 * 4 + i], 1e-6f);
                    assertEquals("frame 1 of [" + c1 + "," + c2 + "]",
                            stored[c1 * perC1 + c2 * perC2 + perKt + i], f1[c1 * after + c2 * 4 + i], 1e-6f);
                }
            }
        }
    }

    @Test
    public void aQwen25vlCheckpointNamesItsWindowPattern() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2_5_VLForConditionalGeneration'],'model_type':'qwen2_5_vl','hidden_size':" + TEXT
                + ",'rms_norm_eps':1e-06,'vision_config':{'depth':4,'num_heads':2,'hidden_size':8,"
                + "'intermediate_size':16,'patch_size':2,'spatial_merge_size':2,"
                + "'fullatt_block_indexes':[1,3]}}"));
        f32(b, 150, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 151, "visual.merger.ln_q.weight", 8);
        f32(b, 152, "visual.merger.ln_q.bias", 8);
        f32(b, 153, "visual.merger.mlp.0.weight", 16, 32);
        f32(b, 154, "visual.merger.mlp.0.bias", 16);
        f32(b, 155, "visual.merger.mlp.2.weight", 8, 16);
        f32(b, 156, "visual.merger.mlp.2.bias", 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("qwen25vl").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("qwen2.5vl_merger", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_USE_SILU), false));
            assertEquals("the tower FFN stays intermediate_size without embed_dim (qwenvl.py:58-61)", V_FF,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH), 0));
            assertEquals("fullatt_block_indexes[0] + 1 (qwenvl.py:71)", 2,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_N_WA_PATTERN), 0));
        }
    }

    @Test
    public void anIrregularQwen25vlWindowPatternIsRefused() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2_5_VLForConditionalGeneration'],'model_type':'qwen2_5_vl','hidden_size':" + TEXT
                + ",'vision_config':{'depth':4,'num_heads':2,'hidden_size':8,'intermediate_size':16,"
                + "'patch_size':2,'fullatt_block_indexes':[1,4]}}"));
        f32(b, 160, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 161, "visual.merger.ln_q.weight", 8);
        f32(b, 162, "visual.merger.mlp.0.weight", 16, 32);
        f32(b, 163, "visual.merger.mlp.2.weight", 8, 16);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("qwen25vl-bad").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("fullatt_block_indexes"));
    }

    // ------------------------------------------------------------------ Qwen3-VL

    private Path qwen3vl() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen3VLForConditionalGeneration'],'model_type':'qwen3_vl','text_config':{'hidden_size':"
                + TEXT + ",'rms_norm_eps':1e-06},'vision_config':{'depth':4,'num_heads':2,'hidden_size':8,"
                + "'intermediate_size':16,'patch_size':2,'spatial_merge_size':2,'num_position_embeddings':16,"
                + "'deepstack_visual_indexes':[1,3]}}"));
        f32(b, 170, "model.visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 171, "model.visual.patch_embed.proj.bias", 8);
        f32(b, 172, "visual.pos_embed.weight", 16, 8);
        for (int n = 0; n < 2; n++) {
            f32(b, 173 + n, "visual.blocks." + n + ".norm1.weight", 8);
            f32(b, 175 + n, "visual.blocks." + n + ".attn.qkv.weight", 24, 8);
            f32(b, 177 + n, "visual.blocks." + n + ".attn.qkv.bias", 24);
        }
        f32(b, 180, "visual.merger.norm.weight", 8);
        f32(b, 181, "visual.merger.linear_fc1.weight", 16, 32);
        f32(b, 182, "visual.merger.linear_fc1.bias", 16);
        f32(b, 183, "visual.merger.linear_fc2.weight", 8, 16);
        f32(b, 184, "visual.merger.linear_fc2.bias", 8);
        f32(b, 185, "model.visual.deepstack_merger_list.0.norm.weight", 8);
        f32(b, 186, "model.visual.deepstack_merger_list.0.linear_fc1.weight", 16, 8);
        f32(b, 187, "model.visual.deepstack_merger_list.0.linear_fc2.weight", 8, 16);
        f32(b, 188, "model.visual.deepstack_merger_list.1.norm.weight", 8);
        f32(b, 189, "model.visual.deepstack_merger_list.1.linear_fc1.weight", 16, 8);
        f32(b, 190, "model.visual.deepstack_merger_list.1.linear_fc2.weight", 8, 16);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        return b.write(tmp.newFolder("qwen3vl").toPath());
    }

    @Test
    public void qwen3vlViewComputesTheImageSizeAndMapsTheDeepstackTaps() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(qwen3vl(), options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("qwen3vl_merger", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals("sqrt(num_position_embeddings) * patch_size (qwen3vl.py:55)", 8,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_IMAGE_SIZE), 0));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_USE_GELU), false));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE), 0));
            assertEquals("the text_config epsilon (qwen3vl.py:67)", 1e-6f,
                    kv.f32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON), 0f), 1e-9f);
            assertEquals("one flag per block, true at the deepstack taps (qwen3vl.py:66)",
                    List.of(false, true, false, true), flags(s, GgufKeys.CLIP_VISION_IS_DEEPSTACK_LAYERS));
            assertEquals(4, kv.u32(kv.key(GgufKeys.CLIP_VISION_BLOCK_COUNT), 0));
            assertEquals("the tower heads come from num_heads (qwen3vl.py:61)", HEADS,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT), 0));

            assertNotNull(s.plan().tensor("v.position_embd.weight"));
            assertNotNull("the patch bias of the qwen3vl projector (qwen3vl.py:131)",
                    s.plan().tensor("v.patch_embd.bias"));
            assertNotNull(s.plan().tensor("mm.0.weight"));
            assertNotNull(s.plan().tensor("mm.2.bias"));
            assertNotNull(s.plan().tensor("v.post_ln.weight"));
            assertEquals("the deepstack taps at their absolute block indexes (qwen3vl.py:117)", 8L,
                    s.plan().ne("v.deepstack.1.norm.weight")[0]);
            assertNotNull(s.plan().tensor("v.deepstack.3.fc1.weight"));
            assertFalse("the list indexes are replaced, not kept", s.plan().has("v.deepstack.0.norm.weight"));
            assertArrayEquals(new long[]{8, 24}, s.plan().ne("v.blk.0.attn_qkv.weight"));
        }
    }

    private static List<Boolean> flags(HfModelSource s, String key) {
        List<?> v = (List<?>) s.getMetadata().get(key);
        assertNotNull(v);
        return v.stream().map(o -> (Boolean) o).collect(java.util.stream.Collectors.toList());
    }

    // ------------------------------------------------------------------ GLM-4V

    @Test
    public void glm4vMapsItsMergerThroughTheConverterNameMap() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Glm4vForConditionalGeneration'],'model_type':'glm4v','hidden_size':" + TEXT
                + ",'vision_config':{'depth':4,'num_heads':2,'hidden_size':8,'intermediate_size':16,"
                + "'patch_size':2,'spatial_merge_size':2,'hidden_act':'gelu','rms_norm_eps':1e-05,'image_size':64}}"));
        f32(b, 200, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 201, "visual.patch_embed.proj.bias", 8);
        f32(b, 202, "visual.post_conv_layernorm.weight", 8);
        f32(b, 203, "visual.merger.ln_q.weight", 8);
        f32(b, 204, "visual.merger.proj.weight", 16, 32);
        f32(b, 205, "visual.blocks.0.attn.qkv.weight", 24, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("glm4v").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("glm4v", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_USE_GELU), false));
            assertEquals(1e-5f, kv.f32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON), 0f), 1e-9f);
            assertNotNull(s.plan().tensor("mm.model.fc.weight"));
            assertNotNull(s.plan().tensor("v.post_ln.weight"));
            assertNotNull(s.plan().tensor("v.norm_embd.weight"));
            assertNotNull(s.plan().tensor("v.blk.0.attn_qkv.weight"));
            assertNull("glm4v writes no deepstack keys",
                    s.getMetadata().get(GgufKeys.CLIP_VISION_IS_DEEPSTACK_LAYERS));
        }
    }

    // ------------------------------------------------------------------ Pixtral (LLaVA)

    private Path pixtral(String architectures, String rootModelType, String isCausal) throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':[" + architectures
                + "],'model_type':'" + rootModelType + "','hidden_size':" + TEXT + (isCausal == null ? ""
                : ",'is_causal':" + isCausal) + ",'spatial_merge_size':2,'vision_config':{'model_type':'pixtral',"
                + "'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,'image_size':64,'patch_size':4,"
                + "'hidden_act':'gelu','num_hidden_layers':4}}"));
        f32(b, 210, "vision_tower.patch_conv.weight", 16, 3, 4, 4);
        f32(b, 211, "vision_tower.transformer.layers.0.attention.q_proj.weight", 8, 8);
        f32(b, 212, "vision_tower.transformer.layers.0.attention.k_proj.weight", 8, 8);
        f32(b, 213, "vision_tower.transformer.layers.0.attention.v_proj.weight", 8, 8);
        f32(b, 214, "vision_tower.transformer.layers.0.attention_norm.weight", 8);
        f32(b, 215, "multi_modal_projector.linear_1.weight", 8, 8);
        f32(b, 216, "multi_modal_projector.linear_2.weight", 8, 8);
        f32(b, 217, "model.embed_tokens.weight", 16, 8);
        file(b, "tokenizer_config.json", "{'added_tokens_decoder':{'7':{'content':'[IMG_BREAK]',"
                + "'special':true}}}");
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        return b.write(tmp.newFolder("pixtral").toPath());
    }

    @Test
    public void thePixtralViewPermutesQKRowsAndExtractsTheImageBreakRow() throws IOException {
        try (HfModelSource s = HfModelSource.openProjector(
                pixtral("'LlavaForConditionalGeneration'", "llava", null), options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("pixtral", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_USE_GELU), false));
            assertEquals(1e-5f, kv.f32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON), 0f), 1e-9f);
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE), 0));

            assertPermutedRows(floats(bytes(s, "v.blk.0.attn_q.weight")), SyntheticHf.values(211, 64), 8, 2, 8);
            assertPermutedRows(floats(bytes(s, "v.blk.0.attn_k.weight")), SyntheticHf.values(212, 64), 8, 2, 8);
            assertArrayEquals(SyntheticHf.values(213, 64), floats(bytes(s, "v.blk.0.attn_v.weight")), 1e-6f);
            assertNotNull(s.plan().tensor("mm.1.weight"));
            assertNotNull(s.plan().tensor("mm.2.weight"));
            assertArrayEquals("the [IMG_BREAK] embedding is the one-dimensional row (llava.py:123-127)",
                    new long[]{8}, s.plan().ne("v.token_embd.img_break"));
            assertArrayEquals(row(SyntheticHf.values(217, 128), 8, 7),
                    floats(bytes(s, "v.token_embd.img_break")), 1e-6f);
            assertFalse("the rest of the embedding is dropped", s.plan().has("token_embd.weight"));
        }
    }

    /**
     * The rotate-half to interleaved row permutation of {@code LlamaModel.permute} over a {@code rows x width}
     * weight: output row {@code h * headWidth + w} is stored row {@code h * headWidth + (w % 2) * half + w / 2}.
     */
    private static void assertPermutedRows(float[] got, float[] stored, int rows, int heads, int width) {
        int headWidth = rows / heads;
        int half = headWidth / 2;
        float[] want = new float[stored.length];
        for (int h = 0; h < heads; h++) {
            for (int w = 0; w < headWidth; w++) {
                for (int c = 0; c < width; c++) {
                    want[(h * headWidth + w) * width + c] =
                            stored[(h * headWidth + (w % 2) * half + w / 2) * width + c];
                }
            }
        }
        assertArrayEquals(want, got, 1e-6f);
    }

    private static float[] row(float[] data, int width, int index) {
        return Arrays.copyOfRange(data, index * width, (index + 1) * width);
    }

    private static float[] floats(byte[] le) {
        ByteBuffer b = ByteBuffer.wrap(le).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[le.length / 4];
        for (int i = 0; i < out.length; i++) {
            out[i] = b.getFloat();
        }
        return out;
    }

    @Test
    public void aNonPixtralVisionTowerIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['LlavaForConditionalGeneration'],'hidden_size':8,'vision_config':{'model_type':"
                + "'clip_vision_model','hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'image_size':64,'patch_size':4,'hidden_act':'gelu','num_hidden_layers':4}}"));
        f32(b, 220, "vision_model.embeddings.patch_embedding.weight", 8, 3, 4, 4);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("llava-clip").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("clip_vision_model"));
    }

    @Test
    public void theEmbeddingVariantNeedsAHeadlessTextModel() throws IOException {
        Path dir = pixtral("'Mistral3Model'", "mistral3", "true");
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("is_causal"));
    }

    // ------------------------------------------------------------------ Gemma 3

    @Test
    public void gemma3ShiftsItsProjectorNormAndScalesThePool() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Gemma3ForConditionalGeneration'],'model_type':'gemma3','hidden_size':" + TEXT
                + ",'vision_config':{'model_type':'siglip_vision_model','hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'image_size':64,'patch_size':4,'layer_norm_eps':1e-06,"
                + "'num_hidden_layers':2}}"));
        f32(b, 230, "vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 3, 4, 4);
        f32(b, 231, "vision_tower.vision_model.embeddings.position_embedding.weight", 256, 8);
        f32(b, 232, "vision_tower.vision_model.post_layernorm.weight", 8);
        f32(b, 233, "vision_tower.vision_model.post_layernorm.bias", 8);
        f32(b, 234, "multi_modal_projector.mm_soft_emb_norm.weight", 8);
        f32(b, 235, "multi_modal_projector.mm_input_projection.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'image_seq_length':64}");
        Path dir = b.write(tmp.newFolder("gemma3").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("gemma3", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals("(64 / 4) / sqrt(64) = 2 (gemma.py:275-279)", 2,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR), 0));
            float[] norm = floats(bytes(s, "mm.soft_emb_norm.weight"));
            float[] stored = SyntheticHf.values(234, 8);
            for (int i = 0; i < norm.length; i++) {
                assertEquals("the Gemma3RMSNorm multiplier restored (gemma.py:306)", stored[i] + 1f, norm[i],
                        1e-4f);
            }
            assertEquals("the projector linears keep F16 (gemma.py:281-282)", GGMLType.F16,
                    s.plan().type("mm.input_projection.weight"));
        }
    }

    @Test
    public void gemma3KeepsItsTowerEmbeddingTensorsF32() throws IOException {
        // ".embeddings." of the checkpoint is F32 whatever the output type (gemma.py:280-284): the SigLIP patch
        // kernel and position table, while the projector linear keeps F16 (gemma.py:281).
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Gemma3ForConditionalGeneration'],'model_type':'gemma3','hidden_size':" + TEXT
                + ",'vision_config':{'model_type':'siglip_vision_model','hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'image_size':64,'patch_size':4,'layer_norm_eps':1e-06,"
                + "'num_hidden_layers':2}}"));
        f32(b, 236, "vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 3, 4, 4);
        f32(b, 237, "vision_tower.vision_model.embeddings.position_embedding.weight", 256, 8);
        f32(b, 238, "multi_modal_projector.mm_input_projection.weight", 8, 8);
        file(b, "preprocessor_config.json", COLORS);
        Path dir = b.write(tmp.newFolder("gemma3-f16").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options(ModelSourceOptions.OutType.F16))) {
            assertEquals(GGMLType.F32, s.plan().type("v.patch_embd.weight"));
            assertEquals(GGMLType.F32, s.plan().type("v.position_embd.weight"));
            assertEquals(GGMLType.F16, s.plan().type("mm.input_projection.weight"));
        }
    }

    // ------------------------------------------------------------------ SmolVLM

    @Test
    public void smolvlmKeepsTheConnectorAndNamesItsScale() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['SmolVLMForConditionalGeneration'],'model_type':'idefics3','scale_factor':2,'hidden_size':"
                + TEXT + ",'vision_config':{'model_type':'idefics3_vision','hidden_size':8,"
                + "'intermediate_size':16,'num_attention_heads':2,'image_size':32,'patch_size':2,"
                + "'layer_norm_eps':1e-05,'num_hidden_layers':2}}"));
        f32(b, 240, "model.vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 3, 2, 2);
        f32(b, 241, "model.vision_tower.vision_model.embeddings.position_embedding.weight", 256, 8);
        f32(b, 242, "model.vision_tower.vision_model.post_layernorm.weight", 8);
        f32(b, 243, "model.vision_tower.vision_model.post_layernorm.bias", 8);
        f32(b, 244, "model.connector.modality_projection.proj.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5],"
                + "'size':{'longest_edge':512}}");
        Path dir = b.write(tmp.newFolder("smolvlm").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("idefics3", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR), 0));
            assertEquals(512, kv.u32(kv.key(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE), 0));
            assertNotNull(s.plan().tensor("mm.model.fc.weight"));
            assertNotNull(s.plan().tensor("v.position_embd.weight"));
        }
    }

    @Test
    public void smolvlmKeepsItsTowerEmbeddingTensorsF32() throws IOException {
        // ".embeddings." of the checkpoint is F32 whatever the output type (smolvlm.py:43-46)
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['SmolVLMForConditionalGeneration'],'model_type':'idefics3','scale_factor':2,'hidden_size':"
                + TEXT + ",'vision_config':{'model_type':'idefics3_vision','hidden_size':8,"
                + "'intermediate_size':16,'num_attention_heads':2,'image_size':32,'patch_size':2,"
                + "'layer_norm_eps':1e-05,'num_hidden_layers':2}}"));
        f32(b, 245, "model.vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 3, 2, 2);
        f32(b, 246, "model.vision_tower.vision_model.embeddings.position_embedding.weight", 256, 8);
        f32(b, 247, "model.connector.modality_projection.proj.weight", 8, 8);
        file(b, "preprocessor_config.json", COLORS);
        Path dir = b.write(tmp.newFolder("smolvlm-f16").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options(ModelSourceOptions.OutType.F16))) {
            assertEquals(GGMLType.F32, s.plan().type("v.patch_embd.weight"));
            assertEquals(GGMLType.F32, s.plan().type("v.position_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ InternVL

    @Test
    public void internvlSplitsItsQkvAndScalesByTheDownsampleRatio() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['InternVisionModel'],'model_type':'internvl_chat','hidden_size':8,'downsample_ratio':0.5,"
                + "'min_dynamic_patch':1,'max_dynamic_patch':12,'vision_config':{'hidden_size':8,"
                + "'intermediate_size':16,'num_attention_heads':2,'image_size':448,'patch_size':14,"
                + "'layer_norm_eps':1e-06,'hidden_act':'gelu','num_hidden_layers':4}}"));
        f32(b, 250, "vision_model.embeddings.patch_embedding.weight", 8, 3, 14, 14);
        f32(b, 251, "vision_model.embeddings.class_embedding", 8);
        f32(b, 252, "vision_model.embeddings.position_embedding.weight", 1024, 8);
        f32(b, 253, "vision_model.encoder.layers.0.attn.qkv.weight", 24, 8);
        f32(b, 254, "vision_model.encoder.layers.0.attn.qkv.bias", 24);
        f32(b, 255, "vision_model.encoder.layers.0.ls1.weight", 8);
        f32(b, 256, "mlp1.0.weight", 8);
        f32(b, 257, "mlp1.1.weight", 8, 8);
        f32(b, 258, "mlp1.3.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("internvl").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("internvl", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR), 0));
            assertEquals(1, kv.u32(kv.key(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES), 0));
            assertEquals(12, kv.u32(kv.key(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES), 0));
            assertFusedSplit(s, "v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight",
                    f32Bytes(253, 24, 8), 8);
            assertNotNull(s.plan().tensor("v.class_embd"));
            assertNotNull(s.plan().tensor("v.blk.0.ls1.weight"));
            assertNotNull(s.plan().tensor("mm.model.mlp.0.weight"));
            assertNotNull(s.plan().tensor("mm.model.mlp.1.weight"));
            assertNotNull(s.plan().tensor("mm.model.mlp.3.weight"));
        }
    }

    // ------------------------------------------------------------------ Llama 4

    @Test
    public void llama4RoutesItsProjectorThroughTheFcTensor() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Llama4ForConditionalGeneration'],'model_type':'llama4','hidden_size':8,'vision_config':"
                + "{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,'image_size':32,'patch_size':2,"
                + "'norm_eps':1e-05,'pixel_shuffle_ratio':0.5,'hidden_act':'gelu','num_hidden_layers':4}}"));
        f32(b, 260, "vision_model.patch_embedding.linear.weight", 8, 3, 2, 2);
        f32(b, 261, "vision_model.class_embedding", 8);
        f32(b, 262, "vision_model.positional_embedding_vlm", 256, 8);
        f32(b, 263, "vision_model.model.layers.0.self_attn.q_proj.weight", 8, 8);
        f32(b, 264, "multi_modal_projector.linear_1.weight", 8, 8);
        f32(b, 265, "vision_model.vision_adapter.mlp.fc1.weight", 8, 8);
        f32(b, 266, "vision_model.vision_adapter.mlp.fc2.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("llama4").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("llama4", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR), 0));
            assertEquals(1e-5f, kv.f32(kv.key(GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON), 0f), 1e-9f);
            assertNotNull("the single fully connected layer keeps the fc name (llama4.py:24-26)",
                    s.plan().tensor("mm.model.fc.weight"));
            assertNotNull(s.plan().tensor("mm.model.mlp.2.weight"));
            assertNotNull(s.plan().tensor("v.position_embd.weight"));
            assertNotNull(s.plan().tensor("v.class_embd"));
        }
    }

    // ------------------------------------------------------------------ Kimi-VL

    @Test
    public void kimivlFlattensItsPositionTableAndSplitsWqkv() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['KimiVLForConditionalGeneration'],'model_type':'kimi_vl','hidden_size':8,'vision_config':"
                + "{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,'patch_size':16,"
                + "'depth':4}}"));
        f32(b, 270, "vision_tower.patch_embed.proj.weight", 8, 3, 16, 16);
        f32(b, 271, "vision_tower.patch_embed.pos_emb.weight", 4, 2, 8);
        f32(b, 272, "vision_tower.encoder.blocks.0.wqkv.weight", 24, 8);
        f32(b, 273, "vision_tower.encoder.blocks.0.norm0.weight", 8);
        f32(b, 274, "vision_tower.encoder.blocks.0.norm1.weight", 8);
        f32(b, 275, "multi_modal_projector.linear_1.weight", 8, 8);
        f32(b, 276, "multi_modal_projector.linear_2.weight", 8, 8);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5],'image_std':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("kimivl").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertEquals("kimivl", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals("the compatibility tile is 64 * 14, whatever the patch size (kimivl.py:24)", 896,
                    kv.u32(kv.key(GgufKeys.CLIP_VISION_IMAGE_SIZE), 0));
            assertEquals(16, kv.u32(kv.key(GgufKeys.CLIP_VISION_PATCH_SIZE), 0));
            assertEquals(2, kv.u32(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR), 0));
            assertArrayEquals("the [h, w, c] table flattens to [h * w, c] (kimivl.py:55-56)",
                    new long[]{8, 8}, s.plan().ne("v.position_embd.weight"));
            byte[] pos = f32Bytes(271, 4, 2, 8);
            assertArrayEquals(pos, bytes(s, "v.position_embd.weight"));
            assertFusedSplit(s, "v.blk.0.attn_q.weight", "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight",
                    f32Bytes(272, 24, 8), 8);
            assertNotNull(s.plan().tensor("mm.1.weight"));
            assertNotNull(s.plan().tensor("mm.2.weight"));
        }
    }

    // ------------------------------------------------------------------ Whisper audio

    @Test
    public void theQwen2AudioTowerWidensItsConvBiasesAndNamesTheProjector() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2AudioForConditionalGeneration'],'model_type':'qwen2_audio','hidden_size':8,"
                + "'audio_config':{'hidden_size':8,'d_model':8,'encoder_ffn_dim':16,'encoder_attention_heads':2,"
                + "'num_mel_bins':4,'num_hidden_layers':4,'layer_norm_eps':1e-05}}"));
        f32(b, 280, "audio_tower.conv1.weight", 8, 4, 3);
        f32(b, 281, "audio_tower.conv1.bias", 8);
        f32(b, 282, "audio_tower.conv2.weight", 8, 8, 3);
        f32(b, 283, "audio_tower.conv2.bias", 8);
        f32(b, 284, "audio_tower.embed_positions.weight", 300, 8);
        f32(b, 285, "audio_tower.layers.0.self_attn.q_proj.weight", 8, 8);
        f32(b, 286, "audio_tower.layer_norm.weight", 8);
        f32(b, 287, "multi_modal_projector.linear.weight", 8, 8);
        f32(b, 288, "multi_modal_projector.linear.bias", 8);
        Path dir = b.write(tmp.newFolder("qwen2audio").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertFalse("an audio-only file names no vision encoder", kv.has(kv.key(GgufKeys.CLIP_HAS_VISION_ENCODER)));
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertEquals("qwen2a", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals(MEL, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS), 0));
            assertEquals(TEXT, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_PROJECTION_DIM), 0));
            assertEquals(4, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_BLOCK_COUNT), 0));
            assertArrayEquals("the conv bias becomes a one-wide row (ultravox.py:129-131)",
                    new long[]{1, 8}, s.plan().ne("a.conv1d.1.bias"));
            assertArrayEquals(f32Bytes(281, 8), produceRow(s, "a.conv1d.1.bias"));
            assertNotNull(s.plan().tensor("a.position_embd.weight"));
            assertNotNull(s.plan().tensor("a.post_ln.weight"));
            assertNotNull(s.plan().tensor("mm.a.fc.weight"));
            assertNotNull(s.plan().tensor("mm.a.fc.bias"));
            assertNotNull(s.plan().tensor("a.blk.0.attn_q.weight"));
        }
    }

    /** The single row of a one-row planned tensor. */
    private static byte[] produceRow(HfModelSource s, String name) throws IOException {
        return bytes(s, name);
    }

    @Test
    public void meralionReadsItsSpeechConfigAndAdapterNames() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MERaLiON2ForConditionalGeneration'],'model_type':'meralion','hidden_size':" + TEXT
                + ",'speech_mlp_scale_factor':20,'speech_config':{'hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'num_mel_bins':4,'num_hidden_layers':3,'layer_norm_eps':1e-05}}"));
        f32(b, 310, "speech_encoder.conv1.weight", 8, 4, 3);
        f32(b, 311, "speech_encoder.conv1.bias", 8);
        f32(b, 312, "speech_encoder.layers.0.self_attn.q_proj.weight", 8, 8);
        f32(b, 313, "ln_speech.weight", 8);
        f32(b, 314, "speech_audio_adapter.mlp_adapter.0.weight", 8, 8);
        f32(b, 315, "speech_audio_adapter.gate_proj.weight", 8, 8);
        f32(b, 316, "speech_audio_adapter.pool_proj.weight", 8, 8);
        f32(b, 317, "speech_audio_adapter.out_proj.weight", 8, 8);
        Path dir = b.write(tmp.newFolder("meralion").toPath());
        try (HfModelSource s = HfModelSource.openProjector(dir, options())) {
            GgufKv kv = GgufKv.of(s);
            assertTrue(kv.bool(kv.key(GgufKeys.CLIP_HAS_AUDIO_ENCODER), false));
            assertFalse(kv.has(kv.key(GgufKeys.CLIP_HAS_VISION_ENCODER)));
            assertEquals("meralion", kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null));
            assertEquals(MEL, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_NUM_MEL_BINS), 0));
            assertEquals("the speech_config block count (ultravox.py:150-162)", 3,
                    kv.u32(kv.key(GgufKeys.CLIP_AUDIO_BLOCK_COUNT), 0));
            assertEquals(20, kv.u32(kv.key(GgufKeys.CLIP_AUDIO_PROJECTOR_STACK_FACTOR), 0));
            assertNotNull(s.plan().tensor("a.conv1d.1.weight"));
            assertNotNull(s.plan().tensor("a.blk.0.attn_q.weight"));
            assertNotNull(s.plan().tensor("mm.a.norm_pre.weight"));
            assertNotNull(s.plan().tensor("mm.a.mlp.0.weight"));
            assertNotNull(s.plan().tensor("mm.a.mlp.1.weight"));
            assertNotNull(s.plan().tensor("mm.a.mlp.2.weight"));
            assertNotNull(s.plan().tensor("mm.a.mlp.3.weight"));
        }
    }

    // ------------------------------------------------------------------ refusals

    @Test
    public void aVisionCheckpointWithoutItsConfigIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2VLForConditionalGeneration'],'model_type':'qwen2_vl','hidden_size':8,"
                + "'audio_config':{'hidden_size':8,'intermediate_size':16,'num_attention_heads':2,"
                + "'num_hidden_layers':2,'num_mel_bins':4}}"));
        f32(b, 290, "audio_tower.conv1.weight", 8, 4, 3);
        Path dir = b.write(tmp.newFolder("no-vision").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("vision_config"));
    }

    @Test
    public void aMissingPreprocessorColorIsRefusedByItsKey() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Qwen2VLForConditionalGeneration'],'model_type':'qwen2_vl','hidden_size':8,"
                + "'vision_config':{'depth':4,'num_heads':2,'embed_dim':8,'hidden_size':16,'patch_size':2}}"));
        f32(b, 291, "visual.patch_embed.proj.weight", 8, 3, 2, 2, 2);
        f32(b, 292, "visual.merger.mlp.0.weight", 16, 32);
        f32(b, 293, "visual.merger.mlp.2.weight", 8, 16);
        file(b, "preprocessor_config.json", "{'image_mean':[0.5,0.5,0.5]}");
        Path dir = b.write(tmp.newFolder("no-std").toPath());
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> HfModelSource.openProjector(dir, options()));
        assertTrue(e.getMessage(), e.getMessage().contains("image_std"));
    }
}
