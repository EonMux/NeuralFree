package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * The projector views of the vision-tower classes ported from llama.cpp's remaining {@code MmprojModel}
 * subclasses: the {@code clip.*} hyperparameters, the projector types and the tensor names the converter rewrites.
 * Each test builds a synthetic checkpoint with the configuration the class reads and the tensors it rewrites, opens
 * it through {@link HfModelSource#openProjector}, and checks the GGUF keys and planned tensor names.
 */
public class HfVisionTowerSourcesTest {

    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int HEADS = 2;
    private static final int BLOCKS = 2;

    private static final String COLORS =
            "'image_mean':[0.5,0.5,0.5],'image_std':[0.25,0.5,0.75]";

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    // ------------------------------------------------------------------ helpers

    private static void f32(SyntheticHf.Builder b, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = shape[i];
        }
        b.tensor(name, StDType.F32, dims, SyntheticHf.f32(SyntheticHf.values(seed, count(shape))));
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

    private ModelSourceOptions options(String name) throws IOException {
        return ModelSourceOptions.builder().cacheDir(tmp.newFolder("cache-" + name).toPath()).build();
    }

    private HfModelSource open(SyntheticHf.Builder b, String name) throws IOException {
        Path dir = b.write(tmp.newFolder(name).toPath());
        return HfModelSource.openProjector(dir, options(name));
    }

    private static String type(HfModelSource s) {
        GgufKv kv = GgufKv.of(s);
        String type = kv.str(kv.key(GgufKeys.CLIP_PROJECTOR_TYPE), null);
        return type != null ? type : kv.str(kv.key(GgufKeys.CLIP_VISION_PROJECTOR_TYPE), null);
    }

    private static int u32(HfModelSource s, String key, int dflt) {
        GgufKv kv = GgufKv.of(s);
        return kv.u32(kv.key(key), dflt);
    }

    private static float f32(HfModelSource s, String key, float dflt) {
        GgufKv kv = GgufKv.of(s);
        return kv.f32(kv.key(key), dflt);
    }

    private static boolean bool(HfModelSource s, String key) {
        GgufKv kv = GgufKv.of(s);
        return kv.bool(kv.key(key), false);
    }

    /** The base vision keys every {@code set_gguf_parameters} writes. */
    private static void assertTower(HfModelSource s, String projector, int imageSize, int patchSize, int blocks) {
        assertTower(s, projector, imageSize, patchSize, blocks, DIM, FF, HEADS);
    }

    private static void assertTower(HfModelSource s, String projector, int imageSize, int patchSize, int blocks,
                                    int hidden, int ffn, int heads) {
        assertEquals(projector, type(s));
        assertEquals("clip.vision.image_size", imageSize, u32(s, GgufKeys.CLIP_VISION_IMAGE_SIZE, -1));
        assertEquals("clip.vision.patch_size", patchSize, u32(s, GgufKeys.CLIP_VISION_PATCH_SIZE, -1));
        assertEquals("clip.vision.block_count", blocks, u32(s, GgufKeys.CLIP_VISION_BLOCK_COUNT, -1));
        assertEquals("clip.vision.embedding_length", hidden, u32(s, GgufKeys.CLIP_VISION_EMBEDDING_LENGTH, -1));
        assertEquals("clip.vision.feed_forward_length", ffn, u32(s, GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH, -1));
        assertEquals("clip.vision.attention.head_count", heads,
                u32(s, GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT, -1));
        assertEquals("clip.vision.projection_dim", DIM, u32(s, GgufKeys.CLIP_VISION_PROJECTION_DIM, -1));
        assertTrue("clip.has_vision_encoder", bool(s, GgufKeys.CLIP_HAS_VISION_ENCODER));
    }

    private static List<?> list(HfModelSource s, String key) {
        Object value = s.getMetadata().get(key);
        assertNotNull(key + " is missing", value);
        return (List<?>) value;
    }

    // ------------------------------------------------------------------ Qwen3-VL derived towers

    private static final String QWEN_VISION = "'vision_config':{'depth':" + BLOCKS + ",'num_heads':" + HEADS
            + ",'hidden_size':" + DIM + ",'intermediate_size':" + FF + ",'patch_size':2,"
            + "'spatial_merge_size':2,'num_position_embeddings':16,'deepstack_visual_indexes':[]}";

    private static SyntheticHf.Builder qwenCheckpoint(String architecture) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':['"
                + architecture + "'],'model_type':'qwen3_vl','text_config':{'hidden_size':" + DIM
                + ",'rms_norm_eps':1e-06}," + QWEN_VISION + "}"));
        f32(b, 1, "visual.patch_embed.proj.weight", DIM, 3, 2, 2, 2);
        f32(b, 2, "visual.blocks.0.attn.qkv.weight", 3 * DIM, DIM);
        f32(b, 3, "visual.merger.linear_fc1.weight", FF, 4 * DIM);
        f32(b, 4, "visual.merger.linear_fc2.weight", DIM, FF);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'size':{'shortest_edge':4096,"
                + "'longest_edge':16384}}");
        return b;
    }

    private static void assertQwenTower(HfModelSource s, String projector) {
        assertTower(s, projector, 8, 2, BLOCKS);
        assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
        assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
        assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
        assertTrue(s.plan().has("v.patch_embd.weight"));
        assertTrue(s.plan().has("v.blk.0.attn_qkv.weight"));
        assertTrue(s.plan().has("mm.0.weight"));
    }

    @Test
    public void clefTowerUsesTheQwen3vlProjector() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("ClefModel"), "clef")) {
            assertQwenTower(s, "qwen3vl_merger");
        }
    }

    @Test
    public void h2oLightningTowerUsesTheQwen3vlProjector() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("H2OLightningModel"), "h2o")) {
            assertQwenTower(s, "qwen3vl_merger");
        }
    }

    @Test
    public void qwen4ExpTowerUsesTheQwen3vlProjector() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("Qwen4ExpForConditionalGeneration"), "qwen4exp")) {
            assertQwenTower(s, "qwen3vl_merger");
        }
    }

    @Test
    public void pplxDeciderReadsItsProcessorPixelBudget() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("PplxDeciderModel"), "pplx")) {
            assertQwenTower(s, "qwen3vl_merger");
            assertEquals(4096, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(16384, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
        }
    }

    @Test
    public void vllmD3WritesItsFixedPixelBudget() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("VllmD3Model"), "vllm-d3")) {
            assertQwenTower(s, "qwen3vl_merger");
            assertEquals(65_536, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1_638_400, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
        }
    }

    @Test
    public void miniCpmOcrSequenceClassifierReadsItsProcessorPixelBudget() throws IOException {
        try (HfModelSource s = open(qwenCheckpoint("MiniCPMOCRForSequenceClassification"), "ocr-seqcls")) {
            assertQwenTower(s, "qwen3vl_merger");
            assertEquals(4096, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(16384, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
        }
    }

    // ------------------------------------------------------------------ EXAONE 4.5

    private static SyntheticHf.Builder exaoneCheckpoint() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Exaone4_5_ForConditionalGeneration'],'model_type':'exaone4_5','hidden_size':" + DIM
                + ",'rms_norm_eps':1e-06,'vision_config':{'embed_dim':" + DIM + ",'hidden_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_key_value_heads':1,'num_hidden_layers':" + BLOCKS
                + ",'patch_size':2,'image_size':8,'fullatt_block_indexes':[1]}}"));
        f32(b, 1, "model.visual.patch_embed.proj.weight", DIM, 3, 2, 2, 2);
        f32(b, 2, "model.visual.blocks.0.attn.qkv.weight", 3 * DIM, DIM);
        f32(b, 3, "model.visual.merger.mlp.0.weight", FF, 4 * DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b;
    }

    @Test
    public void exaone45WritesItsFusedQkvAndWindowPeriod() throws IOException {
        try (HfModelSource s = open(exaoneCheckpoint(), "exaone")) {
            assertTower(s, "exaone4_5", 8, 2, BLOCKS);
            assertTrue("clip.use_silu", bool(s, GgufKeys.CLIP_USE_SILU));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(1, u32(s, GgufKeys.CLIP_VISION_ATTENTION_HEAD_COUNT_KV, -1));
            assertEquals("one-plus-the-first-fullatt index", 2,
                    u32(s, GgufKeys.CLIP_VISION_N_WA_PATTERN, -1));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertTrue(s.plan().has("v.blk.0.attn_qkv.weight"));
            assertTrue(s.plan().has("mm.0.weight"));
        }
    }

    // ------------------------------------------------------------------ Bailing-MoE (Ling)

    private static SyntheticHf.Builder bailingCheckpoint() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['BailingMoeV3VLForConditionalGeneration'],'text_config':{'hidden_size':" + DIM
                + ",'rms_norm_eps':1e-06},'vision_config':{'disable_merger_proj':true,'out_hidden_size':" + DIM
                + ",'depth':" + BLOCKS + ",'num_heads':" + HEADS + ",'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'patch_size':2,'spatial_merge_size':2,"
                + "'num_position_embeddings':16}}"));
        f32(b, 1, "linear_proj.0.weight", DIM, 4 * DIM);
        f32(b, 2, "linear_proj.2.weight", DIM, FF);
        f32(b, 3, "model.visual.merger.norm.weight", DIM);
        f32(b, 4, "model.visual.blocks.0.attn.qkv.weight", 3 * DIM, DIM);
        f32(b, 5, "model.visual.patch_embed.proj.weight", DIM, 3, 2, 2, 2);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b;
    }

    @Test
    public void bailingMoeRenamesItsTopLevelProjectorAndMergerNorm() throws IOException {
        try (HfModelSource s = open(bailingCheckpoint(), "bailing")) {
            assertTower(s, "ling3vl", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.2.weight"));
            assertTrue(s.plan().has("mm.input_norm.weight"));
            assertTrue(s.plan().has("v.blk.0.attn_qkv.weight"));
            assertTrue(s.plan().has("v.patch_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ CogVLM

    @Test
    public void cogVlmKeepsOnlyItsVisionTower() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['CogVLMForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-05}}"));
        f32(b, 1, "model.vision.patch_embedding.proj.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.vision.transformer.layers.0.attention.query_key_value.weight", 3 * DIM, DIM);
        f32(b, 3, "model.language_model.embed_tokens.weight", 16, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "cogvlm")) {
            assertTower(s, "cogvlm", 8, 2, BLOCKS);
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertFalse("the text embedding leaked", s.plan().has("token_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ Cohere2

    @Test
    public void cohere2ReadsItsDownsampleTileLimit() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Cohere2VisionForConditionalGeneration'],'hidden_size':" + DIM
                + ",'downsample_factor':2,'vision_config':{'hidden_size':" + DIM + ",'layer_norm_eps':1e-05"
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'image_size':8}}"));
        f32(b, 1, "model.vision_tower.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.multi_modal_projector.linear_1.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'max_patches':12}");
        try (HfModelSource s = open(b, "cohere2")) {
            assertTower(s, "cohere2v", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(12, u32(s, GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ DotsOCR

    @Test
    public void dotsOcrRenamesItsFfnToGateUpDown() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['DotsOCRForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'spatial_merge_size':2,'rms_norm_eps':1e-05}}"));
        f32(b, 1, "vision_tower.blocks.0.mlp.fc1.weight", FF, DIM);
        f32(b, 2, "vision_tower.blocks.0.mlp.fc2.weight", DIM, FF);
        f32(b, 3, "vision_tower.blocks.0.mlp.fc3.weight", FF, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        try (HfModelSource s = open(b, "dotsocr")) {
            assertTower(s, "dots_ocr", 0, 2, BLOCKS);
            assertTrue("clip.use_silu", bool(s, GgufKeys.CLIP_USE_SILU));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.blk.0.ffn_gate.weight"));
            assertTrue(s.plan().has("v.blk.0.ffn_down.weight"));
            assertTrue(s.plan().has("v.blk.0.ffn_up.weight"));
        }
    }

    // ------------------------------------------------------------------ PaddleOCR-VL

    @Test
    public void paddleOcrDerivesItsImageSizeFromThePixelLimit() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['PaddleOCRVisionModel'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'rms_norm_eps':1e-05}}"));
        f32(b, 1, "model.vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "vision_model.head.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        try (HfModelSource s = open(b, "paddle")) {
            assertTower(s, "paddleocr", 32, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ Janus-Pro

    @Test
    public void janusProDerivesItsFfnAndMapsItsAligner() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['JanusForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'mlp_ratio':2.0,'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS
                + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-06,'hidden_act':'gelu'}}"));
        f32(b, 1, "model.aligner.fc1.weight", DIM, DIM);
        f32(b, 2, "model.aligner.hidden_layers.0.weight", DIM, DIM);
        f32(b, 3, "model.vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 4, "model.generation_head.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "janus")) {
            assertTower(s, "janus_pro", 8, 2, BLOCKS);
            assertEquals("intermediate_size = hidden * mlp_ratio", FF,
                    u32(s, GgufKeys.CLIP_VISION_FEED_FORWARD_LENGTH, -1));
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.1.weight"));
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertFalse("the generation head leaked", s.plan().has("mm.2.weight"));
        }
    }

    // ------------------------------------------------------------------ Youtu-VL

    @Test
    public void youtuVlWritesItsWindowPatternAndDropsTheTextModel() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['YoutuVLForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS
                + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,'image_size':8,"
                + "'fullatt_block_indexes':[1],'hidden_act':'gelu'}}"));
        f32(b, 1, "vision_tower.patch_embed.proj.weight", DIM, 3, 2, 2);
        f32(b, 2, "vision_tower.blocks.0.attn.qkv.weight", 3 * DIM, DIM);
        f32(b, 3, "model.embed_tokens.weight", 16, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "youtu")) {
            assertTower(s, "youtuvl", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(List.of(1), list(s, GgufKeys.CLIP_VISION_WA_LAYER_INDEXES));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertFalse("the text embedding leaked", s.plan().has("token_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ Step 3-VL

    @Test
    public void step3VlMapsItsDownsamplersAndForcesItsTypes() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['StepVLForConditionalGeneration'],'hidden_size':" + DIM + ",'understand_projector_stride':2,"
                + "'vision_config':{'hidden_size':1536,'intermediate_size':8960,'num_attention_heads':12,"
                + "'num_hidden_layers':47,'patch_size':14,'image_size':728}}"));
        f32(b, 1, "vision_model.vit_downsampler1.weight", DIM, DIM);
        f32(b, 2, "vision_model.vit_downsampler2.weight", DIM, DIM);
        f32(b, 3, "vision_model.positional_embedding", 16, DIM);
        f32(b, 4, "vision_model.transformer.resblocks.0.ls_1.gamma", DIM);
        f32(b, 5, "vit_large_projector.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "step3")) {
            assertTower(s, "step3vl", 728, 14, 47, 1536, 8960, 12);
            assertEquals(4, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(3024, u32(s, GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.1.weight"));
            assertTrue(s.plan().has("mm.model.fc.weight"));
            assertTrue(s.plan().has("v.position_embd.weight"));
            assertTrue(s.plan().has("v.blk.0.ls1.weight"));
            assertEquals(GGMLType.F32, s.plan().type("v.position_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ Hunyuan-VL

    @Test
    public void hunyuanVlStripsTheClsRowOfItsPositionTable() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['HunYuanVLForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'image_size':8,"
                + "'patch_size':2,'hidden_size':" + DIM + ",'intermediate_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS
                + ",'rms_norm_eps':1e-05,'spatial_merge_size':2}}"));
        f32(b, 1, "vit.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "vit.embeddings.position_embedding.weight", 17, DIM);
        f32(b, 3, "vit.layers.0.self_attn.q_proj.weight", DIM, DIM);
        f32(b, 4, "vit.layers.0.self_attn.k_proj.weight", DIM, DIM);
        f32(b, 5, "vit.layers.0.self_attn.v_proj.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        try (HfModelSource s = open(b, "hunyuan")) {
            assertTower(s, "hunyuanvl", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(16, s.plan().ne("v.position_embd.weight")[s.plan().ne("v.position_embd.weight").length - 1]);
            assertTrue(s.plan().has("v.patch_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ SenseNova-U1

    @Test
    public void senseNovaU1MapsItsFourEmbeddingTensors() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['NEOVisionModel'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':1,'num_hidden_layers':0,'patch_size':2,"
                + "'image_size':8,'downsample_ratio':0.5,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "vision_model.embeddings.patch_embedding.bias", DIM);
        f32(b, 3, "vision_model.embeddings.dense_embedding.weight", DIM, DIM);
        f32(b, 4, "vision_model.embeddings.dense_embedding.bias", DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "sensenova")) {
            assertTower(s, "sensenova_u1", 8, 2, 0, DIM, FF, 1);
            assertEquals("head_dim = hidden", DIM, u32(s, GgufKeys.CLIP_VISION_ATTENTION_HEAD_DIM, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertTrue(s.plan().has("v.patch_embd.bias"));
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.0.bias"));
        }
    }

    // ------------------------------------------------------------------ LFM2-VL

    @Test
    public void lfm2ReshapesItsPatchEmbeddingAndWritesItsTiling() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Lfm2VlForConditionalGeneration'],'hidden_size':" + DIM + ",'downsample_factor':2,"
                + "'vision_feature_layer':-1,'vision_config':{'image_size':256,'patch_size':16,'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS
                + ",'num_hidden_layers':" + BLOCKS + ",'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_tower.vision_model.embeddings.patch_embedding.weight", DIM, 3 * 16 * 16);
        f32(b, 2, "model.multi_modal_projector.linear_1.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'resample':2,'do_image_splitting':true,"
                + "'min_tiles':2,'max_tiles':4,'tile_size':64,'max_pixels_tolerance':1.5}");
        try (HfModelSource s = open(b, "lfm2")) {
            assertTower(s, "lfm2", 256, 16, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals("bilinear", GgufKv.of(s).str(GgufKv.of(s).key(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO),
                    null));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, -1));
            assertEquals(4, u32(s, GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, -1));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_PREPROC_TILE_SIZE, -1));
            assertEquals(1.5f, f32(s, GgufKeys.CLIP_VISION_PREPROC_MAX_PIXELS_TOLERANCE, 0f), 1e-6f);
            assertArrayEquals(new long[]{16, 16, 3, DIM}, s.plan().ne("v.patch_embd.weight"));
        }
    }

    // ------------------------------------------------------------------ Kimi-K2.5 / GLM-5.2-Vision

    private static SyntheticHf.Builder kimiCheckpoint(String architecture) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':['" + architecture
                + "'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM + ",'intermediate_size':"
                + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':4,'patch_size':2,"
                + "'init_pos_emb_height':2,'init_pos_emb_width':2,'init_pos_emb_time':4,'merge_kernel_size':[2,2],"
                + "'projector_ln_eps':1e-05}}"));
        f32(b, 1, "vision_tower.encoder.blocks.0.wqkv.weight", 3 * HEADS * 8, DIM);
        f32(b, 3, "mm_projector.proj.linear_0.weight", DIM, DIM);
        f32(b, 4, "mm_projector.proj.linear_2.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'in_patch_limit':100}");
        return b;
    }

    @Test
    public void kimiK25WritesItsPositionGeometryAndProjector() throws IOException {
        try (HfModelSource s = open(kimiCheckpoint("KimiK25ForConditionalGeneration"), "kimi")) {
            assertTower(s, "kimik25", 4, 2, 4);
            assertEquals(2, s.getMetadata().get("vision.pos_emb_height"));
            assertEquals(2, s.getMetadata().get("vision.pos_emb_width"));
            assertEquals(4, s.getMetadata().get("vision.pos_emb_time"));
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(8L * 4L, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(100L * 4L, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.blk.0.attn_qkv.weight"));
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.2.weight"));
        }
    }

    @Test
    public void glm5vRenamesItsProjectorLayers() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Glm5vForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':"
                + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'init_pos_emb_height':2,'merge_kernel_size':[2,2]}}"));
        f32(b, 1, "vision_tower.encoder.blocks.0.wqkv.weight", 3 * HEADS * 8, DIM);
        f32(b, 2, "mm_projector.linear_0.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "glm5v")) {
            assertTower(s, "kimik25", 4, 2, BLOCKS);
            assertTrue(s.plan().has("mm.0.weight"));
        }
    }

    // ------------------------------------------------------------------ MiniMax-M3

    @Test
    public void miniMaxM3SplitsItsTemporalPatchEmbeddingAndPermutesQk() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MiniMaxM3SparseForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':"
                + "{'hidden_size':" + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS
                + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,'image_size':8,'layer_norm_eps':1e-05,"
                + "'img_token_compression_config':{'spatial_merge_size':2}}}"));
        f32(b, 1, "vision_tower.vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2, 2);
        f32(b, 2, "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight", 2 * 80, DIM);
        f32(b, 3, "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.weight", 2 * 80, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "minimax")) {
            assertTower(s, "minimax_m3", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertArrayEquals("one Conv2d kernel per temporal slice", new long[]{2, 2, 3, DIM},
                    s.plan().ne("v.patch_embd.weight"));
            assertTrue(s.plan().has("v.patch_embd.weight.1"));
            assertTrue(s.plan().has("v.blk.0.attn_q.weight"));
            assertTrue(s.plan().has("v.blk.0.attn_k.weight"));
        }
    }

    // ------------------------------------------------------------------ Muse-Glimmer / Muse-Spark

    private static SyntheticHf.Builder museCheckpoint(String architecture, boolean audio) {
        String audioConfig = audio ? ",'audio_config':{'hidden_sizes':[8,16],'mlp_ratio':2.0,'depths':[1,1],"
                + "'rms_norm_eps':1e-05,'conv_kernel_size':3,'patch_sizes':[2,2]}" : "";
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':['"
                + architecture + "'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'patch_size':2,'merge_size':2,'pos_emb_height':2,'patch_temporal':1,"
                + "'layer_norm_eps':1e-05}" + audioConfig + "}"));
        f32(b, 1, "model.vision_tower.patch_embedder.patch_embedding.weight", DIM, 3 * 2 * 2);
        f32(b, 2, "model.vision_tower.layers.0.attn.q_proj.weight", HEADS * 4, DIM);
        f32(b, 3, "model.vision_tower.layers.0.attn.k_proj.weight", HEADS * 4, DIM);
        f32(b, 4, "model.vision_adapter.fc1.weight", DIM, DIM);
        f32(b, 5, "model.vision_adapter.fc2.weight", DIM, DIM);
        f32(b, 6, "model.vision_projection.weight", DIM, DIM);
        if (audio) {
            f32(b, 7, "model.audio_tower.norm.weight", 16);
            f32(b, 8, "model.audio_tower.proj.weight", DIM, 16);
            f32(b, 9, "model.audio_tower.post_proj_norm.weight", DIM);
            f32(b, 10, "model.audio_tower.stages.0.patch_embedder.proj.weight", 16, 2);
        }
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b;
    }

    @Test
    public void museGlimmerSumsThePatchSlabsAndMapsItsAdapters() throws IOException {
        try (HfModelSource s = open(museCheckpoint("MuseGlimmerForConditionalGeneration", false), "muse")) {
            assertTower(s, "muse-glimmer", 8, 2, BLOCKS);
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(1e-5f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertArrayEquals(new long[]{2, 2, 3, DIM}, s.plan().ne("v.patch_embd.weight"));
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.1.weight"));
            assertTrue(s.plan().has("mm.2.weight"));
        }
    }

    @Test
    public void museSparkAddsItsAudioStages() throws IOException {
        try (HfModelSource s = open(museSparkCheckpoint(), "muse-spark")) {
            assertTower(s, "muse-glimmer", 8, 2, BLOCKS);
            assertEquals("muse-spark-audio", GgufKv.of(s).str(
                    GgufKv.of(s).key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(16, u32(s, GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, -1));
            assertEquals(32, u32(s, GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, -1));
            assertEquals(1, u32(s, GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_AUDIO_BLOCK_COUNT, -1));
            assertEquals(1, u32(s, GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, -1));
            assertTrue(s.plan().has("a.pre_ln.weight"));
            assertTrue(s.plan().has("a.post_ln.weight"));
            assertTrue(s.plan().has("mm.a.fc.weight"));
            assertTrue(s.plan().has("a.patch.0.weight"));
        }
    }

    private static SyntheticHf.Builder museSparkCheckpoint() {
        return museCheckpoint("MuseSparkForConditionalGeneration", true);
    }

    // ------------------------------------------------------------------ Phi-4

    @Test
    public void phi4TruncatesItsTowerAndReshapesItsPatchKernel() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Phi4ForCausalLMV'],'hidden_size':" + DIM + ",'vision_config':{'hidden_size':" + DIM
                + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':3,"
                + "'patch_size':4}}"));
        f32(b, 1, "model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight", DIM, 3 * 4 * 4);
        f32(b, 2, "model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight", 256, DIM);
        f32(b, 3, "model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight", FF, DIM);
        f32(b, 4, "model.vision_tower.vision_tower.vision_model.encoder.layers.0.mlp.fc2.weight", DIM, FF);
        f32(b, 5, "model.vision_tower.vision_tower.vision_model.encoder.layers.2.mlp.fc1.weight", FF, DIM);
        f32(b, 6, "model.mm_projector.0.weight", DIM, DIM);
        f32(b, 7, "model.mm_projector.2.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'patch_size':4,'min_num_patches':16,"
                + "'max_num_patches':64}");
        try (HfModelSource s = open(b, "phi4")) {
            assertTower(s, "phi4", 64, 4, 2);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(256, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertArrayEquals(new long[]{4, 4, 3, DIM}, s.plan().ne("v.patch_embd.weight"));
            assertTrue(s.plan().has("mm.0.weight"));
            assertTrue(s.plan().has("mm.2.weight"));
            assertFalse("the last block is dropped", s.plan().has("v.blk.2.mlp_fc1.weight"));
        }
    }

    // ------------------------------------------------------------------ Granite Vision 4.1

    @Test
    public void graniteWritesItsProjectorMapAndPinpoints() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Granite4VisionForConditionalGeneration'],'hidden_size':" + DIM + ",'downsample_rate':'2/2',"
                + "'deepstack_layer_map':[[-1,-1]],'spatial_target_layers':[],"
                + "'image_grid_pinpoints':[[8,8],[16,16]],'text_config':{'num_hidden_layers':2},"
                + "'vision_config':{'hidden_size':" + DIM + ",'intermediate_size':" + FF
                + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':" + BLOCKS + ",'patch_size':2,"
                + "'image_size':8,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_model.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.layerwise_projectors.0.query.weight", DIM, DIM);
        f32(b, 3, "model.vision_model.head.weight", DIM, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        try (HfModelSource s = open(b, "granite")) {
            assertTower(s, "granite4_vision", 8, 2, BLOCKS);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(8, u32(s, GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_QUERY_SIDE, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_WINDOW_SIDE, -1));
            assertEquals("the tapped vision layer", List.of(1), list(s, GgufKeys.CLIP_VISION_FEATURE_LAYER));
            assertEquals("no spatial projector", List.of(-1),
                    list(s, GgufKeys.CLIP_VISION_PROJECTOR_SPATIAL_OFFSETS));
            assertEquals("h/w pairs flattened as w,h", List.of(8, 8, 16, 16),
                    list(s, GgufKeys.CLIP_VISION_IMAGE_GRID_PINPOINTS));
            assertTrue(s.plan().has("v.patch_embd.weight"));
            assertTrue(s.plan().has("v.proj_blk.0.query.weight"));
            assertFalse("the vision head is dropped", s.plan().has("v.head.weight"));
        }
    }

    // ------------------------------------------------------------------ Nemotron Nano V2 VL

    private static SyntheticHf.Builder nemotronCheckpoint(boolean audio) {
        String sound = audio ? ",'sound_config':{'hidden_size':8,'intermediate_size':16,"
                + "'num_attention_heads':2,'num_hidden_layers':2,'num_mel_bins':4,'subsampling_factor':8,"
                + "'conv_kernel_size':3}" : "";
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['NemotronH_Nano_VL_V2'],'hidden_size':" + DIM + ",'force_image_size':64,"
                + "'downsample_ratio':0.5,'vision_config':{'patch_size':8}" + sound + "}"));
        f32(b, 1, "vision_model.radio_model.model.patch_generator.pos_embed.weight", 1, 16, DIM);
        f32(b, 2, "vision_model.radio_model.model.patch_generator.embedder.weight", DIM, 3 * 8 * 8);
        f32(b, 3, "vision_model.radio_model.model.blocks.0.norm1.weight", DIM);
        if (audio) {
            f32(b, 4, "sound_projection.linear0.weight", DIM, DIM);
            f32(b, 5, "sound_encoder.encoder.layers.0.self_attn.q_proj.weight", DIM, DIM);
        }
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        return b;
    }

    @Test
    public void nemotronResamplesItsPositionTableAndWritesItsFixedGeometry() throws IOException {
        try (HfModelSource s = open(nemotronCheckpoint(false), "nemotron")) {
            assertTower(s, "nemotron_v2_vl", 64, 8, 32, 1280, 5120, 16);
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertArrayEquals("the position grid is resampled to 8 x 8", new long[]{8, 64, 1},
                    s.plan().ne("v.position_embd.weight"));
            assertArrayEquals("the flat embedder becomes a kernel", new long[]{8, 8, 3, 8},
                    s.plan().ne("v.patch_embd.weight"));
            assertTrue(s.plan().has("v.blk.0.ln1.weight"));
        }
    }

    @Test
    public void nemotronAudioWritesItsParakeetStages() throws IOException {
        try (HfModelSource s = open(nemotronCheckpoint(true), "nemotron-audio")) {
            assertEquals("nemotron_v2_vl", GgufKv.of(s).str(
                    GgufKv.of(s).key(GgufKeys.CLIP_VISION_PROJECTOR_TYPE), null));
            assertEquals("parakeet", GgufKv.of(s).str(
                    GgufKv.of(s).key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(8, u32(s, GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, -1));
            assertEquals(16, u32(s, GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_AUDIO_BLOCK_COUNT, -1));
            assertEquals(4, u32(s, GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, -1));
            assertEquals(8, u32(s, GgufKeys.CLIP_AUDIO_SUBSAMPLING_FACTOR, -1));
            assertTrue(s.plan().has("mm.a.mlp.0.weight"));
            assertTrue(s.plan().has("a.blk.0.attn_q.weight"));
        }
    }

    // ------------------------------------------------------------------ MiniCPM-V 4.6 / 4.7 / OCR 1.0

    private static SyntheticHf.Builder minicpmCheckpoint(String architecture, String mode) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':['"
                + architecture + "'],'hidden_size':" + DIM + ",'vision_config':{'image_size':8,'patch_size':2,"
                + "'hidden_size':" + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS
                + ",'num_hidden_layers':8,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_tower.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.vision_tower.encoder.layers.0.mlp.fc1.weight", FF, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'downsample_mode':'" + mode + "'}");
        return b;
    }

    @Test
    public void miniCpmV46WritesItsInsertionLayerAndScale() throws IOException {
        try (HfModelSource s = open(minicpmCheckpoint("MiniCPMV4_6ForConditionalGeneration", "16x"), "minicpm46")) {
            assertTower(s, "minicpmv4_6", 8, 2, 8);
            assertTrue("clip.use_gelu", bool(s, GgufKeys.CLIP_USE_GELU));
            assertEquals(4, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(List.of(6), list(s, GgufKeys.CLIP_VISION_WA_LAYER_INDEXES));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.patch_embd.weight"));
        }
    }

    @Test
    public void miniCpmV46FourTimesDropsTheVitMerger() throws IOException {
        SyntheticHf.Builder b = minicpmCheckpoint("MiniCPMV4_6ForConditionalGeneration", "4x");
        f32(b, 3, "model.vision_tower.vit_merger.linear_1.weight", FF, DIM);
        try (HfModelSource s = open(b, "minicpm46-4x")) {
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertFalse("the ViT merger is unused at 4x", s.plan().has("v.merger.linear_1.weight"));
        }
    }

    @Test
    public void miniCpmV47DefaultsItsSliceResolutionAndGathersTheSeparators() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['MiniCPMV4_7ForConditionalGeneration'],'hidden_size':" + DIM + ",'vision_config':{'image_size':8,"
                + "'patch_size':2,'hidden_size':" + DIM + ",'intermediate_size':" + FF + ",'num_attention_heads':"
                + HEADS + ",'num_hidden_layers':8,'layer_norm_eps':1e-06}}"));
        f32(b, 1, "model.vision_tower.embeddings.patch_embedding.weight", DIM, 3, 2, 2);
        f32(b, 2, "model.language_model.embed_tokens.weight", 32, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + "}");
        b.tokenizerJson(SyntheticHf.json("{'added_tokens':["
                + "{'id':7,'content':'</image>','special':true,'normalized':false},"
                + "{'id':8,'content':'<slice>','special':true,'normalized':false},"
                + "{'id':9,'content':'</slice>','special':true,'normalized':false},"
                + "{'id':10,'content':'\\n','special':false,'normalized':false}],"
                + "'normalizer':null,'pre_tokenizer':{'type':'ByteLevel','add_prefix_space':false,"
                + "'trim_offsets':true,'use_regex':true},'post_processor':null,"
                + "'decoder':{'type':'ByteLevel','add_prefix_space':true,'trim_offsets':true,'use_regex':true},"
                + "'model':{'type':'BPE','unk_token':null,'byte_fallback':false,'ignore_merges':false,"
                + "'vocab':{'<s>':0,'</s>':1,'a':2},'merges':[]}}"));
        try (HfModelSource s = open(b, "minicpm47")) {
            assertTower(s, "minicpmv4_7", 448, 2, 8);
            assertEquals(4, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
            assertEquals(4, s.plan().ne("v.tok_embd_sep")[1]);
            assertFalse("the text embedding is consumed into the separators",
                    s.plan().has("model.language_model.embed_tokens.weight"));
        }
    }

    @Test
    public void miniCpmOcrUsesItsOwnProjectorType() throws IOException {
        try (HfModelSource s = open(minicpmCheckpoint("MiniCPMOcrForConditionalGeneration", "16x"), "minicpm-ocr")) {
            assertTower(s, "minicpm_ocr_1_0", 8, 2, 8);
            assertEquals(4, u32(s, GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR, -1));
        }
    }

    // ------------------------------------------------------------------ Dots3-Note

    private static SyntheticHf.Builder dots3Checkpoint() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['Dots3NoteForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'adapter_type':'patch_merger',"
                + "'pre_pixel_shuffle':true,'router_scoring_func':'sigmoid','router_scale':1.0,"
                + "'temporal_patch_size':1,'use_bias':false,'use_qk_norm':true,'rms_norm_eps':1e-06,"
                + "'capacity_factor':1.0,'pyramid_num_routed':[1,1],'embed_dim':" + DIM + ",'patch_size':2,"
                + "'intermediate_size':" + FF + ",'num_attention_heads':" + HEADS + ",'num_hidden_layers':"
                + BLOCKS + ",'spatial_merge_size':2},'audio_config':{'use_conv2d_stem':true,'use_rope':true,"
                + "'use_rms_norm':true,'use_causal':false,'merge_factor':1,'chunk_seconds':60,"
                + "'rope_parameters':{'partial_rotary_factor':0.5,'rope_theta':10000.0},"
                + "'whisper_config':{'activation_function':'swiglu','d_model':8,'encoder_ffn_dim':16,"
                + "'encoder_attention_heads':2,'encoder_layers':2,'num_mel_bins':4}}}"));
        f32(b, 1, "vision_encoder.blocks.0.mlp.gate_weight", 2, DIM);
        f32(b, 2, "vision_encoder.blocks.0.mlp.experts.0.fc1.weight", FF, DIM);
        f32(b, 3, "vision_encoder.blocks.0.mlp.experts.0.fc2.weight", DIM, FF);
        f32(b, 4, "vision_encoder.blocks.0.mlp.experts.0.fc3.weight", FF, DIM);
        f32(b, 5, "audio_encoder.dots_encoder.speech_encoder.layers.0.fc1.weight", 2 * FF, DIM);
        file(b, "preprocessor_config.json", "{" + COLORS + ",'min_pixels':64,'max_pixels':1024}");
        return b;
    }

    @Test
    public void dots3NoteStacksItsExpertsAndSplitsItsAudioFc1() throws IOException {
        try (HfModelSource s = open(dots3Checkpoint(), "dots3")) {
            assertTower(s, "dots3note_v", 0, 2, BLOCKS);
            assertTrue("clip.use_silu", bool(s, GgufKeys.CLIP_USE_SILU));
            assertEquals(64, u32(s, GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, -1));
            assertEquals(1024, u32(s, GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE, -1));
            assertEquals(List.of(1, 1), list(s, GgufKeys.CLIP_VISION_EXPERT_COUNT_PER_LAYER));
            assertEquals(1, u32(s, GgufKeys.CLIP_VISION_EXPERT_USED_COUNT, -1));
            assertEquals(1e-6f, f32(s, GgufKeys.CLIP_VISION_ATTENTION_LAYER_NORM_EPSILON, 0f), 1e-9f);
            assertTrue(s.plan().has("v.blk.0.ffn_gate_inp.weight"));
            assertTrue(s.plan().has("v.blk.0.ffn_gate_exps.weight"));
            assertTrue(s.plan().has("v.blk.0.ffn_down_exps.weight"));
            assertTrue(s.plan().has("v.blk.0.ffn_up_exps.weight"));
            assertEquals("dots3note_a", GgufKv.of(s).str(
                    GgufKv.of(s).key(GgufKeys.CLIP_AUDIO_PROJECTOR_TYPE), null));
            assertEquals(8, u32(s, GgufKeys.CLIP_AUDIO_EMBEDDING_LENGTH, -1));
            assertEquals(16, u32(s, GgufKeys.CLIP_AUDIO_FEED_FORWARD_LENGTH, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_AUDIO_ATTENTION_HEAD_COUNT, -1));
            assertEquals(2, u32(s, GgufKeys.CLIP_AUDIO_BLOCK_COUNT, -1));
            assertEquals(4, u32(s, GgufKeys.CLIP_AUDIO_NUM_MEL_BINS, -1));
            assertEquals("the fused fc1 splits into gate and up", 2,
                    s.plan().ne("a.blk.0.ffn_gate.weight").length);
            assertTrue(s.plan().has("a.blk.0.ffn_up.weight"));
        }
    }
}
