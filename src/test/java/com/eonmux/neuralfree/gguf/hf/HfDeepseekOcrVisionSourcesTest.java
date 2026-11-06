package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek-OCR projector export ({@code DeepseekOCRVisionModel}, deepseek.py:22-102): the merged vision
 * configuration, the SAM geometry keys, the SAM and CLIP tower tensors, the single-layer projector and the image
 * sentinels.
 */
public class HfDeepseekOcrVisionSourcesTest {

    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int BLOCKS = 4;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theDeepseekOcrProjectorConvertsItsSamClipTowerAndProjector() throws IOException {
        Path dir = checkpoint("DeepseekOCRForCausalLM", 9).write(tmp.newFolder("ocr").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals("clip", src.getString("general.architecture"));
            assertEquals("deepseekocr", src.getString("clip.projector_type"));
            assertEquals(BLOCKS, number(src, "clip.vision.block_count"));
            assertEquals(DIM, number(src, "clip.vision.embedding_length"));
            assertEquals(FF, number(src, "clip.vision.feed_forward_length"));
            assertEquals(2, number(src, "clip.vision.attention.head_count"));
            assertEquals(32, number(src, "clip.vision.image_size"));
            assertEquals(4, number(src, "clip.vision.patch_size"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("clip.use_gelu"));
            assertEquals(14, number(src, "clip.vision.window_size"));
            assertEquals(2, number(src, "clip.vision.preproc_min_tiles"));
            assertEquals(9, number(src, "clip.vision.preproc_max_tiles"));
            assertEquals(BLOCKS, number(src, "clip.vision.sam.block_count"));
            assertEquals(DIM, number(src, "clip.vision.sam.embedding_length"));
            assertEquals(2, number(src, "clip.vision.sam.head_count"));

            Set<String> t = names(src);
            for (String name : new String[]{"v.sam.patch_embd.weight", "v.sam.pos_embd.weight",
                    "v.sam.blk.0.attn.qkv.weight", "v.sam.blk.0.attn.out.weight", "v.sam.blk.0.attn.pos_h.weight",
                    "v.sam.blk.0.mlp.lin1.weight", "v.sam.neck.0.weight", "v.sam.neck.3.weight",
                    "v.sam.net_2.weight", "v.sam.net_3.weight", "v.class_embd", "v.position_embd",
                    "v.blk.0.attn_q.weight", "v.blk.1.attn_q.weight", "mm.model.fc.weight", "mm.model.fc.bias",
                    "v.image_newline", "v.view_seperator"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("a text tensor leaked: " + name, name.startsWith("blk."));
            }
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("v.sam.pos_embd.weight"));
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("v.sam.blk.0.attn.pos_h.weight"));
        }
    }

    @Test
    public void unlimitedOcrRaisesTheTileLimit() throws IOException {
        Path dir = checkpoint("UnlimitedOCRForCausalLM", 32).write(tmp.newFolder("unlimited").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-unlimited").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals(32, number(src, "clip.vision.preproc_max_tiles"));
        }
    }

    private static SyntheticHf.Builder checkpoint(String arch, int maxTiles) {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures': ['" + arch
                + "'], 'hidden_size': " + DIM + ", 'vision_config': {'image_size': 32, 'patch_size': 4,"
                + " 'n_layers': " + BLOCKS + ", 'width': {'sam_vit_b': {'layers': " + BLOCKS + ", 'width': " + DIM
                + ", 'heads': 2}, 'clip-l-14-224': {'hidden_size': " + DIM + ", 'num_heads': 2,"
                + " 'intermediate_size': " + FF + "}}}}"));
        b.file("preprocessor_config.json", SyntheticHf.json("{'image_mean': [0.5, 0.5, 0.5],"
                + " 'image_std': [0.5, 0.5, 0.5], 'image_seq_length': 256}"));
        f32(b, 10, "model.sam_model.patch_embed.proj.weight", DIM, 3, 4, 4);
        f32(b, 11, "model.sam_model.pos_embed", 16, DIM);
        int seed = 20;
        for (int l = 0; l < BLOCKS; l++) {
            String p = "model.sam_model.blocks." + l + ".";
            f32(b, seed++, p + "norm1.weight", DIM);
            f32(b, seed++, p + "norm2.weight", DIM);
            f32(b, seed++, p + "attn.qkv.weight", 3 * DIM, DIM);
            f32(b, seed++, p + "attn.proj.weight", DIM, DIM);
            f32(b, seed++, p + "attn.rel_pos_h", DIM, DIM);
            f32(b, seed++, p + "attn.rel_pos_w", DIM, DIM);
            f32(b, seed++, p + "mlp.lin1.weight", FF, DIM);
            f32(b, seed++, p + "mlp.lin2.weight", DIM, FF);
            String c = "model.vision_model.encoder.layers." + l + ".";
            f32(b, seed++, c + "self_attn.q_proj.weight", DIM, DIM);
            f32(b, seed++, c + "self_attn.k_proj.weight", DIM, DIM);
            f32(b, seed++, c + "self_attn.v_proj.weight", DIM, DIM);
            f32(b, seed++, c + "self_attn.out_proj.weight", DIM, DIM);
        }
        for (int n = 0; n < 4; n++) {
            f32(b, 60 + n, "model.sam_model.neck." + n + ".weight", DIM, DIM);
        }
        f32(b, 70, "model.sam_model.net_2.weight", DIM, DIM);
        f32(b, 71, "model.sam_model.net_3.weight", DIM, DIM);
        f32(b, 72, "model.vision_model.embeddings.class_embedding", DIM);
        f32(b, 73, "model.vision_model.embeddings.position_embedding", 16, DIM);
        f32(b, 74, "model.projector.layers.weight", DIM, DIM);
        f32(b, 75, "model.projector.layers.bias", DIM);
        f32(b, 76, "image_newline", DIM);
        f32(b, 77, "model.view_seperator", DIM);
        f32(b, 80, "model.layers.0.mlp.up_proj.weight", FF, DIM);
        return b;
    }


    @Test
    public void deepseekOcr2WritesItsQwen2DefaultsAndKvHeads() throws IOException {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + "['DeepseekOCR2ForCausalLM'],'hidden_size':" + DIM + ",'vision_config':{'image_size':32,"
                + "'width':{'sam_vit_b':{'layers':" + BLOCKS + ",'width':" + DIM + ",'heads':2},"
                + "'qwen2-0_5b':{'dim':" + DIM + "}}}}"));
        b.file("preprocessor_config.json", SyntheticHf.json("{'image_mean':[0.5,0.5,0.5],"
                + "'image_std':[0.5,0.5,0.5],'image_seq_length':256}"));
        f32(b, 10, "model.sam_model.patch_embed.proj.weight", DIM, 3, 4, 4);
        f32(b, 11, "model.sam_model.pos_embed", 16, DIM);
        f32(b, 12, "model.projector.layers.weight", DIM, DIM);
        f32(b, 13, "model.projector.layers.bias", DIM);
        f32(b, 14, "image_newline", DIM);
        f32(b, 15, "model.view_seperator", DIM);
        Path dir = b.write(tmp.newFolder("ocr2").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-ocr2").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals("deepseekocr2", src.getString("clip.projector_type"));
            assertEquals("the Qwen2 encoder's default patch", 16,
                    number(src, "clip.vision.patch_size"));
            assertEquals("the Qwen2 encoder's default FFN", 4864,
                    number(src, "clip.vision.feed_forward_length"));
            assertEquals("the Qwen2 encoder's default heads", 14,
                    number(src, "clip.vision.attention.head_count"));
            assertEquals("the Qwen2 encoder's default blocks", 24,
                    number(src, "clip.vision.block_count"));
            assertEquals("the Qwen2 encoder's GQA", 2,
                    number(src, "clip.vision.attention.head_count_kv"));
            assertEquals(DIM, number(src, "clip.vision.embedding_length"));
            assertEquals("OCR2's tile limit", 6, number(src, "clip.vision.preproc_max_tiles"));
            assertEquals(BLOCKS, number(src, "clip.vision.sam.block_count"));
        }
    }

    private static void f32(SyntheticHf.Builder b, int seed, String name, int... shape) {
        long[] dims = new long[shape.length];
        int count = 1;
        for (int i = 0; i < shape.length; i++) {
            dims[i] = shape[i];
            count *= shape[i];
        }
        b.tensor(name, StDType.F32, dims, SyntheticHf.f32(SyntheticHf.values(seed, count)));
    }

    private static Set<String> names(ModelSource src) {
        Set<String> out = new HashSet<>();
        for (GGUFTensorInfo t : src.getTensors()) {
            out.add(t.name());
        }
        return out;
    }

    private static long number(ModelSource src, String key) {
        Object v = src.getMetadata().get(key);
        assertTrue(key + " is missing", v instanceof Number);
        return ((Number) v).longValue();
    }
}
