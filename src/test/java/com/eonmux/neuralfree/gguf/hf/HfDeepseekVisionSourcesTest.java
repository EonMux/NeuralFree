package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hf.tensor.PlannedTensor;
import com.eonmux.neuralfree.gguf.testutil.SyntheticHf;
import com.eonmux.neuralfree.io.safetensors.StDType;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * The DeepSeek-V4.1 vision projector export ({@code DeepseekV41VisionModel}, deepseek41.py:427-459): the
 * {@code deepseek41v} clip keys, the patchifier reshape, the fused SwiGLU split, the aligner and the image sentinels.
 */
public class HfDeepseekVisionSourcesTest {

    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int BLOCKS = 3;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theV41VisionTowerConvertsWithItsAlignerAndSentinels() throws IOException {
        Path dir = checkpoint().write(tmp.newFolder("vision").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals("clip", src.getString("general.architecture"));
            assertEquals("deepseek41v", src.getString("clip.projector_type"));
            assertEquals(BLOCKS, number(src, "clip.vision.block_count"));
            assertEquals(DIM, number(src, "clip.vision.embedding_length"));
            assertEquals(FF, number(src, "clip.vision.feed_forward_length"));
            assertEquals(2, number(src, "clip.vision.attention.head_count"));
            assertEquals(2, number(src, "clip.vision.patch_size"));
            assertEquals(64, number(src, "clip.vision.image_size"));
            assertEquals(2, number(src, "clip.vision.projector.scale_factor"));
            assertEquals(256, number(src, "clip.vision.image_min_pixels"));
            assertEquals(java.util.Arrays.asList(0.5f, 0.5f, 0.5f),
                    src.getMetadata().get("clip.vision.image_mean"));

            Set<String> t = names(src);
            for (String name : new String[]{"v.patch_embd.weight", "v.blk.0.ln1.weight", "v.blk.0.ln2.weight",
                    "v.blk.0.attn_qkv.weight", "v.blk.0.attn_out.weight", "v.blk.0.ffn_gate.weight",
                    "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight", "v.blk.2.attn_qkv.weight",
                    "mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias", "v.image_newline",
                    "v.token_embd.img_start", "v.token_embd.img_end"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("a text tensor leaked: " + name, name.startsWith("blk."));
            }

            PlannedTensor patch = ((HfModelSource) src).plan().tensor("v.patch_embd.weight");
            assertArrayEquals(new long[]{DIM, 3, 2, 2}, patch.source().shape());
            PlannedTensor gate = ((HfModelSource) src).plan().tensor("v.blk.0.ffn_gate.weight");
            PlannedTensor up = ((HfModelSource) src).plan().tensor("v.blk.0.ffn_up.weight");
            assertEquals(FF / 2, gate.source().shape()[0]);
            assertEquals(FF / 2, up.source().shape()[0]);
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("v.patch_embd.weight"));
        }
    }

    private static SyntheticHf.Builder checkpoint() {
        SyntheticHf.Builder b = SyntheticHf.builder().config(SyntheticHf.json("{'architectures':"
                + " ['DeepseekV41ForCausalLM'], 'model_type': 'deepseek_v41', 'hidden_size': " + DIM + ","
                + " 'vision_config': {'num_hidden_layers': " + BLOCKS + ", 'hidden_size': " + DIM + ","
                + "   'num_attention_heads': 2, 'intermediate_size': " + FF + ", 'patch_size': 2,"
                + "   'downsample_ratio': 2, 'min_pixels': 256, 'max_image_tokens': 1024,"
                + "   'max_wh_ratio': null, 'rope_theta': 10000.0}}"));
        f32(b, 10, "vision.patch_embed.proj.weight", DIM, 3 * 2 * 2);
        int seed = 20;
        for (int l = 0; l < BLOCKS; l++) {
            String p = "vision.blocks." + l + ".";
            f32(b, seed++, p + "norm1.weight", DIM);
            f32(b, seed++, p + "norm2.weight", DIM);
            f32(b, seed++, p + "attn.wqkv.weight", 3 * DIM, DIM);
            f32(b, seed++, p + "attn.wo.weight", DIM, DIM);
            f32(b, seed++, p + "mlp.w1.weight", FF, DIM);
            f32(b, seed++, p + "mlp.w2.weight", DIM, FF / 2);
        }
        f32(b, 40, "aligner.w1.weight", FF, DIM);
        f32(b, 41, "aligner.w1.bias", FF);
        f32(b, 42, "aligner.w2.weight", DIM, FF);
        f32(b, 43, "aligner.w2.bias", DIM);
        f32(b, 44, "image_newline", DIM);
        f32(b, 45, "image_start", DIM);
        f32(b, 46, "image_end", DIM);
        f32(b, 50, "model.layers.0.self_attn.q_proj.weight", DIM, DIM);
        return b;
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
