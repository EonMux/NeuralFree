package com.eonmux.neuralfree.gguf.hf;

import com.eonmux.neuralfree.gguf.GGUFTensorInfo;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.ModelSource;
import com.eonmux.neuralfree.gguf.ModelSourceOptions;
import com.eonmux.neuralfree.gguf.ModelSources;
import com.eonmux.neuralfree.gguf.hf.HfModelSource;
import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The LFM3-VL projector export ({@code LFM3VLMmprojModel}, lfm3.py:271-311, over the shared
 * {@code LFM3VLVisionMixin}, lfm3.py:179-266): the {@code lfm3vl} clip keys, the two-frame temporal patchifier, the
 * conv and attention block tensors and the LayerNorm-GELU-linear projector.
 */
public class HfLfm3VisionSourcesTest {

    private static final int DIM = 8;
    private static final int FF = 16;
    private static final int BLOCKS = 2;

    @Rule
    public final TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void theLfm3VlProjectorConvertsItsTowerAndProjector() throws IOException {
        Path dir = checkpoint().write(tmp.newFolder("lfm3vl").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals("clip", src.getString("general.architecture"));
            assertEquals("lfm3vl", src.getString("clip.projector_type"));
            assertEquals(BLOCKS, number(src, "clip.vision.block_count"));
            assertEquals(DIM, number(src, "clip.vision.embedding_length"));
            assertEquals(FF, number(src, "clip.vision.feed_forward_length"));
            assertEquals(2, number(src, "clip.vision.attention.head_count"));
            assertEquals(2, number(src, "clip.vision.attention.head_count_kv"));
            assertEquals(512, number(src, "clip.vision.image_size"));
            assertEquals(2, number(src, "clip.vision.patch_size"));
            assertEquals(2, number(src, "clip.vision.spatial_merge_size"));
            assertEquals(256, number(src, "clip.vision.image_min_pixels"));
            assertEquals(4096, number(src, "clip.vision.image_max_pixels"));
            assertEquals("bilinear", src.getString("clip.vision.image_resize_algo"));
            assertEquals(Boolean.TRUE, src.getMetadata().get("clip.use_silu"));

            Set<String> t = names(src);
            for (String name : new String[]{"v.patch_embd.weight", "v.patch_embd.weight.1",
                    "v.embedding_norm.weight", "v.blk.0.ln1.weight", "v.blk.0.ln2.weight",
                    "v.blk.0.conv2d_in_proj.weight", "v.blk.0.conv2d_spatial.weight",
                    "v.blk.0.conv2d_out_proj.weight", "v.blk.0.ffn_up.weight", "v.blk.1.attn_q.weight",
                    "v.blk.1.attn_k.weight", "v.blk.1.attn_v.weight", "v.blk.1.attn_out.weight",
                    "mm.input_norm.weight", "mm.input_norm.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight",
                    "mm.2.bias"}) {
                assertTrue(name + " is missing from " + t, t.contains(name));
            }
            for (String name : t) {
                assertFalse("a text tensor leaked: " + name, name.startsWith("blk."));
            }

            assertEquals(GGMLType.F16, ((HfModelSource) src).plan().type("v.patch_embd.weight"));
            assertEquals(GGMLType.F16, ((HfModelSource) src).plan().type("v.patch_embd.weight.1"));
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("v.blk.0.conv2d_spatial.weight"));
        }
    }

    @Test
    public void theOmniProjectorRunsTheVisionAndAudioQuantizationRules() throws IOException {
        SyntheticHf.Builder b = checkpoint()
                .config(SyntheticHf.json(omniConfig("'projector_hidden_act': 'gelu'")));
        b.file("preprocessor_config.json", SyntheticHf.json(omniPreprocessor()));
        Path dir = b.write(tmp.newFolder("lfm3omni").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-omni").toPath()).build();
        try (ModelSource src = ModelSources.openProjector(dir, options)) {
            assertEquals("lfm3vl", src.getString("clip.vision.projector_type"));
            assertEquals("lfm3a", src.getString("clip.audio.projector_type"));
            // the Omni force map runs the vision mixin's patch rules before the audio tower's (lfm3.py:257-264)
            assertEquals(GGMLType.F16, ((HfModelSource) src).plan().type("v.patch_embd.weight"));
            assertEquals(GGMLType.F16, ((HfModelSource) src).plan().type("v.patch_embd.weight.1"));
            assertEquals(GGMLType.F32, ((HfModelSource) src).plan().type("v.blk.0.conv2d_spatial.weight"));
        }
    }

    @Test
    public void aProjectorWithAnotherActivationIsRefusedByName() throws IOException {
        SyntheticHf.Builder b = checkpoint()
                .config(SyntheticHf.json(config("'projector_hidden_act': 'silu'")));
        Path dir = b.write(tmp.newFolder("bad").toPath());
        ModelSourceOptions options = ModelSourceOptions.builder()
                .cacheDir(tmp.newFolder("cache-bad").toPath()).build();
        GgufFormatException e = assertThrows(GgufFormatException.class,
                () -> ModelSources.openProjector(dir, options));
        assertTrue(e.getMessage(), e.getMessage().contains("projector_hidden_act"));
    }

    private static String config(String projectorActivation) {
        return "{'architectures': ['Lfm3VlForConditionalGeneration'], 'model_type': 'lfm3_vl',"
                + " 'hidden_size': " + DIM + ", " + projectorActivation + ", 'projector_bias': true,"
                + " 'projector_layer_norm_eps': 1e-6, 'downsample_factor': 2,"
                + " 'vision_config': {'image_size': 8, 'patch_size': 2, 'temporal_patch_size': 2,"
                + "   'in_channels': 3, 'conv_kernel_size': 3, 'num_key_value_heads': 2,"
                + "   'num_attention_heads': 2, 'num_hidden_layers': " + BLOCKS + ", 'hidden_size': " + DIM + ","
                + "   'intermediate_size': " + FF + ", 'norm_eps': 1e-6,"
                + "   'rope_parameters': {'rope_theta': 1000000.0}}}";
    }

    private static String omniConfig(String projectorActivation) {
        return "{'architectures': ['Lfm3OmniForConditionalGeneration'], 'model_type': 'lfm3_omni',"
                + " 'hidden_size': " + DIM + ", " + projectorActivation + ", 'projector_bias': true,"
                + " 'projector_layer_norm_eps': 1e-6, 'downsample_factor': 2,"
                + " 'vision_config': {'image_size': 8, 'patch_size': 2, 'temporal_patch_size': 2,"
                + "   'in_channels': 3, 'conv_kernel_size': 3, 'num_key_value_heads': 2,"
                + "   'num_attention_heads': 2, 'num_hidden_layers': " + BLOCKS + ", 'hidden_size': " + DIM + ","
                + "   'intermediate_size': " + FF + ", 'norm_eps': 1e-6,"
                + "   'rope_parameters': {'rope_theta': 1000000.0}},"
                + " 'audio_encoder_config': {'num_mel_bins': 4, 'sliding_window': 8,"
                + "   'text_config': {'hidden_size': 8, 'intermediate_size': 16, 'num_attention_heads': 2,"
                + "     'num_key_value_heads': 1, 'num_hidden_layers': 2, 'norm_eps': 1e-6,"
                + "     'block_auto_adjust_ff_dim': false, 'block_ffn_dim_multiplier': null,"
                + "     'block_multiple_of': 1, 'rope_parameters': {'rope_theta': 1e6}}}}";
    }

    private static String omniPreprocessor() {
        return "{'image_mean': [0.5, 0.5, 0.5], 'image_std': [0.5, 0.5, 0.5],"
                + " 'size': {'shortest_edge': 256, 'longest_edge': 4096}, 'resample': 2,"
                + " 'feature_extractor': {'feature_extractor_type': 'Lfm3OmniFeatureExtractor',"
                + "   'sampling_rate': 16000, 'feature_size': 128, 'n_fft': 400, 'win_length': 400,"
                + "   'hop_length': 160, 'log_floor': -12.0, 'normalization_offset': 4.0,"
                + "   'normalization_divisor': 4.0, 'encoder_stride': 8}}";
    }

    private static SyntheticHf.Builder checkpoint() {        SyntheticHf.Builder b = SyntheticHf.builder()
                .config(SyntheticHf.json(config("'projector_hidden_act': 'gelu'")));
        b.file("preprocessor_config.json", SyntheticHf.json("{'image_mean': [0.5, 0.5, 0.5],"
                + " 'image_std': [0.5, 0.5, 0.5], 'size': {'shortest_edge': 256, 'longest_edge': 4096},"
                + " 'resample': 2}"));
        f32(b, 10, "model.vision_tower.patchifier.weight", DIM, 3, 2, 2, 2);
        f32(b, 11, "model.vision_tower.embedding_norm.weight", DIM);
        int seed = 20;
        for (int l = 0; l < BLOCKS; l++) {
            String p = "model.vision_tower.layers." + l + ".";
            f32(b, seed++, p + "operator_norm.weight", DIM);
            f32(b, seed++, p + "ffn_norm.weight", DIM);
            f32(b, seed++, p + "feed_forward.w1.weight", FF, DIM);
            f32(b, seed++, p + "feed_forward.w3.weight", FF, DIM);
            f32(b, seed++, p + "feed_forward.w2.weight", DIM, FF);
            if (l == 0) {
                f32(b, seed++, p + "conv.in_proj.weight", 3 * DIM, DIM);
                f32(b, seed++, p + "conv.spatial_conv.weight", DIM, 1, 3, 3);
                f32(b, seed++, p + "conv.out_proj.weight", DIM, DIM);
            } else {
                f32(b, seed++, p + "self_attn.q_proj.weight", DIM, DIM);
                f32(b, seed++, p + "self_attn.k_proj.weight", DIM, DIM);
                f32(b, seed++, p + "self_attn.v_proj.weight", DIM, DIM);
                f32(b, seed++, p + "self_attn.out_proj.weight", DIM, DIM);
            }
        }
        f32(b, 40, "model.multi_modal_projector.layer_norm.weight", DIM);
        f32(b, 41, "model.multi_modal_projector.layer_norm.bias", DIM);
        f32(b, 42, "model.multi_modal_projector.linear_1.weight", FF, DIM);
        f32(b, 43, "model.multi_modal_projector.linear_1.bias", FF);
        f32(b, 44, "model.multi_modal_projector.linear_2.weight", DIM, FF);
        f32(b, 45, "model.multi_modal_projector.linear_2.bias", DIM);
        f32(b, 50, "model.layers.0.mlp.up_proj.weight", FF, DIM);
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
