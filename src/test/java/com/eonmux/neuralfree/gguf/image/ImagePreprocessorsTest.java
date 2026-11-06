package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The preprocessor selection of llama.cpp's {@code mtmd_context::init_vision}: the projector family, the grid
 * pinpoints and the longest edge decide the pipeline.
 */
public class ImagePreprocessorsTest {

    private static Map<String, Object> metadata(String projector, Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", projector);
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams params(String projector, Object... extra) {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(projector, extra)), ImageTokenBudget.none());
    }

    @Test
    public void fixedSizeFamilies() {
        for (String p : new String[] {"mlp", "ldp", "ldpv2", "adapter", "gemma3", "gemma3nv", "yasa2",
                "nemotron_v2_vl"}) {
            assertEquals(p, true, ImagePreprocessors.forParams(params(p)) instanceof FixedSizePreprocessor);
        }
    }

    @Test
    public void pinpointsSwitchTheLlavaFamilyToSlicing() {
        ImageProcessorParams p = params("mlp",
                "clip.vision.image_grid_pinpoints", new int[] {336, 336, 672, 672});
        assertEquals(true, ImagePreprocessors.forParams(p) instanceof LlavaUhdPreprocessor);
        assertEquals(false, ImagePreprocessors.forParams(p) instanceof Cohere2vPreprocessor);
    }

    @Test
    public void llavaUhdFamilies() {
        for (String p : new String[] {"resampler", "minicpmv4_6", "minicpmv4_7", "minicpm_ocr_1_0"}) {
            assertEquals(p, true, ImagePreprocessors.forParams(params(p)) instanceof MiniCpmvPreprocessor);
        }
        for (String p : new String[] {"llama4", "internvl"}) {
            assertEquals(p, true, ImagePreprocessors.forParams(params(p)) instanceof LlavaUhdPreprocessor);
        }
        assertEquals(true, ImagePreprocessors.forParams(params("cohere2v",
                "clip.vision.preproc_max_tiles", 4, "clip.vision.projector.scale_factor", 2))
                instanceof Cohere2vPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(params("idefics3",
                "clip.vision.preproc_image_size", 1344)) instanceof Idefics3Preprocessor);
        assertEquals(true, ImagePreprocessors.forParams(params("idefics3")) instanceof Idefics3Preprocessor);
    }

    @Test
    public void dynamicSizeFamilies() {
        for (String p : new String[] {"qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger", "ling3vl",
                "mimovl", "phi4", "kimivl", "kimik25", "glm4v", "hunyuanvl", "exaone4_5", "paddleocr",
                "dots_ocr", "dots3note_v", "nemotron_omni", "gemma4v", "gemma4uv", "minimax_m3"}) {
            assertEquals(p, true, ImagePreprocessors.forParams(converted(p)) instanceof DynSizePreprocessor);
        }
    }

    @Test
    public void theTiledAndSearchedFamiliesHaveTheirOwnPreprocessor() {
        assertEquals(true, ImagePreprocessors.forParams(converted("lfm2")) instanceof Lfm2Preprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("d1omni_v")) instanceof Lfm2Preprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("lfm3vl")) instanceof Lfm3vlPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("step3vl")) instanceof Step3vlPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("glm5v")) instanceof Glm5vPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("youtuvl")) instanceof YoutuVlPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("muse-glimmer"))
                instanceof MuseGlimmerPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("deepseekocr"))
                instanceof DeepseekOcrPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("deepseekocr2"))
                instanceof DeepseekOcrPreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(converted("lightonocr"))
                instanceof LongestEdgePreprocessor);
        for (String p : new String[] {"adapter", "janus_pro", "cogvlm"}) {
            assertEquals(p, true, ImagePreprocessors.forParams(converted(p)) instanceof FixedSizePreprocessor);
        }
    }

    @Test
    public void pixtralUsesTheLongestEdgeOnlyWithAPreprocImageSize() {
        assertEquals(true, ImagePreprocessors.forParams(params("pixtral")) instanceof DynSizePreprocessor);
        assertEquals(true, ImagePreprocessors.forParams(params("pixtral",
                "clip.vision.preproc_image_size", 1024)) instanceof LongestEdgePreprocessor);
    }

    @Test
    public void everySupportedFamilyPreparesATinyImage() {
        for (ProjectorType projector : ProjectorType.preprocessed()) {
            PreparedImages out = ImagePreprocessors.forParams(converted(projector.typeName()))
                    .prepare(RasterImage.filled(64, 64, 9, 8, 7));
            assertEquals(projector.typeName(), true, out.totalTokens() > 0);
        }
    }

    /**
     * The params of a file of the projector, with the keys its converter always writes (pixel limits, merge factor,
     * resize algorithm, Cohere2-Vision tile cap, Granite Vision pinpoints and QFormer sides) supplied when the bare
     * metadata does not load.
     */
    private static ImageProcessorParams converted(String projector) {
        Map<String, Object> m = metadata(projector);
        if (projector.equals("cohere2v")) {
            m.put("clip.vision.preproc_max_tiles", 4);
        }
        if (projector.equals("granite4_vision")) {
            m.put("clip.vision.image_grid_pinpoints", new int[] {336, 336, 672, 336});
            m.put("clip.vision.projector.window_side", 8);
            m.put("clip.vision.projector.query_side", 4);
        }
        try {
            return ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        } catch (RuntimeException first) {
            m.put("clip.vision.image_min_pixels", 1024);
            m.put("clip.vision.image_max_pixels", 4096 * 784);
            m.put("clip.vision.projector.scale_factor", 2);
            m.put("clip.vision.spatial_merge_size", 2);
            m.put("clip.vision.image_resize_algo", "bilinear");
            try {
                return ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
            } catch (RuntimeException second) {
                throw new AssertionError(projector + ": " + second.getMessage(), second);
            }
        }
    }
}
