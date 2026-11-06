package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The {@code load_hparams} defaults of the families whose preprocessing was ported from llama.cpp tools/mtmd
 * (clip.cpp): resize algorithms, pads, tile ranges, token limits as pixel limits, and the refusals that name the
 * offending metadata key.
 */
public class FamilyDefaultsTest {

    private static Map<String, Object> metadata(String projector, int patch, int imageSize, Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", projector);
        m.put("clip.vision.patch_size", patch);
        m.put("clip.vision.image_size", imageSize);
        m.put("clip.vision.image_mean", new float[] {0.5f, 0.5f, 0.5f});
        m.put("clip.vision.image_std", new float[] {0.5f, 0.5f, 0.5f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams params(Map<String, Object> m) {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
    }

    private static String refusal(Map<String, Object> m) {
        return assertThrows(IllegalArgumentException.class, () -> params(m)).getMessage();
    }

    @Test
    public void lfm2ResizesLikeTorchvisionAndTilesWithTheProcessorDefaults() {
        ImageProcessorParams p = params(metadata("lfm2", 16, 512, "clip.vision.projector.scale_factor", 2));
        assertEquals(ResizeAlgo.TORCH_BILINEAR, p.resizeAlgo());
        assertEquals(ResizeAlgo.TORCH_BILINEAR, p.overviewResizeAlgo());
        assertEquals(ResizeAlgo.TORCH_BILINEAR, p.refinedResizeAlgo());
        assertEquals(PadStyle.NONE, p.refinedPadStyle());
        assertEquals(2, p.minTiles());
        assertEquals(10, p.maxTiles());
        assertEquals(512, p.tileSize());
        assertEquals(2.0f, p.maxPixelsTolerance(), 0.0f);
        // set_limit_image_tokens(64, 256) over the 32x32 merged patch
        assertEquals(64 * 1024, p.minPixels());
        assertEquals(256 * 1024, p.maxPixels());
        assertFalse(p.overviewFirst());

        assertEquals(ResizeAlgo.TORCH_BICUBIC, params(metadata("lfm2", 16, 512,
                "clip.vision.image_resize_algo", "bicubic")).resizeAlgo());
        assertEquals(ResizeAlgo.LANCZOS, params(metadata("d1omni_v", 16, 512,
                "clip.vision.image_resize_algo", "lanczos")).resizeAlgo());
        assertTrue(refusal(metadata("lfm2", 16, 512, "clip.vision.image_resize_algo", "nearest"))
                .contains("'clip.vision.image_resize_algo'"));
        assertTrue(refusal(metadata("lfm2", 16, 512, "clip.vision.preproc_min_tiles", 4,
                "clip.vision.preproc_max_tiles", 3)).contains("'clip.vision.preproc_min_tiles'"));
    }

    @Test
    public void aTokenBudgetReplacesTheLfm2TokenLimits() {
        ImageProcessorParams p = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata("lfm2", 16, 512,
                "clip.vision.projector.scale_factor", 2)), ImageTokenBudget.builder().maxTokens(128).build());
        assertEquals(64 * 1024, p.minPixels());
        assertEquals(128 * 1024, p.maxPixels());
    }

    @Test
    public void lfm3vlRequiresItsAlgorithmMergeAndPixelLimits() {
        ImageProcessorParams p = params(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                "clip.vision.image_resize_algo", "bicubic", "clip.vision.image_min_pixels", 65536,
                "clip.vision.image_max_pixels", 8 * 1024 * 1024));
        assertEquals(ResizeAlgo.TORCH_BICUBIC, p.resizeAlgo());
        assertEquals(PadStyle.NONE, p.padStyle());
        assertEquals(64 * 1024, p.minPixels());
        // 8192 tokens capped at 4096
        assertEquals(4096 * 1024, p.maxPixels());

        assertTrue(refusal(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                "clip.vision.image_min_pixels", 65536, "clip.vision.image_max_pixels", 1048576))
                .contains("'clip.vision.image_resize_algo'"));
        assertTrue(refusal(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                "clip.vision.image_resize_algo", "lanczos", "clip.vision.image_min_pixels", 65536,
                "clip.vision.image_max_pixels", 1048576)).contains("'lanczos'"));
        assertTrue(refusal(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 3,
                "clip.vision.image_resize_algo", "bilinear", "clip.vision.image_min_pixels", 65536,
                "clip.vision.image_max_pixels", 1048576)).contains("'clip.vision.spatial_merge_size'"));
        assertTrue(refusal(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                "clip.vision.image_resize_algo", "bilinear", "clip.vision.image_min_pixels", 65536))
                .contains("'clip.vision.image_max_pixels'"));
        assertTrue(refusal(metadata("lfm3vl", 16, 512, "clip.vision.spatial_merge_size", 2,
                "clip.vision.image_resize_algo", "bilinear", "clip.vision.image_min_pixels", 2048000,
                "clip.vision.image_max_pixels", 1048576)).contains("0 < min <= max"));
    }

    @Test
    public void theLongestEdgeFamiliesDefaultTheirEdge() {
        ImageProcessorParams step = params(metadata("step3vl", 14, 728));
        assertEquals(3024, step.longestEdge());
        assertEquals(4, step.mergeSize());
        assertFalse(step.overviewFirst());
        assertEquals(2000, params(metadata("step3vl", 14, 728, "clip.vision.preproc_image_size", 2000))
                .longestEdge());

        ImageProcessorParams ocr = params(metadata("lightonocr", 14, 1540, "clip.vision.spatial_merge_size", 2));
        assertEquals(1540, ocr.longestEdge());
        assertEquals(ResizeAlgo.BICUBIC, ocr.resizeAlgo());
        assertEquals(1024, params(metadata("lightonocr", 14, 1540, "clip.vision.preproc_image_size", 1024))
                .longestEdge());
    }

    @Test
    public void llama4ResizesBilinearOverEveryGridUpToThreeTiles() {
        ImageProcessorParams p = params(metadata("llama4", 14, 336, "clip.vision.projector.scale_factor", 2));
        assertEquals(ResizeAlgo.BILINEAR, p.resizeAlgo());
        assertEquals(ResizeAlgo.BILINEAR, p.overviewResizeAlgo());
        assertEquals(ResizeAlgo.BILINEAR, p.refinedResizeAlgo());
        assertFalse(p.overviewFirst());
        // set_llava_uhd_res_candidates(model, 3): x outer, y inner, the single tile skipped
        assertEquals(Arrays.asList(new ImageSize(336, 672), new ImageSize(336, 1008), new ImageSize(672, 336),
                new ImageSize(672, 672), new ImageSize(672, 1008), new ImageSize(1008, 336),
                new ImageSize(1008, 672), new ImageSize(1008, 1008)), p.gridPinpoints());
    }

    @Test
    public void glmEdgePadsBlackAndJanusProPadsGray() {
        ImageProcessorParams edge = params(metadata("adapter", 14, 672));
        assertArrayEquals(new int[] {0, 0, 0}, edge.padColor());
        assertEquals(PadStyle.CEIL, edge.padStyle());
        assertEquals(ResizeAlgo.BICUBIC, edge.resizeAlgo());
        ImageProcessorParams janus = params(metadata("janus_pro", 16, 384));
        assertArrayEquals(new int[] {127, 127, 127}, janus.padColor());
        assertEquals(ResizeAlgo.BICUBIC, janus.resizeAlgo());
        assertArrayEquals(new int[] {122, 116, 104}, params(metadata("mlp", 14, 336)).padColor());
        assertArrayEquals(new int[] {122, 116, 104}, params(metadata("cogvlm", 14, 490)).padColor());
    }

    @Test
    public void theMiniCpmMergersStretchWithPythonRounding() {
        ImageProcessorParams merger = params(metadata("minicpmv4_6", 14, 448));
        assertEquals(4, merger.mergeSize());
        assertEquals(PadStyle.NONE, merger.overviewPadStyle());
        assertEquals(PadStyle.NONE, merger.refinedPadStyle());
        assertTrue(merger.roundHalfEven());
        assertEquals(9, merger.maxSliceNums());
        assertEquals(4, params(metadata("minicpm_ocr_1_0", 14, 448, "clip.vision.max_slice_nums", 4))
                .maxSliceNums());
        assertEquals(2, params(metadata("minicpm_ocr_1_0", 14, 448, "clip.vision.projector.scale_factor", 2))
                .mergeSize());
        assertTrue(refusal(metadata("minicpmv4_7", 14, 448, "clip.vision.projector.scale_factor", 3))
                .contains("'clip.vision.projector.scale_factor'"));
    }

    @Test
    public void deepseekOcrFixesItsViewsAndReadsItsTileRange() {
        ImageProcessorParams v1 = params(metadata("deepseekocr", 14, 336));
        assertEquals(16, v1.patchSize());
        assertEquals(1024, v1.imageSize());
        assertEquals(640, v1.tileSize());
        assertEquals(2, v1.minTiles());
        assertEquals(9, v1.maxTiles());
        assertArrayEquals(new int[] {127, 127, 127}, v1.padColor());
        assertFalse(v1.overviewFirst());
        ImageProcessorParams v2 = params(metadata("deepseekocr2", 14, 336));
        assertEquals(768, v2.tileSize());
        assertEquals(6, v2.maxTiles());
        assertEquals(32, params(metadata("deepseekocr", 16, 1024, "clip.vision.preproc_max_tiles", 32))
                .maxTiles());
        assertTrue(refusal(metadata("deepseekocr", 16, 1024, "clip.vision.preproc_max_tiles", 300))
                .contains("'clip.vision.preproc_max_tiles'"));
    }

    @Test
    public void tokenLimitFamiliesTurnTheirTokenRangeIntoPixels() {
        ImageProcessorParams gemma = params(metadata("gemma4v", 16, 768));
        assertEquals(3, gemma.mergeSize());
        assertEquals(70L * 48 * 48, gemma.minPixels());
        assertEquals(1120L * 48 * 48, gemma.maxPixels());
        assertEquals(PadStyle.NONE, gemma.padStyle());
        // the unified variant folds the 3x3 pooling into a 48-pixel patch
        ImageProcessorParams unified = params(metadata("gemma4uv", 16, 768));
        assertEquals(48, unified.patchSize());
        assertEquals(1, unified.mergeSize());
        assertEquals(70L * 48 * 48, unified.minPixels());
        assertEquals(1120L * 48 * 48, unified.maxPixels());

        ImageProcessorParams youtu = params(metadata("youtuvl", 16, 560));
        assertEquals(ResizeAlgo.BILINEAR, youtu.resizeAlgo());
        assertEquals(1024, youtu.minPixels());
        assertEquals(62500L * 1024, youtu.maxPixels());
        ImageProcessorParams minimax = params(metadata("minimax_m3", 14, 672));
        assertEquals(8L * 784, minimax.minPixels());
        assertEquals(576L * 784, minimax.maxPixels());
        ImageProcessorParams muse = params(metadata("muse-glimmer", 16, 512));
        assertEquals(ResizeAlgo.LANCZOS, muse.resizeAlgo());
        assertEquals(1024, muse.minPixels());
        assertEquals(4096L * 1024, muse.maxPixels());
    }

    @Test
    public void theGgufLimitFamiliesRequireTheirPixelKeys() {
        for (String projector : new String[] {"glm5v", "dots3note_v"}) {
            String message = refusal(metadata(projector, 14, 336, "clip.vision.spatial_merge_size", 2));
            assertTrue(message, message.contains("'clip.vision.image_min_pixels'"));
            assertTrue(message, message.contains("'" + projector + "'"));
        }
        ImageProcessorParams glm = params(metadata("glm5v", 14, 336, "clip.vision.image_min_pixels", 3136,
                "clip.vision.image_max_pixels", 1003520));
        assertEquals(3136, glm.minPixels());
        assertEquals(1003520, glm.maxPixels());
        assertEquals(ResizeAlgo.BICUBIC, glm.resizeAlgo());
    }
}
