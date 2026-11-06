package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * The dynamic-size preprocessor against llama.cpp's {@code mtmd_image_preprocessor_dyn_size} over the Qwen2-VL
 * family defaults: a single tile at the smart-resize size (round half to even, no padding), whose merged-conv
 * grid is the token count.
 */
public class DynSizePreprocessorTest {

    private static Map<String, Object> metadata(Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "qwen2vl_merger");
        m.put("clip.vision.patch_size", 14);
        m.put("clip.vision.image_size", 336);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static ImageProcessorParams qwen(Object... extra) {
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(extra)), ImageTokenBudget.none());
    }

    @Test
    public void pixelLimitsDriveTheTargetSize() {
        // the transformers smart-resize example through the qwen limits: min 56*56, max 14*14*4*1280
        ImageProcessorParams params = qwen(
                "clip.vision.image_min_pixels", 56L * 56,
                "clip.vision.image_max_pixels", 14L * 14 * 4 * 1280);
        DynSizePreprocessor pre = (DynSizePreprocessor) ImagePreprocessors.forParams(params);
        assertEquals(new ImageSize(812, 1204), pre.targetSize(new ImageSize(1556, 2352)));

        PreparedImages out = pre.prepare(RasterImage.filled(1556, 2352, 1, 2, 3));
        assertEquals(1, out.encodedTiles().size());
        assertEquals(812, out.encodedTiles().get(0).width());
        assertEquals(1204, out.encodedTiles().get(0).height());
        // (812 / 28) * (1204 / 28) merged-conv tokens
        assertEquals(29 * 43, out.totalTokens());
    }

    @Test
    public void defaultTokenLimitsShapeTheTarget() {
        // qwen defaults 8..4096 tokens over the (14*2)^2 = 784 pixel area: 6272..3211264 pixels
        DynSizePreprocessor pre = (DynSizePreprocessor) ImagePreprocessors.forParams(qwen());
        ImageSize small = pre.targetSize(new ImageSize(100, 100));
        // 100x100 rounds to 112x112 = 12544 >= 6272, within limits
        assertEquals(new ImageSize(112, 112), small);
    }

    @Test
    public void largeImagesCapAtMaxPixels() {
        DynSizePreprocessor pre = (DynSizePreprocessor) ImagePreprocessors.forParams(qwen());
        // 4000x3000 = 12M pixels > 3211264: beta shrinks to the aligned ceiling of the cap
        ImageSize out = pre.targetSize(new ImageSize(4000, 3000));
        assertEquals(true, (long) out.width() * out.height() <= 4096L * 784L);
        assertEquals(0, out.width() % 28);
        assertEquals(0, out.height() % 28);
    }

    @Test
    public void roundHalfEvenShapesTheSides() {
        // the min budget would otherwise push 70x28 up, so the limits are disabled here:
        // 70/28 = 2.5 -> Python round gives 2 -> width 56, height 28
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(
                "clip.vision.image_min_pixels", 0,
                "clip.vision.image_max_pixels", 0)), ImageTokenBudget.none());
        DynSizePreprocessor pre = (DynSizePreprocessor) ImagePreprocessors.forParams(params);
        assertEquals(new ImageSize(56, 28), pre.targetSize(new ImageSize(70, 28)));
    }

    @Test
    public void tokenBudgetReplacesThePixelLimits() {
        // --image-min-tokens 1024/--image-max-tokens 2048 over the merged patch area
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata()),
                ImageTokenBudget.builder().minTokens(1024).maxTokens(2048).build());
        DynSizePreprocessor pre = (DynSizePreprocessor) ImagePreprocessors.forParams(params);
        // 100x100: without the budget it would round to 112x112; the min budget pushes it to 1024*784 pixels
        ImageSize out = pre.targetSize(new ImageSize(100, 100));
        assertEquals(true, (long) out.width() * out.height() >= 1024L * 784L);
        assertEquals(true, (long) out.width() * out.height() <= 2048L * 784L);
        assertEquals(0, out.width() % 28);
        assertEquals(0, out.height() % 28);
    }

    @Test
    public void tokenGridFollowsTheMergedConv() {
        // limits disabled: 56x56 stays, the merged-conv grid is (56/28)^2 = 4 tokens
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata(
                "clip.vision.image_min_pixels", 0,
                "clip.vision.image_max_pixels", 0)), ImageTokenBudget.none());
        PreparedImages out = ImagePreprocessors.forParams(params).prepare(RasterImage.filled(56, 56, 0, 0, 0));
        assertEquals(new ImageSize(2, 2), out.tokenGrid());
        assertEquals(4, out.totalTokens());
    }
}
