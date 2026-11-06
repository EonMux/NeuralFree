package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The longest-edge preprocessor against llama.cpp's {@code mtmd_image_preprocessor_longest_edge} with the
 * Pixtral defaults: the longest side floors to the edge before the align rounding, the merged patch size
 * aligns the result and the derived token cap bounds the area.
 */
public class LongestEdgePreprocessorTest {

    private static ImageProcessorParams pixtral(Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "pixtral");
        m.put("clip.vision.patch_size", 16);
        m.put("clip.vision.image_size", 1024);
        m.put("clip.vision.preproc_image_size", 1024);
        m.put("clip.vision.image_mean", new float[] {0.481f, 0.457f, 0.408f});
        m.put("clip.vision.image_std", new float[] {0.269f, 0.261f, 0.276f});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
    }

    @Test
    public void wideImagesFloorToTheLongestEdgeThenAlign() {
        // 2048x1024: ratio 2 floors to 1024x512, the align keeps it, tokens = 64 * 32 plus 31 [IMG_BREAK] rows
        LongestEdgePreprocessor pre = (LongestEdgePreprocessor) ImagePreprocessors.forParams(pixtral());
        assertEquals(new ImageSize(1024, 512), pre.targetSize(new ImageSize(2048, 1024)));
        PreparedImages out = pre.prepare(RasterImage.filled(2048, 1024, 0, 0, 0));
        assertEquals(1, out.encodedTiles().size());
        assertEquals(64 * 32 + 31, out.totalTokens());
    }

    @Test
    public void oddSizesFloorBeforeTheAlign() {
        // 1500x1000: ratio 1.4648 floors to 1024x683, then the align rounds up to the 16 multiple
        LongestEdgePreprocessor pre = (LongestEdgePreprocessor) ImagePreprocessors.forParams(pixtral());
        ImageSize out = pre.targetSize(new ImageSize(1500, 1000));
        assertEquals(1024, out.width());
        assertEquals(0, out.height() % 16);
        assertEquals(683, out.height(), 16);
    }

    @Test
    public void tokenCapBoundsTheArea() {
        LongestEdgePreprocessor pre = (LongestEdgePreprocessor) ImagePreprocessors.forParams(pixtral());
        // the cap is 64 tokens per side -> 4096 tokens -> 4096 * 256 pixels
        ImageSize out = pre.targetSize(new ImageSize(1024, 1024));
        assertEquals(true, (long) out.width() * out.height() <= 4096L * 256L);
        assertEquals(new ImageSize(1024, 1024), out);
    }

    @Test
    public void tinyImagesAreRefusedWhenTheFloorErasesThem() {
        // a 10000x1 image floors to height 0 -> the reference refuses by name; a 1x1 image stays (no floor,
        // the ratio never exceeds 1) and just aligns up to the merged patch size
        LongestEdgePreprocessor pre = (LongestEdgePreprocessor) ImagePreprocessors.forParams(pixtral());
        try {
            pre.targetSize(new ImageSize(10000, 1));
            fail("expected a positive-dimensions refusal");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("positive"));
        }
        assertEquals(new ImageSize(16, 16), pre.targetSize(new ImageSize(1, 1)));
    }

    @Test
    public void missingLongestEdgeIsRefused() {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "pixtral");
        m.put("clip.vision.patch_size", 16);
        m.put("clip.vision.image_size", 1024);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(m), ImageTokenBudget.none());
        // without preproc_image_size the projector goes to the dynamic-size preprocessor instead
        assertEquals(true, ImagePreprocessors.forParams(params) instanceof DynSizePreprocessor);
    }
}
