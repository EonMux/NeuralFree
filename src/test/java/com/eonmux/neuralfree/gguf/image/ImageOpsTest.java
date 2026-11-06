package com.eonmux.neuralfree.gguf.image;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.fail;

/**
 * The image operations against llama.cpp's {@code img_tool} (tools/mtmd/mtmd-image.cpp): the padded resize
 * (fit, center, fill), the crop and the smart resize, with the transformers smart-resize example
 * (2352 x 1556, factor 28, min 56*56, max 14*14*4*1280 -&gt; 812 x 1204) as the parity pin.
 */
public class ImageOpsTest {

    @Test
    public void cropExtractsTheExactRectangle() {
        RasterImage src = RasterImage.filled(4, 4, 0, 0, 0);
        RasterImage marked = ImageOps.composite(src, RasterImage.filled(2, 2, 255, 0, 0), 2, 2);
        RasterImage crop = ImageOps.crop(marked, 2, 2, 2, 2);
        assertEquals(2, crop.width());
        assertEquals(2, crop.height());
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                assertEquals(0xFF0000, crop.pixel(x, y));
            }
        }
    }

    @Test
    public void cropRefusesEmptyAndOutOfBoundsRectangles() {
        RasterImage src = RasterImage.filled(4, 4, 0, 0, 0);
        try {
            ImageOps.crop(src, 0, 0, 0, 2);
            fail("expected an empty rectangle rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
        try {
            ImageOps.crop(src, 3, 3, 2, 2);
            fail("expected an out-of-bounds rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
    }

    @Test
    public void paddedResizeFitsCentersAndFills() {
        // 300x100 into 336x336, pad color 122/116/104: scale 1.12 -> 336x112, centered at y=112
        RasterImage src = RasterImage.filled(300, 100, 0, 0, 255);
        RasterImage out = ImageOps.resize(src, new ImageSize(336, 336), ResizeAlgo.BILINEAR, PadStyle.CEIL,
                new int[] {122, 116, 104});
        assertEquals(336, out.width());
        assertEquals(336, out.height());
        assertEquals(0x7A7468, out.pixel(0, 0));          // 122, 116, 104 border
        assertEquals(0x7A7468, out.pixel(167, 111));
        assertEquals(0x0000FF, out.pixel(167, 167));      // content starts at row 112
        assertEquals(0x7A7468, out.pixel(167, 224));      // content ends at row 223
    }

    @Test
    public void paddedResizeRoundFitsInsideTheTarget() {
        // 100x300 into 336x336 with NEAREST: fitted sides round to nearest -> 112x336, offset (112, 0)
        RasterImage src = RasterImage.filled(100, 300, 0, 255, 0);
        RasterImage out = ImageOps.resize(src, new ImageSize(336, 336), ResizeAlgo.BILINEAR, PadStyle.NEAREST,
                new int[] {0, 0, 0});
        assertEquals(0, out.red(111, 167));
        assertEquals(0x00FF00, out.pixel(112, 167));
    }

    @Test
    public void padNoneStretches() {
        RasterImage src = RasterImage.filled(100, 200, 255, 255, 255);
        RasterImage out = ImageOps.resize(src, new ImageSize(336, 336), ResizeAlgo.BILINEAR, PadStyle.NONE,
                new int[] {0, 0, 0});
        assertEquals(336, out.width());
        assertEquals(336, out.height());
        assertEquals(0xFFFFFF, out.pixel(335, 335));
    }

    @Test
    public void sameSizeResizeSkipsTheResample() {
        RasterImage src = RasterImage.filled(8, 8, 1, 2, 3);
        assertSame(src, ImageOps.resize(src, new ImageSize(8, 8), ResizeAlgo.BILINEAR, PadStyle.NONE,
                new int[] {0, 0, 0}));
    }

    @Test
    public void smartResizeMatchesTheTransformersExample() {
        // the documented example of the transformers image processors: 2352x1556 (h x w), factor 28,
        // min 56*56, max 14*14*4*1280 -> (812, 1204); the area first rounds to 2352x1568 = 3.69M pixels,
        // which exceeds the 1,003,520 max, so beta = sqrt(h*w/max) shrinks both sides onto the multiple
        ImageSize out = ImageOps.smartResize(new ImageSize(1556, 2352), SmartResizeOptions.builder(28)
                .minPixels(56L * 56)
                .maxPixels(14L * 14 * 4 * 1280)
                .roundHalfEven(true)
                .build());
        assertEquals(new ImageSize(812, 1204), out);
    }

    @Test
    public void smartResizeUpscalesTinyImagesToMinPixels() {
        // 16x16, factor 28: rounds to 28x28 = 784 < 3136, so beta = sqrt(3136/256) = 3.5 scales up to 56x56
        ImageSize out = ImageOps.smartResize(new ImageSize(16, 16), SmartResizeOptions.builder(28)
                .minPixels(56L * 56)
                .roundHalfEven(true)
                .build());
        assertEquals(new ImageSize(56, 56), out);
    }

    @Test
    public void smartResizeRoundsHalfAwayByDefaultAndHalfToEvenWhenSet() {
        // 70/28 = 2.5: std::round gives 3 -> 84, Python round gives 2 -> 56
        ImageSize halfAway = ImageOps.smartResize(new ImageSize(70, 28), SmartResizeOptions.alignTo(28));
        assertEquals(new ImageSize(84, 28), halfAway);
        ImageSize halfEven = ImageOps.smartResize(new ImageSize(70, 28), SmartResizeOptions.builder(28)
                .roundHalfEven(true)
                .build());
        assertEquals(new ImageSize(56, 28), halfEven);
    }

    @Test
    public void smartResizeCapsAtTheLongestEdgeBeforeTheLimits() {
        // longest edge 1024 scales 4096x1024 down to 1024x256, then the align rounds both sides up to it
        ImageSize out = ImageOps.smartResize(new ImageSize(4096, 1024), SmartResizeOptions.builder(16)
                .longestEdge(1024)
                .build());
        assertEquals(new ImageSize(1024, 256), out);
    }

    @Test
    public void smartResizeEmptyInputGivesEmptySize() {
        assertEquals(ImageSize.EMPTY, ImageOps.smartResize(new ImageSize(0, 10), SmartResizeOptions.alignTo(28)));
    }

    @Test
    public void smartResizeRefusesNonPositiveAlign() {
        try {
            ImageOps.smartResize(new ImageSize(8, 8), SmartResizeOptions.builder(0).build());
            fail("expected an align rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
    }
}
