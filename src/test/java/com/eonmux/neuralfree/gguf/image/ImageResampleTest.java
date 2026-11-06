package com.eonmux.neuralfree.gguf.image;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The resample kernels against the Pillow separable algorithm the references run (mtmd-image.cpp
 * {@code resize_pillow}, Pillow Resample.c). The 2x bilinear upscale has exact quarter weights, so its output
 * pins the pixel values; the bicubic and lanczos pins use their documented kernel values
 * (k(0.5) = 9/16 bicubic, sinc-squared lanczos), both halving to 128 on a two-pixel edge.
 */
public class ImageResampleTest {

    /** A single-row grayscale ramp over the full [0, 255] range. */
    private static RasterImage gradientRow(int width) {
        byte[] rgb = new byte[width * 3];
        for (int x = 0; x < width; x++) {
            int v = x * 255 / (width - 1);
            rgb[x * 3] = (byte) v;
            rgb[x * 3 + 1] = (byte) v;
            rgb[x * 3 + 2] = (byte) v;
        }
        return RasterImage.wrap(width, 1, rgb);
    }

    @Test
    public void sameSizeReturnsTheSameImage() {
        RasterImage img = gradientRow(4);
        assertSame(img, ImageResample.resample(img, 4, 1, ResizeAlgo.BILINEAR));
    }

    @Test
    public void bilinearUpscaleFollowsThePillowWeights() {
        // support 1: output xx reads inputs around center (xx + 0.5) / 2 with weight 1 - |distance|;
        // for 4 -> 8 the weights are exact quarters: .25/.75 pairs, edges reproduce the first/last pixel
        RasterImage out = ImageResample.resample(gradientRow(4), 8, 1, ResizeAlgo.BILINEAR);
        int[] in = {0, 85, 170, 255};
        int[] expected = {
                in[0],
                (int) Math.round(0.75 * in[0] + 0.25 * in[1]),   // 21
                (int) Math.round(0.25 * in[0] + 0.75 * in[1]),   // 64
                (int) Math.round(0.75 * in[1] + 0.25 * in[2]),   // 106
                (int) Math.round(0.25 * in[1] + 0.75 * in[2]),   // 149
                (int) Math.round(0.75 * in[2] + 0.25 * in[3]),   // 191
                (int) Math.round(0.25 * in[2] + 0.75 * in[3]),   // 234
                in[3]};
        for (int x = 0; x < 8; x++) {
            assertEquals("column " + x, expected[x], out.red(x, 0));
        }
    }

    @Test
    public void bilinearIsSeparableRowThenColumn() {
        RasterImage src = RasterImage.filled(1, 1, 90, 130, 170);
        RasterImage out = ImageResample.resample(src, 3, 3, ResizeAlgo.BILINEAR);
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                assertEquals(0x5A82AA, out.pixel(x, y));
            }
        }
    }

    @Test
    public void bilinearDownscaleWidensTheKernel() {
        // 8 -> 4: filterscale 2 widens the support to 2; output 1 centers on 3.0 and reads pixels 1..4 with
        // distances 1.5/0.5/0.5/1.5 pixels -> normalized weights 1/8, 3/8, 3/8, 1/8 over values 36, 72, 109, 145
        RasterImage out = ImageResample.resample(gradientRow(8), 4, 1, ResizeAlgo.BILINEAR);
        assertEquals(91, out.red(1, 0));
        // output 0 reads pixels 0..2 with weights 3/7, 3/7, 1/7 over values 0, 36, 72 -> 26
        assertEquals(26, out.red(0, 0));
    }

    @Test
    public void bicubicHalfPixelWeightsHalveTheEdge() {
        // Pillow bicubic a = -0.5: k(0.5) = 9/16, so on a 2 -> 3 upscale output 1 sits at pixel-center
        // distance 0.5 from both inputs, both weights normalize to 1/2 -> the edge value halves;
        // output 2 rings past the source and clamps to 255
        RasterImage out = ImageResample.resample(gradientRow(2), 3, 1, ResizeAlgo.BICUBIC);
        assertEquals(128, out.red(1, 0));
        assertEquals(255, out.red(2, 0));
    }

    @Test
    public void bicubicStaysInsideTheSourceRangeOnARamp() {
        // the cubic kernel rings, but the clamp to [0, 255] and the monotone ramp keep every value in range
        RasterImage out = ImageResample.resample(gradientRow(4), 8, 1, ResizeAlgo.BICUBIC);
        for (int x = 0; x < 8; x++) {
            assertTrue(out.red(x, 0) >= 0 && out.red(x, 0) <= 255);
        }
        assertEquals(0, out.red(0, 0));
        assertEquals(255, out.red(7, 0));
    }

    @Test
    public void lanczosHalfPixelWeightsHalveTheEdge() {
        // Lanczos3: k(0.5) = sinc(0.5) * sinc(1/6) -> both edge weights equal, normalized 1/2 each
        RasterImage out = ImageResample.resample(gradientRow(2), 3, 1, ResizeAlgo.LANCZOS);
        assertEquals(128, out.red(1, 0));
    }

    @Test
    public void constantImageSurvivesEveryKernel() {
        for (ResizeAlgo algo : ResizeAlgo.values()) {
            RasterImage out = ImageResample.resample(RasterImage.filled(2, 2, 37, 200, 255), 7, 5, algo);
            for (int y = 0; y < 5; y++) {
                for (int x = 0; x < 7; x++) {
                    assertEquals(algo.name(), 0x25C8FF, out.pixel(x, y));
                }
            }
        }
    }

    @Test
    public void targetBoundsAreRefused() {
        RasterImage src = gradientRow(4);
        try {
            ImageResample.resample(src, 0, 4, ResizeAlgo.BILINEAR);
            fail("expected a target rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
        try {
            ImageResample.resample(src, 4, 70000, ResizeAlgo.BILINEAR);
            fail("expected a target rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
    }
}
