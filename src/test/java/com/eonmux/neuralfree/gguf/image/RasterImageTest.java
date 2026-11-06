package com.eonmux.neuralfree.gguf.image;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;
import static org.junit.Assert.fail;

public class RasterImageTest {

    private static RasterImage twoByTwo() {
        return new RasterImage(2, 2, new byte[] {
                10, 20, 30,  40, 50, 60,
                70, 80, 90,  100, 110, 120});
    }

    @Test
    public void pixelAccessReadsInterleavedRgbRowMajor() {
        RasterImage img = twoByTwo();
        assertEquals(0x0A141E, img.pixel(0, 0));
        assertEquals(0x28323C, img.pixel(1, 0));
        assertEquals(0x46505A, img.pixel(0, 1));
        assertEquals(0x646E78, img.pixel(1, 1));
        assertEquals(10, img.red(0, 0));
        assertEquals(110, img.green(1, 1));
        assertEquals(120, img.blue(1, 1));
    }

    @Test
    public void rgbHandsOutACopy() {
        RasterImage img = twoByTwo();
        byte[] bytes = img.rgb();
        bytes[0] = -1;
        assertEquals(10, img.red(0, 0));
        assertNotSame(bytes, img.rgb());
    }

    @Test
    public void constructorCopiesThePixelsIn() {
        byte[] bytes = new byte[] {1, 2, 3, 4, 5, 6};
        RasterImage img = new RasterImage(1, 2, bytes);
        bytes[0] = -1;
        assertEquals(1, img.red(0, 0));
    }

    @Test
    public void rejectsWrongArrayLengthAndSizes() {
        try {
            new RasterImage(2, 2, new byte[3]);
            fail("expected a length rejection");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("needs 12"));
        }
        try {
            new RasterImage(0, 1, new byte[0]);
            fail("expected a width rejection");
        } catch (IllegalArgumentException e) {
            // named in the message
        }
    }

    @Test
    public void normalizedSubtractsMeanAndDividesStdPerChannel() {
        RasterImage img = twoByTwo();
        float[] out = img.normalized(new float[] {0.5f, 0.5f, 0.5f}, new float[] {0.25f, 0.5f, 1.0f});
        // pixel (0,0): r=10, g=20, b=30
        assertEquals((10 / 255.0f - 0.5f) / 0.25f, out[0], 1e-6f);
        assertEquals((20 / 255.0f - 0.5f) / 0.5f, out[1], 1e-6f);
        assertEquals((30 / 255.0f - 0.5f) / 1.0f, out[2], 1e-6f);
        // channel order repeats every pixel
        assertEquals((70 / 255.0f - 0.5f) / 0.25f, out[6], 1e-6f);
    }

    @Test
    public void normalizedRejectsWrongMeanStdAndZeroStd() {
        RasterImage img = twoByTwo();
        try {
            img.normalized(new float[2], new float[] {1, 1, 1});
            fail("expected a mean rejection");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("mean"));
        }
        try {
            img.normalized(new float[] {1, 1, 1}, new float[] {1, 0, 1});
            fail("expected a zero std rejection");
        } catch (IllegalArgumentException e) {
            assertEquals(true, e.getMessage().contains("std"));
        }
    }

    @Test
    public void filledPaintsEveryPixel() {
        RasterImage img = RasterImage.filled(3, 2, 122, 116, 104);
        assertArrayEquals(new byte[] {122, 116, 104}, new byte[] {
                (byte) img.red(2, 1), (byte) img.green(2, 1), (byte) img.blue(2, 1)});
        assertEquals(6, img.size().area());
    }
}
