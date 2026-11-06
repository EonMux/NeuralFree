package com.eonmux.neuralfree.gguf.image;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * Decoding through ImageIO for the formats stb_image covers (PNG, JPEG, BMP, GIF), transparency composited
 * over white as vLLM does, and the named refusals (TIFF, ICO, unknown magic); the WebP canvas parsing follows
 * the RIFF container spec, and a real WebP decodes through ffmpeg when it is on the PATH.
 */
public class ImageDecoderTest {

    private static byte[] encode(BufferedImage image, String format) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, format, out);
        return out.toByteArray();
    }

    private static BufferedImage solid(int width, int height, int type, Color color) {
        BufferedImage img = new BufferedImage(width, height, type);
        Graphics2D g = img.createGraphics();
        try {
            g.setColor(color);
            g.fillRect(0, 0, width, height);
        } finally {
            g.dispose();
        }
        return img;
    }

    private static void assertDecodesToSolidRed(byte[] data, int width, int height) {
        RasterImage img = ImageDecoder.decode(data);
        assertEquals(width, img.width());
        assertEquals(height, img.height());
        // JPEG and GIF quantize, so the solid red survives within a small tolerance per channel
        assertPixelClose(0xFF0000, img.pixel(0, 0), 2);
        assertPixelClose(0xFF0000, img.pixel(width - 1, height - 1), 2);
    }

    private static void assertPixelClose(int expectedRgb, int actualRgb, int tolerance) {
        assertTrue("pixel " + Integer.toHexString(actualRgb) + " vs " + Integer.toHexString(expectedRgb),
                Math.abs(((actualRgb >>> 16) & 0xFF) - ((expectedRgb >>> 16) & 0xFF)) <= tolerance
                        && Math.abs(((actualRgb >>> 8) & 0xFF) - ((expectedRgb >>> 8) & 0xFF)) <= tolerance
                        && Math.abs((actualRgb & 0xFF) - (expectedRgb & 0xFF)) <= tolerance);
    }

    @Test
    public void decodesPngJpegBmpGif() throws IOException {
        for (String format : new String[] {"png", "jpg", "bmp", "gif"}) {
            byte[] data = encode(solid(3, 2, BufferedImage.TYPE_INT_RGB, Color.RED), format);
            assertDecodesToSolidRed(data, 3, 2);
        }
    }

    @Test
    public void transparencyCompositesOverWhite() throws IOException {
        BufferedImage source = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = source.createGraphics();
        try {
            g.setColor(new Color(255, 0, 0, 0));   // fully transparent red
            g.fillRect(0, 0, 2, 2);
        } finally {
            g.dispose();
        }
        RasterImage img = ImageDecoder.decode(encode(source, "png"));
        assertEquals(0xFFFFFF, img.pixel(0, 0));
    }

    @Test
    public void theBackgroundColourOfTheLoaderOptionsReplacesWhite() throws IOException {
        BufferedImage source = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
        source.setRGB(0, 0, 0x00FF0000);          // transparent red
        source.setRGB(1, 0, 0xFF00FF00);          // opaque green
        ImageLoadOptions blue = ImageLoadOptions.of(ImageLoadOptions.Mode.RGB, 0x0000FF);
        RasterImage img = ImageDecoder.decode(encode(source, "png"), blue);
        assertEquals(0x0000FF, img.pixel(0, 0));
        assertEquals(0x00FF00, img.pixel(1, 0));
    }

    @Test
    public void grayscaleModeTakesPilsLumaOfTheColourChannelsAndIgnoresAlpha() throws IOException {
        BufferedImage source = new BufferedImage(2, 1, BufferedImage.TYPE_INT_ARGB);
        source.setRGB(0, 0, 0xFFFF0000);          // opaque red
        source.setRGB(1, 0, 0x0000FF00);          // transparent green: PIL's convert("L") keeps its colour
        ImageLoadOptions gray = ImageLoadOptions.of(ImageLoadOptions.Mode.L, ImageLoadOptions.WHITE);
        RasterImage img = ImageDecoder.decode(encode(source, "png"), gray);
        assertEquals("L = (19595 R + 38470 G + 7471 B + 0x8000) >> 16, on three channels", 0x4C4C4C, img.pixel(0, 0));
        assertEquals(0x969696, img.pixel(1, 0));
    }

    @Test
    public void rawRgbaPixelsCompositeOverTheBackgroundOrTurnGray() {
        byte[] rgba = {(byte) 255, 0, 0, (byte) 128, 0, 0, (byte) 255, (byte) 255};
        RasterImage over = ImageDecoder.fromRgba(2, 1, rgba, ImageLoadOptions.defaults());
        assertEquals("half-transparent red over white", 0xFF7F7F, over.pixel(0, 0));
        assertEquals(0x0000FF, over.pixel(1, 0));
        RasterImage gray = ImageDecoder.fromRgba(2, 1, rgba,
                ImageLoadOptions.of(ImageLoadOptions.Mode.L, ImageLoadOptions.WHITE));
        assertEquals(0x4C4C4C, gray.pixel(0, 0));
        assertEquals(0x1D1D1D, gray.pixel(1, 0));
    }

    @Test
    public void opaqueImagePixelsSurvive() throws IOException {
        BufferedImage source = solid(2, 1, BufferedImage.TYPE_INT_RGB, new Color(1, 2, 3));
        RasterImage img = ImageDecoder.decode(encode(source, "png"));
        assertEquals(0x010203, img.pixel(1, 0));
    }

    @Test
    public void emptyBufferIsRefusedByName() {
        try {
            ImageDecoder.decode(new byte[0]);
            fail("expected an empty buffer rejection");
        } catch (ImageDecodeException e) {
            assertEquals(true, e.getMessage().contains("empty"));
        }
        try {
            ImageDecoder.decode(null);
            fail("expected a null buffer rejection");
        } catch (ImageDecodeException e) {
            // named in the message
        }
    }

    @Test
    public void unsupportedFormatsAreRefusedByName() throws IOException {
        assertRefused("TIFF", new byte[] {'I', 'I', 0x2A, 0x00, 0, 0, 0, 0});
        assertRefused("TIFF", new byte[] {'M', 'M', 0x00, 0x2A, 0, 0, 0, 0});
        assertRefused("ICO", new byte[] {0x00, 0x00, 0x01, 0x00, 1, 0, 0, 0});
        assertRefused("unknown", new byte[] {1, 2, 3, 4, 5, 6, 7, 8});
    }

    private static void assertRefused(String name, byte[] data) {
        try {
            ImageDecoder.decode(data);
            fail("expected a refusal for " + name);
        } catch (ImageDecodeException e) {
            assertEquals(true, e.getMessage().contains(name));
            assertEquals(true, e.getMessage().contains("PNG, JPEG, BMP, GIF, WebP"));
        }
    }

    @Test
    public void corruptPngIsRefused() {
        byte[] truncated = new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13};
        try {
            ImageDecoder.decode(truncated);
            fail("expected a corruption refusal");
        } catch (ImageDecodeException e) {
            assertEquals(true, e.getMessage().contains("PNG"));
        }
    }

    @Test
    public void webpCanvasSizeParsesEveryChunkFormat() {
        // VP8 (lossy): frame tag, sync code, then the 16-bit width and height, masked to 14 bits
        byte[] vp8 = riff("VP8 ", new byte[] {0, 0, 0, (byte) 0x9D, 0x01, 0x2A, 0x37, 0x01, (byte) 0xC7, 0x00});
        assertEquals(new ImageSize(311, 199), ImageDecoder.webpCanvasSize(vp8));
        // VP8L (lossless): signature 0x2f then packed 14-bit sizes minus one
        byte[] vp8l = riff("VP8L", new byte[] {0x2F, 0x2A, (byte) 0x80, 0x13, 0x00});
        assertEquals(new ImageSize(43, 79), ImageDecoder.webpCanvasSize(vp8l));
        // VP8X (extended): 24-bit canvas sizes minus one after the flags; one extra byte stands for the
        // next chunk's first byte, which the 4-byte height read touches
        byte[] vp8x = riff("VP8X", new byte[] {0x10, 0, 0, 0, 0x0B, 0x00, 0x00, (byte) 0xC7, 0x00, 0x00, 0x00});
        assertEquals(new ImageSize(12, 200), ImageDecoder.webpCanvasSize(vp8x));
    }

    private static byte[] riff(String chunk, byte[] payload) {
        byte[] out = new byte[12 + 8 + payload.length + (payload.length & 1)];
        out[0] = 'R';
        out[1] = 'I';
        out[2] = 'F';
        out[3] = 'F';
        int size = out.length - 8;
        out[4] = (byte) size;
        out[5] = (byte) (size >>> 8);
        out[6] = (byte) (size >>> 16);
        out[7] = (byte) (size >>> 24);
        out[8] = 'W';
        out[9] = 'E';
        out[10] = 'B';
        out[11] = 'P';
        for (int i = 0; i < 4; i++) {
            out[12 + i] = (byte) chunk.charAt(i);
        }
        int payloadSize = payload.length;
        out[16] = (byte) payloadSize;
        out[17] = (byte) (payloadSize >>> 8);
        out[18] = (byte) (payloadSize >>> 16);
        out[19] = (byte) (payloadSize >>> 24);
        for (int i = 0; i < payload.length; i++) {
            out[20 + i] = payload[i];
        }
        return out;
    }

    @Test
    public void corruptWebpWithoutCanvasChunkIsRefused() {
        byte[] data = new byte[24];
        data[0] = 'R';
        data[1] = 'I';
        data[2] = 'F';
        data[3] = 'F';
        data[8] = 'W';
        data[9] = 'E';
        data[10] = 'B';
        data[11] = 'P';
        try {
            ImageDecoder.webpCanvasSize(data);
            fail("expected a canvas refusal");
        } catch (ImageDecodeException e) {
            assertEquals(true, e.getMessage().contains("WebP"));
        }
    }

    @Test
    public void webpDecodeNeedsFfmpegOrAFailingProcess() throws IOException {
        byte[] data = riff("VP8 ", new byte[] {0, 0, 0, (byte) 0x9D, 0x01, 0x2A, 0x37, 0x01, (byte) 0xC7, 0x00});
        // the body is garbage, so decode fails either way; the message names ffmpeg on both paths
        try {
            ImageDecoder.decode(data);
            fail("expected a WebP decode failure");
        } catch (ImageDecodeException e) {
            assertEquals(true, e.getMessage().contains("ffmpeg"));
        }
    }

    @Test
    public void theWebpDecodeIsGuardedAtItsInput() {
        assertEquals(java.util.Arrays.asList(
                "ffmpeg",
                "-protocol_whitelist", "pipe",
                "-format_whitelist", "webp_pipe,webp_anim,image2,image2pipe",
                "-i", "pipe:0",
                "-f", "rawvideo", "-pix_fmt", "rgba", "-v", "error", "pipe:1"),
                ImageDecoder.webpCommand());
    }

    @Test
    public void realWebpDecodesThroughFfmpegWhenAvailable() throws Exception {
        Path dir = Files.createTempDirectory("nf-webp-test");
        Path png = dir.resolve("in.png");
        Path webp = dir.resolve("out.webp");
        Files.write(png, encode(solid(4, 3, BufferedImage.TYPE_INT_RGB, Color.RED), "png"));
        Process ffmpeg;
        try {
            ffmpeg = new ProcessBuilder("ffmpeg", "-y", "-v", "error", "-i", png.toString(), webp.toString())
                    .start();
        } catch (IOException e) {
            // no ffmpeg on the PATH: llama.cpp cannot decode WebP here either; skip the parity run
            org.junit.Assume.assumeTrue("ffmpeg is not on the PATH", false);
            return;
        }
        try {
            if (ffmpeg.waitFor() != 0) {
                org.junit.Assume.assumeTrue("ffmpeg cannot encode WebP here", false);
                return;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted", e);
        }
        assertTrue(Files.size(webp) > 0);
        assertDecodesToSolidRed(Files.readAllBytes(webp), 4, 3);
    }
}
