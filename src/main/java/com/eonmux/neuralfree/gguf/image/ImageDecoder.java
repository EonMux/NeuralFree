package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.media.FfmpegInput;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * Decodes image bytes into a {@link RasterImage}, the decode step of llama.cpp's
 * {@code mtmd_helper_bitmap_init_from_buf} (tools/mtmd/mtmd-helper.cpp): PNG, JPEG, BMP and GIF decode through
 * ImageIO (the JDK's readers for the formats stb_image covers), and WebP — which ImageIO cannot read — goes
 * through an external {@code ffmpeg}, as llama.cpp decodes it. Transparency is composited over white, as vLLM's
 * {@code rgba_to_rgb} does, or over the background and in the colour mode {@link ImageLoadOptions} set. Every other
 * format is refused by name, including TIFF and ICO, whose magic bytes are recognized.
 *
 * <p>All methods are static; the class holds no state.</p>
 */
public final class ImageDecoder {

    private ImageDecoder() {
    }

    /**
     * Decodes an image buffer in RGB mode over white ({@link ImageLoadOptions#defaults()}).
     *
     * @throws ImageDecodeException for an empty or truncated buffer, an unsupported format (named in the message),
     *                               a corrupt image, or a WebP buffer when no usable ffmpeg is on the PATH
     */
    public static RasterImage decode(byte[] data) {
        return decode(data, ImageLoadOptions.defaults());
    }

    /**
     * Decodes an image buffer in the colour mode and over the background of {@code options}.
     *
     * @throws ImageDecodeException for an empty or truncated buffer, an unsupported format (named in the message),
     *                               a corrupt image, or a WebP buffer when no usable ffmpeg is on the PATH
     */
    public static RasterImage decode(byte[] data, ImageLoadOptions options) {
        Objects.requireNonNull(options, "options");
        if (data == null || data.length == 0) {
            throw new ImageDecodeException("image buffer is empty");
        }
        String format = sniffFormat(data);
        switch (format) {
            case "PNG":
            case "JPEG":
            case "BMP":
            case "GIF":
                return decodeWithImageIo(data, format, options);
            case "WEBP":
                return decodeWebp(data, options);
            default:
                throw new ImageDecodeException("unsupported image format: " + format
                        + " (supported: PNG, JPEG, BMP, GIF, WebP)");
        }
    }

    /** The format the magic bytes name; never null, unknown buffers report their leading bytes. */
    static String sniffFormat(byte[] data) {
        if (matches(data, new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A})) return "PNG";
        if (matches(data, new byte[] {(byte) 0xFF, (byte) 0xD8, (byte) 0xFF})) return "JPEG";
        if (startsWith(data, "GIF87a") || startsWith(data, "GIF89a")) return "GIF";
        if (matches(data, new byte[] {'B', 'M'})) return "BMP";
        if (data.length >= 12 && matches(data, new byte[] {'R', 'I', 'F', 'F'})
                && matchesAt(data, 8, new byte[] {'W', 'E', 'B', 'P'})) return "WEBP";
        if (matches(data, new byte[] {'I', 'I', 0x2A, 0x00}) || matches(data, new byte[] {'M', 'M', 0x00, 0x2A})) {
            return "TIFF";
        }
        if (matches(data, new byte[] {0x00, 0x00, 0x01, 0x00})) return "ICO";
        return "unknown (" + hexPrefix(data) + ")";
    }

    private static boolean matches(byte[] data, byte[] magic) {
        if (data.length < magic.length) return false;
        for (int i = 0; i < magic.length; i++) {
            if (data[i] != magic[i]) return false;
        }
        return true;
    }

    private static boolean matchesAt(byte[] data, int offset, byte[] magic) {
        if (data.length < offset + magic.length) return false;
        for (int i = 0; i < magic.length; i++) {
            if (data[offset + i] != magic[i]) return false;
        }
        return true;
    }

    private static boolean startsWith(byte[] data, String text) {
        byte[] magic = text.getBytes(StandardCharsets.US_ASCII);
        return matches(data, magic);
    }

    private static String hexPrefix(byte[] data) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(8, data.length); i++) {
            if (i > 0) sb.append(' ');
            sb.append(String.format(Locale.ROOT, "%02X", data[i]));
        }
        return sb.toString();
    }

    /**
     * ImageIO decode: in RGB mode composited onto an opaque canvas of the background colour when the source carries
     * transparency, in L mode the luma of the colour channels, alpha ignored.
     */
    private static RasterImage decodeWithImageIo(byte[] data, String format, ImageLoadOptions options) {
        BufferedImage source;
        try {
            source = ImageIO.read(new ByteArrayInputStream(data));
        } catch (IOException e) {
            throw new ImageDecodeException("corrupt " + format + " image: " + e.getMessage(), e);
        } catch (RuntimeException e) {
            throw new ImageDecodeException("corrupt " + format + " image: " + e.getMessage(), e);
        }
        if (source == null) {
            throw new ImageDecodeException("corrupt " + format + " image: no reader decoded it");
        }
        if (source.getWidth() <= 0 || source.getHeight() <= 0) {
            throw new ImageDecodeException("corrupt " + format + " image: " + source.getWidth() + "x"
                    + source.getHeight());
        }
        if (options.mode() == ImageLoadOptions.Mode.L) {
            return grayscale(source);
        }
        BufferedImage rgb = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = rgb.createGraphics();
        try {
            g.setColor(new java.awt.Color(options.background()));
            g.fillRect(0, 0, rgb.getWidth(), rgb.getHeight());
            g.drawImage(source, 0, 0, null);
        } finally {
            g.dispose();
        }
        return toRaster(rgb);
    }

    /** The luma of every pixel's colour channels (alpha ignored, as PIL's {@code convert("L")}) on three channels. */
    private static RasterImage grayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] argb = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] out = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            int p = argb[i];
            byte l = (byte) ImageLoadOptions.luma((p >>> 16) & 0xFF, (p >>> 8) & 0xFF, p & 0xFF);
            out[i * 3] = l;
            out[i * 3 + 1] = l;
            out[i * 3 + 2] = l;
        }
        return RasterImage.wrap(width, height, out);
    }

    private static RasterImage toRaster(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[] argb = image.getRGB(0, 0, width, height, null, 0, width);
        byte[] out = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            int p = argb[i];
            out[i * 3] = (byte) (p >>> 16);
            out[i * 3 + 1] = (byte) (p >>> 8);
            out[i * 3 + 2] = (byte) p;
        }
        return RasterImage.wrap(width, height, out);
    }

    /**
     * WebP decode through ffmpeg (llama.cpp decodes WebP the same way, stb_image has no WebP reader): the
     * canvas size comes from the RIFF header, the first frame comes back as raw RGBA bytes on ffmpeg's stdout and
     * converts to the mode of {@code options} ({@link #fromRgba}). The input is fed on a writer thread, so a full
     * stdout pipe cannot deadlock the feed.
     */
    private static RasterImage decodeWebp(byte[] data, ImageLoadOptions options) {
        ImageSize canvas = webpCanvasSize(data);
        Process process;
        try {
            process = new ProcessBuilder(webpCommand()).start();
        } catch (IOException e) {
            throw new ImageDecodeException("WebP decoding requires ffmpeg on the PATH (llama.cpp decodes WebP"
                    + " through it too): " + e.getMessage(), e);
        }
        byte[] rgba = new byte[canvas.width() * canvas.height() * 4];
        int read;
        try {
            read = readFirstFrame(process, data, rgba);
        } catch (IOException e) {
            process.destroyForcibly();
            throw new ImageDecodeException("ffmpeg WebP decode failed: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            process.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new ImageDecodeException("ffmpeg WebP decode was interrupted");
        }
        int exit = process.exitValue();
        if (exit != 0 || read < rgba.length) {
            throw new ImageDecodeException("ffmpeg WebP decode failed with exit code " + exit
                    + " (" + read + " of " + rgba.length + " frame bytes): " + stderr(process));
        }
        return fromRgba(canvas.width(), canvas.height(), rgba, options);
    }

    /**
     * The arguments of one guarded ffmpeg WebP decode: the client's bytes on standard input, read through the pipe
     * protocol and the image demuxers only, then raw RGBA on standard output.
     */
    static List<String> webpCommand() {
        List<String> command = new ArrayList<>(List.of("ffmpeg"));
        command.addAll(FfmpegInput.ffmpegStdin(FfmpegInput.Media.IMAGE));
        command.addAll(List.of("-f", "rawvideo", "-pix_fmt", "rgba", "-v", "error", "pipe:1"));
        return command;
    }

    /**
     * The raster of raw RGBA pixels: in RGB mode each pixel composited over the background colour, in L mode the
     * luma of its colour channels, alpha ignored.
     */
    static RasterImage fromRgba(int width, int height, byte[] rgba, ImageLoadOptions options) {
        int bg = options.background();
        boolean gray = options.mode() == ImageLoadOptions.Mode.L;
        byte[] out = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++) {
            int r = rgba[i * 4] & 0xFF;
            int g = rgba[i * 4 + 1] & 0xFF;
            int b = rgba[i * 4 + 2] & 0xFF;
            int a = rgba[i * 4 + 3] & 0xFF;
            if (gray) {
                r = ImageLoadOptions.luma(r, g, b);
                g = r;
                b = r;
            } else if (a != 255) {
                r = ImageLoadOptions.over(r, a, (bg >>> 16) & 0xFF);
                g = ImageLoadOptions.over(g, a, (bg >>> 8) & 0xFF);
                b = ImageLoadOptions.over(b, a, bg & 0xFF);
            }
            out[i * 3] = (byte) r;
            out[i * 3 + 1] = (byte) g;
            out[i * 3 + 2] = (byte) b;
        }
        return RasterImage.wrap(width, height, out);
    }

    /** Feeds the buffer on a writer thread, reads the first frame, and waits for ffmpeg to exit. */
    private static int readFirstFrame(Process process, byte[] data, byte[] frame) throws IOException,
            InterruptedException {
        Thread writer = new Thread(() -> {
            try (OutputStream stdin = process.getOutputStream()) {
                stdin.write(data);
            } catch (IOException ignored) {
                // ffmpeg exiting early surfaces through the exit code and the stderr message
            }
        }, "webp-ffmpeg-feed");
        writer.start();
        int read = 0;
        try (InputStream stdout = process.getInputStream()) {
            while (read < frame.length) {
                int n = stdout.read(frame, read, frame.length - read);
                if (n < 0) {
                    break;
                }
                read += n;
            }
        }
        process.waitFor();
        writer.join();
        return read;
    }

    private static String stderr(Process process) {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] chunk = new byte[4096];
        try (InputStream err = process.getErrorStream()) {
            int n;
            while ((n = err.read(chunk)) >= 0) {
                buffer.write(chunk, 0, n);
            }
        } catch (IOException ignored) {
            // the error message stays empty when the stream itself is unreadable
        }
        String text = new String(buffer.toByteArray(), StandardCharsets.UTF_8).trim();
        return text.isEmpty() ? "no error output" : text;
    }

    /**
     * The canvas size from the RIFF chunk headers: the VP8 (lossy), VP8L (lossless) and VP8X (extended) formats
     * each carry it in their first chunk (developers.google.com/speed/webp/docs/riff_container).
     */
    static ImageSize webpCanvasSize(byte[] data) {
        int offset = 12;
        while (offset + 8 <= data.length) {
            String chunk = new String(data, offset, 4, StandardCharsets.US_ASCII);
            int payload = readIntLe(data, offset + 4);
            int body = offset + 8;
            if ("VP8 ".equals(chunk) && body + 10 <= data.length && matchesAt(data, body + 3,
                    new byte[] {(byte) 0x9D, 0x01, 0x2A})) {
                // the VP8 frame header carries the width and height directly, 14 bits with a 2-bit scale
                int width = readU16Le(data, body + 6) & 0x3FFF;
                int height = readU16Le(data, body + 8) & 0x3FFF;
                return checkedSize(width, height);
            }
            if ("VP8L".equals(chunk) && body + 5 <= data.length && data[body] == 0x2F) {
                int bits = readIntLe(data, body + 1);
                int width = (bits & 0x3FFF) + 1;
                int height = ((bits >> 14) & 0x3FFF) + 1;
                return checkedSize(width, height);
            }
            if ("VP8X".equals(chunk) && body + 11 <= data.length) {
                int width = readIntLe(data, body + 4) & 0xFFFFFF;
                int height = readIntLe(data, body + 7) & 0xFFFFFF;
                return checkedSize(width + 1, height + 1);
            }
            offset = body + payload + (payload & 1);
        }
        throw new ImageDecodeException("corrupt WebP image: no VP8/VP8L/VP8X chunk carries the canvas size");
    }

    private static int readIntLe(byte[] data, int offset) {
        return (data[offset] & 0xFF) | (data[offset + 1] & 0xFF) << 8 | (data[offset + 2] & 0xFF) << 16
                | (data[offset + 3] & 0xFF) << 24;
    }

    private static int readU16Le(byte[] data, int offset) {
        return (data[offset] & 0xFF) | (data[offset + 1] & 0xFF) << 8;
    }

    private static ImageSize checkedSize(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new ImageDecodeException("corrupt WebP image: canvas size " + width + "x" + height);
        }
        return new ImageSize(width, height);
    }
}
