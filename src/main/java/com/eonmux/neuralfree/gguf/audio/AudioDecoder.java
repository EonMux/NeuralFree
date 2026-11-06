package com.eonmux.neuralfree.gguf.audio;

import com.eonmux.neuralfree.gguf.media.FfmpegInput;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

/**
 * Decoder of compressed audio bytes to a mono {@link AudioClip}: WAV/AIFF/AU PCM files decode through
 * {@code javax.sound.sampled} (8/16/24/32-bit integer and 32-bit float, any channel count mixed to mono); MP3,
 * FLAC, OGG and M4A decode through one {@code ffmpeg} subprocess per file (mono float32 at the source rate) when
 * the {@code ffmpeg} binary is on the PATH, and are refused by name otherwise. Static and stateless.
 */
public final class AudioDecoder {

    /** Samples one decode returns at most (an hour of 48 kHz mono floats). */
    private static final int MAX_SAMPLES = 60 * 60 * 48000;
    /** Sample rate in ffmpeg's stream banner, e.g. {@code Stream #0:0 ... 16000 Hz, mono, flp}. */
    private static final Pattern RATE_IN_BANNER = Pattern.compile("(\\d{4,6}) Hz");

    private AudioDecoder() {
    }

    /**
     * The bytes as a mono clip at the container's rate.
     *
     * @throws UnsupportedAudioFormatException when the container is recognized but unsupported, has no decoder
     *                                          available, or holds no audio
     * @throws IOException                      when the bytes cannot be read or the ffmpeg subprocess fails
     */
    public static AudioClip decode(byte[] bytes, String name) throws IOException {
        String file = name == null || name.isEmpty() ? "<unnamed>" : name;
        if (bytes == null || bytes.length == 0) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' is empty");
        }
        String kind = sniff(bytes);
        if (kind == null) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' is not a supported container"
                    + " (recognized: WAV, AIFF, AU, MP3, FLAC, OGG, M4A); the bytes start with "
                    + sniffPrefix(bytes));
        }
        return decodeContainer(bytes, file, kind);
    }

    private static AudioClip decodeContainer(byte[] bytes, String file, String kind) throws IOException {
        if ("RIFF_WAVE".equals(kind) || "AIFF".equals(kind) || "AU".equals(kind)) {
            return decodeJavaSound(bytes, file, kind);
        }
        return decodeFfmpeg(bytes, file, kind);
    }

    /**
     * Whether the bytes start with the magic of an audio container this decoder recognizes, the check llama.cpp's
     * {@code mtmd_helper_bitmap_init_from_buf} makes to tell an untyped media buffer's audio from an image.
     */
    public static boolean isAudio(byte[] bytes) {
        return bytes != null && sniff(bytes) != null;
    }

    /** Container magic of the supported formats; {@code null} when unrecognized. */
    static String sniff(byte[] bytes) {
        if (bytes.length >= 12 && startsWithAt(bytes, 0, "RIFF") && startsWithAt(bytes, 8, "WAVE")) {
            return "RIFF_WAVE";
        }
        if (bytes.length >= 12 && startsWithAt(bytes, 0, "FORM") && startsWithAt(bytes, 8, "AIFF")) {
            return "AIFF";
        }
        if (startsWithAt(bytes, 0, ".snd")) {
            return "AU";
        }
        if (startsWithAt(bytes, 0, "ID3")
                || (bytes.length > 2 && (bytes[0] & 0xFF) == 0xFF && (bytes[1] & 0xE6) == 0xE2)) {
            return "MP3";
        }
        if (startsWithAt(bytes, 0, "fLaC")) {
            return "FLAC";
        }
        if (startsWithAt(bytes, 0, "OggS")) {
            return "OGG";
        }
        if (startsWithAt(bytes, 4, "ftyp")) {
            return "M4A";
        }
        return null;
    }

    private static boolean startsWithAt(byte[] b, int off, String s) {
        if (b.length < off + s.length()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (b[off + i] != s.charAt(i)) {
                return false;
            }
        }
        return true;
    }

    private static String sniffPrefix(byte[] bytes) {
        StringBuilder sb = new StringBuilder("0x");
        for (int i = 0; i < Math.min(4, bytes.length); i++) {
            sb.append(String.format("%02x", bytes[i] & 0xFF));
        }
        return sb.toString();
    }

    // ================================================================== javax.sound.sampled path

    private static AudioClip decodeJavaSound(byte[] bytes, String file, String kind) throws IOException {
        AudioFormat format;
        byte[] pcm;
        try (AudioInputStream in = AudioSystem.getAudioInputStream(new ByteArrayInputStream(bytes))) {
            format = in.getFormat();
            pcm = readAll(in, file, MAX_SAMPLES * 8L);
        } catch (UnsupportedAudioFileException e) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' is a " + kind
                    + " file whose codec javax.sound.sampled cannot decode (" + e.getMessage() + ")");
        } catch (IOException e) {
            throw new IOException("audio input '" + file + "' is a " + kind + " file that failed to decode: "
                    + e.getMessage(), e);
        }
        if (format == null || pcm.length == 0) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' holds no audio frames");
        }
        if (format.getSampleRate() <= 0) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' has no sample rate (" + format
                    + ")");
        }
        int frameSize = format.getFrameSize();
        if (frameSize <= 0 || pcm.length % frameSize != 0) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' has a truncated frame ("
                    + pcm.length + " bytes, frame size " + frameSize + ")");
        }
        int frames = pcm.length / frameSize;
        int channels = format.getChannels();
        int width = frameSize / channels;
        float[] samples = new float[frames];
        for (int f = 0; f < frames; f++) {
            float sum = 0.0f;
            for (int c = 0; c < channels; c++) {
                sum += sampleOf(pcm, f * frameSize + c * width, width, format.isBigEndian(),
                        format.getEncoding() == AudioFormat.Encoding.PCM_FLOAT);
            }
            samples[f] = sum / channels;
        }
        return new AudioClip(samples, Math.round(format.getSampleRate()));
    }

    /** One PCM sample in {@code [-1, 1)} from the container's encoding. */
    private static float sampleOf(byte[] pcm, int off, int width, boolean bigEndian, boolean float32) {
        if (float32 && width == 4) {
            int bits = (pcm[off] & 0xFF) | (pcm[off + 1] & 0xFF) << 8 | (pcm[off + 2] & 0xFF) << 16
                    | (pcm[off + 3]) << 24;
            return Float.intBitsToFloat(bits);
        }
        if (width == 1) {
            return ((pcm[off] & 0xFF) - 128) / 128.0f;
        }
        int v;
        if (bigEndian) {
            v = pcm[off] & 0xFF;
            for (int i = 1; i < width; i++) {
                v = (v << 8) | (pcm[off + i] & 0xFF);
            }
        } else {
            v = 0;
            for (int i = width - 1; i >= 0; i--) {
                v = (v << 8) | (pcm[off + i] & 0xFF);
            }
        }
        // move the two's-complement sign byte to the int's sign position and back
        v <<= 8 * (4 - width);
        v >>= 8 * (4 - width);
        return v / (float) (1L << (8 * width - 1));
    }

    private static byte[] readAll(InputStream in, String file, long maxBytes) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[64 * 1024];
        while (true) {
            int n = in.read(buf);
            if (n < 0) {
                break;
            }
            if (out.size() + n > maxBytes) {
                throw new IOException("audio input '" + file + "' exceeds the decode limit of " + maxBytes
                        + " PCM bytes");
            }
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }

    // ================================================================== ffmpeg path

    /**
     * The arguments of one guarded ffmpeg decode: the client's bytes on standard input, read through the pipe
     * protocol and the audio demuxers only, then mono float32 on standard output.
     */
    static List<String> ffmpegCommand() {
        List<String> command = new ArrayList<>(List.of("ffmpeg", "-hide_banner"));
        command.addAll(FfmpegInput.ffmpegStdin(FfmpegInput.Media.AUDIO));
        command.addAll(List.of("-vn", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"));
        return command;
    }

    private static AudioClip decodeFfmpeg(byte[] bytes, String file, String kind) throws IOException {
        if (!ffmpegAvailable()) {
            throw new UnsupportedAudioFormatException("audio input '" + file + "' is " + kind + ", which needs"
                    + " the 'ffmpeg' binary on the PATH (none was found); convert it to WAV or PCM to decode"
                    + " without ffmpeg");
        }
        Process p;
        try {
            // The banner on stderr carries the stream's rate; stdout carries mono float32 at that rate.
            p = new ProcessBuilder(ffmpegCommand()).redirectErrorStream(false).start();
        } catch (IOException e) {
            throw new IOException("audio input '" + file + "': ffmpeg failed to start: " + e.getMessage(), e);
        }
        try {
            Thread writer = writeInput(p, bytes);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[64 * 1024];
            try (InputStream stdout = p.getInputStream()) {
                while (true) {
                    int n = stdout.read(buf);
                    if (n < 0) {
                        break;
                    }
                    if (out.size() + n > (long) MAX_SAMPLES * 4) {
                        throw new IOException("audio input '" + file + "' exceeds " + MAX_SAMPLES + " samples");
                    }
                    out.write(buf, 0, n);
                }
            }
            join(writer, file);
            String banner = drain(p.getErrorStream());
            if (!p.waitFor(60, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                throw new IOException("audio input '" + file + "': ffmpeg timed out");
            }
            if (p.exitValue() != 0) {
                throw new IOException("audio input '" + file + "' is " + kind + " and ffmpeg exit code "
                        + p.exitValue() + " rejected it" + (banner.isEmpty() ? "" : (": " + tail(banner))));
            }
            int rate = rateOf(banner, file, kind);
            byte[] pcm = out.toByteArray();
            if (pcm.length < 4) {
                throw new UnsupportedAudioFormatException("audio input '" + file + "' holds no audio frames");
            }
            if (pcm.length % 4 != 0) {
                throw new IOException("audio input '" + file + "': ffmpeg returned a truncated frame ("
                        + pcm.length + " bytes)");
            }
            float[] samples = new float[pcm.length / 4];
            ByteBuffer bb = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < samples.length; i++) {
                samples[i] = bb.getFloat();
            }
            return new AudioClip(samples, rate);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("audio input '" + file + "': interrupted while ffmpeg ran", e);
        } finally {
            p.destroyForcibly();
        }
    }

    private static Thread writeInput(Process p, byte[] bytes) {
        Thread t = new Thread(() -> {
            try (OutputStream in = p.getOutputStream()) {
                in.write(bytes);
            } catch (IOException ignored) {
                // The reader may have closed the pipe after an error; the exit code carries the failure.
            }
        }, "audio-decoder-writer");
        t.setDaemon(true);
        t.start();
        return t;
    }

    private static void join(Thread writer, String file) throws IOException, InterruptedException {
        writer.join(30_000);
        if (writer.isAlive()) {
            throw new IOException("audio input '" + file + "': writing to ffmpeg stalled");
        }
    }

    private static int rateOf(String banner, String file, String kind) throws UnsupportedAudioFormatException {
        Matcher m = RATE_IN_BANNER.matcher(banner);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        throw new UnsupportedAudioFormatException("audio input '" + file + "' is " + kind + " and ffmpeg reported"
                + " no sample rate for it (no audio stream)");
    }

    private static String tail(String s) {
        String trimmed = s.trim();
        return trimmed.length() > 400 ? trimmed.substring(trimmed.length() - 400) : trimmed;
    }

    private static String drain(InputStream in) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            while (true) {
                int n = in.read(buf);
                if (n < 0) {
                    break;
                }
                out.write(buf, 0, n);
            }
            return out.toString(StandardCharsets.UTF_8.name());
        } catch (IOException e) {
            return "";
        }
    }

    private static Boolean ffmpegPresent;

    /** Whether {@code ffmpeg -version} succeeds; probed once per process. */
    static synchronized boolean ffmpegAvailable() {
        if (ffmpegPresent == null) {
            try {
                Process p = new ProcessBuilder("ffmpeg", "-nostdin", "-version").redirectErrorStream(true).start();
                if (!p.waitFor(10, TimeUnit.SECONDS)) {
                    p.destroyForcibly();
                    ffmpegPresent = false;
                } else {
                    ffmpegPresent = p.exitValue() == 0;
                }
            } catch (IOException e) {
                ffmpegPresent = false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                ffmpegPresent = false;
            }
        }
        return ffmpegPresent;
    }
}
