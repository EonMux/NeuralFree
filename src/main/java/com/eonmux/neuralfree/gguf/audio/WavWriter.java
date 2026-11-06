package com.eonmux.neuralfree.gguf.audio;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * RIFF WAVE writer of 16-bit PCM mono, byte-compatible with llama.cpp's TTS output
 * (tools/mtmd/mtmd-helper-gen.cpp:61-87): a 44-byte canonical header ({@code RIFF/WAVE}, one
 * {@code fmt } chunk with format 1 PCM, one channel, the sample rate, byte rate {@code 2 * rate}, block
 * align 2, 16 bits, and a {@code data} chunk), then each sample as the little-endian {@code int16} of
 * {@code clamp(value, -1, 1) * 32767}.
 *
 * <p>The one-shot form sizes its chunks exactly and refuses audio whose byte size does not fit the 32-bit
 * RIFF fields (more than {@code (2^32 - 36) / 2} samples), like {@code write_wav16}. The streaming form
 * emits the header once with the {@code 0xFFFFFFFF} unknown-length sentinel in both size fields — the
 * ffmpeg-on-a-pipe convention the reference streams with — followed by one {@code appendChunk} per PCM run;
 * a streamed file is only valid once the writer closes it, and callers that cannot patch the header keep the
 * sentinel.
 */
public final class WavWriter {

    /** Bytes of the canonical 44-byte header. */
    public static final int HEADER_BYTES = 44;
    /** Largest sample count a one-shot WAV can hold: the 32-bit RIFF size fields minus the 36 header bytes. */
    public static final long MAX_SAMPLES = 0xFFFFFFFFL - 36 < 0 ? 0 : (0xFFFFFFFFL - 36) / 2;

    private WavWriter() {
    }

    /**
     * The complete WAV file of {@code pcm} at {@code sampleRate} Hz.
     *
     * @throws IllegalArgumentException when the rate is not positive or the audio exceeds {@link #MAX_SAMPLES}
     */
    public static byte[] bytes(float[] pcm, int sampleRate) {
        requireRate(sampleRate);
        Objects.requireNonNull(pcm, "pcm");
        if (pcm.length > MAX_SAMPLES) {
            throw new IllegalArgumentException(pcm.length + " samples exceed the 32-bit RIFF limit of "
                    + MAX_SAMPLES + " (mtmd-helper-gen.cpp:82)");
        }
        ByteBuffer out = ByteBuffer.allocate(HEADER_BYTES + 2 * pcm.length).order(ByteOrder.LITTLE_ENDIAN);
        header(out, pcm.length * 2, sampleRate);
        for (float v : pcm) {
            out.putShort(sample(v));
        }
        return out.array();
    }

    /** Writes {@code bytes(pcm, sampleRate)} to {@code path}. */
    public static void write(Path path, float[] pcm, int sampleRate) throws IOException {
        Objects.requireNonNull(path, "path");
        Files.write(path, bytes(pcm, sampleRate));
    }

    /**
     * The streaming header: the canonical 44 bytes with the {@code 0xFFFFFFFF} unknown-length sentinel in the
     * RIFF and data size fields, as llama.cpp's streamed TTS writes it ({@code write_wav16_header(buf,
     * UINT32_MAX, rate)}).
     */
    public static byte[] streamingHeader(int sampleRate) {
        requireRate(sampleRate);
        ByteBuffer out = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header(out, 0xFFFFFFFF, sampleRate);
        return out.array();
    }

    /**
     * One streamed PCM run: {@code clamp(v, -1, 1) * 32767} per sample as little-endian {@code int16}
     * ({@code append_wav16_pcm}).
     */
    public static byte[] pcmChunk(float[] pcm, int from, int length) {
        Objects.requireNonNull(pcm, "pcm");
        if (from < 0 || length < 0 || (long) from + length > pcm.length) {
            throw new IllegalArgumentException("chunk [" + from + ", +" + length + ") is outside "
                    + pcm.length + " samples");
        }
        ByteBuffer out = ByteBuffer.allocate(2 * length).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < length; i++) {
            out.putShort(sample(pcm[from + i]));
        }
        return out.array();
    }

    /** Reads the data-chunk sample count of a canonical or sentinel WAV header, {@code -1} for the sentinel. */
    public static int dataSamples(byte[] wav) {
        Objects.requireNonNull(wav, "wav");
        if (wav.length < HEADER_BYTES) {
            throw new IllegalArgumentException("a WAV needs at least " + HEADER_BYTES + " bytes, got " + wav.length);
        }
        ByteBuffer in = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        if (!Arrays.equals(Arrays.copyOfRange(wav, 0, 4), new byte[]{'R', 'I', 'F', 'F'})
                || !Arrays.equals(Arrays.copyOfRange(wav, 8, 12), new byte[]{'W', 'A', 'V', 'E'})) {
            throw new IllegalArgumentException("not a RIFF/WAVE header");
        }
        int dataSz = in.getInt(40);
        return dataSz == 0xFFFFFFFF ? -1 : dataSz / 2;
    }

    private static void header(ByteBuffer out, int dataSz, int sampleRate) {
        int riffSz = dataSz == 0xFFFFFFFF ? 0xFFFFFFFF : 36 + dataSz;
        out.put("RIFF".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        out.putInt(riffSz);
        out.put("WAVE".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        out.put("fmt ".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        out.putInt(16);
        out.putShort((short) 1);
        out.putShort((short) 1);
        out.putInt(sampleRate);
        out.putInt(2 * sampleRate);
        out.putShort((short) 2);
        out.putShort((short) 16);
        out.put("data".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        out.putInt(dataSz);
    }

    private static short sample(float v) {
        float clamped = Math.max(-1.0f, Math.min(1.0f, v));
        return (short) (clamped * 32767.0f);
    }

    private static void requireRate(int sampleRate) {
        if (sampleRate <= 0 || sampleRate > 0xFFFFFFFFL / 2) {
            throw new IllegalArgumentException("sample rate must be positive and fit the byte-rate field, got "
                    + sampleRate);
        }
    }
}
