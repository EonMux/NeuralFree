package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/** The WAV bytes of llama.cpp's {@code write_wav16} (mtmd-helper-gen.cpp:61-87): header, samples, refusals. */
public class WavWriterTest {

    @Test
    public void theHeaderIsTheCanonical44ByteForm() {
        byte[] wav = WavWriter.bytes(new float[] {0f}, 24000);
        assertEquals(46, wav.length);
        assertEquals("RIFF", ascii(wav, 0, 4));
        assertEquals(36 + 2, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(4));
        assertEquals("WAVE", ascii(wav, 8, 4));
        assertEquals("fmt ", ascii(wav, 12, 4));
        assertEquals(16, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(16));
        assertEquals(1, shortAt(wav, 20)); // PCM
        assertEquals(1, shortAt(wav, 22)); // mono
        assertEquals(24000, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(24));
        assertEquals(48000, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(28));
        assertEquals(2, shortAt(wav, 32));
        assertEquals(16, shortAt(wav, 34));
        assertEquals("data", ascii(wav, 36, 4));
        assertEquals(2, ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getInt(40));
    }

    @Test
    public void samplesConvertWithTheReferenceClamp() {
        byte[] wav = WavWriter.bytes(new float[] {0f, 0.5f, -1.5f, 1f, -1f}, 8000);
        short[] expected = {0, 16383, -32767, 32767, -32767};
        for (int i = 0; i < expected.length; i++) {
            assertEquals("sample " + i, expected[i], shortAt(wav, 44 + 2 * i));
        }
    }

    @Test
    public void theStreamingHeaderCarriesTheSentinel() {
        byte[] header = WavWriter.streamingHeader(32000);
        assertEquals(44, header.length);
        ByteBuffer in = ByteBuffer.wrap(header).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0xFFFFFFFF, in.getInt(4));
        assertEquals(0xFFFFFFFF, in.getInt(40));
        assertEquals(-1, WavWriter.dataSamples(header));
        byte[] chunk = WavWriter.pcmChunk(new float[] {0.25f, -0.25f}, 0, 2);
        assertEquals(4, chunk.length);
        assertEquals(8191, shortAt(chunk, 0));
        assertEquals(-8191, shortAt(chunk, 2));
    }

    @Test
    public void aFileRoundTripsItsOwnHeader() throws Exception {
        float[] pcm = new float[256];
        Arrays.fill(pcm, 0.125f);
        Path file = Files.createTempFile("nf-tts", ".wav");
        WavWriter.write(file, pcm, 16000);
        byte[] wav = Files.readAllBytes(file);
        assertEquals(256, WavWriter.dataSamples(wav));
        assertArrayEquals(WavWriter.bytes(pcm, 16000), wav);
        Files.deleteIfExists(file);
    }

    @Test
    public void refusalsNameTheLimit() {
        assertThrows(IllegalArgumentException.class, () -> WavWriter.bytes(new float[1], 0));
        assertThrows(IllegalArgumentException.class, () -> WavWriter.pcmChunk(new float[3], 2, 2));
        assertThrows(IllegalArgumentException.class, () -> WavWriter.dataSamples(new byte[10]));
        assertThrows(NullPointerException.class, () -> WavWriter.bytes(null, 24000));
    }

    private static String ascii(byte[] wav, int from, int len) {
        return new String(wav, from, len, java.nio.charset.StandardCharsets.US_ASCII);
    }

    private static int shortAt(byte[] wav, int off) {
        return ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN).getShort(off);
    }
}
