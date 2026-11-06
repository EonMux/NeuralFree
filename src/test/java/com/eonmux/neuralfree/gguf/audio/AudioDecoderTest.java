package com.eonmux.neuralfree.gguf.audio;

import org.junit.Assume;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The WAV/PCM decode path (values, mono mixing, sample widths) and the refusals of unsupported containers and
 * broken inputs, named by container.
 */
public class AudioDecoderTest {

    /** RIFF WAV bytes of a PCM16 clip. */
    private static byte[] wavPcm16(int[] samples, int channels, int rate) {
        int dataBytes = samples.length * 2;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(36 + dataBytes);
        bb.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        bb.put("fmt ".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(16);
        bb.putShort((short) 1); // PCM
        bb.putShort((short) channels);
        bb.putInt(rate);
        bb.putInt(rate * channels * 2);
        bb.putShort((short) (channels * 2));
        bb.putShort((short) 16);
        bb.put("data".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(dataBytes);
        for (int s : samples) {
            bb.putShort((short) s);
        }
        return bb.array();
    }

    @Test
    public void decodesPcm16MonoValues() throws IOException {
        int[] raw = { 0, 16384, -16384, 32767, -32768 };
        AudioClip clip = AudioDecoder.decode(wavPcm16(raw, 1, 16000), "tone.wav");
        assertEquals(16000, clip.sampleRate());
        assertEquals(raw.length, clip.length());
        assertEquals(0.0f, clip.samples()[0], 1e-9);
        assertEquals(16384 / 32768.0f, clip.samples()[1], 1e-6);
        assertEquals(-16384 / 32768.0f, clip.samples()[2], 1e-6);
        assertEquals(32767 / 32768.0f, clip.samples()[3], 1e-6);
        assertEquals(-1.0f, clip.samples()[4], 1e-6);
    }

    @Test
    public void mixesStereoToMono() throws IOException {
        int[] raw = { 16384, -16384, 8192, 8192 };
        AudioClip clip = AudioDecoder.decode(wavPcm16(raw, 2, 8000), "st.wav");
        assertEquals(8000, clip.sampleRate());
        assertEquals(2, clip.length());
        assertEquals(0.0f, clip.samples()[0], 1e-6);
        assertEquals(8192 / 32768.0f, clip.samples()[1], 1e-6);
    }

    @Test
    public void refusesUnknownContainerByName() {
        byte[] ogg = new byte[64];
        System.arraycopy("OggS".getBytes(StandardCharsets.US_ASCII), 0, ogg, 0, 4);
        // OGG is recognized but decodes only through ffmpeg: without the binary the refusal is a named
        // unsupported format, with it the fake payload is a decode failure, both naming the file
        Exception e = assertThrows(Exception.class, () -> AudioDecoder.decode(ogg, "song.ogg"));
        assertTrue(e.getMessage(), e.getMessage().contains("song.ogg"));
        byte[] junk = { 1, 2, 3, 4, 5 };
        UnsupportedAudioFormatException e2 = assertThrows(UnsupportedAudioFormatException.class,
                () -> AudioDecoder.decode(junk, "noise.bin"));
        assertTrue(e2.getMessage(), e2.getMessage().contains("0x01020304"));
    }

    @Test
    public void refusesEmptyInput() {
        UnsupportedAudioFormatException e = assertThrows(UnsupportedAudioFormatException.class,
                () -> AudioDecoder.decode(new byte[0], "empty.wav"));
        assertTrue(e.getMessage(), e.getMessage().contains("empty"));
        assertThrows(UnsupportedAudioFormatException.class, () -> AudioDecoder.decode(null, "null.wav"));
    }

    @Test
    public void refusesBrokenWavByName() {
        byte[] wav = wavPcm16(new int[] { 100, -100 }, 1, 16000);
        byte[] cut = new byte[30];
        System.arraycopy(wav, 0, cut, 0, cut.length);
        // a file cut inside the header is not decodable; the failure names the file
        Exception e = assertThrows(Exception.class, () -> AudioDecoder.decode(cut, "cut.wav"));
        assertTrue(e.getMessage(), e.getMessage().contains("cut.wav"));
    }

    @Test
    public void detectsContainersByMagic() {
        assertEquals("RIFF_WAVE", AudioDecoder.sniff(wavPcm16(new int[] { 0 }, 1, 8000)));
        byte[] mp3 = new byte[16];
        mp3[0] = (byte) 0xFF;
        mp3[1] = (byte) 0xFB;
        assertEquals("MP3", AudioDecoder.sniff(mp3));
        byte[] id3 = new byte[16];
        System.arraycopy("ID3".getBytes(StandardCharsets.US_ASCII), 0, id3, 0, 3);
        assertEquals("MP3", AudioDecoder.sniff(id3));
        byte[] flac = new byte[16];
        System.arraycopy("fLaC".getBytes(StandardCharsets.US_ASCII), 0, flac, 0, 4);
        assertEquals("FLAC", AudioDecoder.sniff(flac));
        byte[] ogg = new byte[16];
        System.arraycopy("OggS".getBytes(StandardCharsets.US_ASCII), 0, ogg, 0, 4);
        assertEquals("OGG", AudioDecoder.sniff(ogg));
        byte[] mp4 = new byte[16];
        System.arraycopy("ftyp".getBytes(StandardCharsets.US_ASCII), 0, mp4, 4, 4);
        assertEquals("M4A", AudioDecoder.sniff(mp4));
        assertEquals(null, AudioDecoder.sniff(new byte[] { 9, 9, 9, 9 }));
    }

    @Test
    public void decodesPcmFloat32Mono() throws IOException {
        int frames = 5;
        int dataBytes = frames * 4;
        ByteBuffer bb = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        bb.put("RIFF".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(36 + dataBytes);
        bb.put("WAVE".getBytes(StandardCharsets.US_ASCII));
        bb.put("fmt ".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(16);
        bb.putShort((short) 3); // IEEE float
        bb.putShort((short) 1);
        bb.putInt(16000);
        bb.putInt(16000 * 4);
        bb.putShort((short) 4);
        bb.putShort((short) 32);
        bb.put("data".getBytes(StandardCharsets.US_ASCII));
        bb.putInt(dataBytes);
        float[] values = { 0.0f, 0.5f, -0.25f, 1.0f, -1.0f };
        for (float v : values) {
            bb.putFloat(v);
        }
        AudioClip clip = AudioDecoder.decode(bb.array(), "float.wav");
        assertEquals(5, clip.length());
        for (int i = 0; i < values.length; i++) {
            assertEquals(values[i], clip.samples()[i], 1e-6);
        }
    }

    @Test
    public void silenceRoundTrip() throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int[] zeros = new int[160];
        out.write(wavPcm16(zeros, 1, 16000));
        AudioClip clip = AudioDecoder.decode(out.toByteArray(), "silence.wav");
        assertEquals(160, clip.length());
        for (int i = 0; i < clip.length(); i++) {
            assertEquals(0.0f, clip.samples()[i], 1e-9);
        }
    }

    @Test
    public void theFfmpegDecodeIsGuardedAtItsInput() {
        assertEquals(java.util.Arrays.asList(
                "ffmpeg", "-hide_banner",
                "-protocol_whitelist", "pipe",
                "-format_whitelist", "wav,aiff,au,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm",
                "-i", "pipe:0",
                "-vn", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"),
                AudioDecoder.ffmpegCommand());
    }

    @Test
    public void everyFfmpegContainerDecodesThroughTheGuardedPipe() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", AudioDecoder.ffmpegAvailable());
        Path dir = Files.createTempDirectory("nf-audio-formats-");
        try {
            String[][] formats = {{"mp3", "libmp3lame"}, {"flac", "flac"}, {"ogg", "libvorbis"},
                    {"m4a", "aac"}};
            for (String[] format : formats) {
                Path file = dir.resolve("tone." + format[0]);
                encode(file, format[1]);
                AudioClip clip = AudioDecoder.decode(Files.readAllBytes(file), file.getFileName().toString());
                assertEquals(format[0], 16000, clip.sampleRate());
                assertTrue(format[0], clip.length() > 0);
            }
        } finally {
            if (dir.toFile().listFiles() != null) {
                for (java.io.File file : dir.toFile().listFiles()) {
                    Files.deleteIfExists(file.toPath());
                }
            }
            Files.deleteIfExists(dir);
        }
    }

    @Test
    public void playlistsAndConcatScriptsAreRefusedByName() {
        String hls = "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\nfile:///etc/passwd\n#EXT-X-ENDLIST\n";
        String concat = "ffconcat version 1.0\nsafe 0\nfile '/etc/passwd'\n";
        for (String script : new String[]{hls, concat}) {
            UnsupportedAudioFormatException e = assertThrows(UnsupportedAudioFormatException.class,
                    () -> AudioDecoder.decode(script.getBytes(StandardCharsets.US_ASCII), "upload.m3u8"));
            assertTrue(e.getMessage(), e.getMessage().contains("upload.m3u8"));
            assertTrue(e.getMessage(), e.getMessage().contains("not a supported container"));
        }
    }

    /** Encodes a 0.2 s 440 Hz mono tone as {@code codec} into {@code file}. */
    private static void encode(Path file, String codec) throws IOException, InterruptedException {
        Process p = new ProcessBuilder("ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=16000", "-t", "0.2", "-ac", "1", "-c:a", codec,
                file.toString()).redirectErrorStream(false).start();
        byte[] buf = new byte[4096];
        try (InputStream err = p.getErrorStream()) {
            while (err.read(buf) >= 0) {
                // the encode's log drains away
            }
        }
        assertTrue("ffmpeg failed to encode " + codec, p.waitFor(60, TimeUnit.SECONDS));
        assertEquals("ffmpeg failed to encode " + codec, 0, p.exitValue());
        assertTrue(Files.size(file) > 0);
    }
}
