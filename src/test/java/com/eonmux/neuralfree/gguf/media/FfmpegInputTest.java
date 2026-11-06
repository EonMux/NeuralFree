package com.eonmux.neuralfree.gguf.media;

import org.junit.Assume;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * The input arguments every ffmpeg and ffprobe run over client-supplied media is built from: the protocol
 * whitelist of the input transport, the demuxer whitelist of the media kind, and the option that keeps stdin from
 * being read when the bytes come from a staged file. A real ffmpeg must leave a piped HLS playlist and a piped
 * ffconcat script undecoded.
 */
public class FfmpegInputTest {

    private static final String VIDEO_DEMUXERS =
            "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,flv,mpegts";
    private static final String AUDIO_DEMUXERS =
            "wav,aiff,au,mp3,flac,ogg,mov,mp4,m4a,3gp,3g2,mj2,matroska,webm";
    private static final String IMAGE_DEMUXERS = "webp_pipe,webp_anim,image2,image2pipe";

    private static final Path CLIP = Paths.get("clip.mp4");

    @Test
    public void anFfmpegRunOverAStagedFileAddsNoStdinAndTheFileProtocol() {
        assertEquals(Arrays.asList(
                "-nostdin",
                "-protocol_whitelist", "file",
                "-format_whitelist", VIDEO_DEMUXERS,
                "-i", CLIP.toString()),
                FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO, CLIP));
    }

    @Test
    public void anFfprobeRunOverAStagedFileOmitsTheNoStdinOptionItDoesNotHave() {
        assertEquals(Arrays.asList(
                "-protocol_whitelist", "file",
                "-format_whitelist", VIDEO_DEMUXERS,
                "-i", CLIP.toString()),
                FfmpegInput.ffprobeFile(FfmpegInput.Media.VIDEO, CLIP));
    }

    @Test
    public void anFfmpegRunOverStdinReadsThroughThePipeProtocol() {
        assertEquals(Arrays.asList(
                "-protocol_whitelist", "pipe",
                "-format_whitelist", AUDIO_DEMUXERS,
                "-i", "pipe:0"),
                FfmpegInput.ffmpegStdin(FfmpegInput.Media.AUDIO));
    }

    @Test
    public void imagesUseTheWebpAndImagePipeDemuxers() {
        assertEquals(Arrays.asList(
                "-protocol_whitelist", "pipe",
                "-format_whitelist", IMAGE_DEMUXERS,
                "-i", "pipe:0"),
                FfmpegInput.ffmpegStdin(FfmpegInput.Media.IMAGE));
    }

    @Test
    public void theAudioWhitelistKeepsEveryContainerTheDecoderForwardsToFfmpeg() {
        List<String> formats = Arrays.asList(AUDIO_DEMUXERS.split(","));
        for (String container : new String[]{"wav", "aiff", "au", "mp3", "flac", "ogg",
                "mov", "mp4", "m4a", "3gp", "3g2", "matroska", "webm"}) {
            assertTrue(container, formats.contains(container));
        }
        for (String indirect : new String[]{"hls", "concat", "image2"}) {
            assertFalse(indirect, formats.contains(indirect));
        }
    }

    @Test
    public void theImageWhitelistCarriesOnlyStillImageDemuxers() {
        List<String> formats = Arrays.asList(IMAGE_DEMUXERS.split(","));
        assertTrue(formats.contains("webp_pipe"));
        assertTrue(formats.contains("image2pipe"));
        for (String indirect : new String[]{"hls", "concat", "mov", "wav"}) {
            assertFalse(indirect, formats.contains(indirect));
        }
    }

    @Test
    public void aPipedHlsPlaylistAndConcatScriptDecodeNothing() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        Path dir = Files.createTempDirectory("nf-ffmpeg-guard-");
        try {
            Path secret = dir.resolve("secret.wav");
            encodeTone(secret);
            String hls = "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\nfile://" + secret
                    + "\n#EXT-X-ENDLIST\n";
            String concat = "ffconcat version 1.0\nsafe 0\nfile '" + secret + "'\n";
            for (String script : new String[]{hls, concat}) {
                Result result = runGuardedAudio(script.getBytes(StandardCharsets.US_ASCII));
                assertNotEquals("the guard must refuse: " + script, 0, result.exit);
                assertEquals("no audio may come out of: " + script, 0, result.stdout.length);
                assertFalse("ffmpeg must report why it refused", result.stderr.trim().isEmpty());
            }
        } finally {
            deleteTree(dir);
        }
    }

    @Test
    public void aStagedConcatScriptIsRefusedByTheFormatWhitelist() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        Path dir = Files.createTempDirectory("nf-ffmpeg-staged-guard-");
        try {
            Path secret = dir.resolve("secret.wav");
            encodeTone(secret);
            Path script = dir.resolve("script.ffconcat");
            Files.write(script, ("ffconcat version 1.0\nsafe 0\nfile '" + secret + "'\n")
                    .getBytes(StandardCharsets.US_ASCII));
            Result result = runGuardedAudioFile(script);
            assertNotEquals("a concat script posing as audio must be refused", 0, result.exit);
            assertEquals(0, result.stdout.length);
            assertTrue(result.stderr.contains("whitelist"));
        } finally {
            deleteTree(dir);
        }
    }

    private static boolean ffmpegOnPath() {
        try {
            Process p = new ProcessBuilder("ffmpeg", "-version").redirectErrorStream(false).start();
            p.waitFor(30, TimeUnit.SECONDS);
            return p.exitValue() == 0;
        } catch (IOException e) {
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private static void encodeTone(Path file) throws IOException, InterruptedException {
        Process p = new ProcessBuilder("ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", "sine=frequency=440:sample_rate=16000", "-t", "0.2", "-ac", "1", file.toString())
                .redirectErrorStream(false).start();
        drainQuietly(p.getErrorStream());
        assertTrue("ffmpeg failed to encode the fixture", p.waitFor(60, TimeUnit.SECONDS));
        assertEquals(0, p.exitValue());
    }

    /** Runs the guarded audio pipe over {@code stdin} exactly as {@code AudioDecoder} does. */
    private static Result runGuardedAudio(byte[] stdin) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>(List.of("ffmpeg", "-hide_banner"));
        command.addAll(FfmpegInput.ffmpegStdin(FfmpegInput.Media.AUDIO));
        command.addAll(List.of("-vn", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"));
        return run(command, stdin);
    }

    /** Runs the guarded audio decode over a staged file. */
    private static Result runGuardedAudioFile(Path file) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>(List.of("ffmpeg", "-hide_banner"));
        command.addAll(FfmpegInput.ffmpegFile(FfmpegInput.Media.AUDIO, file));
        command.addAll(List.of("-vn", "-ac", "1", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"));
        return run(command, null);
    }

    private static Result run(List<String> command, byte[] stdin) throws IOException, InterruptedException {
        Process process = new ProcessBuilder(command).start();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();
        Thread drain = new Thread(() -> {
            try (InputStream in = process.getErrorStream()) {
                copy(in, stderr);
            } catch (IOException ignored) {
                // the exit code carries the failure
            }
        }, "ffmpeg-guard-test-stderr");
        drain.setDaemon(true);
        drain.start();
        if (stdin != null) {
            try (OutputStream in = process.getOutputStream()) {
                in.write(stdin);
            } catch (IOException ignored) {
                // ffmpeg may reject the input before the pipe is written
            }
        } else {
            process.getOutputStream().close();
        }
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        try (InputStream in = process.getInputStream()) {
            copy(in, stdout);
        }
        assertTrue("ffmpeg did not finish", process.waitFor(60, TimeUnit.SECONDS));
        drain.join(10_000);
        return new Result(process.exitValue(), stdout.toByteArray(), stderr.toString(StandardCharsets.UTF_8));
    }

    private static void copy(InputStream in, ByteArrayOutputStream out) throws IOException {
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
    }

    private static void drainQuietly(InputStream in) throws IOException {
        byte[] buf = new byte[4096];
        while (in.read(buf) >= 0) {
            // the encode's log drains away
        }
        in.close();
    }

    private static void deleteTree(Path dir) throws IOException {
        Files.deleteIfExists(dir.resolve("secret.wav"));
        Files.deleteIfExists(dir.resolve("script.ffconcat"));
        Files.deleteIfExists(dir);
    }

    /** The exit code, stdout and stderr of one ffmpeg run. */
    private static final class Result {
        private final int exit;
        private final byte[] stdout;
        private final String stderr;

        private Result(int exit, byte[] stdout, String stderr) {
            this.exit = exit;
            this.stdout = stdout;
            this.stderr = stderr;
        }
    }
}
