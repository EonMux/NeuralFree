package com.eonmux.neuralfree.gguf.video;

import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.media.FfmpegInput;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;
import org.junit.Assume;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The video decoder: the container sniff that tells an untyped buffer's video from its audio and images, and the
 * probe and decode of a real MP4 through ffmpeg, sampled by a plan the way llama.cpp's helper reads frames.
 * Skipped when no ffmpeg is on the PATH.
 */
public class VideoDecoderTest {

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

    /** A 8x8 2 fps MP4 of 3 seconds (6 frames), encoded with the test's ffmpeg. */
    private static byte[] mp4() throws IOException, InterruptedException {
        return mp4(8, 8, 2, 3);
    }

    /** A {@code width}x{@code height} MP4 of {@code seconds} seconds at {@code rate} fps. */
    private static byte[] mp4(int width, int height, int rate, int seconds) throws IOException,
            InterruptedException {
        Path out = Files.createTempFile("nf-video-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc=size=" + width + "x" + height + ":rate=" + rate, "-t",
                    String.valueOf(seconds), "-pix_fmt", "yuv420p", out.toString())
                    .redirectErrorStream(false).start();
            drain(p.getErrorStream());
            assertTrue("ffmpeg failed to encode the fixture", p.waitFor(120, TimeUnit.SECONDS));
            assertEquals(0, p.exitValue());
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    /** A {@code width}x{@code height} MP4 of exactly {@code frames} frames at {@code rate} fps. */
    private static byte[] mp4Frames(int width, int height, int rate, int frames) throws IOException,
            InterruptedException {
        Path out = Files.createTempFile("nf-video-frames-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc=size=" + width + "x" + height + ":rate=" + rate, "-frames:v",
                    String.valueOf(frames), "-pix_fmt", "yuv420p", out.toString())
                    .redirectErrorStream(false).start();
            drain(p.getErrorStream());
            assertTrue("ffmpeg failed to encode the fixture", p.waitFor(120, TimeUnit.SECONDS));
            assertEquals(0, p.exitValue());
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    /** An 8x8 2 fps MP4 of 3 seconds (6 frames) with a 440 Hz mono AAC soundtrack at 16 kHz. */
    private static byte[] mp4WithAudio() throws IOException, InterruptedException {
        Path out = Files.createTempFile("nf-video-audio-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin",
                    "-f", "lavfi", "-i", "testsrc=size=8x8:rate=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "3", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", out.toString())
                    .redirectErrorStream(false).start();
            drain(p.getErrorStream());
            assertTrue("ffmpeg failed to encode the fixture", p.waitFor(120, TimeUnit.SECONDS));
            assertEquals(0, p.exitValue());
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    private static void drain(InputStream in) throws IOException {
        byte[] buf = new byte[4096];
        while (in.read(buf) > 0) {
            // the encode's log drains away
        }
        in.close();
    }

    @Test
    public void theContainerSniffTellsVideosFromAudioAndImages() {
        VideoDecoder decoder = new VideoDecoder();
        // MP4 whose major brand is a video one
        assertTrue(VideoDecoder.isVideo(bytes("....ftypisom........")));
        // MP4 whose major brand is an audio-only one: the audio sniff takes it
        assertFalse(VideoDecoder.isVideo(bytes("....ftypM4A .......")));
        // Matroska and WebM (EBML), AVI, FLV and MPEG-TS
        assertTrue(VideoDecoder.isVideo(bytes(new byte[] {(byte) 0x1A, 0x45, (byte) 0xDF, (byte) 0xA3, 1, 2})));
        assertTrue(VideoDecoder.isVideo(bytes("RIFF....AVI LIST")));
        assertTrue(VideoDecoder.isVideo(bytes("FLV....")));
        byte[] ts = new byte[189];
        ts[0] = 0x47;
        ts[188] = 0x47;
        assertTrue(VideoDecoder.isVideo(ts));
        // audio and image magic is no video
        assertFalse(VideoDecoder.isVideo(bytes("RIFF....WAVEfmt ")));
        assertFalse(VideoDecoder.isVideo(bytes(new byte[] {(byte) 0x89, 'P', 'N', 'G'})));
        assertFalse(VideoDecoder.isVideo(new byte[0]));
    }

    @Test
    public void theProbeReadsTheStreamMetadata() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoMeta meta = new VideoDecoder().probe(video, "video_url.url");
        assertEquals(8, meta.width());
        assertEquals(8, meta.height());
        assertEquals(2.0, meta.fps(), 0.01);
        assertEquals(6, meta.nFrames());
        assertEquals(3.0, meta.duration(), 0.01);
    }

    @Test
    public void theDecodeKeepsThePlannedFramesAtTheProbedSize() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoDecoder decoder = new VideoDecoder();
        VideoMeta meta = decoder.probe(video, "video_url.url");
        // 3 of the 6 frames: round(linspace(0, 5, 3)) = [0, 3, 5]... step 2.5 rounds to [0, 2, 5] half even
        VideoSamplePlan plan = VideoConfig.defaults().plan(meta, 1.0, null, 0, "video_url.url");
        assertEquals(3, plan.nFrames());
        assertArrayEquals(new int[] {0, 2, 5}, plan.indices());
        DecodedVideo decoded = decoder.decode(video, plan, "video_url.url");
        assertEquals(3, decoded.frames().size());
        for (RasterImage frame : decoded.frames()) {
            assertEquals(8, frame.width());
            assertEquals(8, frame.height());
        }
        // the frames differ over time (the test pattern moves)
        assertFalse(java.util.Arrays.equals(decoded.frames().get(0).rgb(), decoded.frames().get(1).rgb()));
        assertTrue(decoded.soundtrack().isEmpty());
    }

    @Test
    public void theQwen2VlFamilyDecodesThePatchFlooredFrames() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4Frames(8, 8, 2, 7);
        VideoDecoder decoder = new VideoDecoder();
        VideoMeta meta = decoder.probe(video, "video_url.url");
        assertEquals(7, meta.nFrames());
        // vLLM's Qwen2-VL loader: 7 frames at 2 fps give n = 7, floored to the 6-frame patch multiple, and
        // torch.arange(0, 7, 7 / 6) takes frames 0 to 5
        VideoSamplePlan plan = VideoConfig.forProjector(ProjectorType.QWEN2VL, false)
                .plan(meta, null, null, 0, "video_url.url");
        assertArrayEquals(new int[] {0, 1, 2, 3, 4, 5}, plan.indices());
        DecodedVideo decoded = decoder.decode(video, plan, "video_url.url");
        assertEquals(6, decoded.frames().size());
        for (RasterImage frame : decoded.frames()) {
            assertEquals(8, frame.width());
            assertEquals(8, frame.height());
        }
    }

    @Test
    public void theDecodeScalesFramesToTheRecommendedSizeLikeLlamaCpp() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4(64, 48, 2, 3);
        VideoDecoder decoder = new VideoDecoder();
        VideoMeta meta = decoder.probe(video, "video_url.url");
        VideoSamplePlan plan = VideoConfig.defaults().plan(meta, 1.0, null, 0, "video_url.url");
        ImageSize size = new ImageSize(16, 12);
        DecodedVideo decoded = decoder.decode(video, plan, size, 0, "video_url.url");
        assertEquals(3, decoded.frames().size());

        // the reference filter chain of mtmd_helper_video::start_ffmpeg over the same staged file
        byte[] raw = ffmpegStream(video, "format=rgb24,scale=16:12:flags=bicubic+accurate_rnd+full_chroma_int"
                + ":param0=0:param1=0.5", "6");
        int frameBytes = 16 * 12 * 3;
        assertEquals(6 * frameBytes, raw.length);
        int[] indices = plan.indices();
        for (int i = 0; i < indices.length; i++) {
            RasterImage frame = decoded.frames().get(i);
            assertEquals(16, frame.width());
            assertEquals(12, frame.height());
            assertArrayEquals(("frame " + indices[i]), slice(raw, indices[i] * frameBytes, frameBytes),
                    frame.rgb());
        }
    }

    @Test
    public void theSoundtrackSlicesFollowTheTemporalGroups() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4WithAudio();
        VideoDecoder decoder = new VideoDecoder();
        VideoMeta meta = decoder.probe(video, "video_url.url");
        assertEquals(0.0, meta.startTime(), 0.01);
        VideoSamplePlan plan = VideoConfig.forProjector(ProjectorType.LFM3VL, true)
                .plan(meta, null, null, 0, "video_url.url");
        assertEquals(6, plan.nFrames());
        assertEquals(3, plan.groups());
        DecodedVideo decoded = decoder.decode(video, plan, null, 16000, "video_url.url");
        assertEquals(3, decoded.soundtrack().size());

        // the reference reads the whole track at the tower's rate; each group takes its span
        float[] pcm = ffmpegPcm(video, 16000);
        int[] starts = {0, 16000, 32000};
        int end = Math.min(pcm.length, 48000);
        for (int g = 0; g < 3; g++) {
            int to = g == 2 ? end : starts[g + 1];
            float[] want = java.util.Arrays.copyOfRange(pcm, starts[g], Math.min(to, end));
            float[] got = decoded.soundtrack().get(g).samples();
            assertEquals("group " + g + " samples", want.length, got.length);
            assertArrayEquals("group " + g, want, got, 0f);
        }
    }

    @Test
    public void aVideoWithoutAnAudioTrackHasNoSoundtrack() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoDecoder decoder = new VideoDecoder();
        VideoMeta meta = decoder.probe(video, "video_url.url");
        VideoSamplePlan plan = VideoConfig.forProjector(ProjectorType.LFM3VL, true)
                .plan(meta, null, null, 0, "video_url.url");
        assertTrue(decoder.decode(video, plan, null, 16000, "video_url.url").soundtrack().isEmpty());
    }

    /** The raw bytes one ffmpeg run over {@code video} with {@code vf} and {@code frames} outputs. */
    private static byte[] ffmpegStream(byte[] video, String vf, String frames)
            throws IOException, InterruptedException {
        List<String> command = new ArrayList<>(List.of("ffmpeg"));
        command.addAll(FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO, stage(video)));
        command.addAll(List.of("-map", "0:v:0", "-vf", vf, "-frames:v", frames,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-loglevel", "error", "pipe:1"));
        return run(command);
    }

    /** The mono float32 samples of the fixture's audio track at {@code rate}, as the reference decodes them. */
    private static float[] ffmpegPcm(byte[] video, int rate) throws IOException, InterruptedException {
        List<String> command = new ArrayList<>(List.of("ffmpeg"));
        command.addAll(FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO, stage(video)));
        command.addAll(List.of("-map", "0:a:0", "-vn", "-ac", "1", "-ar",
                String.valueOf(rate), "-f", "f32le", "-loglevel", "error", "pipe:1"));
        byte[] raw = run(command);
        java.nio.FloatBuffer floats = java.nio.ByteBuffer.wrap(raw)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        float[] out = new float[floats.remaining()];
        floats.get(out);
        return out;
    }

    private static Path stage(byte[] video) throws IOException {
        Path file = Files.createTempFile("nf-video-reference-", ".mp4");
        Files.write(file, video);
        file.toFile().deleteOnExit();
        return file;
    }

    private static byte[] run(List<String> command) throws IOException, InterruptedException {
        Process p = new ProcessBuilder(command).redirectError(ProcessBuilder.Redirect.DISCARD).start();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (InputStream in = p.getInputStream()) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
        }
        assertTrue("ffmpeg failed", p.waitFor(120, TimeUnit.SECONDS));
        assertEquals(0, p.exitValue());
        return out.toByteArray();
    }

    private static byte[] slice(byte[] raw, int from, int length) {
        return java.util.Arrays.copyOfRange(raw, from, from + length);
    }

    @Test
    public void aNonVideoRefusesAtTheProbe() {
        VideoDecoder decoder = new VideoDecoder();
        MediaInputException e = assertThrows(MediaInputException.class,
                () -> decoder.probe("this is not a video".getBytes(StandardCharsets.US_ASCII), "video_url.url"));
        assertTrue(e.getMessage().contains("video_url.url"));
    }

    @Test
    public void everyRunReadsTheStagedFileThroughTheContainerDemuxersOnly() {
        List<String> guards = FfmpegInput.ffmpegFile(FfmpegInput.Media.VIDEO,
                java.nio.file.Paths.get("clip.mp4"));
        assertEquals("file", guards.get(guards.indexOf("-protocol_whitelist") + 1));
        String formats = guards.get(guards.indexOf("-format_whitelist") + 1);
        for (String container : new String[]{"mp4", "matroska", "webm", "avi", "flv", "mpegts"}) {
            assertTrue(container, formats.contains(container));
        }
        for (String indirect : new String[]{"hls", "concat", "image2"}) {
            assertFalse(indirect, java.util.Arrays.asList(formats.split(",")).contains(indirect));
        }
    }

    @Test
    public void aPlaylistOrConcatScriptCannotOpenAnotherFile() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        Path secret = Files.createTempFile("nf-video-secret-", ".mp4");
        try {
            Files.write(secret, mp4());
            String hls = "#EXTM3U\n#EXT-X-TARGETDURATION:3\n#EXTINF:3.0,\nfile://" + secret + "\n#EXT-X-ENDLIST\n";
            String concat = "ffconcat version 1.0\nsafe 0\nfile '" + secret + "'\n";
            VideoDecoder decoder = new VideoDecoder();
            for (String script : new String[]{hls, concat}) {
                MediaInputException e = assertThrows(MediaInputException.class,
                        () -> decoder.probe(bytes(script), "video_url.url"));
                assertTrue(e.getMessage().contains("video_url.url"));
            }
        } finally {
            Files.deleteIfExists(secret);
        }
    }

    @Test
    public void theSoundtrackBufferCountsTheStartOffsetAndSaturatesOnHugeMetadata() {
        assertEquals((16_000L + 32_000L) * Float.BYTES, VideoDecoder.bufferedBytes(16_000L, 32_000L));
        assertEquals(Long.MAX_VALUE, VideoDecoder.bufferedBytes(Long.MAX_VALUE, 1L));
        assertEquals(Long.MAX_VALUE, VideoDecoder.bufferedBytes(Long.MAX_VALUE / 2, Long.MAX_VALUE / 2));
        MediaInputException e = assertThrows(MediaInputException.class, () -> VideoDecoder.checkSoundtrackBytes(
                VideoDecoder.bufferedBytes(Long.MAX_VALUE, Long.MAX_VALUE), 64L << 20, "video_url.url"));
        assertTrue(e.getMessage().contains("video_url.url"));
    }

    @Test
    public void keptFramesAboveTheHeapShareAreRefusedByField() {
        ImageSize size = new ImageSize(1920, 1080);
        VideoDecoder.checkKeptBytes(64L << 20, 64L << 20, size, "video_url.url");
        MediaInputException e = assertThrows(MediaInputException.class,
                () -> VideoDecoder.checkKeptBytes((64L << 20) + 1, 64L << 20, size, "input_video.data"));
        assertTrue(e.getMessage().contains("input_video.data"));
        assertTrue(e.getMessage().contains("64 MiB limit"));
        assertTrue(e.getMessage().contains("1920x1080"));
        // the budget applies to the decoded size, not the source's
        MediaInputException scaled = assertThrows(MediaInputException.class,
                () -> VideoDecoder.checkKeptBytes(65L << 20, 64L << 20, new ImageSize(160, 90),
                        "video_url.url"));
        assertTrue(scaled.getMessage().contains("160x90"));
    }

    private static byte[] bytes(String text) {
        return text.getBytes(StandardCharsets.US_ASCII);
    }

    private static byte[] bytes(byte[] raw) {
        return raw;
    }
}
