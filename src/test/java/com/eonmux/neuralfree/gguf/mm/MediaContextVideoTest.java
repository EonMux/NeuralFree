package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.MediaChunk;
import com.eonmux.neuralfree.gguf.engine.MediaPrompt;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessor;
import com.eonmux.neuralfree.gguf.image.ImagePreprocessors;
import com.eonmux.neuralfree.gguf.image.ImageProcessorParams;
import com.eonmux.neuralfree.gguf.image.ImageSize;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.ImageTokenBudget;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.media.MediaIoOptions;
import com.eonmux.neuralfree.gguf.media.MediaLimits;
import com.eonmux.neuralfree.gguf.ops.rope.MediaRopePositions;
import com.eonmux.neuralfree.gguf.video.DecodedVideo;
import com.eonmux.neuralfree.gguf.video.VideoConfig;
import com.eonmux.neuralfree.gguf.video.VideoDecoder;
import com.eonmux.neuralfree.gguf.video.VideoMeta;
import com.eonmux.neuralfree.gguf.video.VideoOptions;
import com.eonmux.neuralfree.gguf.video.VideoPruner;
import com.eonmux.neuralfree.gguf.video.VideoSamplePlan;
import org.junit.Assume;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStream;
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
 * Video input over the real projector graphs ({@link MediaContext#appendVideo}): a llava projector takes every
 * frame as its own chunk behind the {@code "Video:"} prefix with an {@code [XmY.YYs]} stamp at the interval, the
 * Qwen2-VL projector merges frame pairs through its temporal patch kernel (the rows equal a direct pair encode)
 * and stamps {@code <X.X seconds>} per pair, a configured pruning rate keeps the first frame's rows and the exact
 * retained count, a repeated video reads the caches, and a projector without a vision encoder refuses video.
 * Skipped when no ffmpeg is on the PATH.
 */
public class MediaContextVideoTest {

    private static final int WIDTH = 16;

    /** A text encoder recording every segment it tokenizes, one token per character. */
    private static final class RecordingText implements MediaContext.TextEncoder {
        final List<String> seen = new ArrayList<>();

        @Override
        public int[] encode(String text) {
            seen.add(text);
            int[] out = new int[text.length()];
            for (int i = 0; i < out.length; i++) {
                out[i] = text.charAt(i);
            }
            return out;
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

    /** An 8x8 2 fps MP4 of 3 seconds (6 frames), encoded with the test's ffmpeg. */
    private static byte[] mp4() throws Exception {
        Path out = Files.createTempFile("nf-media-video-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin", "-f", "lavfi",
                    "-i", "testsrc=size=8x8:rate=2", "-t", "3", "-pix_fmt", "yuv420p", out.toString())
                    .redirectErrorStream(false).start();
            drain(p);
            assertTrue("ffmpeg failed to encode the fixture", p.waitFor(120, TimeUnit.SECONDS));
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    /** An 8x8 2 fps MP4 of 3 seconds with a 440 Hz mono AAC soundtrack at 16 kHz. */
    private static byte[] mp4WithAudio() throws Exception {
        Path out = Files.createTempFile("nf-media-video-audio-test-", ".mp4");
        try {
            Process p = new ProcessBuilder("ffmpeg", "-y", "-nostdin",
                    "-f", "lavfi", "-i", "testsrc=size=8x8:rate=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
                    "-t", "3", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", out.toString())
                    .redirectErrorStream(false).start();
            drain(p);
            assertTrue("ffmpeg failed to encode the fixture", p.waitFor(120, TimeUnit.SECONDS));
            return Files.readAllBytes(out);
        } finally {
            Files.deleteIfExists(out);
        }
    }

    private static void drain(Process p) throws IOException {
        try (InputStream err = p.getErrorStream()) {
            byte[] buf = new byte[4096];
            while (err.read(buf) > 0) {
                // the encode's log drains away
            }
        }
    }

    private static MediaContext open(Path file, boolean mrope, VideoOptions video) throws IOException {
        return open(file, WIDTH, mrope, video);
    }

    private static MediaContext open(Path file, int textEmbd, boolean mrope, VideoOptions video)
            throws IOException {
        return MediaContext.open(file.toString(), textEmbd, mrope, ImageTokenBudget.none(), t -> false,
                MmprojOptions.cpu(), MediaContext.DEFAULT_CACHE_BYTES, MediaIoOptions.defaults(), video);
    }

    @Test
    public void aLlavaVideoTakesThePrefixAChunkPerFrameAndTheIntervalStamps() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        // the interval stamps at 1 s: the frames play at 2 fps, so frames 2 and 4 cross the thresholds
        VideoOptions options = VideoOptions.builder().timestampIntervalMs(1000).build();
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false, options)) {
            assertTrue(ctx.supportsVideo());
            VideoConfig config = ctx.videoConfig();
            assertEquals(1, config.temporalMerge());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", text);
            MediaPrompt built = prompt.build();

            assertEquals("Video:", text.seen.get(0));
            assertEquals("[0m1.00s]", text.seen.get(1));
            assertEquals("[0m2.00s]", text.seen.get(2));
            // six frames of four rows each, the stamps between them
            List<MediaChunk> chunks = built.chunks();
            int media = 0;
            for (MediaChunk chunk : chunks) {
                if (!chunk.isText()) {
                    media++;
                    assertEquals(4, chunk.nTokens());
                    assertFalse(chunk.nonCausal());
                    assertEquals(MediaRopePositions.normal(chunk.positions().pos0(), chunk.nTokens()),
                            chunk.positions());
                }
            }
            assertEquals(6, media);
        }
    }

    @Test
    public void aQwenVideoMergesFramePairsThroughTheTemporalKernel() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoOptions options = VideoOptions.builder().timestampIntervalMs(1000).build();
        try (MediaContext ctx = open(MmprojTestFiles.qwen2vl(WIDTH), true, options)) {
            assertEquals(VideoConfig.forProjector(ProjectorType.QWEN2VL, false), ctx.videoConfig());
            RecordingText text = new RecordingText();
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", text);
            MediaPrompt built = prompt.build();

            List<MediaChunk> pairs = new ArrayList<>();
            for (MediaChunk chunk : built.chunks()) {
                if (chunk.isText()) {
                    continue;
                }
                pairs.add(chunk);
                // the 12x12 frames the dyn-size preprocessing makes encode to a 3x3 merged token grid
                assertEquals(MediaRopePositions.image2d(chunk.positions().pos0(), 3, 3), chunk.positions());
            }
            assertEquals(3, pairs.size());
            // Qwen2-VL keeps llama.cpp's default template: the "Video:" prefix and interval stamps
            assertEquals("Video:", text.seen.get(0));
            assertEquals(List.of("[0m1.50s]", "[0m2.50s]"),
                    text.seen.subList(1, text.seen.size()).stream()
                            .filter(s -> s.startsWith("[")).collect(java.util.stream.Collectors.toList()));
            assertEquals("<|vision_start|>", text.seen.get(text.seen.indexOf("[0m1.50s]") + 1));

            // every pair's rows equal a direct encode of its two frames through the temporal kernels
            VideoDecoder decoder = new VideoDecoder();
            VideoMeta meta = decoder.probe(video, "video_url.url");
            VideoSamplePlan plan = ctx.videoConfig().plan(meta, null, null, 0, "video_url.url");
            List<RasterImage> frames = decoder.decode(video, plan, "video_url.url").frames();
            try (MmprojModel mm = MmprojModel.open(MmprojTestFiles.qwen2vl(WIDTH).toString());
                 CpuVisionEncoder enc = mm.openCpuVisionEncoder()) {
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(),
                        ImageTokenBudget.none());
                // the pairs encode as one batch: each one's rows land on its own chunk, in frame order
                for (int p = 0; p < pairs.size(); p++) {
                    ImageTile a = ImagePreprocessors.forParams(params).prepare(frames.get(2 * p)).encodedTiles()
                            .get(0);
                    ImageTile b = ImagePreprocessors.forParams(params).prepare(frames.get(2 * p + 1))
                            .encodedTiles().get(0);
                    float[][] expected = enc.encode(ClipImage.of(a.width(), a.height(),
                            a.normalized(params.mean(), params.std()))
                            .withSecondFrame(ClipImage.of(b.width(), b.height(),
                                    b.normalized(params.mean(), params.std())).pixels()));
                    assertEquals(9, expected.length);
                    assertEquals(expected.length, pairs.get(p).nTokens());
                    for (int r = 0; r < expected.length; r++) {
                        assertArrayEquals("pair " + p + " row " + r, expected[r], pairs.get(p).embeddings()[r],
                                0f);
                    }
                }
            }
        }
    }

    @Test
    public void anLfm3OmniVideoEncodesTheSoundtrackAfterEachFrameGroup() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4WithAudio();
        String field = "video_url.url";
        Path projector = Lfm3OmniFixture.write();
        try (MediaContext ctx = open(projector, Lfm3OmniFixture.WIDTH, false, VideoOptions.defaults())) {
            assertTrue(ctx.supportsVideo());
            assertTrue(ctx.supportsAudio());
            assertTrue(ctx.videoConfig().soundtrack());

            RecordingText text = new RecordingText();
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, field, text);
            List<MediaChunk> media = new ArrayList<>();
            for (MediaChunk chunk : prompt.build().chunks()) {
                if (!chunk.isText()) {
                    media.add(chunk);
                }
            }

            // the same decode the context runs, to compute the reference row split
            VideoDecoder decoder = new VideoDecoder();
            VideoMeta meta = decoder.probe(video, field);
            VideoSamplePlan plan = ctx.videoConfig().plan(meta, null, null, 0,
                    ctx.videoOptions().fpsTarget(), field);
            assertEquals(6, plan.nFrames());
            assertEquals(3, plan.groups());
            try (MmprojModel mm = MmprojModel.open(projector.toString())) {
                ImageProcessorParams params = ImageProcessorParams.fromGguf(mm.metadata(),
                        ImageTokenBudget.none());
                ImagePreprocessor pre = ImagePreprocessors.forParams(params);
                ImageSize frameSize = pre.videoFrameSize(plan.nFrames(), meta.width(), meta.height());
                assertEquals(new ImageSize(28, 28), frameSize);
                int rate = mm.audioHparams().audio().sampleRate();
                DecodedVideo decoded = decoder.decode(video, plan, frameSize, rate, field);
                assertEquals(3, decoded.soundtrack().size());

                float[] samples = new float[sumLengths(decoded.soundtrack())];
                int at = 0;
                for (AudioClip clip : decoded.soundtrack()) {
                    System.arraycopy(clip.samples(), 0, samples, at, clip.length());
                    at += clip.length();
                }
                try (CpuAudioEncoder audio = mm.openAudioEncoder()) {
                    AudioClip track = new AudioClip(samples, rate);
                    float[][] rows = audio.embed(track).get(0);
                    assertEquals("the soundtrack's rows", audio.nTokens(track), rows.length);
                    double samplesPerRow = 8.0 * mm.audioHparams().audio().hopLen();
                    assertSoundtrackRows(rows, media, decoded.soundtrack(), samplesPerRow);
                }
            }

            // the group timestamps precede their groups
            List<String> stamps = new ArrayList<>();
            for (String seen : text.seen) {
                if (seen.endsWith(" seconds>")) {
                    stamps.add(seen);
                }
            }
            assertEquals(List.of("<0.2 seconds>", "<1.2 seconds>", "<2.2 seconds>"), stamps);
        }
    }

    /** Asserts the prompt's image/audio chunks alternate and the audio rows are the reference's per-group spans. */
    private static void assertSoundtrackRows(float[][] rows, List<MediaChunk> media, List<AudioClip> slices,
                                             double samplesPerRow) {
        assertEquals("image and audio chunks", 6, media.size());
        int begin = 0;
        int cumulative = 0;
        for (int g = 0; g < 3; g++) {
            MediaChunk image = media.get(2 * g);
            MediaChunk audio = media.get(2 * g + 1);
            // the 28x28 frame pairs merge to a 7x7 token grid
            assertEquals("group " + g + " image rows", 49, image.nTokens());
            cumulative += slices.get(g).length();
            int end = g == 2 ? rows.length
                    : (int) Math.max(begin, Math.min(rows.length, Math.rint(cumulative / samplesPerRow)));
            assertEquals("group " + g + " audio rows", end - begin, audio.nTokens());
            for (int r = begin; r < end; r++) {
                assertArrayEquals("group " + g + " row " + r, rows[r], audio.embeddings()[r - begin], 0f);
            }
            begin = end;
        }
    }

    private static int sumLengths(List<AudioClip> clips) {
        int total = 0;
        for (AudioClip clip : clips) {
            total += clip.length();
        }
        return total;
    }

    @Test
    public void theFallbackSamplingRateTakesTheFramesItNames() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        // six frames at 2 fps replayed at 1 fps: three frames, round(linspace(0, 5, 3)) = [0, 2, 5]
        VideoOptions options = VideoOptions.builder().fpsTarget(1.0).build();
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false, options)) {
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", new RecordingText());
            int frames = 0;
            for (MediaChunk chunk : prompt.build().chunks()) {
                if (!chunk.isText()) {
                    frames++;
                }
            }
            assertEquals(3, frames);
        }
    }

    @Test
    public void pruningKeepsTheFirstFramesRowsAndTheExactCount() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoOptions options = VideoOptions.builder().pruningRate(0.5).build();
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false, options)) {
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", new RecordingText());
            List<MediaChunk> chunks = prompt.build().chunks();
            List<MediaChunk> frames = new ArrayList<>();
            for (MediaChunk chunk : chunks) {
                if (!chunk.isText()) {
                    frames.add(chunk);
                }
            }
            // a fully pruned frame drops its chunk (its stamp stays); the others keep theirs
            assertTrue("the pruned video holds " + frames.size() + " frame chunks", frames.size() >= 5);
            // 24 rows at the rate 0.5 keep 12, the first frame's four among them
            int kept = 0;
            for (MediaChunk chunk : frames) {
                kept += chunk.nTokens();
            }
            assertEquals(12, kept);
            assertEquals(4, frames.get(0).nTokens());
        }
        // vidcom2 keeps the same count
        VideoOptions vidcom2 = VideoOptions.builder().pruningRate(0.5)
                .pruningMethod(VideoPruner.Method.VIDCOM2).build();
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false, vidcom2)) {
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", new RecordingText());
            int kept = 0;
            for (MediaChunk chunk : prompt.build().chunks()) {
                if (!chunk.isText()) {
                    kept += chunk.nTokens();
                }
            }
            assertEquals(12, kept);
        }
    }

    @Test
    public void pruningAnMropeVideoKeepsTheKeptCellsPositions() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        VideoOptions options = VideoOptions.builder().pruningRate(0.5).build();
        try (MediaContext ctx = open(MmprojTestFiles.qwen2vl(WIDTH), true, options)) {
            MediaPrompt.Builder prompt = MediaPrompt.builder();
            ctx.appendVideo(prompt, video, "video_url.url", new RecordingText());
            int kept = 0;
            int chunks = 0;
            for (MediaChunk chunk : prompt.build().chunks()) {
                if (chunk.isText()) {
                    continue;
                }
                chunks++;
                kept += chunk.nTokens();
                // the kept cells keep their own M-RoPE positions, one per row
                assertEquals(chunk.nTokens(), chunk.positions().nTokens());
                assertEquals(MediaRopePositions.Kind.IMAGE_2D, chunk.positions().kind());
            }
            // three 3x3 pair chunks keep int(27 * 0.5) = 13 rows
            assertEquals(13, kept);
            assertTrue(chunks >= 1);
        }
    }

    @Test
    public void aRepeatedVideoReadsTheCachesInsteadOfDecodingAgain() throws Exception {
        Assume.assumeTrue("no ffmpeg on the PATH", ffmpegOnPath());
        byte[] video = mp4();
        try (MediaContext ctx = open(MmprojTestFiles.llava(WIDTH), false, VideoOptions.defaults())) {
            RecordingText first = new RecordingText();
            MediaPrompt.Builder one = MediaPrompt.builder();
            ctx.appendVideo(one, "clip-1", () -> video, "video_url.url", first);
            List<float[]> rows = rowsOf(one.build());
            assertEquals(0, ctx.cacheHits());
            assertEquals(1, ctx.cacheQueries());

            // the same uuid with its data again reads the cached encoding
            RecordingText second = new RecordingText();
            MediaPrompt.Builder two = MediaPrompt.builder();
            ctx.appendVideo(two, "clip-1", () -> video, "video_url.url", second);
            assertSameRows(rows, rowsOf(two.build()));
            assertTrue(ctx.hasCached(MediaLimits.Modality.VIDEO, "clip-1"));

            // a third append under the uuid alone needs no bytes
            MediaPrompt.Builder three = MediaPrompt.builder();
            ctx.appendVideo(three, "clip-1", null, "video_url.url", new RecordingText());
            assertSameRows(rows, rowsOf(three.build()));
            assertTrue(ctx.cacheHits() >= 2);
        }
    }

    /** Two row lists of the same values (the cache serves fresh copies of the same rows). */
    private static void assertSameRows(List<float[]> want, List<float[]> got) {
        assertEquals(want.size(), got.size());
        for (int r = 0; r < want.size(); r++) {
            assertArrayEquals("row " + r, want.get(r), got.get(r), 0f);
        }
    }

    private static List<float[]> rowsOf(MediaPrompt prompt) {
        List<float[]> out = new ArrayList<>();
        for (MediaChunk chunk : prompt.chunks()) {
            if (!chunk.isText()) {
                for (float[] row : chunk.embeddings()) {
                    out.add(row);
                }
            }
        }
        return out;
    }

    @Test
    public void aProjectorWithoutAVisionEncoderRefusesVideo() throws IOException {
        try (MediaContext ctx = open(MmprojTestFiles.qwen2aAudio(WIDTH), false, VideoOptions.defaults())) {
            assertFalse(ctx.supportsVideo());
            assertTrue(ctx.supportsAudio());
            MediaInputException e = assertThrows(MediaInputException.class,
                    () -> ctx.appendVideo(MediaPrompt.builder(), new byte[0], "f", new RecordingText()));
            assertTrue(e.getMessage().contains("vision"));
        }
    }
}
