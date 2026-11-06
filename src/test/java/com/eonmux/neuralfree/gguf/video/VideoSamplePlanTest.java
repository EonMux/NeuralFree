package com.eonmux.neuralfree.gguf.video;

import com.eonmux.neuralfree.gguf.media.MediaInputException;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

/**
 * The video sampling plan and the per-model video config: the frame count of the HF processors (the source's
 * frames replayed at the target rate, clamped to the family's range) and their {@code round(linspace)} indices,
 * the Qwen2-VL loader's patch-floored count and {@code torch.arange} indices, the Qwen3-VL loader's 24 fps
 * assumption for a source without a rate, the group midpoint timestamps of the LFM3-VL and Qwen3-VL templates, the
 * timestamp texts of llama.cpp's helper, the LFM3-Omni frame clamp and the max_duration refusal.
 */
public class VideoSamplePlanTest {

    /** A source of {@code frames} frames at {@code fps}, running {@code frames / fps} seconds. */
    private static VideoMeta meta(int frames, double fps) {
        return VideoMeta.of(8, 8, fps, frames, frames / fps);
    }

    @Test
    public void theLfm3ConfigRecommendsTwoFpsWithTheOmniClamp() {
        VideoConfig lfm3 = VideoConfig.forProjector(ProjectorType.LFM3VL, false);
        assertEquals(VideoTemplate.GROUP, lfm3.template());
        assertEquals(2, lfm3.temporalMerge());
        assertEquals(2.0, lfm3.fps(), 0);
        assertEquals(4, lfm3.minFrames());
        assertEquals(768, lfm3.maxFrames());

        VideoConfig omni = VideoConfig.forProjector(ProjectorType.LFM3VL, true);
        assertEquals(384, omni.maxFrames());
        assertTrue(omni.soundtrack());
        assertFalse(lfm3.soundtrack());

        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        assertEquals(VideoTemplate.DEFAULT, qwen.template());
        assertEquals(2, qwen.temporalMerge());
        assertEquals(2.0, qwen.fps(), 0);
        assertEquals(4, qwen.minFrames());
        assertEquals(768, qwen.maxFrames());

        VideoConfig qwen3 = VideoConfig.forProjector(ProjectorType.QWEN3VL, false);
        assertEquals(VideoTemplate.GROUP, qwen3.template());
        assertEquals(2, qwen3.temporalMerge());
        assertEquals(2.0, qwen3.fps(), 0);

        VideoConfig plain = VideoConfig.forProjector(ProjectorType.MLP, false);
        assertEquals(VideoTemplate.DEFAULT, plain.template());
        assertEquals(1, plain.temporalMerge());
    }

    @Test
    public void thePlanReplaysTheSourceAtTheTargetRateWithTheProcessorIndices() {
        // 100 frames at 10 fps sampled at 2 fps: the loader's int(total / fps * target) = 20 frames, and
        // torch.arange(0, 100, 5) takes every fifth source frame
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        VideoSamplePlan plan = qwen.plan(meta(100, 10), 2.0, null, 0, "video_url.url");
        assertEquals(20, plan.nFrames());
        assertEquals(2.0, plan.targetFps(), 1e-9);
        int[] indices = plan.indices();
        assertEquals(0, indices[0]);
        assertEquals(95, indices[indices.length - 1]);
        assertEquals(20, indices[4]);
        assertEquals(20, indices.length);
        for (int i = 1; i < indices.length; i++) {
            assertTrue(indices[i - 1] <= indices[i]);
        }
        // the families that keep the HF processors' indices take round(linspace(0, 99, 20))
        VideoSamplePlan lfm3 = VideoConfig.forProjector(ProjectorType.LFM3VL, false)
                .plan(meta(100, 10), 2.0, null, 0, "video_url.url");
        assertEquals(99, lfm3.indices()[lfm3.nFrames() - 1]);
        // the fifth index is rint(4 * 99 / 19)
        assertEquals((int) Math.rint(4 * 99.0 / 19), lfm3.indices()[4]);
    }

    @Test
    public void theFrameCountClampsToTheFamilyRangeAndTheSource() {
        VideoConfig cfg = VideoConfig.forProjector(ProjectorType.LFM3VL, false);
        // a long video clamps at 768 frames
        assertEquals(768, cfg.plan(meta(10000, 10), null, null, 0, "f").nFrames());
        // a short video keeps every frame, the clamp floor never pads
        assertEquals(6, cfg.plan(meta(6, 2), null, null, 0, "f").nFrames());
        // the model's own 2 fps rate decides the count; the request's count is ignored (fps-driven loaders)
        assertEquals(768, cfg.plan(meta(10000, 10), null, 5000, 0, "f").nFrames());
        assertEquals(20, cfg.plan(meta(100, 10), null, 2, 0, "f").nFrames());
        // the omni variant clamps at 384
        assertEquals(384, VideoConfig.forProjector(ProjectorType.LFM3VL, true)
                .plan(meta(10000, 10), null, null, 0, "f").nFrames());
    }

    @Test
    public void aRecommendedFamilyIgnoresTheRequestedFrameCount() {
        // vLLM's Qwen2-VL and Qwen3-VL loaders are fps-driven and ignore num_frames
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        assertEquals(20, qwen.plan(meta(100, 10), null, 2, 0, "f").nFrames());
        assertEquals(20, qwen.plan(meta(100, 10), null, 500, 0, "f").nFrames());
        // llama.cpp's LFM3-VL helper samples at the family's own recommended rate as well
        VideoConfig lfm3 = VideoConfig.forProjector(ProjectorType.LFM3VL, false);
        assertEquals(20, lfm3.plan(meta(100, 10), null, 2, 0, "f").nFrames());
        // the request's rate still overrides the recommendation
        assertEquals(10, lfm3.plan(meta(100, 10), 1.0, null, 0, "f").nFrames());
    }

    @Test
    public void theQwen2VlLoaderFloorsTheCountToTheTemporalPatchAndIndexesWithArange() {
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        // 105 frames at 10 fps sampled at 2 fps: vLLM's n = 21, floored to 20; torch.arange(0, 105, 5.25)
        VideoSamplePlan plan = qwen.plan(meta(105, 10), null, null, 0, "f");
        assertEquals(20, plan.nFrames());
        assertArrayEquals(new int[] {0, 5, 10, 15, 21, 26, 31, 36, 42, 47, 52, 57, 63, 68, 73, 78, 84, 89, 94,
                99}, plan.indices());
        // the same video through round(linspace(0, 104, 20)) would end at 104, not at vLLM's arange tail
        assertEquals(99, plan.indices()[plan.nFrames() - 1]);
        // Qwen2.5-VL shares the loader
        assertArrayEquals(plan.indices(), VideoConfig.forProjector(ProjectorType.QWEN25VL, false)
                .plan(meta(105, 10), null, null, 0, "f").indices());
    }

    @Test
    public void theQwen2VlLoaderIndexesWithFloat32ProductsLikeTorch() {
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        // 118 frames at 2 fps sampled at 1 fps: n = 59 floored to 58, step = 118 / 58. torch.arange evaluates
        // 29 * step in float32 (59.000004) and truncates to 59; the same product in double is 58.99999... and
        // truncates to 58. The float32 step also emits one index past the count (59 values for 58).
        VideoSamplePlan plan = qwen.plan(meta(118, 2), 1.0, null, 0, "f");
        assertEquals(59, plan.nFrames());
        assertArrayEquals(new int[] {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
                40, 42, 44, 46, 48, 50, 52, 54, 56, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89,
                91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117}, plan.indices());

        // 59 frames at 1 fps sampled at 2 fps: the even frame range (58) caps the count, and the float32 step
        // emits one index past the count, clamped to the last frame
        VideoSamplePlan capped = qwen.plan(meta(59, 1), null, null, 0, "f");
        int[] indices = capped.indices();
        assertEquals(59, indices.length);
        for (int i = 0; i < indices.length; i++) {
            assertEquals(i, indices[i]);
        }
    }

    @Test
    public void theQwen2VlLoaderCapsTheSamplingRateAtThirtyFps() {
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        // vLLM's _MAX_FPS: a request for 60 fps samples at 30, so 105 frames at 10 fps give 104 (the floored cap)
        VideoSamplePlan plan = qwen.plan(meta(105, 10), 60.0, null, 0, "f");
        assertEquals(104, plan.nFrames());
        int[] indices = plan.indices();
        assertEquals(0, indices[0]);
        assertEquals(103, indices[indices.length - 1]);
        for (int i = 0; i < indices.length; i++) {
            assertEquals(i, indices[i]);
        }
    }

    @Test
    public void aVideoShorterThanTheQwen2VlTemporalPatchIsRefusedByName() {
        VideoConfig qwen = VideoConfig.forProjector(ProjectorType.QWEN2VL, false);
        // the floored frame range of a one-frame video is 0: vLLM divides by zero, NeuralFree refuses by field
        MediaInputException e = assertThrows(MediaInputException.class,
                () -> qwen.plan(meta(1, 1), null, null, 0, "video_url.url"));
        assertTrue(e.getMessage().contains("video_url.url"));
        assertTrue(e.getMessage().contains("Qwen2-VL"));
        assertTrue(e.getMessage().contains("1 frame"));
        // two frames are one patch, the smallest the loader samples
        assertEquals(2, qwen.plan(meta(2, 1), null, null, 0, "f").nFrames());
        // three frames cap at one patch as well: the floored maximum wins over the four-frame floor
        assertArrayEquals(new int[] {0, 1}, qwen.plan(meta(3, 1), null, null, 0, "f").indices());
    }

    @Test
    public void theQwen3VlLoaderAssumesTwentyFourFpsForASourceWithoutARate() {
        // ffprobe reports no r_frame_rate (a variable or unknown rate) but the frame count and duration
        VideoMeta source = VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "nb_frames", "240", "duration", "10"));
        assertFalse(source.hasFrameRate());
        VideoConfig qwen3 = VideoConfig.forProjector(ProjectorType.QWEN3VL, false);
        assertEquals(24.0, qwen3.assumedSourceFps(), 0);
        VideoSamplePlan plan = qwen3.plan(source, null, null, 0, "video_url.url");
        // vLLM's Qwen3VLVideoBackend assumes 24 fps: int(240 / 24 * 2) = 20 frames, round(linspace(0, 239, 20))
        assertEquals(20, plan.nFrames());
        assertArrayEquals(new int[] {0, 13, 25, 38, 50, 63, 75, 88, 101, 113, 126, 138, 151, 164, 176, 189, 201,
                214, 226, 239}, plan.indices());
        assertEquals(24.0, plan.source().fps(), 0);
        assertEquals(240, plan.source().nFrames());
        // the group stamps use the assumed rate: indices 0 and 13 at 24 fps midpoint at 0.2708 s
        assertEquals(13 / 48.0, plan.groupTimestamp(0), 1e-12);
        assertEquals(63 / 48.0, plan.groupTimestamp(1), 1e-12);

        // a source without a duration samples the same count, playing at the assumed rate
        VideoMeta noDuration = VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "nb_frames", "240"));
        assertEquals(0.0, noDuration.duration(), 0);
        VideoSamplePlan timeless = qwen3.plan(noDuration, null, null, 0, "f");
        assertEquals(20, timeless.nFrames());
        assertEquals(24.0, timeless.targetFps(), 0);
    }

    @Test
    public void aSourceWithoutARateIsRefusedByNameWhereTheReferenceRefusesIt() {
        VideoMeta source = VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "nb_frames", "240", "duration", "10"));
        // vLLM's Qwen2-VL loader raises: the Qwen2 family has no assumed rate
        for (ProjectorType type : new ProjectorType[] {ProjectorType.QWEN2VL, ProjectorType.QWEN25VL}) {
            VideoConfig cfg = VideoConfig.forProjector(type, false);
            assertEquals(0.0, cfg.assumedSourceFps(), 0);
            MediaInputException e = assertThrows(MediaInputException.class,
                    () -> cfg.plan(source, null, null, 0, "video_url.url"));
            assertTrue(e.getMessage().contains("video_url.url"));
            assertTrue(e.getMessage().contains("Qwen2-VL"));
        }
        // llama.cpp's helper rejects a probe without a rate, and so do the generic projectors
        for (ProjectorType type : new ProjectorType[] {ProjectorType.LFM3VL, ProjectorType.MLP}) {
            MediaInputException e = assertThrows(MediaInputException.class,
                    () -> VideoConfig.forProjector(type, false).plan(source, null, null, 0, "video_url.url"));
            assertTrue(e.getMessage().contains("video_url.url"));
            assertTrue(e.getMessage().contains("no frame rate"));
        }
        // a known rate passes through untouched
        VideoMeta known = meta(100, 10);
        assertSame(known, VideoConfig.forProjector(ProjectorType.QWEN2VL, false)
                .resolveSource(known, "video_url.url"));
    }

    @Test
    public void videoMetadataCarriesAnUnknownRateUntilTheFamilyResolvesIt() {
        VideoMeta meta = VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "nb_frames", "240", "duration", "10"));
        assertEquals(0.0, meta.fps(), 0);
        assertFalse(meta.hasFrameRate());
        assertEquals(240, meta.nFrames());
        assertEquals(10.0, meta.duration(), 0);
        assertThrows(IllegalStateException.class, () -> meta.timeOf(3));
        // a 0/0 rational is no rate either, and a hand-built metadata may carry the unknown rate
        assertFalse(VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "r_frame_rate", "0/0", "nb_frames", "10")).hasFrameRate());
        assertFalse(VideoMeta.of(8, 8, 0, 240, 10).hasFrameRate());
        // the family's assumption keeps the frame count and the duration
        VideoMeta resolved = meta.withFps(24.0);
        assertEquals(24.0, resolved.fps(), 0);
        assertEquals(240, resolved.nFrames());
        assertEquals(10.0, resolved.duration(), 0);
        assertEquals(3 / 24.0, resolved.timeOf(3), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> meta.withFps(0));
        // a plan needs the resolved rate
        assertThrows(IllegalArgumentException.class,
                () -> VideoSamplePlan.of(meta, new int[] {0}, 1.0, VideoTemplate.DEFAULT, 1));
        // neither a frame count nor a rate can count the frames
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8", "duration", "10")));
        assertTrue(e.getMessage().contains("frame rate"));
    }

    @Test
    public void theGenericBackendTakesTheSmallerOfTheCountAndTheFpsCount() {
        VideoConfig generic = VideoConfig.forProjector(ProjectorType.MLP, false);
        // a 10 s video at 2 fps derives 20 frames; the requested count caps it when smaller
        assertEquals(10, generic.plan(meta(100, 10), 2.0, 10, 0, "f").nFrames());
        // and the fps-derived count caps the request when smaller
        assertEquals(20, generic.plan(meta(100, 10), 2.0, 50, 0, "f").nFrames());
        // a count above the source's frames clamps to the source first
        assertEquals(20, generic.plan(meta(100, 10), 2.0, 500, 0, "f").nFrames());
        // a count alone samples that count, a rate alone its derived count
        assertEquals(10, generic.plan(meta(100, 10), null, 10, 0, "f").nFrames());
        assertEquals(20, generic.plan(meta(100, 10), 2.0, null, 0, "f").nFrames());
    }

    @Test
    public void aVideoOverTheMaxDurationRefusesNamingTheField() {
        VideoConfig cfg = VideoConfig.forProjector(ProjectorType.MLP, false);
        MediaInputException e = assertThrows(MediaInputException.class,
                () -> cfg.plan(meta(100, 10), null, null, 5, "video_url.url"));
        assertTrue(e.getMessage().contains("video_url.url"));
        assertTrue(e.getMessage().contains("max_duration"));
        // 0 takes no limit
        assertEquals(100, cfg.plan(meta(100, 10), null, null, 0, "f").nFrames());
    }

    @Test
    public void groupTimestampsAreTheMidpointTimesOfEachPair() {
        VideoConfig cfg = VideoConfig.forProjector(ProjectorType.QWEN3VL, false);
        VideoSamplePlan plan = cfg.plan(meta(6, 2), null, null, 0, "f");
        assertEquals(3, plan.groups());
        // frames 0..5 at 2 fps: the group midpoints are (t0 + t1) / 2 = 0.25, 1.25 and 2.25 s
        assertEquals(0.25, plan.groupTimestamp(0), 1e-9);
        assertEquals(1.25, plan.groupTimestamp(1), 1e-9);
        assertEquals(2.25, plan.groupTimestamp(2), 1e-9);
    }

    @Test
    public void anOddFrameCountFormsItsLastGroupFromOneFrame() {
        VideoSamplePlan plan = VideoSamplePlan.ofAll(meta(5, 1), VideoTemplate.GROUP, 2);
        assertEquals(3, plan.groups());
        // the last group's midpoint repeats its only frame's time, as the HF padding repeats the last index
        assertEquals(4.0, plan.groupTimestamp(2), 1e-9);
    }

    @Test
    public void theTimestampTextsMatchTheReferenceFormats() {
        assertEquals("Video:", VideoTemplate.DEFAULT.videoPrefix());
        assertEquals("[0m0.00s]", VideoTemplate.DEFAULT.frameTimestamp(0));
        assertEquals("[1m50.25s]", VideoTemplate.DEFAULT.frameTimestamp(110.25));
        // the exact halves round half to even, as C's snprintf and the HF formatters do
        assertEquals("<0.2 seconds>", VideoTemplate.GROUP.groupTimestamp(0.25));
        assertEquals("<0.3 seconds>", VideoTemplate.GROUP.groupTimestamp(0.35));
        assertEquals("<2.2 seconds>", VideoTemplate.GROUP.groupTimestamp(2.25));
        assertThrows(IllegalArgumentException.class, () -> VideoTemplate.GROUP.groupTimestamp(-1));
        assertThrows(IllegalArgumentException.class, () -> VideoTemplate.GROUP.frameTimestamp(0));
        assertThrows(IllegalArgumentException.class, () -> VideoTemplate.DEFAULT.groupTimestamp(0));
    }

    @Test
    public void theDefaultTemplateStampsEveryInterval() {
        // 6 frames playing at 2 fps: elapsed 0, 500, 1000, 1500, 2000, 2500 ms over a 1000 ms interval
        VideoSamplePlan plan = VideoSamplePlan.ofAll(meta(6, 2), VideoTemplate.DEFAULT, 1);
        assertEquals(2, plan.frameTimestamps(6, 1000));
        assertEquals(0, plan.frameTimestamps(6, 0));
        assertEquals(0, VideoSamplePlan.ofAll(meta(6, 2), VideoTemplate.GROUP, 2).frameTimestamps(6, 1000));
        assertEquals(0.0, plan.frameTime(0), 0);
        assertEquals(2.0, plan.frameTime(4), 1e-9);
    }

    @Test
    public void aFrameRateRequestOverridesTheModelRecommendation() {
        VideoConfig cfg = VideoConfig.forProjector(ProjectorType.LFM3VL, false);
        VideoSamplePlan plan = cfg.plan(meta(100, 10), 5.0, null, 0, "f");
        assertEquals(50, plan.nFrames());
        // the source rate replays unchanged with no request and no model recommendation
        VideoSamplePlan plain = VideoConfig.forProjector(ProjectorType.MLP, false)
                .plan(meta(12, 4), null, null, 0, "f");
        assertEquals(12, plain.nFrames());
        assertEquals(4.0, plain.targetFps(), 0);
    }

    @Test
    public void theCallerFallbackRateDrivesAFamilyWithoutARecommendation() {
        VideoConfig plain = VideoConfig.forProjector(ProjectorType.MLP, false);
        // llama.cpp's --video-fps default of 4: 100 frames at 10 fps replay as 40
        assertEquals(40, plain.plan(meta(100, 10), null, null, 0, 4.0, "f").nFrames());
        // the family's recommendation wins over the caller's rate
        VideoConfig lfm3 = VideoConfig.forProjector(ProjectorType.LFM3VL, false);
        assertEquals(20, lfm3.plan(meta(100, 10), null, null, 0, 4.0, "f").nFrames());
        // no fallback keeps the source rate
        assertEquals(100, plain.plan(meta(100, 10), null, null, 0, 0, "f").nFrames());
        // a request rate beats both
        assertEquals(10, plain.plan(meta(100, 10), 1.0, null, 0, 4.0, "f").nFrames());
    }

    @Test
    public void videoMetadataValidatesAndParsesTheProbeFields() {
        VideoMeta meta = VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8",
                "r_frame_rate", "30000/1001", "nb_frames", "N/A", "duration", "3.0"));
        assertEquals(8, meta.width());
        assertEquals(30000.0 / 1001, meta.fps(), 1e-9);
        assertEquals((int) Math.round(3.0 * 30000.0 / 1001), meta.nFrames());
        assertEquals(3.0, meta.duration(), 0);
        assertEquals(0.0, meta.timeOf(0), 0);
        assertThrows(IllegalArgumentException.class,
                () -> VideoMeta.ofProbe(java.util.Map.of("width", "0", "height", "8",
                        "r_frame_rate", "25")));
        assertThrows(IllegalArgumentException.class,
                () -> VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8")));
        assertThrows(IllegalArgumentException.class,
                () -> VideoMeta.ofProbe(java.util.Map.of("width", "8", "height", "8", "r_frame_rate", "N/A")));
    }

    @Test
    public void planAccessorsRefuseOutsideIndices() {
        VideoSamplePlan plan = VideoSamplePlan.ofAll(meta(4, 2), VideoTemplate.DEFAULT, 1);
        assertThrows(IndexOutOfBoundsException.class, () -> plan.groupTimestamp(0));
        assertThrows(IndexOutOfBoundsException.class, () -> plan.frameTime(4));
        assertArrayEquals(new int[] {0, 1, 2, 3}, plan.indices());
    }
}
