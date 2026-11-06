package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * The LFM3-VL video frame sizing against llama.cpp's {@code mtmd_image_preprocessor_lfm3vl}
 * ({@code get_video_frame_size} and {@code preprocess_video_frame}): the reference video processor's fixed pixel
 * limits, the check spanning the frame count rounded to temporal groups, and frames the decode already sized
 * snapping to the merged patch multiple instead of the image's pixel limits.
 */
public class Lfm3vlPreprocessorTest {

    private static Map<String, Object> metadata(Object... extra) {
        Map<String, Object> m = new HashMap<>();
        m.put("general.architecture", "clip");
        m.put("clip.vision.projector_type", "lfm3vl");
        m.put("clip.vision.patch_size", 16);
        m.put("clip.vision.image_size", 512);
        m.put("clip.vision.spatial_merge_size", 2);
        m.put("clip.vision.image_resize_algo", "bilinear");
        m.put("clip.vision.image_min_pixels", 65536);
        m.put("clip.vision.image_max_pixels", 1048576);
        m.put("clip.vision.image_mean", new float[] {0, 0, 0});
        m.put("clip.vision.image_std", new float[] {1, 1, 1});
        for (int i = 0; i < extra.length; i += 2) {
            m.put((String) extra[i], extra[i + 1]);
        }
        return m;
    }

    private static Lfm3vlPreprocessor preprocessor() {
        ImageProcessorParams params = ImageProcessorParams.fromGguf(GgufKv.ofMap(metadata()),
                ImageTokenBudget.none());
        return (Lfm3vlPreprocessor) ImagePreprocessors.forParams(params);
    }

    @Test
    public void theVideoSizeFollowsTheReferenceProcessor() {
        Lfm3vlPreprocessor pre = preprocessor();
        // 1000x800 already snaps to the 32-pixel multiple and stays within the fixed video limits
        assertEquals(new ImageSize(992, 800), pre.videoFrameSize(10, 1000, 800));
        assertEquals(new ImageSize(96, 64), pre.videoFrameSize(4, 100, 60));
        // 640x480 is already on the merged patch grid
        assertEquals(new ImageSize(640, 480), pre.videoFrameSize(8, 640, 480));
        // an edge below one merged patch first scales up, then the fixed 4096-pixel floor raises the sides
        assertEquals(new ImageSize(32, 32), pre.videoFrameSize(4, 1, 1));
        // 4000x3000 over 120 frames exceeds the fixed 25165824 maximum pixel count
        assertEquals(new ImageSize(512, 384), pre.videoFrameSize(120, 4000, 3000));
    }

    @Test
    public void aFrameTheDecodeSizedOnlySnapsToTheMergedPatchGrid() {
        Lfm3vlPreprocessor pre = preprocessor();
        // a frame the decode already sized only snaps to the merged patch multiple; the still-image path would
        // stretch it to the params' 65536-pixel minimum instead
        PreparedImages video = pre.prepareVideoFrame(RasterImage.filled(28, 28, 1, 2, 3));
        assertEquals(1, video.encodedTiles().size());
        assertEquals(32, video.encodedTiles().get(0).width());
        assertEquals(32, video.encodedTiles().get(0).height());
        // the still-image path applies the params' pixel limits instead
        assertEquals(256, pre.prepare(RasterImage.filled(8, 8, 1, 2, 3)).encodedTiles().get(0).width());
    }
}
