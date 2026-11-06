package com.eonmux.neuralfree.gguf.image;

/**
 * One model-specific image preprocessing pipeline ({@code mtmd_image_preprocessor}, tools/mtmd/mtmd-image.h):
 * it turns a decoded image into the tiles the projector's vision encoder consumes, in the order they encode,
 * with the token counts the prompt builder needs. Implementations are stateless and thread-safe.
 */
public interface ImagePreprocessor {

    /**
     * Preprocesses a decoded image.
     *
     * @throws IllegalArgumentException when the image cannot serve the projector's geometry (named in the message)
     */
    PreparedImages prepare(RasterImage image);

    /**
     * The frame size a video of {@code nFrames} sampled frames decodes at, the counterpart of
     * {@code mtmd_image_preprocessor::get_video_frame_size}: the video helper hands this size to ffmpeg so the kept
     * frames arrive already resized, exactly as llama.cpp's mtmd helper does.
     *
     * @return the recommended frame size, or null when the family recommends none and the frames keep the source
     *         size
     */
    default ImageSize videoFrameSize(int nFrames, int width, int height) {
        return null;
    }

    /**
     * Preprocesses one frame of a video, whose decode may already have sized it to
     * {@link #videoFrameSize(int, int, int)}; the counterpart of {@code mtmd_image_preprocessor::
     * preprocess_video_frame}. The default runs {@link #prepare(RasterImage)}.
     *
     * @throws IllegalArgumentException when the frame cannot serve the projector's geometry (named in the message)
     */
    default PreparedImages prepareVideoFrame(RasterImage frame) {
        return prepare(frame);
    }
}
