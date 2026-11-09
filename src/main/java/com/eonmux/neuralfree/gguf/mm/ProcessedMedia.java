package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.image.FloatImage;
import com.eonmux.neuralfree.gguf.image.ImageTile;
import com.eonmux.neuralfree.gguf.image.PreparedImages;
import com.eonmux.neuralfree.gguf.image.RasterImage;
import com.eonmux.neuralfree.gguf.video.DecodedVideo;

/**
 * A processed medium: an image's prepared tiles, an audio clip's decoded samples or a video's sampled frames,
 * and the bytes they hold. Immutable.
 */
final class ProcessedMedia {
    final PreparedImages image;
    final AudioClip audio;
    final DecodedVideo video;
    final long bytes;

    private ProcessedMedia(PreparedImages image, AudioClip audio, DecodedVideo video, long bytes) {
        this.image = image;
        this.audio = audio;
        this.video = video;
        this.bytes = bytes;
    }

    /** Prepared tiles, sized by their pixels: 3 bytes per pixel, or 3 floats for a tile normalized early. */
    static ProcessedMedia image(PreparedImages prepared) {
        long bytes = 0;
        for (ImageTile tile : prepared.encodedTiles()) {
            long channels = 3L * tile.width() * tile.height();
            bytes += tile instanceof FloatImage ? channels * Float.BYTES : channels;
        }
        return new ProcessedMedia(prepared, null, null, bytes);
    }

    /** Decoded samples, sized by their floats. */
    static ProcessedMedia audio(AudioClip clip) {
        return new ProcessedMedia(null, clip, null, (long) clip.length() * Float.BYTES);
    }

    /** Sampled video frames and their soundtrack slices, sized by their RGB bytes and sample floats. */
    static ProcessedMedia video(DecodedVideo video) {
        long bytes = 0;
        for (RasterImage frame : video.frames()) {
            bytes += 3L * frame.width() * frame.height();
        }
        for (AudioClip clip : video.soundtrack()) {
            bytes += (long) clip.length() * Float.BYTES;
        }
        return new ProcessedMedia(null, null, video, bytes);
    }

    long bytes() {
        return bytes;
    }
}
