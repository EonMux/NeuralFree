package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.media.MediaInputException;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The soundtrack rows of one LFM3-Omni video, the reference's {@code add_soundtrack_piece} (tools/mtmd/mtmd.cpp):
 * the temporal groups' slices concatenate into one clip, the audio tower encodes it once, and each group takes the
 * rows from its accumulated samples' position to the next group's, the last group the rest of the track. The rows
 * per row of audio are the reference's {@code 8 * audio_hop_len}. Static methods only.
 */
final class VideoSoundtrack {

    /** Mel frames whose audio the reference folds into one embedding row. */
    private static final int MEL_FRAMES_PER_ROW = 8;

    private VideoSoundtrack() {
    }

    /**
     * The embedding rows of every group's span.
     *
     * @param slices the group slices, in group order (empty clips allowed)
     * @param encoder the projector's audio tower
     * @param field  the request field the video came from, named in every refusal
     * @return one row block per group, a group outside the encoded track's rows having none
     * @throws MediaInputException when the tower encodes the soundtrack as more than one window or names no
     *                             samples per row
     */
    static float[][][] rows(List<AudioClip> slices, CpuAudioEncoder encoder, String field) {
        Objects.requireNonNull(slices, "slices");
        Objects.requireNonNull(encoder, "encoder");
        Objects.requireNonNull(field, "field");
        int total = 0;
        for (AudioClip clip : slices) {
            total += clip.length();
        }
        if (total == 0 || slices.isEmpty()) {
            return emptyRows(slices.size());
        }
        float[] samples = new float[total];
        int at = 0;
        for (AudioClip clip : slices) {
            System.arraycopy(clip.samples(), 0, samples, at, clip.length());
            at += clip.length();
        }
        AudioClip track = new AudioClip(samples, slices.get(0).sampleRate());
        List<float[][]> blocks = encoder.embed(track);
        if (blocks.size() != 1) {
            throw new MediaInputException(field + ": the video soundtrack encodes to " + blocks.size() + " audio"
                    + " windows; the " + encoder.model().hparams().projectorType() + " tower encodes it as one");
        }
        float[][] rows = blocks.get(0);
        ClipAudioHparams audio = encoder.model().hparams().audio();
        double samplesPerRow = (double) MEL_FRAMES_PER_ROW * audio.hopLen();
        if (samplesPerRow <= 0) {
            throw new MediaInputException(field + ": the " + encoder.model().hparams().projectorType() + " audio"
                    + " tower names no samples per row, so the soundtrack cannot follow the video's frames");
        }
        float[][][] out = new float[slices.size()][][];
        int begin = 0;
        int cumulative = 0;
        for (int g = 0; g < slices.size(); g++) {
            cumulative += slices.get(g).length();
            int end = g == slices.size() - 1 ? rows.length
                    : (int) Math.max(begin, Math.min(rows.length, Math.rint(cumulative / samplesPerRow)));
            out[g] = end > begin ? Arrays.copyOfRange(rows, begin, end) : new float[0][];
            begin = end;
        }
        return out;
    }

    private static float[][][] emptyRows(int groups) {
        float[][][] out = new float[groups][][];
        for (int g = 0; g < groups; g++) {
            out[g] = new float[0][];
        }
        return out;
    }
}
