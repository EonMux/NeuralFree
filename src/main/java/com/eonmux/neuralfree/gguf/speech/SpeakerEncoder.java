package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;

/**
 * A speaker (voice-prompt) encoder of a speech generator file: a reference waveform becomes the conditioning the
 * generator's backbone consumes. It is the counterpart of llama.cpp's audio encoder half of an mtmd projector
 * (tools/mtmd/models/qwen3tts-spkenc.cpp and pockettts-spkenc.cpp): a qwen3tts file yields one x-vector row of the
 * backbone width, a pockettts file yields one row per 12.5 Hz model frame.
 *
 * <p>An implementation is immutable over bound host weights, so {@link #encode(AudioClip)} may run on any thread;
 * it is owned by the generator file that opened it.</p>
 */
public interface SpeakerEncoder {

    /** The family name, for logs ({@code qwen3tts_spkenc}, {@code pockettts_spkenc}). */
    String name();

    /** The sample rate {@link #encode(AudioClip)} expects. */
    int sampleRate();

    /** The width of every returned row, the backbone's embedding width. */
    int width();

    /**
     * Encodes one reference clip.
     *
     * @param clip a mono clip at {@link #sampleRate()}; a clip at another rate is resampled
     * @return one or more rows of {@link #width()} values in the backbone's embedding space
     * @throws IllegalArgumentException when the clip is empty or too short for the encoder's front-end
     */
    float[][] encode(AudioClip clip);
}
