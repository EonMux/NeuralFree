package com.eonmux.neuralfree.gguf.audio;

import java.util.Locale;

/**
 * A decoded mono PCM signal: {@code samples[i]} is the amplitude of frame {@code i} at {@code sampleRate} Hz.
 * Immutable; the samples are finite and stored in a private array.
 */
public final class AudioClip {

    private final float[] samples;
    private final int sampleRate;

    /**
     * Takes ownership of nothing: {@code samples} is copied.
     *
     * @throws IllegalArgumentException when the rate is not positive or a sample is not finite
     */
    public AudioClip(float[] samples, int sampleRate) {
        requireRate(sampleRate);
        if (samples == null) {
            throw new IllegalArgumentException("samples must not be null");
        }
        for (int i = 0; i < samples.length; i++) {
            if (!Float.isFinite(samples[i])) {
                throw new IllegalArgumentException("sample " + i + " is not finite (" + samples[i] + ")");
            }
        }
        this.samples = samples.clone();
        this.sampleRate = sampleRate;
    }

    private static void requireRate(int sampleRate) {
        if (sampleRate <= 0) {
            throw new IllegalArgumentException("sample rate must be positive, got " + sampleRate);
        }
    }

    /** Frame amplitudes; the caller must not modify the returned array. */
    public float[] samples() {
        return samples;
    }

    /** Frames in the clip. */
    public int length() {
        return samples.length;
    }

    /** Frames per second. */
    public int sampleRate() {
        return sampleRate;
    }

    /** Duration in seconds. */
    public double durationSeconds() {
        return (double) samples.length / sampleRate;
    }

    /** A silent clip of {@code length} frames at {@code sampleRate}. */
    public static AudioClip silent(int length, int sampleRate) {
        requireRate(sampleRate);
        if (length < 0) {
            throw new IllegalArgumentException("length must be non-negative, got " + length);
        }
        return new AudioClip(new float[length], sampleRate);
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "AudioClip{%d frames @ %d Hz, %.2fs}",
                samples.length, sampleRate, durationSeconds());
    }

}
