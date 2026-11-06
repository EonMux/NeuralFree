package com.eonmux.neuralfree.gguf.audio;

import java.util.Objects;

/**
 * A log-mel spectrogram, or one encoder-ready chunk of it: {@code data[m * nLen + t]} is mel bin {@code m} in frame
 * {@code t} (mel-major, the layout llama.cpp's {@code mtmd_audio_mel} copies into the encoder input unchanged).
 * {@link LogMel} computes them and {@link MelFrontend} cuts them into encoder chunks. Immutable to its users.
 */
public final class MelChunk {

    private final int nMel;
    private final int nLen;
    private final float[] data;

    MelChunk(int nMel, int nLen, float[] data) {
        this.nMel = nMel;
        this.nLen = nLen;
        this.data = data;
    }

    /**
     * A chunk over a copy of {@code data}.
     *
     * @throws IllegalArgumentException for a non-positive size or a data length other than {@code nMel * nLen}
     */
    public static MelChunk of(int nMel, int nLen, float[] data) {
        Objects.requireNonNull(data, "data");
        if (nMel <= 0 || nLen <= 0) {
            throw new IllegalArgumentException("a mel chunk needs positive sizes, got " + nMel + " x " + nLen);
        }
        if ((long) nMel * nLen != data.length) {
            throw new IllegalArgumentException("a " + nMel + " x " + nLen + " mel chunk needs " + (long) nMel * nLen
                    + " values, got " + data.length);
        }
        return new MelChunk(nMel, nLen, data.clone());
    }

    /** Mel bin count. */
    public int nMel() {
        return nMel;
    }

    /** Frame count. */
    public int nLen() {
        return nLen;
    }

    /** The mel-major values, shared with the chunk; treat the array as read-only. */
    public float[] data() {
        return data;
    }

    /** Value of bin {@code m} in frame {@code t}. */
    public float value(int m, int t) {
        if (m < 0 || m >= nMel || t < 0 || t >= nLen) {
            throw new IndexOutOfBoundsException("mel value (" + m + ',' + t + ") outside " + nMel + 'x' + nLen);
        }
        return data[m * nLen + t];
    }
}
