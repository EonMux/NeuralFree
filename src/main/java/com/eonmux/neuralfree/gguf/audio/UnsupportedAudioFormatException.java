package com.eonmux.neuralfree.gguf.audio;

/**
 * An audio container or codec the decoders cannot handle. The message names the format so a caller can convert
 * the file (WAV/PCM decodes natively; MP3, FLAC, OGG and M4A decode through an {@code ffmpeg} subprocess when one
 * is installed).
 */
public final class UnsupportedAudioFormatException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    public UnsupportedAudioFormatException(String message) {
        super(message);
    }

    public UnsupportedAudioFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
