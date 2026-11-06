package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;

import java.util.function.Consumer;

/**
 * A text-to-speech generator: text plus {@link SpeechOptions} becomes an {@link AudioClip}. The server's
 * {@code /v1/audio/speech} and {@code /tts} routes and the demo's {@code --mode audio} call this interface; no
 * route or command knows a generator family, so the models that plug in here ({@code qwen3tts} and
 * {@code pockettts}) share the same surface.
 *
 * <p>The streaming form delivers the waveform in decoder-window chunks that concatenate to exactly the samples of
 * the one-shot form; a generator that cannot split its output leaves the default, which delivers one clip. An
 * implementation is bound to one text model and one loaded generator file; it is not thread-safe, because it
 * drives the model's forward backend, and is closed by its owner.
 */
public interface SpeechGenerator extends AutoCloseable {

    /** The generator family, for logs and metadata ({@code qwen3tts}, {@code pockettts}). */
    String name();

    /** The sample rate of every clip the generator produces. */
    int sampleRate();

    /**
     * Synthesizes the whole waveform of {@code text}.
     *
     * @param text    the text to speak; must not be blank
     * @param options the language, voice, sampling and frame cap
     * @return the mono waveform at the generator file's sample rate
     * @throws IllegalArgumentException      when {@code text} is blank, {@code options} name a preset the generator
     *                                       does not have, or the request does not fit the model
     * @throws UnsupportedOperationException when the generator cannot speak the text (a missing token, an
     *                                       unsupported language), naming the cause
     */
    AudioClip synthesize(String text, SpeechOptions options);

    /**
     * {@link #synthesize(String, SpeechOptions)} delivered in chunks: every clip holds consecutive samples of the
     * one-shot result, at the same sample rate, and concatenating them gives that result. The default delivers the
     * one-shot clip.
     *
     * @throws IllegalArgumentException      as {@link #synthesize(String, SpeechOptions)}
     * @throws UnsupportedOperationException as {@link #synthesize(String, SpeechOptions)}
     */
    default void synthesize(String text, SpeechOptions options, Consumer<AudioClip> sink) {
        sink.accept(synthesize(text, options));
    }

    /** Releases the generator's file and weights; idempotent. */
    @Override
    default void close() {
    }
}
