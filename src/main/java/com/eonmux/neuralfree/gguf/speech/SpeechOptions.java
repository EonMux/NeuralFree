package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.audio.AudioClip;
import com.eonmux.neuralfree.gguf.engine.SamplerConfig;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The options of one speech synthesis: the caller-facing fields of llama.cpp's {@code /tts} body (server-context.cpp
 * {@code post_tts}) that reach a {@link SpeechGenerator}.
 *
 * <p>{@code language} selects the codec language token of a qwen3tts backbone and names the language a pockettts
 * pack speaks (a request for another language is refused by name). It accepts the ISO codes llama.cpp
 * maps ({@code zh}, {@code en}, {@code de}, {@code it}, {@code pt}, {@code es}, {@code ja}, {@code ko}, {@code fr},
 * {@code ru}, mtmd-helper-gen.cpp:27-41) as well as the full names themselves; an empty value leaves the
 * generator's default, English for qwen3tts, the pack's own language for pockettts, and the speech template's own
 * default for lfm3a. {@code voice} names a generator-provided speaker preset; a generator that has no presets
 * refuses a non-empty voice by name rather than ignoring it, while a speech template binds it as the {@code voice}
 * variable. {@code seed} drives the generator's draws (the backbone and code predictor of qwen3tts, the flow
 * noise of pockettts, the text steps of an lfm3a turn), {@code topK}/{@code topP} the code predictor's truncated
 * sampling (pockettts decodes continuous latents, so they do not apply), and {@code maxFrames} the frame cap of
 * the generation (llama.cpp's default {@code n_predict} of 512). {@code chunkFrames} asks a streaming caller for
 * decoder windows of that many frames; 0 leaves each generator's own window. {@code speakerReference} is the
 * reference waveform a generator with a speaker encoder clones the voice from (llama.cpp's {@code speaker_ref});
 * a generator without one refuses it by name rather than ignoring it. {@code sampler} replaces the text-step
 * sampler a generator would build from the other fields, so a server request's whole sampling configuration
 * reaches the model.
 *
 * <p>Immutable; validate ranges at construction.
 */
public final class SpeechOptions {

    /** llama.cpp's default number of generated frames when the body names none ({@code n_predict}, 512). */
    public static final int DEFAULT_MAX_FRAMES = 512;
    /** mtmd-helper-gen.cpp's {@code mtmd_gen_inp_default} top-k of the code predictor. */
    public static final int DEFAULT_TOP_K = 50;
    /** mtmd-helper-gen.cpp's {@code mtmd_gen_inp_default} top-p of the code predictor. */
    public static final float DEFAULT_TOP_P = 1.0f;
    /** The default seed: a fixed draw, so a request without a seed repeats itself. */
    public static final long DEFAULT_SEED = 0L;

    private static final Map<String, String> LANGUAGE_CODES;

    static {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("zh", "chinese");
        m.put("en", "english");
        m.put("de", "german");
        m.put("it", "italian");
        m.put("pt", "portuguese");
        m.put("es", "spanish");
        m.put("ja", "japanese");
        m.put("ko", "korean");
        m.put("fr", "french");
        m.put("ru", "russian");
        LANGUAGE_CODES = Collections.unmodifiableMap(m);
    }

    private final String language;
    private final String voice;
    private final long seed;
    private final int topK;
    private final float topP;
    private final int maxFrames;
    private final int chunkFrames;
    private final AudioClip speakerReference;
    private final SamplerConfig sampler;

    private SpeechOptions(String language, String voice, long seed, int topK, float topP, int maxFrames,
                          int chunkFrames, AudioClip speakerReference, SamplerConfig sampler) {
        this.language = language;
        this.voice = voice;
        this.seed = seed;
        this.topK = topK;
        this.topP = topP;
        this.maxFrames = maxFrames;
        this.chunkFrames = chunkFrames;
        this.speakerReference = speakerReference;
        this.sampler = sampler;
    }

    /** The defaults: English, no voice, seed 0, top-k 50, top-p 1, 512 steps, the generator's window. */
    public static SpeechOptions defaults() {
        return new SpeechOptions("", "", DEFAULT_SEED, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_MAX_FRAMES, 0, null,
                null);
    }

    /** {@link #defaults()} with {@code language}, {@code voice}, {@code seed}, {@code topK} and {@code topP}. */
    public static SpeechOptions of(String language, String voice, long seed, int topK, float topP) {
        return defaults().withLanguage(language).withVoice(voice).withSeed(seed).withTopK(topK).withTopP(topP);
    }

    /** A copy speaking {@code language}; null and blank speak the reference's default, English. */
    public SpeechOptions withLanguage(String language) {
        String value = language == null ? "" : language.trim();
        return new SpeechOptions(value, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference, sampler);
    }

    /** A copy naming the speaker preset {@code voice}; null and blank mean none. */
    public SpeechOptions withVoice(String voice) {
        return new SpeechOptions(language, voice == null ? "" : voice.trim(), seed, topK, topP, maxFrames,
                chunkFrames, speakerReference, sampler);
    }

    /** A copy seeded with {@code seed}. */
    public SpeechOptions withSeed(long seed) {
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /**
     * A copy sampling with {@code topK} and {@code topP}.
     *
     * @throws IllegalArgumentException when {@code topK} is negative or {@code topP} outside {@code [0, 1]}
     */
    public SpeechOptions withTopK(int topK) {
        if (topK < 0) {
            throw new IllegalArgumentException("\"top_k\" must not be negative, got " + topK);
        }
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /** {@link #withTopK(int)} with the top-p value. */
    public SpeechOptions withTopP(float topP) {
        if (!(topP >= 0f) || topP > 1f || Float.isNaN(topP)) {
            throw new IllegalArgumentException("\"top_p\" must be in [0, 1], got " + topP);
        }
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /**
     * A copy generating at most {@code maxFrames} steps: speech-code frames, and the text tokens a turn samples
     * between them (llama.cpp's {@code n_predict}).
     *
     * @throws IllegalArgumentException when {@code maxFrames} is below one
     */
    public SpeechOptions withMaxFrames(int maxFrames) {
        if (maxFrames < 1) {
            throw new IllegalArgumentException("\"max_tokens\" must be at least 1 step, got " + maxFrames);
        }
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /**
     * A copy asking a streaming caller for chunks of {@code chunkFrames} frames.
     *
     * @throws IllegalArgumentException when {@code chunkFrames} is negative
     */
    public SpeechOptions withChunkFrames(int chunkFrames) {
        if (chunkFrames < 0) {
            throw new IllegalArgumentException("the stream chunk must not be negative, got " + chunkFrames);
        }
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /**
     * A copy cloning the voice of {@code reference}; null clears it.
     *
     * @throws IllegalArgumentException when the clip is empty
     */
    public SpeechOptions withSpeakerReference(AudioClip reference) {
        if (reference != null && reference.length() == 0) {
            throw new IllegalArgumentException("the speaker reference is empty");
        }
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, reference, sampler);
    }

    /**
     * A copy whose text steps sample with {@code sampler} instead of the {@code seed}/{@code topK}/{@code topP}
     * above; null restores the built configuration.
     */
    public SpeechOptions withSampler(SamplerConfig sampler) {
        return new SpeechOptions(language, voice, seed, topK, topP, maxFrames, chunkFrames, speakerReference,
                sampler);
    }

    /**
     * The codec language name of {@code language}: the ISO code mapped as llama.cpp's {@code tts_lang_codes} maps
     * it, otherwise the value as written; blank is {@code english}.
     *
     * @throws IllegalArgumentException when the value is not a known ISO code and holds characters outside lower-case
     *                                  letters, digits, {@code _} or {@code -}
     */
    public static String resolveLanguage(String language) {
        if (language == null || language.trim().isEmpty()) {
            return "english";
        }
        String value = language.trim();
        String mapped = LANGUAGE_CODES.get(value.toLowerCase(Locale.ROOT));
        if (mapped != null) {
            return mapped;
        }
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'
                    || c == '-')) {
                throw new IllegalArgumentException("language \"" + language + "\" is neither an ISO code ("
                        + String.join(", ", LANGUAGE_CODES.keySet()) + ") nor a language name");
            }
        }
        return value.toLowerCase(Locale.ROOT);
    }

    /** The language as given; blank speaks English. */
    public String language() {
        return language;
    }

    /** The speaker preset, or an empty string when none is named. */
    public String voice() {
        return voice;
    }

    /** True when a speaker preset is named. */
    public boolean hasVoice() {
        return !voice.isEmpty();
    }

    /** The draw seed. */
    public long seed() {
        return seed;
    }

    /** The code predictor's top-k; 0 disables the cut. */
    public int topK() {
        return topK;
    }

    /** The code predictor's top-p; 1 disables the cut. */
    public float topP() {
        return topP;
    }

    /** The frame cap of the generation. */
    public int maxFrames() {
        return maxFrames;
    }

    /** The streaming chunk in frames, or 0 for the generator's own window. */
    public int chunkFrames() {
        return chunkFrames;
    }

    /** The reference waveform to clone the voice from, or null when none was given. */
    public AudioClip speakerReference() {
        return speakerReference;
    }

    /** The text-step sampler override, or null when the generator builds one from the fields above. */
    public SamplerConfig sampler() {
        return sampler;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SpeechOptions)) {
            return false;
        }
        SpeechOptions other = (SpeechOptions) o;
        return seed == other.seed && topK == other.topK && Float.compare(topP, other.topP) == 0
                && maxFrames == other.maxFrames && chunkFrames == other.chunkFrames
                && language.equals(other.language) && voice.equals(other.voice)
                && sameReference(speakerReference, other.speakerReference)
                && Objects.equals(sampler, other.sampler);
    }

    /** Two references are the same clip: the same rate and samples, or the same object (or both null). */
    private static boolean sameReference(AudioClip a, AudioClip b) {
        if (a == b) {
            return true;
        }
        return a != null && b != null && a.sampleRate() == b.sampleRate()
                && java.util.Arrays.equals(a.samples(), b.samples());
    }

    @Override
    public int hashCode() {
        int reference = speakerReference == null ? 0
                : 31 * speakerReference.sampleRate() + java.util.Arrays.hashCode(speakerReference.samples());
        return Objects.hash(language, voice, seed, topK, topP, maxFrames, chunkFrames, reference, sampler);
    }

    @Override
    public String toString() {
        return "SpeechOptions{language='" + language + "', voice='" + voice + "', seed=" + seed + ", topK=" + topK
                + ", topP=" + topP + ", maxFrames=" + maxFrames + ", chunkFrames=" + chunkFrames
                + ", speakerReference=" + (speakerReference == null ? "none" : speakerReference)
                + (sampler == null ? "" : ", sampler=" + sampler) + '}';
    }
}
