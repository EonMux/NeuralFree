package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

import java.util.Locale;
import java.util.Objects;

/**
 * The per-pack generation settings of pockettts that live in the reference's pack YAML rather than in the
 * checkpoint (llama.cpp tools/mtmd/mtmd-helper-gen.cpp:522-545): the flow temperature, the number of frames kept
 * after the end-of-speech probe fires, and whether very short prompts are padded.
 *
 * <p>All three are zero or false for a pack that does not tune them; the pipeline then uses the model defaults
 * (temperature {@value PocketTtsFlow#DEFAULT_TEMPERATURE}, a tail guessed from the prompt length, no padding).
 * The variant is the {@code clip.gen.audio.model_variant} metadata, the checkpoint directory name written by the
 * converter. Immutable.
 */
public final class PocketTtsPack {

    private final String variant;
    private final float temperature;
    private final int framesAfterEos;
    private final boolean padShortText;

    private PocketTtsPack(String variant, float temperature, int framesAfterEos, boolean padShortText) {
        this.variant = variant;
        this.temperature = temperature;
        this.framesAfterEos = framesAfterEos;
        this.padShortText = padShortText;
    }

    /** The settings of {@code variant}; an unlisted variant tunes nothing. */
    public static PocketTtsPack forVariant(String variant) {
        String name = variant == null ? "" : variant.trim();
        switch (name) {
            case "english":
                return new PocketTtsPack(name, 0.3f, 0, false);
            case "english_2026-01":
                return new PocketTtsPack(name, 0.7f, 0, true);
            case "english_2026-04":
                return new PocketTtsPack(name, 0.3f, 0, false);
            case "french_24l":
                return new PocketTtsPack(name, 0.7f, 8, false);
            default:
                return new PocketTtsPack(name, 0f, 0, false);
        }
    }

    /** The pack of an opened generator file ({@link GgufKeys#CLIP_GEN_AUDIO_MODEL_VARIANT}). */
    public static PocketTtsPack of(GgufKv kv) {
        Objects.requireNonNull(kv, "kv");
        return forVariant(kv.str(GgufKeys.CLIP_GEN_AUDIO_MODEL_VARIANT, null));
    }

    /** The variant name, or an empty string when the file names none. */
    public String variant() {
        return variant;
    }

    /** The flow temperature: the pack's value, else {@value PocketTtsFlow#DEFAULT_TEMPERATURE}. */
    public float temperature() {
        return temperature > 0f ? temperature : PocketTtsFlow.DEFAULT_TEMPERATURE;
    }

    /**
     * Frames kept after the end-of-speech probe fires: the pack's value, else 5 for a prompt of at most 6 tokens
     * and 3 otherwise (mtmd-helper-gen.cpp:900-903). The frame of the step that fired the probe is discarded, so
     * this many frames follow it.
     */
    public int framesAfterEos(int nTokens) {
        return framesAfterEos > 0 ? framesAfterEos : (nTokens <= 6 ? 5 : 3);
    }

    /** True when a prompt shorter than 5 words gets eight leading spaces (mtmd-helper-gen.cpp:978-981). */
    public boolean padShortText() {
        return padShortText;
    }

    /** The language the pack speaks: {@code french} for a French variant, {@code english} otherwise. */
    public String language() {
        return variant.toLowerCase(Locale.ROOT).startsWith("french") ? "french" : "english";
    }

    @Override
    public String toString() {
        return "PocketTtsPack{variant='" + variant + "', temp=" + temperature + ", framesAfterEos=" + framesAfterEos
                + ", padShortText=" + padShortText + '}';
    }
}
