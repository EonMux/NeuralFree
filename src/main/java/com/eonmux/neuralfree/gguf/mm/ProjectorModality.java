package com.eonmux.neuralfree.gguf.mm;

/**
 * Modality of one clip-file encoder (llama.cpp {@code clip_modality}, tools/mtmd/clip-model.h). An mmproj file can
 * hold one encoder per media modality; the encoder-level metadata keys of {@code ClipHparams} are prefixed by the
 * modality ({@code clip.vision.*}, {@code clip.audio.*}, {@code clip.gen.audio.*}). {@link #TEXT} is the text tower
 * of a CLIP two-tower embedding file ({@link ClipDualEncoder}), which no mmproj encoder opens for.
 */
public enum ProjectorModality {

    /** Image input (the {@code clip.vision.*} keys, tensor prefix {@code v.}). */
    VISION,

    /** Audio input (the {@code clip.audio.*} keys, tensor prefix {@code a.}). */
    AUDIO,

    /** Audio generation (the {@code clip.gen.audio.*} keys, tensor prefix {@code a.gen.}). */
    GEN_AUDIO,

    /** The text tower of a CLIP two-tower file (the {@code clip.text.*} keys, tensor prefix {@code t.}). */
    TEXT;

    /** True for the modalities an mmproj file feeds a language model with: every modality but {@link #TEXT}. */
    public boolean media() {
        return this != TEXT;
    }

    /** The {@code clip.%s.*} metadata prefix of this modality (llama.cpp {@code load_hparams}). */
    public String keyPrefix() {
        switch (this) {
            case VISION:
                return "vision";
            case AUDIO:
                return "audio";
            case TEXT:
                return "text";
            default:
                return "gen.audio";
        }
    }

    /** The GGUF tensor prefix of this modality. */
    public String tensorPrefix() {
        switch (this) {
            case VISION:
                return "v";
            case AUDIO:
                return "a";
            case TEXT:
                return "t";
            default:
                return "a.gen";
        }
    }
}
