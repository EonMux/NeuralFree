package com.eonmux.neuralfree.gguf.mm;

import java.util.Objects;

/**
 * The image preprocessing of a vision projector type: the preprocessing family whose pipeline and defaults prepare
 * its images (llama.cpp {@code clip_model_loader::load_hparams} and {@code mtmd_context::init_vision}), the
 * formula that turns one prepared tile into image tokens ({@code clip_n_output_tokens}), the grid those tokens lie
 * on ({@code clip_n_output_tokens_x/y}), and the rule its per-side patch merge is read with. Every
 * {@link ProjectorType} NeuralFree preprocesses owns one; a plugin row declares one for its own registry key through
 * {@link ProjectorEntry}. Immutable.
 */
public final class VisionPreprocessing {

    /**
     * The preprocessing families: each fixes the preprocessing pipeline (fixed size, llava-uhd slicing, dynamic
     * resolution, longest edge) and the family defaults of the resize algorithm, padding, pixel limits and tiling.
     * Named after the projector type that leads the family.
     */
    public enum Family {
        /** LLaVA projectors and CogVLM: one fixed tile, or llava-1.6 slicing over the grid pinpoints. */
        LLAVA,
        /** GLM-Edge: the LLaVA pipeline padded with black. */
        GLM_EDGE,
        /** Janus-Pro: the LLaVA pipeline padded with gray. */
        JANUS_PRO,
        /** MiniCPM-V resampler: llava-uhd slicing, one fixed query count per tile. */
        MINICPMV,
        /** MiniCPM-V 4.6/4.7 unified merger: MiniCPM-V slicing aligned to four patches, without padding. */
        MINICPMV_MERGER,
        /** MiniCPM-OCR 1.0: the MiniCPM-V merger slicing aligned to the merged patch. */
        MINICPM_OCR,
        /** InternVL: dynamic-resolution tile candidates, overview last. */
        INTERNVL,
        /** Idefics3 / SmolVLM: longest-edge resize, then {@code image_size} tiles with the overview last. */
        IDEFICS3,
        /** LLaMA 4 vision: llava-uhd slicing over the up-to-3x3 tile candidates, overview last. */
        LLAMA4,
        /** LFM2-VL and d1-omni: a smart resize, or {@code preproc_tile_size} tiles and a thumbnail when large. */
        LFM2,
        /** LFM3-VL: the transformers smart resize in double precision, required pixel limits. */
        LFM3VL,
        /** Step3-VL: window crops of a capped image, float bilinear tiles, overview last. */
        STEP3VL,
        /** LightOnOCR: the longest-edge resize, the edge defaulting to {@code image_size}. */
        LIGHTONOCR,
        /** Gemma 4 vision: bicubic smart resize without padding. */
        GEMMA4V,
        /** Gemma 4 unified vision: as Gemma 4, the patch convolution merging the pooling kernel. */
        GEMMA4UV,
        /** GLM-5V: an aligned canvas fitted to the pixel budget, the image at its top-left. */
        GLM5V,
        /** Youtu-VL: a linear scale search under the patch budget. */
        YOUTUVL,
        /** DeepSeek-OCR: a padded global view and aspect-ratio tiles fused per row, overview last. */
        DEEPSEEK_OCR,
        /** DeepSeek-OCR 2: a padded global view and separate aspect-ratio tiles, overview last. */
        DEEPSEEK_OCR2,
        /** dots3-note: smart resize, required pixel limits. */
        DOTS3NOTE_V,
        /** MiniMax-M3: smart resize without padding. */
        MINIMAX_M3,
        /** SenseNova-U1: bicubic dynamic resolution, 16 to 4096 merged tokens behind required pixel keys. */
        SENSENOVA_U1,
        /**
         * DeepSeek-V4: the largest patch-aligned size whose token block fits the cap (room kept for the lead pads),
         * padded around the image with gray, or stretched past the aspect ratio limit.
         */
        DEEPSEEK4V,
        /** DeepSeek-V4.1: the reading-order block's grid plan, the image fitted and centered on gray. */
        DEEPSEEK41V,
        /**
         * Granite Vision 4.1: llava-next anyres over the grid pinpoints, the overview and every tile of the best
         * resolution encoded as one image whose tile grid is cropped to the original aspect ratio.
         */
        GRANITE4V,
        /** Muse Glimmer: the merged patch grid closest to the aspect ratio under the token cap. */
        MUSE_GLIMMER,
        /** Cohere2 vision: its own tile-grid choice, tiles before the overview. */
        COHERE2V,
        /** Gemma 3 vision: one fixed bilinear tile. */
        GEMMA3,
        /** Qwen2-VL, Qwen2.5-VL and Qwen3-VL mergers: smart resize, optional pixel limits. */
        QWEN_MERGER,
        /** MiMo-VL: the Qwen smart resize with required pixel limits. */
        MIMOVL,
        /** GLM-4V: smart resize with padding, optional pixel limits. */
        GLM4V,
        /** HunyuanVL: Lanczos smart resize, optional pixel limits. */
        HUNYUANVL,
        /** EXAONE 4.5: smart resize, required pixel limits. */
        EXAONE4_5,
        /** PaddleOCR-VL: smart resize with padding, required pixel limits. */
        PADDLEOCR,
        /** dots.ocr: smart resize, required pixel limits. */
        DOTS_OCR,
        /** Nemotron v2 VL: one fixed tile. */
        NEMOTRON_V2_VL,
        /** Nemotron Omni: dynamic resolution without padding, required pixel limits. */
        NEMOTRON_OMNI,
        /** Kimi-VL and Kimi K2.5: dynamic resolution, optional pixel limits. */
        KIMIVL,
        /** Phi-4 multimodal: bilinear dynamic resolution, required pixel limits. */
        PHI4,
        /** Pixtral: dynamic resolution, or the longest-edge resize when the file sets one. */
        PIXTRAL,
        /** YASA2: one fixed tile stretched without padding. */
        YASA2
    }

    private final Family family;
    private final ImageTokenFormula tokenFormula;
    private final ImageTokenGrid tokenGrid;
    private final PatchMergeRule merge;

    private VisionPreprocessing(Family family, ImageTokenFormula tokenFormula, ImageTokenGrid tokenGrid,
                                PatchMergeRule merge) {
        this.family = family;
        this.tokenFormula = tokenFormula;
        this.tokenGrid = tokenGrid;
        this.merge = merge;
    }

    /**
     * The preprocessing of {@code family} counting tokens with {@code tokenFormula} on the formula's
     * {@link ImageTokenFormula#defaultGrid() default grid}, merging by {@code merge}.
     */
    public static VisionPreprocessing of(Family family, ImageTokenFormula tokenFormula, PatchMergeRule merge) {
        Objects.requireNonNull(tokenFormula, "tokenFormula");
        return of(family, tokenFormula, tokenFormula.defaultGrid(), merge);
    }

    /** The preprocessing of {@code family} counting tokens with {@code tokenFormula} on {@code tokenGrid}. */
    public static VisionPreprocessing of(Family family, ImageTokenFormula tokenFormula, ImageTokenGrid tokenGrid,
                                         PatchMergeRule merge) {
        return new VisionPreprocessing(Objects.requireNonNull(family, "family"),
                Objects.requireNonNull(tokenFormula, "tokenFormula"), Objects.requireNonNull(tokenGrid, "tokenGrid"),
                Objects.requireNonNull(merge, "merge"));
    }

    /** The preprocessing family. */
    public Family family() {
        return family;
    }

    /** How one prepared tile turns into image tokens. */
    public ImageTokenFormula tokenFormula() {
        return tokenFormula;
    }

    /** The grid one tile's tokens lie on. */
    public ImageTokenGrid tokenGrid() {
        return tokenGrid;
    }

    /** How the per-side patch merge is read. */
    public PatchMergeRule merge() {
        return merge;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof VisionPreprocessing)) {
            return false;
        }
        VisionPreprocessing other = (VisionPreprocessing) o;
        return family == other.family && tokenFormula == other.tokenFormula && tokenGrid == other.tokenGrid
                && merge.equals(other.merge);
    }

    @Override
    public int hashCode() {
        return Objects.hash(family, tokenFormula, tokenGrid, merge);
    }

    @Override
    public String toString() {
        return "VisionPreprocessing{" + family + ", tokens=" + tokenFormula + ", grid=" + tokenGrid + ", merge="
                + merge + "}";
    }
}
