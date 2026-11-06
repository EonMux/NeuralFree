package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

/**
 * The preprocessor factory, the projector switch of llama.cpp's {@code mtmd_context::init_vision}
 * (tools/mtmd/mtmd.cpp): the {@link VisionPreprocessing.Family} of the params' projector picks its preprocessor.
 * Projector types without image preprocessing are refused by name when their params load
 * ({@link ImageProcessorParams#fromGguf}).
 */
public final class ImagePreprocessors {

    private ImagePreprocessors() {
    }

    /**
     * The preprocessor for the params' preprocessing family: the fixed-size, llava-uhd, dynamic-size or
     * longest-edge pipeline, with the family variants (MiniCPM-V, Cohere2, Idefics3, LFM2) or the family's own
     * pipeline (Step3-VL, GLM-5V, Youtu-VL, LFM3-VL, DeepSeek-OCR, DeepSeek-V4, Granite Vision anyres, Muse
     * Glimmer).
     */
    public static ImagePreprocessor forParams(ImageProcessorParams params) {
        switch (params.preprocessing().family()) {
            case LLAVA:
            case GLM_EDGE:
            case JANUS_PRO:
                // with grid pinpoints the llava family slices (llava-1.6); without, one fixed tile
                return params.gridPinpoints().isEmpty()
                        ? new FixedSizePreprocessor(params)
                        : new LlavaUhdPreprocessor(params);
            case MINICPMV:
            case MINICPMV_MERGER:
            case MINICPM_OCR:
                return new MiniCpmvPreprocessor(params);
            case LLAMA4:
            case INTERNVL:
                return new LlavaUhdPreprocessor(params);
            case COHERE2V:
                return new Cohere2vPreprocessor(params);
            case IDEFICS3:
                return new Idefics3Preprocessor(params);
            case LFM2:
                return new Lfm2Preprocessor(params);
            case STEP3VL:
                return new Step3vlPreprocessor(params);
            case DEEPSEEK_OCR:
            case DEEPSEEK_OCR2:
                return new DeepseekOcrPreprocessor(params);
            case DEEPSEEK4V:
            case DEEPSEEK41V:
                return new DeepseekV4Preprocessor(params);
            case GRANITE4V:
                return new Granite4AnyresPreprocessor(params);
            case GEMMA3:
            case YASA2:
            case NEMOTRON_V2_VL:
                return new FixedSizePreprocessor(params);
            case PIXTRAL:
                return params.longestEdge() > 0
                        ? new LongestEdgePreprocessor(params)
                        : new DynSizePreprocessor(params);
            case LIGHTONOCR:
                return new LongestEdgePreprocessor(params);
            case GLM5V:
                return new Glm5vPreprocessor(params);
            case YOUTUVL:
                return new YoutuVlPreprocessor(params);
            case LFM3VL:
                return new Lfm3vlPreprocessor(params);
            case MUSE_GLIMMER:
                return new MuseGlimmerPreprocessor(params);
            case QWEN_MERGER:
            case MIMOVL:
            case GLM4V:
            case HUNYUANVL:
            case EXAONE4_5:
            case PADDLEOCR:
            case DOTS_OCR:
            case DOTS3NOTE_V:
            case NEMOTRON_OMNI:
            case KIMIVL:
            case PHI4:
            case GEMMA4V:
            case GEMMA4UV:
            case MINIMAX_M3:
            case SENSENOVA_U1:
                return new DynSizePreprocessor(params);
            default:
                throw new IllegalStateException("unhandled preprocessing family " + params.preprocessing().family()
                        + " of projector type '" + params.projectorName() + "'");
        }
    }
}
