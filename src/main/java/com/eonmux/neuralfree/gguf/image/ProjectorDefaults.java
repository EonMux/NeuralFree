package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.mm.ImageTokenFormula;
import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

import java.util.ArrayList;
import java.util.List;

/**
 * The per-family preprocessing defaults of llama.cpp's {@code clip_model_loader::load_hparams}
 * (tools/mtmd/clip.cpp): for the {@link VisionPreprocessing.Family} of the projector it sets the resize algorithm,
 * pad style and color, the dynamic-resolution pixel limits, the resolution candidates and the ordering, then reads
 * the GGUF keys over them and applies the {@link ImageTokenBudget} last (a set token limit replaces the pixel
 * limits, the way {@code clip_hparams::set_limit_image_tokens} gives {@code custom_image_min_tokens}
 * precedence). The per-side merge is already set from the type's patch-merge rule. Package-private;
 * {@link ImageProcessorParams#fromGguf} is the only caller.
 */
final class ProjectorDefaults {

    /** The Step3-VL longest edge when the GGUF sets none ({@code processing_step3.py}). */
    static final int STEP3VL_LONGEST_EDGE = 3024;

    /** The tile grids LLaMA 4 slices into, per side ({@code set_llava_uhd_res_candidates(model, 3)}). */
    private static final int LLAMA4_MAX_TILES_PER_SIDE = 3;

    /** The LFM3-VL token cap, an out-of-memory guard over the processor's pixel limit. */
    private static final int LFM3VL_MAX_TOKENS = 4096;

    /** The DeepSeek-V4 token block cap ({@code vision_max_n_token}, which the converter asserts). */
    static final int DEEPSEEK4V_MAX_TOKENS = 384;

    /** The DeepSeek-V4.1 token block cap ({@code vision_max_n_token}, which the converter asserts). */
    static final int DEEPSEEK41V_MAX_TOKENS = 1024;

    /** The smallest token block cap a budget sets for DeepSeek-V4 and V4.1, room for the resize solver. */
    private static final int DEEPSEEK_V4_MIN_MAX_TOKENS = 16;

    private ProjectorDefaults() {
    }

    static void apply(ImageProcessorParams.Builder b, GgufKv kv, ImageTokenBudget budget) {
        switch (b.preprocessing.family()) {
            case LLAVA:
            case GLM_EDGE:
            case JANUS_PRO:
                applyLlava(b, kv);
                break;
            case MINICPMV:
                // llava-uhd defaults; the overview-only variant applies for small images at preprocess time
                b.minicpmvQueryNum(ImageTokenFormula.resamplerQueryNum(kv));
                readTiles(kv, b);
                break;
            case MINICPMV_MERGER:
            case MINICPM_OCR:
                applyMinicpmvMerger(b, kv);
                break;
            case INTERNVL:
                b.minTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 1));
                b.maxTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, 12));
                b.gridPinpoints(dhrCandidates(b.imageSize, b.minTiles, b.maxTiles));
                b.overviewFirst(false);
                break;
            case LLAMA4:
                applyLlama4(b, kv);
                break;
            case IDEFICS3:
                // Lanczos tiles behind the optional longest edge; the token budget alone sets the pixel limits
                b.resizeAlgo(ResizeAlgo.LANCZOS);
                b.longestEdge(kv.u32(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, 0));
                b.overviewFirst(false);
                break;
            case COHERE2V:
                applyCohere2v(b, kv);
                break;
            case LFM2:
                applyLfm2(b, kv);
                break;
            case STEP3VL:
                b.longestEdge(kv.u32(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, 0));
                if (b.longestEdge == 0) {
                    b.longestEdge(STEP3VL_LONGEST_EDGE);
                }
                b.overviewFirst(false);
                break;
            case DEEPSEEK_OCR:
            case DEEPSEEK_OCR2:
                applyDeepseekOcr(b, kv);
                break;
            case GRANITE4V:
                applyGranite4v(b, kv);
                break;
            case DEEPSEEK4V:
            case DEEPSEEK41V:
                // the family applies the token budget itself
                applyDeepseekV4(b, kv, budget);
                return;
            default:
                applyDynamic(b, kv);
                break;
        }
        applyPixelLimits(b, kv, budget);
        applyBudget(b, budget);
    }

    /** The families that resize the whole image once: resize algorithm, padding and the longest edge. */
    private static void applyDynamic(ImageProcessorParams.Builder b, GgufKv kv) {
        switch (b.preprocessing.family()) {
            case GEMMA3:
            case PHI4:
                b.resizeAlgo(ResizeAlgo.BILINEAR);
                break;
            case QWEN_MERGER:
            case MIMOVL:
            case EXAONE4_5:
            case DOTS_OCR:
                // the HF processor resizes to the smart_resize size without padding, rounding like Python round()
                b.resizeAlgo(ResizeAlgo.BICUBIC).padStyle(PadStyle.NONE).roundHalfEven(true);
                break;
            case HUNYUANVL:
                b.resizeAlgo(ResizeAlgo.LANCZOS).padStyle(PadStyle.NONE);
                break;
            case NEMOTRON_OMNI:
                b.padStyle(PadStyle.NONE);
                break;
            case PIXTRAL:
                b.resizeAlgo(ResizeAlgo.BICUBIC);
                b.longestEdge(kv.u32(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, 0));
                break;
            case LIGHTONOCR:
                b.resizeAlgo(ResizeAlgo.BICUBIC);
                b.longestEdge(kv.u32(GgufKeys.CLIP_VISION_PREPROC_IMAGE_SIZE, b.imageSize));
                break;
            case YASA2:
            case GEMMA4V:
            case MINIMAX_M3:
                b.resizeAlgo(ResizeAlgo.BICUBIC).padStyle(PadStyle.NONE);
                break;
            case GEMMA4UV:
                // the unified variant merges the pooling kernel in the patch convolution
                b.resizeAlgo(ResizeAlgo.BICUBIC).padStyle(PadStyle.NONE);
                b.patchSize(b.patchSize * b.mergeSize).mergeSize(1);
                break;
            case YOUTUVL:
                b.resizeAlgo(ResizeAlgo.BILINEAR).padStyle(PadStyle.NONE);
                break;
            case LFM3VL:
                applyLfm3vl(b, kv);
                break;
            case MUSE_GLIMMER:
                b.resizeAlgo(ResizeAlgo.LANCZOS);
                break;
            case GLM4V:
            case GLM5V:
            case PADDLEOCR:
            case KIMIVL:
            case DOTS3NOTE_V:
            case SENSENOVA_U1:
                b.resizeAlgo(ResizeAlgo.BICUBIC);
                break;
            case NEMOTRON_V2_VL:
                break;
            default:
                throw new IllegalStateException("unhandled preprocessing family " + b.preprocessing.family()
                        + " of projector type '" + b.projectorName + "'");
        }
    }

    /**
     * The llava family: the CLIP-average pad color (CogVLM too), GLM-Edge's black or Janus-Pro's gray; with
     * pinpoints, the llava-1.6 slice split.
     */
    private static void applyLlava(ImageProcessorParams.Builder b, GgufKv kv) {
        switch (b.preprocessing.family()) {
            case GLM_EDGE:
                b.padStyle(PadStyle.CEIL).resizeAlgo(ResizeAlgo.BICUBIC);
                break;
            case JANUS_PRO:
                b.padColor(new int[] {127, 127, 127}).resizeAlgo(ResizeAlgo.BICUBIC);
                break;
            default:
                b.padColor(new int[] {122, 116, 104});
                if (!b.gridPinpoints.isEmpty()) {
                    b.padStyle(PadStyle.CEIL).resizeAlgo(ResizeAlgo.BICUBIC)
                            .overviewPadStyle(PadStyle.NONE).refinedPadStyle(PadStyle.CEIL)
                            .refinedPadColor(new int[] {122, 116, 104});
                }
                break;
        }
        readTiles(kv, b);
    }

    /**
     * The MiniCPM-V 4.6/4.7 merger and MiniCPM-OCR 1.0: no padding anywhere, Python rounding in the slice planner,
     * a 2x or 4x merge.
     */
    private static void applyMinicpmvMerger(ImageProcessorParams.Builder b, GgufKv kv) {
        if (b.mergeSize != 2 && b.mergeSize != 4) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_PROJECTOR_SCALE_FACTOR
                    + "' of the '" + b.projectorName + "' projector must be 2 or 4, got " + b.mergeSize);
        }
        b.overviewPadStyle(PadStyle.NONE).refinedPadStyle(PadStyle.NONE).roundHalfEven(true);
        readTiles(kv, b);
    }

    /**
     * LLaMA 4: bilinear for the overview and the refined image as well (llama.cpp sets only
     * {@code image_resize_algo}, which its llava-uhd slicing does not read), every tile grid up to 3x3 but the
     * single tile as candidates, overview last.
     */
    private static void applyLlama4(ImageProcessorParams.Builder b, GgufKv kv) {
        b.resizeAlgo(ResizeAlgo.BILINEAR).overviewResizeAlgo(ResizeAlgo.BILINEAR)
                .refinedResizeAlgo(ResizeAlgo.BILINEAR);
        readTiles(kv, b);
        List<ImageSize> candidates = new ArrayList<>(b.gridPinpoints);
        for (int x = 1; x <= LLAMA4_MAX_TILES_PER_SIDE; x++) {
            for (int y = 1; y <= LLAMA4_MAX_TILES_PER_SIDE; y++) {
                if (x != 1 || y != 1) {
                    candidates.add(new ImageSize(x * b.imageSize, y * b.imageSize));
                }
            }
        }
        b.gridPinpoints(candidates);
        b.overviewFirst(false);
    }

    /**
     * LFM2-VL and d1-omni: the GGUF resize algorithm (torchvision bilinear by default), stretched tiles, the tiling
     * keys with the defaults of the LFM2.5-VL processor config, the thumbnail last.
     */
    private static void applyLfm2(ImageProcessorParams.Builder b, GgufKv kv) {
        String name = kv.str(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO, "bilinear");
        ResizeAlgo algo = namedAlgo(b, name);
        b.resizeAlgo(algo).overviewResizeAlgo(algo).refinedResizeAlgo(algo).refinedPadStyle(PadStyle.NONE);
        b.minTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 2));
        b.maxTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, 10));
        b.tileSize(kv.u32(GgufKeys.CLIP_VISION_PREPROC_TILE_SIZE, 512));
        b.maxPixelsTolerance(kv.f32(GgufKeys.CLIP_VISION_PREPROC_MAX_PIXELS_TOLERANCE, 2.0f));
        if (b.minTiles < 1 || b.minTiles > b.maxTiles) {
            throw new IllegalArgumentException("metadata keys '" + GgufKeys.CLIP_VISION_PREPROC_MIN_TILES + "' ("
                    + b.minTiles + ") and '" + GgufKeys.CLIP_VISION_PREPROC_MAX_TILES + "' (" + b.maxTiles
                    + ") must satisfy 1 <= min <= max");
        }
        if (b.tileSize <= 0) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_PREPROC_TILE_SIZE
                    + "' must be positive, got " + b.tileSize);
        }
        b.overviewFirst(false);
    }

    /** The LFM2 resize algorithm names: bilinear and bicubic resize as torchvision does, Lanczos as Pillow. */
    private static ResizeAlgo namedAlgo(ImageProcessorParams.Builder b, String name) {
        if ("bilinear".equals(name) || "bicubic".equals(name)) {
            return ResizeAlgo.fromName(name).torch();
        }
        if ("lanczos".equals(name) && b.preprocessing.family() == VisionPreprocessing.Family.LFM2) {
            return ResizeAlgo.LANCZOS;
        }
        throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO + "' of the '"
                + b.projectorName + "' projector names an unsupported resize algo: '" + name + "'");
    }

    /**
     * LFM3-VL: the required torchvision resize algorithm, no padding, a 2x2 merge, and the required pixel limits
     * that become the token limits (the max capped at 4096 tokens).
     */
    private static void applyLfm3vl(ImageProcessorParams.Builder b, GgufKv kv) {
        if (!kv.has(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO)) {
            throw new IllegalArgumentException("the '" + b.projectorName + "' projector requires the metadata key '"
                    + GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO + "'");
        }
        b.resizeAlgo(namedAlgo(b, kv.str(GgufKeys.CLIP_VISION_IMAGE_RESIZE_ALGO, ""))).padStyle(PadStyle.NONE);
        if (b.mergeSize != 2) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_SPATIAL_MERGE_SIZE
                    + "' of the '" + b.projectorName + "' projector must be 2, got " + b.mergeSize);
        }
    }

    /**
     * DeepSeek-OCR v1 and v2: a 1024 global view of 16-pixel patches padded with gray, local tiles of 640 (v1)
     * or 768 (v2) pixels, the tile range from the GGUF over the per-version defaults, the global view last.
     */
    private static void applyDeepseekOcr(ImageProcessorParams.Builder b, GgufKv kv) {
        boolean v1 = b.preprocessing.family() == VisionPreprocessing.Family.DEEPSEEK_OCR;
        b.patchSize(16).imageSize(1024).resizeAlgo(ResizeAlgo.BICUBIC).padColor(new int[] {127, 127, 127});
        b.tileSize(v1 ? 640 : 768);
        b.minTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 2));
        b.maxTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, v1 ? 9 : 6));
        if (b.minTiles > b.maxTiles || b.maxTiles > 256) {
            throw new IllegalArgumentException("metadata keys '" + GgufKeys.CLIP_VISION_PREPROC_MIN_TILES + "' ("
                    + b.minTiles + ") and '" + GgufKeys.CLIP_VISION_PREPROC_MAX_TILES + "' (" + b.maxTiles
                    + ") must satisfy 0 <= min <= max <= 256");
        }
        b.overviewFirst(false);
    }

    /**
     * DeepSeek-V4 and V4.1: bicubic on gray, the required {@code image_min_pixels} (the budget's min tokens
     * replace it), the token block cap as the max pixels over the merged patch area (the budget's max tokens,
     * at least 16, replace the family cap), the min pixels never above the max.
     */
    private static void applyDeepseekV4(ImageProcessorParams.Builder b, GgufKv kv, ImageTokenBudget budget) {
        if (!kv.has(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS)) {
            throw new IllegalArgumentException("the '" + b.projectorName + "' projector requires the metadata key '"
                    + GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS + "'");
        }
        b.resizeAlgo(ResizeAlgo.BICUBIC).padColor(new int[] {127, 127, 127});
        long area = patchArea(b);
        long minPixels = budget.minTokens() > 0 ? budget.minPixels(area)
                : kv.u32Long(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, 0);
        int cap = b.preprocessing.family() == VisionPreprocessing.Family.DEEPSEEK4V ? DEEPSEEK4V_MAX_TOKENS
                : DEEPSEEK41V_MAX_TOKENS;
        int maxTokens = budget.maxTokens() > 0 ? Math.max(budget.maxTokens(), DEEPSEEK_V4_MIN_MAX_TOKENS) : cap;
        long maxPixels = maxTokens * area;
        b.maxPixels(maxPixels).minPixels(Math.min(minPixels, maxPixels));
    }

    /**
     * Granite Vision 4.1: the required grid pinpoints as the anyres candidates, a stretched bicubic overview, the
     * bicubic refined image padded with black, and the QFormer window and query sides the token count reads.
     */
    private static void applyGranite4v(ImageProcessorParams.Builder b, GgufKv kv) {
        if (b.gridPinpoints.isEmpty()) {
            throw new IllegalArgumentException("the '" + b.projectorName + "' projector requires the metadata key '"
                    + GgufKeys.CLIP_VISION_IMAGE_GRID_PINPOINTS + "' (its anyres resolution candidates)");
        }
        b.resizeAlgo(ResizeAlgo.BICUBIC).padStyle(PadStyle.CEIL)
                .overviewResizeAlgo(ResizeAlgo.BICUBIC).overviewPadStyle(PadStyle.NONE)
                .refinedResizeAlgo(ResizeAlgo.BICUBIC).refinedPadStyle(PadStyle.CEIL)
                .refinedPadColor(new int[] {0, 0, 0});
        b.windows(positiveKey(b, kv, GgufKeys.CLIP_VISION_PROJECTOR_WINDOW_SIDE),
                positiveKey(b, kv, GgufKeys.CLIP_VISION_PROJECTOR_QUERY_SIDE));
    }

    private static int positiveKey(ImageProcessorParams.Builder b, GgufKv kv, String key) {
        int v = kv.u32Required(key);
        if (v <= 0) {
            throw new IllegalArgumentException("metadata key '" + key + "' of the '" + b.projectorName
                    + "' projector must be positive, got " + v);
        }
        return v;
    }

    private static void applyCohere2v(ImageProcessorParams.Builder b, GgufKv kv) {
        b.refinedPadStyle(PadStyle.NONE);
        b.maxTiles(kv.u32Required(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES));
        if (b.maxTiles <= 0 || b.maxTiles > 256) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_PREPROC_MAX_TILES
                    + "' must be in range [1, 256], got " + b.maxTiles);
        }
        b.overviewFirst(false);
    }

    /**
     * The dynamic-resolution pixel limits of the family: the token defaults its preprocessor config carries, the
     * GGUF keys its converter always writes, or the Pixtral longest-edge cap.
     */
    private static void applyPixelLimits(ImageProcessorParams.Builder b, GgufKv kv, ImageTokenBudget budget) {
        switch (b.preprocessing.family()) {
            case QWEN_MERGER:
            case GLM4V:
                limitPixels(b, kv, budget, 8, 4096);
                break;
            case HUNYUANVL:
                limitPixels(b, kv, budget, 256, 16384);
                break;
            case KIMIVL:
                limitPixels(b, kv, budget, 8, 1024);
                break;
            case LFM2:
                tokenLimits(b, 64, 256);
                break;
            case GEMMA4V:
            case GEMMA4UV:
                tokenLimits(b, 70, 1120);
                break;
            case YOUTUVL:
                tokenLimits(b, 1, 62500);
                break;
            case MINIMAX_M3:
                tokenLimits(b, 8, 576);
                break;
            case MUSE_GLIMMER:
                tokenLimits(b, 1, 4096);
                break;
            case SENSENOVA_U1:
                // the converter writes the pixel keys and the loader requires them, but the token limits replace
                // their values
                requirePixelLimits(b, kv);
                tokenLimits(b, 16, 4096);
                break;
            case LFM3VL:
                lfm3vlLimits(b, kv);
                break;
            case MIMOVL:
            case EXAONE4_5:
            case PADDLEOCR:
            case DOTS_OCR:
            case NEMOTRON_OMNI:
            case PHI4:
            case GLM5V:
            case DOTS3NOTE_V:
                requirePixelLimits(b, kv);
                break;
            case PIXTRAL:
                if (b.longestEdge > 0) {
                    b.padStyle(PadStyle.NONE).floorBeforeAlign(true);
                    long perSide = (b.longestEdge + (long) b.patchSize * b.mergeSize - 1)
                            / ((long) b.patchSize * b.mergeSize);
                    b.minPixels(0).maxPixels(perSide * perSide * patchArea(b));
                } else {
                    limitPixels(b, kv, budget, 8, 1024);
                }
                break;
            default:
                break;
        }
    }

    /**
     * The family's token limits as pixel limits over the merged patch area, the GGUF pixel keys unread
     * ({@code set_limit_image_tokens(min, max)}); the budget replaces them afterwards.
     */
    private static void tokenLimits(ImageProcessorParams.Builder b, int minTokens, int maxTokens) {
        long area = patchArea(b);
        b.minPixels(minTokens * area).maxPixels(maxTokens * area);
    }

    /** LFM3-VL: the required pixel limits as whole merged tokens, the max capped at 4096 tokens. */
    private static void lfm3vlLimits(ImageProcessorParams.Builder b, GgufKv kv) {
        requirePixelLimits(b, kv);
        if (b.minPixels <= 0 || b.maxPixels < b.minPixels) {
            throw new IllegalArgumentException("metadata keys '" + GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS + "' ("
                    + b.minPixels + ") and '" + GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS + "' (" + b.maxPixels
                    + ") of the '" + b.projectorName + "' projector must satisfy 0 < min <= max");
        }
        long area = patchArea(b);
        tokenLimits(b, (int) (b.minPixels / area), (int) Math.min(LFM3VL_MAX_TOKENS, b.maxPixels / area));
    }

    /** llava-uhd tile bounds from the GGUF (minicpmv slice cap and internvl tile range). */
    private static void readTiles(GgufKv kv, ImageProcessorParams.Builder b) {
        b.maxSliceNums(kv.u32(GgufKeys.CLIP_VISION_MAX_SLICE_NUMS, 9));
        b.minTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MIN_TILES, 0));
        b.maxTiles(kv.u32(GgufKeys.CLIP_VISION_PREPROC_MAX_TILES, 0));
    }

    /**
     * The pixel limits: the GGUF values when they carry them, the family defaults derived from the token
     * defaults through the merged patch area otherwise; a set budget limit replaces its pixel limit per side
     * (llama.cpp {@code custom_image_min_tokens} wins over the GGUF key).
     */
    private static void limitPixels(ImageProcessorParams.Builder b, GgufKv kv, ImageTokenBudget budget,
                                    int defaultMinTokens, int defaultMaxTokens) {
        long area = patchArea(b);
        if (budget.minTokens() > 0) {
            b.minPixels(budget.minPixels(area));
        } else {
            b.minPixels(kv.u32Long(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, (long) defaultMinTokens * area));
        }
        if (budget.maxTokens() > 0) {
            b.maxPixels(budget.maxPixels(area));
        } else {
            b.maxPixels(kv.u32Long(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, (long) defaultMaxTokens * area));
        }
    }

    /** Families whose converters always write the pixel limits; absent keys are refused by name. */
    private static void requirePixelLimits(ImageProcessorParams.Builder b, GgufKv kv) {
        if (!kv.has(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS) || !kv.has(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS)) {
            throw new IllegalArgumentException("the '" + b.projectorName
                    + "' projector requires the metadata keys '" + GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS + "' and '"
                    + GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS + "'");
        }
        b.minPixels(kv.u32Long(GgufKeys.CLIP_VISION_IMAGE_MIN_PIXELS, 0));
        b.maxPixels(kv.u32Long(GgufKeys.CLIP_VISION_IMAGE_MAX_PIXELS, 0));
    }

    /** The budget as pixel limits over the merged patch area; unset entries keep what the family set. */
    private static void applyBudget(ImageProcessorParams.Builder b, ImageTokenBudget budget) {
        if (budget.minTokens() > 0) {
            b.minPixels(budget.minPixels(patchArea(b)));
        }
        if (budget.maxTokens() > 0) {
            b.maxPixels(budget.maxPixels(patchArea(b)));
        }
    }

    private static long patchArea(ImageProcessorParams.Builder b) {
        long area = (long) b.patchSize * b.mergeSize;
        return area * area;
    }

    /** InternVL dynamic high resolution: every {@code a x b} tile grid in the tile range (clip.cpp). */
    private static List<ImageSize> dhrCandidates(int imageSize, int minTiles, int maxTiles) {
        if (minTiles < 1) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_PREPROC_MIN_TILES
                    + "' must be at least 1, got " + minTiles);
        }
        List<ImageSize> out = new ArrayList<>();
        for (int a = minTiles; a <= maxTiles; a++) {
            int bLo = Math.max((minTiles + a - 1) / a, minTiles);
            int bHi = Math.min(maxTiles / a, maxTiles);
            for (int b = bLo; b <= bHi; b++) {
                out.add(new ImageSize(a * imageSize, b * imageSize));
            }
        }
        return out;
    }
}
