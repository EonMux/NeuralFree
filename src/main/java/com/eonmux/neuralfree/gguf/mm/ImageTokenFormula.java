package com.eonmux.neuralfree.gguf.mm;

import com.eonmux.neuralfree.gguf.hparams.GgufFormatException;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.hparams.GgufKv;

/**
 * How one prepared image tile of a vision projector turns into image tokens (llama.cpp
 * {@code clip_n_output_tokens}, tools/mtmd/clip.cpp): the patch grid of the tile, divided by the spatial merge
 * where the projector merges patches, with the extra rows some projectors emit (break, newline, begin/end rows),
 * resampled to a fixed query count, or pooled. Every formula reads the tile size in pixels and the
 * {@link ImageTokenGeometry} of the projector, and whether the tile is the overview of a sliced image (the
 * {@code add_viewsep} flag of DeepSeek-OCR's global view). Below, {@code px x py} is the tile's patch grid
 * ({@code width / patch}, {@code height / patch}) and {@code m} the per-side merge.
 */
public enum ImageTokenFormula {

    /** {@code px * py}; the sides must be patch multiples. */
    PATCHES,
    /** CogVLM: {@code px * py} plus its BOI and EOI rows; the sides must be patch multiples. */
    PATCHES_BOI_EOI,
    /** {@code px * py / m^2}; the patch count must be a multiple of {@code m^2}. */
    PATCHES_OVER_MERGE,
    /** {@code (px / m) * (py / m)}, the merged token grid; the sides must be multiples of {@code patch * m}. */
    MERGED_CONV,
    /** The LDP projectors: {@code px * py / 4}. */
    LDP,
    /** The GLM-Edge adapter: {@code px * py / 4}, plus its BOI and EOI rows. */
    LDP_BOI_EOI,
    /** A fixed query count per tile, the MiniCPM-V resampler. */
    RESAMPLER,
    /** Always 64 tokens, the YASA2 8x8 adaptive pool. */
    POOLED_8X8,
    /**
     * Pixtral: the {@code (px / m) x (py / m)} merged tokens plus one {@code [IMG_BREAK]} row after every token
     * row but the last, {@code (px/m)*(py/m) + py/m - 1}; the sides must be multiples of {@code patch * m}.
     */
    MERGED_ROWS_WITH_BREAKS,
    /**
     * Kimi-VL, Kimi K2.5, LFM2: the merged grid with each side aligned up,
     * {@code ceil(width / (patch*m)) * ceil(height / (patch*m))}.
     */
    ALIGNED_MERGED_GRID,
    /** PaddleOCR-VL and dots.ocr: the patch count aligned up to {@code m^2}, {@code ceil(px * py / m^2)}. */
    ALIGNED_PATCHES_OVER_MERGE,
    /** Gemma 3n (MobileNetV5): {@code image_size / patch_size} tokens for any tile. */
    IMAGE_SIZE_OVER_PATCH,
    /**
     * HunyuanVL: the {@code ow x oh} merged grid ({@code ow = px / m}, {@code oh = py / m}, floored) with one
     * newline row per token row, plus the begin and end rows, {@code (ow + 1) * oh + 2}.
     */
    MERGED_CONV_WITH_NEWLINES,
    /**
     * DeepSeek-OCR: the SAM convolutions reduce the patches by 16 ({@code n = px * py / 16}). The global view
     * ({@code h = floor(sqrt(n))}) adds a newline row per token row and the view separator, {@code h * (h + 1) + 1};
     * a fused tile row ({@code height} a multiple of {@code width}, {@code g = height / width} tiles of
     * {@code t = width / (patch * 4)} tokens per side) adds a newline per token row, {@code (t * g + 1) * t}; any
     * other tile is {@code n}.
     */
    DEEPSEEK_OCR,
    /** DeepSeek-OCR 2: {@code px * py / 16} query tokens, plus the view separator after the global view. */
    DEEPSEEK_OCR2,
    /**
     * DeepSeek-V4: the token block over the {@code w x h} aligner grid ({@code w = ceil(px / m)},
     * {@code h = ceil(py / m)}) without lead pads: start, the rows rounded up to an even count each with a
     * newline, the pads keeping the block even, end. The prompt adds up to 3 lead pads that align the image to
     * 4 tokens.
     */
    DEEPSEEK4V_BLOCK,
    /** DeepSeek-V4.1: the reading-order block over the aligner grid, {@code h * (w + 1) + 2}. */
    DEEPSEEK41V_BLOCK,
    /**
     * Granite Vision 4.1: each {@code window x window} patch window becomes {@code query x query} tokens,
     * {@code ((px / window) * query) * ((py / window) * query)}; the patch grid must be a whole number of windows.
     * An anyres image counts its overview and unpadded tile grid with {@link AnyresGrid#tokens}.
     */
    GRANITE4_WINDOWS;

    /**
     * The image token count of one {@code width x height} tile that is not the overview of a sliced image.
     *
     * @throws IllegalArgumentException when the tile is empty, its sides are not multiples of the geometry the
     *                                  formula divides by, a resampler has no query count, or the count does not
     *                                  fit an int
     */
    public int tokens(int width, int height, ImageTokenGeometry geometry) {
        return tokens(width, height, geometry, false);
    }

    /**
     * The image token count of one {@code width x height} tile; {@code overview} marks the overview (global view)
     * of a sliced image.
     *
     * @throws IllegalArgumentException when the tile is empty, its sides are not multiples of the geometry the
     *                                  formula divides by, a resampler has no query count, or the count does not
     *                                  fit an int
     */
    public int tokens(int width, int height, ImageTokenGeometry geometry, boolean overview) {
        requireTile(width, height);
        int patch = geometry.patchSize();
        int merge = geometry.mergeSize();
        long patches = (long) (width / patch) * (height / patch);
        switch (this) {
            case PATCHES:
                return toTokens((long) divide(width, patch, "width") * divide(height, patch, "height"));
            case PATCHES_BOI_EOI:
                return toTokens((long) divide(width, patch, "width") * divide(height, patch, "height") + 2);
            case PATCHES_OVER_MERGE:
                return patchesOverMerge(width, height, geometry);
            case MERGED_CONV:
                return toTokens((long) divide(width, geometry.mergedPatchSize(), "width")
                        * divide(height, geometry.mergedPatchSize(), "height"));
            case LDP:
                return toTokens(patches / 4);
            case LDP_BOI_EOI:
                return toTokens(patches / 4 + 2);
            case RESAMPLER:
                if (geometry.queryNum() <= 0) {
                    throw new IllegalArgumentException("the resampler query count is unset; the resampler token"
                            + " count is unknown");
                }
                return geometry.queryNum();
            case POOLED_8X8:
                return 64;
            case MERGED_ROWS_WITH_BREAKS: {
                long px = divide(width, geometry.mergedPatchSize(), "width");
                long py = divide(height, geometry.mergedPatchSize(), "height");
                return toTokens(px * py + py - 1);
            }
            case ALIGNED_MERGED_GRID:
                return toTokens((long) ceilDiv(width, geometry.mergedPatchSize())
                        * ceilDiv(height, geometry.mergedPatchSize()));
            case ALIGNED_PATCHES_OVER_MERGE: {
                long stride = (long) merge * merge;
                return toTokens((patches + stride - 1) / stride);
            }
            case IMAGE_SIZE_OVER_PATCH:
                if (geometry.imageSize() <= 0) {
                    throw new IllegalArgumentException("the image size over patch formula needs a positive image"
                            + " size, got " + geometry.imageSize());
                }
                return geometry.imageSize() / patch;
            case MERGED_CONV_WITH_NEWLINES: {
                long ow = (width / patch) / merge;
                long oh = (height / patch) / merge;
                return toTokens((ow + 1) * oh + 2);
            }
            case DEEPSEEK_OCR:
                return deepseekOcr(width, height, patch, patches / 16, overview);
            case DEEPSEEK_OCR2:
                return toTokens(patches / 16 + (overview ? 1 : 0));
            case DEEPSEEK4V_BLOCK:
                return DeepseekV4Layout.tokens4v(DeepseekV4Layout.gridSide(width / patch, merge),
                        DeepseekV4Layout.gridSide(height / patch, merge), 0);
            case DEEPSEEK41V_BLOCK:
                return DeepseekV4Layout.tokens41v(DeepseekV4Layout.gridSide(width / patch, merge),
                        DeepseekV4Layout.gridSide(height / patch, merge));
            case GRANITE4_WINDOWS:
                return toTokens((long) windowTokens(width, geometry, "width")
                        * windowTokens(height, geometry, "height"));
            default:
                throw new IllegalStateException("unhandled token formula: " + this);
        }
    }

    /**
     * The token grid a formula lays its tokens on unless the projector type says otherwise
     * ({@code clip_n_output_tokens_x/y}): the merged grid for the merger convolutions, one row otherwise.
     */
    public ImageTokenGrid defaultGrid() {
        return this == MERGED_CONV || this == MERGED_CONV_WITH_NEWLINES ? ImageTokenGrid.MERGED : ImageTokenGrid.ROW;
    }

    /**
     * The MiniCPM-V resampler query count of a file, the {@link #RESAMPLER} token count per tile:
     * {@code clip.minicpmv_query_num}, or the legacy per-version value when the key is absent or 0 (64 for
     * {@code clip.minicpmv_version} 3, 4, 5, 6 and 100045, 96 otherwise; clip.cpp load_hparams).
     *
     * @throws GgufFormatException naming the key for a negative query count
     */
    public static int resamplerQueryNum(GgufKv kv) {
        int n = kv.u32(GgufKeys.CLIP_MINICPMV_QUERY_NUM, 0);
        if (n < 0) {
            throw new GgufFormatException("metadata key '" + GgufKeys.CLIP_MINICPMV_QUERY_NUM
                    + "' must not be negative, got " + n);
        }
        if (n > 0) {
            return n;
        }
        switch (kv.i32(GgufKeys.CLIP_MINICPMV_VERSION, 2)) {
            case 3:
            case 4:
            case 5:
            case 6:
            case 100045:
                return 64;
            default:
                return 96;
        }
    }

    /** The DeepSeek-OCR count of the global view, a fused tile row, or any other tile. */
    private static int deepseekOcr(int width, int height, int patch, long n, boolean overview) {
        if (overview) {
            long h = (long) Math.sqrt((float) n);
            return toTokens(h * (h + 1) + 1);
        }
        if (height >= width && height % width == 0) {
            long tiles = height / width;
            long side = width / (patch * 4L);
            return toTokens((side * tiles + 1) * side);
        }
        return toTokens(n);
    }

    /**
     * The Granite Vision tokens along one side: {@code (side / patch / window) * query}.
     *
     * @throws IllegalArgumentException when the geometry has no window sides or the side is not a whole number of
     *                                  windows
     */
    public static int windowTokens(int side, ImageTokenGeometry geometry, String what) {
        if (geometry.windowSide() <= 0 || geometry.querySide() <= 0) {
            throw new IllegalArgumentException("the QFormer window and query sides are unset; the window token count"
                    + " is unknown");
        }
        int patches = divide(side, geometry.patchSize(), what);
        return divide(patches, geometry.windowSide(), what + " in patches") * geometry.querySide();
    }

    private static int patchesOverMerge(int width, int height, ImageTokenGeometry geometry) {
        long patches = (long) (width / geometry.patchSize()) * (height / geometry.patchSize());
        long merged = (long) geometry.mergeSize() * geometry.mergeSize();
        if (patches % merged != 0) {
            throw new IllegalArgumentException("tile " + width + "x" + height + " yields " + patches
                    + " patches, not a multiple of merge^2 (" + merged + ")");
        }
        return toTokens(patches / merged);
    }

    private static void requireTile(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("tile " + width + "x" + height + " is empty");
        }
    }

    private static int divide(int side, int unit, String what) {
        if (side % unit != 0) {
            throw new IllegalArgumentException(what + " " + side + " is not a multiple of " + unit);
        }
        return side / unit;
    }

    private static int ceilDiv(int side, int unit) {
        return (side + unit - 1) / unit;
    }

    private static int toTokens(long tokens) {
        if (tokens > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("the tile needs " + tokens + " tokens, more than "
                    + Integer.MAX_VALUE);
        }
        return (int) tokens;
    }
}
