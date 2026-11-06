package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.hparams.GgufKv;
import com.eonmux.neuralfree.gguf.hparams.GgufKeys;
import com.eonmux.neuralfree.gguf.mm.AnyresGrid;
import com.eonmux.neuralfree.gguf.mm.ImageTokenFormula;
import com.eonmux.neuralfree.gguf.mm.ImageTokenGeometry;
import com.eonmux.neuralfree.gguf.mm.ImageTokenGrid;
import com.eonmux.neuralfree.gguf.mm.ProjectorModality;
import com.eonmux.neuralfree.gguf.mm.ProjectorRegistry;
import com.eonmux.neuralfree.gguf.mm.ProjectorType;
import com.eonmux.neuralfree.gguf.mm.UnsupportedProjectorException;
import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * The image preprocessing parameters of one vision projector, the image-relevant part of llama.cpp's
 * {@code clip_hparams} (tools/mtmd/clip-model.h): the projector type name and its {@link VisionPreprocessing}
 * (preprocessing family and image token formula), the patch and merge geometry, the pixel limits (set directly
 * by the GGUF or derived from {@link ImageTokenBudget}), the resolution candidates, the slice cap, the tile range,
 * tile size and pixel tolerance of the tiling families, the Granite Vision QFormer window and query sides, the
 * resize algorithm, pad style and pad color, the normalize mean/std, and the llava-uhd overview/refined resize
 * split.
 *
 * <p>Immutable. {@link #fromGguf(GgufKv, ImageTokenBudget)} loads the values the way llama.cpp's
 * {@code clip_model_loader::load_hparams} does: the merge through the type's patch-merge rule, family defaults,
 * GGUF keys over them, the token budget last (a set token limit replaces the pixel limits,
 * {@code custom_image_min_tokens} wins over the GGUF values).</p>
 */
public final class ImageProcessorParams {

    private final String projectorName;
    private final VisionPreprocessing preprocessing;
    private final int patchSize;
    private final int mergeSize;
    private final int imageSize;
    private final int longestEdge;
    private final long minPixels;
    private final long maxPixels;
    private final List<ImageSize> gridPinpoints;
    private final int maxSliceNums;
    private final int minTiles;
    private final int maxTiles;
    private final int tileSize;
    private final float maxPixelsTolerance;
    private final ResizeAlgo resizeAlgo;
    private final PadStyle padStyle;
    private final int[] padColor;
    private final boolean roundHalfEven;
    private final boolean floorBeforeAlign;
    private final boolean overviewFirst;
    private final int minicpmvQueryNum;
    private final int windowSide;
    private final int querySide;
    private final ResizeAlgo overviewResizeAlgo;
    private final ResizeAlgo refinedResizeAlgo;
    private final PadStyle overviewPadStyle;
    private final PadStyle refinedPadStyle;
    private final int[] refinedPadColor;
    private final float[] mean;
    private final float[] std;

    private ImageProcessorParams(Builder b) {
        this.projectorName = b.projectorName;
        this.preprocessing = b.preprocessing;
        this.patchSize = b.patchSize;
        this.mergeSize = b.mergeSize;
        this.imageSize = b.imageSize;
        this.longestEdge = b.longestEdge;
        this.minPixels = b.minPixels;
        this.maxPixels = b.maxPixels;
        this.gridPinpoints = Collections.unmodifiableList(new ArrayList<>(b.gridPinpoints));
        this.maxSliceNums = b.maxSliceNums;
        this.minTiles = b.minTiles;
        this.maxTiles = b.maxTiles;
        this.tileSize = b.tileSize;
        this.maxPixelsTolerance = b.maxPixelsTolerance;
        this.resizeAlgo = b.resizeAlgo;
        this.padStyle = b.padStyle;
        this.padColor = b.padColor.clone();
        this.roundHalfEven = b.roundHalfEven;
        this.floorBeforeAlign = b.floorBeforeAlign;
        this.overviewFirst = b.overviewFirst;
        this.minicpmvQueryNum = b.minicpmvQueryNum;
        this.windowSide = b.windowSide;
        this.querySide = b.querySide;
        this.overviewResizeAlgo = b.overviewResizeAlgo;
        this.refinedResizeAlgo = b.refinedResizeAlgo;
        this.overviewPadStyle = b.overviewPadStyle;
        this.refinedPadStyle = b.refinedPadStyle;
        this.refinedPadColor = b.refinedPadColor.clone();
        this.mean = b.mean.clone();
        this.std = b.std.clone();
    }

    /**
     * Loads the parameters from projector GGUF metadata with the given token budget; see the class comment. The
     * vision projector type ({@link ProjectorType#declaredName}) resolves to its image preprocessing through
     * {@link ProjectorRegistry#visionPreprocessing}: a built-in type's own, or a plugin row's.
     *
     * @throws UnsupportedProjectorException naming the projector type when it has no image preprocessing
     * @throws IllegalArgumentException      for a missing required key (named), an unsupported resize algorithm
     *                                       (named), a pinpoints array with an odd length, or a budget that is
     *                                       inconsistent with the family
     */
    public static ImageProcessorParams fromGguf(GgufKv kv, ImageTokenBudget budget) {
        Objects.requireNonNull(kv, "kv");
        String name = ProjectorType.declaredName(kv, ProjectorModality.VISION);
        return fromGguf(kv, name, ProjectorRegistry.visionPreprocessing(name), budget);
    }

    /**
     * Loads the parameters of the projector {@code projectorName} prepared with {@code preprocessing} (already
     * resolved, for example from a {@link com.eonmux.neuralfree.gguf.mm.ProjectorEntry}); see the class comment.
     *
     * @throws IllegalArgumentException for a missing required key (named), a pinpoints array with an odd length,
     *                                  or a budget that is inconsistent with the family
     */
    public static ImageProcessorParams fromGguf(GgufKv kv, String projectorName, VisionPreprocessing preprocessing,
                                                ImageTokenBudget budget) {
        Objects.requireNonNull(kv, "kv");
        Objects.requireNonNull(budget, "budget");
        Builder b = new Builder(projectorName, preprocessing)
                .patchSize(kv.u32Required(GgufKeys.CLIP_VISION_PATCH_SIZE))
                .imageSize(kv.u32Required(GgufKeys.CLIP_VISION_IMAGE_SIZE))
                .mergeSize(preprocessing.merge().read(kv))
                .mean(requireArray(kv, GgufKeys.CLIP_VISION_IMAGE_MEAN, "image_mean"))
                .std(requireArray(kv, GgufKeys.CLIP_VISION_IMAGE_STD, "image_std"));
        b.gridPinpoints(pinpoints(kv));
        ProjectorDefaults.apply(b, kv, budget);
        return b.build();
    }

    private static float[] requireArray(GgufKv kv, String key, String what) {
        float[] v = kv.f32Array(key);
        if (v == null || v.length < 3) {
            throw new IllegalArgumentException("metadata key '" + key + "' must hold at least 3 floats, got "
                    + (v == null ? "nothing" : v.length + " (" + what + ")"));
        }
        return v;
    }

    /** The flat even-length {@code clip.vision.image_grid_pinpoints} array as width/height pairs. */
    private static List<ImageSize> pinpoints(GgufKv kv) {
        int[] flat = kv.i32Array(GgufKeys.CLIP_VISION_IMAGE_GRID_PINPOINTS);
        List<ImageSize> out = new ArrayList<>();
        if (flat == null) {
            return out;
        }
        if (flat.length % 2 != 0) {
            throw new IllegalArgumentException("metadata key '" + GgufKeys.CLIP_VISION_IMAGE_GRID_PINPOINTS
                    + "' must have an even number of elements, got " + flat.length);
        }
        for (int i = 0; i < flat.length; i += 2) {
            out.add(new ImageSize(flat[i], flat[i + 1]));
        }
        return out;
    }

    /** A new builder over a copy of these values. */
    public Builder toBuilder() {
        Builder b = new Builder(projectorName, preprocessing)
                .patchSize(patchSize)
                .mergeSize(mergeSize)
                .imageSize(imageSize)
                .longestEdge(longestEdge)
                .minPixels(minPixels)
                .maxPixels(maxPixels)
                .gridPinpoints(gridPinpoints)
                .maxSliceNums(maxSliceNums)
                .minTiles(minTiles)
                .maxTiles(maxTiles)
                .tileSize(tileSize)
                .maxPixelsTolerance(maxPixelsTolerance)
                .resizeAlgo(resizeAlgo)
                .padStyle(padStyle)
                .padColor(padColor)
                .roundHalfEven(roundHalfEven)
                .floorBeforeAlign(floorBeforeAlign)
                .overviewFirst(overviewFirst)
                .minicpmvQueryNum(minicpmvQueryNum)
                .windows(windowSide, querySide)
                .overviewResizeAlgo(overviewResizeAlgo)
                .refinedResizeAlgo(refinedResizeAlgo)
                .overviewPadStyle(overviewPadStyle)
                .refinedPadStyle(refinedPadStyle)
                .refinedPadColor(refinedPadColor)
                .mean(mean)
                .std(std);
        return b;
    }

    /** The vision projector type name ({@code clip.projector_type}, or a plugin row's key). */
    public String projectorName() { return projectorName; }

    /** The preprocessing family, image token formula and patch-merge rule of the projector. */
    public VisionPreprocessing preprocessing() { return preprocessing; }

    /** The vision patch size; positive. */
    public int patchSize() { return patchSize; }

    /** The patch merges per side ({@code n_merge}); positive. */
    public int mergeSize() { return mergeSize; }

    /** The CLIP input size the encoder was trained at; positive, or 0 for SenseNova-U1, whose file sets none. */
    public int imageSize() { return imageSize; }

    /** The preprocessor longest edge; {@code 0} when the model has none. */
    public int longestEdge() { return longestEdge; }

    /** The smallest accepted pixel area; {@code 0} when the model has no limit. */
    public long minPixels() { return minPixels; }

    /** The largest accepted pixel area; {@code 0} when the model has no limit. */
    public long maxPixels() { return maxPixels; }

    /** The resolution candidates (grid pinpoints); unmodifiable, possibly empty. */
    public List<ImageSize> gridPinpoints() { return gridPinpoints; }

    /** The llava-uhd slice cap ({@code clip.vision.max_slice_nums}); positive. */
    public int maxSliceNums() { return maxSliceNums; }

    /** The smallest tile-grid size; {@code 0} when the model tiles by slice cap instead. */
    public int minTiles() { return minTiles; }

    /** The largest tile-grid size; {@code 0} when the model tiles by slice cap instead. */
    public int maxTiles() { return maxTiles; }

    /** The side of one local tile ({@code preproc_tile_size}: LFM2, DeepSeek-OCR); {@code 0} when unused. */
    public int tileSize() { return tileSize; }

    /**
     * How far over {@link #maxPixels()} the smart-resized image may reach before LFM2 tiles it
     * ({@code preproc_max_pixels_tolerance}); {@code 1} by default.
     */
    public float maxPixelsTolerance() { return maxPixelsTolerance; }

    public ResizeAlgo resizeAlgo() { return resizeAlgo; }

    public PadStyle padStyle() { return padStyle; }

    /** The pad color as R, G, B. */
    public int[] padColor() { return padColor.clone(); }

    /** Whether the smart resize rounds half to even (Python {@code round}). */
    public boolean roundHalfEven() { return roundHalfEven; }

    /** Whether the longest edge floors the sides before the align rounding (Pixtral). */
    public boolean floorBeforeAlign() { return floorBeforeAlign; }

    /** Whether the llava-uhd overview encodes before the slices; false for InternVL, LLaMA 4 and Cohere2. */
    public boolean overviewFirst() { return overviewFirst; }

    /** The MiniCPM-V resampler query count per tile; {@code 0} when not a resampler model. */
    public int minicpmvQueryNum() { return minicpmvQueryNum; }

    /** The Granite Vision QFormer window side in patches; {@code 0} for every other family. */
    public int windowSide() { return windowSide; }

    /** The Granite Vision QFormer query side in tokens per window side; {@code 0} for every other family. */
    public int querySide() { return querySide; }

    public ResizeAlgo overviewResizeAlgo() { return overviewResizeAlgo; }

    public ResizeAlgo refinedResizeAlgo() { return refinedResizeAlgo; }

    public PadStyle overviewPadStyle() { return overviewPadStyle; }

    public PadStyle refinedPadStyle() { return refinedPadStyle; }

    /** The refined-image pad color as R, G, B. */
    public int[] refinedPadColor() { return refinedPadColor.clone(); }

    /** The normalize mean as R, G, B. */
    public float[] mean() { return mean.clone(); }

    /** The normalize std as R, G, B. */
    public float[] std() { return std.clone(); }

    /** {@code patchSize * mergeSize} squared: the pixel area of one output token. */
    public long mergedPatchArea() {
        long area = (long) patchSize * mergeSize;
        return area * area;
    }

    /** The geometry the projector's token formula counts with. */
    public ImageTokenGeometry tokenGeometry() {
        ImageTokenGeometry geometry = ImageTokenGeometry.of(patchSize, mergeSize, imageSize, minicpmvQueryNum);
        return windowSide > 0 ? geometry.withWindows(windowSide, querySide) : geometry;
    }

    /**
     * The image token count of one prepared tile that is not the overview of a sliced image
     * ({@code clip_n_output_tokens}).
     *
     * @throws IllegalArgumentException when the tile sides are not multiples of the geometry the projector
     *                                  family divides by
     */
    public int tokensFor(ImageTile tile) {
        return tokensFor(tile.size());
    }

    /** {@link #tokensFor(ImageTile)} over a size. */
    public int tokensFor(ImageSize size) {
        return tokensFor(size, false);
    }

    /**
     * The image token count of one prepared tile of {@code size}; {@code overview} marks the overview of a sliced
     * image, which DeepSeek-OCR follows with its view separator.
     *
     * @throws IllegalArgumentException when the tile sides are not multiples of the geometry the projector
     *                                  family divides by
     */
    public int tokensFor(ImageSize size, boolean overview) {
        return preprocessing.tokenFormula().tokens(size.width(), size.height(), tokenGeometry(), overview);
    }

    /**
     * The token count of an anyres image ({@code clip_n_output_tokens} of a tiled Granite Vision entry): the
     * overview tile's tokens, then the {@code gridX x gridY} tile grid cropped to the original aspect ratio with a
     * newline row per token row ({@link AnyresGrid#tokens}).
     *
     * @throws IllegalArgumentException when the projector counts no QFormer windows, the tile is not a whole number
     *                                  of windows, or a size is not positive
     */
    public int anyresTokens(ImageSize tile, int gridX, int gridY, ImageSize original) {
        int side = ImageTokenFormula.windowTokens(tile.width(), tokenGeometry(), "width");
        return AnyresGrid.tokens(side, gridX, gridY, original.width(), original.height());
    }

    /**
     * The token grid of one prepared tile ({@code clip_n_output_tokens_x/y}): the per-side merged token count
     * for the projectors that lay their tokens on the merged grid, the total token count and one row otherwise.
     */
    public ImageSize tokenGridFor(ImageSize size) {
        ImageTokenGeometry geometry = tokenGeometry();
        ImageTokenGrid grid = preprocessing.tokenGrid();
        return new ImageSize(grid.columns(size.width(), size.height(), geometry, tokensFor(size)),
                grid.rows(size.width(), size.height(), geometry));
    }

    @Override
    public String toString() {
        return "ImageProcessorParams(" + projectorName + ", patch=" + patchSize + ", merge=" + mergeSize
                + ", imageSize=" + imageSize + ", longestEdge=" + longestEdge + ", pixels=[" + minPixels + ", "
                + maxPixels + "], pinpoints=" + gridPinpoints.size() + ", algo=" + resizeAlgo + ", pad=" + padStyle
                + ")";
    }

    /** Builder of an immutable {@link ImageProcessorParams}. */
    public static final class Builder {

        // package-private: ProjectorDefaults reads the loaded values while applying family defaults
        final String projectorName;
        final VisionPreprocessing preprocessing;
        int patchSize = 0;
        int mergeSize = 1;
        int imageSize = 0;
        int longestEdge = 0;
        long minPixels = 0;
        long maxPixels = 0;
        List<ImageSize> gridPinpoints = new ArrayList<>();
        int minTiles = 0;
        int maxTiles = 0;
        int tileSize = 0;
        float maxPixelsTolerance = 1.0f;
        int minicpmvQueryNum = 0;
        int windowSide = 0;
        int querySide = 0;
        private int maxSliceNums = 9;
        private ResizeAlgo resizeAlgo = ResizeAlgo.BICUBIC;
        private PadStyle padStyle = PadStyle.CEIL;
        private int[] padColor = {0, 0, 0};
        private boolean roundHalfEven = false;
        private boolean floorBeforeAlign = false;
        private boolean overviewFirst = true;
        private ResizeAlgo overviewResizeAlgo = ResizeAlgo.BICUBIC;
        private ResizeAlgo refinedResizeAlgo = ResizeAlgo.BICUBIC;
        private PadStyle overviewPadStyle = PadStyle.NONE;
        private PadStyle refinedPadStyle = PadStyle.CEIL;
        private int[] refinedPadColor = {0, 0, 0};
        private float[] mean = {0.0f, 0.0f, 0.0f};
        private float[] std = {1.0f, 1.0f, 1.0f};

        Builder(String projectorName, VisionPreprocessing preprocessing) {
            Objects.requireNonNull(projectorName, "projectorName");
            if (projectorName.isBlank()) {
                throw new IllegalArgumentException("the projector type name is blank");
            }
            this.projectorName = projectorName;
            this.preprocessing = Objects.requireNonNull(preprocessing, "preprocessing");
        }

        /** Sets the patch size; positive. */
        public Builder patchSize(int patchSize) {
            requirePositive("patchSize", patchSize);
            this.patchSize = patchSize;
            return this;
        }

        /** Sets the per-side patch merge; positive. */
        public Builder mergeSize(int mergeSize) {
            requirePositive("mergeSize", mergeSize);
            this.mergeSize = mergeSize;
            return this;
        }

        /** Sets the CLIP image size; 0 when the file sets none, which only SenseNova-U1 accepts. */
        public Builder imageSize(int imageSize) {
            if (imageSize < 0) throw new IllegalArgumentException("imageSize must not be negative, got " + imageSize);
            this.imageSize = imageSize;
            return this;
        }

        /** Sets the preprocessor longest edge; zero unsets, negative is refused. */
        public Builder longestEdge(int longestEdge) {
            if (longestEdge < 0) throw new IllegalArgumentException("longestEdge must be non-negative");
            this.longestEdge = longestEdge;
            return this;
        }

        /** Sets the smallest accepted pixel area; zero disables, negative is refused. */
        public Builder minPixels(long minPixels) {
            if (minPixels < 0) throw new IllegalArgumentException("minPixels must be non-negative");
            this.minPixels = minPixels;
            return this;
        }

        /** Sets the largest accepted pixel area; zero disables, negative is refused. */
        public Builder maxPixels(long maxPixels) {
            if (maxPixels < 0) throw new IllegalArgumentException("maxPixels must be non-negative");
            this.maxPixels = maxPixels;
            return this;
        }

        /** Replaces the resolution candidates. */
        public Builder gridPinpoints(List<ImageSize> pinpoints) {
            this.gridPinpoints = pinpoints == null ? new ArrayList<>() : new ArrayList<>(pinpoints);
            return this;
        }

        /** Sets the llava-uhd slice cap; positive. */
        public Builder maxSliceNums(int maxSliceNums) {
            requirePositive("maxSliceNums", maxSliceNums);
            this.maxSliceNums = maxSliceNums;
            return this;
        }

        /** Sets the smallest tile-grid size; non-negative. */
        public Builder minTiles(int minTiles) {
            if (minTiles < 0) throw new IllegalArgumentException("minTiles must be non-negative");
            this.minTiles = minTiles;
            return this;
        }

        /** Sets the largest tile-grid size; non-negative. */
        public Builder maxTiles(int maxTiles) {
            if (maxTiles < 0) throw new IllegalArgumentException("maxTiles must be non-negative");
            this.maxTiles = maxTiles;
            return this;
        }

        /** Sets the local tile side; non-negative. */
        public Builder tileSize(int tileSize) {
            if (tileSize < 0) throw new IllegalArgumentException("tileSize must be non-negative");
            this.tileSize = tileSize;
            return this;
        }

        /** Sets the LFM2 tiling tolerance over the pixel limit; positive and finite. */
        public Builder maxPixelsTolerance(float maxPixelsTolerance) {
            if (!(maxPixelsTolerance > 0.0f) || Float.isInfinite(maxPixelsTolerance)) {
                throw new IllegalArgumentException("maxPixelsTolerance must be positive and finite, got "
                        + maxPixelsTolerance);
            }
            this.maxPixelsTolerance = maxPixelsTolerance;
            return this;
        }

        public Builder resizeAlgo(ResizeAlgo algo) {
            this.resizeAlgo = Objects.requireNonNull(algo, "algo");
            return this;
        }

        public Builder padStyle(PadStyle padStyle) {
            this.padStyle = Objects.requireNonNull(padStyle, "padStyle");
            return this;
        }

        /** Sets the pad color from R, G, B. */
        public Builder padColor(int[] padColor) {
            this.padColor = requireTriple(padColor, "padColor");
            return this;
        }

        public Builder roundHalfEven(boolean roundHalfEven) {
            this.roundHalfEven = roundHalfEven;
            return this;
        }

        public Builder floorBeforeAlign(boolean floorBeforeAlign) {
            this.floorBeforeAlign = floorBeforeAlign;
            return this;
        }

        public Builder overviewFirst(boolean overviewFirst) {
            this.overviewFirst = overviewFirst;
            return this;
        }

        /** Sets the resampler query count; zero unsets, negative is refused. */
        public Builder minicpmvQueryNum(int minicpmvQueryNum) {
            if (minicpmvQueryNum < 0) throw new IllegalArgumentException("minicpmvQueryNum must be non-negative");
            this.minicpmvQueryNum = minicpmvQueryNum;
            return this;
        }

        /**
         * Sets the Granite Vision QFormer window side (patches) and query side (tokens per window side); both zero
         * unset, both positive set them.
         */
        public Builder windows(int windowSide, int querySide) {
            if (windowSide < 0 || querySide < 0 || (windowSide == 0) != (querySide == 0)) {
                throw new IllegalArgumentException("the window side (" + windowSide + ") and query side ("
                        + querySide + ") must both be positive or both zero");
            }
            this.windowSide = windowSide;
            this.querySide = querySide;
            return this;
        }

        public Builder overviewResizeAlgo(ResizeAlgo algo) {
            this.overviewResizeAlgo = Objects.requireNonNull(algo, "algo");
            return this;
        }

        public Builder refinedResizeAlgo(ResizeAlgo algo) {
            this.refinedResizeAlgo = Objects.requireNonNull(algo, "algo");
            return this;
        }

        public Builder overviewPadStyle(PadStyle padStyle) {
            this.overviewPadStyle = Objects.requireNonNull(padStyle, "padStyle");
            return this;
        }

        public Builder refinedPadStyle(PadStyle padStyle) {
            this.refinedPadStyle = Objects.requireNonNull(padStyle, "padStyle");
            return this;
        }

        /** Sets the refined-image pad color from R, G, B. */
        public Builder refinedPadColor(int[] refinedPadColor) {
            this.refinedPadColor = requireTriple(refinedPadColor, "refinedPadColor");
            return this;
        }

        /** Sets the normalize mean from R, G, B. */
        public Builder mean(float[] mean) {
            this.mean = requireTripleF(mean, "mean");
            return this;
        }

        /** Sets the normalize std from R, G, B. */
        public Builder std(float[] std) {
            this.std = requireTripleF(std, "std");
            return this;
        }

        /**
         * Builds the validated parameters.
         *
         * @throws IllegalArgumentException when a required value is unset or the pixel limits are inverted
         */
        public ImageProcessorParams build() {
            requirePositive("patchSize", patchSize);
            requirePositive("mergeSize", mergeSize);
            // the SenseNova-U1 converter writes image_size 0: its dynamic resolution never reads it
            if (preprocessing.family() != VisionPreprocessing.Family.SENSENOVA_U1) {
                requirePositive("imageSize", imageSize);
            }
            if (minPixels > 0 && maxPixels > 0 && minPixels > maxPixels) {
                throw new IllegalArgumentException("minPixels (" + minPixels + ") exceeds maxPixels ("
                        + maxPixels + ")");
            }
            for (ImageSize p : gridPinpoints) {
                if (p.isEmpty()) {
                    throw new IllegalArgumentException("grid pinpoints hold an empty resolution");
                }
            }
            RasterImage.requireMeanStd(mean, std);
            return new ImageProcessorParams(this);
        }

        private static void requirePositive(String name, int value) {
            if (value <= 0) throw new IllegalArgumentException(name + " must be positive, got " + value);
        }

        private static int[] requireTriple(int[] v, String name) {
            if (v == null || v.length != 3) {
                throw new IllegalArgumentException(name + " must hold 3 values, got "
                        + (v == null ? "null" : v.length));
            }
            return v.clone();
        }

        private static float[] requireTripleF(float[] v, String name) {
            if (v == null || v.length != 3) {
                throw new IllegalArgumentException(name + " must hold 3 values, got "
                        + (v == null ? "null" : v.length));
            }
            return v.clone();
        }
    }
}
