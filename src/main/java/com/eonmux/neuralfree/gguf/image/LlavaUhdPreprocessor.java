package com.eonmux.neuralfree.gguf.image;

import java.util.ArrayList;
import java.util.List;

/**
 * The llava-uhd preprocessor ({@code mtmd_image_preprocessor_llava_uhd}, tools/mtmd/mtmd-image.cpp), the slice
 * layout of LLaVA-NeXT (arxiv.org/pdf/2403.11703): one overview tile always, and — once a side exceeds the
 * CLIP image size — a grid of refined slices. Two planners exist:
 *
 * <ul>
 *   <li>pinpoints ({@code clip.vision.image_grid_pinpoints}, LLaVA-1.6 and InternVL): the best resolution
 *   candidate for the original size is picked by the least wasted area, then stepped over in
 *   {@code imageSize} squares;</li>
 *   <li>the dynamic grid (MiniCPM-V): the slice count derives from the image area, the grid shape follows the
 *   aspect ratio and the refined size aligns every slice to the merge factor.</li>
 * </ul>
 *
 * <p>Subclasses replace the planner; the slicing, ordering and token computation stay here.</p>
 */
public class LlavaUhdPreprocessor implements ImagePreprocessor {

    private final ImageProcessorParams params;

    /** Creates the preprocessor over the params. */
    public LlavaUhdPreprocessor(ImageProcessorParams params) {
        this.params = params;
    }

    /** The params this preprocessor serves. */
    protected ImageProcessorParams params() { return params; }

    @Override
    public PreparedImages prepare(RasterImage image) {
        SliceInstructions inst = sliceInstructions(image.size());
        return withOverview(inst, sliceImage(image, inst));
    }

    /** The overview and slices of a plan as one group, on the plan's grid ({@code preprocess}). */
    protected PreparedImages withOverview(SliceInstructions inst, SliceOutput sliced) {
        return new PreparedImages.Builder(params)
                .overview(sliced.overview)
                .tiles(sliced.slices)
                .grid(inst.gridSize.width(), inst.gridSize.height())
                .build();
    }

    /** The align unit of the slice planner: one merger token per side ({@code get_slice_align}). */
    protected int sliceAlign() {
        return params.patchSize() * params.mergeSize();
    }

    /**
     * The slice plan for an original size ({@code get_slice_instructions}): the overview size, the refined
     * size (empty when nothing is sliced), the grid and the slice rectangles. Overridden by the MiniCPM-V and
     * Cohere2 variants.
     */
    protected SliceInstructions sliceInstructions(ImageSize original) {
        SliceInstructions res = new SliceInstructions();
        int sliceSize = params.imageSize();
        boolean hasSlices = original.width() > sliceSize || original.height() > sliceSize;
        boolean hasPinpoints = !params.gridPinpoints().isEmpty();

        if (!hasSlices) {
            res.overviewSize = new ImageSize(sliceSize, sliceSize);
            return res;
        }

        if (hasPinpoints) {
            ImageSize refineSize = selectBestResolution(original, params.gridPinpoints());
            res.overviewSize = new ImageSize(sliceSize, sliceSize);
            res.refinedSize = refineSize;
            for (int y = 0; y < refineSize.height(); y += sliceSize) {
                for (int x = 0; x < refineSize.width(); x += sliceSize) {
                    res.slices.add(new SliceRect(x, y,
                            Math.min(sliceSize, refineSize.width() - x),
                            Math.min(sliceSize, refineSize.height() - y)));
                }
            }
            res.gridSize = new ImageSize(refineSize.width() / sliceSize, refineSize.height() / sliceSize);
            return res;
        }

        // no pinpoints: the dynamic grid (MiniCPM-V)
        res.overviewSize = bestResize(original, sliceSize, sliceAlign(), !hasSlices);
        int maxSliceNums = params.maxSliceNums() > 0 ? params.maxSliceNums() : 9;
        float logRatio = (float) Math.log((double) original.width() / original.height());
        float ratio = (float) original.width() * original.height() / (sliceSize * sliceSize);
        int multiple = Math.min((int) Math.ceil(ratio), maxSliceNums);

        ImageSize bestGrid = bestGrid(maxSliceNums, multiple, logRatio);
        ImageSize refineSize = refineSize(original, bestGrid, sliceSize, sliceAlign(), true);
        res.gridSize = bestGrid;
        res.refinedSize = refineSize;

        int cellWidth = refineSize.width() / bestGrid.width();
        int cellHeight = refineSize.height() / bestGrid.height();
        for (int y = 0, row = 0; y < refineSize.height() && row < bestGrid.height(); y += cellHeight, row++) {
            for (int x = 0, col = 0; x < refineSize.width() && col < bestGrid.width(); x += cellWidth, col++) {
                res.slices.add(new SliceRect(x, y, cellWidth, cellHeight));
            }
        }
        return res;
    }

    /** The overview and the slices of a plan ({@code slice_image}). */
    protected SliceOutput sliceImage(RasterImage image, SliceInstructions inst) {
        SliceOutput output = new SliceOutput();
        output.overview = ImageOps.resize(image, inst.overviewSize, params.overviewResizeAlgo(),
                params.overviewPadStyle(), params.padColor());
        if (inst.slices.isEmpty()) {
            return output;
        }
        RasterImage refined = ImageOps.resize(image, inst.refinedSize, params.refinedResizeAlgo(),
                params.refinedPadStyle(), params.refinedPadColor());
        for (SliceRect slice : inst.slices) {
            output.slices.add(ImageOps.crop(refined, slice.x, slice.y, slice.width, slice.height));
        }
        return output;
    }

    /**
     * The best aspect-preserving fit under {@code scaleResolution}, upscaled only when allowed
     * ({@code get_best_resize}); sides align to {@code patchSize}.
     */
    protected ImageSize bestResize(ImageSize original, int scaleResolution, int patchSize, boolean allowUpscale) {
        int width = original.width();
        int height = original.height();
        if ((width * height > scaleResolution * scaleResolution) || allowUpscale) {
            float r = (float) width / height;
            height = (int) (scaleResolution / Math.sqrt(r));
            width = (int) (height * r);
        }
        return new ImageSize(ensureDivide(width, patchSize), ensureDivide(height, patchSize));
    }

    /**
     * Rounds to the nearest patch multiple, never below one patch ({@code ensure_divide}); ties round half to
     * even when the params ask for Python rounding ({@code align_round} of the MiniCPM-V planner).
     */
    protected int ensureDivide(int length, int patchSize) {
        double q = (double) length / patchSize;
        long rounded = params.roundHalfEven() ? (long) Math.rint(q) : Math.round(q);
        return Math.max((int) rounded * patchSize, patchSize);
    }

    /**
     * The candidate with the most effective resolution, ties by the least wasted area
     * ({@code select_best_resolution}): the classic LLaVA-NeXT dynamic-resolution choice.
     */
    protected ImageSize selectBestResolution(ImageSize original, List<ImageSize> candidates) {
        ImageSize best = ImageSize.EMPTY;
        int minWastedArea = Integer.MAX_VALUE;
        int maxEffectiveResolution = 0;
        for (ImageSize candidate : candidates) {
            ImageSize target = resizeMaintainAspectRatio(original, candidate);
            int effective = Math.min(target.width() * target.height(), original.width() * original.height());
            int wasted = candidate.area() - effective;
            if (effective > maxEffectiveResolution
                    || (effective == maxEffectiveResolution && wasted < minWastedArea)) {
                maxEffectiveResolution = effective;
                minWastedArea = wasted;
                best = candidate;
            }
        }
        return best;
    }

    private ImageSize resizeMaintainAspectRatio(ImageSize original, ImageSize targetMax) {
        float scaleWidth = (float) targetMax.width() / original.width();
        float scaleHeight = (float) targetMax.height() / original.height();
        float scale = Math.min(scaleWidth, scaleHeight);
        return new ImageSize((int) (original.width() * scale), (int) (original.height() * scale));
    }

    /**
     * The refined size of the dynamic grid: every cell resizes to the CLIP size aligned to the merge factor,
     * so the whole refined image divides into whole slices ({@code get_refine_size}).
     */
    protected ImageSize refineSize(ImageSize original, ImageSize grid, int scaleResolution, int patchSize,
                                   boolean allowUpscale) {
        int refineWidth = ensureDivide(original.width(), grid.width());
        int refineHeight = ensureDivide(original.height(), grid.height());
        ImageSize cell = new ImageSize(refineWidth / grid.width(), refineHeight / grid.height());
        ImageSize bestCell = bestResize(cell, scaleResolution, patchSize, allowUpscale);
        return new ImageSize(bestCell.width() * grid.width(), bestCell.height() * grid.height());
    }

    /**
     * The grid whose shape best matches the aspect ratio, among the factorizations of the slice counts around
     * the area multiple ({@code get_best_grid}).
     */
    protected ImageSize bestGrid(int maxSliceNums, int multiple, float logRatio) {
        List<Integer> candidateSplitGridsNums = new ArrayList<>();
        for (int i : new int[] {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > maxSliceNums) {
                continue;
            }
            candidateSplitGridsNums.add(i);
        }

        List<ImageSize> candidateGrids = new ArrayList<>();
        for (int splitGridsNums : candidateSplitGridsNums) {
            for (int m = 1; m <= splitGridsNums; m++) {
                if (splitGridsNums % m == 0) {
                    candidateGrids.add(new ImageSize(m, splitGridsNums / m));
                }
            }
        }

        ImageSize best = new ImageSize(1, 1);
        float minError = Float.POSITIVE_INFINITY;
        for (ImageSize grid : candidateGrids) {
            float error = (float) Math.abs(logRatio - Math.log(1.0 * grid.width() / grid.height()));
            if (error < minError) {
                best = grid;
                minError = error;
            }
        }
        return best;
    }

    /** One slice rectangle over the refined image ({@code slice_coordinates}). */
    static final class SliceRect {
        final int x;
        final int y;
        final int width;
        final int height;

        SliceRect(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
    }

    /** The slice plan ({@code slice_instructions}); empty sizes mark "not applicable". */
    static final class SliceInstructions {
        ImageSize overviewSize = ImageSize.EMPTY;
        ImageSize refinedSize = ImageSize.EMPTY;
        ImageSize gridSize = ImageSize.EMPTY;
        final List<SliceRect> slices = new ArrayList<>();
    }

    /** The plan's images ({@code slice_output}). */
    static final class SliceOutput {
        RasterImage overview;
        final List<RasterImage> slices = new ArrayList<>();
    }
}
