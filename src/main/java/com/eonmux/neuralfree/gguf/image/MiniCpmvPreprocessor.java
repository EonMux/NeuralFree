package com.eonmux.neuralfree.gguf.image;

import com.eonmux.neuralfree.gguf.mm.VisionPreprocessing;

/**
 * The MiniCPM-V preprocessor ({@code mtmd_image_preprocessor_minicpmv} and
 * {@code mtmd_image_preprocessor_minicpm_ocr_1_0}, tools/mtmd/mtmd-image.cpp): an image whose area fits one slice
 * skips the slicing and encodes as a single overview tile at its best aspect-aligned size; larger images go
 * through the generic llava-uhd grid. The 4.6/4.7 merger aligns every size to four patches whatever its merge
 * factor, MiniCPM-OCR 1.0 to the merged patch, the resampler to one patch.
 */
public final class MiniCpmvPreprocessor extends LlavaUhdPreprocessor {

    /** Creates the preprocessor over the params. */
    public MiniCpmvPreprocessor(ImageProcessorParams params) {
        super(params);
    }

    @Override
    protected int sliceAlign() {
        if (params().preprocessing().family() == VisionPreprocessing.Family.MINICPMV_MERGER) {
            return params().patchSize() * 4;
        }
        return super.sliceAlign();
    }

    @Override
    protected SliceInstructions sliceInstructions(ImageSize original) {
        int sliceSize = params().imageSize();
        float ratio = (float) original.width() * original.height() / (sliceSize * sliceSize);
        if (ratio <= 1.0f) {
            SliceInstructions inst = new SliceInstructions();
            inst.overviewSize = bestResize(original, sliceSize, sliceAlign(), true);
            return inst;
        }
        return super.sliceInstructions(original);
    }
}
