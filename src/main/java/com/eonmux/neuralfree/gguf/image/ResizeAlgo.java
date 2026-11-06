package com.eonmux.neuralfree.gguf.image;

import java.util.Locale;

/**
 * Resampling filter of the image resize, the {@code resize_algo} of llama.cpp (tools/mtmd/clip-model.h): Pillow's
 * separable resampling with bilinear, bicubic (a = -0.5) or Lanczos3 kernels
 * ({@code img_tool::resize_pillow}, tools/mtmd/mtmd-image.cpp). The {@code TORCH_} variants are the same filters
 * with torchvision's int16 weights ({@code RESIZE_ALGO_FLAG_TORCH}, the antialiased uint8 resize of
 * aten/src/ATen/native/cpu/UpSampleKernel.cpp); torchvision has no Lanczos.
 */
public enum ResizeAlgo {

    BILINEAR,
    BICUBIC,
    LANCZOS,
    TORCH_BILINEAR,
    TORCH_BICUBIC;

    /**
     * The algorithm named by a GGUF {@code clip.vision.image_resize_algo} value.
     *
     * @throws IllegalArgumentException for any other name
     */
    public static ResizeAlgo fromName(String name) {
        if ("bilinear".equals(name)) return BILINEAR;
        if ("bicubic".equals(name)) return BICUBIC;
        if ("lanczos".equals(name)) return LANCZOS;
        throw new IllegalArgumentException("unsupported image resize algo: '" + name + "'"
                + " (supported: bilinear, bicubic, lanczos)");
    }

    /**
     * The torchvision variant of this filter ({@code set_image_resize_algo(algo, true)}).
     *
     * @throws IllegalArgumentException for Lanczos, which torchvision does not have
     */
    public ResizeAlgo torch() {
        switch (this) {
            case BILINEAR:
            case TORCH_BILINEAR:
                return TORCH_BILINEAR;
            case BICUBIC:
            case TORCH_BICUBIC:
                return TORCH_BICUBIC;
            default:
                throw new IllegalArgumentException("torchvision has no " + name().toLowerCase(Locale.ROOT)
                        + " resize");
        }
    }

    /** The filter kernel without the weight precision: {@code BILINEAR}, {@code BICUBIC} or {@code LANCZOS}. */
    public ResizeAlgo kernel() {
        switch (this) {
            case TORCH_BILINEAR:
                return BILINEAR;
            case TORCH_BICUBIC:
                return BICUBIC;
            default:
                return this;
        }
    }

    /** Whether the weights round to torchvision's int16 precision. */
    public boolean int16Weights() {
        return this == TORCH_BILINEAR || this == TORCH_BICUBIC;
    }
}
