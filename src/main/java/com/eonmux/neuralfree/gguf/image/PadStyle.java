package com.eonmux.neuralfree.gguf.image;

/**
 * How a resize fits the source into a target canvas, the {@code pad_style} of llama.cpp
 * (tools/mtmd/clip-model.h): the source is scaled to fit inside the target preserving its aspect ratio, placed at
 * the center and the border is filled with the pad color. {@code NONE} stretches instead, with no canvas.
 */
public enum PadStyle {

    /** Stretch to the target size, no padding. */
    NONE,

    /** The remaining border is filled; the fitted sides round up ({@code ceil}). */
    CEIL,

    /** The remaining border is filled; the fitted sides round to nearest ({@code nearest}). */
    NEAREST
}
