package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * A tensor-scoped quantizer of the online-quantization seam: its scales span more than one row of the
 * tensor — fp8_per_tensor's per-expert scale, fp8_per_block's 128x128 tile scales and nvfp4_per_token's
 * per-expert global scales (vLLM online fp8.py and nvfp4.py). {@link OnlineQuantizers} registers a prototype;
 * {@link #open} starts the scan of one tensor, or of the virtual tensor a fused projection's members form
 * ({@link RequantPolicy#quantizeFused}), the requantization producer feeds every stored element once, in any
 * order, as rows ({@link #scanRow}) or as column runs of a transposed member ({@link #scanColumn}) (the zero
 * padding rows of a fused group are never fed), {@link #freeze} ends the scan, and the same instance is
 * afterwards the frozen quantizer of the scan's scales: immutable and thread-safe then, and not thread-safe while
 * scanning. Rows are expert-major: a tensor stacked over {@code experts} experts keeps rows
 * {@code [e * rows / experts, (e + 1) * rows / experts)} for expert {@code e}, the layout vLLM quantizes each
 * expert's 2-D weight with. {@link #outputScale()} becomes the planned tensor's output multiplier (the
 * {@code <base>.scale} sibling).
 */
public interface ScanningQuantizer {

    /** The ggml storage type the quantized rows of whole blocks are written as. */
    GGMLType type();

    /**
     * The storage type of a stored row of {@code n} values that is a row of the scanned tensor: {@link #type()},
     * unless the quantizer stores rows that are no whole number of its blocks in a finer type (the FP8
     * quantizers: {@code F8_E4M3_B32} for rows of whole 32-value runs).
     */
    default GGMLType typeFor(int n) {
        return type();
    }

    /**
     * The storage type of a stored row of {@code n} values that runs down a column of the scanned tensor (a
     * transposed member, {@link #quantizeColumn}), when every such row starts at a scanned row that is a multiple
     * of {@code startAlign} (0 when all start at row 0); null when its blocks cannot carry the scales exactly.
     */
    GGMLType columnTypeFor(int n, long startAlign);

    /**
     * A fresh scanner over a tensor of {@code rows} rows of {@code inner} floats stacked over {@code experts}
     * experts (1 for a plain 2-D weight). The registered instance of a scanning target is the target's prototype:
     * every open scan is a fresh instance of it.
     *
     * @throws IllegalArgumentException when {@code rows} is not a whole number of {@code experts}, or rows of
     *                                  {@code inner} values have no storage type
     */
    ScanningQuantizer open(long rows, int inner, long experts);

    /** Accumulates row {@code row} (its {@code n} floats at {@code x[off]}) into the scales. */
    void scanRow(long row, float[] x, int off, int n);

    /**
     * Accumulates {@code n} floats at {@code x[off]} that are column {@code col} of rows
     * {@code [row, row + n)} into the scales.
     *
     * @throws IllegalArgumentException when the quantizer's scales do not run along columns
     */
    void scanColumn(long row, int col, float[] x, int off, int n);

    /** Ends the scan: every stored element was fed, and the scales are final. */
    void freeze();

    /**
     * Quantizes row {@code row} ({@code n} floats at {@code x[off]}) into {@code dst} at {@code dstOff},
     * {@code typeFor(n).getBytesForElements(n)} bytes. Only valid after {@link #freeze}.
     *
     * @throws IllegalArgumentException when {@code n} is not a whole number of blocks of {@link #typeFor}
     */
    void quantizeRow(long row, float[] x, int off, int n, byte[] dst, int dstOff);

    /**
     * Quantizes the {@code n} floats at {@code x[off]} that are column {@code col} of rows
     * {@code [row, row + n)} into one stored row of {@code type} at {@code dst[dstOff]},
     * {@code type.getBytesForElements(n)} bytes; {@code type} is the {@link #columnTypeFor column type} the
     * transposed member stores every row as. Only valid after {@link #freeze}.
     *
     * @throws IllegalArgumentException when a block of {@code type} cannot carry the scales of its run exactly
     */
    void quantizeColumn(long row, int col, float[] x, int off, int n, GGMLType type, byte[] dst, int dstOff);

    /**
     * The output multiplier of the frozen weights: 1 value, or 1 per expert of the stacked dimension; null when
     * the blocks carry the full scale. Only valid after {@link #freeze}.
     */
    default float[] outputScale() {
        return null;
    }
}
