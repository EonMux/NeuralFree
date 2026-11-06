package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * One row weight quantizer of the online-quantization seam: quantizes a float row into the ggml row the
 * reference quantizers of llama.cpp's row kernels store ({@link com.eonmux.neuralfree.gguf.memory.KvBlockCodec}'s
 * block quantizers). {@link OnlineQuantizers} is the
 * registry.
 */
public interface OnlineQuantizer {

    /** The ggml storage type the quantized rows are written as. */
    GGMLType type();

    /**
     * The storage type of a row of {@code n} values: {@link #type()}, unless the quantizer stores rows that are
     * no whole number of its blocks in a finer type (the FP8 quantizers: {@code F8_E4M3_B32} for rows of whole
     * 32-value runs).
     */
    default GGMLType typeFor(int n) {
        return type();
    }

    /**
     * Quantizes {@code n} values of {@code x} at {@code off} into {@code dst} at {@code dstOff},
     * {@code typeFor(n).getBytesForElements(n)} bytes. {@code n} is a whole number of blocks of the row type.
     *
     * @throws IllegalArgumentException when {@code n} is not a whole number of blocks of {@link #typeFor}
     */
    void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff);
}
