package com.eonmux.neuralfree.gguf.quant;

import com.eonmux.neuralfree.gguf.GGMLType;

/**
 * One row weight quantizer of the online-quantization seam (QZ-1): quantizes a float row into the ggml row the
 * reference quantizers of llama.cpp's row kernels store ({@link com.eonmux.neuralfree.gguf.memory.KvBlockCodec}'s
 * block quantizers). PAR QN-4 registers the K-quant, IQ, TQ and FP8 quantizers; {@link OnlineQuantizers} is the
 * registry.
 */
public interface OnlineQuantizer {

    /** The ggml storage type the quantized rows are written as. */
    GGMLType type();

    /**
     * Quantizes {@code n} values of {@code x} at {@code off} into {@code dst} at {@code dstOff},
     * {@code type().getBytesForElements(n)} bytes. {@code n} is a whole number of blocks of the row type.
     *
     * @throws IllegalArgumentException when {@code n} is not a whole number of blocks of {@link #type()}
     */
    void quantizeRow(float[] x, int off, int n, byte[] dst, int dstOff);
}
