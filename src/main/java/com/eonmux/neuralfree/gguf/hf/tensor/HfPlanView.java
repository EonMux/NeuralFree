package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.io.IOException;
import java.util.List;

/**
 * A read-only view of the GGUF tensors planned for a Hugging Face checkpoint, for the hooks that run after tensor
 * planning (metadata and vocabulary hooks): names, GGUF shapes and types, and the values of small tensors (llama.cpp
 * metadata derived from tensor data, such as rope factors or alibi slopes).
 */
public interface HfPlanView {

    /** The largest element count a caller passes to {@link #readF32} or {@link #readI64}. */
    int MAX_READ_ELEMENTS = 262_144;

    /** The planned GGUF tensor names, in plan order. */
    List<String> names();

    /** True when {@code ggufName} is planned. */
    boolean has(String ggufName);

    /**
     * The GGUF shape ({@code ne}, innermost first) of {@code ggufName} (a copy).
     *
     * @throws IllegalArgumentException when the name is not planned
     */
    long[] ne(String ggufName);

    /**
     * The planned GGUF type of {@code ggufName}.
     *
     * @throws IllegalArgumentException when the name is not planned
     */
    GGMLType type(String ggufName);

    /**
     * The values of {@code ggufName} as floats, decoded from its source format (any float, integer or ggml type), in
     * row-major order.
     *
     * @param maxElements the most elements the caller accepts, at most {@link #MAX_READ_ELEMENTS}
     * @throws IllegalArgumentException when the name is not planned or the tensor has more than {@code maxElements}
     *                                  elements
     * @throws IOException              when the data cannot be read
     */
    float[] readF32(String ggufName, int maxElements) throws IOException;

    /**
     * The values of the integer tensor {@code ggufName}, read exactly, in row-major order.
     *
     * @param maxElements the most elements the caller accepts, at most {@link #MAX_READ_ELEMENTS}
     * @throws IllegalArgumentException when the name is not planned, its source is not an integer format, or the
     *                                  tensor has more than {@code maxElements} elements
     * @throws IOException              when the data cannot be read
     */
    long[] readI64(String ggufName, int maxElements) throws IOException;
}
