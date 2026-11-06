package com.eonmux.neuralfree.gguf.hub.slice;

import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Arrays;
import java.util.Objects;

/**
 * One tensor of a slice plan: the output name after renumbering, the output dtype and shape (possibly row-sliced),
 * and the contiguous byte span of the source shard that holds the kept bytes. A row slice keeps a prefix of the
 * first dimension, so the span is always {@code [sourceBegin, sourceBegin + byteLength)}; the output offset is
 * assigned by the writer. Immutable.
 */
final class TensorSlice {

    private final String sourceName;
    private final String outputName;
    private final StDType dtype;
    private final long[] outputShape;
    private final long[] originalShape;
    private final long sourceBegin;
    private final long byteLength;
    private final String sourceFile;

    TensorSlice(String sourceName, String outputName, StDType dtype, long[] outputShape, long[] originalShape,
                long sourceBegin, long byteLength, String sourceFile) {
        this.sourceName = Objects.requireNonNull(sourceName, "sourceName");
        this.outputName = Objects.requireNonNull(outputName, "outputName");
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.outputShape = Objects.requireNonNull(outputShape, "outputShape").clone();
        this.originalShape = Objects.requireNonNull(originalShape, "originalShape").clone();
        this.sourceBegin = sourceBegin;
        this.byteLength = byteLength;
        this.sourceFile = Objects.requireNonNull(sourceFile, "sourceFile");
        if (sourceBegin < 0 || byteLength < 0) {
            throw new IllegalArgumentException("tensor '" + sourceName + "': the source span [" + sourceBegin
                    + ", +" + byteLength + ") is not a valid range");
        }
    }

    /** The tensor's name in the source shard. */
    String sourceName() {
        return sourceName;
    }

    /** The tensor's name in the slice. */
    String outputName() {
        return outputName;
    }

    /** The output dtype, unchanged from the source. */
    StDType dtype() {
        return dtype;
    }

    /** The output shape, outermost dimension first (a copy). */
    long[] outputShape() {
        return outputShape.clone();
    }

    /** The source shape, outermost dimension first (a copy). */
    long[] originalShape() {
        return originalShape.clone();
    }

    /** Absolute offset in {@link #sourceFile()} of the first kept byte. */
    long sourceBegin() {
        return sourceBegin;
    }

    /** Byte length of the kept span, which is also the output byte length. */
    long byteLength() {
        return byteLength;
    }

    /** The repository path of the shard the bytes are read from. */
    String sourceFile() {
        return sourceFile;
    }

    /** Whether the span is shorter than the source tensor's. */
    boolean rowSliced() {
        return byteLength != product(originalShape) * dtype.bits() / 8;
    }

    private static long product(long[] shape) {
        long n = 1L;
        for (long d : shape) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    @Override
    public String toString() {
        return outputName + " " + dtype.headerName() + " " + Arrays.toString(outputShape);
    }
}
