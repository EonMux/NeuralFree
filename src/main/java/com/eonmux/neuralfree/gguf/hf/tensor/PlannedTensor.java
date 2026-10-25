package com.eonmux.neuralfree.gguf.hf.tensor;

import com.eonmux.neuralfree.gguf.GGMLType;

import java.util.Arrays;
import java.util.Objects;

/**
 * One GGUF tensor of a {@link TensorPlan}: its name, its source in the planned type, the GGUF shape ({@code ne},
 * innermost first) and its byte size. Immutable.
 */
public final class PlannedTensor {

    private final String ggufName;
    private final RowSource source;
    private final GGMLType type;
    private final long[] ne;
    private final long byteSize;

    /**
     * @throws IllegalArgumentException when the source's format is not {@code type}, or {@code ne} and
     *                                  {@code byteSize} do not describe the source
     */
    public PlannedTensor(String ggufName, RowSource source, GGMLType type, long[] ne, long byteSize) {
        this.ggufName = Objects.requireNonNull(ggufName, "ggufName");
        this.source = Objects.requireNonNull(source, "source");
        this.type = Objects.requireNonNull(type, "type");
        this.ne = ne.clone();
        this.byteSize = byteSize;
        if (source.format().ggmlType() != type) {
            throw new IllegalArgumentException(ggufName + ": source " + source.describe() + " holds "
                    + source.format() + ", not the planned type " + type);
        }
        if (!Arrays.equals(this.ne, neOf(source.shape()))) {
            throw new IllegalArgumentException(ggufName + ": ne " + Arrays.toString(ne) + " is not the reverse of the"
                    + " source shape " + Arrays.toString(source.shape()));
        }
        if (byteSize != Math.multiplyExact(source.rows(), source.rowBytes())) {
            throw new IllegalArgumentException(ggufName + ": " + byteSize + " bytes do not match " + source.rows()
                    + " rows of " + source.rowBytes() + " bytes");
        }
    }

    /** The GGUF {@code ne} of a PyTorch shape: the reverse, and {@code {1}} for a scalar. */
    public static long[] neOf(long[] shape) {
        if (shape.length == 0) {
            return new long[]{1};
        }
        long[] ne = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ne[i] = shape[shape.length - 1 - i];
        }
        return ne;
    }

    public String ggufName() {
        return ggufName;
    }

    /** The bytes of the tensor, in {@link #type()}. */
    public RowSource source() {
        return source;
    }

    public GGMLType type() {
        return type;
    }

    /** The GGUF shape, innermost first (a copy). */
    public long[] ne() {
        return ne.clone();
    }

    public long byteSize() {
        return byteSize;
    }

    /** The element count. */
    public long elements() {
        long n = 1;
        for (long d : ne) {
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    @Override
    public String toString() {
        return "PlannedTensor{" + ggufName + ", " + type + ", ne " + Arrays.toString(ne) + ", " + byteSize
                + " bytes, from " + source.describe() + "}";
    }
}
