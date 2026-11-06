package com.eonmux.neuralfree.gguf.hf.st;

import com.eonmux.neuralfree.gguf.hf.tensor.ElementFormat;
import com.eonmux.neuralfree.gguf.hf.tensor.RowSource;
import com.eonmux.neuralfree.io.safetensors.StDType;

import java.util.Arrays;
import java.util.Objects;

/**
 * One tensor of a checkpoint container before the container is built: its name, its element encoding and its bytes as
 * a {@link RowSource}. The dtype and shape are the source's, so a derived source (a split or a reassembly) carries
 * the shape and encoding of what it produces. Immutable.
 */
public final class StoredTensor {

    private final String name;
    private final StDType dtype;
    private final long[] shape;
    private final RowSource source;

    private StoredTensor(String name, StDType dtype, long[] shape, RowSource source) {
        this.name = name;
        this.dtype = dtype;
        this.shape = shape;
        this.source = source;
    }

    /**
     * The tensor {@code name} served by {@code source}, whose encoding must be a safetensors dtype.
     *
     * @throws IllegalArgumentException when {@code name} is empty, the source is not stored as a safetensors dtype, or
     *                                  its shape and row width do not describe a whole number of element bytes
     */
    public static StoredTensor of(String name, RowSource source) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(source, "source");
        if (name.isEmpty()) {
            throw new IllegalArgumentException("a stored tensor needs a name");
        }
        ElementFormat format = source.format();
        StDType dtype = format.stDType();
        if (dtype == null) {
            throw new IllegalArgumentException("tensor '" + name + "' is stored as " + format
                    + ", which has no safetensors dtype");
        }
        long[] shape = source.shape();
        long numel = 1;
        try {
            for (long d : shape) {
                if (d < 0) {
                    throw new IllegalArgumentException("tensor '" + name + "' has a negative dimension in "
                            + Arrays.toString(shape));
                }
                numel = Math.multiplyExact(numel, d);
            }
            long bits = Math.multiplyExact(numel, (long) dtype.bits());
            if (bits % 8 != 0) {
                throw new IllegalArgumentException("tensor '" + name + "': " + numel + " " + dtype.headerName()
                        + " elements are " + bits + " bits, not a whole number of bytes");
            }
            long expected = bits / 8;
            long rows = source.rows();
            long rowBytes = source.rowBytes();
            long actual = Math.multiplyExact(rows, rowBytes);
            if (expected != actual) {
                throw new IllegalArgumentException("tensor '" + name + "': " + actual + " bytes as " + rows
                        + " rows of " + rowBytes + " do not hold " + numel + " " + dtype.headerName()
                        + " elements (" + expected + " bytes)");
            }
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("tensor '" + name + "': the byte size of shape "
                    + Arrays.toString(shape) + " overflows a 64-bit integer", e);
        }
        return new StoredTensor(name, dtype, shape, source);
    }

    /** The tensor's name. */
    public String name() {
        return name;
    }

    /** The element dtype. */
    public StDType dtype() {
        return dtype;
    }

    /** The shape, outermost dimension first (a copy). */
    public long[] shape() {
        return shape.clone();
    }

    /** The bytes, row by row. */
    public RowSource source() {
        return source;
    }

    @Override
    public String toString() {
        return "StoredTensor{" + name + ", " + dtype.headerName() + " " + Arrays.toString(shape) + "}";
    }
}
