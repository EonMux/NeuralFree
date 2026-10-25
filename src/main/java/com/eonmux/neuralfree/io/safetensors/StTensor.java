package com.eonmux.neuralfree.io.safetensors;

import java.util.Arrays;
import java.util.Objects;

/**
 * One tensor entry of a safetensors header: name, dtype, shape in PyTorch order (outermost dimension first) and the
 * byte range {@code [begin, begin + byteLength)} relative to the start of the file's data section. Immutable.
 */
public final class StTensor {

    private final String name;
    private final StDType dtype;
    private final long[] shape;
    private final long begin;
    private final long byteLength;
    private final long numel;

    /**
     * @throws IllegalArgumentException when a dimension or {@code begin} is negative, the element count overflows, or
     *                                  {@code byteLength} is not the byte size of {@code numel} elements of
     *                                  {@code dtype}
     */
    public StTensor(String name, StDType dtype, long[] shape, long begin, long byteLength) {
        this.name = Objects.requireNonNull(name, "name");
        this.dtype = Objects.requireNonNull(dtype, "dtype");
        this.shape = Objects.requireNonNull(shape, "shape").clone();
        if (begin < 0 || byteLength < 0 || byteLength > Long.MAX_VALUE - begin) {
            throw new IllegalArgumentException("tensor '" + name + "': byte range [" + begin + ", +" + byteLength
                    + ") is not a valid range");
        }
        this.begin = begin;
        this.byteLength = byteLength;
        this.numel = elementCount(name, this.shape);
        long expected = byteSize(name, dtype, numel);
        if (expected != byteLength) {
            throw new IllegalArgumentException("tensor '" + name + "': " + byteLength + " bytes do not hold "
                    + numel + " " + dtype.headerName() + " elements (" + expected + " bytes)");
        }
    }

    /** The product of {@code shape}, 1 for a scalar. */
    static long elementCount(String name, long[] shape) {
        long n = 1;
        for (long d : shape) {
            if (d < 0) {
                throw new IllegalArgumentException("tensor '" + name + "': negative dimension in shape "
                        + Arrays.toString(shape));
            }
            try {
                n = Math.multiplyExact(n, d);
            } catch (ArithmeticException e) {
                throw new IllegalArgumentException("tensor '" + name + "': element count of shape "
                        + Arrays.toString(shape) + " overflows a 64-bit integer", e);
            }
        }
        return n;
    }

    /** Bytes of {@code numel} elements of {@code dtype}; the bit count must be a whole number of bytes. */
    static long byteSize(String name, StDType dtype, long numel) {
        long bits;
        try {
            bits = Math.multiplyExact(numel, (long) dtype.bits());
        } catch (ArithmeticException e) {
            throw new IllegalArgumentException("tensor '" + name + "': " + numel + " " + dtype.headerName()
                    + " elements overflow a 64-bit bit count", e);
        }
        if (bits % 8 != 0) {
            throw new IllegalArgumentException("tensor '" + name + "': " + numel + " " + dtype.headerName()
                    + " elements are " + bits + " bits, not a whole number of bytes");
        }
        return bits / 8;
    }

    /** The tensor's name in the header. */
    public String name() { return name; }

    /** The element dtype. */
    public StDType dtype() { return dtype; }

    /** The shape, outermost dimension first (a copy). */
    public long[] shape() { return shape.clone(); }

    /** The number of dimensions; 0 for a scalar. */
    public int rank() { return shape.length; }

    /** Byte offset of the first byte, relative to the start of the data section. */
    public long begin() { return begin; }

    /** Byte offset one past the last byte, relative to the start of the data section. */
    public long end() { return begin + byteLength; }

    /** Byte length of the tensor's data. */
    public long byteLength() { return byteLength; }

    /** The number of elements: the product of the shape, 1 for a scalar. */
    public long numel() { return numel; }

    /** The shape in GGUF order, innermost dimension first (a reversed copy). */
    public long[] ne() {
        long[] ne = new long[shape.length];
        for (int i = 0; i < shape.length; i++) {
            ne[i] = shape[shape.length - 1 - i];
        }
        return ne;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof StTensor)) {
            return false;
        }
        StTensor t = (StTensor) o;
        return name.equals(t.name) && dtype == t.dtype && Arrays.equals(shape, t.shape) && begin == t.begin
                && byteLength == t.byteLength;
    }

    @Override
    public int hashCode() {
        int h = name.hashCode();
        h = 31 * h + dtype.hashCode();
        h = 31 * h + Arrays.hashCode(shape);
        h = 31 * h + Long.hashCode(begin);
        return 31 * h + Long.hashCode(byteLength);
    }

    @Override
    public String toString() {
        return "StTensor{" + name + ", " + dtype.headerName() + " " + Arrays.toString(shape) + ", data=[" + begin
                + ", " + end() + ")}";
    }
}
