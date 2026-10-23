package com.eonmux.neuralfree.tensor;

import java.util.Arrays;
import java.util.Objects;

public final class Strides {
    private final long[] strides;
    private final long offset;

    private Strides(long[] strides, long offset) {
        this.strides = strides.clone();
        this.offset = offset;
    }

    public static Strides scalar() {
        return new Strides(new long[0], 0);
    }

    public static Strides contiguous(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        int rank = shape.rank();
        if (rank == 0) {
            return scalar();
        }
        
        long[] strides = new long[rank];
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape.dim(i);
        }
        
        return new Strides(strides, 0);
    }

    /**
     * Element strides and element offset of an arbitrary view, for example the stride-0 dimensions of a
     * broadcast ({@link Tensor#expand}).
     *
     * @throws IllegalArgumentException when a stride or the offset is negative
     */
    public static Strides of(long[] strides, long offset) {
        Objects.requireNonNull(strides, "Strides cannot be null");
        for (int i = 0; i < strides.length; i++) {
            if (strides[i] < 0) {
                throw new IllegalArgumentException("Stride at axis " + i + " must be non-negative: " + strides[i]);
            }
        }
        if (offset < 0) {
            throw new IllegalArgumentException("Offset must be non-negative: " + offset);
        }
        return new Strides(strides, offset);
    }

    public static Strides fortran(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        int rank = shape.rank();
        if (rank == 0) {
            return scalar();
        }
        
        long[] strides = new long[rank];
        long stride = 1;
        for (int i = 0; i < rank; i++) {
            strides[i] = stride;
            stride *= shape.dim(i);
        }
        
        return new Strides(strides, 0);
    }

    public Strides transpose(int... axes) {
        Objects.requireNonNull(axes, "Axes cannot be null");
        
        if (axes.length != strides.length) {
            throw new IllegalArgumentException("Number of axes (" + axes.length + 
                ") must match rank (" + strides.length + ")");
        }
        
        boolean[] used = new boolean[strides.length];
        long[] newStrides = new long[strides.length];
        
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= strides.length) {
                throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + strides.length);
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Duplicate axis " + axis + " in transpose");
            }
            used[axis] = true;
            newStrides[i] = strides[axis];
        }
        
        return new Strides(newStrides, offset);
    }

    public Strides withOffset(long newOffset) {
        if (newOffset < 0) {
            throw new IllegalArgumentException("Offset must be non-negative: " + newOffset);
        }
        return new Strides(strides, newOffset);
    }

    public boolean isContiguous(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (shape.rank() != strides.length) {
            return false;
        }
        
        if (offset != 0) {
            return false;
        }
        
        long expectedStride = 1;
        for (int i = shape.rank() - 1; i >= 0; i--) {
            int dim = shape.dim(i);
            if (strides[i] != expectedStride) {
                if (dim == 1) {
                    continue;
                }
                return false;
            }
            expectedStride *= dim;
        }
        
        return true;
    }

    public boolean isFortran(Shape shape) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        
        if (shape.rank() != strides.length) {
            return false;
        }
        
        if (offset != 0) {
            return false;
        }
        
        long expectedStride = 1;
        for (int i = 0; i < shape.rank(); i++) {
            int dim = shape.dim(i);
            if (strides[i] != expectedStride) {
                if (dim == 1) {
                    continue;
                }
                return false;
            }
            expectedStride *= dim;
        }
        
        return true;
    }

    public long linearIndex(int... indices) {
        Objects.requireNonNull(indices, "Indices cannot be null");
        
        if (indices.length != strides.length) {
            throw new IllegalArgumentException("Number of indices (" + indices.length + 
                ") must match rank (" + strides.length + ")");
        }
        
        long linearIdx = offset;
        for (int i = 0; i < indices.length; i++) {
            linearIdx += (long) indices[i] * strides[i];
        }
        
        return linearIdx;
    }

    public long linearIndex(long... indices) {
        Objects.requireNonNull(indices, "Indices cannot be null");
        
        if (indices.length != strides.length) {
            throw new IllegalArgumentException("Number of indices (" + indices.length + 
                ") must match rank (" + strides.length + ")");
        }
        
        long linearIdx = offset;
        for (int i = 0; i < indices.length; i++) {
            linearIdx += indices[i] * strides[i];
        }
        
        return linearIdx;
    }

    public int rank() {
        return strides.length;
    }

    public long stride(int axis) {
        if (axis < 0 || axis >= strides.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + strides.length);
        }
        return strides[axis];
    }

    public long[] strides() {
        return strides.clone();
    }

    public long offset() {
        return offset;
    }

    public Strides slice(int axis, long start, long end) {
        if (axis < 0 || axis >= strides.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + strides.length);
        }
        if (start < 0) {
            throw new IllegalArgumentException("Start must be non-negative: " + start);
        }
        if (end < start) {
            throw new IllegalArgumentException("End (" + end + ") must be >= start (" + start + ")");
        }
        
        long newOffset = offset + start * strides[axis];
        return new Strides(strides, newOffset);
    }

    public Strides squeeze(int axis) {
        if (axis < 0 || axis >= strides.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + strides.length);
        }
        
        long[] newStrides = new long[strides.length - 1];
        System.arraycopy(strides, 0, newStrides, 0, axis);
        System.arraycopy(strides, axis + 1, newStrides, axis, strides.length - axis - 1);
        
        return new Strides(newStrides, offset);
    }

    public Strides unsqueeze(int axis) {
        if (axis < 0 || axis > strides.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for unsqueeze (valid: 0 to " + strides.length + ")");
        }
        
        long[] newStrides = new long[strides.length + 1];
        System.arraycopy(strides, 0, newStrides, 0, axis);
        newStrides[axis] = 1;
        System.arraycopy(strides, axis, newStrides, axis + 1, strides.length - axis);
        
        return new Strides(newStrides, offset);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Strides that = (Strides) o;
        return offset == that.offset && Arrays.equals(strides, that.strides);
    }

    @Override
    public int hashCode() {
        int result = Arrays.hashCode(strides);
        result = 31 * result + Long.hashCode(offset);
        return result;
    }

    @Override
    public String toString() {
        return "Strides{" +
            "strides=" + Arrays.toString(strides) +
            ", offset=" + offset +
            '}';
    }
}
