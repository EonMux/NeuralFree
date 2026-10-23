package com.eonmux.neuralfree.tensor;

import java.util.Arrays;
import java.util.Objects;

public final class Shape {
    private final int[] dims;

    private Shape(int[] dims) {
        this.dims = dims.clone();
    }

    public static Shape scalar() {
        return new Shape(new int[0]);
    }

    public static Shape vector(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Vector dimension must be non-negative: " + n);
        }
        return new Shape(new int[]{n});
    }

    public static Shape matrix(int rows, int cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("Matrix dimensions must be non-negative: rows=" + rows + ", cols=" + cols);
        }
        return new Shape(new int[]{rows, cols});
    }

    public static Shape of(int... dims) {
        Objects.requireNonNull(dims, "Dimensions array cannot be null");
        for (int i = 0; i < dims.length; i++) {
            if (dims[i] < 0) {
                throw new IllegalArgumentException("Dimension at index " + i + " must be non-negative: " + dims[i]);
            }
        }
        return new Shape(dims);
    }

    public int rank() {
        return dims.length;
    }

    public int dim(int axis) {
        if (axis < 0 || axis >= dims.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for shape with rank " + dims.length);
        }
        return dims[axis];
    }

    public int[] dims() {
        return dims.clone();
    }

    public long numElements() {
        if (dims.length == 0) {
            return 1;
        }
        long product = 1;
        for (int dim : dims) {
            product *= dim;
        }
        return product;
    }

    public long byteSize(DType dtype) {
        Objects.requireNonNull(dtype, "DType cannot be null");
        return numElements() * dtype.byteSize();
    }

    public Shape reshape(int... newDims) {
        Objects.requireNonNull(newDims, "New dimensions cannot be null");
        
        long newNumElements = 1;
        int inferIndex = -1;
        
        for (int i = 0; i < newDims.length; i++) {
            if (newDims[i] == -1) {
                if (inferIndex != -1) {
                    throw new IllegalArgumentException("Can only infer one dimension, but multiple -1 found");
                }
                inferIndex = i;
            } else if (newDims[i] < 0) {
                throw new IllegalArgumentException("Dimension at index " + i + " must be non-negative or -1 for inference: " + newDims[i]);
            } else {
                newNumElements *= newDims[i];
            }
        }
        
        long currentNumElements = numElements();
        
        if (inferIndex != -1) {
            if (currentNumElements % newNumElements != 0) {
                throw new IllegalArgumentException("Cannot reshape shape " + Arrays.toString(dims) + 
                    " with " + currentNumElements + " elements to shape with inferred dimension");
            }
            newDims = newDims.clone();
            newDims[inferIndex] = (int) (currentNumElements / newNumElements);
            newNumElements = currentNumElements;
        }
        
        if (newNumElements != currentNumElements) {
            throw new IllegalArgumentException("Cannot reshape shape " + Arrays.toString(dims) + 
                " with " + currentNumElements + " elements to shape " + Arrays.toString(newDims) + 
                " with " + newNumElements + " elements");
        }
        
        return new Shape(newDims);
    }

    public Shape transpose(int... axes) {
        Objects.requireNonNull(axes, "Axes cannot be null");
        
        if (axes.length != dims.length) {
            throw new IllegalArgumentException("Number of axes (" + axes.length + 
                ") must match rank (" + dims.length + ")");
        }
        
        boolean[] used = new boolean[dims.length];
        int[] newDims = new int[dims.length];
        
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i];
            if (axis < 0 || axis >= dims.length) {
                throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + dims.length);
            }
            if (used[axis]) {
                throw new IllegalArgumentException("Duplicate axis " + axis + " in transpose");
            }
            used[axis] = true;
            newDims[i] = dims[axis];
        }
        
        return new Shape(newDims);
    }

    public Shape transpose() {
        if (dims.length < 2) {
            throw new UnsupportedOperationException("Cannot transpose shape with rank < 2");
        }
        
        int[] axes = new int[dims.length];
        for (int i = 0; i < dims.length; i++) {
            axes[i] = dims.length - 1 - i;
        }
        return transpose(axes);
    }

    public Shape broadcast(Shape other) {
        Objects.requireNonNull(other, "Other shape cannot be null");
        
        int maxRank = Math.max(this.dims.length, other.dims.length);
        int[] resultDims = new int[maxRank];
        
        for (int i = 0; i < maxRank; i++) {
            int thisIdx = this.dims.length - maxRank + i;
            int otherIdx = other.dims.length - maxRank + i;
            
            int thisDim = thisIdx >= 0 ? this.dims[thisIdx] : 1;
            int otherDim = otherIdx >= 0 ? other.dims[otherIdx] : 1;
            
            if (thisDim != otherDim && thisDim != 1 && otherDim != 1) {
                throw new IllegalArgumentException(
                    "Shapes " + Arrays.toString(this.dims) + " and " + Arrays.toString(other.dims) + 
                    " are not broadcast-compatible at dimension " + i + 
                    ": " + thisDim + " vs " + otherDim);
            }
            
            resultDims[i] = Math.max(thisDim, otherDim);
        }
        
        return new Shape(resultDims);
    }

    public static boolean isBroadcastCompatible(Shape a, Shape b) {
        Objects.requireNonNull(a, "Shape a cannot be null");
        Objects.requireNonNull(b, "Shape b cannot be null");
        
        int maxRank = Math.max(a.dims.length, b.dims.length);
        
        for (int i = 0; i < maxRank; i++) {
            int aIdx = a.dims.length - maxRank + i;
            int bIdx = b.dims.length - maxRank + i;
            
            int aDim = aIdx >= 0 ? a.dims[aIdx] : 1;
            int bDim = bIdx >= 0 ? b.dims[bIdx] : 1;
            
            if (aDim != bDim && aDim != 1 && bDim != 1) {
                return false;
            }
        }
        
        return true;
    }

    public Shape squeeze() {
        int nonOneCount = 0;
        for (int dim : dims) {
            if (dim != 1) {
                nonOneCount++;
            }
        }
        
        int[] newDims = new int[nonOneCount];
        int idx = 0;
        for (int dim : dims) {
            if (dim != 1) {
                newDims[idx++] = dim;
            }
        }
        
        return new Shape(newDims);
    }

    public Shape squeeze(int axis) {
        if (axis < 0 || axis >= dims.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for rank " + dims.length);
        }
        
        if (dims[axis] != 1) {
            throw new IllegalArgumentException("Cannot squeeze axis " + axis + " with dimension " + dims[axis] + " (must be 1)");
        }
        
        int[] newDims = new int[dims.length - 1];
        int idx = 0;
        for (int i = 0; i < dims.length; i++) {
            if (i != axis) {
                newDims[idx++] = dims[i];
            }
        }
        
        return new Shape(newDims);
    }

    public Shape unsqueeze(int axis) {
        if (axis < 0 || axis > dims.length) {
            throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds for unsqueeze (valid: 0 to " + dims.length + ")");
        }
        
        int[] newDims = new int[dims.length + 1];
        System.arraycopy(dims, 0, newDims, 0, axis);
        newDims[axis] = 1;
        System.arraycopy(dims, axis, newDims, axis + 1, dims.length - axis);
        
        return new Shape(newDims);
    }

    public boolean isScalar() {
        return dims.length == 0;
    }

    public boolean isVector() {
        return dims.length == 1;
    }

    public boolean isMatrix() {
        return dims.length == 2;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Shape shape = (Shape) o;
        return Arrays.equals(dims, shape.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        return "Shape" + Arrays.toString(dims);
    }
}
