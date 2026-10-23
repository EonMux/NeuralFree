package com.eonmux.neuralfree.tensor;

import java.util.Arrays;
import java.util.Objects;

public final class BroadcastEngine {

    private BroadcastEngine() {
    }

    public static Shape broadcastShape(Shape a, Shape b) {
        Objects.requireNonNull(a, "Shape a cannot be null");
        Objects.requireNonNull(b, "Shape b cannot be null");

        int maxRank = Math.max(a.rank(), b.rank());
        int[] resultDims = new int[maxRank];

        for (int i = 0; i < maxRank; i++) {
            int aIdx = a.rank() - maxRank + i;
            int bIdx = b.rank() - maxRank + i;

            int aDim = aIdx >= 0 ? a.dim(aIdx) : 1;
            int bDim = bIdx >= 0 ? b.dim(bIdx) : 1;

            if (aDim != bDim && aDim != 1 && bDim != 1) {
                throw new IllegalArgumentException(
                    "Shapes " + Arrays.toString(a.dims()) + " and " + Arrays.toString(b.dims()) +
                    " are not broadcast-compatible at dimension " + i +
                    ": " + aDim + " vs " + bDim);
            }

            resultDims[i] = Math.max(aDim, bDim);
        }

        return Shape.of(resultDims);
    }

    public static long[] broadcastStrides(Shape sourceShape, Shape targetShape, Strides sourceStrides) {
        Objects.requireNonNull(sourceShape, "Source shape cannot be null");
        Objects.requireNonNull(targetShape, "Target shape cannot be null");
        Objects.requireNonNull(sourceStrides, "Source strides cannot be null");

        if (!isBroadcastableTo(sourceShape, targetShape)) {
            throw new IllegalArgumentException(
                "Shape " + sourceShape + " cannot be broadcast to " + targetShape);
        }

        int targetRank = targetShape.rank();
        long[] resultStrides = new long[targetRank];

        for (int i = 0; i < targetRank; i++) {
            int sourceIdx = sourceShape.rank() - targetRank + i;

            if (sourceIdx < 0) {
                resultStrides[i] = 0;
            } else if (sourceShape.dim(sourceIdx) == 1) {
                resultStrides[i] = 0;
            } else {
                resultStrides[i] = sourceStrides.stride(sourceIdx);
            }
        }

        return resultStrides;
    }

    public static boolean isBroadcastableTo(Shape source, Shape target) {
        Objects.requireNonNull(source, "Source shape cannot be null");
        Objects.requireNonNull(target, "Target shape cannot be null");

        if (source.rank() > target.rank()) {
            return false;
        }

        int offset = target.rank() - source.rank();

        for (int i = 0; i < source.rank(); i++) {
            int sourceDim = source.dim(i);
            int targetDim = target.dim(offset + i);

            if (sourceDim != targetDim && sourceDim != 1) {
                return false;
            }
        }

        return true;
    }

    public static BroadcastInfo computeBroadcast(Shape a, Shape b) {
        Objects.requireNonNull(a, "Shape a cannot be null");
        Objects.requireNonNull(b, "Shape b cannot be null");

        Shape resultShape = broadcastShape(a, b);

        Strides contiguousStridesA = Strides.contiguous(a);
        Strides contiguousStridesB = Strides.contiguous(b);

        long[] broadcastStridesA = broadcastStrides(a, resultShape, contiguousStridesA);
        long[] broadcastStridesB = broadcastStrides(b, resultShape, contiguousStridesB);

        return new BroadcastInfo(resultShape, broadcastStridesA, broadcastStridesB);
    }

    public static Shape reduceShape(Shape shape, int[] dims, boolean keepDim) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dims, "Dims cannot be null");

        if (dims.length == 0) {
            return keepDim ? shape : Shape.scalar();
        }

        int[] normalizedDims = normalizeAxes(dims, shape.rank());

        if (keepDim) {
            int[] resultDims = shape.dims();
            for (int dim : normalizedDims) {
                resultDims[dim] = 1;
            }
            return Shape.of(resultDims);
        } else {
            int resultRank = shape.rank() - normalizedDims.length;
            if (resultRank == 0) {
                return Shape.scalar();
            }

            int[] resultDims = new int[resultRank];
            int resultIdx = 0;
            for (int i = 0; i < shape.rank(); i++) {
                boolean isReduced = false;
                for (int dim : normalizedDims) {
                    if (dim == i) {
                        isReduced = true;
                        break;
                    }
                }
                if (!isReduced) {
                    resultDims[resultIdx++] = shape.dim(i);
                }
            }
            return Shape.of(resultDims);
        }
    }

    public static int[] normalizeAxes(int[] axes, int rank) {
        int[] normalized = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            if (axes[i] < 0) {
                normalized[i] = axes[i] + rank;
            } else {
                normalized[i] = axes[i];
            }
            if (normalized[i] < 0 || normalized[i] >= rank) {
                throw new IndexOutOfBoundsException(
                    "Axis " + axes[i] + " out of bounds for rank " + rank);
            }
        }
        return normalized;
    }

    public static Shape squeezeShape(Shape shape, int[] dims) {
        Objects.requireNonNull(shape, "Shape cannot be null");
        Objects.requireNonNull(dims, "Dims cannot be null");

        if (dims.length == 0) {
            return shape.squeeze();
        }

        int[] normalizedDims = normalizeAxes(dims, shape.rank());
        Arrays.sort(normalizedDims);

        int[] resultDims = new int[shape.rank() - normalizedDims.length];
        int resultIdx = 0;
        int skipIdx = 0;

        for (int i = 0; i < shape.rank(); i++) {
            if (skipIdx < normalizedDims.length && normalizedDims[skipIdx] == i) {
                skipIdx++;
            } else {
                resultDims[resultIdx++] = shape.dim(i);
            }
        }

        return resultDims.length == 0 ? Shape.scalar() : Shape.of(resultDims);
    }

    public static final class BroadcastInfo {
        private final Shape resultShape;
        private final long[] stridesA;
        private final long[] stridesB;

        public BroadcastInfo(Shape resultShape, long[] stridesA, long[] stridesB) {
            this.resultShape = Objects.requireNonNull(resultShape, "Result shape cannot be null");
            this.stridesA = stridesA.clone();
            this.stridesB = stridesB.clone();
        }

        public Shape resultShape() {
            return resultShape;
        }

        public long[] stridesA() {
            return stridesA.clone();
        }

        public long[] stridesB() {
            return stridesB.clone();
        }

        public long strideA(int axis) {
            if (axis < 0 || axis >= stridesA.length) {
                throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds");
            }
            return stridesA[axis];
        }

        public long strideB(int axis) {
            if (axis < 0 || axis >= stridesB.length) {
                throw new IndexOutOfBoundsException("Axis " + axis + " out of bounds");
            }
            return stridesB[axis];
        }

        @Override
        public String toString() {
            return "BroadcastInfo{" +
                "resultShape=" + resultShape +
                ", stridesA=" + Arrays.toString(stridesA) +
                ", stridesB=" + Arrays.toString(stridesB) +
                '}';
        }
    }
}
