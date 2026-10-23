package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * CPU reference of {@link TensorOps}' concatenation, split and stack, plus the host helpers every {@code Cpu*}
 * reference class shares: result construction, row-major index arithmetic and dim normalization.
 *
 * <p>Every reference reads its operands in row-major order ({@link Tensor#toFloatArray()}, so any view works),
 * computes in Java float arithmetic and uploads one new contiguous result to the first operand's backend.</p>
 */
final class CpuShapeOps {

    private CpuShapeOps() {
    }

    // ------------------------------------------------------------------ cat, split, stack

    /**
     * Concatenation along {@code dim}; the caller has validated the list ({@link #catShape}).
     *
     * @param resultShape the concatenated shape
     */
    static Tensor cat(List<Tensor> tensors, int dim, Shape resultShape) {
        float[] result = new float[hostLength(resultShape.numElements())];
        int offset = 0;
        for (Tensor t : tensors) {
            float[] data = t.toFloatArray();
            int tDimSize = t.shape().dim(dim);
            copySlice(data, t.shape(), result, resultShape, dim, 0, offset, tDimSize);
            offset += tDimSize;
        }
        return floats(tensors.get(0).backend(), resultShape, result, false);
    }

    /** {@code chunks} equal parts of {@code a} along {@code dim}, each copied into a fresh contiguous tensor. */
    static List<Tensor> split(Tensor a, int chunks, int dim) {
        int chunkSize = a.shape().dim(dim) / chunks;
        float[] data = a.toFloatArray();
        List<Tensor> result = new ArrayList<>(chunks);
        try {
            for (int i = 0; i < chunks; i++) {
                result.add(splitSlice(data, a, dim, i * chunkSize, (i + 1) * chunkSize));
            }
        } catch (RuntimeException e) {
            throw closeAll(result, e);
        }
        return result;
    }

    /**
     * The tensors stacked along a new axis {@code dim} (in {@code [0, rank]}); the caller has validated that they
     * share one shape. Tensor {@code t} lands at index {@code t} of the new axis for every outer and inner index.
     */
    static Tensor stack(List<Tensor> tensors, int dim, Shape resultShape) {
        Shape shape = tensors.get(0).shape();
        int outer = product(shape, 0, dim);
        int inner = product(shape, dim, shape.rank());
        int count = tensors.size();
        float[] result = new float[hostLength(resultShape.numElements())];
        for (int t = 0; t < count; t++) {
            float[] data = tensors.get(t).toFloatArray();
            for (int o = 0; o < outer; o++) {
                System.arraycopy(data, o * inner, result, (o * count + t) * inner, inner);
            }
        }
        return floats(tensors.get(0).backend(), resultShape, result, false);
    }

    /**
     * Validates a concatenation (non-empty list, one backend, equal ranks, equal dims except {@code dim}) and returns
     * its shape; {@code dim} is already normalized.
     */
    static Shape catShape(List<Tensor> tensors, int dim) {
        Tensor first = tensors.get(0);
        int rank = first.shape().rank();
        int[] otherDims = first.shape().dims();
        long totalDim = 0;
        for (Tensor t : tensors) {
            requireSameBackend(first, t);
            if (t.shape().rank() != rank) {
                throw new IllegalArgumentException("All tensors must have the same rank");
            }
            for (int i = 0; i < rank; i++) {
                if (i != dim && t.shape().dim(i) != otherDims[i]) {
                    throw new IllegalArgumentException("All dimensions except dim must match");
                }
            }
            totalDim += t.shape().dim(dim);
        }
        if (totalDim > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Concatenated dimension " + totalDim + " exceeds Integer.MAX_VALUE");
        }
        int[] resultDims = otherDims.clone();
        resultDims[dim] = (int) totalDim;
        return Shape.of(resultDims);
    }

    /** Validates a stack (non-empty list, one backend, one shape) and returns its shape; {@code dim} in [0, rank]. */
    static Shape stackShape(List<Tensor> tensors, int dim) {
        Tensor first = tensors.get(0);
        Shape shape = first.shape();
        for (Tensor t : tensors) {
            requireSameBackend(first, t);
            if (!t.shape().equals(shape)) {
                throw new IllegalArgumentException("All tensors must have the same shape for stack");
            }
        }
        int rank = shape.rank();
        int[] resultDims = new int[rank + 1];
        int j = 0;
        for (int i = 0; i < rank + 1; i++) {
            resultDims[i] = i == dim ? tensors.size() : shape.dim(j++);
        }
        return Shape.of(resultDims);
    }

    private static void copySlice(float[] src, Shape srcShape, float[] dst, Shape dstShape,
                                  int dim, int srcStart, int dstStart, int size) {
        int rank = srcShape.rank();
        int[] srcStrides = contiguousStrides(srcShape);
        int[] dstStrides = contiguousStrides(dstShape);
        int[] indices = new int[rank];
        copySliceRecursive(src, srcShape, srcStrides, dst, dstStrides, dim, srcStart, dstStart, size, indices, 0, rank);
    }

    private static void copySliceRecursive(float[] src, Shape srcShape, int[] srcStrides,
                                           float[] dst, int[] dstStrides, int dim,
                                           int srcStart, int dstStart, int size,
                                           int[] indices, int currentDim, int rank) {
        if (currentDim == rank) {
            int srcLinearIdx = 0;
            int dstLinearIdx = 0;
            for (int i = 0; i < rank; i++) {
                int srcIdx = (i == dim) ? indices[i] + srcStart : indices[i];
                int dstIdx = (i == dim) ? indices[i] + dstStart : indices[i];
                srcLinearIdx += srcIdx * srcStrides[i];
                dstLinearIdx += dstIdx * dstStrides[i];
            }
            dst[dstLinearIdx] = src[srcLinearIdx];
            return;
        }
        int limit = (currentDim == dim) ? size : srcShape.dim(currentDim);
        for (int i = 0; i < limit; i++) {
            indices[currentDim] = i;
            copySliceRecursive(src, srcShape, srcStrides, dst, dstStrides, dim,
                    srcStart, dstStart, size, indices, currentDim + 1, rank);
        }
    }

    private static Tensor splitSlice(float[] data, Tensor a, int dim, int start, int end) {
        Shape shape = a.shape();
        int[] resultDims = shape.dims();
        resultDims[dim] = end - start;
        Shape resultShape = Shape.of(resultDims);

        int rank = shape.rank();
        int[] srcStrides = contiguousStrides(shape);
        int[] dstStrides = contiguousStrides(resultShape);

        float[] result = new float[hostLength(resultShape.numElements())];
        int[] indices = new int[rank];

        for (long i = 0; i < resultShape.numElements(); i++) {
            long linearIdx = i;
            for (int j = rank - 1; j >= 0; j--) {
                indices[j] = (int) (linearIdx % resultShape.dim(j));
                linearIdx /= resultShape.dim(j);
            }
            int srcLinearIdx = 0;
            int dstLinearIdx = 0;
            for (int j = 0; j < rank; j++) {
                int srcIdx = (j == dim) ? indices[j] + start : indices[j];
                srcLinearIdx += srcIdx * srcStrides[j];
                dstLinearIdx += indices[j] * dstStrides[j];
            }
            result[dstLinearIdx] = data[srcLinearIdx];
        }
        return floats(a.backend(), resultShape, result, false);
    }

    // ------------------------------------------------------------------ shared host helpers

    /** A new contiguous FLOAT32 tensor of {@code shape} holding {@code data} (row-major), uploaded once. */
    static Tensor floats(ComputeBackend backend, Shape shape, float[] data, boolean requiresGrad) {
        requireCount(shape, data.length);
        TensorStorage storage = TensorStorage.fromHostFloat(backend, data);
        return Tensor.wrap(shape, DType.FLOAT32, Strides.contiguous(shape), storage, backend, requiresGrad);
    }

    /** A new contiguous INT32 tensor of {@code shape} holding {@code data} (row-major), uploaded once. */
    static Tensor ints(ComputeBackend backend, Shape shape, int[] data) {
        requireCount(shape, data.length);
        ByteBuffer buffer = ByteBuffer.allocate(hostLength((long) data.length * DType.INT32.byteSize()))
                .order(ByteOrder.nativeOrder());
        buffer.asIntBuffer().put(data);
        return Tensor.fromStorage(backend, shape, DType.INT32, TensorStorage.fromHost(backend, buffer, DType.INT32));
    }

    /** A new contiguous tensor of {@code shape} and {@code dtype} holding the elements' bytes in native order. */
    static Tensor bytes(ComputeBackend backend, Shape shape, DType dtype, byte[] data) {
        requireCount(shape, data.length / dtype.byteSize());
        TensorStorage storage = TensorStorage.fromHost(backend, ByteBuffer.wrap(data), dtype);
        return Tensor.fromStorage(backend, shape, dtype, storage);
    }

    /** Row-major strides of {@code shape} in elements. */
    static int[] contiguousStrides(Shape shape) {
        int rank = shape.rank();
        int[] strides = new int[rank];
        int stride = 1;
        for (int i = rank - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape.dim(i);
        }
        return strides;
    }

    /** Row-major coordinates of element {@code linearIdx} of {@code shape}, written into {@code indices}. */
    static void unravelIndex(long linearIdx, Shape shape, int[] indices) {
        long rest = linearIdx;
        for (int j = indices.length - 1; j >= 0; j--) {
            indices[j] = (int) (rest % shape.dim(j));
            rest /= shape.dim(j);
        }
    }

    /** Product of {@code shape}'s dims {@code [from, to)}. */
    static int product(Shape shape, int from, int to) {
        long p = 1;
        for (int i = from; i < to; i++) {
            p *= shape.dim(i);
        }
        return hostLength(p);
    }

    /**
     * {@code dim} normalized to {@code [0, rank)}.
     *
     * @throws IndexOutOfBoundsException when {@code dim} is outside {@code [-rank, rank)}
     */
    static int normalizeDim(int dim, int rank) {
        return BroadcastEngine.normalizeAxes(new int[]{dim}, rank)[0];
    }

    static void requireSameBackend(Tensor a, Tensor b) {
        if (a.backend() != b.backend()) {
            throw new IllegalArgumentException("Tensors must be on the same backend");
        }
    }

    /**
     * Validates gather/scatter shapes (equal ranks, every index dim but {@code dim} within the data's) and returns
     * {@code dim} normalized to {@code [0, rank)}; the one rule both the device dispatch and the CPU reference apply.
     *
     * @throws IllegalArgumentException  when the ranks differ or an index dim other than {@code dim} exceeds the data's
     * @throws IndexOutOfBoundsException when {@code dim} is outside {@code [-rank, rank)}
     */
    static int checkIndexShapes(Shape data, int dim, Shape index) {
        int rank = data.rank();
        if (index.rank() != rank) {
            throw new IllegalArgumentException("Input and index must have the same rank");
        }
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) {
            throw new IndexOutOfBoundsException("Axis " + d + " out of bounds for shape with rank " + rank);
        }
        for (int k = 0; k < rank; k++) {
            if (k != d && index.dim(k) > data.dim(k)) {
                throw new IllegalArgumentException("Index shape " + index + " exceeds shape " + data + " at dimension "
                        + k + " (only dimension " + d + " may differ freely)");
            }
        }
        return d;
    }

    /** A host array length, checked against the Java array limit. */
    static int hostLength(long n) {
        if (n < 0 || n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException(n + " elements exceed the Java array limit");
        }
        return (int) n;
    }

    /** Closes every tensor of {@code tensors} and returns {@code failure} with close failures suppressed into it. */
    static RuntimeException closeAll(List<Tensor> tensors, RuntimeException failure) {
        for (int i = tensors.size() - 1; i >= 0; i--) {
            try {
                tensors.get(i).close();
            } catch (RuntimeException e) {
                failure.addSuppressed(e);
            }
        }
        return failure;
    }

    private static void requireCount(Shape shape, long count) {
        if (shape.numElements() != count) {
            throw new IllegalArgumentException(count + " values for a result of shape " + shape);
        }
    }
}
