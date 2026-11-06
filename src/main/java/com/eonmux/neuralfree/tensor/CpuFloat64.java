package com.eonmux.neuralfree.tensor;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * The FLOAT64 host path of the forward tensor ops: broadcasting arithmetic, the elementwise functions and
 * activations, {@code sum}/{@code mean}, {@code matmul}/{@code bmm} and {@code cat}; the convolution, pooling and
 * batch-norm forwards are in {@link CpuFloat64Conv}. Each op has the shapes, argument checks and loop order of its
 * FLOAT32 CPU reference, with every value, product and accumulator a Java {@code double}. It is the precision
 * reference of the gradient check's finite differences ({@code GradientCheckUtil.gradCheckFloat64}); the device
 * kernels and the backward primitives stay FLOAT32-only.
 *
 * <p>The FLOAT32 CPU entry points call {@link #applies} first: an op whose operands are all FLOAT64 runs here on the
 * operands' backend (values cross through the host), an op that mixes FLOAT64 with another dtype is refused, and
 * every other call keeps its FLOAT32 path. Results are new contiguous FLOAT64 tensors that do not require grad.</p>
 */
final class CpuFloat64 {

    private CpuFloat64() {
    }

    /**
     * Whether {@code op} takes the FLOAT64 path: true when a non-null operand is FLOAT64 and every non-null operand
     * is, false when none is.
     *
     * @throws IllegalArgumentException when FLOAT64 is mixed with another dtype
     */
    static boolean applies(String op, Tensor... operands) {
        boolean any = false;
        DType other = null;
        for (Tensor t : operands) {
            if (t == null) {
                continue;
            }
            if (t.dtype() == DType.FLOAT64) {
                any = true;
            } else {
                other = t.dtype();
            }
        }
        if (any && other != null) {
            throw new IllegalArgumentException(op + " needs operands of one dtype; got FLOAT64 and " + other);
        }
        return any;
    }

    /** {@link #applies} over a list of operands. */
    static boolean applies(String op, List<Tensor> operands) {
        return applies(op, operands.toArray(new Tensor[0]));
    }

    /** A new contiguous FLOAT64 tensor of {@code shape} holding {@code data} (row-major) on {@code backend}. */
    static Tensor doubles(ComputeBackend backend, Shape shape, double[] data) {
        if (shape.numElements() != data.length) {
            throw new IllegalArgumentException(data.length + " values for shape " + shape);
        }
        ByteBuffer bytes = ByteBuffer.allocate(CpuShapeOps.hostLength((long) data.length * DType.FLOAT64.byteSize()))
                .order(ByteOrder.nativeOrder());
        bytes.asDoubleBuffer().put(data);
        TensorStorage storage = TensorStorage.fromHost(backend, bytes, DType.FLOAT64);
        return Tensor.fromStorage(backend, shape, DType.FLOAT64, storage);
    }

    // ------------------------------------------------------------------ elementwise

    /** {@code op} over the broadcast of {@code a} and {@code b}, as {@code CpuElementwise}'s binary ops. */
    static Tensor binary(Tensor a, Tensor b, DoubleBinaryOperator op) {
        BroadcastEngine.BroadcastInfo broadcast = BroadcastEngine.computeBroadcast(a.shape(), b.shape());
        Shape resultShape = broadcast.resultShape();
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        double[] result = new double[CpuShapeOps.hostLength(resultShape.numElements())];
        CpuElementwise.BroadcastWalk walk = new CpuElementwise.BroadcastWalk(broadcast);
        for (int i = 0; i < result.length; i++) {
            walk.locate(i);
            result[i] = op.applyAsDouble(dataA[walk.indexA], dataB[walk.indexB]);
        }
        return doubles(a.backend(), resultShape, result);
    }

    /** {@code f} applied to every element of {@code a} in row-major order. */
    static Tensor map(Tensor a, DoubleUnaryOperator f) {
        double[] data = a.toDoubleArray();
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = f.applyAsDouble(data[i]);
        }
        return doubles(a.backend(), a.shape(), result);
    }

    static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    static double softplus(double x) {
        return Math.log(1.0 + Math.exp(x));
    }

    /** GELU with the tanh approximation of the FLOAT32 path. */
    static double gelu(double x) {
        double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + Math.tanh(inner));
    }

    // ------------------------------------------------------------------ reductions

    /**
     * The sum ({@code mean} false) or mean of {@code a} over the normalized, distinct {@code dims}, or over every
     * element into a scalar when {@code dims} is empty; terms are added in row-major order.
     */
    static Tensor sumOrMean(Tensor a, int[] dims, boolean mean) {
        Shape shape = a.shape();
        double[] data = a.toDoubleArray();
        if (dims.length == 0) {
            double total = 0.0;
            for (double v : data) {
                total += v;
            }
            return doubles(a.backend(), Shape.scalar(), new double[]{mean ? total / data.length : total});
        }
        Shape resultShape = BroadcastEngine.reduceShape(shape, dims, false);
        int rank = shape.rank();
        boolean[] reduced = new boolean[rank];
        long count = 1;
        for (int d : dims) {
            reduced[d] = true;
            count *= shape.dim(d);
        }
        long[] outStride = new long[rank];
        long stride = 1;
        for (int k = rank - 1; k >= 0; k--) {
            if (!reduced[k]) {
                outStride[k] = stride;
                stride *= shape.dim(k);
            }
        }
        double[] result = new double[CpuShapeOps.hostLength(resultShape.numElements())];
        int[] idx = new int[rank];
        long out = 0;
        for (int i = 0; i < data.length; i++) {
            result[(int) out] += data[i];
            for (int k = rank - 1; k >= 0; k--) {
                out += outStride[k];
                if (++idx[k] < shape.dim(k)) {
                    break;
                }
                out -= outStride[k] * shape.dim(k);
                idx[k] = 0;
            }
        }
        if (mean) {
            for (int i = 0; i < result.length; i++) {
                result[i] /= count;
            }
        }
        return doubles(a.backend(), resultShape, result);
    }

    // ------------------------------------------------------------------ products

    /**
     * {@code a @ b} with the rules and result shapes of {@code CpuLinalg.matmul}: rank-1 and rank-2 operands follow
     * the vector/matrix rules; above rank 2 the leading dims broadcast and the result is {@code batch + [M, N]}.
     *
     * @throws IllegalArgumentException when an operand has rank 0, the inner dims differ, or the batch dims do not
     *                                  broadcast
     */
    static Tensor matmul(Tensor a, Tensor b) {
        Shape shapeA = a.shape();
        Shape shapeB = b.shape();
        if (shapeA.rank() < 1 || shapeB.rank() < 1) {
            throw new IllegalArgumentException("Both tensors must have at least 1 dimension for matmul");
        }
        int k = shapeA.dim(shapeA.rank() - 1);
        int kB = shapeB.rank() > 1 ? shapeB.dim(shapeB.rank() - 2) : shapeB.dim(0);
        if (k != kB) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication: A has " + k
                    + " columns, B has " + kB + " rows");
        }
        int m = shapeA.rank() > 1 ? shapeA.dim(shapeA.rank() - 2) : 1;
        int n = shapeB.rank() > 1 ? shapeB.dim(shapeB.rank() - 1) : 1;
        int[] batchA = CpuLinalg.getBatchDims(shapeA);
        int[] batchB = CpuLinalg.getBatchDims(shapeB);
        int[] batch = CpuLinalg.broadcastBatchShapes(batchA, batchB);
        int[] stridesA = CpuLinalg.computeBatchStrides(batchA, batch);
        int[] stridesB = CpuLinalg.computeBatchStrides(batchB, batch);
        int batchSize = 1;
        for (int d : batch) {
            batchSize *= d;
        }
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        double[] result = new double[CpuShapeOps.hostLength((long) batchSize * m * n)];
        int[] batchIndex = new int[batch.length];
        for (int t = 0; t < batchSize; t++) {
            int rest = t;
            for (int i = batch.length - 1; i >= 0; i--) {
                batchIndex[i] = rest % batch[i];
                rest /= batch[i];
            }
            multiply(dataA, CpuLinalg.computeBatchOffset(batchIndex, stridesA) * m * k, dataB,
                    CpuLinalg.computeBatchOffset(batchIndex, stridesB) * k * n, result, t * m * n, m, n, k);
        }
        return doubles(a.backend(), matmulShape(shapeA, shapeB, batch, m, n), result);
    }

    /** The result shape of {@link #matmul}: vector/matrix rules up to rank 2, else {@code batch + [M, N]}. */
    private static Shape matmulShape(Shape shapeA, Shape shapeB, int[] batch, int m, int n) {
        if (shapeA.rank() <= 2 && shapeB.rank() <= 2) {
            if (shapeA.rank() == 1) {
                return Shape.vector(n);
            }
            return shapeB.rank() == 1 ? Shape.vector(m) : Shape.matrix(m, n);
        }
        int[] dims = new int[batch.length + 2];
        System.arraycopy(batch, 0, dims, 0, batch.length);
        dims[batch.length] = m;
        dims[batch.length + 1] = n;
        return Shape.of(dims);
    }

    /**
     * {@code a @ b} for two rank-3 operands with equal batch, as {@code CpuLinalg.bmm}.
     *
     * @throws IllegalArgumentException when an operand is not rank 3 or the batches or inner dimensions differ
     */
    static Tensor bmm(Tensor a, Tensor b) {
        Shape shapeA = a.shape();
        Shape shapeB = b.shape();
        if (shapeA.rank() != 3 || shapeB.rank() != 3) {
            throw new IllegalArgumentException("Both tensors must be 3D for bmm: got ranks " + shapeA.rank()
                    + " and " + shapeB.rank());
        }
        if (shapeA.dim(0) != shapeB.dim(0)) {
            throw new IllegalArgumentException("Batch dimensions must match: " + shapeA.dim(0) + " vs "
                    + shapeB.dim(0));
        }
        if (shapeA.dim(2) != shapeB.dim(1)) {
            throw new IllegalArgumentException("Inner dimensions must match: " + shapeA.dim(2) + " vs "
                    + shapeB.dim(1));
        }
        return matmul(a, b);
    }

    /** Row-major {@code [m, k] @ [k, n]} at the given element offsets into {@code c}, ascending {@code k}. */
    private static void multiply(double[] a, int offsetA, double[] b, int offsetB, double[] c, int offsetC,
                                 int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int kk = 0; kk < k; kk++) {
                    sum += a[offsetA + i * k + kk] * b[offsetB + kk * n + j];
                }
                c[offsetC + i * n + j] = sum;
            }
        }
    }

    // ------------------------------------------------------------------ shape

    /** The concatenation of {@code tensors} along {@code dim} into {@code resultShape}, validated by the caller. */
    static Tensor cat(List<Tensor> tensors, int dim, Shape resultShape) {
        double[] result = new double[CpuShapeOps.hostLength(resultShape.numElements())];
        int outer = 1;
        for (int k = 0; k < dim; k++) {
            outer *= resultShape.dim(k);
        }
        int inner = 1;
        for (int k = dim + 1; k < resultShape.rank(); k++) {
            inner *= resultShape.dim(k);
        }
        int resultRow = resultShape.dim(dim) * inner;
        int offset = 0;
        for (Tensor t : tensors) {
            double[] data = t.toDoubleArray();
            int row = t.shape().dim(dim) * inner;
            for (int o = 0; o < outer; o++) {
                System.arraycopy(data, o * row, result, o * resultRow + offset, row);
            }
            offset += row;
        }
        return doubles(tensors.get(0).backend(), resultShape, result);
    }
}
