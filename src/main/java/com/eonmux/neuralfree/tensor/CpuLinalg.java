package com.eonmux.neuralfree.tensor;

import java.util.Arrays;

/**
 * CPU reference of {@link TensorOps#matmul} and {@link TensorOps#bmm}: each output is the sum of its K separately
 * rounded products, added in ascending k from {@code 0.0f}. Results never require grad, whatever the operands' flags.
 */
final class CpuLinalg {

    private CpuLinalg() {
    }

    /**
     * {@code a @ b}: rank-1 and rank-2 operands follow the vector/matrix rules (a rank-1 {@code a} of length K times
     * {@code [K, N]} gives {@code [N]}, {@code [M, K]} times a rank-1 {@code b} gives {@code [M]}); above rank 2 the
     * leading dims broadcast and the result is {@code batch + [M, N]}.
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

        int kA = shapeA.dim(shapeA.rank() - 1);
        int kB = shapeB.rank() > 1 ? shapeB.dim(shapeB.rank() - 2) : shapeB.dim(0);

        if (kA != kB) {
            throw new IllegalArgumentException("Matrix dimensions incompatible for multiplication: " +
                "A has " + kA + " columns, B has " + kB + " rows");
        }

        int m = shapeA.rank() > 1 ? shapeA.dim(shapeA.rank() - 2) : 1;
        int n = shapeB.rank() > 1 ? shapeB.dim(shapeB.rank() - 1) : 1;
        int k = kA;

        if (shapeA.rank() <= 2 && shapeB.rank() <= 2) {
            return matmul2D(a, b, m, n, k);
        }

        int[] batchDimsA = getBatchDims(shapeA);
        int[] batchDimsB = getBatchDims(shapeB);
        int[] broadcastBatchDims = broadcastBatchShapes(batchDimsA, batchDimsB);

        float[] dataA = a.toFloatArray();
        float[] dataB = b.toFloatArray();

        int batchSize = 1;
        for (int dim : broadcastBatchDims) {
            batchSize *= dim;
        }

        int[] resultDims = new int[broadcastBatchDims.length + 2];
        System.arraycopy(broadcastBatchDims, 0, resultDims, 0, broadcastBatchDims.length);
        resultDims[resultDims.length - 2] = m;
        resultDims[resultDims.length - 1] = n;

        float[] result = new float[CpuShapeOps.hostLength((long) batchSize * m * n)];

        int[] batchIndices = new int[broadcastBatchDims.length];
        int[] batchStridesA = computeBatchStrides(batchDimsA, broadcastBatchDims);
        int[] batchStridesB = computeBatchStrides(batchDimsB, broadcastBatchDims);

        int matrixSizeA = shapeA.rank() > 1 ? shapeA.dim(shapeA.rank() - 2) * k : k;
        int matrixSizeB = shapeB.rank() > 1 ? kB * n : n;
        int matrixSizeResult = m * n;

        for (int batch = 0; batch < batchSize; batch++) {
            int tempBatch = batch;
            for (int i = broadcastBatchDims.length - 1; i >= 0; i--) {
                batchIndices[i] = tempBatch % broadcastBatchDims[i];
                tempBatch /= broadcastBatchDims[i];
            }

            int offsetA = computeBatchOffset(batchIndices, batchStridesA) * matrixSizeA;
            int offsetB = computeBatchOffset(batchIndices, batchStridesB) * matrixSizeB;
            int offsetResult = batch * matrixSizeResult;

            if (shapeA.rank() == 1) {
                for (int j = 0; j < n; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < k; kk++) {
                        sum += dataA[kk] * dataB[offsetB + kk * n + j];
                    }
                    result[offsetResult + j] = sum;
                }
            } else if (shapeB.rank() == 1) {
                for (int i = 0; i < m; i++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < k; kk++) {
                        sum += dataA[offsetA + i * k + kk] * dataB[kk];
                    }
                    result[offsetResult + i] = sum;
                }
            } else {
                multiply(dataA, offsetA, dataB, offsetB, result, offsetResult, m, n, k);
            }
        }

        return CpuShapeOps.floats(a.backend(), Shape.of(resultDims), result, false);
    }

    /**
     * {@code a @ b} for two rank-3 operands with equal batch, as a new {@code [batch, M, N]} tensor.
     *
     * @throws IllegalArgumentException when an operand is not rank 3 or the batches or inner dimensions differ
     */
    static Tensor bmm(Tensor a, Tensor b) {
        Shape shapeA = a.shape();
        Shape shapeB = b.shape();

        if (shapeA.rank() != 3 || shapeB.rank() != 3) {
            throw new IllegalArgumentException("Both tensors must be 3D for bmm: got ranks " +
                shapeA.rank() + " and " + shapeB.rank());
        }

        int batch = shapeA.dim(0);
        if (batch != shapeB.dim(0)) {
            throw new IllegalArgumentException("Batch dimensions must match: " + batch + " vs " + shapeB.dim(0));
        }

        int m = shapeA.dim(1);
        int k = shapeA.dim(2);
        int k2 = shapeB.dim(1);
        int n = shapeB.dim(2);

        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must match: " + k + " vs " + k2);
        }

        float[] dataA = a.toFloatArray();
        float[] dataB = b.toFloatArray();
        float[] result = new float[CpuShapeOps.hostLength((long) batch * m * n)];

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            multiply(dataA, bIdx * m * k, dataB, bIdx * k * n, result, bIdx * m * n, m, n, k);
        }

        return CpuShapeOps.floats(a.backend(), Shape.of(batch, m, n), result, false);
    }

    private static Tensor matmul2D(Tensor a, Tensor b, int m, int n, int k) {
        float[] dataA = a.toFloatArray();
        float[] dataB = b.toFloatArray();
        float[] result = new float[CpuShapeOps.hostLength((long) m * n)];

        if (a.shape().rank() == 1) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; kk++) {
                    sum += dataA[kk] * dataB[kk * n + j];
                }
                result[j] = sum;
            }
            return CpuShapeOps.floats(a.backend(), Shape.vector(n), result, false);
        } else if (b.shape().rank() == 1) {
            for (int i = 0; i < m; i++) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; kk++) {
                    sum += dataA[i * k + kk] * dataB[kk];
                }
                result[i] = sum;
            }
            return CpuShapeOps.floats(a.backend(), Shape.vector(m), result, false);
        }
        multiply(dataA, 0, dataB, 0, result, 0, m, n, k);
        return CpuShapeOps.floats(a.backend(), Shape.matrix(m, n), result, false);
    }

    /** Row-major {@code [m, k] @ [k, n]} at the given element offsets into {@code c}. */
    private static void multiply(float[] a, int offsetA, float[] b, int offsetB, float[] c, int offsetC,
                                 int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int kk = 0; kk < k; kk++) {
                    sum += a[offsetA + i * k + kk] * b[offsetB + kk * n + j];
                }
                c[offsetC + i * n + j] = sum;
            }
        }
    }

    private static int[] getBatchDims(Shape shape) {
        if (shape.rank() <= 2) {
            return new int[0];
        }
        int[] batchDims = new int[shape.rank() - 2];
        for (int i = 0; i < batchDims.length; i++) {
            batchDims[i] = shape.dim(i);
        }
        return batchDims;
    }

    private static int[] broadcastBatchShapes(int[] batchDimsA, int[] batchDimsB) {
        int maxLen = Math.max(batchDimsA.length, batchDimsB.length);
        int[] result = new int[maxLen];

        for (int i = 0; i < maxLen; i++) {
            int dimA = i < (maxLen - batchDimsA.length) ? 1 : batchDimsA[i - (maxLen - batchDimsA.length)];
            int dimB = i < (maxLen - batchDimsB.length) ? 1 : batchDimsB[i - (maxLen - batchDimsB.length)];

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException("Batch dimensions cannot be broadcast: " +
                    Arrays.toString(batchDimsA) + " vs " + Arrays.toString(batchDimsB));
            }
            result[i] = Math.max(dimA, dimB);
        }
        return result;
    }

    /** Matrix-count strides of an operand's batch dims over the broadcast batch dims (0 on broadcast dims). */
    private static int[] computeBatchStrides(int[] batchDims, int[] broadcastDims) {
        int[] strides = new int[broadcastDims.length];
        int offset = broadcastDims.length - batchDims.length;

        for (int i = 0; i < broadcastDims.length; i++) {
            if (i < offset) {
                strides[i] = 0;
            } else {
                int batchIdx = i - offset;
                if (batchDims[batchIdx] == 1) {
                    strides[i] = 0;
                } else {
                    strides[i] = 1;
                    for (int j = batchIdx + 1; j < batchDims.length; j++) {
                        strides[i] *= batchDims[j];
                    }
                }
            }
        }
        return strides;
    }

    private static int computeBatchOffset(int[] indices, int[] strides) {
        int offset = 0;
        for (int i = strides.length - 1; i >= 0; i--) {
            if (strides[i] > 0) {
                offset += indices[i] * strides[i];
            }
        }
        return offset;
    }
}
