package com.eonmux.neuralfree.tensor;

import java.util.Arrays;

/**
 * CPU reference of {@link TensorOps}' reductions: sum, mean, max, min, the arg-reductions, variance, softmax and
 * log-softmax (plain and masked), their backwards, max backward and {@code sumToShape}. Accumulation is sequential
 * Java float arithmetic in ascending index order. Results never require grad, whatever the inputs' flags.
 */
final class CpuReductions {

    private enum ReductionOp {
        SUM, MEAN, MAX, MIN
    }

    private CpuReductions() {
    }

    // ------------------------------------------------------------------ sum, mean, max, min

    /** Sum over {@code dims}; no dims sums every element into a scalar. */
    static Tensor sum(Tensor a, int... dims) {
        return sumOrMean(a, dims, ReductionOp.SUM);
    }

    /** Mean over {@code dims} (the sum, then one division); no dims averages every element into a scalar. */
    static Tensor mean(Tensor a, int... dims) {
        return sumOrMean(a, dims, ReductionOp.MEAN);
    }

    static Tensor max(Tensor a, int dim) {
        return extreme(a, dim, ReductionOp.MAX);
    }

    static Tensor min(Tensor a, int dim) {
        return extreme(a, dim, ReductionOp.MIN);
    }

    private static Tensor sumOrMean(Tensor a, int[] dims, ReductionOp op) {
        Shape shape = a.shape();
        int[] normalizedDims = normalizeDims(dims, shape.rank());
        float[] data = a.toFloatArray();
        if (dims.length == 0) {
            float total = 0.0f;
            for (float v : data) {
                total += v;
            }
            float value = op == ReductionOp.MEAN ? total / data.length : total;
            return CpuShapeOps.floats(a.backend(), Shape.scalar(), new float[]{value}, false);
        }
        Shape resultShape = BroadcastEngine.reduceShape(shape, normalizedDims, false);
        float[] result = computeReduction(data, shape, normalizedDims, op);
        return CpuShapeOps.floats(a.backend(), resultShape, result, false);
    }

    private static Tensor extreme(Tensor a, int dim, ReductionOp op) {
        Shape shape = a.shape();
        int[] normalizedDims = BroadcastEngine.normalizeAxes(new int[]{dim}, shape.rank());
        Shape resultShape = BroadcastEngine.reduceShape(shape, normalizedDims, false);
        float[] result = computeReduction(a.toFloatArray(), shape, normalizedDims, op);
        return CpuShapeOps.floats(a.backend(), resultShape, result, false);
    }

    /**
     * Population variance over {@code dims} (no Bessel correction): the mean of the squared deviations from the mean
     * over the same dims. The mean keeps the reduced dims as size 1, so it broadcasts against {@code a} for any
     * choice of dims; no dims reduces every element into a scalar.
     */
    static Tensor var(Tensor a, int... dims) {
        Shape shape = a.shape();
        int[] reduced = dims.length == 0 ? allDims(shape.rank()) : normalizeDims(dims, shape.rank());
        Shape keptShape = BroadcastEngine.reduceShape(shape, reduced, true);
        try (Tensor meanTensor = mean(a, dims);
             Tensor meanKept = meanTensor.reshape(keptShape);
             Tensor centered = CpuElementwise.sub(a, meanKept);
             Tensor squared = CpuElementwise.mul(centered, centered)) {
            return mean(squared, dims);
        }
    }

    // ------------------------------------------------------------------ arg-reductions

    /**
     * INT32 index of the maximum along {@code dim}: starting from {@code (-inf, 0)}, an index replaces the best only
     * when its value is strictly greater, so NaN is never selected, ties keep the first index, and a row with no
     * such value gives 0.
     */
    static Tensor argmax(Tensor a, int dim) {
        return argReduce(a, dim, true);
    }

    /** {@link #argmax} with the comparison reversed, starting from {@code +inf}. */
    static Tensor argmin(Tensor a, int dim) {
        return argReduce(a, dim, false);
    }

    private static Tensor argReduce(Tensor a, int dim, boolean isMax) {
        Shape shape = a.shape();
        int d = CpuShapeOps.normalizeDim(dim, shape.rank());
        Shape resultShape = BroadcastEngine.reduceShape(shape, new int[]{d}, false);
        float[] data = a.toFloatArray();
        Rows rows = new Rows(shape, d);
        int[] result = new int[rows.count()];
        for (int r = 0; r < result.length; r++) {
            int bestIdx = 0;
            float bestVal = isMax ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
            for (int k = 0; k < rows.length; k++) {
                float val = data[rows.index(r, k)];
                if ((isMax && val > bestVal) || (!isMax && val < bestVal)) {
                    bestVal = val;
                    bestIdx = k;
                }
            }
            result[r] = bestIdx;
        }
        return CpuShapeOps.ints(a.backend(), resultShape, result);
    }

    // ------------------------------------------------------------------ softmax family

    /** Softmax along {@code dim}: max-subtract, exp, sum, divide. */
    static Tensor softmax(Tensor a, int dim) {
        return softmaxRows(a, dim, false, false);
    }

    /** Log-softmax along {@code dim}: {@code x - (max + log(sum))}. */
    static Tensor logSoftmax(Tensor a, int dim) {
        return softmaxRows(a, dim, true, false);
    }

    /**
     * Softmax along {@code dim} with {@code -inf} entries masked out ({@code MultiHeadAttention}'s attention
     * softmax): they are left out of the sum and get exactly 0, so a row of only {@code -inf} gives all 0.
     */
    static Tensor maskedSoftmax(Tensor a, int dim) {
        return softmaxRows(a, dim, false, true);
    }

    private static Tensor softmaxRows(Tensor a, int dim, boolean log, boolean masked) {
        Shape shape = a.shape();
        Rows rows = new Rows(shape, CpuShapeOps.normalizeDim(dim, shape.rank()));
        float[] data = a.toFloatArray();
        float[] result = new float[data.length];
        for (int r = 0; r < rows.count(); r++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int k = 0; k < rows.length; k++) {
                maxVal = Math.max(maxVal, data[rows.index(r, k)]);
            }
            float sum = 0.0f;
            for (int k = 0; k < rows.length; k++) {
                float x = data[rows.index(r, k)];
                if (!(masked && x == Float.NEGATIVE_INFINITY)) {
                    sum += (float) Math.exp(x - maxVal);
                }
            }
            float logSum = log ? maxVal + (float) Math.log(sum) : 0.0f;
            for (int k = 0; k < rows.length; k++) {
                int idx = rows.index(r, k);
                float x = data[idx];
                if (log) {
                    result[idx] = x - logSum;
                } else if (masked && x == Float.NEGATIVE_INFINITY) {
                    result[idx] = 0.0f;
                } else {
                    result[idx] = (float) Math.exp(x - maxVal) / sum;
                }
            }
        }
        return CpuShapeOps.floats(a.backend(), shape, result, false);
    }

    /**
     * Gradient of softmax ({@code log == false}, {@code GradFunctions.SoftmaxGrad}: {@code gx = y * (gy - sum(y*gy))})
     * or log-softmax ({@code log == true}, {@code LogSoftmaxGrad}: {@code gx = gy - exp(y) * sum(gy)}) along
     * {@code dim} (negative dims count from the end).
     *
     * @throws IllegalArgumentException when the backends or shapes differ
     */
    static Tensor softmaxBackward(Tensor y, Tensor gy, int dim, boolean log) {
        CpuShapeOps.requireSameBackend(y, gy);
        if (!y.shape().equals(gy.shape())) {
            throw new IllegalArgumentException("softmax output " + y.shape() + " and gradient " + gy.shape()
                    + " must have the same shape");
        }
        Rows rows = new Rows(y.shape(), CpuShapeOps.normalizeDim(dim, y.shape().rank()));
        float[] outputData = y.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] result = new float[outputData.length];
        for (int r = 0; r < rows.count(); r++) {
            float acc = 0.0f;
            for (int k = 0; k < rows.length; k++) {
                int idx = rows.index(r, k);
                acc += log ? gradData[idx] : outputData[idx] * gradData[idx];
            }
            for (int k = 0; k < rows.length; k++) {
                int idx = rows.index(r, k);
                result[idx] = log ? gradData[idx] - (float) Math.exp(outputData[idx]) * acc
                        : outputData[idx] * (gradData[idx] - acc);
            }
        }
        return CpuShapeOps.floats(gy.backend(), y.shape(), result, false);
    }

    /**
     * Gradient of {@link #max} ({@code GradFunctions.MaxGrad}): zeros of {@code x}'s shape, except {@code gy[row]} at
     * the first index along {@code dim} whose value {@code ==} {@code maxOut[row]} (a NaN max routes nowhere).
     * {@code maxOut} and {@code gy} hold one value per reduced row in row-major order of the kept dims.
     *
     * @throws IllegalArgumentException when the backends differ or {@code maxOut} or {@code gy} has the wrong
     *                                  element count
     */
    static Tensor maxBackward(Tensor x, Tensor maxOut, Tensor gy, int dim) {
        CpuShapeOps.requireSameBackend(x, maxOut);
        CpuShapeOps.requireSameBackend(x, gy);
        Rows rows = new Rows(x.shape(), CpuShapeOps.normalizeDim(dim, x.shape().rank()));
        requireRowValues("max output", maxOut, rows.count());
        requireRowValues("gradient", gy, rows.count());
        float[] inputData = x.toFloatArray();
        float[] outputData = maxOut.toFloatArray();
        float[] gradData = gy.toFloatArray();
        float[] result = new float[inputData.length];
        for (int r = 0; r < rows.count(); r++) {
            for (int k = 0; k < rows.length; k++) {
                int idx = rows.index(r, k);
                if (inputData[idx] == outputData[r]) {
                    result[idx] = gradData[r];
                    break;
                }
            }
        }
        return CpuShapeOps.floats(x.backend(), x.shape(), result, false);
    }

    /**
     * {@code t} summed down to {@code target}, the shape of an operand that was broadcast to {@code t}'s shape (the
     * gradient of a broadcast operand): the prepended dims and the dims where {@code target} has 1 and
     * {@code t} more are summed. When no dim is summed, {@code target} must have {@code t}'s element count and the
     * result is a copy of {@code t}'s elements. Always a new tensor.
     *
     * @throws IllegalArgumentException when the shapes are not broadcast-compatible, or {@code target} neither
     *                                  broadcasts to {@code t}'s shape nor has its element count
     */
    static Tensor sumToShape(Tensor t, Shape target) {
        Shape shape = t.shape();
        if (!shape.equals(target) && BroadcastEngine.broadcastShape(target, shape).equals(shape)) {
            int[] axes = broadcastAxes(target, shape);
            if (axes.length > 0) {
                try (Tensor summed = sum(t, axes)) {
                    return summed.reshape(target);
                }
            }
        }
        if (shape.numElements() != target.numElements()) {
            throw new IllegalArgumentException("target shape " + target + " neither broadcasts to " + shape
                    + " nor has its " + shape.numElements() + " elements");
        }
        return CpuShapeOps.floats(t.backend(), target, t.toFloatArray(), false);
    }

    /**
     * Dims of {@code larger} that a broadcast of {@code smaller} expanded: the prepended dims and the dims where
     * {@code smaller} has 1 and {@code larger} more.
     */
    private static int[] broadcastAxes(Shape smaller, Shape larger) {
        int rankDiff = larger.rank() - smaller.rank();
        int[] axes = new int[larger.rank()];
        int count = 0;
        for (int i = 0; i < rankDiff; i++) {
            axes[count++] = i;
        }
        for (int i = 0; i < smaller.rank(); i++) {
            if (smaller.dim(i) == 1 && larger.dim(rankDiff + i) > 1) {
                axes[count++] = rankDiff + i;
            }
        }
        return Arrays.copyOf(axes, count);
    }

    // ------------------------------------------------------------------ helpers

    /**
     * {@code dims} normalized to {@code [0, rank)}.
     *
     * @throws IndexOutOfBoundsException when a dim is outside the rank
     * @throws IllegalArgumentException  when a dim repeats
     */
    private static int[] normalizeDims(int[] dims, int rank) {
        int[] normalized = BroadcastEngine.normalizeAxes(dims, rank);
        boolean[] seen = new boolean[rank];
        for (int d : normalized) {
            if (seen[d]) {
                throw new IllegalArgumentException("dim " + d + " is reduced twice in " + Arrays.toString(dims));
            }
            seen[d] = true;
        }
        return normalized;
    }

    private static int[] allDims(int rank) {
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = i;
        }
        return dims;
    }

    private static void requireRowValues(String what, Tensor t, int expected) {
        if (t.numElements() != expected) {
            throw new IllegalArgumentException(what + " must hold one value per reduced row (" + expected
                    + "), got shape " + t.shape());
        }
    }

    private static float[] computeReduction(float[] data, Shape shape, int[] dims, ReductionOp op) {
        int rank = shape.rank();

        boolean[] isReduced = new boolean[rank];
        for (int d : dims) {
            isReduced[d] = true;
        }

        int resultRank = 0;
        for (int i = 0; i < rank; i++) {
            if (!isReduced[i]) {
                resultRank++;
            }
        }

        int[] resultDims = new int[resultRank];
        int resultIdx = 0;
        for (int i = 0; i < rank; i++) {
            if (!isReduced[i]) {
                resultDims[resultIdx++] = shape.dim(i);
            }
        }

        int resultSize = 1;
        for (int d : resultDims) {
            resultSize *= d;
        }
        float[] result = initResultArray(resultSize, op);

        int[] indices = new int[rank];
        for (long i = 0; i < shape.numElements(); i++) {
            long linearIdx = i;
            for (int j = rank - 1; j >= 0; j--) {
                indices[j] = (int) (linearIdx % shape.dim(j));
                linearIdx /= shape.dim(j);
            }

            int resultLinearIdx = 0;
            int resultStride = 1;
            for (int j = resultRank - 1; j >= 0; j--) {
                int srcDim = -1;
                int count = 0;
                for (int k = 0; k < rank; k++) {
                    if (!isReduced[k]) {
                        if (count == j) {
                            srcDim = k;
                            break;
                        }
                        count++;
                    }
                }
                if (srcDim >= 0) {
                    resultLinearIdx += indices[srcDim] * resultStride;
                    resultStride *= resultDims[j];
                }
            }

            applyReduction(result, resultLinearIdx, data[(int) i], op);
        }

        if (op == ReductionOp.MEAN) {
            long reductionSize = 1;
            for (int d : dims) {
                reductionSize *= shape.dim(d);
            }
            for (int i = 0; i < result.length; i++) {
                result[i] /= reductionSize;
            }
        }

        return result;
    }

    private static float[] initResultArray(int size, ReductionOp op) {
        float[] result = new float[size];
        if (op == ReductionOp.MAX) {
            Arrays.fill(result, Float.NEGATIVE_INFINITY);
        } else if (op == ReductionOp.MIN) {
            Arrays.fill(result, Float.POSITIVE_INFINITY);
        }
        return result;
    }

    private static void applyReduction(float[] result, int idx, float value, ReductionOp op) {
        switch (op) {
            case SUM:
            case MEAN:
                result[idx] += value;
                break;
            case MAX:
                result[idx] = Math.max(result[idx], value);
                break;
            case MIN:
                result[idx] = Math.min(result[idx], value);
                break;
            default:
                throw new IllegalStateException("unknown reduction " + op);
        }
    }

    /**
     * The rows of one dim: row {@code r} (row-major over the other dims, {@code r = outer * inner + in}) holds the
     * {@code length} elements along the dim at stride {@code inner}.
     */
    private static final class Rows {
        final int outer;
        final int length;
        final int inner;

        Rows(Shape shape, int dim) {
            this.outer = CpuShapeOps.product(shape, 0, dim);
            this.length = shape.dim(dim);
            this.inner = CpuShapeOps.product(shape, dim + 1, shape.rank());
        }

        int count() {
            return CpuShapeOps.hostLength((long) outer * inner);
        }

        /** Row-major element index of position {@code k} along the dim in row {@code r}. */
        int index(int r, int k) {
            int o = r / inner;
            int in = r - o * inner;
            return (o * length + k) * inner + in;
        }
    }
}
