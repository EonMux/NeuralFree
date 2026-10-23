package com.eonmux.neuralfree.distributed;

import com.eonmux.neuralfree.autograd.ComputationGraph;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.GpuOptimizerOps;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorValues;

import java.util.Objects;

/**
 * The tensor side of the {@link DistributedBackend} collectives: reading a tensor's values to the host, reducing
 * the per-rank values in rank order, and writing a result back into the caller's tensor. Collectives exchange host
 * floats and are not differentiable operations: nothing here records autograd nodes.
 */
final class CollectiveTensors {

    private CollectiveTensors() {
    }

    /**
     * The values of {@code tensor} in row-major order, as floats ({@link TensorValues#toFloats}): FLOAT16, BFLOAT16
     * and FLOAT64 values are converted to FLOAT32 for the exchange.
     *
     * @throws IllegalArgumentException when {@code tensor} is not a floating-point tensor
     */
    static float[] read(Tensor tensor) {
        Objects.requireNonNull(tensor, "Tensor cannot be null");
        if (!tensor.dtype().isFloatingPoint()) {
            throw new IllegalArgumentException("Collectives exchange floating-point tensors, got " + tensor.dtype());
        }
        return TensorValues.toFloats(tensor);
    }

    /**
     * Checks, before any rank exchanges values, that a collective can write its result into {@code target}.
     *
     * @throws IllegalArgumentException when {@code target} is not FLOAT32
     */
    static void requireWritable(Tensor target) {
        Objects.requireNonNull(target, "Tensor cannot be null");
        if (target.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("In-place collective results are written to FLOAT32 tensors, got "
                + target.dtype());
        }
    }

    /**
     * Writes {@code values} (row-major) into the elements {@code target} addresses, in place and under
     * {@link ComputationGraph#noGrad()}, as PyTorch's in-place collectives write their tensor.
     *
     * @throws IllegalArgumentException when {@code target} is not FLOAT32 (the dtype gradients and parameters use) or
     *                                  the number of values differs from its element count
     */
    static void writeBack(Tensor target, float[] values) {
        requireWritable(target);
        if (values.length != target.numElements()) {
            throw new IllegalArgumentException("Collective result has " + values.length + " values for a tensor of "
                + target.numElements() + " elements " + target.shape());
        }
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad()) {
            GpuOptimizerOps.storeFloats(target, values);
        }
    }

    /**
     * A new tensor holding {@code values} with the shape, backend and dtype of {@code like}; it does not require
     * grad. The caller owns it.
     */
    static Tensor shapedLike(Tensor like, float[] values) {
        try (ComputationGraph.NoGradScope noGrad = ComputationGraph.noGrad();
             Tensor flat = Tensor.fromArray(like.backend(), values)) {
            Tensor shaped = flat.reshape(like.shape());
            if (like.dtype() == DType.FLOAT32) {
                return shaped;
            }
            try (Tensor full = shaped) {
                return full.to(like.dtype());
            }
        }
    }

    /**
     * The element-wise reduction of {@code perRank} (one array per rank, all of one length) in rank order:
     * {@code SUM} and {@code PRODUCT} fold ranks {@code 0, 1, ...} left to right, {@code MEAN} is that sum divided
     * once by the number of ranks, {@code MAX} and {@code MIN} follow {@link Math#max(float, float)} and
     * {@link Math#min(float, float)} (NaN propagates). Every rank evaluating it on the same arrays gets bit-identical
     * results.
     *
     * @throws IllegalArgumentException when the arrays differ in length or there are none
     */
    static float[] reduce(float[][] perRank, ReduceOp op) {
        Objects.requireNonNull(op, "ReduceOp cannot be null");
        if (perRank.length == 0) {
            throw new IllegalArgumentException("A reduction needs at least one rank's values");
        }
        int length = perRank[0].length;
        for (int r = 1; r < perRank.length; r++) {
            if (perRank[r].length != length) {
                throw new IllegalArgumentException("Rank " + r + " contributed " + perRank[r].length
                    + " values, rank 0 contributed " + length + "; every rank must reduce a tensor of the same size");
            }
        }
        float[] result = perRank[0].clone();
        for (int r = 1; r < perRank.length; r++) {
            float[] next = perRank[r];
            for (int i = 0; i < length; i++) {
                result[i] = combine(result[i], next[i], op);
            }
        }
        if (op == ReduceOp.MEAN) {
            float ranks = perRank.length;
            for (int i = 0; i < length; i++) {
                result[i] /= ranks;
            }
        }
        return result;
    }

    private static float combine(float a, float b, ReduceOp op) {
        switch (op) {
            case SUM:
            case MEAN:
                return a + b;
            case PRODUCT:
                return a * b;
            case MAX:
                return Math.max(a, b);
            case MIN:
                return Math.min(a, b);
            default:
                throw new IllegalArgumentException("Unknown ReduceOp: " + op);
        }
    }
}
