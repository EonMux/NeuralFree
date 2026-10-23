package com.eonmux.neuralfree.nn.multimodal;

import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Objects;

/**
 * Multi-head scaled dot-product attention of the multimodal modules, composed from {@link TensorOps}: heads are split
 * by reshape and permute views, scores come from a batched matmul against a transposed key view, the softmax runs
 * along the key axis, the value product is a second batched matmul, and the heads are merged by a permute and one
 * contiguous copy. On a backend with the train kernels every step stays on the device; elsewhere each call takes its
 * CPU path. Every intermediate is released before returning.
 *
 * <p>Per (batch, head, query) row the result is {@code score_j = (sum_d q_d * k_jd) * scale} with
 * {@code scale = 1 / (float) Math.sqrt(headDim)}, an optional key mask that sets {@code score_j = -inf} where
 * {@code mask[b, j] == 0}, {@code p = softmax(score)} and {@code out_d = sum_j p_j * v_jd}. The sums are in the
 * GEMM tolerance class and the softmax in the transcendental class of the
 * {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 */
final class ScaledDotProductAttention {

    private ScaledDotProductAttention() {
    }

    /**
     * Attention of {@code [batch, queryLen, hidden]} queries over {@code [batch, keyLen, hidden]} keys and values.
     *
     * @param numHeads    number of heads; {@code hidden} must be divisible by it
     * @param keyMask     null, or a FLOAT32 tensor of {@code batch * keyLen} values in row-major
     *                    {@code [batch, keyLen]} order whose zero entries mask the key out
     * @param maskedRows  whether masked entries leave the softmax sum and become exactly 0, so that a fully masked row
     *                    is all 0 ({@link TensorOps#maskedSoftmax}); otherwise the plain {@link TensorOps#softmax}
     * @return a new contiguous {@code [batch, queryLen, hidden]} tensor
     * @throws IllegalArgumentException when a shape does not match or {@code hidden} is not divisible by the heads
     */
    static Tensor attend(Tensor q, Tensor k, Tensor v, int numHeads, Tensor keyMask, boolean maskedRows) {
        Objects.requireNonNull(q, "query cannot be null");
        Objects.requireNonNull(k, "key cannot be null");
        Objects.requireNonNull(v, "value cannot be null");
        requireRank3(q, "query");
        requireRank3(k, "key");
        requireRank3(v, "value");
        int batch = q.shape().dim(0);
        int queryLen = q.shape().dim(1);
        int hidden = q.shape().dim(2);
        int keyLen = k.shape().dim(1);
        if (k.shape().dim(0) != batch || v.shape().dim(0) != batch) {
            throw new IllegalArgumentException("query, key and value batch sizes differ: " + q.shape() + ", "
                    + k.shape() + ", " + v.shape());
        }
        if (v.shape().dim(1) != keyLen) {
            throw new IllegalArgumentException("key length " + keyLen + " and value length " + v.shape().dim(1)
                    + " differ");
        }
        if (k.shape().dim(2) != hidden || v.shape().dim(2) != hidden) {
            throw new IllegalArgumentException("query, key and value feature sizes differ: " + q.shape() + ", "
                    + k.shape() + ", " + v.shape());
        }
        requireDivisible(hidden, numHeads);
        int headDim = hidden / numHeads;
        float scale = 1.0f / (float) Math.sqrt(headDim);

        try (Tensor qh = heads(q, numHeads, headDim, false);
             Tensor kt = heads(k, numHeads, headDim, true);
             Tensor vh = heads(v, numHeads, headDim, false);
             Tensor raw = TensorOps.matmul(qh, kt);
             Tensor scores = TensorOps.mul(raw, scale);
             Tensor masked = keyMask == null ? null : withKeyMask(scores, keyMask, batch, keyLen);
             Tensor probs = softmax(masked != null ? masked : scores, maskedRows);
             Tensor context = TensorOps.matmul(probs, vh);
             Tensor merged = context.permute(0, 2, 1, 3)) {
            return merged.reshape(Shape.of(batch, queryLen, hidden));
        }
    }

    /**
     * Throws unless {@code hidden} splits evenly into {@code numHeads} heads.
     *
     * @throws IllegalArgumentException when it does not
     */
    static void requireDivisible(int hidden, int numHeads) {
        if (numHeads <= 0 || hidden % numHeads != 0) {
            throw new IllegalArgumentException("hiddenDim (" + hidden + ") must be divisible by numHeads ("
                    + numHeads + ")");
        }
    }

    /**
     * {@code [b, len, heads * headDim]} as a {@code [b, heads, len, headDim]} view, or as
     * {@code [b, heads, headDim, len]} when {@code transposed}.
     */
    private static Tensor heads(Tensor t, int numHeads, int headDim, boolean transposed) {
        Shape s = t.shape();
        try (Tensor split = t.reshape(Shape.of(s.dim(0), s.dim(1), numHeads, headDim))) {
            return transposed ? split.permute(0, 2, 3, 1) : split.permute(0, 2, 1, 3);
        }
    }

    private static Tensor softmax(Tensor scores, boolean maskedRows) {
        return maskedRows ? TensorOps.maskedSoftmax(scores, 3) : TensorOps.softmax(scores, 3);
    }

    /**
     * {@code scores + bias}, where the {@code [batch, 1, 1, keyLen]} bias is {@code -inf} at masked keys and
     * {@code +0} elsewhere, so masked scores become {@code -inf} and the others keep their value.
     */
    private static Tensor withKeyMask(Tensor scores, Tensor keyMask, int batch, int keyLen) {
        if (keyMask.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("attention mask must be FLOAT32, got " + keyMask.dtype());
        }
        if (keyMask.numElements() != (long) batch * keyLen) {
            throw new IllegalArgumentException("attention mask of shape " + keyMask.shape() + " does not hold "
                    + batch + " x " + keyLen + " key flags");
        }
        Shape biasShape = Shape.of(batch, 1, 1, keyLen);
        try (Tensor mask = keyMask.reshape(biasShape);
             Tensor zero = Tensor.scalar(scores.backend(), 0.0f);
             Tensor dropped = TensorOps.eq(mask, zero);
             Tensor zeros = Tensor.zeros(scores.backend(), biasShape);
             Tensor negInf = TensorOps.add(zeros, Float.NEGATIVE_INFINITY);
             Tensor bias = TensorOps.where(dropped, negInf, zeros)) {
            return TensorOps.add(scores, bias);
        }
    }

    private static void requireRank3(Tensor t, String what) {
        if (t.shape().rank() != 3) {
            throw new IllegalArgumentException(what + " must be [batch, length, features], got " + t.shape());
        }
    }
}
