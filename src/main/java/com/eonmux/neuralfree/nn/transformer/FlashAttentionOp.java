package com.eonmux.neuralfree.nn.transformer;

import com.eonmux.neuralfree.tensor.BroadcastEngine;
import com.eonmux.neuralfree.tensor.DType;
import com.eonmux.neuralfree.tensor.Shape;
import com.eonmux.neuralfree.tensor.Tensor;
import com.eonmux.neuralfree.tensor.TensorOps;

import java.util.Collections;
import java.util.Objects;

/**
 * Scaled dot-product attention over {@code [batch, heads, seq, headDim]} tensors as a composition of {@link TensorOps}
 * calls, so it runs on the train kernels of the tensors' backend and falls back to the CPU path where they are
 * unavailable. {@link MultiHeadAttention} uses the same composition.
 *
 * <p><b>Composition.</b> Scores are {@code bmm(Q, K^T)} over the merged {@code batch * heads} dim (K^T is a transposed
 * view), then {@code * scale}. An additive {@code mask} is added next, then causal masking sets every key position
 * {@code j > i} of query row {@code i} to {@code -inf} by {@code where}. Attention weights are
 * {@link TensorOps#maskedSoftmax} along the key dim, so a {@code -inf} entry gets exactly 0 and a row of only
 * {@code -inf} gets all 0. The output is {@code bmm(P, V)}. Every product is a sum of separately rounded terms in
 * ascending order on the CPU path, as the original host loops computed it; on the device the results are within the
 * GEMM and softmax tolerance classes of the {@linkplain com.eonmux.neuralfree.tensor.gpu numerical contract}.</p>
 *
 * <p><b>Autograd.</b> Every step is a recording {@link TensorOps} call or Tensor view, so in grad mode the
 * attention output is a recorded node output and the backward pass of the composition gives the gradients of the
 * query, key and value (and of a mask that requires grad): {@code dV = P^T dO}, {@code dP = dO V^T},
 * {@code dS = P * (dP - sum_j P dP)} (the masked softmax's backward, 0 at masked entries), {@code dQ = scale * dS K}
 * and {@code dK = scale * dS^T Q}. The causal mask and the masked {@code -inf} entries are constants.</p>
 *
 * <p><b>Masks.</b> A mask is a FLOAT32 additive attention bias on the scores' backend that broadcasts to
 * {@code [batch, heads, querySeq, keySeq]} (for example {@code [keySeq]}, {@code [querySeq, keySeq]} or
 * {@code [batch, 1, 1, keySeq]} for padding): {@code -inf} entries exclude a key, finite entries bias the scores (an
 * {@link ALiBiPE} bias is such a mask).</p>
 */
public final class FlashAttentionOp {

    private static final int FLASH_ATTENTION_THRESHOLD = 256;

    private FlashAttentionOp() {
    }

    /** Whether {@code seqLen} reaches {@link #getThreshold()}, the length from which attention counts as long. */
    public static boolean shouldUseFlashAttention(int seqLen) {
        return seqLen >= FLASH_ATTENTION_THRESHOLD;
    }

    public static int getThreshold() {
        return FLASH_ATTENTION_THRESHOLD;
    }

    /**
     * Attention of {@code query} {@code [B, H, Sq, D]} over {@code key} and {@code value} {@code [B, H, Skv, D]}.
     * The result's {@code output} is {@code [B, H, Sq, D]}, {@code scores} the masked scaled scores
     * {@code [B, H, Sq, Skv]} (the softmax input) and {@code l} the softmax normalizer {@code [B, H, Sq]}:
     * {@code sum_j exp(s_ij - max_j s_ij)} over the unmasked keys, 0 for a fully masked row.
     *
     * <p>{@code output} is computed by the recording composition of the class description, so in grad mode with an
     * input that requires grad it is the output of recorded nodes and a backward pass from it delivers the gradients
     * of {@code query}, {@code key}, {@code value} and a differentiable {@code mask}. {@code scores} and {@code l} are
     * inspection values: {@code scores} is a {@link Tensor#detach()} handle of the recorded scores and {@code l} is
     * computed from it, so neither records anything and both stay valid after the backward pass.</p>
     *
     * @throws IllegalArgumentException when the shapes do not form an attention, or the mask is not a FLOAT32 tensor on
     *                                  the same backend that broadcasts to the score shape
     */
    public static FlashAttentionResult apply(Tensor query, Tensor key, Tensor value,
                                             float scale, boolean causal, Tensor mask) {
        requireAttentionShapes(query, key, value);
        Tensor output = null;
        Tensor scores = null;
        try (Tensor recordedScores = maskedScores(query, key, scale, causal, mask);
             Tensor weights = TensorOps.maskedSoftmax(recordedScores, 3)) {
            output = weightedValues(weights, value);
            scores = recordedScores.detach();
            Tensor l = normalizer(scores);
            return new FlashAttentionResult(output, scores, l, query, key, value, scale, causal);
        } catch (RuntimeException e) {
            closeAll(e, output, scores);
            throw e;
        }
    }

    // ------------------------------------------------------------------ shared composition

    /**
     * The masked scaled scores {@code [B, H, Sq, Skv]} of {@code query} {@code [B, H, Sq, D]} against {@code key}
     * {@code [B, H, Skv, D]} (any views): {@code bmm(Q, K^T) * scale}, plus {@code mask} when given, with
     * {@code -inf} at {@code j > i} when {@code causal}. A new tensor the caller owns.
     */
    static Tensor maskedScores(Tensor query, Tensor key, float scale, boolean causal, Tensor mask) {
        Shape qShape = query.shape();
        int b = qShape.dim(0);
        int h = qShape.dim(1);
        int sq = qShape.dim(2);
        int d = qShape.dim(3);
        int skv = key.shape().dim(2);
        Shape scoreShape = Shape.of(b, h, sq, skv);
        if (mask != null) {
            requireMask(mask, query, scoreShape);
        }

        Tensor scores;
        try (Tensor q3 = query.reshape(Shape.of(b * h, sq, d));
             Tensor k3 = key.reshape(Shape.of(b * h, skv, d));
             Tensor kT = k3.transpose(0, 2, 1);
             Tensor raw = TensorOps.bmm(q3, kT);
             Tensor raw4 = raw.reshape(scoreShape)) {
            scores = TensorOps.mul(raw4, scale);
        }
        if (mask != null) {
            try {
                Tensor biased = TensorOps.add(scores, mask);
                scores.close();
                scores = biased;
            } catch (RuntimeException e) {
                scores.close();
                throw e;
            }
        }
        if (causal) {
            try {
                Tensor masked = causalMasked(scores);
                scores.close();
                scores = masked;
            } catch (RuntimeException e) {
                scores.close();
                throw e;
            }
        }
        return scores;
    }

    /** {@code bmm(P, V)}: attention weights {@code [B, H, Sq, Skv]} times values {@code [B, H, Skv, D]}. */
    static Tensor weightedValues(Tensor weights, Tensor value) {
        Shape wShape = weights.shape();
        int b = wShape.dim(0);
        int h = wShape.dim(1);
        int sq = wShape.dim(2);
        int skv = wShape.dim(3);
        int d = value.shape().dim(3);
        try (Tensor p3 = weights.reshape(Shape.of(b * h, sq, skv));
             Tensor v3 = value.reshape(Shape.of(b * h, skv, d));
             Tensor out3 = TensorOps.bmm(p3, v3)) {
            return out3.reshape(Shape.of(b, h, sq, d));
        }
    }

    /**
     * A contiguous tensor with {@code t}'s elements that the caller closes independently of {@code t}: a new copy when
     * {@code t} is not contiguous, otherwise a new reference to {@code t}'s storage.
     */
    static Tensor contiguousOwned(Tensor t) {
        Tensor c = t.contiguous();
        return c != t ? c : t.reshape(t.shape());
    }

    /** A new contiguous FLOAT32 tensor with {@code t}'s elements that shares no storage with {@code t}. */
    static Tensor copyOf(Tensor t) {
        return TensorOps.cat(Collections.singletonList(t), 0);
    }

    /** {@code scores} with {@code -inf} at every key position {@code j > i} of query row {@code i} (last two dims). */
    private static Tensor causalMasked(Tensor scores) {
        Shape shape = scores.shape();
        int sq = shape.dim(2);
        int skv = shape.dim(3);
        try (Tensor rows = Tensor.fromArray(scores.backend(), positions(sq));
             Tensor rowCol = rows.reshape(Shape.of(sq, 1));
             Tensor cols = Tensor.fromArray(scores.backend(), positions(skv));
             Tensor colRow = cols.reshape(Shape.of(1, skv));
             Tensor future = TensorOps.gt(colRow, rowCol);
             Tensor futureAll = future.expand(shape);
             Tensor negInf = Tensor.scalar(scores.backend(), Float.NEGATIVE_INFINITY);
             Tensor negInfAll = negInf.expand(shape)) {
            return TensorOps.where(futureAll, negInfAll, scores);
        }
    }

    /** {@code sum_j exp(s_ij - m_i)} along the last dim with {@code m_i} the row maximum, 0 for an all-masked row. */
    private static Tensor normalizer(Tensor scores) {
        try (Tensor rowMax = TensorOps.max(scores, 3);
             Tensor negInf = Tensor.scalar(scores.backend(), Float.NEGATIVE_INFINITY);
             Tensor emptyRow = TensorOps.eq(rowMax, negInf);
             Tensor zero = Tensor.scalar(scores.backend(), 0.0f);
             Tensor zeros = zero.expand(rowMax.shape());
             Tensor safeMax = TensorOps.where(emptyRow, zeros, rowMax);
             Tensor safeMaxCol = safeMax.unsqueeze(3);
             Tensor shifted = TensorOps.sub(scores, safeMaxCol);
             Tensor exps = TensorOps.exp(shifted)) {
            return TensorOps.sum(exps, 3);
        }
    }

    private static float[] positions(int n) {
        float[] p = new float[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    private static void requireAttentionShapes(Tensor query, Tensor key, Tensor value) {
        Objects.requireNonNull(query, "Query cannot be null");
        Objects.requireNonNull(key, "Key cannot be null");
        Objects.requireNonNull(value, "Value cannot be null");
        Shape q = query.shape();
        Shape k = key.shape();
        Shape v = value.shape();
        if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4) {
            throw new IllegalArgumentException("Query, key and value must be 4D [batch, heads, seq, head_dim], got "
                + q + ", " + k + " and " + v);
        }
        if (q.dim(0) != k.dim(0) || q.dim(1) != k.dim(1) || q.dim(3) != k.dim(3)) {
            throw new IllegalArgumentException("Key " + k + " does not match query " + q
                + " in batch, heads or head_dim");
        }
        if (!k.equals(v)) {
            throw new IllegalArgumentException("Value shape " + v + " must equal key shape " + k);
        }
        if (q.dim(2) == 0 || k.dim(2) == 0) {
            throw new IllegalArgumentException("Query and key sequence lengths must be positive, got " + q.dim(2)
                + " and " + k.dim(2));
        }
    }

    private static void requireMask(Tensor mask, Tensor like, Shape scoreShape) {
        if (mask.dtype() != DType.FLOAT32) {
            throw new IllegalArgumentException("Attention mask must be a FLOAT32 additive bias, got " + mask.dtype());
        }
        if (mask.backend() != like.backend()) {
            throw new IllegalArgumentException("Attention mask lives on a different backend than the query");
        }
        if (!BroadcastEngine.isBroadcastableTo(mask.shape(), scoreShape)) {
            throw new IllegalArgumentException("Attention mask " + mask.shape()
                + " does not broadcast to the score shape " + scoreShape);
        }
    }

    private static void closeAll(RuntimeException failure, Tensor... tensors) {
        for (Tensor t : tensors) {
            if (t != null) {
                try {
                    t.close();
                } catch (RuntimeException e) {
                    failure.addSuppressed(e);
                }
            }
        }
    }

    /**
     * The tensors of one {@link #apply} call. {@code output}, {@code scores} and {@code l} are new tensors that
     * {@link #close()} releases (closing is idempotent); {@code query}, {@code key} and {@code value} are the caller's
     * inputs and are never closed here. A recorded {@code output} is valid until a loss computed from it is
     * back-propagated (the backward pass releases it unless it is the root); {@code scores} and {@code l} are detached
     * and stay valid until {@link #close()}.
     */
    public static class FlashAttentionResult implements AutoCloseable {
        public final Tensor output;
        public final Tensor scores;
        public final Tensor l;
        public final Tensor query;
        public final Tensor key;
        public final Tensor value;
        public final float scale;
        public final boolean causal;

        public FlashAttentionResult(Tensor output, Tensor scores, Tensor l,
                                    Tensor query, Tensor key, Tensor value,
                                    float scale, boolean causal) {
            this.output = output;
            this.scores = scores;
            this.l = l;
            this.query = query;
            this.key = key;
            this.value = value;
            this.scale = scale;
            this.causal = causal;
        }

        /** Closes {@code output}, {@code scores} and {@code l}. */
        @Override
        public void close() {
            output.close();
            scores.close();
            l.close();
        }
    }
}
