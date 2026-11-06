package com.eonmux.neuralfree.gguf.spec;

import com.eonmux.neuralfree.gguf.ops.rope.RopeMath;

import java.util.Arrays;
import java.util.Objects;

/**
 * Fourier position embedding (FoPE) parameters of one layer, resolved from hparams and the tensor catalog: the
 * learned per-KV-head coefficient tensors that replace the analytic cos/sin of each rotation pair, and the
 * frequencies they are projected from.
 *
 * <p>The cos and sin rows of head {@code h} at position {@code p} are
 * {@code sum_d cos/sin(p * invFreq[d]) * coef[row(h), d, i]} for pair {@code i < invFreq.length}, and 1 for the
 * remaining pairs, where {@code row(h)} is {@code h} for a key head and the query head's KV group otherwise
 * ({@code FourierRotaryEmbedding._compute_cos_sin_cache} in vLLM, rotary_embedding/fope.py). The rotation itself is
 * the NEOX pair layout. Immutable.
 */
public final class FopeSpec {

    private final int numKeyValueHeads;
    private final boolean separateHeads;
    private final float[] invFreq;
    private final TensorRef cosCoef;
    private final TensorRef sinCoef;

    private FopeSpec(int numKeyValueHeads, boolean separateHeads, float[] invFreq, TensorRef cosCoef, TensorRef sinCoef) {
        this.numKeyValueHeads = numKeyValueHeads;
        this.separateHeads = separateHeads;
        this.invFreq = invFreq.clone();
        this.cosCoef = cosCoef;
        this.sinCoef = sinCoef;
    }

    /**
     * FoPE parameters of a layer whose coefficient tensors hold {@code numKeyValueHeads} rows of
     * {@code invFreq.length x invFreq.length}.
     *
     * @param numKeyValueHeads number of coefficient rows ({@code num_key_value_heads})
     * @param separateHeads    true when each KV head has its own row ({@code fope_sep_head}); false broadcasts row 0
     * @param invFreq          frequencies the coefficients project, as {@link RopeMath#fopeInvFreq} returns them
     * @param cosCoef          the cosine coefficient tensor, {@code [n, D, D]} in ggml {@code ne} order
     * @param sinCoef          the sine coefficient tensor, the same shape
     * @throws IllegalArgumentException for a non-positive head count, an empty frequency list, or coefficient shapes
     *                                  that do not match the head count and frequency count
     */
    public static FopeSpec of(int numKeyValueHeads, boolean separateHeads, float[] invFreq, TensorRef cosCoef,
                              TensorRef sinCoef) {
        Objects.requireNonNull(invFreq, "invFreq");
        Objects.requireNonNull(cosCoef, "cosCoef");
        Objects.requireNonNull(sinCoef, "sinCoef");
        if (numKeyValueHeads <= 0) {
            throw new IllegalArgumentException("FoPE needs a positive KV head count, got " + numKeyValueHeads);
        }
        if (invFreq.length == 0) {
            throw new IllegalArgumentException("FoPE needs at least one frequency");
        }
        final int d = invFreq.length;
        checkCoef("cosine", cosCoef, numKeyValueHeads, d);
        checkCoef("sine", sinCoef, numKeyValueHeads, d);
        return new FopeSpec(numKeyValueHeads, separateHeads, invFreq, cosCoef, sinCoef);
    }

    private static void checkCoef(String what, TensorRef ref, int heads, int dim) {
        if (ref.ne(0) != dim || ref.ne(1) != dim || ref.ne(2) != heads) {
            throw new IllegalArgumentException("FoPE " + what + " coefficient tensor " + ref.name() + " has shape "
                    + Arrays.toString(ref.ne()) + ", expected [" + dim + ", " + dim + ", " + heads + "]");
        }
    }

    /** Number of coefficient rows: one per KV head. */
    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }

    /** Whether every KV head has its own coefficient row ({@code fope_sep_head}); row 0 serves all heads otherwise. */
    public boolean separateHeads() {
        return separateHeads;
    }

    /** Frequencies the coefficients project, in increasing pair order (copy). */
    public float[] invFreq() {
        return invFreq.clone();
    }

    /** Number of projected pairs {@code D}; pairs at or above it rotate with cos = sin = 1. */
    public int numInvFreq() {
        return invFreq.length;
    }

    public TensorRef cosCoef() {
        return cosCoef;
    }

    public TensorRef sinCoef() {
        return sinCoef;
    }

    /**
     * Index of the coefficient row head {@code h} reads: {@code h} for the KV heads themselves, the KV group of the
     * query head for a wider query ({@code group = nHeads / numKeyValueHeads} heads share a row); always 0 when the
     * heads are not separate.
     *
     * @throws IllegalArgumentException when {@code nHeads} is not a multiple of the KV head count
     */
    public int rowForHead(int nHeads, int h) {
        if (!separateHeads) {
            return 0;
        }
        if (nHeads <= 0 || nHeads % numKeyValueHeads != 0) {
            throw new IllegalArgumentException("FoPE rotates " + nHeads + " heads with " + numKeyValueHeads
                    + " coefficient rows; the head count must be a multiple of the KV head count");
        }
        return h / (nHeads / numKeyValueHeads);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FopeSpec)) {
            return false;
        }
        FopeSpec f = (FopeSpec) o;
        return numKeyValueHeads == f.numKeyValueHeads && separateHeads == f.separateHeads
                && Arrays.equals(invFreq, f.invFreq) && cosCoef.equals(f.cosCoef) && sinCoef.equals(f.sinCoef);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * (31 * numKeyValueHeads + Boolean.hashCode(separateHeads)) + Arrays.hashCode(invFreq))
                + 31 * cosCoef.hashCode() + sinCoef.hashCode();
    }

    @Override
    public String toString() {
        return "FopeSpec{kvHeads=" + numKeyValueHeads + ", sep=" + separateHeads + ", nInv=" + invFreq.length
                + ", cos=" + cosCoef.name() + ", sin=" + sinCoef.name() + '}';
    }
}
