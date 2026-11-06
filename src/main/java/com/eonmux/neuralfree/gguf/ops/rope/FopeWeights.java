package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.spec.FopeSpec;

import java.util.Arrays;
import java.util.Objects;

/**
 * The learned FoPE coefficients of one layer, bound to their data: the {@link FopeSpec} parameters plus the cosine
 * and sine coefficient matrices every layer of the model shares.
 *
 * <p>{@link #cosSin} reproduces the cos/sin row of vLLM's {@code FourierRotaryEmbedding._compute_cos_sin_cache}:
 * pair {@code i} of head {@code h} at position {@code p} rotates by
 * {@code (sum_d cos(p * invFreq[d]) * cosCoef[row, d, i], sum_d sin(p * invFreq[d]) * sinCoef[row, d, i])} when
 * {@code i < D}, and by {@code (1, 1)} otherwise. Immutable and shareable across threads.
 */
public final class FopeWeights {

    private final FopeSpec spec;
    private final float[] cosCoef;
    private final float[] sinCoef;
    private final int dim;

    private FopeWeights(FopeSpec spec, float[] cosCoef, float[] sinCoef) {
        this.spec = spec;
        this.cosCoef = cosCoef;
        this.sinCoef = sinCoef;
        this.dim = spec.numInvFreq();
    }

    /**
     * Coefficients of {@code spec}, read from its tensors.
     *
     * @throws IllegalArgumentException when a matrix does not hold {@code nKv * D * D} values
     */
    public static FopeWeights of(FopeSpec spec, float[] cosCoef, float[] sinCoef) {
        Objects.requireNonNull(spec, "spec");
        Objects.requireNonNull(cosCoef, "cosCoef");
        Objects.requireNonNull(sinCoef, "sinCoef");
        final long n = (long) spec.numKeyValueHeads() * spec.numInvFreq() * spec.numInvFreq();
        if (cosCoef.length < n || sinCoef.length < n) {
            throw new IllegalArgumentException("FoPE coefficients hold " + cosCoef.length + " and " + sinCoef.length
                    + " values, need " + n + " (" + spec.numKeyValueHeads() + " heads of "
                    + spec.numInvFreq() + "x" + spec.numInvFreq() + ")");
        }
        return new FopeWeights(spec, cosCoef, sinCoef);
    }

    public FopeSpec spec() {
        return spec;
    }

    /** Coefficient rows ({@code num_key_value_heads}). */
    public int rows() {
        return spec.numKeyValueHeads();
    }

    /** Projected pairs {@code D}. */
    public int dim() {
        return dim;
    }

    /** Cosine coefficient matrix in ggml {@code ne} order, {@code [D, D, rows]} (shared). */
    public float[] cosCoef() {
        return cosCoef;
    }

    /** Sine coefficient matrix in ggml {@code ne} order, {@code [D, D, rows]} (shared). */
    public float[] sinCoef() {
        return sinCoef;
    }

    /** Frequencies the coefficients project (copy). */
    public float[] invFreq() {
        return spec.invFreq();
    }

    /** Values a caller-owned scratch array must hold for {@link #cosSin}. */
    public int basisSize() {
        return 2 * dim;
    }

    /**
     * Fills {@code cs[2i] = cos} and {@code cs[2i + 1] = sin} of every rotation pair for position {@code p} and
     * head {@code h} of {@code nHeads} rotated heads. {@code basis} is caller-owned scratch of at least
     * {@link #basisSize()} values, filled with {@code cos(p * invFreq[d])} then {@code sin(p * invFreq[d])}.
     *
     * @throws IllegalArgumentException when {@code cs} or {@code basis} is too short, or the head count does not
     *                                  match the rows
     */
    public void cosSin(int p, int nHeads, int h, int pairs, float[] cs, float[] basis) {
        Objects.requireNonNull(cs, "cs");
        Objects.requireNonNull(basis, "basis");
        if (cs.length < 2 * pairs) {
            throw new IllegalArgumentException("cos/sin row holds " + cs.length + " values, need " + (2 * pairs));
        }
        if (basis.length < basisSize()) {
            throw new IllegalArgumentException("FoPE scratch holds " + basis.length + " values, need " + basisSize());
        }
        final int row = spec.rowForHead(nHeads, h);
        final int rowOff = row * dim * dim;
        final float[] inv = spec.invFreq();
        for (int d = 0; d < dim; d++) {
            final double angle = (double) p * inv[d];
            basis[d] = (float) Math.cos(angle);
            basis[dim + d] = (float) Math.sin(angle);
        }
        for (int i = 0; i < pairs; i++) {
            if (i >= dim) {
                cs[2 * i] = 1.0f;
                cs[2 * i + 1] = 1.0f;
                continue;
            }
            float c = 0f;
            float s = 0f;
            for (int d = 0; d < dim; d++) {
                final int off = rowOff + d * dim + i;
                c += basis[d] * cosCoef[off];
                s += basis[dim + d] * sinCoef[off];
            }
            cs[2 * i] = c;
            cs[2 * i + 1] = s;
        }
    }

    @Override
    public String toString() {
        return "FopeWeights{" + spec + ", values=" + cosCoef.length + "/" + sinCoef.length + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof FopeWeights)) {
            return false;
        }
        FopeWeights f = (FopeWeights) o;
        return spec.equals(f.spec) && Arrays.equals(cosCoef, f.cosCoef) && Arrays.equals(sinCoef, f.sinCoef);
    }

    @Override
    public int hashCode() {
        return 31 * (31 * spec.hashCode() + Arrays.hashCode(cosCoef)) + Arrays.hashCode(sinCoef);
    }
}
