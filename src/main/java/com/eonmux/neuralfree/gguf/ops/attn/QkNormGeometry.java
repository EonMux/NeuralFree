package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.arch.Arch;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.QkNormKind;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Objects;
import java.util.Set;

/**
 * Row geometry of the Q or K norm of one softmax-attention layer, shared by {@link GpuAttentionMixer} and
 * {@link CpuAttentionMixer}, which only differ in how they load and broadcast the tensors. The projection
 * ({@code [n_tokens][n_heads * head_dim]}, dense) is normalized as {@code n_tokens * rowsPerToken} rows of
 * {@code rowLen} values ({@code LLM_RowNorm} / {@code NormMath.rowNorm}) with a weight of {@code wRows} rows, row
 * {@code r} using weight row {@code r % wRows}:
 * <ul>
 *   <li>{@link QkNormKind#RMS_HEAD}, {@link QkNormKind#LN_HEAD}: one row per head; the weight is one
 *       {@code head_dim} row shared by the heads, one row per head ({@code {head_dim, n_head}}, k2-horizon,
 *       models/k2-horizon.cpp:75-76), or one scalar per head ({@code {1, n_head}}, talkie, models/talkie.cpp:26 and
 *       87), which the mixers expand to {@code n_heads} rows ({@link #perHeadScalar});</li>
 *   <li>{@link QkNormKind#RMS_SPLIT_HALF}: one row per half head with a weight of two rows (sensenova-u1,
 *       models/sensenova-u1.cpp:103-107);</li>
 *   <li>{@link QkNormKind#RMS_FULL}, {@link QkNormKind#LN_FULL}: one row of the whole projection
 *       ({@code n_head * head_dim} for Q, {@code n_head_kv * head_dim} for K), as the BERT graph and mpt reshape Q
 *       and K to 2-D before {@code build_norm(LLM_NORM)} (models/bert.cpp:115-129, models/mpt.cpp:106-118).</li>
 * </ul>
 * The LayerNorm kinds use {@code f_norm_eps}, the RMS kinds {@code f_norm_rms_eps}. Immutable.
 */
public final class QkNormGeometry {

    /**
     * Graphs that normalize Q only when {@code attn_q_norm} exists and K only when {@code attn_k_norm} exists:
     * k2-horizon (models/k2-horizon.cpp:255-261), stablelm (models/stablelm.cpp:84-97), command-r
     * (models/command-r.cpp:79-89), chameleon (models/chameleon.cpp:90-104), glm4-moe (models/glm4-moe.cpp:197-204,
     * 325-332) and step35 (models/step35.cpp:225-232, 435-442).
     */
    private static final Set<Arch> PER_TENSOR = Collections.unmodifiableSet(EnumSet.of(
            Arch.K2_HORIZON, Arch.STABLELM, Arch.COMMAND_R, Arch.CHAMELEON, Arch.GLM4_MOE, Arch.STEP35));

    private final QkNormKind kind;
    private final NormKind normKind;
    private final float eps;
    private final int nHeads;
    private final int rowsPerToken;
    private final int rowLen;
    private final int layer;

    private QkNormGeometry(QkNormKind kind, NormKind normKind, float eps, int nHeads, int rowsPerToken, int rowLen,
                           int layer) {
        this.kind = kind;
        this.normKind = normKind;
        this.eps = eps;
        this.nHeads = nHeads;
        this.rowsPerToken = rowsPerToken;
        this.rowLen = rowLen;
        this.layer = layer;
    }

    /**
     * Geometry of a Q ({@code nHeads = n_head}) or K ({@code nHeads = n_head_kv}) norm of {@code kind}.
     *
     * @throws IllegalArgumentException for {@link QkNormKind#NONE}, non-positive sizes, a projection wider than
     *                                  {@code 2^31 - 1}, or a split-half norm over an odd head dim
     */
    public static QkNormGeometry of(QkNormKind kind, int headDim, int nHeads, float normEps, float rmsEps,
                                    int layer) {
        Objects.requireNonNull(kind, "kind");
        if (headDim <= 0 || nHeads <= 0) {
            throw new IllegalArgumentException("layer " + layer + ": Q/K norm needs positive heads and head dim, got "
                    + nHeads + " x " + headDim);
        }
        long full = (long) nHeads * headDim;
        if (full > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("layer " + layer + ": Q/K projection of " + nHeads + " x " + headDim
                    + " exceeds 2^31 - 1 values");
        }
        switch (kind) {
            case RMS_HEAD:
                return new QkNormGeometry(kind, NormKind.RMS, rmsEps, nHeads, nHeads, headDim, layer);
            case LN_HEAD:
                return new QkNormGeometry(kind, NormKind.LAYER, normEps, nHeads, nHeads, headDim, layer);
            case RMS_SPLIT_HALF:
                if (headDim % 2 != 0) {
                    throw new IllegalArgumentException("layer " + layer + ": split-half QK norm needs an even head"
                            + " dim, got " + headDim);
                }
                return new QkNormGeometry(kind, NormKind.RMS, rmsEps, nHeads, 2 * nHeads, headDim / 2, layer);
            case RMS_FULL:
                return new QkNormGeometry(kind, NormKind.RMS, rmsEps, nHeads, 1, (int) full, layer);
            case LN_FULL:
                return new QkNormGeometry(kind, NormKind.LAYER, normEps, nHeads, 1, (int) full, layer);
            case NONE:
            default:
                throw new IllegalArgumentException("layer " + layer + ": no Q/K norm geometry for " + kind);
        }
    }

    /**
     * True when the norm of a projection runs. The graphs that check the norm tensors follow their checks: the BERT
     * graph ({@link QkNormKind#LN_FULL}) and {@link #PER_TENSOR} normalize Q when {@code attn_q_norm} exists and K
     * when {@code attn_k_norm} exists; mpt normalizes both when {@code attn_q_norm} exists, K with whatever
     * {@code attn_k_norm} it has (models/mpt.cpp:106-118). The other graphs normalize both projections, a missing
     * weight giving the weightless norm (talkie's K, models/talkie.cpp:87-91; llama's {@code use_kq_norm}).
     *
     * @param key true for the K norm, false for the Q norm
     */
    public static boolean applies(QkNormKind kind, Arch arch, boolean key, boolean hasQWeight, boolean hasKWeight) {
        Objects.requireNonNull(kind, "kind");
        if (kind == QkNormKind.NONE) {
            return false;
        }
        if (arch == Arch.MPT) {
            return hasQWeight;
        }
        if (kind == QkNormKind.LN_FULL || PER_TENSOR.contains(arch)) {
            return key ? hasKWeight : hasQWeight;
        }
        return true;
    }

    /**
     * True when a weight or bias tensor of shape {@code {cols, rows}} holds one scalar per head ({@code {1, n_heads}})
     * of a per-head kind; ggml broadcasts it over each head's values, so the mixers expand it to
     * {@code [n_heads][rowLen]} and use {@link #nHeads()} weight rows.
     */
    public boolean perHeadScalar(long cols, long rows) {
        return QkvProjection.isPerHeadScalar(kind, cols, rows, nHeads, rowLen);
    }

    /**
     * Weight rows of a (not per-head-scalar) weight of {@code elements} values: whole rows of {@code rowLen} whose
     * count divides {@link #rowsPerToken()}; a split-half weight holds exactly two rows.
     *
     * @throws IllegalArgumentException naming the tensor otherwise
     */
    public int weightRows(String name, long elements) {
        if (elements % rowLen != 0 || elements / rowLen < 1 || rowsPerToken % (elements / rowLen) != 0) {
            throw new IllegalArgumentException("layer " + layer + ": norm weight " + name + " of " + elements
                    + " values does not tile rows of " + rowLen);
        }
        int wRows = (int) (elements / rowLen);
        if (kind == QkNormKind.RMS_SPLIT_HALF && wRows != 2) {
            throw new IllegalArgumentException("layer " + layer + ": split-half norm weight " + name
                    + " must hold one row per half head (" + 2 * rowLen + " values)");
        }
        return wRows;
    }

    /**
     * Checks a bias of {@code elements} values against the weight layout: {@code wRows * rowLen} values (the bias
     * rows follow the weight rows).
     *
     * @throws IllegalArgumentException naming the tensor otherwise
     */
    public void checkBias(String name, long elements, int wRows) {
        QkvProjection.checkLength(name, elements, (long) wRows * rowLen);
    }

    public QkNormKind kind() { return kind; }
    /** {@link NormKind#LAYER} for the LayerNorm kinds, {@link NormKind#RMS} otherwise. */
    public NormKind normKind() { return normKind; }
    public float eps() { return eps; }
    /** Heads of the projection ({@code n_head} for Q, {@code n_head_kv} for K). */
    public int nHeads() { return nHeads; }
    /** Normalized rows per token. */
    public int rowsPerToken() { return rowsPerToken; }
    /** Values per normalized row. */
    public int rowLen() { return rowLen; }

    @Override
    public String toString() {
        return "QkNormGeometry{" + kind + ", " + normKind + " eps " + eps + ", " + rowsPerToken + " x " + rowLen
                + " per token}";
    }
}
