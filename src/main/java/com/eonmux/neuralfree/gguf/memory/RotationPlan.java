package com.eonmux.neuralfree.gguf.memory;

import java.util.Objects;

/**
 * Hadamard rotation of quantized KV rows (src/llama-kv-cache.cpp:315-360; graph use llama-graph.cpp:3225-3231 and
 * 3270). When {@link #rotateK()}, Q and K are multiplied by a block-diagonal Hadamard matrix of size
 * {@link #nRotK()} before K is stored; when {@link #rotateV()}, V is rotated with blocks of {@link #nRotV()} before
 * it is stored and the attention output is rotated back. Rotation spreads outliers so the quantization error of the
 * block types (Q8_0, Q4_0, Q4_1, IQ4_NL, Q5_0, Q5_1) drops; F32, F16 and BF16 are not rotated. Scores are unchanged
 * because the matrix is orthogonal. Immutable.
 */
public final class RotationPlan {

    /** No rotation. */
    public static final RotationPlan NONE = new RotationPlan(0, 0);

    /** Smallest rotation block. */
    public static final int BASE_BLOCK = 64;

    private final int nRotK;
    private final int nRotV;

    private RotationPlan(int nRotK, int nRotV) {
        this.nRotK = nRotK;
        this.nRotV = nRotV;
    }

    /**
     * Plan for one cache, following llama.cpp:
     * <ul>
     *   <li>K: rotated when enabled, every cached layer has the same K head dim ({@code headDimKAll > 0}),
     *       {@code typeK} is quantized and {@code headDimK % 64 == 0}; or always when {@code forceK} (the DeepSeek
     *       lightning-indexer archs with {@code n_embd_head_k_full == indexer_head_size}). {@code nRotK} starts at 64
     *       and doubles while {@code headDimKAll % (2 * nRotK) == 0}.</li>
     *   <li>V: rotated when enabled, {@code headDimVAll > 0}, {@code typeV} is quantized and
     *       {@code headDimV % 64 == 0}; {@code nRotV = 64}.</li>
     * </ul>
     *
     * @param headDimKAll the K head dim shared by every cached layer, or -1 when it varies
     * @param headDimVAll the V head dim shared by every cached layer, -1 when it varies, 0 for MLA caches
     * @param headDimK    the model K head dim ({@code n_embd_head_k()})
     * @param headDimV    the model V head dim ({@code n_embd_head_v()})
     * @param enabled     {@code neuralfree.kv.rotation}
     * @param forceK      rotate K regardless of {@code enabled} and the type
     */
    public static RotationPlan of(KvDType typeK, KvDType typeV, int headDimKAll, int headDimVAll, int headDimK,
                                  int headDimV, boolean enabled, boolean forceK) {
        Objects.requireNonNull(typeK, "typeK");
        Objects.requireNonNull(typeV, "typeV");
        boolean rotK = forceK
                || (enabled && headDimKAll > 0 && typeK.quantized() && headDimK % BASE_BLOCK == 0);
        boolean rotV = enabled && headDimVAll > 0 && typeV.quantized() && headDimV % BASE_BLOCK == 0;
        int nRotK = 0;
        if (rotK) {
            nRotK = BASE_BLOCK;
            while (headDimKAll > 0 && headDimKAll % (2 * nRotK) == 0) {
                nRotK *= 2;
            }
        }
        int nRotV = rotV ? BASE_BLOCK : 0;
        return nRotK == 0 && nRotV == 0 ? NONE : new RotationPlan(nRotK, nRotV);
    }

    public boolean rotateK() { return nRotK > 0; }
    public boolean rotateV() { return nRotV > 0; }
    /** K / Q rotation block size; 0 when K is not rotated. */
    public int nRotK() { return nRotK; }
    /** V rotation block size; 0 when V is not rotated. */
    public int nRotV() { return nRotV; }

    @Override
    public String toString() {
        return this == NONE ? "RotationPlan.NONE" : "RotationPlan{nRotK=" + nRotK + ", nRotV=" + nRotV + '}';
    }
}
