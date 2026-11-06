package com.eonmux.neuralfree.gguf.ops.rope;

import com.eonmux.neuralfree.gguf.spec.DualChunkSpec;

import java.util.Objects;

/**
 * Positions of the five dual-chunk query rotations and of the key rotation (vLLM
 * {@code DualChunkRotaryEmbedding._compute_cos_sin_cache} and {@code forward_native}).
 *
 * <p>With {@code chunkLen = chunk_size - local_size}, a key at position {@code p} rotates at
 * {@code p % chunkLen}. The query variants of a token at {@code p} use:
 * <ul>
 *   <li>{@code query}: {@code p % chunkLen};</li>
 *   <li>{@code query_succ}: {@code min(p % chunkLen + chunkLen, chunk_size)};</li>
 *   <li>{@code query_inter}: {@code min(2 * chunkLen - 1, chunk_size)}, the same for every token;</li>
 *   <li>{@code query_succ_critical}: {@code p % chunkLen + chunkLen};</li>
 *   <li>{@code query_inter_critical}: {@code p % chunkLen + chunk_size}.</li>
 * </ul>
 * The critical variants feed vLLM's sparse-selection score; they are rotated for every token even though the dense
 * attention does not read them, so the query layout is the reference's.
 */
public final class DualChunkRope {

    /** Query variants of one token ({@code query}, {@code query_succ}, {@code query_inter}, the two critical ones). */
    public static final int VARIANTS = 5;

    private DualChunkRope() {
    }

    /** {@code min(2 * chunkLen - 1, chunk_size)}, the constant {@code query_inter} position. */
    public static int interPosition(DualChunkSpec dc) {
        Objects.requireNonNull(dc, "dc");
        int chunkLen = dc.chunkLength();
        return Math.min(2 * chunkLen - 1, dc.chunkSize());
    }

    /**
     * Fills {@code out[s * n + t]} with variant {@code s} of token {@code t}, whose position is
     * {@code pos[posOff + t]}.
     *
     * @param out at least {@code 5 * n} ints
     * @throws IllegalArgumentException when the position is negative or the array is too short
     */
    public static void queryPositions(DualChunkSpec dc, int[] pos, int posOff, int n, int[] out) {
        Objects.requireNonNull(dc, "dc");
        Objects.requireNonNull(pos, "pos");
        Objects.requireNonNull(out, "out");
        if (out.length < (long) VARIANTS * n) {
            throw new IllegalArgumentException("dual-chunk position scratch holds " + out.length + " values, need "
                    + VARIANTS + " x " + n);
        }
        if (n > 0 && posOff + n > pos.length) {
            throw new IllegalArgumentException("positions [" + posOff + ", " + (posOff + n) + ") exceed the "
                    + pos.length + "-value buffer");
        }
        final int chunkLen = dc.chunkLength();
        final int chunkSize = dc.chunkSize();
        final int inter = interPosition(dc);
        for (int t = 0; t < n; t++) {
            int p = pos[posOff + t];
            if (p < 0) {
                throw new IllegalArgumentException("dual-chunk attention needs non-negative positions, got " + p);
            }
            int base = p % chunkLen;
            out[t] = base;
            out[n + t] = Math.min(base + chunkLen, chunkSize);
            out[2 * n + t] = inter;
            out[3 * n + t] = base + chunkLen;
            out[4 * n + t] = base + chunkSize;
        }
    }

    /**
     * Fills {@code out[t]} with the key rotation position {@code pos[posOff + t] % chunkLen}.
     *
     * @throws IllegalArgumentException when the position is negative or the array is too short
     */
    public static void keyPositions(DualChunkSpec dc, int[] pos, int posOff, int n, int[] out) {
        Objects.requireNonNull(dc, "dc");
        Objects.requireNonNull(pos, "pos");
        Objects.requireNonNull(out, "out");
        if (out.length < n) {
            throw new IllegalArgumentException("dual-chunk key position scratch holds " + out.length + " values, need "
                    + n);
        }
        if (n > 0 && posOff + n > pos.length) {
            throw new IllegalArgumentException("positions [" + posOff + ", " + (posOff + n) + ") exceed the "
                    + pos.length + "-value buffer");
        }
        final int chunkLen = dc.chunkLength();
        for (int t = 0; t < n; t++) {
            int p = pos[posOff + t];
            if (p < 0) {
                throw new IllegalArgumentException("dual-chunk attention needs non-negative positions, got " + p);
            }
            out[t] = p % chunkLen;
        }
    }

    /**
     * Rotates the five query planes of {@code n} tokens starting at array offset 0 (plane {@code s} starts at
     * {@code s * planeStride}): plane 0 arrives with the projected query and planes 1..4 are copies of it, each then
     * rotated at its own position with {@code table} (a {@link RopeFreqTable#forDualChunk} table).
     *
     * @param qRowStride values per token within a plane
     * @param nOffs      first rotated dim within each head
     * @param posScratch at least {@code 5 * n} ints, overwritten
     * @throws IllegalArgumentException on inconsistent geometry
     */
    public static void applyQuery(RopeFreqTable table, DualChunkSpec dc, float[] q, int planeStride, int n,
                                  int nHeads, int headDimK, int qRowStride, int nOffs, int[] pos, int posOff,
                                  int[] posScratch) {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(q, "q");
        if (planeStride < (long) nHeads * headDimK) {
            throw new IllegalArgumentException("dual-chunk query plane stride " + planeStride + " is below the query"
                    + " width " + nHeads * headDimK);
        }
        int rowElems = n * qRowStride;
        for (int variant = 1; variant < VARIANTS; variant++) {
            System.arraycopy(q, 0, q, variant * planeStride, rowElems);
        }
        queryPositions(dc, pos, posOff, n, posScratch);
        for (int variant = 0; variant < VARIANTS; variant++) {
            CpuRope.apply(table, q, variant * planeStride, n, nHeads, headDimK, qRowStride, nOffs,
                    posScratch, variant * n, 1);
        }
    }

    /**
     * Rotates the keys of {@code n} tokens at their positions modulo the chunk length.
     *
     * @param kRowStride values per token
     * @throws IllegalArgumentException on inconsistent geometry
     */
    public static void applyKey(RopeFreqTable table, DualChunkSpec dc, float[] k, int n, int nHeads, int headDimK,
                                int kRowStride, int nOffs, int[] pos, int posOff, int[] posScratch) {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(k, "k");
        keyPositions(dc, pos, posOff, n, posScratch);
        CpuRope.apply(table, k, 0, n, nHeads, headDimK, kRowStride, nOffs, posScratch, 0, 1);
    }
}
