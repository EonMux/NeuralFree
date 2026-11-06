package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU twin of {@code LLM_InklingQueryScale} and {@code LLM_InklingRelLogits}: the per-token log scaling of the
 * full-attention layers and the query-side relative-logit table of Inkling.
 *
 * <p>The log scaling is vLLM's {@code compute_log_scaling_tau}: {@code tau = 1 + alpha * log(max(1, (pos + 1) /
 * n_floor))}, {@code 1} when {@code n_floor <= 0}. It multiplies the normalized queries and the relative logits of a
 * full-attention layer; a local layer applies neither.
 *
 * <p>The relative table of query {@code t} and head {@code h} is {@code rel[t][h][e] = tau[t] * sum_j r[t][h][j] *
 * proj[j][e]} with the shared {@code [d_rel][rel_extent]} projection {@code attn_rel_proj} (vLLM's
 * {@code RelLogitsProj}, einsum {@code "thd,de->the"}). Rows are the tokens of the step, which is the row index the
 * attention kernels add the bias by.
 */
public final class InklingMath {

    /** Tokens (and rows of the table) per parallel task of the relative projection. */
    private static final int REL_GRAIN = 8;

    private InklingMath() {
    }

    /**
     * Log-scaling factor of a token at {@code pos}; 1 when {@code nFloor <= 0}.
     *
     * @throws IllegalArgumentException when {@code pos} is negative
     */
    public static float tau(int pos, int nFloor, float alpha) {
        if (pos < 0) {
            throw new IllegalArgumentException("log scaling needs a non-negative position, got " + pos);
        }
        if (nFloor <= 0) {
            return 1.0f;
        }
        float x = (pos + 1) / (float) nFloor;
        return 1.0f + alpha * (float) Math.log(Math.max(x, 1.0f));
    }

    /**
     * {@code q[t][h][d] *= tau(pos[t])} over {@code n} token rows of {@code nHead} heads.
     *
     * @param pos positions of the step's tokens from {@code posOff}
     */
    public static void scaleQueries(int n, int nHead, int headDim, int rowStride, float[] q, int qOff,
                                    int[] pos, int posOff, int nFloor, float alpha) {
        if (nFloor <= 0) {
            return;
        }
        final int width = nHead * headDim;
        CpuParallel.parallelFor(n, 1, (from, to) -> {
            for (int t = from; t < to; t++) {
                float tau = tau(pos[posOff + t], nFloor, alpha);
                int base = qOff + t * rowStride;
                for (int i = 0; i < width; i++) {
                    q[base + i] *= tau;
                }
            }
        });
    }

    /**
     * Writes the relative-logit table of {@code n} token rows: {@code out[(t * nHead + h) * relExtent + e] =
     * tau[pos[t]] * sum_j r[t][h][j] * proj[j][e]}.
     *
     * @param r      the {@code attn_rel} output, {@code [t][nHead][dRel]} from index 0
     * @param proj   the shared {@code [dRel][relExtent]} relative projection
     * @param pos    positions of the step's tokens from {@code posOff}
     * @param out    the table, {@code [t][nHead][relExtent]} from index 0
     * @param nFloor log-scaling floor; {@code <= 0} leaves the table unscaled
     * @throws IllegalArgumentException when an array does not hold the rows the geometry needs
     */
    public static void relLogits(int n, int nHead, int dRel, int relExtent, float[] r, float[] proj, int[] pos,
                                 int posOff, int nFloor, float alpha, float[] out) {
        requireSpan("attn_rel output", r, (long) n * nHead * dRel);
        requireSpan("attn_rel_proj", proj, (long) dRel * relExtent);
        requireSpan("relative-logit table", out, (long) n * nHead * relExtent);
        if (n > 0 && posOff + n > pos.length) {
            throw new IllegalArgumentException("positions need [" + posOff + ", " + (posOff + (long) n) + ") of "
                    + pos.length);
        }
        final boolean scaled = nFloor > 0;
        CpuParallel.parallelFor(n, REL_GRAIN, (from, to) -> {
            for (int t = from; t < to; t++) {
                float tau = scaled ? tau(pos[posOff + t], nFloor, alpha) : 1.0f;
                for (int h = 0; h < nHead; h++) {
                    int rBase = (t * nHead + h) * dRel;
                    int oBase = (t * nHead + h) * relExtent;
                    for (int e = 0; e < relExtent; e++) {
                        float acc = 0.0f;
                        for (int j = 0; j < dRel; j++) {
                            acc += r[rBase + j] * proj[j * relExtent + e];
                        }
                        out[oBase + e] = acc * tau;
                    }
                }
            }
        });
    }

    private static void requireSpan(String what, float[] a, long count) {
        if (a == null || a.length < count) {
            throw new IllegalArgumentException(what + " holds " + (a == null ? 0 : a.length) + " floats, need "
                    + count);
        }
    }
}
