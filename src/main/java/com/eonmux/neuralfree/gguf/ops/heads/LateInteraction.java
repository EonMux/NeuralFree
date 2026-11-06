package com.eonmux.neuralfree.gguf.ops.heads;

/**
 * Late-interaction scoring of two per-token embedding sets (vLLM's late interaction scorer, over the L2-normalized
 * token rows of a {@code token_embed} model):
 * <pre>
 *   token_scores = q @ d^T        (one dot product per query/document token pair)
 *   score        = token_scores.amax(dim=-1).sum()
 * </pre>
 * the sum over the query tokens of the maximum dot product over the document tokens. The sides encode
 * independently; the score needs no cross-attention. Pure and stateless.
 */
public final class LateInteraction {

    private LateInteraction() {
    }

    /**
     * The MaxSim score of one query's per-token rows against one document's: {@code sum_t max_u (q_t . d_u)}, the
     * rows packed as {@code [tokens][width]}.
     *
     * @throws IllegalArgumentException when a side is empty, a row is null, or the sides have different widths
     */
    public static float maxSim(float[][] queryRows, float[][] docRows) {
        requireRows("query", queryRows);
        requireRows("document", docRows);
        int width = queryRows[0].length;
        float sum = 0.0f;
        for (float[] q : queryRows) {
            if (q == null || q.length != width) {
                throw new IllegalArgumentException("the query rows must share one width, got "
                        + (q == null ? "null" : String.valueOf(q.length)) + " against " + width);
            }
            float best = Float.NEGATIVE_INFINITY;
            for (float[] d : docRows) {
                if (d.length != width) {
                    throw new IllegalArgumentException("the document rows must share the query width " + width
                            + ", got " + d.length);
                }
                float dot = 0.0f;
                for (int c = 0; c < width; c++) {
                    dot += q[c] * d[c];
                }
                best = Math.max(best, dot);
            }
            sum += best;
        }
        return sum;
    }

    private static void requireRows(String name, float[][] rows) {
        if (rows == null || rows.length == 0) {
            throw new IllegalArgumentException("the " + name + " side needs at least one token row, got "
                    + (rows == null ? "null" : "0"));
        }
        if (rows[0] == null || rows[0].length == 0) {
            throw new IllegalArgumentException("the " + name + " side's first row is empty");
        }
    }
}
