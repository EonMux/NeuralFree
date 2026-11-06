package com.eonmux.neuralfree.gguf.mm;

/**
 * Encoder output before the projector: {@code nTokens} rows of {@code width} values ({@code n_embd}, or a
 * multiple of it for stacked feature layers), the patches of a {@code pw x ph} grid in the order the family's
 * graph lays them out. Package-private executor support.
 */
final class Encoded {

    final float[] rows;
    final int nTokens;
    final int width;
    final int pw;
    final int ph;

    Encoded(float[] rows, int nTokens, int pw, int ph) {
        if (nTokens <= 0 || rows.length % nTokens != 0) {
            throw new IllegalArgumentException(rows.length + " values do not split into " + nTokens + " rows");
        }
        this.rows = rows;
        this.nTokens = nTokens;
        this.width = rows.length / nTokens;
        this.pw = pw;
        this.ph = ph;
    }

    /** Refuses rows that are not exactly the patch grid (a class token left in by the encoder). */
    void requireGrid(String what) {
        if (nTokens != pw * ph) {
            throw new IllegalStateException(what + " reads a " + pw + "x" + ph + " patch grid, the encoder"
                    + " produced " + nTokens + " rows");
        }
    }
}
