package com.eonmux.neuralfree.gguf.hparams;

/**
 * Four-section RoPE positions of text tokens for {@code MROPE}/{@code IMROPE} layers. Positions are laid out
 * section-major, {@code pos[s * nTokens + t]}, matching llama.cpp src/llama-batch.cpp:146-157.
 */
public enum MropeTextPositions {
    /** {@code [p, p, p, 0]}: temporal, height and width carry the token position (llama.cpp default). */
    PPP0,
    /** {@code [p, 0, 0, 0]}: only the temporal section carries the token position. */
    P000;

    /** Number of position sections per token. */
    public static final int SECTIONS = 4;

    /** Position of section {@code section} (0..3) for a text token at position {@code pos}. */
    public int position(int section, int pos) {
        if (section < 0 || section >= SECTIONS) {
            throw new IllegalArgumentException("section " + section + " outside 0.." + (SECTIONS - 1));
        }
        switch (this) {
            case PPP0:
                return section < 3 ? pos : 0;
            case P000:
                return section == 0 ? pos : 0;
            default:
                throw new IllegalStateException("unhandled pattern " + this);
        }
    }

    /**
     * Fills {@code dst[0 .. 4 * nTokens)} with the section-major positions of {@code nTokens} consecutive
     * text tokens starting at {@code pos0}.
     */
    public void fill(int pos0, int nTokens, int[] dst) {
        if (nTokens < 0 || dst.length < SECTIONS * nTokens) {
            throw new IllegalArgumentException("destination holds " + dst.length + " positions, need "
                    + SECTIONS * nTokens);
        }
        for (int s = 0; s < SECTIONS; s++) {
            for (int t = 0; t < nTokens; t++) {
                dst[s * nTokens + t] = position(s, pos0 + t);
            }
        }
    }
}
