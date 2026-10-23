package com.eonmux.neuralfree.gguf.ops.flow.dflash;

/**
 * The decoder stack of a dflash draft, in the two modes of its llama.cpp graph (dflash.cpp:803-805, 1108-1110):
 * an embedding batch of fused target features writes K/V into the draft cache, and a token batch (the noise
 * block) attends over that cache and produces the draft logits. Implementations write into the
 * {@link DflashKvCache} they were built with; the caller trims it.
 */
interface DflashStack {

    /**
     * Writes the cache rows of {@code nRows} fused feature rows {@code inpG[t*nEmbd ..]} (the encoder output
     * {@code rms(fc . features) * enc.output_norm}) at positions {@code pos0 + t}.
     */
    void inject(float[] inpG, int nRows, int pos0);

    /**
     * Runs the noise block {@code tokens} at positions {@code pos0 + t}; its own K/V rows are written to the cache
     * at those positions.
     */
    Block forward(int[] tokens, int pos0);

    /** Output of one noise block. Arrays are owned by the block. */
    final class Block {
        private final int nTokens;
        private final float[] logits;
        private final int vocab;
        private final float[] tEmbd;
        private final int nEmbd;

        Block(int nTokens, float[] logits, int vocab, float[] tEmbd, int nEmbd) {
            this.nTokens = nTokens;
            this.logits = logits;
            this.vocab = vocab;
            this.tEmbd = tEmbd;
            this.nEmbd = nEmbd;
        }

        int nTokens() { return nTokens; }

        /** Output-head logits {@code [nTokens][vocab]} in draft-head rows (before any {@code d2t} scatter). */
        float[] logits() { return logits; }

        /** Rows of the output head. */
        int vocab() { return vocab; }

        /**
         * {@code t_embd} rows {@code [nTokens][nEmbd]}: the normalized final hidden state (dflash.cpp:1050), the
         * pre-norm collapsed state for the DSV4 backbone (dflash.cpp:1256-1257).
         */
        float[] tEmbd() { return tEmbd; }

        int nEmbd() { return nEmbd; }
    }
}
