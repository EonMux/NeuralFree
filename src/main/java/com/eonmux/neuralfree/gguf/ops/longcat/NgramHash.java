package com.eonmux.neuralfree.gguf.ops.longcat;

import com.eonmux.neuralfree.gguf.memory.SequenceState;

import java.util.Objects;

/**
 * The hashed n-gram ids of LongCat-Flash-Lite's input embedding, the host twin of vLLM's
 * {@code ngram_compute_n_gram_ids} kernel (csrc/ngram_embedding_kernels.cu:16-66).
 *
 * <p>There are {@code splits * (neighbors - 1)} embedders, one per (order, split): embedder
 * {@code e = order * splits + split} hashes the current token and the {@code order + 1} tokens before it. With
 * {@code m} the base table size, embedder {@code e} uses modulus {@code m + 2 * e + 1} and the weights
 * {@code vocab^d} for shift {@code d}, and its global row is
 * {@code (sum_d token[pos - d] * vocab^d mod modulus) mod modulus} plus the embedder's offset in the concatenated
 * table, {@code sum_{q < e} (m + 2q + 1) = e * m + e^2}. The walk stops at the request start, at an embedding row and
 * at the end-of-sequence token, which the kernel records as a negative table entry (longcat_flash_ngram.py:303-304,
 * 360-374).</p>
 */
final class NgramHash {

    private final int splits;
    private final int neighbors;
    private final int embedders;
    private final int eosTokenId;
    private final int[] mods;
    private final long[] offsets;
    private final long[][] powers;
    private final int[] window;

    /**
     * @param base        base table size {@code m = ngram_vocab_size_ratio * vocab_size}
     * @param splits      tables per n-gram order ({@code emb_split_num})
     * @param neighbors   tokens per hash, the current one included ({@code emb_neighbor_num})
     * @param eosTokenId  token that ends a hash walk
     * @param vocab       vocabulary size, the hash base
     * @param ubatch      largest step of the model, for the working window
     * @throws IllegalArgumentException for a geometry whose tables or offsets do not fit the GGUF row count
     */
    NgramHash(long base, int splits, int neighbors, int eosTokenId, long vocab, int ubatch) {
        if (splits < 1 || neighbors < 2) {
            throw new IllegalArgumentException("n-gram embedding needs splits >= 1 and neighbors >= 2, got " + splits
                    + " and " + neighbors);
        }
        if (eosTokenId < 0 || eosTokenId > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("n-gram end-of-sequence token must be non-negative, got "
                    + eosTokenId);
        }
        if (vocab <= 0 || base <= 0) {
            throw new IllegalArgumentException("n-gram hash needs a positive vocabulary and base table size, got "
                    + vocab + " and " + base);
        }
        if (ubatch < 1) {
            throw new IllegalArgumentException("n-gram window needs a positive ubatch, got " + ubatch);
        }
        this.splits = splits;
        this.neighbors = neighbors;
        this.eosTokenId = eosTokenId;
        this.embedders = Math.multiplyExact(splits, neighbors - 1);
        this.mods = new int[embedders];
        this.offsets = new long[embedders];
        this.powers = new long[embedders][neighbors];
        long total = 0;
        for (int e = 0; e < embedders; e++) {
            long mod = base + 2L * e + 1;
            if (mod > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("n-gram embedder " + e + " has modulus " + mod
                        + ", beyond the row count of one table");
            }
            mods[e] = (int) mod;
            offsets[e] = e * base + (long) e * e;
            long totalE = offsets[e] + mod;
            if (totalE > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("n-gram table rows reach " + totalE + ", beyond the row count of"
                        + " one table");
            }
            total = Math.max(total, totalE);
            long p = 1;
            for (int d = 0; d < neighbors; d++) {
                powers[e][d] = p;
                p = p * vocab % mod;
            }
        }
        this.window = new int[Math.addExact(ubatch, neighbors - 1)];
        if (total != expectedRows(base, splits, neighbors)) {
            throw new IllegalStateException("n-gram table layout mismatch: computed " + total + " rows");
        }
    }

    /** Rows of the concatenated table: {@code embedders * base + embedders^2}. */
    static long expectedRows(long base, int splits, int neighbors) {
        long embedders = (long) splits * (neighbors - 1);
        return embedders * base + embedders * embedders;
    }

    int embedders() {
        return embedders;
    }

    int splits() {
        return splits;
    }

    int neighbors() {
        return neighbors;
    }

    int eosTokenId() {
        return eosTokenId;
    }

    /**
     * Writes the {@code n * embedders()} global table rows of the step in token-major order, {@code out[t * E + e]},
     * hashing the {@code n} tokens {@code step[tokStart .. tokStart + n)} against the sequence record.
     *
     * @param record     the sequence's stored tokens (the step's own rows are not stored yet)
     * @param contiguous true for the contiguous record, whose entries are positions
     * @param pos0       position of the segment's first row
     * @throws IllegalStateException when the record does not hold the tokens before {@code pos0}
     */
    void ids(int[] step, int tokStart, int n, SequenceState record, boolean contiguous, int pos0, int[] out) {
        Objects.requireNonNull(step, "step");
        Objects.requireNonNull(record, "record");
        Objects.requireNonNull(out, "out");
        long need = (long) n * embedders;
        if (out.length < need) {
            throw new IllegalArgumentException("n-gram id buffer holds " + out.length + " values, need " + need);
        }
        fillWindow(step, tokStart, n, record, contiguous, pos0);
        int back = neighbors - 1;
        for (int t = 0; t < n; t++) {
            int current = back + t;
            for (int e = 0; e < embedders; e++) {
                int order = e / splits;
                int mod = mods[e];
                long id = 0;
                for (int d = 0; d <= order + 1; d++) {
                    int at = current - d;
                    if (at < 0) {
                        break;
                    }
                    int token = window[at];
                    if (token < 0 || token == eosTokenId) {
                        break;
                    }
                    id = (id + (long) token * powers[e][d]) % mod;
                }
                out[t * embedders + e] = (int) (id % mod + offsets[e]);
            }
        }
    }

    /** The hash window: the {@code neighbors - 1} predecessors first, then the step's tokens. */
    private void fillWindow(int[] step, int tokStart, int n, SequenceState record, boolean contiguous, int pos0) {
        int back = neighbors - 1;
        if (pos0 > 0 && (contiguous ? record.nPast() < pos0 : record.nPast() == 0)) {
            throw new IllegalStateException("n-gram embedding at position " + pos0 + " needs the tokens before it"
                    + " in the sequence record, which holds " + record.nPast());
        }
        for (int w = 0; w < back; w++) {
            int pos = pos0 - back + w;
            window[w] = pos < 0 ? -1 : record.tokenAtOrBefore(pos);
        }
        System.arraycopy(step, tokStart, window, back, n);
    }
}
