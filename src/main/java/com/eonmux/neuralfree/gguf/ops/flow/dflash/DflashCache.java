package com.eonmux.neuralfree.gguf.ops.flow.dflash;

/**
 * The position bookkeeping a dflash drafter drives on its own KV cache, wherever the rows live: the host
 * {@link DflashKvCache} or the device {@link GpuDflashKvCache}. Rows of rejected tokens are dropped by position
 * ({@link #trim}, llama.cpp's {@code seq_rm(seq, pos, -1)} on the draft context).
 */
interface DflashCache {

    /** Positions a full-attention layer can hold. */
    int capacity();

    /** True when positions {@code [0, pos + n)} fit the full-attention layers. */
    boolean fits(int pos, int n);

    /** Forgets every stored position {@code >= pos}. */
    void trim(int pos);

    /** Forgets every position. */
    void clear();
}
