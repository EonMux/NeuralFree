package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.KvLayout;

/**
 * Heap KV cache of one owning layer, stored in the {@link com.eonmux.neuralfree.gguf.memory.KvDType} of its
 * {@link KvLayout} (F32, F16, BF16 or a 32-value block type). Reads dequantize to F32; writes quantize. Row
 * {@code r} holds the position {@link CpuMemory#positionOfRow(int, int)} reports for this cache's group. Head
 * {@code h} of a K row is values {@code [h * headDimK, (h + 1) * headDimK)}; for an MLA latent cache there is one
 * head and V reads the first {@code widthV} values of the K row.
 */
public interface KvAccessor {

    /** Layer that owns this cache. */
    int layer();

    KvLayout layout();

    /** KV group addressing this cache. */
    int group();

    /** KV heads per row. */
    int nHeadKv();

    int headDimK();

    int headDimV();

    /**
     * Dequantizes K head {@code head} of row {@code row} into {@code dst[off, off + headDimK)}.
     *
     * @throws IndexOutOfBoundsException for a row, head or destination range outside the cache
     */
    void readK(int row, int head, float[] dst, int off);

    /**
     * Dequantizes K values {@code [from, from + n)} of row {@code row} into {@code dst[off, off + n)}, a view of part
     * of a stored row (for example the compressed KV of a DeepSeek-V4.1 row that also holds the index key).
     *
     * <p>The default serves a whole single-head row and refuses a partial range; a cache that stores several value
     * groups in one row overrides it.</p>
     *
     * @throws IndexOutOfBoundsException     for a row, range or destination range outside the cache
     * @throws UnsupportedOperationException for a partial range on an accessor without a slice reader
     */
    default void readKSlice(int row, int from, int n, float[] dst, int off) {
        if (from != 0 || n != headDimK()) {
            throw new UnsupportedOperationException("partial K rows [" + from + ", " + (from + n)
                    + ") are not readable from " + getClass().getSimpleName());
        }
        readK(row, 0, dst, off);
    }

    /**
     * Dequantizes V head {@code head} of row {@code row} into {@code dst[off, off + headDimV)}.
     *
     * @throws IndexOutOfBoundsException for a row, head or destination range outside the cache
     */
    void readV(int row, int head, float[] dst, int off);

    /** Stores a full K row ({@code widthK} values from {@code src[off]}) into row {@code row}. */
    void writeK(int row, float[] src, int off);

    /**
     * Stores K values {@code [from, from + n)} of row {@code row} from {@code src[off, off + n)}, leaving the rest of
     * the row's stored bits unchanged (a {@code ggml_set_rows} into a view of part of the row). For a block type,
     * {@code from} and {@code n} are multiples of its block size.
     *
     * @throws IndexOutOfBoundsException for a row or range outside the cache
     * @throws IllegalArgumentException  for a block-type range that splits a block
     */
    void writeKSlice(int row, int from, int n, float[] src, int off);

    /**
     * Stores a full V row ({@code widthV} values from {@code src[off]}).
     *
     * @throws UnsupportedOperationException for a latent cache, whose V is part of the K row
     */
    void writeV(int row, float[] src, int off);

    /**
     * The raw int8 view of this cache, or null when its K and V rows are not both {@code INT8_PER_TOKEN_HEAD}. The
     * fused int8 attention reads a key or value row in its stored format through the view instead of dequantizing it
     * first.
     */
    default KvInt8Rows int8Rows() {
        return null;
    }

    /**
     * Moves {@code n} whole rows starting at {@code srcRow} to the same-width run starting at {@code dstRow}
     * ({@code System.arraycopy} per row, K and V), the host form of a {@code PageCopy}: stored bits move unchanged,
     * as the device copy kernel moves them.
     *
     * @throws IndexOutOfBoundsException for a source or destination row outside the cache
     */
    default void copyRows(int srcRow, int dstRow, int n) {
        throw new UnsupportedOperationException("row copies run on the page caches, not on "
                + getClass().getSimpleName());
    }
}
