package com.eonmux.neuralfree.gguf.memory.cpu;

/**
 * Raw int8 view of a KV cache: the stored signed codes of K and V and their per-(row, KV head) scales, without the
 * dequantization {@link KvAccessor#readK} and {@link KvAccessor#readV} apply. The fused int8 attention kernel reads
 * through this view, so a key row is consumed in its stored format and the exact int8 dot product with the
 * quantized query replaces the float product of the dequantized rows.
 *
 * <p>Only a cache whose K and V dtypes are {@code INT8_PER_TOKEN_HEAD} has this view
 * ({@link KvAccessor#int8Rows()}); a row holds one signed byte per value ({@code -127..127}), and the value of code
 * {@code c} is {@code c * scale(row, head)}. Rows are addressed as in {@link KvAccessor} and carry their positions in
 * the memory, not in the view.</p>
 */
public interface KvInt8Rows {

    /** KV heads per row. */
    int nHeadKv();

    int headDimK();

    int headDimV();

    /**
     * Reads the {@code headDimK} codes of K head {@code head} of {@code row} into {@code dst[off..]}.
     *
     * @throws IndexOutOfBoundsException for a row, head or destination range outside the cache
     */
    void readKCodes(int row, int head, byte[] dst, int off);

    /**
     * Reads the {@code headDimV} codes of V head {@code head} of {@code row} into {@code dst[off..]}.
     *
     * @throws IndexOutOfBoundsException for a row, head or destination range outside the cache
     */
    void readVCodes(int row, int head, byte[] dst, int off);

    /** The stored K scale of {@code head} in {@code row}, positive. */
    float kScale(int row, int head);

    /** The stored V scale of {@code head} in {@code row}, positive. */
    float vScale(int row, int head);
}
