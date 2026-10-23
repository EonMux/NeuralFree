package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Abstraction over tensor data storage — single-file or multi-shard GGUF.
 *
 * <p>All {@code tensorOffset} parameters are offsets relative to the logical start
 * of the tensor data region (i.e. {@link GGUFTensorInfo#offset()} directly,
 * without adding any base offset).</p>
 */
public interface WeightStore extends AutoCloseable {

    /**
     * Reads {@code size} bytes starting at {@code tensorOffset} into {@code dst}.
     *
     * @param tensorOffset logical offset from the start of the tensor data region
     * @param size         number of bytes to read
     * @param dst          destination array (must have length ≥ {@code size})
     */
    void readTensor(long tensorOffset, int size, byte[] dst) throws IOException;

    /**
     * Reads {@code size} bytes starting at {@code tensorOffset} into {@code dst}
     * at array position {@code dstOffset}.
     */
    void readTensor(long tensorOffset, int size, byte[] dst, int dstOffset) throws IOException;

    /**
     * Reads {@code size} bytes starting at {@code tensorOffset} into a direct
     * {@link ByteBuffer} (zero-copy GPU staging path).
     */
    void readTensorDirect(long tensorOffset, int size, ByteBuffer dst) throws IOException;

    @Override
    void close() throws IOException;
}
