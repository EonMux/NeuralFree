package com.eonmux.neuralfree.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * Byte regions of a model's tensors for host-side readers ({@code CpuWeights}): each region is mapped in place or
 * copied into a buffer the reader keeps. Implementations are thread-safe.
 */
public interface TensorRegions extends AutoCloseable {

    /**
     * The {@code length} bytes of {@code info} starting {@code byteOff} bytes into it, in a buffer ordered
     * little-endian and positioned at 0.
     *
     * @throws IOException when the range lies outside the tensor's stored data or the bytes cannot be mapped or read;
     *                     the message names the tensor
     */
    ByteBuffer region(GGUFTensorInfo info, long byteOff, long length) throws IOException;

    /** The model the regions belong to, for messages. */
    String description();

    /** Releases the files; buffers handed out earlier stay readable until they are no longer referenced. */
    @Override
    void close() throws IOException;
}
