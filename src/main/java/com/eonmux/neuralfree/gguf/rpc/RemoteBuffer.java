package com.eonmux.neuralfree.gguf.rpc;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

/**
 * A device buffer of a remote backend: the id the server allocated it under and its size. Closing it releases the
 * server-side buffer once; closing again has no effect.
 */
final class RemoteBuffer implements GpuBuffer {

    private final RemoteComputeBackend backend;
    private final long id;
    private final long sizeInBytes;
    private boolean closed;

    RemoteBuffer(RemoteComputeBackend backend, long id, long sizeInBytes) {
        this.backend = backend;
        this.id = id;
        this.sizeInBytes = sizeInBytes;
    }

    long id() {
        return id;
    }

    @Override
    public long sizeInBytes() {
        return sizeInBytes;
    }

    @Override
    public long bufferId() {
        return id;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        backend.releaseBuffer(this);
    }

    @Override
    public String toString() {
        return "RemoteBuffer{" + id + ", " + sizeInBytes + " bytes}";
    }
}
