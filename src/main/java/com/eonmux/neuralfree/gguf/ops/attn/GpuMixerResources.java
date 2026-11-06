package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.utils.Closeables;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The prepared dispatches and device buffers one attention mixer owns: created while it binds and released
 * together when it closes (or when the bind fails). Not thread-safe.
 */
final class GpuMixerResources {

    private final GpuOps ops;
    private final ComputeBackend backend;
    private final List<PreparedDispatch> prepared = new ArrayList<>();
    private final List<GpuBuffer> owned = new ArrayList<>();

    GpuMixerResources(GpuOps ops, ComputeBackend backend) {
        this.ops = Objects.requireNonNull(ops, "ops");
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    /** A dispatch of {@code kernel} over the given grid with its bound arguments, released by {@link #release}. */
    PreparedDispatch prepare(String kernel, long[] global, long[] local, Object[] args) {
        PreparedDispatch d = ops.prepare(kernel, global, local, args);
        prepared.add(d);
        return d;
    }

    /** A device buffer of {@code bytes} bytes, released by {@link #release}. */
    GpuBuffer allocate(long bytes) {
        GpuBuffer b = backend.allocateBuffer(bytes);
        owned.add(b);
        return b;
    }

    /**
     * Closes every prepared dispatch, then every owned buffer, and forgets them.
     *
     * @param failure the failure of an earlier release step, or null
     * @return {@code failure}, or the first failure of this release when {@code failure} is null
     */
    RuntimeException release(RuntimeException failure) {
        RuntimeException f = Closeables.closeAll(prepared, failure);
        prepared.clear();
        f = Closeables.closeAll(owned, f);
        owned.clear();
        return f;
    }
}
