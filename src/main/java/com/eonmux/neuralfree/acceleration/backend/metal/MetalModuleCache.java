package com.eonmux.neuralfree.acceleration.backend.metal;

import com.eonmux.neuralfree.acceleration.NotThreadSafe;
import com.eonmux.neuralfree.acceleration.metal.MetalNative;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Compiled Metal libraries of one backend, keyed by program key.
 *
 * <p>A {@code MTLLibrary} is the compilation of one MSL source (for a kernel library, one
 * {@code .metal} file). Several kernels can live in one file, so the library is compiled once per
 * program key and the pipeline of each kernel is built from it. Libraries are released when the
 * backend closes.</p>
 */
@NotThreadSafe("Guarded internally; a backend serialises compilation through KernelSet's load lock.")
final class MetalModuleCache {

    private final MetalComputeBackend backend;
    private final Map<String, Long> libraries = new HashMap<>();
    private final List<Long> released = new ArrayList<>();
    private boolean closed;

    MetalModuleCache(MetalComputeBackend backend) {
        if (backend == null) {
            throw new IllegalArgumentException("Metal backend must not be null");
        }
        this.backend = backend;
    }

    /**
     * The library of {@code programKey}, compiled on first use.
     *
     * @param programKey unique id of the source and its options
     * @param source     MSL text
     * @param options    {@code -DNAME=VALUE} compile options
     * @param kernelName kernel the source must provide, for the error message
     * @throws IllegalStateException when the source does not compile
     */
    long library(String programKey, Supplier<String> source, String options, String kernelName) {
        synchronized (this) {
            if (closed) {
                throw new IllegalStateException("Metal kernel " + kernelName + " was used after its backend"
                        + " was closed");
            }
            Long cached = libraries.get(programKey);
            if (cached != null) {
                return cached;
            }
        }
        String text = source.get();
        long handle = MetalNative.newLibrary(backend.device().getIndex(), text, options == null ? "" : options);
        synchronized (this) {
            if (closed) {
                MetalNative.releaseObject(handle);
                throw new IllegalStateException("Metal kernel " + kernelName + " was used after its backend"
                        + " was closed");
            }
            Long raced = libraries.get(programKey);
            if (raced != null) {
                MetalNative.releaseObject(handle);
                return raced;
            }
            libraries.put(programKey, handle);
            return handle;
        }
    }

    /** Releases every compiled library. Idempotent. */
    void close() {
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            released.addAll(libraries.values());
            libraries.clear();
        }
        for (Long library : released) {
            MetalNative.releaseObject(library);
        }
    }

    /** Number of compiled libraries, for diagnostics. */
    int size() {
        synchronized (this) {
            return libraries.size();
        }
    }
}
