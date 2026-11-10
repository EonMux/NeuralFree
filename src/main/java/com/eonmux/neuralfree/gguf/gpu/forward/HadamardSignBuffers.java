package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device sign vectors of the Hadamard transforms a {@link GpuOps} runs: one read-only buffer per
 * {@link HadamardSpec#signsName()}, uploaded on first use. Owned by one {@link GpuOps}, which closes it after its
 * queue has drained; not thread-safe.
 */
final class HadamardSignBuffers {

    private final ComputeBackend backend;
    private final ComputeQueue queue;
    private final Map<String, float[]> values = new HashMap<>();
    private final Map<String, GpuBuffer> buffers = new HashMap<>();
    private final Map<HadamardSpec, GpuBuffer> bySpec = new IdentityHashMap<>();

    HadamardSignBuffers(ComputeBackend backend, ComputeQueue queue) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.queue = Objects.requireNonNull(queue, "queue");
    }

    /**
     * The sign vector of {@code h}. Transforms seen before resolve by identity without touching their sign arrays.
     *
     * @throws IllegalArgumentException when two transforms share a sign vector name with different signs
     */
    GpuBuffer of(HadamardSpec h) {
        GpuBuffer cached = bySpec.get(h);
        if (cached != null) {
            return cached;
        }
        String name = h.signsName();
        float[] signs = h.signs();
        float[] known = values.get(name);
        if (known != null) {
            if (!Arrays.equals(known, signs)) {
                throw new IllegalArgumentException("two Hadamard transforms share the sign vector name " + name
                        + " with different signs");
            }
            GpuBuffer b = buffers.get(name);
            bySpec.put(h, b);
            return b;
        }
        GpuBuffer b = backend.allocateBuffer((long) signs.length * Float.BYTES, true);
        try {
            GpuEvent e = queue.enqueueWriteBuffer(b, 0, (long) signs.length * Float.BYTES, signs);
            if (e != null) {
                e.close();
            }
        } catch (RuntimeException ex) {
            b.close();
            throw ex;
        }
        values.put(name, signs);
        buffers.put(name, b);
        bySpec.put(h, b);
        return b;
    }

    /** Hands over the uploaded buffers for closing and forgets every transform. */
    List<GpuBuffer> release() {
        List<GpuBuffer> owned = new ArrayList<>(buffers.values());
        buffers.clear();
        values.clear();
        bySpec.clear();
        return owned;
    }
}
