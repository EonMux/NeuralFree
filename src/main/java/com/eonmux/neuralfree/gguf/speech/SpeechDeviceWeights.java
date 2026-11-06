package com.eonmux.neuralfree.gguf.speech;

import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeight;
import com.eonmux.neuralfree.gguf.gpu.weights.GpuWeightUploader;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device view of an opened speech generator file (qwen3tts, pockettts or lfm3a): the uploaded matrices and
 * embedding tables (through {@link GpuWeightUploader}, so they keep their stored type) and the F32 vectors and
 * convolution kernels the speech kernels read, copied once into device buffers. Sessions look weights up by their
 * GGUF names.
 *
 * <p>Owned by the clip's device binding; {@link #close} releases every buffer this class allocated, and the owner
 * releases the uploader. Buffers are shared by every graph of one clip.
 */
final class SpeechDeviceWeights implements AutoCloseable {

    private final String family;
    private final GpuOps ops;
    private final ComputeQueue queue;
    private final GpuWeightUploader uploader;
    private final List<GpuBuffer> owned = new ArrayList<>();
    private final Map<String, GpuBuffer> f32 = new HashMap<>();
    private final Map<String, Integer> f32Lengths = new HashMap<>();
    private final Map<String, GpuBuffer> f32RowCache = new HashMap<>();
    private final Map<String, Long> f32RowLengths = new HashMap<>();
    private boolean closed;

    /**
     * @param family the generator family the errors name ({@code qwen3tts}, {@code pockettts}, {@code lfm3a})
     */
    SpeechDeviceWeights(String family, GpuOps ops, GpuWeightUploader uploader) {
        this.family = Objects.requireNonNull(family, "family");
        this.ops = Objects.requireNonNull(ops, "ops");
        this.queue = ops.queue();
        this.uploader = Objects.requireNonNull(uploader, "uploader");
    }

    GpuOps ops() {
        return ops;
    }

    ComputeQueue queue() {
        return queue;
    }

    /** The uploaded matrix or embedding table {@code name} (its stored type on the device). */
    GpuWeight weight(String name) {
        return uploader.upload(name);
    }

    /**
     * {@code name} as a single F32 device vector of {@code values.length} elements.
     *
     * @throws IllegalArgumentException when the same name was uploaded with another length
     */
    GpuBuffer f32(String name, float[] values) {
        ensureOpen();
        Objects.requireNonNull(values, "values");
        GpuBuffer existing = f32.get(name);
        if (existing != null) {
            int length = f32Lengths.get(name);
            if (length != values.length) {
                throw new IllegalArgumentException("device vector " + name + " holds " + length + " floats, "
                        + values.length + " are bound");
            }
            return existing;
        }
        GpuBuffer buffer = allocate((long) values.length * Float.BYTES);
        writeFloats(buffer, values);
        f32.put(name, buffer);
        f32Lengths.put(name, values.length);
        return buffer;
    }

    /** {@code name} as the F32 weight vector the norm kernels take. */
    GpuWeight f32Weight(String name, float[] values) {
        GpuBuffer buffer = f32(name, values);
        return GpuWeight.of(name, GGMLType.F32, new long[]{values.length}, buffer);
    }

    /**
     * {@code name} as the F32 matrix {@code rows} x {@code cols} the matvec kernels take. The values are ggml order
     * ({@code ne[0]} fastest), the order {@link com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights} returns.
     */
    GpuWeight f32Matrix(String name, float[] values, long rows, long cols) {
        if (Math.multiplyExact(rows, cols) != values.length) {
            throw new IllegalArgumentException("device matrix " + name + " holds " + values.length + " values, its "
                    + rows + " x " + cols + " shape needs " + Math.multiplyExact(rows, cols));
        }
        GpuBuffer buffer = f32(name, values);
        return GpuWeight.of(name, GGMLType.F32, new long[]{cols, rows}, buffer);
    }

    /** A new zero-filled weight buffer of {@code floats} elements, released with this view. */
    GpuBuffer buffer(String what, long floats) {
        ensureOpen();
        if (floats <= 0) {
            throw new IllegalArgumentException("device buffer " + what + " needs a positive size, got " + floats);
        }
        GpuBuffer buffer = allocate(Math.multiplyExact(floats, (long) Float.BYTES));
        queue.enqueueFillBuffer(buffer, 0, buffer.sizeInBytes(), 0);
        return buffer;
    }

    /**
     * A zero-filled buffer the caller owns and releases (a session's activations and state), unlike
     * {@link #f32(String, float[])}'s weights, which live until the clip's device binding closes.
     */
    GpuBuffer allocate(String what, long floats) {
        ensureOpen();
        if (floats <= 0) {
            throw new IllegalArgumentException("device buffer " + what + " needs a positive size, got " + floats);
        }
        GpuBuffer buffer = ops.kernels().backend().allocateBuffer(Math.multiplyExact(floats, (long) Float.BYTES));
        queue.enqueueFillBuffer(buffer, 0, buffer.sizeInBytes(), 0);
        return buffer;
    }

    /**
     * {@code w}'s rows as F32: the tensor's own buffer when it is a single F32 shard, else a device decode through
     * the embedding-lookup kernels, decoded once per tensor and reused by every session. A decoded buffer is owned
     * by this view like the other weight buffers; a single F32 shard stays owned by the uploader.
     */
    GpuBuffer f32Rows(GpuWeight w) {
        ensureOpen();
        if (w.type() == GGMLType.F32 && w.isSingleShard()) {
            return w.shard(0);
        }
        long rows = w.rows();
        long cols = w.cols();
        long elements = Math.multiplyExact(rows, cols);
        if (elements > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(family + ": tensor " + w.name() + " holds " + elements
                    + " values, too many for one F32 device buffer");
        }
        GpuBuffer cached = f32RowCache.get(w.name());
        if (cached != null) {
            if (f32RowLengths.get(w.name()).longValue() != elements) {
                throw new IllegalArgumentException(family + ": tensor " + w.name() + " was decoded to "
                        + f32RowLengths.get(w.name()) + " values, " + elements + " are requested");
            }
            return cached;
        }
        GpuBuffer out = buffer("f32." + w.name(), elements);
        int chunk = 4096;
        for (long start = 0; start < rows; start += chunk) {
            int n = (int) Math.min((long) chunk, rows - start);
            int[] ids = new int[n];
            for (int i = 0; i < n; i++) {
                ids[i] = (int) (start + i);
            }
            ops.embed(w, ids, out, (int) (start * cols));
        }
        f32RowCache.put(w.name(), out);
        f32RowLengths.put(w.name(), elements);
        return out;
    }

    /** Uploads {@code values} into {@code dst} from element 0. */
    void writeFloats(GpuBuffer dst, float[] values) {
        ensureOpen();
        closeEvent(queue.enqueueWriteBuffer(dst, 0, (long) values.length * Float.BYTES, values));
    }

    /** Uploads {@code values} (one int each) into {@code dst} from element 0. */
    void writeInts(GpuBuffer dst, int[] values) {
        ensureOpen();
        java.nio.ByteBuffer data = java.nio.ByteBuffer.allocate(values.length * Integer.BYTES)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int v : values) {
            data.putInt(v);
        }
        closeEvent(queue.enqueueWriteBuffer(dst, 0, (long) values.length * Integer.BYTES, data.array()));
    }

    /** Reads {@code n} floats from {@code src} into {@code dst}, waiting for the queue. */
    void readFloats(GpuBuffer src, float[] dst, int n) {
        ensureOpen();
        closeEvent(queue.enqueueReadBuffer(src, 0, (long) n * Float.BYTES, dst));
        queue.finish();
    }

    /**
     * Requires the uploaded tensor {@code name} to have {@code rows} rows of {@code cols} values.
     *
     * @throws IllegalArgumentException when its device shape differs
     */
    void requireShape(String name, long rows, long cols) {
        GpuWeight w = weight(name);
        if (w.rows() != rows || w.cols() != cols) {
            throw new IllegalArgumentException(family + " generator tensor " + name + " is " + w.rows() + " x "
                    + w.cols() + " on the device, the graph needs " + rows + " x " + cols);
        }
    }

    private GpuBuffer allocate(long bytes) {
        GpuBuffer buffer = ops.kernels().backend().allocateBuffer(bytes);
        owned.add(buffer);
        return buffer;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("the " + family + " device weights are closed");
        }
    }

    private static void closeEvent(GpuEvent event) {
        if (event != null) {
            event.close();
        }
    }

    /** Releases every buffer this class allocated. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        RuntimeException failure = null;
        for (GpuBuffer buffer : owned) {
            try {
                buffer.close();
            } catch (RuntimeException e) {
                failure = failure == null ? e : failure;
            }
        }
        owned.clear();
        f32.clear();
        f32Lengths.clear();
        f32RowCache.clear();
        f32RowLengths.clear();
        if (failure != null) {
            throw failure;
        }
    }

    @Override
    public String toString() {
        return "SpeechDeviceWeights{" + family + ", vectors=" + f32.size() + ", buffers=" + owned.size() + '}';
    }
}
