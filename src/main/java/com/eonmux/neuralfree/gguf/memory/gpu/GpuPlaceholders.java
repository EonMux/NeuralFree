package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;

import java.util.Objects;

/**
 * Dummy buffers bound to optional kernel arguments: an LLM kernel never receives a null
 * buffer, so an absent tensor is bound to a placeholder and disabled by its {@code has_x} flag.
 * <ul>
 *   <li>Read-only placeholders ({@link #readOnlyFloats()}, {@link #readOnlyInts()}) hold {@value #ELEMENTS}
 *       zeros and are allocated read-only; they are bound only to read-only bindings, so no backend ever sees a
 *       write to them.</li>
 *   <li>The FP8 KV-scale placeholder ({@link #kvScalesOnes()}) holds ones: the flash and FP8 store kernels read
 *       their KV-cache scales unconditionally, so an unscaled cache binds the identity scales.</li>
 *   <li>Scratch placeholders ({@link #scratchFloats()}, {@link #scratchInts()}) are writable and bound to optional
 *       outputs whose kernel flag disables the write; their contents are meaningless.</li>
 * </ul>
 * One instance per model; buffers live until {@link #close()}.
 */
public final class GpuPlaceholders implements AutoCloseable {

    /** Elements in every placeholder. */
    public static final int ELEMENTS = 4;

    /** K then V scale entries of {@link #kvScalesOnes()} (1024 KV heads). */
    public static final int KV_SCALE_ELEMENTS = 2048;

    private static final long BYTES = (long) ELEMENTS * Integer.BYTES;
    private static final long KV_SCALE_BYTES = (long) KV_SCALE_ELEMENTS * Integer.BYTES;

    /** Device bytes one allocation of the placeholders takes: the five buffers of {@link #allocate}. */
    public static final long TOTAL_BYTES = 4 * BYTES + KV_SCALE_BYTES;

    private final GpuBuffer readOnlyFloats;
    private final GpuBuffer readOnlyInts;
    private final GpuBuffer kvScalesOnes;
    private final GpuBuffer scratchFloats;
    private final GpuBuffer scratchInts;
    private boolean closed;

    private GpuPlaceholders(GpuBuffer readOnlyFloats, GpuBuffer readOnlyInts, GpuBuffer kvScalesOnes,
                            GpuBuffer scratchFloats, GpuBuffer scratchInts) {
        this.readOnlyFloats = readOnlyFloats;
        this.readOnlyInts = readOnlyInts;
        this.kvScalesOnes = kvScalesOnes;
        this.scratchFloats = scratchFloats;
        this.scratchInts = scratchInts;
    }

    /**
     * Allocates the placeholders, the read-only ones zero-filled and {@link #kvScalesOnes()} filled with 1.0, on
     * {@code queue}.
     *
     * @throws RuntimeException from the backend when an allocation fails; nothing stays allocated then
     */
    public static GpuPlaceholders allocate(ComputeBackend backend, ComputeQueue queue) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        GpuBuffer[] made = new GpuBuffer[5];
        try {
            made[0] = backend.allocateBuffer(BYTES, true);
            made[1] = backend.allocateBuffer(BYTES, true);
            made[2] = backend.allocateBuffer(KV_SCALE_BYTES, true);
            made[3] = backend.allocateBuffer(BYTES, false);
            made[4] = backend.allocateBuffer(BYTES, false);
            for (int i = 0; i < made.length; i++) {
                if (i == 2) {
                    float[] ones = new float[KV_SCALE_ELEMENTS];
                    java.util.Arrays.fill(ones, 1.0f);
                    GpuEvent e = queue.enqueueWriteBuffer(made[i], 0, KV_SCALE_BYTES, ones);
                    if (e != null) {
                        e.close();
                    }
                } else {
                    queue.enqueueFillBuffer(made[i], 0, BYTES, 0);
                }
            }
            return new GpuPlaceholders(made[0], made[1], made[2], made[3], made[4]);
        } catch (RuntimeException e) {
            for (GpuBuffer b : made) {
                if (b != null) {
                    b.close();
                }
            }
            throw e;
        }
    }

    /** Read-only {@code float[4]} of zeros. */
    public GpuBuffer readOnlyFloats() {
        return live(readOnlyFloats);
    }

    /** Read-only {@code int[4]} of zeros. */
    public GpuBuffer readOnlyInts() {
        return live(readOnlyInts);
    }

    /**
     * Read-only {@code float[2 * 1024]} of ones, the identity FP8 KV-cache scales (n_head_kv K values then
     * n_head_kv V values) for a cache without checkpoint scales.
     */
    public GpuBuffer kvScalesOnes() {
        return live(kvScalesOnes);
    }

    /** Writable {@code float[4]} for disabled optional outputs. */
    public GpuBuffer scratchFloats() {
        return live(scratchFloats);
    }

    /** Writable {@code int[4]} for disabled optional outputs. */
    public GpuBuffer scratchInts() {
        return live(scratchInts);
    }

    private GpuBuffer live(GpuBuffer b) {
        if (closed) {
            throw new IllegalStateException("GPU placeholders are closed");
        }
        return b;
    }

    /** Releases every placeholder. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        readOnlyFloats.close();
        readOnlyInts.close();
        kvScalesOnes.close();
        scratchFloats.close();
        scratchInts.close();
    }
}
