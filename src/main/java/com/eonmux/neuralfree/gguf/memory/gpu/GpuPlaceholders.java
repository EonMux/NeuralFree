package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Objects;

/**
 * Dummy buffers bound to optional kernel arguments: an LLM kernel never receives a null
 * buffer, so an absent tensor is bound to a placeholder and disabled by its {@code has_x} flag.
 * <ul>
 *   <li>Read-only placeholders ({@link #readOnlyFloats()}, {@link #readOnlyInts()}) hold {@value #ELEMENTS}
 *       zeros and are allocated read-only; they are bound only to read-only bindings, so no backend ever sees a
 *       write to them.</li>
 *   <li>Scratch placeholders ({@link #scratchFloats()}, {@link #scratchInts()}) are writable and bound to optional
 *       outputs whose kernel flag disables the write; their contents are meaningless.</li>
 * </ul>
 * One instance per model; buffers live until {@link #close()}.
 */
public final class GpuPlaceholders implements AutoCloseable {

    /** Elements in every placeholder. */
    public static final int ELEMENTS = 4;

    private static final long BYTES = (long) ELEMENTS * Integer.BYTES;

    private final GpuBuffer readOnlyFloats;
    private final GpuBuffer readOnlyInts;
    private final GpuBuffer scratchFloats;
    private final GpuBuffer scratchInts;
    private boolean closed;

    private GpuPlaceholders(GpuBuffer readOnlyFloats, GpuBuffer readOnlyInts, GpuBuffer scratchFloats,
                            GpuBuffer scratchInts) {
        this.readOnlyFloats = readOnlyFloats;
        this.readOnlyInts = readOnlyInts;
        this.scratchFloats = scratchFloats;
        this.scratchInts = scratchInts;
    }

    /**
     * Allocates the four placeholders and zero-fills them on {@code queue}.
     *
     * @throws RuntimeException from the backend when an allocation fails; nothing stays allocated then
     */
    public static GpuPlaceholders allocate(ComputeBackend backend, ComputeQueue queue) {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(queue, "queue");
        GpuBuffer[] made = new GpuBuffer[4];
        try {
            made[0] = backend.allocateBuffer(BYTES, true);
            made[1] = backend.allocateBuffer(BYTES, true);
            made[2] = backend.allocateBuffer(BYTES, false);
            made[3] = backend.allocateBuffer(BYTES, false);
            for (GpuBuffer b : made) {
                queue.enqueueFillBuffer(b, 0, BYTES, 0);
            }
            return new GpuPlaceholders(made[0], made[1], made[2], made[3]);
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

    /** Releases the four buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        readOnlyFloats.close();
        readOnlyInts.close();
        scratchFloats.close();
        scratchInts.close();
    }
}
