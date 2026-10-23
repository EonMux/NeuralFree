package com.eonmux.neuralfree.gguf.gpu.forward;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;

import java.util.Objects;

/**
 * A named float view into a device scratch buffer: elements {@code [offset, offset + floats)} of {@link #buffer()}.
 * Activations are row-major, {@code [t][width]}, starting at {@link #offset()}.
 *
 * <p>Immutable. The buffer belongs to the {@link GpuScratch} that issued the slot; offsets are in float elements and
 * fit the {@code int} offset arguments of the LLM kernels.</p>
 */
public final class ScratchSlot {

    private final String name;
    private final GpuBuffer buffer;
    private final int offset;
    private final long floats;

    private ScratchSlot(String name, GpuBuffer buffer, int offset, long floats) {
        this.name = name;
        this.buffer = buffer;
        this.offset = offset;
        this.floats = floats;
    }

    /**
     * A view of {@code floats} elements starting at element {@code offset} of {@code buffer}.
     *
     * @throws IllegalArgumentException for an empty name, a negative offset or size, an offset beyond the kernels'
     *                                  {@code int} range, or a view that does not fit the buffer
     */
    public static ScratchSlot of(String name, GpuBuffer buffer, long offset, long floats) {
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("scratch slot needs a name");
        }
        Objects.requireNonNull(buffer, "buffer");
        if (offset < 0 || floats < 0) {
            throw new IllegalArgumentException("scratch slot " + name + ": offset " + offset + " and size " + floats
                    + " must be non-negative");
        }
        if (offset > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("scratch slot " + name + ": offset " + offset
                    + " exceeds the kernels' int offset range");
        }
        long end = offset + floats;
        if (end * Float.BYTES > buffer.sizeInBytes()) {
            throw new IllegalArgumentException("scratch slot " + name + ": elements [" + offset + ", " + end
                    + ") exceed the buffer of " + buffer.sizeInBytes() + " bytes");
        }
        return new ScratchSlot(name, buffer, (int) offset, floats);
    }

    public String name() { return name; }
    public GpuBuffer buffer() { return buffer; }
    /** First element of the view in {@link #buffer()}. */
    public int offset() { return offset; }
    /** Capacity of the view in floats. */
    public long floats() { return floats; }

    /**
     * Sub-view of {@code count} elements starting {@code from} elements into this view.
     *
     * @throws IllegalArgumentException when the sub-view does not fit this view
     */
    public ScratchSlot slice(long from, long count) {
        if (from < 0 || count < 0 || from + count > floats) {
            throw new IllegalArgumentException("scratch slot " + name + ": slice [" + from + ", " + (from + count)
                    + ") exceeds its " + floats + " floats");
        }
        return of(name, buffer, offset + from, count);
    }

    /**
     * Checks that the view holds at least {@code needed} floats.
     *
     * @return this slot
     * @throws IllegalStateException naming the slot, its capacity and the need when it is too small
     */
    public ScratchSlot require(long needed) {
        if (needed > floats) {
            throw new IllegalStateException("scratch slot " + name + " holds " + floats + " floats, " + needed
                    + " are needed");
        }
        return this;
    }

    @Override
    public String toString() {
        return "ScratchSlot{" + name + " @" + offset + " x" + floats + '}';
    }
}
