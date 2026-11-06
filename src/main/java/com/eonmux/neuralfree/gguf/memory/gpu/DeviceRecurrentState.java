package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.ComputeBackend;
import com.eonmux.neuralfree.acceleration.backend.ComputeQueue;
import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.acceleration.backend.GpuEvent;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Device recurrent state of one layer: one zero-initialised F32 buffer per state tensor (single sequence, row 0),
 * plus, when the memory plan reserves snapshots, one shadow buffer per tensor allocated with it, which
 * {@link #snapshot(ComputeQueue)} fills and {@link #restore(ComputeQueue)} copies back with device-to-device copies.
 */
public final class DeviceRecurrentState implements GpuRecurrentState, AutoCloseable {

    private final int layer;
    private final List<LayerMemoryPlan.StateTensor> tensors;
    private final Map<String, GpuBuffer> buffers;
    private final Map<String, GpuBuffer> shadows;
    private boolean saved;
    private boolean closed;

    private DeviceRecurrentState(int layer, List<LayerMemoryPlan.StateTensor> tensors, Map<String, GpuBuffer> buffers,
                                 Map<String, GpuBuffer> shadows) {
        this.layer = layer;
        this.tensors = tensors;
        this.buffers = buffers;
        this.shadows = shadows;
    }

    /**
     * Allocates and zero-fills the state tensors of a layer and, with {@code snapshots}, a shadow of each.
     *
     * @param snapshots allocate the shadows {@link #snapshot(ComputeQueue)} needs
     *                  ({@code MemoryPlan#snapshotsReserved})
     * @throws IllegalArgumentException for an empty tensor list or duplicate tensor names
     */
    static DeviceRecurrentState allocate(ComputeBackend backend, ComputeQueue queue, int layer,
                                         List<LayerMemoryPlan.StateTensor> tensors, boolean snapshots) {
        Objects.requireNonNull(tensors, "tensors");
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("layer " + layer + " has no recurrent state tensors");
        }
        Map<String, GpuBuffer> buffers = new LinkedHashMap<>();
        Map<String, GpuBuffer> shadows = new LinkedHashMap<>();
        try {
            for (LayerMemoryPlan.StateTensor t : tensors) {
                if (buffers.containsKey(t.name())) {
                    throw new IllegalArgumentException("layer " + layer + " declares state tensor " + t.name() + " twice");
                }
                buffers.put(t.name(), DeviceModelMemory.allocateZeroed(backend, queue, t.bytes(),
                        "recurrent state " + t.name() + " of layer " + layer));
                if (snapshots) {
                    shadows.put(t.name(), DeviceModelMemory.allocateZeroed(backend, queue, t.bytes(),
                            "snapshot of state " + t.name() + " of layer " + layer));
                }
            }
        } catch (RuntimeException e) {
            for (GpuBuffer b : buffers.values()) {
                b.close();
            }
            for (GpuBuffer b : shadows.values()) {
                b.close();
            }
            throw e;
        }
        return new DeviceRecurrentState(layer, Collections.unmodifiableList(new ArrayList<>(tensors)), buffers,
                shadows);
    }

    @Override
    public int layer() {
        return layer;
    }

    @Override
    public List<LayerMemoryPlan.StateTensor> tensors() {
        return tensors;
    }

    @Override
    public GpuBuffer tensor(String name) {
        checkOpen();
        GpuBuffer b = buffers.get(name);
        if (b == null) {
            throw new IllegalArgumentException("layer " + layer + " has no state tensor '" + name + "' (has "
                    + buffers.keySet() + ")");
        }
        return b;
    }

    /** Zeroes every state tensor and forgets the snapshot. */
    void reset(ComputeQueue queue) {
        checkOpen();
        for (GpuBuffer b : buffers.values()) {
            queue.enqueueFillBuffer(b, 0, b.sizeInBytes(), 0);
        }
        saved = false;
    }

    /**
     * Copies every state tensor into its shadow.
     *
     * @throws IllegalStateException when the state was allocated without shadows
     */
    void snapshot(ComputeQueue queue) {
        checkOpen();
        if (shadows.isEmpty()) {
            throw new IllegalStateException("recurrent state of layer " + layer + " has no snapshot buffers");
        }
        for (Map.Entry<String, GpuBuffer> e : buffers.entrySet()) {
            queue.enqueueCopyBuffer(e.getValue(), 0, shadows.get(e.getKey()), 0, e.getValue().sizeInBytes());
        }
        saved = true;
    }

    /**
     * Copies every shadow back into its state tensor.
     *
     * @throws IllegalStateException when {@link #snapshot(ComputeQueue)} has not run since the last reset
     */
    void restore(ComputeQueue queue) {
        checkOpen();
        if (!saved) {
            throw new IllegalStateException("layer " + layer + " has no recurrent-state snapshot");
        }
        for (Map.Entry<String, GpuBuffer> e : buffers.entrySet()) {
            queue.enqueueCopyBuffer(shadows.get(e.getKey()), 0, e.getValue(), 0, e.getValue().sizeInBytes());
        }
    }

    /**
     * Copies state slot {@code seq}'s row of every tensor into {@code rows} (one row per tensor, in plan order);
     * the reads enqueue and land when the queue finishes.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the tensor's slots
     */
    void readSlot(ComputeQueue queue, int seq, List<float[]> rows) {
        checkOpen();
        for (LayerMemoryPlan.StateTensor t : tensors) {
            long floats = t.floats();
            Objects.checkIndex(seq, t.slots());
            float[] row = new float[(int) floats];
            GpuEvent e = queue.enqueueReadBuffer(buffers.get(t.name()), seq * floats * Float.BYTES,
                    floats * Float.BYTES, row);
            if (e != null) {
                e.close();
            }
            rows.add(row);
        }
    }

    /**
     * Writes {@code copy}'s rows {@code [first, first + tensors)} into state slot {@code seq} of every tensor.
     *
     * @throws IllegalArgumentException when a row does not match its tensor's width
     * @throws IndexOutOfBoundsException when {@code seq} is outside the tensor's slots
     */
    void writeSlot(ComputeQueue queue, int seq, RecurrentStateCopy copy, int first) {
        checkOpen();
        for (int i = 0; i < tensors.size(); i++) {
            LayerMemoryPlan.StateTensor t = tensors.get(i);
            long floats = t.floats();
            Objects.checkIndex(seq, t.slots());
            if (copy.floats(first + i) != floats) {
                throw new IllegalArgumentException("state copy row " + (first + i) + " holds " + copy.floats(first + i)
                        + " floats, tensor " + t.name() + " of layer " + layer + " keeps " + floats);
            }
            float[] row = new float[copy.floats(first + i)];
            copy.copyInto(first + i, row, 0);
            GpuEvent e = queue.enqueueWriteBuffer(buffers.get(t.name()), seq * floats * Float.BYTES,
                    floats * Float.BYTES, row);
            if (e != null) {
                e.close();
            }
        }
    }

    /** Zeroes state slot {@code seq}'s row of every tensor. */
    void clearSlot(ComputeQueue queue, int seq) {
        checkOpen();
        for (LayerMemoryPlan.StateTensor t : tensors) {
            Objects.checkIndex(seq, t.slots());
            queue.enqueueFillBuffer(buffers.get(t.name()), seq * t.floats() * Float.BYTES,
                    t.floats() * Float.BYTES, 0);
        }
    }

    private void checkOpen() {
        if (closed) {
            throw new IllegalStateException("recurrent state of layer " + layer + " is closed");
        }
    }

    /** Releases the state and shadow buffers. Idempotent. */
    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        for (GpuBuffer b : buffers.values()) {
            b.close();
        }
        for (GpuBuffer b : shadows.values()) {
            b.close();
        }
        shadows.clear();
    }

    @Override
    public String toString() {
        return "DeviceRecurrentState{il=" + layer + ", " + tensors + '}';
    }
}
