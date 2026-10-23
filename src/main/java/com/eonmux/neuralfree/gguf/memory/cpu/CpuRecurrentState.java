package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Heap recurrent state of one layer: one zeroed F32 array per state tensor (single sequence), updated in place by
 * the mixers. When allocated with snapshots, every array has a shadow of the same size: {@link #snapshot()} copies
 * the arrays into it with {@code System.arraycopy} and {@link #restore()} copies it back.
 */
public final class CpuRecurrentState implements RecurrentStateAccessor {

    private static final long MAX_ARRAY = Integer.MAX_VALUE - 8L;

    private final int layer;
    private final List<LayerMemoryPlan.StateTensor> tensors;
    private final Map<String, float[]> arrays;
    private final Map<String, float[]> shadows;
    private boolean saved;

    private CpuRecurrentState(int layer, List<LayerMemoryPlan.StateTensor> tensors, Map<String, float[]> arrays,
                              Map<String, float[]> shadows) {
        this.layer = layer;
        this.tensors = tensors;
        this.arrays = arrays;
        this.shadows = shadows;
    }

    /**
     * Allocates the zeroed state tensors of a layer, without snapshot shadows.
     *
     * @throws IllegalArgumentException for an empty list, duplicate names or a tensor larger than a Java array
     */
    public static CpuRecurrentState allocate(int layer, List<LayerMemoryPlan.StateTensor> tensors) {
        return allocate(layer, tensors, false);
    }

    /**
     * Allocates the zeroed state tensors of a layer and, with {@code snapshots}, a shadow of each for
     * {@link #snapshot()} ({@code MemoryPlan#snapshotsReserved}).
     *
     * @throws IllegalArgumentException for an empty list, duplicate names or a tensor larger than a Java array
     */
    public static CpuRecurrentState allocate(int layer, List<LayerMemoryPlan.StateTensor> tensors,
                                             boolean snapshots) {
        Objects.requireNonNull(tensors, "tensors");
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("layer " + layer + " has no recurrent state tensors");
        }
        Map<String, float[]> arrays = new LinkedHashMap<>();
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (arrays.containsKey(t.name())) {
                throw new IllegalArgumentException("layer " + layer + " declares state tensor " + t.name() + " twice");
            }
            if (t.floats() > MAX_ARRAY) {
                throw new IllegalArgumentException("state tensor " + t.name() + " of layer " + layer + " holds "
                        + t.floats() + " floats, more than a Java array can");
            }
            arrays.put(t.name(), new float[(int) t.floats()]);
        }
        Map<String, float[]> shadows = new LinkedHashMap<>();
        if (snapshots) {
            for (Map.Entry<String, float[]> e : arrays.entrySet()) {
                shadows.put(e.getKey(), new float[e.getValue().length]);
            }
        }
        return new CpuRecurrentState(layer, Collections.unmodifiableList(new ArrayList<>(tensors)), arrays, shadows);
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
    public float[] tensor(String name) {
        float[] a = arrays.get(name);
        if (a == null) {
            throw new IllegalArgumentException("layer " + layer + " has no state tensor '" + name + "' (has "
                    + arrays.keySet() + ")");
        }
        return a;
    }

    /** Zeroes every state tensor and forgets the snapshot. */
    public void reset() {
        for (float[] a : arrays.values()) {
            Arrays.fill(a, 0f);
        }
        saved = false;
    }

    /**
     * Copies every state tensor into its shadow.
     *
     * @throws IllegalStateException when the state was allocated without shadows
     */
    public void snapshot() {
        if (shadows.isEmpty()) {
            throw new IllegalStateException("recurrent state of layer " + layer + " has no snapshot buffers");
        }
        for (Map.Entry<String, float[]> e : arrays.entrySet()) {
            float[] src = e.getValue();
            System.arraycopy(src, 0, shadows.get(e.getKey()), 0, src.length);
        }
        saved = true;
    }

    /**
     * Copies every shadow back into its state tensor.
     *
     * @throws IllegalStateException when {@link #snapshot()} has not run since the last reset
     */
    public void restore() {
        if (!saved) {
            throw new IllegalStateException("layer " + layer + " has no recurrent-state snapshot");
        }
        for (Map.Entry<String, float[]> e : arrays.entrySet()) {
            float[] dst = e.getValue();
            System.arraycopy(shadows.get(e.getKey()), 0, dst, 0, dst.length);
        }
    }

    @Override
    public String toString() {
        return "CpuRecurrentState{il=" + layer + ", " + tensors + '}';
    }
}
