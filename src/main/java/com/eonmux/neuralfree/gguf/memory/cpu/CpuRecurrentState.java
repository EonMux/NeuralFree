package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.RecurrentStateCopy;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Heap recurrent state of one layer: one F32 array per state tensor holding one row of
 * {@code floats()} per slot ({@link #slots()}, single sequence), updated in place by the mixers. When allocated with
 * snapshots, every array has a shadow of the same size: {@link #snapshot()} copies the arrays into it with
 * {@code System.arraycopy} and {@link #restore()} copies it back.
 */
public final class CpuRecurrentState implements RecurrentStateAccessor {

    private static final long MAX_ARRAY = Integer.MAX_VALUE - 8L;

    private final int layer;
    private final List<LayerMemoryPlan.StateTensor> tensors;
    private final Map<String, float[]> arrays;
    private final Map<String, float[]> shadows;
    private final int slots;
    private boolean saved;

    private CpuRecurrentState(int layer, List<LayerMemoryPlan.StateTensor> tensors, Map<String, float[]> arrays,
                              Map<String, float[]> shadows, int slots) {
        this.layer = layer;
        this.tensors = tensors;
        this.arrays = arrays;
        this.shadows = shadows;
        this.slots = slots;
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
     * Allocates the zeroed state tensors of a layer — each holding its {@code slots()} rows — and, with
     * {@code snapshots}, a shadow of each for {@link #snapshot()} ({@code MemoryPlan#snapshotsReserved}).
     *
     * @throws IllegalArgumentException for an empty list, duplicate names or a tensor larger than a Java array
     */
    public static CpuRecurrentState allocate(int layer, List<LayerMemoryPlan.StateTensor> tensors,
                                             boolean snapshots) {
        Objects.requireNonNull(tensors, "tensors");
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("layer " + layer + " has no recurrent state tensors");
        }
        int slots = tensors.get(0).slots();
        Map<String, float[]> arrays = new LinkedHashMap<>();
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (arrays.containsKey(t.name())) {
                throw new IllegalArgumentException("layer " + layer + " declares state tensor " + t.name() + " twice");
            }
            if (t.slots() != slots) {
                throw new IllegalArgumentException("layer " + layer + " holds " + t.slots() + " slots of state tensor "
                        + t.name() + ", its other tensors hold " + slots);
            }
            if (t.floats() * t.slots() > MAX_ARRAY) {
                throw new IllegalArgumentException("state tensor " + t.name() + " of layer " + layer + " holds "
                        + t.floats() * t.slots() + " floats, more than a Java array can");
            }
            arrays.put(t.name(), new float[(int) (t.floats() * t.slots())]);
        }
        Map<String, float[]> shadows = new LinkedHashMap<>();
        if (snapshots) {
            for (Map.Entry<String, float[]> e : arrays.entrySet()) {
                shadows.put(e.getKey(), new float[e.getValue().length]);
            }
        }
        return new CpuRecurrentState(layer, Collections.unmodifiableList(new ArrayList<>(tensors)), arrays, shadows,
                slots);
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

    /** The state slots every tensor holds ({@link RecurrentStateAccessor#slots()}). */
    @Override
    public int slots() {
        return slots;
    }

    /**
     * Zeroes slot {@code seq} of every state tensor (the sequence's live row); the other slots and any snapshot
     * shadow stay.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the slots
     */
    public void clearSlot(int seq) {
        if (seq < 0 || seq >= slots) {
            throw new IndexOutOfBoundsException("state slot " + seq + " outside [0, " + slots + ")");
        }
        for (Map.Entry<String, float[]> e : arrays.entrySet()) {
            int row = (int) rowFloatsOf(e.getKey());
            Arrays.fill(e.getValue(), seq * row, (seq + 1) * row, 0f);
        }
    }

    /**
     * Copies slot {@code seq} of every state tensor, one row per tensor in {@link #tensors()} order.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the slots
     */
    public float[][] copySlot(int seq) {
        if (seq < 0 || seq >= slots) {
            throw new IndexOutOfBoundsException("state slot " + seq + " outside [0, " + slots + ")");
        }
        float[][] out = new float[tensors.size()][];
        for (int i = 0; i < tensors.size(); i++) {
            LayerMemoryPlan.StateTensor t = tensors.get(i);
            int row = (int) t.floats();
            out[i] = Arrays.copyOfRange(arrays.get(t.name()), seq * row, (seq + 1) * row);
        }
        return out;
    }

    /**
     * Writes rows {@code [first, first + tensors().size())} of {@code copy} into slot {@code seq} of the state
     * tensors, in {@link #tensors()} order; the other slots stay.
     *
     * @throws IndexOutOfBoundsException when {@code seq} is outside the slots or the copy holds too few rows
     * @throws IllegalArgumentException  when a row's width differs from its tensor's slot width
     */
    public void restoreSlot(int seq, RecurrentStateCopy copy, int first) {
        if (seq < 0 || seq >= slots) {
            throw new IndexOutOfBoundsException("state slot " + seq + " outside [0, " + slots + ")");
        }
        if (first < 0 || first + tensors.size() > copy.rows()) {
            throw new IndexOutOfBoundsException("state copy " + copy + " holds no rows [" + first + ", "
                    + (first + tensors.size()) + ") for layer " + layer);
        }
        for (int i = 0; i < tensors.size(); i++) {
            LayerMemoryPlan.StateTensor t = tensors.get(i);
            if (copy.floats(first + i) != t.floats()) {
                throw new IllegalArgumentException("state copy row " + (first + i) + " holds " + copy.floats(first + i)
                        + " floats, state tensor " + t.name() + " of layer " + layer + " holds " + t.floats());
            }
        }
        for (int i = 0; i < tensors.size(); i++) {
            LayerMemoryPlan.StateTensor t = tensors.get(i);
            copy.copyInto(first + i, arrays.get(t.name()), (int) (seq * t.floats()));
        }
    }

    private long rowFloatsOf(String name) {
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (t.name().equals(name)) {
                return t.floats();
            }
        }
        throw new IllegalStateException("state tensor '" + name + "' left its tensors list");
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
