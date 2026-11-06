package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.List;

/**
 * Heap recurrent state of one layer: one F32 array per {@link LayerMemoryPlan.StateTensor}, laid out as its
 * {@link LayerMemoryPlan.StateLayout} describes. The arrays are the live runtime state: mixers read and update them
 * in place, {@code reset()} zeroes them and {@code snapshot()} / {@code restore()} copy them.
 *
 * <p>The tensors hold one row per state slot ({@link #slots()}): a contiguous memory has one slot (sequence 0, the
 * row every tensor starts at), a paged memory one row per sequence followed by the rollback snapshot rows, so a
 * mixer addresses sequence {@code seq}'s row at float offset {@code seq * <floats of the tensor>}.</p>
 */
public interface RecurrentStateAccessor {

    int layer();

    List<LayerMemoryPlan.StateTensor> tensors();

    /**
     * Live array of a state tensor ({@code slots() * floats()} values).
     *
     * @throws IllegalArgumentException when the layer has no tensor of that name
     */
    float[] tensor(String name);

    /**
     * State slots every tensor holds: 1 for a contiguous single-sequence memory,
     * {@code parallel x (1 + rollbackSnapshots)} for a paged one ({@code MemoryPlanner}).
     */
    default int slots() {
        return 1;
    }

    /**
     * Floats of one slot of the tensor {@code name} (its declared {@code floats()}).
     *
     * @throws IllegalArgumentException when the layer has no tensor of that name
     */
    default int slotFloats(String name) {
        return rowFloats(tensors(), name, layer());
    }

    /** {@code t.floats()} of the named tensor, or an exception naming the layer and the tensors it has. */
    static int rowFloats(List<LayerMemoryPlan.StateTensor> tensors, String name, int layer) {
        for (LayerMemoryPlan.StateTensor t : tensors) {
            if (t.name().equals(name)) {
                return Math.toIntExact(t.floats());
            }
        }
        throw new IllegalArgumentException("layer " + layer + " has no state tensor '" + name + "' (has "
                + tensors + ")");
    }
}
