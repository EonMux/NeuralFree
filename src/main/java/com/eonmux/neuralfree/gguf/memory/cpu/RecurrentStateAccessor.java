package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.List;

/**
 * Heap recurrent state of one layer: one F32 array per {@link LayerMemoryPlan.StateTensor}, laid out as its
 * {@link LayerMemoryPlan.StateLayout} describes. The arrays are the live runtime state: mixers read and update them
 * in place, {@code reset()} zeroes them and {@code snapshot()} / {@code restore()} copy them.
 */
public interface RecurrentStateAccessor {

    int layer();

    List<LayerMemoryPlan.StateTensor> tensors();

    /**
     * Live array of a state tensor ({@code floats()} values).
     *
     * @throws IllegalArgumentException when the layer has no tensor of that name
     */
    float[] tensor(String name);
}
