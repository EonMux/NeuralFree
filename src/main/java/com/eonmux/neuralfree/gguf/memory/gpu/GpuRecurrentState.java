package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.List;

/**
 * Device recurrent state of one layer: one F32 buffer per {@link LayerMemoryPlan.StateTensor}, laid out as its
 * {@link LayerMemoryPlan.StateLayout} describes, holding one row per sequence (row {@code stateRows[s]} of
 * {@link GpuStepInputs#stateRows()}).
 */
public interface GpuRecurrentState {

    int layer();

    /** The state tensors of this layer, in plan order. */
    List<LayerMemoryPlan.StateTensor> tensors();

    /**
     * Buffer of a state tensor by name ({@link LayerMemoryPlan#STATE_CONV}, {@link LayerMemoryPlan#STATE_SCAN},
     * {@link LayerMemoryPlan#STATE_SHIFT}, {@link LayerMemoryPlan#STATE_PLE_CONV}).
     *
     * @throws IllegalArgumentException when the layer has no tensor of that name
     */
    GpuBuffer tensor(String name);
}
