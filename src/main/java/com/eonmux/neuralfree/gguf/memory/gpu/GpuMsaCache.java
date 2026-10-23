package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

/**
 * Device MSA index-key cache of one layer (minimax-m3): one index key per KV row, written at {@code cell_idx};
 * the position of each row (and therefore its position block) comes from the KV group's {@code kv_pos}.
 */
public interface GpuMsaCache {

    int layer();

    LayerMemoryPlan.MsaPlan plan();

    /** KV group whose {@code kv_pos} gives the position of each row. */
    int group();

    /** Index key rows. */
    GpuBuffer indexKeys();
}
