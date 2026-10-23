package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

/**
 * Device lightning-indexer key cache of one layer (DSA, QSA k-pool): rows of {@link LayerMemoryPlan.IndexerPlan#width()}
 * values in the plan's element type ({@link LayerMemoryPlan.IndexerPlan#dtype()}, the context's K type), one per KV
 * row, written at {@code cell_idx} and addressed through the same KV group as the layer's cache.
 */
public interface GpuIndexerCache {

    int layer();

    LayerMemoryPlan.IndexerPlan plan();

    /** KV group whose {@code kv_pos} gives the position of each row. */
    int group();

    /** Indexer key rows. */
    GpuBuffer keys();
}
