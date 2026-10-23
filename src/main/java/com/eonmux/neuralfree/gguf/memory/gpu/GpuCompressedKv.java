package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

/**
 * Device DeepSeek-V4 compressed cache of one layer (CSA, HCA or the CSA lightning indexer): the K-only compressed
 * rows plus the two F32 compressor tensors, all zeroed at reset. Compressed row {@code r} holds the compression of
 * positions {@code [r * ratio, (r + 1) * ratio)}. Each compressor tensor holds {@link LayerMemoryPlan.CompressedKvPlan#planes()}
 * consecutive planes of {@code stateRows x stateWidth}: plane 0 is the live state the compressor reads and updates,
 * planes {@code 1..snapshots} the rollback snapshots each step writes ({@code LLM_DSV4_StateSnapshot}).
 */
public interface GpuCompressedKv {

    int layer();

    LayerMemoryPlan.CompressedKvPlan plan();

    /** Compressed K rows. */
    GpuBuffer cache();

    /** Compressor kv tensor: {@code planes()} planes of {@code stateRows x stateWidth} F32, the live one first. */
    GpuBuffer stateKv();

    /** Compressor score tensor: {@code planes()} planes of {@code stateRows x stateWidth} F32, the live one first. */
    GpuBuffer stateScore();

    /** Rollback snapshot planes after the live plane ({@code n_rs_seq}). */
    default int snapshots() {
        return plan().snapshots();
    }
}
