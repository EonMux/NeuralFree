package com.eonmux.neuralfree.gguf.memory.gpu;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;

/**
 * Device KV cache of one owning layer: {@link KvLayout#rows()} rows of K and V. A position is stored in row
 * {@code layout().rowOf(pos)}; the row's position is tracked in the group's {@code kv_pos} buffer
 * ({@link GpuStepInputs#kvPos(int)}). For an MLA latent cache {@link #v()} returns the same buffer as {@link #k()}.
 */
public interface GpuKvLayer {

    /** Layer that owns this cache. */
    int layer();

    KvLayout layout();

    /** KV group whose {@code kv_pos} / {@code cell_idx} buffers address this cache. */
    int group();

    /** Hadamard rotation applied to Q/K and V around the store. */
    RotationPlan rotation();

    /** K rows ({@code rows * rowBytesK} bytes). */
    GpuBuffer k();

    /** V rows; {@link #k()} when {@code layout().vAliasesK()}. */
    GpuBuffer v();
}
