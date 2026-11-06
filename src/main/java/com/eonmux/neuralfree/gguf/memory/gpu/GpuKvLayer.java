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

    /**
     * The interleaved per-(row, KV head) scales of a per-token-head cache ({@code rows * 2 * nHeadKv} F32 values,
     * K heads then V heads of each row), or null for every other dtype. The store kernels write them and attention
     * reads them through its {@code kv_scales} binding; page copies, row moves and shifts carry them with the rows.
     */
    default GpuBuffer rowScales() {
        return null;
    }

    /**
     * The side buffer of a vendor cache, or null for every other cache type: its first {@code 2 * nHeadKv} F32
     * values are the cache's checkpoint KV scales (K heads then V heads), the next {@code 16} are the Lloyd-Max
     * centroid table of a TurboQuant key stream, and the packed side-plane rows of K and (separate) V follow at
     * {@link DeviceKvFormats#sidePlaneBase(int)}, {@code kPlaneRowBytes + vPlaneRowBytes} bytes per row. The store
     * kernels write the planes and the attention reads the scales, the table and the planes through its
     * {@code kv_scales} binding, which is this buffer; page copies, row moves and shifts carry the plane rows with
     * the codes.
     */
    default GpuBuffer vendorSide() {
        return null;
    }
}
