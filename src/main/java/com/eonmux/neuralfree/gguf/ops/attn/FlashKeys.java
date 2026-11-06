package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.memory.KvDType;
import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.gpu.GpuKvLayer;

import java.util.Objects;

/**
 * The key and value rows one layer's flash attention reads: the layer's cache rows ({@link #cached}) with their
 * {@code kv_pos} visibility, or the current batch's K and V scratch rows ({@link #batch}) with
 * {@code kv_pos = q_pos}. Immutable.
 */
final class FlashKeys {

    private final KvDType typeK;
    private final KvDType typeV;
    private final GpuBuffer k;
    private final int kRowStride;
    private final GpuBuffer v;
    private final int vRowStride;
    private final GpuBuffer kvPos;
    private final GpuBuffer rowScales;
    private final GpuBuffer vendorSide;
    private final boolean vAliasesK;
    private final int rows;
    private final int rowBase;
    private final boolean cached;

    private FlashKeys(KvDType typeK, KvDType typeV, GpuBuffer k, int kRowStride, GpuBuffer v, int vRowStride,
                      GpuBuffer kvPos, GpuBuffer rowScales, GpuBuffer vendorSide, boolean vAliasesK, int rows,
                      int rowBase, boolean cached) {
        this.typeK = Objects.requireNonNull(typeK, "typeK");
        this.typeV = Objects.requireNonNull(typeV, "typeV");
        this.k = Objects.requireNonNull(k, "k");
        this.kRowStride = kRowStride;
        this.v = Objects.requireNonNull(v, "v");
        this.vRowStride = vRowStride;
        this.kvPos = Objects.requireNonNull(kvPos, "kvPos");
        this.rowScales = rowScales;
        this.vendorSide = vendorSide;
        this.vAliasesK = vAliasesK;
        this.rows = rows;
        this.rowBase = rowBase;
        this.cached = cached;
    }

    /** The cache rows of {@code kv}, whose positions are the step inputs' {@code kvPos} of the cache's group. */
    static FlashKeys cached(GpuKvLayer kv, GpuBuffer kvPos) {
        KvLayout layout = kv.layout();
        return new FlashKeys(layout.dtypeK(), layout.dtypeV(), kv.k(), layout.widthK(), kv.v(), layout.widthV(),
                kvPos, kv.rowScales(), kv.vendorSide(), layout.vAliasesK(), layout.rows(), layout.rowBase(), true);
    }

    /** The F32 K and V rows of the current batch (no cache): {@code ubatch} rows positioned by {@code pos}. */
    static FlashKeys batch(GpuBuffer k, int kDim, GpuBuffer v, int vDim, GpuBuffer pos, int ubatch) {
        return new FlashKeys(KvDType.F32, KvDType.F32, k, kDim, v, vDim, pos, null, null, false, ubatch, 0, false);
    }

    KvDType typeK() {
        return typeK;
    }

    KvDType typeV() {
        return typeV;
    }

    GpuBuffer k() {
        return k;
    }

    /** Floats between consecutive K rows. */
    int kRowStride() {
        return kRowStride;
    }

    GpuBuffer v() {
        return v;
    }

    /** Floats between consecutive V rows. */
    int vRowStride() {
        return vRowStride;
    }

    /** The position of every key row. */
    GpuBuffer kvPos() {
        return kvPos;
    }

    /**
     * The interleaved per-(row, KV head) scale rows of a per-token-head cache ([K heads, V heads] of every row), or
     * null. The flash kernels read it through their {@code kv_scales} binding.
     */
    GpuBuffer rowScales() {
        return rowScales;
    }

    /**
     * The side buffer of a vendor cache, or null: the flash kernels' {@code kv_scales} binding, holding the cache's
     * checkpoint KV scales, the Lloyd-Max table of a TurboQuant key stream and the packed plane rows.
     */
    GpuBuffer vendorSide() {
        return vendorSide;
    }

    /** Whether V is read from the K rows (MLA latent cache). */
    boolean vAliasesK() {
        return vAliasesK;
    }

    /** Key rows addressable: the cache rows, or the micro-batch size. */
    int rows() {
        return rows;
    }

    /** Global position of local row 0; 0 for a whole cache, the block start for a context shard. */
    int rowBase() {
        return rowBase;
    }

    /** Whether the keys are cache rows rather than the current batch. */
    boolean cached() {
        return cached;
    }
}
