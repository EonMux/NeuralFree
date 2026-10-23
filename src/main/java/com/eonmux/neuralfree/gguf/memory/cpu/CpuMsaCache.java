package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;
import com.eonmux.neuralfree.gguf.memory.MemoryPlan;

import java.util.Objects;

/**
 * Heap MSA index-key cache of one layer (minimax-m3): a single-head, key-only cache of {@code width} per row in the
 * plan's KV type, addressed by the layer's KV group; the position of each row (and so its position block) is
 * {@link CpuMemory#positionOfRow(int, int)}. V reads return the key; V writes are rejected.
 */
public final class CpuMsaCache implements KvAccessor {

    private final LayerMemoryPlan.MsaPlan plan;
    private final CpuKvLayer keys;

    private CpuMsaCache(LayerMemoryPlan.MsaPlan plan, CpuKvLayer keys) {
        this.plan = plan;
        this.keys = keys;
    }

    /** Allocates the zeroed cache of a layer whose KV rows are addressed by {@code group}. */
    public static CpuMsaCache allocate(int layer, LayerMemoryPlan.MsaPlan plan, MemoryPlan.KvGroup group) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(group, "group");
        KvLayout layout = KvLayout.latent(plan.rows(), group.ring(), plan.width(), plan.width(), plan.dtype(),
                group.window());
        return new CpuMsaCache(plan, CpuKvLayer.keyOnly(layer, layout, group.index()));
    }

    public LayerMemoryPlan.MsaPlan plan() {
        return plan;
    }

    @Override
    public int layer() {
        return keys.layer();
    }

    @Override
    public KvLayout layout() {
        return keys.layout();
    }

    @Override
    public int group() {
        return keys.group();
    }

    @Override
    public int nHeadKv() {
        return 1;
    }

    @Override
    public int headDimK() {
        return keys.headDimK();
    }

    @Override
    public int headDimV() {
        return keys.headDimV();
    }

    @Override
    public void readK(int row, int head, float[] dst, int off) {
        keys.readK(row, head, dst, off);
    }

    @Override
    public void readV(int row, int head, float[] dst, int off) {
        keys.readV(row, head, dst, off);
    }

    @Override
    public void writeK(int row, float[] src, int off) {
        keys.writeK(row, src, off);
    }

    @Override
    public void writeKSlice(int row, int from, int n, float[] src, int off) {
        keys.writeKSlice(row, from, n, src, off);
    }

    @Override
    public void writeV(int row, float[] src, int off) {
        throw new UnsupportedOperationException("the MSA cache of layer " + layer() + " stores index keys only");
    }

    @Override
    public String toString() {
        return "CpuMsaCache{il=" + layer() + ", group=" + group() + ", " + plan + '}';
    }
}
