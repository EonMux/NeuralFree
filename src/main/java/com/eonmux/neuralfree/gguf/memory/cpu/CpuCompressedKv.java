package com.eonmux.neuralfree.gguf.memory.cpu;

import com.eonmux.neuralfree.gguf.memory.KvLayout;
import com.eonmux.neuralfree.gguf.memory.KvWindow;
import com.eonmux.neuralfree.gguf.memory.LayerMemoryPlan;

import java.util.Arrays;
import java.util.Objects;

/**
 * Heap DeepSeek-V4 compressed cache of one layer and kind (src/llama-kv-cache-dsv4.cpp:1244-1340): single-head,
 * key-only compressed rows (row {@code r} compresses positions {@code [r * ratio, (r + 1) * ratio)}, not addressed
 * by a KV group, so {@link #group()} is -1) plus the two F32 compressor tensors, exposed through {@link #state()} as
 * tensors {@link CpuMemory#COMPRESSOR_KV} and {@link CpuMemory#COMPRESSOR_SCORE}: {@code planes()} consecutive
 * row-major planes of {@code stateRows x stateWidth}, the live plane first, then the rollback snapshots. Rows and
 * planes are zeroed at allocation and at every reset.
 */
public final class CpuCompressedKv implements KvAccessor {

    private final LayerMemoryPlan.CompressedKvPlan plan;
    private final CpuKvLayer rows;
    private final CpuRecurrentState state;

    private CpuCompressedKv(LayerMemoryPlan.CompressedKvPlan plan, CpuKvLayer rows, CpuRecurrentState state) {
        this.plan = plan;
        this.rows = rows;
        this.state = state;
    }

    /**
     * Allocates the zeroed cache and planes. The planes are labelled {@link LayerMemoryPlan.StateLayout#PLE_CONV},
     * the row-major {@code [slot][width]} history layout.
     */
    public static CpuCompressedKv allocate(int layer, LayerMemoryPlan.CompressedKvPlan plan) {
        Objects.requireNonNull(plan, "plan");
        KvLayout layout = KvLayout.latent(plan.rows(), false, plan.width(), plan.width(), plan.dtype(),
                KvWindow.FULL_CAUSAL);
        long tensorFloats = (long) plan.planes() * plan.planeFloats();
        CpuRecurrentState planes = CpuRecurrentState.allocate(layer, Arrays.asList(
                LayerMemoryPlan.StateTensor.of(CpuMemory.COMPRESSOR_KV, LayerMemoryPlan.StateLayout.PLE_CONV,
                        tensorFloats),
                LayerMemoryPlan.StateTensor.of(CpuMemory.COMPRESSOR_SCORE, LayerMemoryPlan.StateLayout.PLE_CONV,
                        tensorFloats)));
        return new CpuCompressedKv(plan, CpuKvLayer.keyOnly(layer, layout, -1), planes);
    }

    public LayerMemoryPlan.CompressedKvPlan plan() {
        return plan;
    }

    /** The compressor planes. */
    public RecurrentStateAccessor state() {
        return state;
    }

    /**
     * Copies rollback snapshot plane {@code d} of both compressor tensors into their live plane (llama.cpp's restore
     * of plane {@code rs_idx}, llama-kv-cache-dsv4.cpp:670-681).
     *
     * @throws IllegalArgumentException when {@code d} is not a snapshot plane ({@code 1..snapshots})
     */
    public void restorePlane(int d) {
        if (d < 1 || d > plan.snapshots()) {
            throw new IllegalArgumentException(plan.kind() + " compressor of layer " + layer() + " has snapshot planes 1.."
                    + plan.snapshots() + ", not " + d);
        }
        int n = plan.planeFloats();
        for (String name : new String[]{CpuMemory.COMPRESSOR_KV, CpuMemory.COMPRESSOR_SCORE}) {
            float[] t = state.tensor(name);
            System.arraycopy(t, d * n, t, 0, n);
        }
    }

    /** Zeroes the compressed rows and both compressor tensors, every plane. */
    public void reset() {
        rows.clear();
        state.reset();
    }

    @Override
    public int layer() {
        return rows.layer();
    }

    @Override
    public KvLayout layout() {
        return rows.layout();
    }

    @Override
    public int group() {
        return -1;
    }

    @Override
    public int nHeadKv() {
        return 1;
    }

    @Override
    public int headDimK() {
        return rows.headDimK();
    }

    @Override
    public int headDimV() {
        return rows.headDimV();
    }

    @Override
    public void readK(int row, int head, float[] dst, int off) {
        rows.readK(row, head, dst, off);
    }

    @Override
    public void readV(int row, int head, float[] dst, int off) {
        rows.readV(row, head, dst, off);
    }

    @Override
    public void writeK(int row, float[] src, int off) {
        rows.writeK(row, src, off);
    }

    @Override
    public void writeKSlice(int row, int from, int n, float[] src, int off) {
        rows.writeKSlice(row, from, n, src, off);
    }

    @Override
    public void writeV(int row, float[] src, int off) {
        throw new UnsupportedOperationException(plan.kind() + " compressed cache of layer " + layer()
                + " stores keys only");
    }

    @Override
    public String toString() {
        return "CpuCompressedKv{il=" + layer() + ", " + plan + '}';
    }
}
