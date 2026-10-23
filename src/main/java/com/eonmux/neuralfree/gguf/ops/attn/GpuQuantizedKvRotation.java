package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.gpu.forward.GpuOps;
import com.eonmux.neuralfree.gguf.gpu.forward.PreparedDispatch;
import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Hadamard rotation around a quantized KV cache on the GPU (llama.cpp src/llama-graph.cpp:3225-3231 and 3270-3271,
 * {@code llama_mul_mat_hadamard} in src/llama-impl.h:57-75): with {@link RotationPlan#rotateK()}, Q and the stored K
 * are rotated in blocks of {@link RotationPlan#nRotK()} before K is stored; with {@link RotationPlan#rotateV()}, the
 * stored V is rotated in blocks of {@link RotationPlan#nRotV()} and the attention output is rotated back with the
 * same matrix ({@code H * H = I}). Scores are unchanged because the matrix is orthonormal; only the quantization
 * error of the cached rows drops.
 *
 * <p>Every transform runs {@value HadamardSpec#KERNEL} in place on a dense {@code [t][width]} view at offset 0, with
 * its dispatch prepared at construction for {@code maxTokens} rows; a step re-sets only the row count and the grid.
 * Not thread-safe.</p>
 */
public final class GpuQuantizedKvRotation implements AutoCloseable {

    private final List<Step> before = new ArrayList<>(3);
    private final Step after;
    private final int local;

    private GpuQuantizedKvRotation(GpuOps ops, RotationPlan plan, View q, View k, View v, View out, int maxTokens) {
        this.local = ops.local();
        Step afterStep = null;
        try {
            if (plan.rotateK()) {
                HadamardSpec hk = spec(plan.nRotK());
                add(ops, hk, q, maxTokens);
                if (k != null) {
                    add(ops, hk, k, maxTokens);
                }
            }
            if (plan.rotateV()) {
                HadamardSpec hv = spec(plan.nRotV());
                if (v != null) {
                    add(ops, hv, v, maxTokens);
                }
                afterStep = new Step(prepare(ops, hv, out, maxTokens), out.width / hv.block());
            }
        } catch (RuntimeException e) {
            closeAll(before, afterStep);
            throw e;
        }
        this.after = afterStep;
    }

    /**
     * Rotation of one attention layer, or null when the plan rotates nothing.
     *
     * @param q         Q rows ({@code n_head * d_k} wide)
     * @param k         K rows to be stored ({@code n_head_kv * d_k}), or null when the layer stores no K
     * @param v         V rows to be stored ({@code n_head_kv * d_v}), or null when the layer stores no V
     * @param out       attention output rows ({@code n_head * d_v})
     * @param maxTokens largest token count of a step (the scratch ubatch)
     * @throws IllegalArgumentException when a width is not a multiple of its block, a block exceeds the kernel, or a
     *                                  view does not hold {@code maxTokens} rows
     */
    public static GpuQuantizedKvRotation of(GpuOps ops, RotationPlan plan, GpuBuffer q, int qWidth, GpuBuffer k,
                                            int kWidth, GpuBuffer v, int vWidth, GpuBuffer out, int outWidth,
                                            int maxTokens) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(plan, "plan");
        if (!plan.rotateK() && !plan.rotateV()) {
            return null;
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("rotation needs a positive token capacity, got " + maxTokens);
        }
        return new GpuQuantizedKvRotation(ops, plan, new View(q, qWidth), k == null ? null : new View(k, kWidth),
                v == null ? null : new View(v, vWidth), new View(out, outWidth), maxTokens);
    }

    /** Rotates Q, and K and V when they are stored, for {@code nTokens} rows: before the KV store. */
    public void beforeStore(int nTokens) {
        for (int i = 0; i < before.size(); i++) {
            before.get(i).run(nTokens, local);
        }
    }

    /** Rotates the attention output back for {@code nTokens} rows (V rotation only): before the output projection. */
    public void afterAttention(int nTokens) {
        if (after != null) {
            after.run(nTokens, local);
        }
    }

    /** True when the attention output is rotated back. */
    public boolean rotatesOutput() {
        return after != null;
    }

    private void add(GpuOps ops, HadamardSpec h, View x, int maxTokens) {
        before.add(new Step(prepare(ops, h, x, maxTokens), x.width / h.block()));
    }

    private PreparedDispatch prepare(GpuOps ops, HadamardSpec h, View x, int maxTokens) {
        h.checkWidth(x.width);
        if (!h.kernelSupported()) {
            throw new IllegalArgumentException("KV rotation block " + h.block() + " exceeds the kernel limit of "
                    + HadamardSpec.MAX_KERNEL_BLOCK);
        }
        long floats = (long) maxTokens * x.width;
        if (floats * Float.BYTES > x.buffer.sizeInBytes()) {
            throw new IllegalArgumentException("rotation view of " + maxTokens + " x " + x.width + " floats exceeds a "
                    + x.buffer.sizeInBytes() + "-byte buffer");
        }
        long groups = floats / h.block();
        return ops.prepare(HadamardSpec.KERNEL, new long[]{groups * local}, new long[]{local},
                maxTokens, x.width, h.block(), h.kernelSignMode(), h.permHd(), h.permNk(), h.permRep(), 0, 0,
                ops.placeholders().readOnlyFloats(), x.buffer, x.buffer);
    }

    private static HadamardSpec spec(int block) {
        return HadamardSpec.rotation(block);
    }

    private static void closeAll(List<Step> list, Step extra) {
        for (Step s : list) {
            s.dispatch.close();
        }
        if (extra != null) {
            extra.dispatch.close();
        }
    }

    @Override
    public void close() {
        closeAll(before, after);
        before.clear();
    }

    /**
     * One prepared transform: one work-group per {@code block} values, {@code blocksPerRow} per token. The row count
     * and grid are re-set only when the token count changes, so a steady decode allocates nothing.
     */
    private static final class Step {
        private final PreparedDispatch dispatch;
        private final int blocksPerRow;
        private int lastTokens = -1;

        private Step(PreparedDispatch dispatch, int blocksPerRow) {
            this.dispatch = dispatch;
            this.blocksPerRow = blocksPerRow;
        }

        private void run(int nTokens, int local) {
            if (nTokens != lastTokens) {
                dispatch.setInt(HadamardSpec.ARG_N_ROWS, nTokens);
                dispatch.setGlobal((long) nTokens * blocksPerRow * local);
                lastTokens = nTokens;
            }
            dispatch.dispatch();
        }
    }

    private static final class View {
        private final GpuBuffer buffer;
        private final int width;

        private View(GpuBuffer buffer, int width) {
            this.buffer = Objects.requireNonNull(buffer, "buffer");
            if (width <= 0) {
                throw new IllegalArgumentException("rotation view width must be positive, got " + width);
            }
            this.width = width;
        }
    }
}
