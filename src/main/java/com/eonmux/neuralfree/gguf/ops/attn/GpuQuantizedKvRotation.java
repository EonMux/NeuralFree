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
 * error of the cached rows drops. {@link #forStore} carries the store half alone, for a write that runs no
 * attention (a drafter's injection): K and V only, no Q rotation and no output step.
 *
 * <p>Every transform runs {@value HadamardSpec#KERNEL} in place on a dense {@code [t][width]} view at offset 0, with
 * its dispatch prepared at construction for {@code maxTokens} rows; a step re-sets only the row count and the grid.
 * Not thread-safe.</p>
 */
public final class GpuQuantizedKvRotation implements AutoCloseable {

    private final List<Step> before = new ArrayList<>(3);
    private final Step after;
    private final int local;

    private GpuQuantizedKvRotation(GpuOps ops, RotationPlan plan, View q, View k, View v, View out, int maxTokens,
                                   boolean storeOnly) {
        this.local = ops.local();
        Step afterStep = null;
        try {
            if (plan.rotateK()) {
                HadamardSpec hk = spec(plan.nRotK());
                if (!storeOnly && q != null) {
                    add(ops, hk, q, maxTokens);
                }
                if (k != null) {
                    add(ops, hk, k, maxTokens);
                }
            }
            if (plan.rotateV()) {
                HadamardSpec hv = spec(plan.nRotV());
                if (v != null) {
                    add(ops, hv, v, maxTokens);
                }
                if (!storeOnly && out != null) {
                    afterStep = new Step(prepare(ops, hv, out, maxTokens, 0), out.width / hv.block());
                }
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
        return ofPlanes(ops, plan, q, qWidth, 1, 0, k, kWidth, v, vWidth, out, outWidth, maxTokens);
    }

    /**
     * Rotation of one attention layer whose query buffer holds {@code qPlanes} planes {@code qPlaneStride} floats
     * apart (dual-chunk attention rotates all five of its query planes like the keys they are scored against), with
     * the other views as {@link #of}.
     *
     * @throws IllegalArgumentException when a plane count or stride is negative, a width is not a multiple of its
     *                                  block, a block exceeds the kernel, or a view does not hold the rows
     */
    public static GpuQuantizedKvRotation ofPlanes(GpuOps ops, RotationPlan plan, GpuBuffer q, int qWidth,
                                                  int qPlanes, int qPlaneStride, GpuBuffer k, int kWidth,
                                                  GpuBuffer v, int vWidth, GpuBuffer out, int outWidth,
                                                  int maxTokens) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(plan, "plan");
        if (!plan.rotateK() && !plan.rotateV()) {
            return null;
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("rotation needs a positive token capacity, got " + maxTokens);
        }
        if (qPlanes < 0 || qPlaneStride < 0) {
            throw new IllegalArgumentException("rotation needs planes >= 0 and a non-negative plane stride, got "
                    + qPlanes + " / " + qPlaneStride);
        }
        return new GpuQuantizedKvRotation(ops, plan, q == null ? null : View.planes(q, qWidth, qPlanes, qPlaneStride),
                k == null ? null : new View(k, kWidth), v == null ? null : new View(v, vWidth),
                new View(out, outWidth), maxTokens, false);
    }

    /**
     * Rotation of the stored K and V rows only, for a write that runs no attention (a drafter's injection): K in
     * blocks of {@code plan.nRotK()} and V in blocks of {@code plan.nRotV()}, no Q rotation and no output step.
     *
     * @param k K rows to be stored ({@code n_head_kv * d_k}), or null when the write stores no K
     * @param v V rows to be stored ({@code n_head_kv * d_v}), or null when the write stores no V
     * @throws IllegalArgumentException when a width is not a multiple of its block, a block exceeds the kernel, or a
     *                                  view does not hold {@code maxTokens} rows
     */
    public static GpuQuantizedKvRotation forStore(GpuOps ops, RotationPlan plan, GpuBuffer k, int kWidth, GpuBuffer v,
                                                  int vWidth, int maxTokens) {
        Objects.requireNonNull(ops, "ops");
        Objects.requireNonNull(plan, "plan");
        if (!plan.rotateK() && !plan.rotateV()) {
            return null;
        }
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("rotation needs a positive token capacity, got " + maxTokens);
        }
        return new GpuQuantizedKvRotation(ops, plan, null, k == null ? null : new View(k, kWidth),
                v == null ? null : new View(v, vWidth), null, maxTokens, true);
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
        for (int offset : x.offsets) {
            before.add(new Step(prepare(ops, h, x, maxTokens, offset), x.width / h.block()));
        }
    }

    private PreparedDispatch prepare(GpuOps ops, HadamardSpec h, View x, int maxTokens, int offset) {
        h.checkWidth(x.width);
        if (!h.kernelSupported()) {
            throw new IllegalArgumentException("KV rotation block " + h.block() + " exceeds the kernel limit of "
                    + HadamardSpec.MAX_KERNEL_BLOCK);
        }
        long floats = (long) maxTokens * x.width;
        if ((offset + floats) * Float.BYTES > x.buffer.sizeInBytes()) {
            throw new IllegalArgumentException("rotation view at offset " + offset + " of " + maxTokens + " x "
                    + x.width + " floats exceeds a " + x.buffer.sizeInBytes() + "-byte buffer");
        }
        long groups = floats / h.block();
        return ops.prepare(HadamardSpec.KERNEL, new long[]{groups * local}, new long[]{local},
                maxTokens, x.width, h.block(), h.kernelSignMode(), h.permHd(), h.permNk(), h.permRep(),
                offset, offset, ops.placeholders().readOnlyFloats(), x.buffer, x.buffer);
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
        private final int[] offsets;

        private View(GpuBuffer buffer, int width) {
            this(buffer, width, new int[]{0});
        }

        private View(GpuBuffer buffer, int width, int[] offsets) {
            this.buffer = Objects.requireNonNull(buffer, "buffer");
            if (width <= 0) {
                throw new IllegalArgumentException("rotation view width must be positive, got " + width);
            }
            this.width = width;
            this.offsets = Objects.requireNonNull(offsets, "offsets");
        }

        private static View planes(GpuBuffer buffer, int width, int planes, int planeStride) {
            int[] offsets = new int[planes];
            for (int plane = 0; plane < planes; plane++) {
                offsets[plane] = plane * planeStride;
            }
            return new View(buffer, width, offsets);
        }
    }
}
