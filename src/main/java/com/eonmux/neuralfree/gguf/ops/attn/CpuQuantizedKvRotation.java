package com.eonmux.neuralfree.gguf.ops.attn;

import com.eonmux.neuralfree.gguf.memory.RotationPlan;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.util.Objects;

/**
 * CPU twin of {@link GpuQuantizedKvRotation} (llama.cpp src/llama-graph.cpp:3225-3231 and 3270-3271): with
 * {@link RotationPlan#rotateK()} Q and the stored K rows are rotated in Hadamard blocks of {@link RotationPlan#nRotK()}
 * before the store; with {@link RotationPlan#rotateV()} the stored V rows are rotated in blocks of
 * {@link RotationPlan#nRotV()} and the attention output is rotated back with the same matrix. Rows are dense
 * {@code [t][width]} arrays from index 0; transforms run in place. Immutable.
 */
public final class CpuQuantizedKvRotation {

    private final HadamardSpec k;
    private final HadamardSpec v;
    private final int qWidth;
    private final int kWidth;
    private final int vWidth;
    private final int outWidth;

    private CpuQuantizedKvRotation(HadamardSpec k, HadamardSpec v, int qWidth, int kWidth, int vWidth, int outWidth) {
        this.k = k;
        this.v = v;
        this.qWidth = qWidth;
        this.kWidth = kWidth;
        this.vWidth = vWidth;
        this.outWidth = outWidth;
    }

    /**
     * Rotation of one attention layer, or null when the plan rotates nothing.
     *
     * @param kWidth K row width, or 0 when the layer stores no K / V (shared-KV reader)
     * @param vWidth V row width, or 0 when the layer stores no K / V
     * @throws IllegalArgumentException when a width is not a multiple of its block
     */
    public static CpuQuantizedKvRotation of(RotationPlan plan, int qWidth, int kWidth, int vWidth, int outWidth) {
        Objects.requireNonNull(plan, "plan");
        if (!plan.rotateK() && !plan.rotateV()) {
            return null;
        }
        HadamardSpec hk = plan.rotateK() ? HadamardSpec.rotation(plan.nRotK()) : null;
        HadamardSpec hv = plan.rotateV() ? HadamardSpec.rotation(plan.nRotV()) : null;
        if (hk != null) {
            hk.checkWidth(qWidth);
            if (kWidth > 0) {
                hk.checkWidth(kWidth);
            }
        }
        if (hv != null) {
            if (vWidth > 0) {
                hv.checkWidth(vWidth);
            }
            hv.checkWidth(outWidth);
        }
        return new CpuQuantizedKvRotation(hk, hv, qWidth, kWidth, vWidth, outWidth);
    }

    /**
     * The store half alone, for a write that runs no attention: K in blocks of {@code plan.nRotK()} and V in blocks
     * of {@code plan.nRotV()}, with no Q rotation and no output step.
     *
     * @param kWidth K row width, or 0 when the write stores no K
     * @param vWidth V row width, or 0 when the write stores no V
     * @throws IllegalArgumentException when a width is not a multiple of its block
     */
    public static CpuQuantizedKvRotation forStore(RotationPlan plan, int kWidth, int vWidth) {
        Objects.requireNonNull(plan, "plan");
        if (!plan.rotateK() && !plan.rotateV()) {
            return null;
        }
        HadamardSpec hk = plan.rotateK() ? HadamardSpec.rotation(plan.nRotK()) : null;
        HadamardSpec hv = plan.rotateV() ? HadamardSpec.rotation(plan.nRotV()) : null;
        if (hk != null && kWidth > 0) {
            hk.checkWidth(kWidth);
        }
        if (hv != null && vWidth > 0) {
            hv.checkWidth(vWidth);
        }
        return new CpuQuantizedKvRotation(hk, hv, 0, kWidth, vWidth, 0);
    }

    /** Rotates Q, and K and V when the layer stores them ({@code kv != null}), for {@code nTokens} rows. */
    public void beforeStore(int nTokens, float[] q, float[] kRows, float[] vRows) {
        beforeStorePlanes(nTokens, q, 1, 0, kRows, vRows);
    }

    /**
     * Rotates the query planes of a multi-plane query buffer, and K and V when the layer stores them, for
     * {@code nTokens} rows: dual-chunk attention keeps its five query rotations {@code planeStride} floats apart and
     * every plane has to be rotated like the keys it is scored against.
     *
     * @param planes      query planes to rotate
     * @param planeStride floats between consecutive planes
     * @throws IllegalArgumentException when the plane stride is negative or the buffer is too short
     */
    public void beforeStorePlanes(int nTokens, float[] q, int planes, int planeStride, float[] kRows,
                                  float[] vRows) {
        if (planes < 0 || planeStride < 0) {
            throw new IllegalArgumentException("rotation needs planes >= 0 and a non-negative plane stride, got "
                    + planes + " / " + planeStride);
        }
        if (k != null) {
            for (int plane = 0; plane < planes; plane++) {
                CpuHadamard.applyInPlace(k, q, plane * planeStride, nTokens, qWidth);
            }
            if (kRows != null && kWidth > 0) {
                CpuHadamard.applyInPlace(k, kRows, 0, nTokens, kWidth);
            }
        }
        if (v != null && vRows != null && vWidth > 0) {
            CpuHadamard.applyInPlace(v, vRows, 0, nTokens, vWidth);
        }
    }

    /**
     * Rotates the stored K and V rows alone for {@code nTokens} rows, for a write that runs no query or attention.
     */
    public void beforeStoreKV(int nTokens, float[] kRows, float[] vRows) {
        if (k != null && kRows != null && kWidth > 0) {
            CpuHadamard.applyInPlace(k, kRows, 0, nTokens, kWidth);
        }
        if (v != null && vRows != null && vWidth > 0) {
            CpuHadamard.applyInPlace(v, vRows, 0, nTokens, vWidth);
        }
    }

    /** Rotates the attention output back for {@code nTokens} rows (V rotation only). */
    public void afterAttention(int nTokens, float[] out) {
        if (v != null) {
            CpuHadamard.applyInPlace(v, out, 0, nTokens, outWidth);
        }
    }

    /** True when the attention output is rotated back. */
    public boolean rotatesOutput() {
        return v != null;
    }
}
