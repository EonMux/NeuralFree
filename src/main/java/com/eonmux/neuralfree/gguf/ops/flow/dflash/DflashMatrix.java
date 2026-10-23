package com.eonmux.neuralfree.gguf.ops.flow.dflash;

import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.weights.CpuWeights;

import java.util.Objects;

/**
 * A weight matrix of the drafter bound to the model that stores it: the draft GGUF, or the target GGUF for the
 * shared {@code token_embd} and {@code output} (dflash.cpp:918-926, 1052-1061). Products and row lookups go through
 * the owner, so its transforms ({@code <base>.scale} output scales, PRISM Hadamards) apply as llama.cpp's
 * {@code build_lora_mm} applies them. A ggml tensor {@code {cols, rows}} has {@link #rows()} rows of
 * {@link #cols()} values. Immutable.
 */
final class DflashMatrix {

    private final CpuWeights owner;
    private final CpuTensorRef ref;
    private final String name;

    private DflashMatrix(CpuWeights owner, CpuTensorRef ref, String name) {
        this.owner = owner;
        this.ref = ref;
        this.name = name;
    }

    /** The tensor {@code name} of {@code owner}; null when the owner is null or has no such tensor. */
    static DflashMatrix orNull(CpuWeights owner, String name) {
        if (owner == null) {
            return null;
        }
        CpuTensorRef ref = owner.getOrNull(name);
        return ref == null ? null : new DflashMatrix(owner, ref, name);
    }

    /**
     * The required tensor {@code name} of {@code owner}.
     *
     * @throws IllegalArgumentException when the model has no such tensor
     */
    static DflashMatrix require(CpuWeights owner, String name) {
        DflashMatrix m = orNull(Objects.requireNonNull(owner, "owner"), name);
        if (m == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor '" + name + "'");
        }
        return m;
    }

    /** Main-stack tensor {@code blk.<il>.<suffix>} of {@code owner}, or null. */
    static DflashMatrix layerOrNull(CpuWeights owner, int il, String suffix) {
        CpuTensorRef ref = owner.layerOrNull(il, suffix);
        return ref == null ? null : new DflashMatrix(owner, ref, "blk." + il + "." + suffix);
    }

    /**
     * Required main-stack tensor {@code blk.<il>.<suffix>} of {@code owner}.
     *
     * @throws IllegalArgumentException when the model has no such tensor
     */
    static DflashMatrix layer(CpuWeights owner, int il, String suffix) {
        DflashMatrix m = layerOrNull(owner, il, suffix);
        if (m == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor 'blk." + il + "." + suffix + "'");
        }
        return m;
    }

    /**
     * Required tensor {@code <stack>.<il>.<suffix>} of {@code owner}.
     *
     * @throws IllegalArgumentException when the model has no such tensor
     */
    static DflashMatrix stack(CpuWeights owner, String stack, int il, String suffix) {
        CpuTensorRef ref = owner.stackOrNull(stack, il, suffix);
        if (ref == null) {
            throw new IllegalArgumentException("dflash drafter: model has no tensor '" + stack + "." + il + "."
                    + suffix + "'");
        }
        return new DflashMatrix(owner, ref, stack + "." + il + "." + suffix);
    }

    int rows() {
        return Math.toIntExact(ref.rows());
    }

    int cols() {
        return Math.toIntExact(ref.cols());
    }

    String name() {
        return name;
    }

    /**
     * Checks the matrix shape.
     *
     * @throws IllegalArgumentException when the shape differs from {@code rows x cols}
     */
    DflashMatrix expect(long rows, long cols) {
        if (ref.rows() != rows || ref.cols() != cols) {
            throw new IllegalArgumentException("dflash drafter: tensor '" + name + "' is " + ref.cols() + " x "
                    + ref.rows() + " (ggml ne0 x ne1), expected " + cols + " x " + rows);
        }
        return this;
    }

    /** {@code y[yOff + t*rows + r] = sum_c w[r][c] * x[xOff + t*cols + c]} for {@code t < nTokens}. */
    void matmul(float[] x, int xOff, float[] y, int yOff, int nTokens) {
        owner.matVec().matmul(ref, x, xOff, y, yOff, nTokens);
    }

    /** One-row product {@code y[yOff + r] = sum_c w[r][c] * x[xOff + c]}. */
    void matvec(float[] x, int xOff, float[] y, int yOff) {
        owner.matVec().matvec(ref, x, xOff, y, yOff);
    }

    /** Product with rows {@code [rowStart, rowStart + count)} only. */
    void matvecRows(long rowStart, int count, float[] x, int xOff, float[] y, int yOff) {
        owner.matVec().matvecRows(ref, rowStart, count, x, xOff, y, yOff);
    }

    /** Product with expert {@code expert} of a 3-D tensor stacking {@code nExpert} experts along its rows. */
    void matvecExpert(int expert, int nExpert, float[] x, int xOff, float[] y, int yOff) {
        owner.matVec().matvecExpert(ref, expert, nExpert, x, xOff, y, yOff);
    }

    /** Row {@code row} as a lookup table ({@code ggml_get_rows}) into {@code dst[dstOff .. dstOff + cols)}. */
    void row(long row, float[] dst, int dstOff) {
        if (row < 0 || row >= ref.rows()) {
            throw new IllegalArgumentException("dflash drafter: row " + row + " is outside the " + ref.rows()
                    + " rows of '" + name + "'");
        }
        owner.embedRow(ref, row, dst, dstOff);
    }

    @Override
    public String toString() {
        return "DflashMatrix{" + name + ", " + ref.cols() + " x " + ref.rows() + '}';
    }
}
