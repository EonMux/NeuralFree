package com.eonmux.neuralfree.gguf.ops.mla;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuActivationQuantizer;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuMatVec;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuTensorRef;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;

import java.util.Objects;

/**
 * Per-head products of the CPU MLA and DSA sparse-MLA mixers over every (token, head) pair of a step in one parallel
 * pass: {@code y[t][h][r] = sum_c w[h*R + r][c] * x[t][h][c]} for {@code r < R}, where the weight stacks one
 * {@code [R][cols]} matrix per head. This is the CPU form of llama.cpp's single batched {@code ggml_mul_mat} over the
 * 3-D {@code attn_k_b} (q_nope absorption, models/deepseek2.cpp:324) and {@code attn_v_b} (the {@code v_mla} expansion
 * of {@code build_attn}).
 *
 * <p>Each pair's activations are quantized once for the weight's vec-dot ({@link CpuActivationQuantizer}); the
 * products then run as one {@link CpuParallel#parallelFor} over (head, row block) tasks, each walking the step's
 * tokens so a head's weight block is reused while it is cached. The weight must carry no input transform and no
 * output scale; the mixers check that at bind, because these products, like llama.cpp's, apply none.</p>
 */
public final class CpuMlaHeadProducts {

    /** Multiply-adds a parallel task should at least perform (the {@link CpuMatVec} task size). */
    static final long MIN_TASK_WORK = 1L << 15;
    /** Activation values a quantization task should at least convert. */
    static final int MIN_QUANT_VALUES = 1 << 13;

    private CpuMlaHeadProducts() {
    }

    /**
     * Runs the products. Pair {@code (t, h)} reads {@code cols = w.cols()} activations at
     * {@code x[xOff + t*xTokenStride + h*xHeadStride]} and writes {@code R} outputs at
     * {@code y[yOff + t*yTokenStride + h*yHeadStride]}.
     *
     * @param w           the per-head weight of {@code nHead * rowsPerHead} rows
     * @param rowsPerHead {@code R}, the output width of one head
     * @throws IllegalArgumentException when the weight is not {@code nHead} stacked heads of {@code rowsPerHead}
     *                                  rows, the counts are not positive, or a view leaves its array
     * @throws UnsupportedOperationException for a weight type without a CPU vec-dot
     */
    public static void run(CpuTensorRef w, int nHead, int rowsPerHead, int nTokens,
                    float[] x, int xOff, int xTokenStride, int xHeadStride,
                    float[] y, int yOff, int yTokenStride, int yHeadStride) {
        Objects.requireNonNull(w, "weight");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (nHead < 1 || rowsPerHead < 1 || nTokens < 1) {
            throw new IllegalArgumentException("per-head product needs positive heads, rows and tokens, got "
                    + nHead + ", " + rowsPerHead + ", " + nTokens);
        }
        if (w.rows() != (long) nHead * rowsPerHead || w.cols() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(w + " is not " + nHead + " stacked heads of " + rowsPerHead + " rows");
        }
        final int cols = (int) w.cols();
        checkView("input", x, xOff, xTokenStride, xHeadStride, nTokens, nHead, cols);
        checkView("output", y, yOff, yTokenStride, yHeadStride, nTokens, nHead, rowsPerHead);
        final CpuVecDot dot = CpuMatVec.vecDot(w.type());

        final int pairs = nTokens * nHead;
        final QuantizedActivations[] acts = new QuantizedActivations[pairs];
        CpuParallel.parallelFor(pairs, Math.max(1, MIN_QUANT_VALUES / cols), (from, to) -> {
            for (int s = from; s < to; s++) {
                int t = s / nHead;
                int h = s % nHead;
                acts[s] = CpuActivationQuantizer.quantizeFor(w.type(), x, xOff + t * xTokenStride + h * xHeadStride,
                        cols);
            }
        });

        long perRow = (long) cols * nTokens;
        final int block = (int) Math.min(rowsPerHead, Math.max(1L, (MIN_TASK_WORK + perRow - 1) / perRow));
        final int blocks = (rowsPerHead + block - 1) / block;
        CpuParallel.parallelFor(nHead * blocks, 1, (from, to) -> {
            for (int i = from; i < to; i++) {
                int h = i / blocks;
                int r0 = (i % blocks) * block;
                int nr = Math.min(block, rowsPerHead - r0);
                long row = (long) h * rowsPerHead + r0;
                for (int t = 0; t < nTokens; t++) {
                    dot.dotRows(w, row, nr, acts[t * nHead + h], y, yOff + t * yTokenStride + h * yHeadStride + r0);
                }
            }
        });
    }

    private static void checkView(String what, float[] a, int off, int tokenStride, int headStride, int nTokens,
                                  int nHead, int width) {
        if (off < 0 || tokenStride < 0 || headStride < 0) {
            throw new IllegalArgumentException(what + " view offset and strides must be non-negative, got " + off
                    + ", " + tokenStride + ", " + headStride);
        }
        long end = off + (long) (nTokens - 1) * tokenStride + (long) (nHead - 1) * headStride + width;
        if (end > a.length) {
            throw new IllegalArgumentException(what + " view of " + nTokens + " tokens x " + nHead + " heads x "
                    + width + " ends at " + end + ", beyond the array length " + a.length);
        }
    }
}
