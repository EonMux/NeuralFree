package com.eonmux.neuralfree.gguf.ops.heads;

import com.eonmux.neuralfree.gguf.arch.PoolingKind;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.flow.FlowKernels;

/**
 * Host row math of the output heads: embedding pooling ({@link PoolingHead}), row selection (the CPU twin of
 * {@link FlowKernels#GATHER_ROWS}) and the graniteswitch switched-LoRA delta (the CPU twin of
 * {@link HeadsKernels#SWITCHED_LORA_ADD}). Each twin keeps its kernel's index math and summation order. Pure and
 * stateless; the caller owns every array.
 *
 * <p>Rows are token-major: element {@code (t, c)} of a view is {@code v[off + t*stride + c]}.
 */
public final class CpuPooling {

    /** Columns per parallel task of the pooling loop. */
    private static final int POOL_GRAIN = 256;
    /** Elements per parallel task of the gather and LoRA up-projection loops. */
    private static final int ELEMENT_GRAIN = 4096;
    /** (token, rank row) dot products per parallel task of the LoRA down-projection. */
    private static final int DOT_GRAIN = 16;

    private CpuPooling() {
    }

    /**
     * The row pooling applied for a model's pooling type (llama-graph.cpp:4120-4144 build_pooling, and the CLS
     * row choice of llm_graph_input_cls, llama-graph.cpp:367-371): {@link PoolingKind#RANK} pools by MEAN when the
     * classifier pooling ({@code pooling_type_cls}) is MEAN, otherwise by LAST for causal attention and CLS for
     * non-causal attention; every other pooling is returned unchanged.
     *
     * @param poolingCls the model's classifier pooling, or null when unspecified
     * @throws IllegalArgumentException when {@code pooling} is null
     */
    public static PoolingKind rowPooling(PoolingKind pooling, PoolingKind poolingCls, boolean causalAttn) {
        if (pooling == null) {
            throw new IllegalArgumentException("pooling is null");
        }
        if (pooling != PoolingKind.RANK) {
            return pooling;
        }
        if (poolingCls == PoolingKind.MEAN) {
            return PoolingKind.MEAN;
        }
        return causalAttn ? PoolingKind.LAST : PoolingKind.CLS;
    }

    /**
     * Pools one sequence of {@code nTokens} rows (llama-graph.cpp:4121-4135 build_pooling; MEAN as the product with
     * the {@code 1/n} column of {@code inp_mean}):
     * <ul>
     *   <li>NONE: {@code out[outOff + t*nEmbd + c] = x[t][c]} for every row (rows packed);</li>
     *   <li>MEAN: {@code out[outOff + c] = sum_t x[t][c] * (1/nTokens)}, t ascending;</li>
     *   <li>CLS: {@code out[outOff + c] = x[0][c]}; LAST: {@code out[outOff + c] = x[nTokens-1][c]}.</li>
     * </ul>
     *
     * @param mode a row pooling; resolve {@link PoolingKind#RANK} with {@link #rowPooling} first
     * @throws IllegalArgumentException for RANK, {@code nTokens < 1}, {@code nEmbd < 1}, overlapping input and output
     *                                  views or a view that does not fit its array
     */
    public static void pool(PoolingKind mode, int nTokens, int nEmbd, float[] x, int xOff, int xStride,
                            float[] out, int outOff) {
        if (mode == null || mode == PoolingKind.RANK) {
            throw new IllegalArgumentException("pooling must be NONE, MEAN, CLS or LAST, got " + mode
                    + "; resolve RANK with rowPooling first");
        }
        if (nTokens < 1 || nEmbd < 1) {
            throw new IllegalArgumentException("pooling needs n_tokens >= 1 and n_embd >= 1: n_tokens=" + nTokens
                    + " n_embd=" + nEmbd);
        }
        final int outRows = mode == PoolingKind.NONE ? nTokens : 1;
        requireRows("x", x, xOff, xStride, nTokens, nEmbd);
        requireRows("out", out, outOff, nEmbd, outRows, nEmbd);
        requireDisjoint("x", x, xOff, lastIndex(xOff, xStride, nTokens, nEmbd),
                "out", out, outOff, lastIndex(outOff, nEmbd, outRows, nEmbd));
        final float inv = 1.0f / (float) nTokens;
        CpuParallel.parallelFor(nEmbd, POOL_GRAIN, (from, to) -> {
            for (int c = from; c < to; c++) {
                int col = xOff + c;
                switch (mode) {
                    case NONE:
                        for (int t = 0; t < nTokens; t++) {
                            out[outOff + t * nEmbd + c] = x[col + t * xStride];
                        }
                        break;
                    case MEAN: {
                        float acc = 0.0f;
                        for (int t = 0; t < nTokens; t++) {
                            acc += x[col + t * xStride] * inv;
                        }
                        out[outOff + c] = acc;
                        break;
                    }
                    case CLS:
                        out[outOff + c] = x[col];
                        break;
                    case LAST:
                        out[outOff + c] = x[col + (nTokens - 1) * xStride];
                        break;
                    default:
                        throw new IllegalStateException("unresolved pooling " + mode);
                }
            }
        });
    }

    /**
     * Twin of {@link FlowKernels#GATHER_ROWS}:
     * {@code dst[dstOff + r*dstStride + c] = rows[r] >= 0 ? src[srcOff + rows[r]*srcStride + c] : 0} for
     * {@code r < nRows}, {@code c < rowLen}.
     *
     * @throws IllegalArgumentException for a non-positive dimension, overlapping source and destination views, a
     *                                  source row beyond {@code src} or a view that does not fit its array
     */
    public static void gatherRows(int nRows, int rowLen, int[] rows, float[] src, int srcOff, int srcStride,
                                  float[] dst, int dstOff, int dstStride) {
        if (nRows < 1 || rowLen < 1) {
            throw new IllegalArgumentException("gather needs n_rows >= 1 and row_len >= 1: n_rows=" + nRows
                    + " row_len=" + rowLen);
        }
        if (rows == null || rows.length < nRows) {
            throw new IllegalArgumentException("gather needs " + nRows + " row indices, got "
                    + (rows == null ? "null" : String.valueOf(rows.length)));
        }
        requireRows("dst", dst, dstOff, dstStride, nRows, rowLen);
        final long dstLast = lastIndex(dstOff, dstStride, nRows, rowLen);
        for (int r = 0; r < nRows; r++) {
            if (rows[r] >= 0) {
                long first = srcOff + (long) rows[r] * srcStride;
                requireRows("src row " + rows[r], src, first, rowLen, 1, rowLen);
                requireDisjoint("src row " + rows[r], src, first, first + rowLen - 1, "dst", dst, dstOff, dstLast);
            }
        }
        CpuParallel.parallelFor(nRows * rowLen, ELEMENT_GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                int r = i / rowLen;
                int c = i - r * rowLen;
                int s = rows[r];
                dst[dstOff + r * dstStride + c] = s >= 0 ? src[srcOff + s * srcStride + c] : 0.0f;
            }
        });
    }

    /**
     * Twin of {@link HeadsKernels#SWITCHED_LORA_ADD} (both stages): per token {@code t} with slot
     * {@code s = clamp(slots[t], 0, nSlots-1)}
     * <pre>
     *   tmp[tmpOff + t*rank + r] = sum_i a[(s*rank + r)*inDim + i] * x[xOff + t*xStride + i]
     *   y[yOff + t*yStride + o] += sum_{r ascending} b[(s*outDim + o)*rank + r] * tmp[tmpOff + t*rank + r]
     * </pre>
     * (models/granite-switch.cpp build_switched_lora_delta: {@code lora_a {inDim, rank, nSlots}},
     * {@code lora_b {rank, outDim, nSlots}}, two mul_mat_id with the slot as expert, then ggml_add onto the base
     * projection).
     *
     * @param tmp scratch of at least {@code tmpOff + nTokens*rank} floats, overwritten
     * @throws IllegalArgumentException on a bad shape, overlapping {@code x}, {@code tmp} and {@code y} views or a view
     *                                  that does not fit its array
     */
    public static void switchedLoraAdd(int nTokens, int inDim, int outDim, int rank, int nSlots, int[] slots,
                                       float[] a, float[] b, float[] x, int xOff, int xStride,
                                       float[] tmp, int tmpOff, float[] y, int yOff, int yStride) {
        HeadsKernels.checkLoraShape(inDim, outDim, rank, nSlots);
        if (nTokens < 1) {
            throw new IllegalArgumentException("switched LoRA needs n_tokens >= 1, got " + nTokens);
        }
        if (slots == null || slots.length < nTokens) {
            throw new IllegalArgumentException("switched LoRA needs " + nTokens + " slot ids, got "
                    + (slots == null ? "null" : String.valueOf(slots.length)));
        }
        requireSpan("lora_a", a, (long) nSlots * rank * inDim);
        requireSpan("lora_b", b, (long) nSlots * outDim * rank);
        requireRows("x", x, xOff, xStride, nTokens, inDim);
        requireRows("tmp", tmp, tmpOff, rank, nTokens, rank);
        requireRows("y", y, yOff, yStride, nTokens, outDim);
        final long xLast = lastIndex(xOff, xStride, nTokens, inDim);
        final long tmpLast = lastIndex(tmpOff, rank, nTokens, rank);
        final long yLast = lastIndex(yOff, yStride, nTokens, outDim);
        requireDisjoint("x", x, xOff, xLast, "tmp", tmp, tmpOff, tmpLast);
        requireDisjoint("x", x, xOff, xLast, "y", y, yOff, yLast);
        requireDisjoint("tmp", tmp, tmpOff, tmpLast, "y", y, yOff, yLast);

        CpuParallel.parallelFor(nTokens * rank, DOT_GRAIN, (from, to) -> {
            for (int g = from; g < to; g++) {
                int t = g / rank;
                int r = g - t * rank;
                int aRow = (slot(slots[t], nSlots) * rank + r) * inDim;
                int xRow = xOff + t * xStride;
                float acc = 0.0f;
                for (int i = 0; i < inDim; i++) {
                    acc += a[aRow + i] * x[xRow + i];
                }
                tmp[tmpOff + t * rank + r] = acc;
            }
        });
        CpuParallel.parallelFor(nTokens * outDim, ELEMENT_GRAIN, (from, to) -> {
            for (int idx = from; idx < to; idx++) {
                int t = idx / outDim;
                int o = idx - t * outDim;
                int bRow = (slot(slots[t], nSlots) * outDim + o) * rank;
                int tRow = tmpOff + t * rank;
                float acc = 0.0f;
                for (int r = 0; r < rank; r++) {
                    acc += b[bRow + r] * tmp[tRow + r];
                }
                int yi = yOff + t * yStride + o;
                y[yi] = y[yi] + acc;
            }
        });
    }

    /** Adapter slot clamped to {@code [0, nSlots)} ({@code ggml_clamp(slot, 0, n_adapters)}). */
    static int slot(int s, int nSlots) {
        return s < 0 ? 0 : Math.min(s, nSlots - 1);
    }

    static void requireRows(String name, float[] a, long off, long stride, int rows, int cols) {
        if (a == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (off < 0 || stride < 0 || (rows > 1 && stride < cols)) {
            throw new IllegalArgumentException(name + ": bad view off=" + off + " stride=" + stride + " for "
                    + cols + " columns");
        }
        long last = lastIndex(off, stride, rows, cols);
        if (rows > 0 && last >= a.length) {
            throw new IllegalArgumentException(name + " needs index " + last + " but holds " + a.length + " floats");
        }
    }

    /** Index of the last element of a {@code rows x cols} view at {@code off} with row stride {@code stride}. */
    static long lastIndex(long off, long stride, int rows, int cols) {
        return off + (long) (rows - 1) * stride + cols - 1;
    }

    /**
     * Rejects two views of one array whose element spans {@code [firstA, lastA]} and {@code [firstB, lastB]}
     * intersect (the kernels require their input, scratch and output views not to overlap).
     */
    static void requireDisjoint(String nameA, float[] a, long firstA, long lastA,
                                String nameB, float[] b, long firstB, long lastB) {
        if (a == b && firstA <= lastB && firstB <= lastA) {
            throw new IllegalArgumentException(nameA + " [" + firstA + ", " + lastA + "] and " + nameB + " [" + firstB
                    + ", " + lastB + "] overlap in one array");
        }
    }

    private static void requireSpan(String name, float[] a, long count) {
        if (a == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (count > a.length) {
            throw new IllegalArgumentException(name + " needs " + count + " floats but holds " + a.length);
        }
    }
}
