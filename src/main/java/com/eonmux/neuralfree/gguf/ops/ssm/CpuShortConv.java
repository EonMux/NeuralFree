package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU math of the LFM2 short convolution between its projections ({@code bcx = in_proj(x)} and
 * {@code out_proj(y)}): the causal form with conv state (models/lfm-base.cpp:9-87) and the non-causal centered
 * form of the decision/encoder graph (models/lfm2.cpp:349-380), plus the CPU twin of
 * {@link SsmKernels#SHORTCONV_CENTERED}. Pure and stateless; the caller owns every array.
 *
 * <p>{@code bcx} rows are {@code [b | c | x]}, {@code nEmbd} floats each, at {@code bcx[bcxOff + t*bcxStride]}.
 * The GPU path is the same sequence of operations: {@code LLM_GLU MUL(b, x)}, the conv kernel, then
 * {@code LLM_GLU MUL(c, conv)}.
 */
public final class CpuShortConv {

    /** Elements per parallel task of the element-wise loops. */
    private static final int GRAIN = 4096;

    private CpuShortConv() {
    }

    /**
     * Causal LFM2 short conv: {@code bx = b*x}, {@code conv = ssm_conv(concat(state, bx), w)} (no bias, no
     * activation) with the conv-state update of {@link CpuSsmConv#forward}, and
     * {@code out[outOff + t*outStride + c] = c * conv}.
     *
     * @param dConv        kernel width ({@code n_shortconv_l_cache}); the state keeps {@code dConv - 1} steps
     * @param w            conv weight {@code w[c*dConv + k]}
     * @param convState    state with layout {@code conv[c][j]}
     * @param stateRowBase float index of the state row
     * @param bx           scratch of at least {@code nTokens * nEmbd} floats, overwritten
     * @throws IllegalArgumentException on a bad shape, aliased {@code bcx}/{@code bx}/{@code out} or an array too
     *                                  short for the indices
     */
    public static void gatedCausal(int nEmbd, int dConv, int nTokens,
                                   float[] bcx, int bcxOff, int bcxStride,
                                   float[] w, float[] convState, int stateRowBase,
                                   float[] bx, float[] out, int outOff, int outStride) {
        SsmKernels.checkConvShape(nEmbd, dConv);
        requireTokens(nTokens);
        requireDistinct(bcx, bx, out);
        if (nTokens == 0) {
            return;
        }
        multiplyChunks(nEmbd, nTokens, bcx, bcxOff, bcxStride, 0, 2, bx, 0, nEmbd);
        CpuSsmConv.forward(nEmbd, dConv, nTokens, 0, bx, 0, nEmbd, w, null, convState, stateRowBase, 0,
                out, outOff, outStride);
        gateInPlace(nEmbd, nTokens, bcx, bcxOff + nEmbd, bcxStride, out, outOff, outStride);
    }

    /**
     * Centered (non-causal) LFM2 short conv without state: {@code bx = b*x},
     * {@code conv = centered(bx, w)} (see {@link #centered}), {@code out[outOff + t*outStride + c] = c * conv}.
     *
     * @param bx scratch of at least {@code nTokens * nEmbd} floats, overwritten
     * @throws IllegalArgumentException on a bad shape, aliased {@code bcx}/{@code bx}/{@code out} or an array too
     *                                  short for the indices
     */
    public static void gatedCentered(int nEmbd, int dConv, int nTokens,
                                     float[] bcx, int bcxOff, int bcxStride,
                                     float[] w, float[] bx, float[] out, int outOff, int outStride) {
        SsmKernels.checkCenteredShape(nEmbd, dConv);
        requireTokens(nTokens);
        requireDistinct(bcx, bx, out);
        if (nTokens == 0) {
            return;
        }
        multiplyChunks(nEmbd, nTokens, bcx, bcxOff, bcxStride, 0, 2, bx, 0, nEmbd);
        centered(nEmbd, dConv, nTokens, bx, 0, nEmbd, w, out, outOff, outStride);
        gateInPlace(nEmbd, nTokens, bcx, bcxOff + nEmbd, bcxStride, out, outOff, outStride);
    }

    /**
     * Twin of {@link SsmKernels#SHORTCONV_CENTERED}: with {@code p = (dConv-1)/2},
     * {@code out[outOff + t*outStride + c] = x[t][c]*w[c*dConv+p] + sum_{k != p, 0 <= t+k-p < nTokens} x[t+k-p][c]*w[c*dConv+k]}
     * where {@code x[t][c] = x[xOff + t*xStride + c]}; the center tap first, the others in ascending {@code k}
     * (llama.cpp {@code bx*tap1 + prv*tap0 + nxt*tap2}).
     *
     * @throws IllegalArgumentException on a bad shape, aliased input/output or an array too short
     */
    public static void centered(int nCh, int dConv, int nTokens,
                                float[] x, int xOff, int xStride,
                                float[] w, float[] out, int outOff, int outStride) {
        SsmKernels.checkCenteredShape(nCh, dConv);
        requireTokens(nTokens);
        if (nTokens == 0) {
            return;
        }
        if (x == out) {
            throw new IllegalArgumentException("centered short conv input and output must be distinct arrays");
        }
        CpuSsmConv.requireRows("x", x, xOff, xStride, nTokens, nCh);
        CpuSsmConv.requireRows("out", out, outOff, outStride, nTokens, nCh);
        CpuSsmConv.requireSpan("w", w, 0, (long) nCh * dConv);
        final int pad = (dConv - 1) / 2;
        CpuParallel.parallelFor(nTokens * nCh, GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                int t = i / nCh;
                int c = i - t * nCh;
                int wBase = c * dConv;
                float sum = x[xOff + t * xStride + c] * w[wBase + pad];
                for (int k = 0; k < dConv; k++) {
                    int src = t + k - pad;
                    if (k != pad && src >= 0 && src < nTokens) {
                        sum += x[xOff + src * xStride + c] * w[wBase + k];
                    }
                }
                out[outOff + t * outStride + c] = sum;
            }
        });
    }

    /**
     * The gated forms read {@code bcx} while writing {@code bx} and {@code out}, so the three arrays must differ.
     *
     * @throws IllegalArgumentException when two of them are the same array
     */
    private static void requireDistinct(float[] bcx, float[] bx, float[] out) {
        if (bx == bcx || out == bcx || out == bx) {
            throw new IllegalArgumentException("short conv bcx, bx scratch and output must be distinct arrays");
        }
    }

    private static void requireTokens(int nTokens) {
        if (nTokens < 0) {
            throw new IllegalArgumentException("short conv n_tokens must be non-negative, got " + nTokens);
        }
    }

    /** {@code dst[dstOff + t*dstStride + c] = row[chunkA][c] * row[chunkB][c]} over the {@code bcx} rows. */
    private static void multiplyChunks(int nEmbd, int nTokens, float[] bcx, int bcxOff, int bcxStride,
                                       int chunkA, int chunkB, float[] dst, int dstOff, int dstStride) {
        CpuSsmConv.requireRows("bcx", bcx, bcxOff, bcxStride, nTokens, 3 * nEmbd);
        CpuSsmConv.requireRows("bx scratch", dst, dstOff, dstStride, nTokens, nEmbd);
        final int offA = bcxOff + chunkA * nEmbd;
        final int offB = bcxOff + chunkB * nEmbd;
        CpuParallel.parallelFor(nTokens * nEmbd, GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                int t = i / nEmbd;
                int c = i - t * nEmbd;
                dst[dstOff + t * dstStride + c] = bcx[offA + t * bcxStride + c] * bcx[offB + t * bcxStride + c];
            }
        });
    }

    /** {@code out[t][c] = gate[t][c] * out[t][c]} (ggml_mul(c, conv)). */
    private static void gateInPlace(int nEmbd, int nTokens, float[] gate, int gateOff, int gateStride,
                                    float[] out, int outOff, int outStride) {
        CpuParallel.parallelFor(nTokens * nEmbd, GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                int t = i / nEmbd;
                int c = i - t * nEmbd;
                int o = outOff + t * outStride + c;
                out[o] = gate[gateOff + t * gateStride + c] * out[o];
            }
        });
    }
}
