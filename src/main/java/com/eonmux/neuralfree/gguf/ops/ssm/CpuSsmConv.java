package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;

/**
 * CPU twin of {@link SsmKernels#SSM_CONV}: the causal depthwise conv1d of Mamba1/2, PLaMo2, GDN, KDA and LFM2
 * short convolution with its conv-state update (ggml_ssm_conv, ggml-cpu/ops.cpp:10022-10066, and the
 * concat / state copy of models/mamba-base.cpp:57-82). Pure and stateless; the caller owns every array.
 *
 * <p>Layouts: activations are token-major ({@code x[xOff + t*xStride + c]}), the weight is
 * {@code w[c*dConv + k]} (ggml {@code ssm_conv1d} {@code {d_conv, n_ch}}), and the conv state has the llama.cpp
 * layout {@code conv[c][j]}, {@code j in [0, dConv-1)}, {@code j = 0} oldest, with channel {@code c} of this call at
 * {@code convState[stateRowBase + (stateOff + c)*(dConv-1) + j]}.
 */
public final class CpuSsmConv {

    /** Channels per parallel task. */
    private static final int GRAIN = 64;

    private CpuSsmConv() {
    }

    /**
     * For each channel {@code c < nCh}: with {@code xs = concat(state[c][0..dConv-2], x[0..nTokens-1][c])},
     * {@code out[outOff + t*outStride + c] = silu?(sum_{k<dConv} xs[t+k] * w[c*dConv+k] (+ b[c]))} in ascending
     * {@code k}, then {@code state[c][j] = xs[nTokens + j]}.
     *
     * @param flags        {@link SsmKernels#CONV_FLAG_BIAS} | {@link SsmKernels#CONV_FLAG_SILU}
     * @param b            bias, read only with {@link SsmKernels#CONV_FLAG_BIAS} (may be null otherwise)
     * @param stateRowBase float index of the state row ({@code row * floats-per-sequence})
     * @param stateOff     channel offset of this call's slice inside the state row
     * @throws IllegalArgumentException on a bad shape, unknown flag bits or an array too short for the indices
     */
    public static void forward(int nCh, int dConv, int nTokens, int flags,
                               float[] x, int xOff, int xStride,
                               float[] w, float[] b,
                               float[] convState, int stateRowBase, int stateOff,
                               float[] out, int outOff, int outStride) {
        SsmKernels.checkConvShape(nCh, dConv);
        if (nTokens < 0) {
            throw new IllegalArgumentException("SSM conv n_tokens must be non-negative, got " + nTokens);
        }
        if ((flags & ~(SsmKernels.CONV_FLAG_BIAS | SsmKernels.CONV_FLAG_SILU)) != 0) {
            throw new IllegalArgumentException("unknown SSM conv flags 0x" + Integer.toHexString(flags));
        }
        final boolean hasBias = (flags & SsmKernels.CONV_FLAG_BIAS) != 0;
        final boolean hasSilu = (flags & SsmKernels.CONV_FLAG_SILU) != 0;
        final int steps = dConv - 1;
        if (nTokens > 0) {
            requireRows("x", x, xOff, xStride, nTokens, nCh);
            requireRows("out", out, outOff, outStride, nTokens, nCh);
        }
        requireSpan("w", w, 0, (long) nCh * dConv);
        if (hasBias) {
            requireSpan("b", b, 0, nCh);
        }
        requireSpan("conv state", convState, stateRowBase + (long) stateOff * steps, (long) nCh * steps);
        if (x == out) {
            throw new IllegalArgumentException("SSM conv input and output must be distinct arrays");
        }

        CpuParallel.parallelFor(nCh, GRAIN, (from, to) -> {
            for (int c = from; c < to; c++) {
                convChannel(c, dConv, nTokens, hasBias, hasSilu, x, xOff, xStride, w, b,
                        convState, stateRowBase + (stateOff + c) * steps, out, outOff, outStride);
            }
        });
    }

    private static void convChannel(int c, int dConv, int nTokens, boolean hasBias, boolean hasSilu,
                                    float[] x, int xOff, int xStride, float[] w, float[] b,
                                    float[] convState, int sBase, float[] out, int outOff, int outStride) {
        final int steps = dConv - 1;
        final int wBase = c * dConv;
        for (int t = 0; t < nTokens; t++) {
            float sum = 0.0f;
            for (int k = 0; k < dConv; k++) {
                int src = t + k - steps;
                float v = src >= 0 ? x[xOff + src * xStride + c] : convState[sBase + t + k];
                sum += v * w[wBase + k];
            }
            if (hasBias) {
                sum += b[c];
            }
            if (hasSilu) {
                sum = ActMath.silu(sum);
            }
            out[outOff + t * outStride + c] = sum;
        }
        for (int j = 0; j < steps; j++) {
            int src = nTokens + j - steps;
            convState[sBase + j] = src >= 0 ? x[xOff + src * xStride + c] : convState[sBase + nTokens + j];
        }
    }

    /**
     * Requires {@code a[off + t*stride + c]} to be in range for every {@code t < rows}, {@code c < cols}.
     *
     * @throws IllegalArgumentException naming the array and the first index out of range
     */
    static void requireRows(String name, float[] a, long off, long stride, int rows, int cols) {
        if (a == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (off < 0 || stride < 0 || (rows > 1 && stride < cols)) {
            throw new IllegalArgumentException(name + ": bad view off=" + off + " stride=" + stride + " for " + cols + " columns");
        }
        long last = off + (long) (rows - 1) * stride + cols - 1;
        if (rows > 0 && last >= a.length) {
            throw new IllegalArgumentException(name + " needs index " + last + " but holds " + a.length + " floats");
        }
    }

    /**
     * Requires {@code a[off .. off+count)} to be in range.
     *
     * @throws IllegalArgumentException naming the array
     */
    static void requireSpan(String name, float[] a, long off, long count) {
        if (a == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (off < 0 || off + count > a.length) {
            throw new IllegalArgumentException(name + " needs floats [" + off + ", " + (off + count) + ") but holds "
                    + a.length);
        }
    }
}
