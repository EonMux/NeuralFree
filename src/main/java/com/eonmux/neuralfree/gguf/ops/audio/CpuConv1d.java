package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;

/**
 * CPU math of the wavtokenizer-dec convolutions, twins of {@link AudioKernels#CONV1D_SAME} and
 * {@link AudioKernels#CONV1D_DEPTHWISE}: {@code ggml_conv_1d_ph} / {@code ggml_conv_1d_dw_ph} with stride 1
 * (ggml.c:4639-4699; im2col tap {@code t + k*dil - K/2}, ggml-cpu/ops.cpp:6802-6875), zero outside the signal, the
 * bias added after the dot product. Pure and stateless; the caller owns every array.
 *
 * <p>Signals are channel-major: element {@code (c, t)} at {@code v[off + c*len + t]}. The output length is
 * {@link AudioKernels#outLen}, {@code L} for an odd kernel width and dilation 1.
 */
public final class CpuConv1d {

    /** Output elements per parallel task. */
    private static final int GRAIN = 1024;

    private CpuConv1d() {
    }

    /**
     * Twin of {@link AudioKernels#CONV1D_SAME}: with {@code p = K/2} and {@code OL = outLen(L, K, dil)},
     * {@code y[yOff + oc*OL + t] = (sum_{ic asc} sum_{k asc, 0 <= t+k*dil-p < L} w[(oc*IC + ic)*K + k]
     * * x[xOff + ic*L + t+k*dil-p]) + b[oc]}.
     *
     * @param w weight {@code [OC][IC][K]} (ggml {@code {K, IC, OC}})
     * @param b bias {@code [OC]}
     * @return the output length {@code OL}
     * @throws IllegalArgumentException on a bad shape, overlapping input and output views or an array too short
     */
    public static int same(int len, int inChannels, int outChannels, int k, int dil,
                           float[] x, int xOff, float[] w, float[] b, float[] y, int yOff) {
        final int ol = AudioKernels.checkConvShape(len, inChannels, outChannels, k, dil);
        requireSpan("x", x, xOff, (long) inChannels * len);
        requireSpan("w", w, 0, (long) outChannels * inChannels * k);
        requireSpan("b", b, 0, outChannels);
        requireSpan("y", y, yOff, (long) outChannels * ol);
        requireDisjoint(x, xOff, (long) inChannels * len, y, yOff, (long) outChannels * ol);
        final int pad = k / 2;
        CpuParallel.parallelFor(outChannels * ol, GRAIN, (from, to) -> {
            for (int idx = from; idx < to; idx++) {
                int oc = idx / ol;
                int t = idx - oc * ol;
                int wOc = oc * inChannels * k;
                float acc = 0.0f;
                for (int ic = 0; ic < inChannels; ic++) {
                    int xRow = xOff + ic * len;
                    int wRow = wOc + ic * k;
                    for (int j = 0; j < k; j++) {
                        int src = t + j * dil - pad;
                        if (src >= 0 && src < len) {
                            acc += w[wRow + j] * x[xRow + src];
                        }
                    }
                }
                y[yOff + idx] = acc + b[oc];
            }
        });
        return ol;
    }

    /**
     * Twin of {@link AudioKernels#CONV1D_DEPTHWISE}: with {@code p = K/2} and {@code OL = outLen(L, K, 1)},
     * {@code y[yOff + c*OL + t] = (sum_{k asc, 0 <= t+k-p < L} w[c*K + k] * x[xOff + c*L + t+k-p]) + b[c]}.
     *
     * @param w weight {@code [C][K]} (ggml {@code {K, 1, C}})
     * @param b bias {@code [C]}
     * @return the output length {@code OL}
     * @throws IllegalArgumentException on a bad shape, overlapping input and output views or an array too short
     */
    public static int depthwise(int len, int channels, int k,
                                float[] x, int xOff, float[] w, float[] b, float[] y, int yOff) {
        final int ol = AudioKernels.checkDepthwiseShape(len, channels, k);
        requireSpan("x", x, xOff, (long) channels * len);
        requireSpan("w", w, 0, (long) channels * k);
        requireSpan("b", b, 0, channels);
        requireSpan("y", y, yOff, (long) channels * ol);
        requireDisjoint(x, xOff, (long) channels * len, y, yOff, (long) channels * ol);
        final int pad = k / 2;
        CpuParallel.parallelFor(channels * ol, GRAIN, (from, to) -> {
            for (int idx = from; idx < to; idx++) {
                int c = idx / ol;
                int t = idx - c * ol;
                int xRow = xOff + c * len;
                int wRow = c * k;
                float acc = 0.0f;
                for (int j = 0; j < k; j++) {
                    int src = t + j - pad;
                    if (src >= 0 && src < len) {
                        acc += w[wRow + j] * x[xRow + src];
                    }
                }
                y[yOff + idx] = acc + b[c];
            }
        });
        return ol;
    }

    /** Rejects input and output views of one array whose element ranges intersect (the kernels forbid overlap). */
    private static void requireDisjoint(float[] x, long xOff, long xCount, float[] y, long yOff, long yCount) {
        if (x == y && xOff < yOff + yCount && yOff < xOff + xCount) {
            throw new IllegalArgumentException("conv1d input [" + xOff + ", " + (xOff + xCount) + ") and output ["
                    + yOff + ", " + (yOff + yCount) + ") overlap in one array");
        }
    }

    private static void requireSpan(String name, float[] a, long off, long count) {
        if (a == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (off < 0 || off + count > a.length) {
            throw new IllegalArgumentException(name + " needs elements [" + off + ", " + (off + count)
                    + ") but holds " + a.length + " floats");
        }
    }
}
