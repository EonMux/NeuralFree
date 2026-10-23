package com.eonmux.neuralfree.gguf.ops.audio;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout and launch geometry of the audio-family kernels ({@code kernels/llm/audio.cl} and the
 * GLSL twin {@code conv1d.comp}): the stride-1 half-padded convolutions of wavtokenizer-dec
 * ({@code ggml_conv_1d_ph}, {@code ggml_conv_1d_dw_ph}). Signals are channel-major, element {@code (c, t)} at
 * {@code v[off + c*len + t]}; offsets are in floats; every scalar is a 32-bit {@code int}; buffers follow the
 * scalars in the order listed.
 *
 * <ul>
 *   <li>{@link #CONV1D_SAME} {@code (L*, IC, OC, K, dil, x_off, y_off; x, w, b, y)}: dense conv, weight
 *       {@code w[OC][IC][K]}, bias {@code b[OC]}, output {@code y[OC][OL]}.
 *       CPU twin {@link CpuConv1d#same}.</li>
 *   <li>{@link #CONV1D_DEPTHWISE} {@code (L*, C, K, x_off, y_off; x, w, b, y)}: depthwise conv, weight
 *       {@code w[C][K]}, bias {@code b[C]}, output {@code y[C][OL]}. CPU twin {@link CpuConv1d#depthwise}.</li>
 * </ul>
 * {@code OL} is {@link #outLen}. Both kernels run one work-item per output element: {@link #convGlobal}.
 * {@code *} marks the sequence length, re-set per decoded code sequence at the matching {@code ARG_*} index.
 */
public final class AudioKernels {

    public static final String CONV1D_SAME = "LLM_Conv1dSame";
    public static final String CONV1D_DEPTHWISE = "LLM_Conv1dDepthwise";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(Arrays.asList(CONV1D_SAME, CONV1D_DEPTHWISE));

    /** {@link #CONV1D_SAME} {@code L}. */
    public static final int ARG_SAME_L = 0;
    /** {@link #CONV1D_SAME} argument count (7 scalars, 4 buffers). */
    public static final int SAME_ARG_COUNT = 11;

    /** {@link #CONV1D_DEPTHWISE} {@code L}. */
    public static final int ARG_DW_L = 0;
    /** {@link #CONV1D_DEPTHWISE} argument count (5 scalars, 4 buffers). */
    public static final int DW_ARG_COUNT = 9;

    /** Local size of both convolutions. */
    public static final int CONV_LOCAL = LlmKernelConstants.LLM_DIMX;

    private AudioKernels() {
    }

    /**
     * Output length of a stride-1 convolution padded by {@code K/2}: {@code L + 2*(K/2) - dil*(K-1)}
     * (ggml_calc_conv_output_size with {@code p = K/2}); {@code L} for an odd {@code K} and {@code dil = 1}.
     */
    public static int outLen(int len, int k, int dil) {
        return len + 2 * (k / 2) - dil * (k - 1);
    }

    /**
     * Validates a {@link #CONV1D_SAME} shape for the kernel and {@link CpuConv1d#same}.
     *
     * @return the output length {@link #outLen}
     * @throws IllegalArgumentException when a dimension is not positive, the output length is not positive, or the
     *                                  signal, output or weight {@code [OC][IC][K]} exceeds the 32-bit index range of
     *                                  the GLSL twin
     */
    public static int checkConvShape(int len, int inChannels, int outChannels, int k, int dil) {
        if (len <= 0 || inChannels <= 0 || outChannels <= 0 || k <= 0 || dil <= 0) {
            throw new IllegalArgumentException("conv1d dimensions must be positive: L=" + len + " IC=" + inChannels
                    + " OC=" + outChannels + " K=" + k + " dil=" + dil);
        }
        int ol = checkedOutLen(len, k, dil);
        if ((long) inChannels * len > Integer.MAX_VALUE || (long) outChannels * ol > Integer.MAX_VALUE
                || (long) outChannels * inChannels * k > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("conv1d of L=" + len + " IC=" + inChannels + " OC=" + outChannels
                    + " K=" + k + " exceeds the 32-bit index range");
        }
        return ol;
    }

    /**
     * Validates a {@link #CONV1D_DEPTHWISE} shape (dilation 1) for the kernel and {@link CpuConv1d#depthwise}.
     *
     * @return the output length {@code outLen(len, k, 1)}
     * @throws IllegalArgumentException when a dimension is not positive, the output length is not positive, or the
     *                                  signal, output or weight {@code [C][K]} exceeds the 32-bit index range of the
     *                                  GLSL twin
     */
    public static int checkDepthwiseShape(int len, int channels, int k) {
        if (len <= 0 || channels <= 0 || k <= 0) {
            throw new IllegalArgumentException("depthwise conv1d dimensions must be positive: L=" + len + " C="
                    + channels + " K=" + k);
        }
        int ol = checkedOutLen(len, k, 1);
        if ((long) channels * len > Integer.MAX_VALUE || (long) channels * ol > Integer.MAX_VALUE
                || (long) channels * k > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("depthwise conv1d of L=" + len + " C=" + channels + " K=" + k
                    + " exceeds the 32-bit index range");
        }
        return ol;
    }

    /** {@link #outLen} in 64-bit arithmetic, rejected when it is not a positive {@code int}. */
    private static int checkedOutLen(int len, int k, int dil) {
        long ol = (long) len + 2L * (k / 2) - (long) dil * (k - 1);
        if (ol <= 0 || ol > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("conv1d of L=" + len + " with K=" + k + " dil=" + dil
                    + " has no valid output (output length " + ol + ")");
        }
        return (int) ol;
    }

    /** Global size of either convolution: one work-item per output element, rounded up to {@link #CONV_LOCAL}. */
    public static long convGlobal(int outChannels, int outLen) {
        if (outChannels <= 0 || outLen <= 0) {
            throw new IllegalArgumentException("conv1d needs positive output channels and length: channels="
                    + outChannels + " OL=" + outLen);
        }
        long n = (long) outChannels * outLen;
        return (n + CONV_LOCAL - 1) / CONV_LOCAL * CONV_LOCAL;
    }
}
