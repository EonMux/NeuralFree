package com.eonmux.neuralfree.gguf.ops.vision;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout and launch geometry of the vision-family kernels ({@code kernels/llm/vision.cl}
 * and the GLSL twins {@code conv2d_depthwise.comp}, {@code pool_2d_avg.comp} and {@code rel_pos_attn.comp}):
 * the device kernels of the convolutional vision encoders. Feature maps are pixel-major F32, element
 * {@code (x, y, c)} at {@code v[off + (y*W + x)*C + c]}; offsets are in floats; every scalar is a 32-bit
 * {@code int}; buffers follow the scalars in the order listed.
 *
 * <ul>
 *   <li>{@link #CONV2D_DEPTHWISE} {@code (W*, H*, OW, OH, C, K, SY, SX, PY, PX, x_off, y_off; x, w, b, y)}:
 *       {@code ggml_conv_2d_dw} with a {@code K x K} kernel, weight {@code w[C][K*K]}, bias {@code b[C]},
 *       taps outside the map reading zero. CPU twin {@code MobileNetV5Maps.depthwise}.</li>
 *   <li>{@link #POOL_2D_AVG} {@code (W*, H*, OW, OH, C, KY, KX, x_off, y_off; x, y)}: {@code ggml_pool_2d}
 *       AVG over whole {@code KY x KX} windows at stride {@code KY}, {@code KX}, no padding. CPU twin
 *       {@code MobileNetV5Maps.avgPool}.</li>
 *   <li>{@link #REL_POS_ATTN} {@code (groups, side, n_head, dh, scale, q_stride, kv_stride, out_stride,
 *       q_off, kv_off, out_off; q, k, v, rw, rh, out)}: the bidirectional softmax attention of the
 *       DeepSeek-OCR SAM backbone, its score of query {@code tq} against key {@code tk} carrying the
 *       decomposed relative-position bias {@code q . rw[qx - kx + side - 1] + q . rh[qy - ky + side - 1]}.
 *       CPU twin {@code SamBackbone#attention}. One workgroup per (group, head, query).</li>
 * </ul>
 * The convolution and pool run one work-item per output element: {@link #elementGlobal}. The attention runs
 * one workgroup of {@link #attnLocal} per (group, head, query): {@link #attnGlobal}. {@code *} marks the
 * arguments re-set per dispatch, at the matching {@code ARG_*} index.
 */
public final class VisionKernels {

    public static final String CONV2D_DEPTHWISE = "LLM_Conv2dDepthwise";
    public static final String POOL_2D_AVG = "LLM_Pool2dAvg";
    public static final String REL_POS_ATTN = "LLM_RelPosAttn";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(Arrays.asList(
            CONV2D_DEPTHWISE, POOL_2D_AVG, REL_POS_ATTN));

    /** {@link #CONV2D_DEPTHWISE} {@code W}. */
    public static final int ARG_DW_W = 0;
    /** {@link #CONV2D_DEPTHWISE} argument count (12 scalars, 4 buffers). */
    public static final int DW_ARG_COUNT = 16;

    /** {@link #POOL_2D_AVG} {@code W}. */
    public static final int ARG_POOL_W = 0;
    /** {@link #POOL_2D_AVG} argument count (9 scalars, 2 buffers). */
    public static final int POOL_ARG_COUNT = 11;

    /** {@link #REL_POS_ATTN} {@code groups}. */
    public static final int ARG_ATTN_GROUPS = 0;
    /** {@link #REL_POS_ATTN} {@code scale}. */
    public static final int ARG_ATTN_SCALE = 4;
    /** {@link #REL_POS_ATTN} argument count (11 scalars, 6 buffers). */
    public static final int ATTN_ARG_COUNT = 17;

    /** Tokens per attention group whose probabilities {@link #REL_POS_ATTN} stages in shared memory. */
    public static final int ATTN_T_MAX = 4096;

    /** Local size of the element-wise vision kernels. */
    public static final int LOCAL = LlmKernelConstants.LLM_DIMX;

    private VisionKernels() {
    }

    /**
     * Global size of the element-wise kernels: one work-item per output element, rounded up to
     * {@link #LOCAL}.
     *
     * @throws IllegalArgumentException when the element count is not positive or exceeds the 32-bit index
     *                                  range of the GLSL twin
     */
    public static long elementGlobal(long elements) {
        if (elements <= 0 || elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("vision kernel elements must be positive and within the 32-bit"
                    + " index range, got " + elements);
        }
        return (elements + LOCAL - 1) / LOCAL * LOCAL;
    }

    /**
     * Global size of {@link #REL_POS_ATTN}: one workgroup per (group, head, query) at local size {@code local},
     * so {@code groups * n_head * side * side} workgroups of {@code local} work-items each.
     *
     * @throws IllegalArgumentException when the workgroup count exceeds the 32-bit range of the GLSL twin
     */
    public static long attnWorkItems(int groups, int nHead, int side, int local) {
        long count = attnGlobal(groups, nHead, side);
        return count * local;
    }

    /**
     * The local size of {@link #REL_POS_ATTN}: a power of two in {@code [32, local]} ({@code local} the
     * device's row-reduce size), as the workgroup reductions need.
     *
     * @throws IllegalArgumentException when {@code local} is smaller than 32
     */
    public static int attnLocal(int local) {
        int l = Integer.highestOneBit(Math.max(1, local));
        if (l < 32) {
            throw new IllegalArgumentException("the rel-pos attention needs a local size of at least 32, got "
                    + local);
        }
        return l;
    }

    /**
     * Validates a {@link #CONV2D_DEPTHWISE} shape for the kernel and {@code MobileNetV5Maps.depthwise}.
     *
     * @throws IllegalArgumentException when a dimension is not positive, an output side is not positive, or
     *                                  the map and output exceed the 32-bit index range of the GLSL twin
     */
    public static void checkDepthwise(int w, int h, int ow, int oh, int c, int k, int sy, int sx, int py, int px) {
        if (w <= 0 || h <= 0 || ow <= 0 || oh <= 0 || c <= 0 || k <= 0 || sy <= 0 || sx <= 0 || py < 0 || px < 0) {
            throw new IllegalArgumentException("depthwise conv2d dimensions must be positive (paddings"
                    + " non-negative): W=" + w + " H=" + h + " OW=" + ow + " OH=" + oh + " C=" + c + " K=" + k
                    + " SY=" + sy + " SX=" + sx + " PY=" + py + " PX=" + px);
        }
        if ((long) w * h * c > Integer.MAX_VALUE || (long) ow * oh * c > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("depthwise conv2d of " + w + "x" + h + " to " + ow + "x" + oh
                    + " at " + c + " channels exceeds the 32-bit index range");
        }
    }

    /**
     * Validates a {@link #POOL_2D_AVG} shape for the kernel and {@code MobileNetV5Maps.avgPool}: whole
     * {@code KY x KY}-high, {@code KX}-wide windows at stride {@code KY}, {@code KX} must tile the map exactly.
     *
     * @throws IllegalArgumentException when a dimension is not positive or the windows do not tile the map
     */
    public static void checkPool(int w, int h, int ow, int oh, int c, int ky, int kx) {
        if (w <= 0 || h <= 0 || ow <= 0 || oh <= 0 || c <= 0 || ky <= 0 || kx <= 0) {
            throw new IllegalArgumentException("pool2d dimensions must be positive: W=" + w + " H=" + h + " OW="
                    + ow + " OH=" + oh + " C=" + c + " KY=" + ky + " KX=" + kx);
        }
        if (ow != w / kx || oh != h / ky) {
            throw new IllegalArgumentException("pool2d needs whole " + ky + "x" + kx + " windows: " + w + "x" + h
                    + " pools to " + (w / kx) + "x" + (h / ky) + ", got " + ow + "x" + oh);
        }
        if ((long) w * h * c > Integer.MAX_VALUE || (long) ow * oh * c > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("pool2d of " + w + "x" + h + " to " + ow + "x" + oh + " at " + c
                    + " channels exceeds the 32-bit index range");
        }
    }

    /**
     * Validates a {@link #REL_POS_ATTN} shape for the kernel and {@code SamBackbone#attention}: the group
     * token count must fit the kernel's shared probabilities.
     *
     * @throws IllegalArgumentException when a dimension is not positive or a group holds more than
     *                                  {@link #ATTN_T_MAX} tokens
     */
    public static void checkRelPosAttn(int groups, int side, int nHead, int dh) {
        if (groups <= 0 || side <= 0 || nHead <= 0 || dh <= 0) {
            throw new IllegalArgumentException("rel-pos attention dimensions must be positive: groups=" + groups
                    + " side=" + side + " n_head=" + nHead + " d_head=" + dh);
        }
        if ((long) side * side > ATTN_T_MAX) {
            throw new IllegalArgumentException("rel-pos attention of " + side + "x" + side + " tokens exceeds the "
                    + ATTN_T_MAX + " tokens the kernel stages");
        }
    }

    /**
     * The number of workgroups of {@link #REL_POS_ATTN}: one per (group, head, query).
     *
     * @throws IllegalArgumentException when the count exceeds the 32-bit range of the GLSL twin
     */
    private static long attnGlobal(int groups, int nHead, int side) {
        long count = (long) groups * nHead * side * side;
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("rel-pos attention of " + groups + " groups of " + side + "x" + side
                    + " tokens exceeds the 32-bit workgroup range");
        }
        return count;
    }
}
