package com.eonmux.neuralfree.gguf.ops.ssm;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout and launch geometry of the SSM-family kernels ({@code kernels/llm/recurrent/ssm_conv.cl},
 * {@code kernels/llm/recurrent/ssm_scan.cl} and their GLSL twins). Offsets and strides are in floats; every
 * scalar is a 32-bit {@code int}; buffers follow the scalars in the order listed.
 *
 * <ul>
 *   <li>{@link #SSM_CONV} {@code (n_ch, d_conv, n_tokens*, flags, x_stride, x_off, out_stride, out_off, state_off,
 *       state_stride; x, w, b, conv_state, state_rows, out)}: causal depthwise conv over
 *       {@code concat(state, x)} with optional bias ({@link #CONV_FLAG_BIAS}) and SiLU ({@link #CONV_FLAG_SILU});
 *       the state row {@code state_rows[0]} ({@code state_stride} floats) is updated in place, its slice starting
 *       {@code state_off} channels in. One work-item per channel: global = roundUp(n_ch, L).
 *       CPU twin {@link CpuSsmConv#forward}.</li>
 *   <li>{@link #SHORTCONV_CENTERED} {@code (n_ch, d_conv, n_tokens*, x_stride, x_off, out_stride; x, w, out)}:
 *       LFM2 non-causal centered conv, zero-padded on both sides, no state. One work-item per (token, channel):
 *       global = roundUp(n_tokens * n_ch, L). CPU twin {@link CpuShortConv#centered}.</li>
 *   <li>{@link #SSM_SCAN} {@code (d_state, head_dim, n_head, n_group, n_tokens*, a_per_state, ch_per_wg, x_stride,
 *       x_off, dt_stride, dt_off, b_stride, c_stride, b_off, c_off, y_stride, has_dt_bias; x, dt, dt_bias, A, B, C,
 *       ssm_state, state_rows, y)}: Mamba1/Mamba2 selective scan with the dt bias and softplus folded in. Launch
 *       geometry from {@link ScanGeometry}. CPU twin {@link CpuSsmScan#scan}.</li>
 *   <li>{@link #MAMBA_GATE_OUT} {@code (n*, width, head_dim, x_stride, x_off, y_off, z_stride, z_off;
 *       y, x, D, z, out)}: {@code out = (y + D[c / head_dim] * x) * silu(z)} over {@code n = n_tokens * width}
 *       elements, out contiguous. global = roundUp(n, L). CPU twin {@link CpuSsmScan#gateOut}.</li>
 * </ul>
 * {@code *} marks the per-step argument whose index is the matching {@code ARG_*} constant. Optional buffers
 * ({@code b} without {@link #CONV_FLAG_BIAS}, {@code dt_bias} with {@code has_dt_bias = 0}) are bound to a
 * read-only placeholder. {@code state_rows} is the single-sequence row buffer of the step inputs.
 */
public final class SsmKernels {

    public static final String SSM_CONV = "LLM_SSM_Conv";
    public static final String SHORTCONV_CENTERED = "LLM_ShortConvCentered";
    public static final String SSM_SCAN = "LLM_SSM_Scan";
    public static final String MAMBA_GATE_OUT = "LLM_MambaGateOut";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.unmodifiableList(
            Arrays.asList(SSM_CONV, SHORTCONV_CENTERED, SSM_SCAN, MAMBA_GATE_OUT));

    /** {@link #SSM_CONV} flag: add {@code b[c]} after the convolution. */
    public static final int CONV_FLAG_BIAS = 1;
    /** {@link #SSM_CONV} flag: apply SiLU after the bias. */
    public static final int CONV_FLAG_SILU = 2;

    /** {@link #SSM_CONV} {@code n_tokens}. */
    public static final int ARG_CONV_N_TOKENS = 2;
    /** {@link #SSM_CONV} argument count (10 scalars, 6 buffers). */
    public static final int CONV_ARG_COUNT = 16;

    /** {@link #SHORTCONV_CENTERED} {@code n_tokens}. */
    public static final int ARG_CENTERED_N_TOKENS = 2;
    /** {@link #SHORTCONV_CENTERED} argument count (6 scalars, 3 buffers). */
    public static final int CENTERED_ARG_COUNT = 9;

    /** {@link #SSM_SCAN} {@code n_tokens}. */
    public static final int ARG_SCAN_N_TOKENS = 4;
    /** {@link #SSM_SCAN} argument count (17 scalars, 9 buffers). */
    public static final int SCAN_ARG_COUNT = 26;

    /** {@link #MAMBA_GATE_OUT} {@code n = n_tokens * width}. */
    public static final int ARG_GATE_N = 0;
    /** {@link #MAMBA_GATE_OUT} argument count (8 scalars, 5 buffers). */
    public static final int GATE_ARG_COUNT = 13;

    /** States one scan lane keeps in registers; {@code NF_SSM_SCAN_MAX_SPL}. */
    public static final int SCAN_MAX_STATES_PER_LANE = 8;
    /** Largest number of lanes per channel the host launches. */
    public static final int SCAN_MAX_LANES = 64;
    /** Smallest scan workgroup: one full CUDA warp, the subgroup-reduction minimum. */
    public static final int SCAN_MIN_LOCAL = 32;
    /** Local size of the barrier-free kernels ({@link #SSM_CONV}, {@link #SHORTCONV_CENTERED}, {@link #MAMBA_GATE_OUT}). */
    public static final int ELEMENTWISE_LOCAL = LlmKernelConstants.LLM_DIMX;

    private SsmKernels() {
    }

    /**
     * Launch geometry of {@link #SSM_SCAN} for one state shape: {@code lanes = min(pow2ceil(d_state), 64)}
     * work-items per channel, {@code ch_per_wg = max(1, 32 / lanes)} channels per workgroup, local size
     * {@code lanes * ch_per_wg} (a power of two in [32, 64]) and one workgroup per {@code ch_per_wg} channels.
     * Immutable.
     */
    public static final class ScanGeometry {
        private final int dState;
        private final int headDim;
        private final int nHead;
        private final int nGroup;
        private final int lanes;
        private final int channelsPerGroup;

        private ScanGeometry(int dState, int headDim, int nHead, int nGroup) {
            this.dState = dState;
            this.headDim = headDim;
            this.nHead = nHead;
            this.nGroup = nGroup;
            this.lanes = Math.min(ceilPow2(dState), SCAN_MAX_LANES);
            this.channelsPerGroup = Math.max(1, SCAN_MIN_LOCAL / lanes);
        }

        /**
         * @param nGroup B/C groups; 1 for Mamba1 and PLaMo2
         * @throws IllegalArgumentException when a dimension is not positive, {@code nGroup} does not divide
         *                                  {@code nHead}, or {@code dState > SCAN_MAX_LANES * SCAN_MAX_STATES_PER_LANE}
         */
        public static ScanGeometry of(int dState, int headDim, int nHead, int nGroup) {
            checkScanShape(dState, headDim, nHead, nGroup);
            return new ScanGeometry(dState, headDim, nHead, nGroup);
        }

        public int dState() { return dState; }
        public int headDim() { return headDim; }
        public int nHead() { return nHead; }
        public int nGroup() { return nGroup; }
        /** Channels {@code n_head * head_dim} (= {@code d_inner}). */
        public int channels() { return nHead * headDim; }
        /** Work-items per channel. */
        public int lanes() { return lanes; }
        /** The kernel's {@code ch_per_wg} argument. */
        public int channelsPerGroup() { return channelsPerGroup; }
        /** States per lane, at most {@link #SCAN_MAX_STATES_PER_LANE}. */
        public int statesPerLane() { return (dState + lanes - 1) / lanes; }
        /** Local work size {@code lanes * ch_per_wg}. */
        public int localSize() { return lanes * channelsPerGroup; }
        /** Global work size {@code ceil(channels / ch_per_wg) * localSize}. */
        public long globalSize() {
            return (long) ((channels() + channelsPerGroup - 1) / channelsPerGroup) * localSize();
        }
        /** Floats of one state row, {@code n_head * head_dim * d_state}. */
        public long stateFloats() { return (long) channels() * dState; }

        @Override
        public String toString() {
            return "ScanGeometry{dState=" + dState + ", headDim=" + headDim + ", nHead=" + nHead + ", nGroup=" + nGroup
                    + ", lanes=" + lanes + ", chPerWg=" + channelsPerGroup + '}';
        }
    }

    /**
     * Validates a scan shape for both the kernel and {@link CpuSsmScan}.
     *
     * @throws IllegalArgumentException with the offending values
     */
    public static void checkScanShape(int dState, int headDim, int nHead, int nGroup) {
        if (dState <= 0 || headDim <= 0 || nHead <= 0 || nGroup <= 0) {
            throw new IllegalArgumentException("SSM scan dimensions must be positive: d_state=" + dState
                    + " head_dim=" + headDim + " n_head=" + nHead + " n_group=" + nGroup);
        }
        if (nHead % nGroup != 0) {
            throw new IllegalArgumentException("SSM scan n_group=" + nGroup + " does not divide n_head=" + nHead);
        }
        if (dState > SCAN_MAX_LANES * SCAN_MAX_STATES_PER_LANE) {
            throw new IllegalArgumentException("SSM scan d_state=" + dState + " exceeds "
                    + SCAN_MAX_LANES * SCAN_MAX_STATES_PER_LANE);
        }
        if ((long) nHead * headDim * dState > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("SSM scan state of " + nHead + "x" + headDim + "x" + dState
                    + " floats exceeds the 32-bit index range");
        }
    }

    /**
     * Validates a conv shape for {@link #SSM_CONV} and {@link CpuSsmConv}.
     *
     * @throws IllegalArgumentException when {@code nCh <= 0} or {@code dConv < 2}
     */
    public static void checkConvShape(int nCh, int dConv) {
        if (nCh <= 0 || dConv < 2) {
            throw new IllegalArgumentException("SSM conv needs n_ch > 0 and d_conv >= 2: n_ch=" + nCh + " d_conv=" + dConv);
        }
    }

    /**
     * Validates a centered-conv shape for {@link #SHORTCONV_CENTERED} and {@link CpuShortConv#centered}.
     *
     * @throws IllegalArgumentException when {@code nCh <= 0} or {@code dConv} is not a positive odd number
     */
    public static void checkCenteredShape(int nCh, int dConv) {
        if (nCh <= 0 || dConv <= 0 || (dConv & 1) == 0) {
            throw new IllegalArgumentException("centered short conv needs n_ch > 0 and an odd d_conv: n_ch=" + nCh
                    + " d_conv=" + dConv);
        }
    }

    /** Global size of a one-work-item-per-element kernel, rounded up to a multiple of {@code local}. */
    public static long elementwiseGlobal(long elements, int local) {
        if (elements < 0 || local <= 0) {
            throw new IllegalArgumentException("elements=" + elements + " local=" + local);
        }
        return (elements + local - 1) / local * local;
    }

    static int ceilPow2(int v) {
        int p = 1;
        while (p < v) {
            p <<= 1;
        }
        return p;
    }
}
