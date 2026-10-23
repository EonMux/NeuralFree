package com.eonmux.neuralfree.gguf.ops.flow;

import com.eonmux.neuralfree.gguf.kernels.LlmKernelConstants;
import com.eonmux.neuralfree.gguf.ops.heads.CpuPooling;

import java.util.Collections;
import java.util.List;

/**
 * Names, argument layout and launch geometry of the flow-family kernels ({@code kernels/llm/flow.cl} and the GLSL
 * twin {@code gather_rows.comp}). Offsets and strides are in floats; every scalar is a 32-bit {@code int}; buffers
 * follow the scalars in the order listed.
 *
 * <ul>
 *   <li>{@link #GATHER_ROWS} {@code (n_rows*, row_len, src_stride, dst_stride, src_off, dst_off; rows, src, dst)}:
 *       {@code dst[r] = rows[r] >= 0 ? src[rows[r]] : 0} row by row (ggml_get_rows on F32): output-row selection of
 *       the logits head and drafter row selection. One work-item per element: {@link #gatherGlobal}.
 *       CPU twin {@link CpuPooling#gatherRows}.</li>
 * </ul>
 * {@code *} marks the argument re-set per dispatch, at the matching {@code ARG_*} index.
 */
public final class FlowKernels {

    public static final String GATHER_ROWS = "LLM_GatherRows";

    /** Every kernel of the family, for prewarming. */
    public static final List<String> ALL = Collections.singletonList(GATHER_ROWS);

    /** {@link #GATHER_ROWS} {@code n_rows}. */
    public static final int ARG_GATHER_N_ROWS = 0;
    /** {@link #GATHER_ROWS} argument count (6 scalars, 3 buffers). */
    public static final int GATHER_ARG_COUNT = 9;

    /** Local size of {@link #GATHER_ROWS}. */
    public static final int GATHER_LOCAL = LlmKernelConstants.LLM_DIMX;

    private FlowKernels() {
    }

    /**
     * Global size of {@link #GATHER_ROWS}: one work-item per gathered element, rounded up to {@link #GATHER_LOCAL}.
     *
     * @throws IllegalArgumentException when a dimension is not positive or the element count exceeds the 32-bit
     *                                  index range of the GLSL twin
     */
    public static long gatherGlobal(int nRows, int rowLen) {
        if (nRows <= 0 || rowLen <= 0) {
            throw new IllegalArgumentException(GATHER_ROWS + " needs n_rows > 0 and row_len > 0: n_rows=" + nRows
                    + " row_len=" + rowLen);
        }
        long n = (long) nRows * rowLen;
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(GATHER_ROWS + " of " + nRows + " rows x " + rowLen
                    + " floats exceeds the 32-bit index range");
        }
        return (n + GATHER_LOCAL - 1) / GATHER_LOCAL * GATHER_LOCAL;
    }
}
