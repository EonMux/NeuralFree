package com.eonmux.neuralfree.gguf.ops.act;

import com.eonmux.neuralfree.acceleration.backend.GpuBuffer;
import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * Names, argument indices and argument lists of the generic activation kernels in
 * {@code kernels/llm/ffn_glu.cl} (GLSL {@code unary.comp}, {@code glu.comp}); signatures:
 * <pre>
 * LLM_Unary(i n, i op, f p0, f p1, f p2, f p3, i in_off, i out_off, Rn in, Wn out)
 * LLM_GLU(i n_cols, i n_rows, i op, f limit, f alpha, f beta, i gate_stride, i gate_off, i up_stride,
 *         i up_off, i out_stride, i out_off, Rn gate, Rn up, Wn out)
 * </pre>
 * Offsets and strides are in floats. {@code op} is the {@link ActOp#code()} / {@link GluOp#code()}.
 * {@code LLM_GLU}'s {@code alpha} is the SWIGLU_OAI slope or, for SITU, {@code situ_beta}; its
 * {@code beta} is the SITU {@code situ_linear_beta} ({@link #gluScalars}).
 *
 * <p>Dispatch: one work-item per output element, 1-D {@code global = roundUp(elements, local)} with
 * {@code local = {L}} ({@link #global1d}); {@code LLM_GLU} also accepts the 2-D range
 * {@code {roundUp(n_cols, L), n_rows}} with {@code local = {L, 1}}.
 */
public final class ActKernels {

    public static final String UNARY = "LLM_Unary";
    public static final String GLU = "LLM_GLU";

    public static final int ARG_UNARY_N = 0;
    public static final int ARG_UNARY_OP = 1;
    public static final int ARG_UNARY_P0 = 2;
    public static final int ARG_UNARY_P1 = 3;
    public static final int ARG_UNARY_P2 = 4;
    public static final int ARG_UNARY_P3 = 5;
    public static final int ARG_UNARY_IN_OFF = 6;
    public static final int ARG_UNARY_OUT_OFF = 7;
    public static final int ARG_UNARY_IN = 8;
    public static final int ARG_UNARY_OUT = 9;
    public static final int UNARY_ARG_COUNT = 10;

    public static final int ARG_GLU_N_COLS = 0;
    public static final int ARG_GLU_N_ROWS = 1;
    public static final int ARG_GLU_OP = 2;
    public static final int ARG_GLU_LIMIT = 3;
    public static final int ARG_GLU_ALPHA = 4;
    public static final int ARG_GLU_BETA = 5;
    public static final int ARG_GLU_GATE_STRIDE = 6;
    public static final int ARG_GLU_GATE_OFF = 7;
    public static final int ARG_GLU_UP_STRIDE = 8;
    public static final int ARG_GLU_UP_OFF = 9;
    public static final int ARG_GLU_OUT_STRIDE = 10;
    public static final int ARG_GLU_OUT_OFF = 11;
    public static final int ARG_GLU_GATE = 12;
    public static final int ARG_GLU_UP = 13;
    public static final int ARG_GLU_OUT = 14;
    public static final int GLU_ARG_COUNT = 15;

    private static final int FLOAT_BYTES = Float.BYTES;

    private ActKernels() {
    }

    /**
     * The {@code (limit, alpha, beta)} scalars of {@code LLM_GLU} for an op: SITU maps
     * {@code (limit, situ_beta = p.beta(), situ_linear_beta = p.lambda())}, every other op
     * {@code (limit, p.alpha(), p.lambda())}; ops without parameters ignore them.
     */
    public static float[] gluScalars(GluOp op, ActParams p) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        if (op == GluOp.SITU) {
            return new float[]{p.limit(), p.beta(), p.lambda()};
        }
        return new float[]{p.limit(), p.alpha(), p.lambda()};
    }

    /**
     * Arguments of {@code LLM_Unary} in signature order: {@code out[outOff + i] = act(in[inOff + i])}, {@code i < n}.
     * {@code out} may be {@code in} with {@code outOff == inOff}.
     *
     * @throws IllegalArgumentException when a view exceeds its buffer or a size is negative
     * @throws IllegalStateException    for SOFTCAP with a non-positive cap
     */
    public static Object[] unaryArgs(ActOp op, ActParams p, int n,
                                     GpuBuffer in, int inOff, GpuBuffer out, int outOff) {
        Objects.requireNonNull(op, "op");
        float[] a = Objects.requireNonNull(p, "params").unaryArgs(op);
        checkFloatView("in", in, inOff, n, 1, n);
        checkFloatView("out", out, outOff, n, 1, n);
        return new Object[]{n, op.code(), a[0], a[1], a[2], a[3], inOff, outOff, in, out};
    }

    /**
     * Arguments of {@code LLM_GLU} in signature order: for {@code r < nRows}, {@code c < nCols},
     * {@code out[outOff + r*outStride + c] = f(gate[gateOff + r*gateStride + c], up[upOff + r*upStride + c])}.
     * The fused gate+up layout passes the same buffer for gate and up with {@code upOff = gateOff + n_ff}.
     *
     * @throws IllegalArgumentException when a view exceeds its buffer, a stride is below {@code nCols} with
     *                                  several rows, or the element count reaches 2^31
     */
    public static Object[] gluArgs(GluOp op, ActParams p, int nRows, int nCols,
                                   GpuBuffer gate, int gateOff, int gateStride,
                                   GpuBuffer up, int upOff, int upStride,
                                   GpuBuffer out, int outOff, int outStride) {
        float[] s = gluScalars(op, p);
        elementCount(nRows, nCols);
        checkFloatView("gate", gate, gateOff, gateStride, nRows, nCols);
        checkFloatView("up", up, upOff, upStride, nRows, nCols);
        checkFloatView("out", out, outOff, outStride, nRows, nCols);
        return new Object[]{nCols, nRows, op.code(), s[0], s[1], s[2],
                gateStride, gateOff, upStride, upOff, outStride, outOff, gate, up, out};
    }

    /**
     * 1-D global size for {@code elements} work-items: {@code roundUp(elements, local)}, at least one
     * work-group.
     *
     * @throws IllegalArgumentException when {@code local} is not positive or {@code elements} is negative
     */
    public static long[] global1d(long elements, int local) {
        if (local <= 0) {
            throw new IllegalArgumentException("local size must be positive, got " + local);
        }
        if (elements < 0) {
            throw new IllegalArgumentException("element count must be non-negative, got " + elements);
        }
        long groups = Math.max(1L, (elements + local - 1) / local);
        return new long[]{groups * local};
    }

    /**
     * {@code rows * cols} as an int element count.
     *
     * @throws IllegalArgumentException when a factor is negative or the product reaches 2^31
     */
    public static int elementCount(long rows, long cols) {
        if (rows < 0 || cols < 0) {
            throw new IllegalArgumentException("shape must be non-negative, got " + rows + " x " + cols);
        }
        long n = rows * cols;
        if (rows != 0 && n / rows != cols || n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("element count " + rows + " x " + cols
                    + " exceeds the kernels' 2^31 - 1 limit");
        }
        return (int) n;
    }

    /**
     * Validates a float view {@code [rows][cols]} at element offset {@code off} with row stride {@code stride}
     * against the buffer size: non-null buffer, non-negative sizes and offset, {@code stride >= cols} when
     * {@code rows > 1}, and the last element inside the buffer. Shared by the element-wise kernel wrappers
     * of every op family.
     *
     * @throws IllegalArgumentException naming the operand when the view is invalid
     */
    public static void checkFloatView(String name, GpuBuffer buf, long off, long stride, long rows, long cols) {
        Objects.requireNonNull(buf, name);
        if (off < 0 || rows < 0 || cols < 0) {
            throw new IllegalArgumentException(name + ": offset and shape must be non-negative, got offset " + off
                    + ", shape " + rows + " x " + cols);
        }
        if (rows > 1 && stride < cols) {
            throw new IllegalArgumentException(name + ": stride " + stride + " is below the row width " + cols);
        }
        if (rows == 0 || cols == 0) {
            return;
        }
        long endElements = off + (rows - 1) * stride + cols;
        if (endElements > buf.sizeInBytes() / FLOAT_BYTES) {
            throw new IllegalArgumentException(name + ": view ends at float " + endElements + " but the buffer holds "
                    + (buf.sizeInBytes() / FLOAT_BYTES) + " floats");
        }
    }
}
