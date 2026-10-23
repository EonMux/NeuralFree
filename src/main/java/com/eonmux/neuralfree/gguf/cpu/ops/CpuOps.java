package com.eonmux.neuralfree.gguf.cpu.ops;

import com.eonmux.neuralfree.gguf.arch.ActOp;
import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.ops.act.ActMath;
import com.eonmux.neuralfree.gguf.ops.act.GluMath;
import com.eonmux.neuralfree.gguf.ops.norm.NormKernels.MixMode;
import com.eonmux.neuralfree.gguf.ops.norm.NormMath;
import com.eonmux.neuralfree.gguf.spec.ActParams;

import java.util.Objects;

/**
 * Row-wise and element-wise CPU operations of the forward pass over row-major {@code [row][width]} float arrays,
 * threaded through {@link CpuParallel}.
 *
 * <p>The math has one home: normalizations delegate to {@link NormMath}, activations to {@link ActMath} and gated
 * activations to {@link GluMath}; this class adds the row sharding, softmax, top-k and the plain vector
 * operations. Stateless and thread-safe; nothing is allocated except where a method says so.</p>
 */
public final class CpuOps {

    /** Elements per parallel task of the element-wise operations. */
    private static final int ELEMENT_GRAIN = 1 << 15;

    private CpuOps() {
    }

    // ============================================================== normalization

    /**
     * Normalizes {@code nRows} rows of {@code width} floats: row {@code r} is read from {@code x[xOff + r*width ..]}
     * and written to {@code y[yOff + r*width ..]} as {@code norm(row) * w + b} ({@link NormMath#rowNorm}), the
     * weight and bias applied when non-null. {@code y} may be {@code x} at the same offset.
     *
     * @throws IllegalArgumentException for a non-row-norm kind, an invalid eps, or a view outside its array
     */
    public static void norm(NormKind kind, float eps, float[] x, int xOff, float[] y, int yOff, int nRows, int width,
                            float[] w, float[] b) {
        Objects.requireNonNull(kind, "kind");
        checkNonNegative(nRows, "row count");
        CpuParallel.parallelFor(nRows, rowGrain(width), (from, to) -> {
            for (int r = from; r < to; r++) {
                NormMath.rowNorm(kind, eps, x, xOff + r * width, y, yOff + r * width, width, w, 0, b, 0);
            }
        });
    }

    /** RMS norm of {@code nRows} rows ({@link #norm} with {@link NormKind#RMS} and no bias). */
    public static void rmsNorm(float[] x, int xOff, float[] y, int yOff, int nRows, int width, float[] w, float eps) {
        norm(NormKind.RMS, eps, x, xOff, y, yOff, nRows, width, w, null);
    }

    /** Layer norm of {@code nRows} rows ({@link #norm} with {@link NormKind#LAYER}). */
    public static void layerNorm(float[] x, int xOff, float[] y, int yOff, int nRows, int width, float[] w, float[] b,
                                 float eps) {
        norm(NormKind.LAYER, eps, x, xOff, y, yOff, nRows, width, w, b);
    }

    /** {@link NormMath#residualMix}: {@code out = mix(inp, cur)} over {@code n} floats. */
    public static void residualMix(MixMode mode, int n, float s, float[] inp, int inpOff, float[] cur, int curOff,
                                   float[] alpha, float[] beta, int abLen, float[] out, int outOff) {
        NormMath.residualMix(mode, n, s, inp, inpOff, cur, curOff, alpha, beta, abLen, out, outOff);
    }

    // ============================================================== softmax

    /**
     * In-place softmax of {@code x[off .. off + n)} as ggml's CPU {@code soft_max} computes it (llama.cpp
     * ggml/src/ggml-cpu/ops.cpp:5980-5999, vec.cpp:625-698): {@code max} over the row, {@code e = expf(x - max)}
     * summed in double, then every value scaled by {@code (float) (1 / sum)}.
     *
     * @throws IllegalArgumentException when the range is outside the array or {@code n < 1}
     */
    public static void softmax(float[] x, int off, int n) {
        checkRange("softmax row", x, off, n);
        if (n < 1) {
            throw new IllegalArgumentException("softmax needs at least one value");
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, x[off + i]);
        }
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            float e = (float) Math.exp(x[off + i] - max);
            x[off + i] = e;
            sum += e;
        }
        float inv = (float) (1.0 / sum);
        for (int i = 0; i < n; i++) {
            x[off + i] *= inv;
        }
    }

    // ============================================================== activations

    /** {@code out[outOff + i] = act(in[inOff + i])} for {@code i < n} ({@link ActMath#apply}). */
    public static void unary(ActOp op, ActParams p, float[] in, int inOff, float[] out, int outOff, int n) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        checkNonNegative(n, "element count");
        checkRange("activation input", in, inOff, n);
        checkRange("activation output", out, outOff, n);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (from, to) ->
                ActMath.apply(op, p, in, inOff + from, out, outOff + from, to - from));
    }

    /**
     * {@code out[outOff + i] = glu(gate[gateOff + i], up[upOff + i])} for {@code i < n} ({@link GluMath#apply}).
     * {@code out} may be {@code gate} or {@code up} at the same offset.
     */
    public static void glu(GluOp op, ActParams p, float[] gate, int gateOff, float[] up, int upOff,
                           float[] out, int outOff, int n) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        checkNonNegative(n, "element count");
        checkRange("gate", gate, gateOff, n);
        checkRange("up", up, upOff, n);
        checkRange("glu output", out, outOff, n);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (from, to) ->
                GluMath.apply(op, p, gate, gateOff + from, up, upOff + from, out, outOff + from, to - from));
    }

    /**
     * Strided GLU over {@code nRows} rows of {@code nCols} ({@link GluMath#applyRows}), sharded on rows; the fused
     * gate+up layout passes the same array with {@code upOff = gateOff + nCols}.
     */
    public static void gluRows(GluOp op, ActParams p, int nRows, int nCols,
                               float[] gate, int gateOff, int gateStride,
                               float[] up, int upOff, int upStride,
                               float[] out, int outOff, int outStride) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(p, "params");
        checkNonNegative(nRows, "row count");
        CpuParallel.parallelFor(nRows, rowGrain(nCols), (from, to) ->
                GluMath.applyRows(op, p, to - from, nCols,
                        gate, gateOff + from * gateStride, gateStride,
                        up, upOff + from * upStride, upStride,
                        out, outOff + from * outStride, outStride));
    }

    // ============================================================== vector operations

    /** {@code out[outOff + i] = a[aOff + i] + b[bOff + i]} for {@code i < n}. */
    public static void add(float[] a, int aOff, float[] b, int bOff, float[] out, int outOff, int n) {
        checkNonNegative(n, "element count");
        checkRange("a", a, aOff, n);
        checkRange("b", b, bOff, n);
        checkRange("sum", out, outOff, n);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                out[outOff + i] = a[aOff + i] + b[bOff + i];
            }
        });
    }

    /** {@code y[yOff + i] += s * x[xOff + i]} for {@code i < n}. */
    public static void axpy(float s, float[] x, int xOff, float[] y, int yOff, int n) {
        checkNonNegative(n, "element count");
        checkRange("x", x, xOff, n);
        checkRange("y", y, yOff, n);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                y[yOff + i] += s * x[xOff + i];
            }
        });
    }

    /** {@code x[off + i] *= s} for {@code i < n}. */
    public static void scale(float[] x, int off, int n, float s) {
        checkNonNegative(n, "element count");
        checkRange("x", x, off, n);
        CpuParallel.parallelFor(n, ELEMENT_GRAIN, (from, to) -> {
            for (int i = from; i < to; i++) {
                x[off + i] *= s;
            }
        });
    }

    /**
     * Adds a bias row to each of {@code nRows} rows: {@code x[xOff + r*width + c] += bias[biasOff + c]}.
     *
     * @throws IllegalArgumentException when a view is outside its array
     */
    public static void addRowBias(float[] x, int xOff, int nRows, int width, float[] bias, int biasOff) {
        checkNonNegative(nRows, "row count");
        checkRange("bias", bias, biasOff, width);
        checkRange("rows", x, xOff, (long) nRows * width);
        CpuParallel.parallelFor(nRows, rowGrain(width), (from, to) -> {
            for (int r = from; r < to; r++) {
                int base = xOff + r * width;
                for (int c = 0; c < width; c++) {
                    x[base + c] += bias[biasOff + c];
                }
            }
        });
    }

    /** {@code sum(a[aOff + i] * b[bOff + i])} over {@code i < n}, accumulated in double. */
    public static float dot(float[] a, int aOff, float[] b, int bOff, int n) {
        checkNonNegative(n, "element count");
        checkRange("a", a, aOff, n);
        checkRange("b", b, bOff, n);
        double s = 0.0;
        for (int i = 0; i < n; i++) {
            s += (double) a[aOff + i] * b[bOff + i];
        }
        return (float) s;
    }

    /**
     * Index (relative to {@code off}) of the largest of {@code x[off .. off + n)}; the first one on ties, as
     * {@code ggml_vec_argmax_f32}.
     *
     * @throws IllegalArgumentException when {@code n < 1} or the range is outside the array
     */
    public static int argmax(float[] x, int off, int n) {
        checkRange("argmax input", x, off, n);
        if (n < 1) {
            throw new IllegalArgumentException("argmax needs at least one value");
        }
        int best = 0;
        float max = x[off];
        for (int i = 1; i < n; i++) {
            if (x[off + i] > max) {
                max = x[off + i];
                best = i;
            }
        }
        return best;
    }

    /**
     * Writes the indices (relative to {@code off}) of the {@code k} largest of {@code x[off .. off + n)} to
     * {@code idx[idxOff ..]}, in descending order of value with lower indices first among equal values.
     *
     * @throws IllegalArgumentException when {@code k} is outside {@code [1, n]} or a range is out of bounds
     */
    public static void topK(float[] x, int off, int n, int k, int[] idx, int idxOff) {
        checkRange("top-k input", x, off, n);
        if (k < 1 || k > n) {
            throw new IllegalArgumentException("top-k of " + k + " needs 1 <= k <= " + n);
        }
        if (idxOff < 0 || idxOff > idx.length - k) {
            throw new IllegalArgumentException("top-k output [" + idxOff + ", +" + k + ") exceeds length "
                    + idx.length);
        }
        int filled = 0;
        for (int i = 0; i < n; i++) {
            float v = x[off + i];
            if (filled == k && !(v > x[off + idx[idxOff + k - 1]])) {
                continue;
            }
            int pos = filled < k ? filled++ : k - 1;
            while (pos > 0 && v > x[off + idx[idxOff + pos - 1]]) {
                idx[idxOff + pos] = idx[idxOff + pos - 1];
                pos--;
            }
            idx[idxOff + pos] = i;
        }
    }

    // ============================================================== validation

    /** Rows per parallel task so that one task covers about {@link #ELEMENT_GRAIN} elements. */
    private static int rowGrain(int width) {
        return Math.max(1, ELEMENT_GRAIN / Math.max(1, width));
    }

    private static void checkNonNegative(int v, String what) {
        if (v < 0) {
            throw new IllegalArgumentException(what + " must be non-negative, got " + v);
        }
    }

    private static void checkRange(String what, float[] a, int off, long n) {
        Objects.requireNonNull(a, what);
        if (off < 0 || n < 0 || off > a.length - n) {
            throw new IllegalArgumentException(what + " range [" + off + ", " + (off + n) + ") exceeds length "
                    + a.length);
        }
    }
}
