package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.cpu.ops.CpuParallel;
import com.eonmux.neuralfree.gguf.cpu.quant.CpuVecDot.QuantizedActivations;
import com.eonmux.neuralfree.gguf.quant.CpuHadamard;
import com.eonmux.neuralfree.gguf.quant.HadamardSpec;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Matrix-vector and matrix-matrix products of in-place weights ({@link CpuTensorRef}) with F32 activations, the CPU
 * twin of {@code ggml_mul_mat} as {@code build_lora_mm} / {@code build_lora_mm_id} use it (llama.cpp
 * src/llama-graph.cpp: 1628-1647, 1706-1733).
 *
 * <p>Per call, each token's activation row is
 * <ol>
 *   <li>transformed by the weight's PRISM input Hadamard when it has one ({@code build_hadamard_input},
 *       llama-graph.cpp:1598-1626, through {@link CpuHadamard});</li>
 *   <li>quantized once into the weight type's {@code vec_dot_type} ({@link CpuActivationQuantizer});</li>
 *   <li>dotted with every requested weight row by the type's {@link CpuVecDot};</li>
 * </ol>
 * and the outputs are multiplied by the weight's output scale {@code w_s} when it has one (per tensor, or per expert
 * for 3-D expert tensors; llama-graph.cpp:1643-1645, 1722-1729). Work is sharded on weight rows over
 * {@link CpuParallel}, so every row is read once per call whatever the token count.</p>
 *
 * <p>Transforms are looked up per weight through the {@link TransformSource} the instance was built with
 * ({@code CpuWeights} binds the PRISM and {@code .scale} tensors of the model); {@link #PLAIN} applies none.
 * Instances are immutable and thread-safe. Each call allocates its activation blocks (and the rotated input when a
 * Hadamard applies); weights are never copied.</p>
 *
 * <p>Layouts are row-major: token {@code t}'s input row starts at {@code x[xOff + t*xStride]} and holds
 * {@code w.cols()} values; its output row starts at {@code y[yOff + t*yStride]}.</p>
 */
public final class CpuMatVec {

    /** Multiply-accumulates per parallel task below which rows are not split further. */
    private static final long MIN_TASK_WORK = 1L << 15;

    /** Vec-dot kernel of every weight type the CPU supports. */
    private static final Map<GGMLType, CpuVecDot> VEC_DOTS = buildVecDots();

    /** Instance that applies no input transform and no output scale. */
    public static final CpuMatVec PLAIN = new CpuMatVec(w -> Transforms.NONE);

    private final TransformSource transforms;

    /** A matvec that resolves each weight's transforms through {@code transforms}. */
    public CpuMatVec(TransformSource transforms) {
        this.transforms = Objects.requireNonNull(transforms, "transforms");
    }

    // ------------------------------------------------------------------ transforms

    /** Resolves the transforms bound to a weight. */
    @FunctionalInterface
    public interface TransformSource {
        /** The transforms of {@code w}; {@link Transforms#NONE} for a weight without any. Never null. */
        Transforms of(CpuTensorRef w);
    }

    /**
     * The per-weight transforms of one matmul weight: the PRISM input Hadamard, the output scale {@code w_s}, and the
     * expert count of a 3-D expert tensor. Immutable.
     */
    public static final class Transforms {

        /** No input transform, no output scale, not an expert tensor. */
        public static final Transforms NONE = new Transforms(null, null, 0);

        private final HadamardSpec input;
        private final float[] scale;
        private final int experts;

        private Transforms(HadamardSpec input, float[] scale, int experts) {
            this.input = input;
            this.scale = scale;
            this.experts = experts;
        }

        /**
         * Transforms of one weight.
         *
         * @param input   PRISM input Hadamard, or null
         * @param scale   output scale {@code w_s}: null, one value for the whole tensor, or one value per expert
         * @param experts number of experts of a 3-D expert tensor ({@code ne[2]}), or 0 for a 2-D weight
         * @throws IllegalArgumentException for a negative expert count, a scale of another length, or a non-finite
         *                                  scale value
         */
        public static Transforms of(HadamardSpec input, float[] scale, int experts) {
            if (experts < 0) {
                throw new IllegalArgumentException("expert count must be non-negative, got " + experts);
            }
            float[] s = null;
            if (scale != null) {
                if (scale.length != 1 && (experts == 0 || scale.length != experts)) {
                    throw new IllegalArgumentException("output scale has " + scale.length
                            + " values; expected 1" + (experts > 0 ? " or " + experts + " (one per expert)" : ""));
                }
                for (float v : scale) {
                    if (!Float.isFinite(v)) {
                        throw new IllegalArgumentException("output scale value " + v + " is not finite");
                    }
                }
                s = scale.clone();
            }
            if (input == null && s == null && experts == 0) {
                return NONE;
            }
            return new Transforms(input, s, experts);
        }

        /** PRISM input Hadamard, or null. */
        public HadamardSpec input() {
            return input;
        }

        /** True when outputs are multiplied by a scale. */
        public boolean hasScale() {
            return scale != null;
        }

        /** Number of experts of a 3-D expert tensor, 0 for a 2-D weight. */
        public int experts() {
            return experts;
        }

        /** Output scale of the rows of {@code expert} (1 without a scale; the tensor scale when it has one value). */
        public float scale(int expert) {
            if (scale == null) {
                return 1.0f;
            }
            return scale.length == 1 ? scale[0] : scale[expert];
        }

        @Override
        public String toString() {
            return "Transforms{input=" + input + ", scale=" + (scale == null ? "none" : scale.length + " value(s)")
                    + ", experts=" + experts + "}";
        }
    }

    // ------------------------------------------------------------------ kernels

    private static Map<GGMLType, CpuVecDot> buildVecDots() {
        EnumMap<GGMLType, CpuVecDot> m = new EnumMap<>(GGMLType.class);
        for (GGMLType t : LegacyVecDot.supportedTypes()) {
            put(m, t, LegacyVecDot.forType(t));
        }
        for (GGMLType t : Q8VecDot.supportedTypes()) {
            put(m, t, Q8VecDot.forType(t));
        }
        for (GGMLType t : Fp4VecDot.supportedTypes()) {
            put(m, t, Fp4VecDot.forType(t));
        }
        for (GGMLType t : KQuantVecDot.supportedTypes()) {
            put(m, t, KQuantVecDot.forType(t));
        }
        for (GGMLType t : IQuantVecDot.supportedTypes()) {
            put(m, t, IQuantVecDot.forType(t));
        }
        for (GGMLType t : TernaryVecDot.supportedTypes()) {
            put(m, t, TernaryVecDot.forType(t));
        }
        for (GGMLType t : NewFpVecDot.supportedTypes()) {
            put(m, t, NewFpVecDot.forType(t));
        }
        for (GGMLType t : Fp16VecDot.supportedTypes()) {
            put(m, t, Fp16VecDot.forType(t));
        }
        return Collections.unmodifiableMap(m);
    }

    private static void put(Map<GGMLType, CpuVecDot> m, GGMLType t, CpuVecDot dot) {
        if (m.put(t, Objects.requireNonNull(dot, "vec-dot of " + t)) != null) {
            throw new IllegalStateException("two CPU vec-dot kernels claim ggml type " + t);
        }
    }

    /** True when weights of {@code type} can be multiplied on the CPU. */
    public static boolean supports(GGMLType type) {
        return type != null && VEC_DOTS.containsKey(type);
    }

    /** Every weight type the CPU matvec supports (unmodifiable). */
    public static Set<GGMLType> supportedTypes() {
        return VEC_DOTS.keySet();
    }

    /**
     * The vec-dot kernel of a weight type.
     *
     * @throws UnsupportedOperationException when the CPU has no kernel for the type
     */
    public static CpuVecDot vecDot(GGMLType type) {
        CpuVecDot dot = type == null ? null : VEC_DOTS.get(type);
        if (dot == null) {
            throw new UnsupportedOperationException("no CPU matmul kernel for ggml type " + type
                    + "; supported: " + VEC_DOTS.keySet());
        }
        return dot;
    }

    /** The transforms this instance applies to {@code w}. */
    public Transforms transformsOf(CpuTensorRef w) {
        Transforms t = transforms.of(Objects.requireNonNull(w, "weight"));
        if (t == null) {
            throw new IllegalStateException("transform source returned null for " + w);
        }
        return t;
    }

    // ------------------------------------------------------------------ public products

    /** {@code y[yOff + r] = sum_c w[r][c] * x[xOff + c]} for every row {@code r} of {@code w}. */
    public void matvec(CpuTensorRef w, float[] x, int xOff, float[] y, int yOff) {
        run(w, 0L, rowCount(w), x, xOff, 0, y, yOff, 0, 1, false);
    }

    /** {@code y[yOff + r] += sum_c w[r][c] * x[xOff + c]} for every row {@code r} of {@code w}. */
    public void matvecAdd(CpuTensorRef w, float[] x, int xOff, float[] y, int yOff) {
        run(w, 0L, rowCount(w), x, xOff, 0, y, yOff, 0, 1, true);
    }

    /**
     * {@code y[yOff + i] = sum_c w[rowStart + i][c] * x[xOff + c]} for {@code i < rowCount}: a block of rows, such as
     * one head of a per-head projection.
     */
    public void matvecRows(CpuTensorRef w, long rowStart, int rowCount, float[] x, int xOff, float[] y, int yOff) {
        run(w, rowStart, rowCount, x, xOff, 0, y, yOff, 0, 1, false);
    }

    /**
     * Product with expert {@code expert} of a 3-D expert tensor whose expert count is known to this instance's
     * {@link TransformSource} (rows {@code [expert*R, (expert+1)*R)} with {@code R = rows / experts}).
     *
     * @throws IllegalArgumentException when the weight is not a known expert tensor or the expert is out of range
     */
    public void matvecExpert(CpuTensorRef w3d, int expert, float[] x, int xOff, float[] y, int yOff) {
        int experts = transformsOf(w3d).experts();
        if (experts == 0) {
            throw new IllegalArgumentException("weight " + w3d + " is not a known expert tensor; pass the expert count");
        }
        matvecExpert(w3d, expert, experts, x, xOff, y, yOff);
    }

    /**
     * Product with expert {@code expert} of a 3-D expert tensor holding {@code nExpert} experts stacked along its
     * rows: {@code y[yOff + i] = sum_c w[expert*R + i][c] * x[xOff + c]} for {@code i < R = rows / nExpert}.
     *
     * @throws IllegalArgumentException when the rows do not split into {@code nExpert} experts, the expert is out of
     *                                  range, or {@code nExpert} contradicts the weight's known expert count
     */
    public void matvecExpert(CpuTensorRef w3d, int expert, int nExpert, float[] x, int xOff, float[] y, int yOff) {
        Objects.requireNonNull(w3d, "weight");
        int known = transformsOf(w3d).experts();
        if (nExpert < 1 || w3d.rows() % nExpert != 0) {
            throw new IllegalArgumentException("expert count " + nExpert + " does not divide the " + w3d.rows()
                    + " rows of " + w3d);
        }
        if (known != 0 && known != nExpert) {
            throw new IllegalArgumentException("expert count " + nExpert + " contradicts the " + known
                    + " experts of " + w3d);
        }
        if (expert < 0 || expert >= nExpert) {
            throw new IllegalArgumentException("expert " + expert + " is outside [0, " + nExpert + ")");
        }
        long perExpert = w3d.rows() / nExpert;
        run(w3d, expert * perExpert, toInt(perExpert, w3d), x, xOff, 0, y, yOff, 0, 1, false);
    }

    /** {@link #matmul(CpuTensorRef, float[], int, int, float[], int, int, int)} over densely packed rows. */
    public void matmul(CpuTensorRef w, float[] x, int xOff, float[] y, int yOff, int nTokens) {
        int rows = rowCount(w);
        run(w, 0L, rows, x, xOff, toInt(w.cols(), w), y, yOff, rows, nTokens, false);
    }

    /**
     * {@code y[yOff + t*yStride + r] = sum_c w[r][c] * x[xOff + t*xStride + c]} for {@code t < nTokens} and every
     * row {@code r} of {@code w}.
     */
    public void matmul(CpuTensorRef w, float[] x, int xOff, int xStride, float[] y, int yOff, int yStride,
                       int nTokens) {
        run(w, 0L, rowCount(w), x, xOff, xStride, y, yOff, yStride, nTokens, false);
    }

    // ------------------------------------------------------------------ core

    private void run(CpuTensorRef w, long rowStart, int rowCount, float[] x, int xOff, int xStride,
                     float[] y, int yOff, int yStride, int nTokens, boolean accumulate) {
        Objects.requireNonNull(w, "weight");
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        int cols = toInt(w.cols(), w);
        checkRows(w, rowStart, rowCount);
        checkTokens(nTokens, xStride, cols, yStride, rowCount);
        checkView("input", x, xOff, xStride, nTokens, cols);
        checkView("output", y, yOff, yStride, nTokens, rowCount);
        if (rowCount == 0 || nTokens == 0) {
            return;
        }
        CpuVecDot dot = vecDot(w.type());
        Transforms t = transformsOf(w);
        QuantizedActivations[] acts = prepare(w.type(), t.input(), x, xOff, xStride, cols, nTokens);
        long perExpert = t.experts() > 0 ? w.rows() / t.experts() : 0L;
        int grain = (int) Math.max(1L, MIN_TASK_WORK / ((long) cols * nTokens));
        CpuParallel.parallelFor(rowCount, grain, (from, to) -> {
            int n = to - from;
            float[] tmp = accumulate || t.hasScale() ? new float[n] : null;
            for (int tok = 0; tok < nTokens; tok++) {
                int yBase = yOff + tok * yStride + from;
                if (tmp == null) {
                    dot.dotRows(w, rowStart + from, n, acts[tok], y, yBase);
                    continue;
                }
                dot.dotRows(w, rowStart + from, n, acts[tok], tmp, 0);
                for (int i = 0; i < n; i++) {
                    float v = tmp[i] * rowScale(t, perExpert, rowStart + from + i);
                    y[yBase + i] = accumulate ? y[yBase + i] + v : v;
                }
            }
        });
    }

    /** Applies the input Hadamard (when present) and quantizes each token's row for the weight type. */
    private static QuantizedActivations[] prepare(GGMLType type, HadamardSpec hadamard, float[] x, int xOff,
                                                  int xStride, int cols, int nTokens) {
        float[] src = x;
        int srcOff = xOff;
        int srcStride = xStride;
        if (hadamard != null) {
            float[] rotated = new float[Math.multiplyExact(nTokens, cols)];
            if (nTokens == 1 || xStride == cols) {
                CpuHadamard.apply(hadamard, x, xOff, rotated, 0, nTokens, cols);
            } else {
                for (int tok = 0; tok < nTokens; tok++) {
                    CpuHadamard.apply(hadamard, x, xOff + tok * xStride, rotated, tok * cols, 1, cols);
                }
            }
            src = rotated;
            srcOff = 0;
            srcStride = cols;
        }
        QuantizedActivations[] acts = new QuantizedActivations[nTokens];
        final float[] s = src;
        final int so = srcOff;
        final int ss = srcStride;
        CpuParallel.parallelFor(nTokens, 1, (from, to) -> {
            for (int tok = from; tok < to; tok++) {
                acts[tok] = CpuActivationQuantizer.quantizeFor(type, s, so + tok * ss, cols);
            }
        });
        return acts;
    }

    private static float rowScale(Transforms t, long perExpert, long row) {
        return t.scale(perExpert > 0 ? (int) (row / perExpert) : 0);
    }

    // ------------------------------------------------------------------ validation

    private static int rowCount(CpuTensorRef w) {
        return toInt(Objects.requireNonNull(w, "weight").rows(), w);
    }

    private static int toInt(long v, CpuTensorRef w) {
        if (v > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("dimension " + v + " of " + w + " exceeds one float array");
        }
        return (int) v;
    }

    private static void checkRows(CpuTensorRef w, long rowStart, int rowCount) {
        if (rowStart < 0 || rowCount < 0 || rowStart > w.rows() - rowCount) {
            throw new IllegalArgumentException("rows [" + rowStart + ", " + (rowStart + rowCount)
                    + ") are outside the " + w.rows() + "-row weight " + w);
        }
    }

    private static void checkTokens(int nTokens, int xStride, int cols, int yStride, int rowCount) {
        if (nTokens < 0) {
            throw new IllegalArgumentException("token count must be non-negative, got " + nTokens);
        }
        if (nTokens > 1 && (xStride < cols || yStride < rowCount)) {
            throw new IllegalArgumentException("strides (x " + xStride + ", y " + yStride + ") are below the row widths"
                    + " (x " + cols + ", y " + rowCount + ") for " + nTokens + " tokens");
        }
    }

    private static void checkView(String what, float[] a, int off, int stride, int nTokens, int width) {
        long end = nTokens == 0 ? off : off + (long) (nTokens - 1) * stride + width;
        if (off < 0 || end > a.length) {
            throw new IllegalArgumentException(what + " view [" + off + ", " + end + ") of " + nTokens + " row(s) x "
                    + width + " exceeds the array length " + a.length);
        }
    }

    @Override
    public String toString() {
        return "CpuMatVec{" + (this == PLAIN ? "plain" : transforms) + "}";
    }
}
