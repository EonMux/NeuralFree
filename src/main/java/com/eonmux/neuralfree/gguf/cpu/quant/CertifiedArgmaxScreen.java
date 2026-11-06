package com.eonmux.neuralfree.gguf.cpu.quant;

import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * The INT8 screen of one LM head weight, the data the certified exact-argmax lane screens with: per row
 * {@code v} the absmax scale {@code s_v = |W_v|max / 127}, the rounded codes {@code q_v} that approximate
 * the row, and the error bound {@code eb_v = (||W_v - s_v q_v||_2 + c_acc ||W_v||_2) (1 + 2^-20)} whose
 * Cauchy-Schwarz product with the hidden norm bounds the screen's logit error. {@code c_acc} carries the
 * float summation error of both the exact head dot and the screen dot
 * ({@value #C_ACC_STEPS} accumulation steps of {@value #EPSILON}, exact and screen dot together), and the
 * {@code 1 + 2^-20} slack covers evaluating the norms themselves. The build decodes every row of the
 * weight once (through {@link CpuTensorRef#decodeRow}), so it costs one full read of the head.
 *
 * <p>Per step, the screen logit of row {@code v} against hidden row {@code h} is
 * {@code s_v * sum_i h_i q_vi}, and a row is a candidate when
 * {@code s_v (h q_v) + ||h|| eb_v >= T - tie} with {@code T = max_v (s_v (h q_v) - ||h|| eb_v)} and
 * {@code tie} one float ulp at the largest possible logit magnitude
 * {@code |T| + 2 ||h|| ebmax} ({@link #TIE_RELATIVE_ULP}; the exact heads run float logits). Every
 * candidate is then rescored with the head's own exact kernel; the certified set contains the argmax.</p>
 *
 * <p>Immutable and thread-safe; the arrays are owned by the instance.</p>
 */
public final class CertifiedArgmaxScreen {

    /** Screen float accumulations whose rounding the bound carries: the exact dot and the screen dot. */
    private static final int C_ACC_STEPS = 2;

    /** Rounding error of one float32 accumulation. */
    private static final float EPSILON = 0x1p-23f;

    /** Slack for the float evaluation of the row norms themselves. */
    private static final float NORM_SLACK = 1.0f + 0x1p-20f;

    /** Largest quantized code magnitude. */
    private static final float CODE_MAX = 127.0f;

    /** Relative width of the tie window: one ulp of a float logit at the largest possible magnitude. */
    public static final float TIE_RELATIVE_ULP = 0x1p-23f;

    /** Floor of the tie window, keeping an all-zero bound from closing it entirely. */
    private static final float TIE_FLOOR = 1.0e-30f;

    /** Code count of one row. */
    private final int cols;

    /** Per-row scale {@code s_v} (row {@code v}'s absmax over {@link #CODE_MAX}). */
    private final float[] scales;

    /** Per-row codes {@code q_v}, {@link #cols} bytes per row. */
    private final byte[] codes;

    /** Per-row bound {@code eb_v}. */
    private final float[] bounds;

    /** Largest {@link #bounds} value. */
    private final float maxBound;

    private CertifiedArgmaxScreen(int cols, float[] scales, byte[] codes, float[] bounds, float maxBound) {
        this.cols = cols;
        this.scales = scales;
        this.codes = codes;
        this.bounds = bounds;
        this.maxBound = maxBound;
    }

    /**
     * Builds the screen of {@code weight}: every row decoded once, absmax-scaled, rounded to
     * {@code [-127, 127]} and bound. The weight must hold at least one row of at least one column.
     *
     * @throws IllegalArgumentException for an empty weight
     */
    public static CertifiedArgmaxScreen of(CpuTensorRef weight) {
        Objects.requireNonNull(weight, "weight");
        int rows = Math.toIntExact(weight.rows());
        int cols = Math.toIntExact(weight.cols());
        if (rows < 1 || cols < 1) {
            throw new IllegalArgumentException("screen weight " + weight + " holds " + rows + " rows of "
                    + cols + " values; both must be positive");
        }
        float cAcc = cols * C_ACC_STEPS * EPSILON;
        float[] scales = new float[rows];
        byte[] codes = new byte[Math.multiplyExact(rows, cols)];
        float[] bounds = new float[rows];
        float[] row = new float[cols];
        double maxBound = 0.0;
        for (int v = 0; v < rows; v++) {
            weight.decodeRow(v, row, 0);
            float absmax = 0.0f;
            for (int i = 0; i < cols; i++) {
                float a = Math.abs(row[i]);
                if (a > absmax) {
                    absmax = a;
                }
            }
            float scale = absmax / CODE_MAX;
            scales[v] = scale;
            int base = v * cols;
            double errSq = 0.0;
            double normSq = 0.0;
            for (int i = 0; i < cols; i++) {
                float w = row[i];
                float q = scale == 0.0f ? 0.0f
                        : Math.max(-CODE_MAX, Math.min(CODE_MAX, Math.round(w / scale)));
                codes[base + i] = (byte) q;
                double diff = w - q * scale;
                errSq += diff * diff;
                normSq += (double) w * w;
            }
            double eb = (Math.sqrt(errSq) + cAcc * Math.sqrt(normSq)) * NORM_SLACK;
            bounds[v] = (float) eb;
            if (eb > maxBound) {
                maxBound = eb;
            }
        }
        return new CertifiedArgmaxScreen(cols, scales, codes, bounds, (float) maxBound);
    }

    /** The screen's row length: the weight's column count. */
    public int cols() {
        return cols;
    }

    /** Row count: the weight's row count. */
    public int rows() {
        return scales.length;
    }

    /** Scale of row {@code v}. */
    public float scale(int row) {
        return scales[row];
    }

    /** Error bound of row {@code v}: the largest screen logit error per unit of hidden norm. */
    public float bound(int row) {
        return bounds[row];
    }

    /** Largest per-row bound. */
    public float maxBound() {
        return maxBound;
    }

    /** Read-only view of the per-row codes: {@link #rows()} * {@link #cols()} bytes from index 0. */
    public ByteBuffer codesView() {
        return ByteBuffer.wrap(codes).asReadOnlyBuffer();
    }

    /** Copy of the per-row scales, {@link #rows()} floats. */
    public float[] scalesCopy() {
        return scales.clone();
    }

    /** Copy of the per-row bounds, {@link #rows()} floats. */
    public float[] boundsCopy() {
        return bounds.clone();
    }

    /**
     * Writes every row's screen logit against hidden row {@code h[hOff .. hOff + cols)} to
     * {@code out[outOff .. outOff + rows())}: {@code s_v * sum_i h_i q_vi}, each sum accumulated in float
     * in element order, so the values never depend on which row is evaluated when.
     *
     * @throws IllegalArgumentException when a view leaves its array
     */
    public void screenLogits(float[] h, int hOff, float[] out, int outOff) {
        checkViews(h, hOff, out, outOff);
        for (int v = 0; v < scales.length; v++) {
            out[outOff + v] = scales[v] * dotRow(v, h, hOff);
        }
    }

    private void checkViews(float[] h, int hOff, float[] out, int outOff) {
        if (hOff < 0 || hOff + cols > h.length) {
            throw new IllegalArgumentException("hidden view [" + hOff + ", " + (hOff + cols)
                    + ") exceeds the array length " + h.length);
        }
        if (outOff < 0 || (long) rows() + outOff > out.length) {
            throw new IllegalArgumentException("screen view of " + rows() + " rows at " + outOff
                    + " exceeds the array length " + out.length);
        }
    }

    /** Unscaled screen dot of row {@code v} with the hidden row at {@code hOff}. */
    private float dotRow(int v, float[] h, int hOff) {
        int base = v * cols;
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) {
            sum += h[hOff + i] * codes[base + i];
        }
        return sum;
    }

    /**
     * The tie window of a row whose screen maximum of {@code screen - bound * hiddenNorm} is {@code t}
     * and whose hidden norm is {@code hiddenNorm}: one float ulp at the largest possible logit magnitude
     * {@code |t| + 2 * hiddenNorm * maxBound()}.
     */
    public float tie(float t, float hiddenNorm) {
        return (Math.abs(t) + 2.0f * hiddenNorm * maxBound) * TIE_RELATIVE_ULP + TIE_FLOOR;
    }

    @Override
    public String toString() {
        return "CertifiedArgmaxScreen{rows=" + scales.length + ", cols=" + cols
                + ", maxBound=" + maxBound + '}';
    }
}
