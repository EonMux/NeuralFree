package com.eonmux.neuralfree.gguf.train;

import com.eonmux.neuralfree.gguf.arch.GluOp;
import com.eonmux.neuralfree.gguf.arch.NormKind;
import com.eonmux.neuralfree.gguf.arch.RopeType;
import com.eonmux.neuralfree.gguf.ops.rope.RopeFreqTable;
import com.eonmux.neuralfree.gguf.spec.RopeSpec;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * Central-difference checks of every {@link TrainMath} kernel: each backward pass must reproduce the derivative of
 * its forward pass element by element.
 */
public class TrainMathTest {

    private static final double EPS = 1e-4;

    /** Absolute rounding floor of a float32 central difference and the relative band a matched derivative needs. */
    private static final double ATOL = 5e-4;
    private static final double RTOL = 5e-3;

    private static float[] random(int n, long seed) {
        Random random = new Random(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) ((random.nextDouble() - 0.5) * 1.5);
        }
        return v;
    }

    /**
     * Asserts {@code actual} equals {@code expected} within the float32 noise of a central difference: the step
     * itself rounds the forward sum, so small derivatives cannot be held to a purely relative band and large ones
     * cannot be held to a purely absolute one.
     */
    private static void assertDerivative(String message, double expected, double actual) {
        assertEquals(message, expected, actual, ATOL + RTOL * Math.abs(expected));
    }

    /** The numeric derivative of {@code f(x[i])} at {@code i}. */
    private static double numeric(int i, float[] x, java.util.function.DoubleSupplier f) {
        float original = x[i];
        x[i] = original + (float) EPS;
        double plus = f.getAsDouble();
        x[i] = original - (float) EPS;
        double minus = f.getAsDouble();
        x[i] = original;
        return (plus - minus) / (2 * EPS);
    }

    @Test
    public void linearInputAndWeightGradients() {
        int rows = 3;
        int cols = 4;
        int n = 2;
        float[] w = random(rows * cols, 1);
        float[] x = random(n * cols, 2);
        float[] dy = random(n * rows, 3);
        float[] y = new float[n * rows];
        TrainMath.linear(w, rows, cols, null, x, n, y);
        float[] dx = new float[n * cols];
        float[] dw = new float[rows * cols];
        TrainMath.linearInputGrad(w, rows, cols, dy, n, dx);
        TrainMath.linearWeightGrad(rows, cols, x, dy, n, dw);
        for (int i = 0; i < x.length; i++) {
            assertDerivative("dx[" + i + "]", numeric(i, x, () -> dot(w, rows, cols, x, dy, n, y)), dx[i]);
        }
        for (int i = 0; i < w.length; i++) {
            int index = i;
            assertDerivative("dw[" + i + "]", numeric(index, w,
                    () -> dot(w, rows, cols, x, dy, n, y)), dw[index]);
        }
    }

    private static double dot(float[] w, int rows, int cols, float[] x, float[] dy, int n, float[] y) {
        TrainMath.linear(w, rows, cols, null, x, n, y);
        double sum = 0;
        for (int i = 0; i < y.length; i++) {
            sum += y[i] * dy[i];
        }
        return sum;
    }

    @Test
    public void rmsNormGradient() {
        checkNorm(NormKind.RMS, 3, 8, 1);
        checkNorm(NormKind.RMS, 4, 8, 2);
    }

    @Test
    public void layerNormGradient() {
        checkNorm(NormKind.LAYER, 3, 8, 1);
        checkNorm(NormKind.LAYER, 4, 8, 2);
    }

    private static void checkNorm(NormKind kind, int rows, int rowLen, int wRows) {
        float[] x = random(rows * rowLen, 10 + rows);
        float[] w = random(wRows * rowLen, 20 + rows);
        float[] dy = random(rows * rowLen, 30 + rows);
        float[] y = new float[rows * rowLen];
        float[] stats = new float[2 * rows];
        TrainMath.normForward(kind, 1e-5f, x, rows, rowLen, wRows, w, null, y, stats);
        float[] dx = new float[x.length];
        float[] dw = new float[w.length];
        TrainMath.normBackward(kind, x, dy, rows, rowLen, wRows, w, stats, dx, dw, null);
        for (int i = 0; i < x.length; i++) {
            assertDerivative(kind + " dx[" + i + "]", numeric(i, x,
                    () -> normDot(kind, rows, rowLen, wRows, w, x, dy, y)), dx[i]);
        }
        for (int i = 0; i < w.length; i++) {
            int index = i;
            assertDerivative(kind + " dw[" + i + "]", numeric(index, w,
                    () -> normDot(kind, rows, rowLen, wRows, w, x, dy, y)), dw[index]);
        }
    }

    private static double normDot(NormKind kind, int rows, int rowLen, int wRows, float[] w, float[] x, float[] dy,
                                  float[] y) {
        TrainMath.normForward(kind, 1e-5f, x, rows, rowLen, wRows, w, null, y, new float[2 * rows]);
        double sum = 0;
        for (int i = 0; i < y.length; i++) {
            sum += y[i] * dy[i];
        }
        return sum;
    }

    @Test
    public void ropeGradient() {
        RopeFreqTable table = RopeFreqTable.of(RopeSpec.builder(RopeType.NORM).nDims(8).freqBase(10000f).build(), null);
        int tokens = 3;
        int heads = 2;
        int headStride = 8;
        float[] x = random(tokens * heads * headStride, 40);
        float[] dy = random(x.length, 41);
        TrainMath.ropeForward(table, x, tokens, heads, headStride, heads * headStride);
        float[] dx = dy.clone();
        TrainMath.ropeBackward(table, dx, tokens, heads, headStride, heads * headStride);
        checkRope(table, tokens, heads, headStride, x, dy, dx);
    }

    private static void checkRope(RopeFreqTable table, int tokens, int heads, int headStride, float[] x,
                                  float[] dy, float[] dx) {
        for (int i = 0; i < x.length; i++) {
            float original = x[i];
            x[i] = original + (float) EPS;
            double plus = ropeDot(table, tokens, heads, headStride, x, dy);
            x[i] = original - (float) EPS;
            double minus = ropeDot(table, tokens, heads, headStride, x, dy);
            x[i] = original;
            assertDerivative("rope dx[" + i + "]", (plus - minus) / (2 * EPS), dx[i]);
        }
    }

    private static double ropeDot(RopeFreqTable table, int tokens, int heads, int headStride, float[] x,
                                  float[] dy) {
        float[] rotated = x.clone();
        TrainMath.ropeForward(table, rotated, tokens, heads, headStride, heads * headStride);
        double sum = 0;
        for (int i = 0; i < rotated.length; i++) {
            sum += rotated[i] * dy[i];
        }
        return sum;
    }

    @Test
    public void causalAttentionGradients() {
        checkAttentionGradients(0, 0f);
    }

    @Test
    public void slidingWindowAttentionGradients() {
        checkAttentionGradients(2, 0f);
    }

    @Test
    public void softcappedAttentionGradients() {
        checkAttentionGradients(0, 1.5f);
    }

    private static void checkAttentionGradients(int window, float softcap) {
        int n = 4;
        int nHead = 2;
        int nHeadKv = 1;
        int headDimK = 3;
        int headDimV = 2;
        float scale = 0.5f;
        int qStride = nHead * headDimK;
        int kStride = nHeadKv * headDimK;
        int vStride = nHeadKv * headDimV;
        int outStride = nHead * headDimV;
        float[] q = random(n * qStride, 50);
        float[] k = random(n * kStride, 51);
        float[] v = random(n * vStride, 52);
        float[] dout = random(n * outStride, 53);
        float[] scores = new float[nHead * n];
        float[] probs = new float[nHead * n];
        float[] dp = new float[n];
        float[] dq = new float[q.length];
        float[] dk = new float[k.length];
        float[] dv = new float[v.length];
        TrainMath.attentionBackward(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride,
                k, kStride, v, vStride, dout, outStride, dq, dk, dv, scores, probs, dp);
        checkAttention(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride, k, kStride, v,
                vStride, dout, outStride, q, dq, scores, probs, "dq");
        checkAttention(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride, k, kStride, v,
                vStride, dout, outStride, k, dk, scores, probs, "dk");
        checkAttention(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride, k, kStride, v,
                vStride, dout, outStride, v, dv, scores, probs, "dv");
    }

    private static void checkAttention(int n, int nHead, int nHeadKv, int headDimK, int headDimV, float scale,
                                       int window, float softcap, float[] q, int qStride, float[] k, int kStride,
                                       float[] v, int vStride, float[] dout, int outStride, float[] operand,
                                       float[] analytic, float[] scores, float[] probs, String name) {
        for (int i = 0; i < operand.length; i++) {
            float original = operand[i];
            operand[i] = original + (float) EPS;
            double plus = attentionDot(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride, k,
                    kStride, v, vStride, dout, outStride, scores, probs);
            operand[i] = original - (float) EPS;
            double minus = attentionDot(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride,
                    k, kStride, v, vStride, dout, outStride, scores, probs);
            operand[i] = original;
            assertDerivative(name + "[" + i + "]", (plus - minus) / (2 * EPS), analytic[i]);
        }
    }

    private static double attentionDot(int n, int nHead, int nHeadKv, int headDimK, int headDimV, float scale,
                                       int window, float softcap, float[] q, int qStride, float[] k, int kStride,
                                       float[] v, int vStride, float[] dout, int outStride, float[] scores,
                                       float[] probs) {
        float[] out = new float[n * outStride];
        TrainMath.attentionForward(n, nHead, nHeadKv, headDimK, headDimV, scale, window, softcap, q, qStride, k,
                kStride, v, vStride, out, outStride, scores, probs);
        double sum = 0;
        for (int i = 0; i < out.length; i++) {
            sum += out[i] * dout[i];
        }
        return sum;
    }

    @Test
    public void expertSelectionTakesTheTopProbabilitiesAndRenormalizes() {
        float[] probs = {0.1f, 0.5f, 0.2f, 0.15f, 0.05f};
        int[] index = new int[2];
        float[] weight = new float[2];
        TrainMath.selectExperts(probs, 0, 5, 2, true, 1f, index, 0, weight, 0);
        assertEquals(1, index[0]);
        assertEquals(2, index[1]);
        assertEquals(0.5f / 0.7f, weight[0], 1e-6f);
        assertEquals(0.2f / 0.7f, weight[1], 1e-6f);
        TrainMath.selectExperts(probs, 0, 5, 2, false, 2f, index, 0, weight, 0);
        assertEquals(1.0f, weight[0], 1e-6f);
        assertEquals(0.4f, weight[1], 1e-6f);
    }

    @Test
    public void gluGradients() {
        int n = 6;
        float[] gate = random(n, 60);
        float[] up = random(n, 61);
        float[] dy = random(n, 62);
        float[] out = new float[n];
        TrainMath.gluForward(GluOp.SWIGLU, gate, up, out, n);
        float[] dGate = new float[n];
        float[] dUp = new float[n];
        TrainMath.gluBackward(GluOp.SWIGLU, gate, up, dy, dGate, dUp, n);
        for (int i = 0; i < n; i++) {
            assertDerivative("dgate[" + i + "]", numeric(i, gate, () -> gluDot(gate, up, dy, out)), dGate[i]);
            int index = i;
            assertDerivative("dup[" + i + "]", numeric(i, up, () -> gluDot(gate, up, dy, out)), dUp[index]);
        }
    }

    private static double gluDot(float[] gate, float[] up, float[] dy, float[] out) {
        TrainMath.gluForward(GluOp.SWIGLU, gate, up, out, out.length);
        double sum = 0;
        for (int i = 0; i < out.length; i++) {
            sum += out[i] * dy[i];
        }
        return sum;
    }

    @Test
    public void crossEntropyGradient() {
        int n = 5;
        float[] logits = random(n, 70);
        float[] probs = new float[n];
        int label = 2;
        double loss = TrainMath.crossEntropy(logits, n, label, probs);
        for (int i = 0; i < n; i++) {
            float original = logits[i];
            logits[i] = original + (float) EPS;
            double plus = TrainMath.crossEntropy(logits, n, label, probs);
            logits[i] = original - (float) EPS;
            double minus = TrainMath.crossEntropy(logits, n, label, probs);
            logits[i] = original;
            double numeric = (plus - minus) / (2 * EPS);
            TrainMath.crossEntropy(logits, n, label, probs);
            double analytic = probs[i] - (i == label ? 1.0 : 0.0);
            assertDerivative("dlogits[" + i + "]", numeric, analytic);
        }
        assertEquals("loss", Math.log(1.0 / Math.max(probs[label], Float.MIN_NORMAL)), loss, 1e-5);
    }
}
