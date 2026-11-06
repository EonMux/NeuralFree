package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import com.eonmux.neuralfree.gguf.testutil.GgufTestWriter;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The screen's certificate over random F16, BF16 and Q8_0 rows and hidden rows: every row's screen logit
 * stays inside its bound, and the row holding the exact argmax is always a candidate of the certified
 * set, for every sampled hidden row.
 */
public class CertifiedArgmaxScreenTest {

    private static final int ROWS = 97;
    private static final int COLS = 128;

    @Test
    public void everyScreenLogitStaysInsideItsBound() {
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}) {
            CpuTensorRef weight = weight(type);
            CertifiedArgmaxScreen screen = CertifiedArgmaxScreen.of(weight);
            float[] h = hidden(1);
            float hiddenNorm = norm(h);
            float[] approx = new float[screen.rows()];
            screen.screenLogits(h, 0, approx, 0);
            float[] row = new float[COLS];
            for (int v = 0; v < screen.rows(); v++) {
                weight.decodeRow(v, row, 0);
                float exact = 0.0f;
                for (int i = 0; i < COLS; i++) {
                    exact += row[i] * h[i];
                }
                assertTrue(type + " row " + v + ": |" + (exact - approx[v]) + "| exceeds bound "
                                + hiddenNorm * screen.bound(v),
                        Math.abs(exact - approx[v]) <= hiddenNorm * screen.bound(v));
            }
        }
    }

    @Test
    public void theExactArgmaxRowIsAlwaysACandidate() {
        for (GGMLType type : new GGMLType[]{GGMLType.F16, GGMLType.BF16, GGMLType.Q8_0}) {
            CpuTensorRef weight = weight(type, 7);
            CertifiedArgmaxScreen screen = CertifiedArgmaxScreen.of(weight);
            float[] row = new float[COLS];
            for (int sample = 0; sample < 32; sample++) {
                float[] h = hidden(sample * 31 + 3);
                float hiddenNorm = norm(h);
                float[] approx = new float[screen.rows()];
                screen.screenLogits(h, 0, approx, 0);
                float lower = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < screen.rows(); v++) {
                    float t = approx[v] - screen.bound(v) * hiddenNorm;
                    if (t > lower) {
                        lower = t;
                    }
                }
                float threshold = lower - screen.tie(lower, hiddenNorm);
                int argmax = 0;
                float best = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < screen.rows(); v++) {
                    float exact = 0.0f;
                    weight.decodeRow(v, row, 0);
                    for (int i = 0; i < COLS; i++) {
                        exact += row[i] * h[i];
                    }
                    boolean candidate = approx[v] + screen.bound(v) * hiddenNorm >= threshold;
                    assertTrue(type + " sample " + sample + " row " + v + " outside the certified set",
                            candidate || exact < lower);
                    if (exact > best) {
                        best = exact;
                        argmax = v;
                    }
                }
                boolean in = approx[argmax] + screen.bound(argmax) * hiddenNorm >= threshold;
                assertTrue(type + " sample " + sample + ": argmax " + argmax + " is not a candidate", in);
            }
        }
    }

    @Test
    public void scalesAndBoundsMatchTheRoundedCodes() {
        CpuTensorRef weight = weight(GGMLType.F16, 5);
        CertifiedArgmaxScreen screen = CertifiedArgmaxScreen.of(weight);
        float[] row = new float[COLS];
        for (int v = 0; v < screen.rows(); v++) {
            weight.decodeRow(v, row, 0);
            float absmax = 0.0f;
            for (float w : row) {
                absmax = Math.max(absmax, Math.abs(w));
            }
            assertEquals("scale of row " + v, absmax / 127.0f, screen.scale(v), 0.0f);
            double errSq = 0.0;
            for (int i = 0; i < COLS; i++) {
                float q = Math.max(-127f, Math.min(127f, Math.round(row[i] / screen.scale(v))));
                double diff = row[i] - q * screen.scale(v);
                errSq += diff * diff;
            }
            assertTrue("bound of row " + v, Math.sqrt(errSq) <= screen.bound(v));
        }
    }

    private static CpuTensorRef weight(GGMLType type) {
        return weight(type, 1);
    }

    private static CpuTensorRef weight(GGMLType type, long seed) {
        Random rnd = new Random(seed);
        float[] values = new float[ROWS * COLS];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (rnd.nextGaussian() * 0.35);
        }
        byte[] bytes = GgufTestWriter.encodeFloats(type, values);
        return CpuTensorRef.of(type, ByteBuffer.wrap(bytes), ROWS, COLS);
    }

    private static float[] hidden(int seed) {
        Random rnd = new Random(seed);
        float[] h = new float[COLS];
        for (int i = 0; i < COLS; i++) {
            h[i] = (float) (rnd.nextGaussian() * 1.4);
        }
        return h;
    }

    private static float norm(float[] h) {
        double sum = 0.0;
        for (float v : h) {
            sum += (double) v * v;
        }
        return (float) Math.sqrt(sum);
    }
}
