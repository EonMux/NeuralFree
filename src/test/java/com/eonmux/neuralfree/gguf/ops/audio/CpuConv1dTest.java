package com.eonmux.neuralfree.gguf.ops.audio;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Reference checks of the wavtokenizer convolutions against a direct transcription of ggml_conv_1d (im2col rows
 * {@code [OL][IC*K]} with zero padding, then the product with the {@code [OC][IC*K]} weight) and ggml_conv_1d_dw,
 * plus the output-length rule.
 */
public class CpuConv1dTest {

    private static final float EPS = 1e-5f;

    private static float[] random(Random r, int n) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            a[i] = r.nextFloat() * 2f - 1f;
        }
        return a;
    }

    /** ggml_conv_1d(w, x, s0 = 1, p0 = K/2, d0 = dil) followed by ggml_add(bias). */
    private static float[] referenceConv(int len, int ic, int oc, int k, int dil, float[] x, float[] w, float[] b) {
        int p = k / 2;
        int ol = (len + 2 * p - dil * (k - 1) - 1) + 1;
        float[][] im2col = new float[ol][ic * k];
        for (int iow = 0; iow < ol; iow++) {
            for (int iic = 0; iic < ic; iic++) {
                for (int ikw = 0; ikw < k; ikw++) {
                    int iiw = iow + ikw * dil - p;
                    im2col[iow][iic * k + ikw] = (iiw < 0 || iiw >= len) ? 0f : x[iic * len + iiw];
                }
            }
        }
        float[] y = new float[oc * ol];
        for (int o = 0; o < oc; o++) {
            for (int t = 0; t < ol; t++) {
                float acc = 0f;
                for (int j = 0; j < ic * k; j++) {
                    acc += im2col[t][j] * w[o * ic * k + j];
                }
                y[o * ol + t] = acc + b[o];
            }
        }
        return y;
    }

    @Test
    public void sameMatchesIm2colProduct() {
        Random r = new Random(5);
        int[][] shapes = {{10, 4, 3, 7, 1}, {9, 3, 5, 3, 1}, {6, 2, 2, 1, 1}, {12, 3, 2, 3, 2}, {8, 2, 3, 4, 1}};
        for (int[] s : shapes) {
            int len = s[0], ic = s[1], oc = s[2], k = s[3], dil = s[4];
            float[] x = random(r, 3 + ic * len);
            float[] xs = new float[ic * len];
            System.arraycopy(x, 3, xs, 0, xs.length);
            float[] w = random(r, oc * ic * k);
            float[] b = random(r, oc);
            int ol = AudioKernels.outLen(len, k, dil);
            float[] y = new float[2 + oc * ol];
            assertEquals(ol, CpuConv1d.same(len, ic, oc, k, dil, x, 3, w, b, y, 2));
            float[] ref = referenceConv(len, ic, oc, k, dil, xs, w, b);
            float[] got = new float[oc * ol];
            System.arraycopy(y, 2, got, 0, got.length);
            assertArrayEquals(ref, got, EPS);
        }
    }

    @Test
    public void depthwiseEqualsDenseWithDiagonalWeights() {
        Random r = new Random(6);
        int len = 11, c = 4, k = 7;
        float[] x = random(r, c * len);
        float[] w = random(r, c * k);
        float[] b = random(r, c);
        float[] dense = new float[c * c * k];
        for (int ch = 0; ch < c; ch++) {
            System.arraycopy(w, ch * k, dense, (ch * c + ch) * k, k);
        }
        float[] y = new float[c * len];
        assertEquals(len, CpuConv1d.depthwise(len, c, k, x, 0, w, b, y, 0));
        float[] ref = referenceConv(len, c, c, k, 1, x, dense, b);
        assertArrayEquals(ref, y, EPS);
    }

    @Test
    public void outputLengthFollowsGgml() {
        assertEquals(100, AudioKernels.outLen(100, 7, 1));
        assertEquals(100, AudioKernels.outLen(100, 3, 1));
        assertEquals(100, AudioKernels.outLen(100, 1, 1));
        assertEquals(101, AudioKernels.outLen(100, 4, 1));
        assertEquals(98, AudioKernels.outLen(100, 3, 2));
        try {
            AudioKernels.checkConvShape(1, 1, 1, 7, 3);
            fail("a convolution without output must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
        assertEquals(1024L, AudioKernels.convGlobal(4, 200));
    }

    @Test
    public void depthwiseShapeBoundsTheChannelTapsOnly() {
        assertEquals(4, AudioKernels.checkDepthwiseShape(4, 50000, 7));
        try {
            AudioKernels.checkDepthwiseShape(2, 0, 3);
            fail("a depthwise convolution without channels must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void overlappingViewsOfOneArrayAreRejectedAndDisjointOnesAccepted() {
        Random r = new Random(7);
        int len = 6, c = 2, k = 3;
        float[] w = random(r, c * k);
        float[] b = random(r, c);
        float[] buf = random(r, 2 * c * len);
        float[] x = new float[c * len];
        System.arraycopy(buf, 0, x, 0, x.length);
        float[] ref = new float[c * len];
        CpuConv1d.depthwise(len, c, k, x, 0, w, b, ref, 0);
        CpuConv1d.depthwise(len, c, k, buf, 0, w, b, buf, c * len);
        float[] got = new float[c * len];
        System.arraycopy(buf, c * len, got, 0, got.length);
        assertArrayEquals(ref, got, 0f);
        try {
            CpuConv1d.depthwise(len, c, k, buf, 0, w, b, buf, c * len - 1);
            fail("overlapping input and output views must be rejected");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }
}
