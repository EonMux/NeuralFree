package com.eonmux.neuralfree.gguf.audio;

import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/** {@link DoubleFft}: the magnitudes of a direct DFT of the zero-padded signal, and power-of-two sizes only. */
public class DoubleFftTest {

    @Test
    public void magnitudesMatchTheDirectTransformOfTheZeroPaddedSignal() {
        int n = 128;
        int len = 100;
        Random r = new Random(3);
        float[] x = new float[len];
        for (int i = 0; i < len; i++) {
            x[i] = r.nextFloat() * 2f - 1f;
        }
        double[] got = new double[n / 2 + 1];
        new DoubleFft(n).magnitudes(x, len, got);
        for (int k = 0; k <= n / 2; k++) {
            double re = 0;
            double im = 0;
            for (int t = 0; t < len; t++) {
                double a = -2.0 * Math.PI * k * t / n;
                re += x[t] * Math.cos(a);
                im += x[t] * Math.sin(a);
            }
            assertEquals("bin " + k, Math.hypot(re, im), got[k], 1e-12);
        }
    }

    @Test
    public void sizesThatAreNoPowerOfTwoAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> new DoubleFft(96));
        assertThrows(IllegalArgumentException.class, () -> new DoubleFft(1));
        assertThrows(IllegalArgumentException.class,
                () -> new DoubleFft(8).magnitudes(new float[16], 16, new double[5]));
    }
}
