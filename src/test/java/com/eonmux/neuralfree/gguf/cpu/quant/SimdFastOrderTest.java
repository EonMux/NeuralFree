package com.eonmux.neuralfree.gguf.cpu.quant;

import com.eonmux.neuralfree.gguf.GGMLType;
import org.junit.Assume;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.Properties;
import java.util.Random;

import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_BYTES;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_D;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.Q8_0_QS;
import static com.eonmux.neuralfree.gguf.quant.BlockLayout.QK8_0;
import static com.eonmux.neuralfree.gguf.quant.FpBlockDecoders.fp16At;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * The opt-in fast summation order of {@value CpuSimd#PROP_ORDER}: the property gate, the dispatch of the fast form,
 * and the fast Q8_0 single-token dots, which sum the exact block integers in float vector lanes and so stay within
 * float rounding of the scalar kernel, give one row the same bits through every entry point, and leave the batched
 * product bit-exact.
 */
public class SimdFastOrderTest {

    private static final int ROWS = 37;

    @Test
    public void thePropertyAcceptsExactAndFastAndRejectsOtherValuesByName() {
        assertFalse(CpuSimd.fastOrder(new Properties()));
        assertFalse(CpuSimd.fastOrder(props(" ")));
        assertFalse(CpuSimd.fastOrder(props("exact")));
        assertTrue(CpuSimd.fastOrder(props("fast")));
        assertTrue(CpuSimd.fastOrder(props(" FAST ")));
        try {
            CpuSimd.fastOrder(props("quick"));
            fail("quick is not an order");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage(), e.getMessage().contains(CpuSimd.PROP_ORDER + "=quick"));
        }
    }

    /**
     * Over row widths with an odd and an even block count, every fast dot lies within the float rounding bound of the
     * two summation orders from the scalar kernel's value: {@code (2 * blocks + 16) * 2^-24} times the sum of the
     * absolute element terms {@code |w_q * y_q * d_x * d_y|}, the error of the scalar chain of block sums plus that
     * of the lane chains and the eight-lane reduction. The row loop, the single-row dot and the byte-row dot agree
     * bit for bit.
     */
    @Test
    public void fastQ8_0DotsStayWithinRoundingOfTheScalarKernelAndAgreeAcrossEntryPoints() {
        Assume.assumeTrue(CpuSimd.available());
        final CpuVecDot scalar = Q8VecDot.Q8_0;
        for (int cols : new int[]{32, 96, 1024, 2560, 9728}) {
            final Random rnd = new Random(cols);
            final ByteBuffer raw = SimdVecDotTest.randomRows(GGMLType.Q8_0, ROWS, cols, rnd);
            final CpuTensorRef w = CpuTensorRef.of(GGMLType.Q8_0, raw, ROWS, cols);
            final CpuVecDot.QuantizedActivations x = CpuActivationQuantizer.quantizeFor(GGMLType.Q8_0,
                    SimdVecDotTest.gaussian(cols, rnd), 0, cols);
            final float[] fast = new float[ROWS];
            fastQ8_0().dotRows(w, 0, ROWS, x, fast, 0);
            final int rowBytes = cols / QK8_0 * Q8_0_BYTES;
            for (int r = 0; r < ROWS; r++) {
                final float want = scalar.dotRow(w, r, x);
                final double bound = Math.scalb((2.0 * (cols / QK8_0) + 16) * absTerms(raw, r * rowBytes, x), -24);
                assertEquals(cols + " cols, row " + r, want, fast[r], bound);
                assertEquals(cols + " cols, row " + r + ": dotRow", Float.floatToRawIntBits(fast[r]),
                        Float.floatToRawIntBits(fastQ8_0().dotRow(w, r, x)));
                final byte[] row = new byte[rowBytes];
                raw.duplicate().position(r * rowBytes).get(row);
                assertEquals(cols + " cols, row " + r + ": dotRowBytes", Float.floatToRawIntBits(fast[r]),
                        Float.floatToRawIntBits(fastQ8_0().dotRowBytes(row, 0, x)));
            }
        }
    }

    /** The fast form's batched product is the exact one: bit-identical to the scalar kernel per token. */
    @Test
    public void theFastFormKeepsTheExactBatchedProduct() {
        Assume.assumeTrue(CpuSimd.available());
        final int cols = 1024;
        final int tokens = 19;
        final Random rnd = new Random(7);
        final CpuTensorRef w = CpuTensorRef.of(GGMLType.Q8_0, SimdVecDotTest.randomRows(GGMLType.Q8_0, ROWS, cols,
                rnd), ROWS, cols);
        final CpuVecDot.QuantizedActivations[] x = new CpuVecDot.QuantizedActivations[tokens];
        for (int t = 0; t < tokens; t++) {
            x[t] = CpuActivationQuantizer.quantizeFor(GGMLType.Q8_0, SimdVecDotTest.gaussian(cols, rnd), 0, cols);
        }
        final float[] y = new float[ROWS * tokens];
        final CpuVecDot fast = fastQ8_0();
        fast.dotTiles(w, 0, ROWS, fast.prepareTiles(x, tokens), x, 0, tokens, y, 0, ROWS);
        for (int t = 0; t < tokens; t++) {
            for (int r = 0; r < ROWS; r++) {
                assertEquals("token " + t + ", row " + r, Float.floatToRawIntBits(Q8VecDot.Q8_0.dotRow(w, r, x[t])),
                        Float.floatToRawIntBits(y[t * ROWS + r]));
            }
        }
    }

    /**
     * A JVM run with the fast order serves the fast Q8_0 form and the exact kernels of every other type, and its
     * status line names the order; without the property it serves the exact form, and an invalid value is rejected by
     * name at the first kernel query.
     */
    @Test
    public void theFastOrderPropertySelectsTheFastQ8_0FormOnly() throws Exception {
        Assume.assumeTrue(CpuSimd.available());
        final String module = "--add-modules=jdk.incubator.vector";
        final String fast = SimdVecDotTest.run(SimdFallbackProbe.class, module, "-D" + CpuSimd.PROP_ORDER + "=fast");
        assertTrue(fast, fast.contains("q8_0=" + fastQ8_0() + "\n"));
        assertTrue(fast, fast.contains("status=" + CpuSimd.describe() + ", fast summation order ("));
        assertTrue(fast, fast.contains("q4_0=" + CpuSimd.vecDot(GGMLType.Q4_0) + "\n"));
        final String exact = SimdVecDotTest.run(SimdFallbackProbe.class, module);
        assertTrue(exact, exact.contains("q8_0=" + CpuSimd.vecDot(GGMLType.Q8_0) + "\n"));
        assertFalse(exact, exact.contains("q8_0=" + fastQ8_0() + "\n"));
        assertTrue(exact, exact.contains("status=" + CpuSimd.status() + "\n"));
        assertFalse(exact, exact.contains("fast summation order"));
        final String bad = SimdVecDotTest.run(SimdFallbackProbe.class, module, "-D" + CpuSimd.PROP_ORDER + "=quick");
        assertTrue(bad, bad.contains("rejected=IllegalArgumentException:" + CpuSimd.PROP_ORDER + "=quick"));
    }

    /** The fast-order Q8_0 form of this JVM's engine. */
    private static CpuVecDot fastQ8_0() {
        final CpuVecDot fast = CpuSimd.fastOrderVecDot(GGMLType.Q8_0);
        assertTrue("the engine has a fast-order Q8_0 form", fast != null);
        return fast;
    }

    /** {@code sum_i |w_q[i] * y_q[i] * d_x * d_y|} of row {@code off} of {@code raw} against {@code x}, in double. */
    private static double absTerms(ByteBuffer raw, int off, CpuVecDot.QuantizedActivations x) {
        final byte[] b = new byte[x.n / QK8_0 * Q8_0_BYTES];
        raw.duplicate().position(off).get(b);
        double s = 0;
        for (int ib = 0; ib < x.n / QK8_0; ib++) {
            long sumAbs = 0;
            for (int j = 0; j < QK8_0; j++) {
                sumAbs += Math.abs(b[ib * Q8_0_BYTES + Q8_0_QS + j] * x.qs[ib * QK8_0 + j]);
            }
            s += sumAbs * Math.abs((double) fp16At(b, ib * Q8_0_BYTES + Q8_0_D) * x.d[ib]);
        }
        return s;
    }

    private static Properties props(String order) {
        final Properties p = new Properties();
        p.setProperty(CpuSimd.PROP_ORDER, order);
        return p;
    }
}
